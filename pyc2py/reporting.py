import dis
import inspect
import sys
from pathlib import Path
from types import CodeType
from typing import Any, Protocol

from pyc2py.bytecode.decoder import decode_instructions, validate_bytecode
from pyc2py.bytecode.metadata import (
    parse_exception_table,
    validate_exception_table,
    validate_line_entries,
)
from pyc2py.cfg import build_cfg, validate_cfg, validate_cfg_analysis
from pyc2py.pyc.validation import iter_code_objects, validate_code_object
from pyc2py.source import validate_source, validate_source_format
from pyc2py.stack import validate_linear_stack_effects
from pyc2py.types import ProgressCallback, VerificationReport, emit_progress


class ValidationResult(Protocol):
    warnings: list[str]

    @property
    def checks(self) -> tuple[str, ...]: ...


def validate_generated_source(
    pyc_path: Path,
    source_path: Path,
    source: str,
    warnings: list[str],
    version: tuple[int, ...] | None = None,
    code: Any | None = None,
    progress: ProgressCallback | None = None,
) -> VerificationReport:
    emit_progress(progress, "validating code object recovery")
    bytecode_checks, bytecode_warnings = validate_code_recovery(code, version)
    warnings.extend(bytecode_warnings)
    if version is not None and version < (3, 0):
        emit_progress(progress, "validating legacy source formatting")
        formatting = validate_source_format(source)
        warnings.extend(formatting.warnings)
        warnings.append(
            f"legacy Python {version[0]}.{version[1]} source was neither compiled "
            "nor compared with the original bytecode (this python cannot parse it)"
        )
        return VerificationReport(
            source_path=source_path,
            pyc_path=pyc_path,
            compiled=True,
            source_valid=not formatting.warnings,
            checks=[
                f"legacy Python source emitted for {version[0]}.{version[1]}",
                *bytecode_checks,
                *formatting.checks,
            ],
            warnings=warnings,
            errors=[],
        )

    emit_progress(progress, "parsing and compiling generated source")
    validation = validate_source(source, str(source_path))
    errors: list[str] = []
    checks = list(validation.checks)
    warnings.extend(validation.warnings)
    if not validation.compiled:
        error = validation.error or "source did not compile"
        emit_progress(progress, f"error: {error}")
        errors.append(error)
    else:
        emit_progress(progress, "comparing recovered bytecode with the original")
        warnings.extend(compare_recovered_bytecode(code, validation.code, version))

    return VerificationReport(
        source_path=source_path,
        pyc_path=pyc_path,
        compiled=validation.compiled,
        source_valid=validation.compiled,
        checks=[*bytecode_checks, *checks],
        warnings=warnings,
        errors=errors,
    )


def compare_recovered_bytecode(
    original: Any | None,
    recovered: object | None,
    version: tuple[int, ...] | None,
) -> list[str]:
    if original is None or not isinstance(recovered, CodeType):
        return []
    if version is None or tuple(version[:2]) != sys.version_info[:2]:
        return [
            "recovered source was not compared with the original bytecode "
            "(the pyc is from another python version)"
        ]
    if not isinstance(original, CodeType):
        return []
    return [
        f"{label}: recovered source compiles to different bytecode"
        for label in differing_code_labels(original, recovered)
    ]


def differing_code_labels(original: CodeType, recovered: CodeType) -> list[str]:
    labels: list[str] = []
    if code_behaviour(original) != code_behaviour(recovered):
        labels.append(code_object_name(original))
    original_children = [c for c in original.co_consts if isinstance(c, CodeType)]
    recovered_children = [c for c in recovered.co_consts if isinstance(c, CodeType)]
    for original_child, recovered_child in zip(
        original_children, recovered_children, strict=False
    ):
        labels.extend(differing_code_labels(original_child, recovered_child))
    return labels


def code_behaviour(code: CodeType) -> tuple[object, ...]:
    instructions, positions, jumps = behaviour_instructions(code)
    return (
        instructions,
        protected_instructions(code, positions, len(instructions), jumps),
        docstring_constant(code),
        code.co_varnames,
        code_signature(code),
    )


SIGNATURE_FLAGS = (
    inspect.CO_VARARGS
    | inspect.CO_VARKEYWORDS
    | inspect.CO_GENERATOR
    | inspect.CO_COROUTINE
    | inspect.CO_ITERABLE_COROUTINE
    | inspect.CO_ASYNC_GENERATOR
)


def code_signature(code: CodeType) -> tuple[object, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags & SIGNATURE_FLAGS,
        code.co_freevars,
        code.co_cellvars,
        code.co_name == "<lambda>",
    )


LINE_FUSED_INSTRUCTIONS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
}
LAYOUT_ONLY_INSTRUCTIONS = frozenset({"EXTENDED_ARG", "NOP"})
JUMP_OPCODES = frozenset(dis.hasjrel) | frozenset(dis.hasjabs)
NAME_OPCODES = frozenset(dis.hasname)
CONSTANT_OPCODES = frozenset(dis.hasconst)


def behaviour_instructions(
    code: CodeType,
) -> tuple[tuple[tuple[str, object], ...], dict[int, int], frozenset[int]]:
    kept: list[tuple[str, object]] = []
    jumps: dict[int, int] = {}
    positions: dict[int, int] = {}
    # only python 3.13+ stores that line, earlier a class can only get the
    # name from its own source
    class_body = sys.version_info >= (3, 13) and is_class_body(code)
    for instruction in dis.get_instructions(code):
        positions[instruction.offset] = len(kept)
        if instruction.opname in LAYOUT_ONLY_INSTRUCTIONS:
            continue
        if class_body and stores_first_line_number(instruction) and kept:
            kept[-1] = (kept[-1][0], None)
            class_body = False
        if instruction.opcode in JUMP_OPCODES:
            jumps[len(kept)] = int(instruction.argval)
        kept.extend(behaviour_arguments(instruction, code))
    for index, target_offset in jumps.items():
        kept[index] = (kept[index][0], positions.get(target_offset, len(kept)))
    return tuple(kept), positions, frozenset(jumps)


def protected_instructions(
    code: CodeType,
    positions: dict[int, int],
    count: int,
    jumps: frozenset[int],
) -> tuple[object, ...]:
    covered: list[object] = [None] * count
    table = getattr(code, "co_exceptiontable", b"")
    for entry in parse_exception_table(table):
        handler = (positions.get(entry.target_offset, count), entry.depth, entry.lasti)
        start = positions.get(entry.start_offset, count)
        for position in range(start, positions.get(entry.end_offset, count)):
            covered[position] = handler
    for position in jumps:
        covered[position] = None
    return tuple(covered)


def behaviour_arguments(
    instruction: dis.Instruction,
    code: CodeType,
) -> list[tuple[str, object]]:
    if instruction.opname == "KW_NAMES" and instruction.arg is not None:
        return [(instruction.opname, constant_key(code.co_consts[instruction.arg]))]
    if instruction.opcode in NAME_OPCODES:
        return [(instruction.opname, instruction.argrepr)]
    if instruction.opcode in CONSTANT_OPCODES:
        return [(instruction.opname, constant_key(instruction.argval))]
    return unfused_instruction(instruction)


def constant_key(value: object) -> object:
    if isinstance(value, CodeType):
        return CodeType
    if isinstance(value, float | complex):
        return (type(value), repr(value))
    if isinstance(value, tuple):
        return tuple(constant_key(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(constant_key(item) for item in value)
    return (type(value), value)


def docstring_constant(code: CodeType) -> object:
    if not code.co_consts:
        return None
    has_docstring = getattr(inspect, "CO_HAS_DOCSTRING", None)
    if has_docstring is None:
        is_function = bool(code.co_flags & inspect.CO_OPTIMIZED)
        return code.co_consts[0] if is_function else None
    return code.co_consts[0] if code.co_flags & has_docstring else None


def is_class_body(code: CodeType) -> bool:
    return code.co_names[:3] == ("__name__", "__module__", "__qualname__")


def stores_first_line_number(instruction: dis.Instruction) -> bool:
    return (
        instruction.opname == "STORE_NAME" and instruction.argval == "__firstlineno__"
    )


def unfused_instruction(instruction: dis.Instruction) -> list[tuple[str, object]]:
    halves = LINE_FUSED_INSTRUCTIONS.get(instruction.opname)
    if halves is None or instruction.arg is None:
        return [(plain_load(instruction.opname), instruction.arg)]
    # both local indexes share the argument, four bits each
    return [
        (plain_load(halves[0]), instruction.arg >> 4),
        (plain_load(halves[1]), instruction.arg & 15),
    ]


def plain_load(opname: str) -> str:
    return "LOAD_FAST" if opname == "LOAD_FAST_BORROW" else opname


def validate_code_recovery(
    code: Any | None,
    version: tuple[int, ...] | None,
) -> tuple[list[str], list[str]]:
    if code is None:
        return [], []

    checks: list[str] = []
    warnings: list[str] = []

    for code_index, code_object in enumerate(iter_code_objects(code)):
        label = code_object_label(code_index, code_object)
        code_checks, code_warnings = validate_single_code_object(code_object, version)
        checks.extend(f"{label}: {check}" for check in code_checks)
        warnings.extend(f"{label}: {warning}" for warning in code_warnings)

    return checks, warnings


def code_object_label(code_index: int, code: Any) -> str:
    return f"code[{code_index}] {code_object_name(code)}"


def code_object_name(code: Any) -> str:
    return str(getattr(code, "co_qualname", None) or getattr(code, "co_name", "code"))


def validate_single_code_object(
    code: Any,
    version: tuple[int, ...] | None,
) -> tuple[list[str], list[str]]:
    checks: list[str] = []
    warnings: list[str] = []
    validations: list[ValidationResult] = [
        validate_code_object(code),
        validate_bytecode(code, version),
    ]

    instructions = decode_instructions(code, version)
    validations.extend(validate_instruction_recovery(code, instructions, version))

    for validation in validations:
        checks.extend(validation.checks)
        warnings.extend(validation.warnings)
    return checks, warnings


def validate_instruction_recovery(
    code: Any,
    instructions: list[Any],
    version: tuple[int, ...] | None,
) -> list[ValidationResult]:
    if not instructions:
        return []

    code_size = len(bytes(getattr(code, "co_code", b"") or b""))
    instruction_offsets = {instruction.offset for instruction in instructions}
    exception_table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    exception_entries = safe_exception_entries(exception_table)
    graph = build_cfg(instructions, exception_entries)
    stacksize = int(getattr(code, "co_stacksize", -1) or -1)
    validations: list[ValidationResult] = [
        validate_line_entries(instructions),
        validate_cfg(graph),
        validate_cfg_analysis(graph),
        validate_linear_stack_effects(
            instructions,
            expected_stacksize=stacksize,
            version=version,
        ),
    ]
    if exception_table:
        validations.append(
            validate_exception_table(exception_table, instruction_offsets, code_size)
        )

    return validations


def safe_exception_entries(exception_table: bytes) -> tuple[Any, ...]:
    if not exception_table:
        return ()
    try:
        return parse_exception_table(exception_table)
    except ValueError:
        return ()
