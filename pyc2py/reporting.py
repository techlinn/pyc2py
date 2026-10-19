import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import CodeType
from typing import Any, Protocol

from pyc2py.bytecode.decoder import (
    code_opcode_table,
    decode_instructions,
    validate_bytecode,
)
from pyc2py.bytecode.metadata import (
    parse_exception_table,
    validate_exception_table,
    validate_line_entries,
)
from pyc2py.bytecode.scanner import scan_code
from pyc2py.cfg import build_cfg, validate_cfg
from pyc2py.code_behaviour import (
    code_behaviour,
    code_shape,
    iter_units,
    portable_behaviour,
)
from pyc2py.compilers import Compiler, CompilerError
from pyc2py.pyc.code import code_objects
from pyc2py.pyc.validation import validate_code_object
from pyc2py.source import validate_source, validate_source_format
from pyc2py.stack import StackValidation, validate_linear_stack_effects
from pyc2py.types import (
    MISSING_BODY,
    ProgressCallback,
    UnitVerdict,
    VerificationReport,
    emit_progress,
)

LEGACY_HELPER_PATTERN = re.compile(r"\b__pyc2py_\w+")
HARMLESS_SKIPS = frozenset({"STORE_LOCALS"})
SKIP_PATTERN = re.compile(
    r"^(?:native: )?(?P<scope>.*?):? ?(?:control-flow|unsupported) opcode skipped "
    r"at (?P<offset>\d+): opcode=(?P<opname>\w+)"
)


def is_host_version(version: tuple[int, ...] | None) -> bool:
    return version is not None and tuple(version[:2]) == sys.version_info[:2]


def skipped_instruction_errors(warnings: list[str]) -> list[str]:
    lost: dict[str, list[str]] = {}

    for warning in warnings:
        match = SKIP_PATTERN.match(warning)
        if match is None or match["opname"] in HARMLESS_SKIPS:
            continue

        scope = match["scope"] or "<module>"
        lost.setdefault(scope, []).append(f"{match['opname']} at {match['offset']}")

    return [
        f"{scope}: {len(items)} instruction(s) could not be placed: {', '.join(items)}"
        for scope, items in lost.items()
    ]


# id(code object) -> (status, reason)
InputFindings = dict[int, tuple[str, str]]


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
    compiler: Compiler | None = None,
) -> VerificationReport:
    # compiler: an interpreter of the file's own version, from --verify-with
    emit_progress(progress, "validating code object recovery")
    bytecode_checks, input_problems, findings = validate_code_recovery(code, version)
    errors = [warning for warning in warnings if MISSING_BODY in warning]
    errors.extend(f"damaged input: {problem}" for problem in input_problems)

    if not is_host_version(version):
        errors.extend(skipped_instruction_errors(warnings))

    if version is not None and version < (3, 0):
        emit_progress(progress, "validating legacy Python source formatting")
        formatting = validate_source_format(source)
        warnings.extend(formatting.warnings)
        errors.extend(
            f"unresolved decompiler helper: {name}"
            for name in sorted(set(LEGACY_HELPER_PATTERN.findall(source)))
        )
        reason = (
            f"legacy Python {version[0]}.{version[1]} source cannot be compiled "
            "by this python, so it was not compared with the original"
        )

        if compiler is None:
            warnings.append(reason)
            legacy_units = unverified_units(code, reason, findings)
        else:
            legacy_units = compared_elsewhere(code, source, compiler, findings, errors)

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
            errors=errors,
            units=legacy_units,
        )

    emit_progress(progress, "parsing and compiling generated source")
    validation = validate_source(source, str(source_path))
    checks = list(validation.checks)
    warnings.extend(validation.warnings)
    units: list[UnitVerdict] = []
    newer_syntax = (
        not validation.compiled
        and validation.syntax_error
        and is_newer_than_host(version)
    )

    if compiler is not None:
        emit_progress(
            progress, f"compiling the recovered source with {compiler.python}"
        )
        units = compared_elsewhere(code, source, compiler, findings, errors)
    elif newer_syntax:
        reason = (
            f"this Python ({sys.version_info[0]}.{sys.version_info[1]}) cannot "
            f"compile the syntax of a newer Python: {validation.error}"
        )
        warnings.append(reason)
        units = unverified_units(code, reason, findings)
    elif not validation.compiled:
        error = validation.error or "source did not compile"
        emit_progress(progress, f"error: {error}")
        errors.append(error)
    else:
        emit_progress(progress, "comparing recovered bytecode with the original")
        units = unit_verdicts(code, validation.code, version, findings)

    return VerificationReport(
        source_path=source_path,
        pyc_path=pyc_path,
        compiled=validation.compiled,
        source_valid=validation.compiled or newer_syntax,
        checks=[*bytecode_checks, *checks],
        warnings=warnings,
        errors=errors,
        units=units,
    )


def compared_elsewhere(
    code: Any | None,
    source: str,
    compiler: Compiler,
    findings: InputFindings,
    errors: list[str],
) -> list[UnitVerdict]:
    # the recovered source compiled by the file's own python, compared there
    try:
        recovered = compiler.compile(source)
    except CompilerError as error:
        errors.append(
            f"{compiler.python} did not compile the recovered source: {error}"
        )
        return []

    def judge(name: str, unit: Any, counterpart: Any) -> UnitVerdict:
        same = portable_behaviour(unit, compiler.version) == portable_behaviour(
            counterpart, compiler.version
        )
        return UnitVerdict(name, *(("verified", "") if same else DIFFERENT_ELSEWHERE))

    return judged_units(code, recovered, False, judge, findings)


DIFFERENT_ELSEWHERE = (
    "different",
    "recompiles to different bytecode with --verify-with",
)
CROSS_VERSION_REASON = (
    "the pyc is from another python version, so its bytecode could not be "
    "compared; only the functions, classes and their arguments were checked"
)


def is_newer_than_host(version: tuple[int, ...] | None) -> bool:
    return version is not None and tuple(version[:2]) > sys.version_info[:2]


def unit_verdicts(
    original: Any | None,
    recovered: object | None,
    version: tuple[int, ...] | None,
    findings: InputFindings | None = None,
) -> list[UnitVerdict]:
    if original is None or not isinstance(recovered, CodeType):
        return []
    if isinstance(original, CodeType) and is_host_version(version):
        return judged_units(original, recovered, False, same_bytecode, findings or {})

    return judged_units(original, recovered, True, same_shape, findings or {})


def same_bytecode(name: str, unit: Any, counterpart: Any) -> UnitVerdict:
    if code_behaviour(unit) != code_behaviour(counterpart):
        return UnitVerdict(name, "different", "recompiles to different bytecode")

    return UnitVerdict(name, "verified", "")


def same_shape(name: str, unit: Any, counterpart: Any) -> UnitVerdict:
    if code_shape(unit) != code_shape(counterpart):
        return UnitVerdict(name, "different", "different arguments or kind")

    return UnitVerdict(name, "unverified", CROSS_VERSION_REASON)


def judged_units(
    original: Any | None,
    recovered: Any,
    structural: bool,
    judge: Callable[[str, Any, Any], UnitVerdict],
    findings: InputFindings,
) -> list[UnitVerdict]:
    counterparts = dict(iter_units(recovered, structural))
    verdicts: list[UnitVerdict] = []

    for name, unit in iter_units(original, structural):
        counterpart = counterparts.get(name)

        if id(unit) in findings:
            verdicts.append(UnitVerdict(name, *findings[id(unit)]))
        elif counterpart is None:
            verdicts.append(UnitVerdict(name, "missing", "not in the recovered source"))
        else:
            try:
                verdicts.append(judge(name, unit, counterpart))
            except Exception as error:
                reason = (
                    f"pyc2py could not compare it ({type(error).__name__}: {error})"
                )
                verdicts.append(UnitVerdict(name, "error", reason))

    return verdicts


def unverified_units(
    code: Any | None, reason: str, findings: InputFindings
) -> list[UnitVerdict]:
    return [
        UnitVerdict(name, *findings.get(id(unit), ("unverified", reason)))
        for name, unit in iter_units(code, structural=False)
    ]


def validate_code_recovery(
    code: Any | None,
    version: tuple[int, ...] | None,
) -> tuple[list[str], list[str], InputFindings]:
    if code is None:
        return [], [], {}

    checks: list[str] = []
    problems: list[str] = []
    findings: InputFindings = {}

    for code_index, code_object in enumerate(code_objects(code)):
        label = code_object_label(code_index, code_object)

        try:
            code_checks, code_problems = validate_single_code_object(
                code_object, version
            )
        except Exception as error:
            # pyc2py's own failure, which the unit's error verdict reports
            reason = f"pyc2py could not check it ({type(error).__name__}: {error})"
            findings[id(code_object)] = ("error", reason)
            continue

        checks.extend(f"{label}: {check}" for check in code_checks)
        problems.extend(f"{label}: {problem}" for problem in code_problems)

        if code_problems:
            reason = f"its bytecode is damaged ({code_problems[0]})"
            findings[id(code_object)] = ("damaged", reason)

    return checks, problems, findings


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
    if not any(validation.warnings for validation in validations):
        instructions = decode_instructions(code, version)
        validations.extend(validate_instruction_recovery(code, instructions, version))

    for validation in validations:
        checks.extend(validation.checks)

        # the stack model is not exact (UPGRADE.md N35)
        if not isinstance(validation, StackValidation):
            warnings.extend(validation.warnings)

    return checks, warnings


def validate_instruction_recovery(
    code: Any,
    instructions: list[Any],
    version: tuple[int, ...] | None,
) -> list[ValidationResult]:
    if not instructions:
        return []

    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    code_size = len(code_bytes)
    # includes instructions the decoder drops, such as NOT_TAKEN
    instruction_offsets = {instruction.offset for instruction in instructions}

    if version is not None:
        table = code_opcode_table(code, version)
        instruction_offsets |= {
            item.offset for item in scan_code(code_bytes, table, table.layout)
        }

    exception_table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    exception_entries = safe_exception_entries(exception_table)
    graph = build_cfg(instructions, exception_entries)
    stacksize = int(getattr(code, "co_stacksize", -1) or -1)
    validations: list[ValidationResult] = [
        validate_line_entries(instructions),
        validate_cfg(graph),
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
