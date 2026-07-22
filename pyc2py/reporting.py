from pathlib import Path
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

    return VerificationReport(
        source_path=source_path,
        pyc_path=pyc_path,
        compiled=validation.compiled,
        source_valid=validation.compiled,
        checks=[*bytecode_checks, *checks],
        warnings=warnings,
        errors=errors,
    )


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
    name = str(getattr(code, "co_qualname", None) or getattr(code, "co_name", "code"))
    return f"code[{code_index}] {name}"


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
