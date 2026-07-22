from dataclasses import dataclass, field
from typing import Any

from pyc2py.constants import MAX_CODE_SIZE
from pyc2py.pyc.code import has_code_shape, read_bytes_attr, read_int_attr


def iter_code_objects(root: Any | None, max_objects: int = 10_000) -> list[Any]:
    if root is None:
        return []
    if max_objects < 1:
        raise ValueError("max_objects must be positive")

    result: list[Any] = []
    work = [root]
    for _step in range(max_objects):
        if not work:
            return result
        code = work.pop()
        result.append(code)
        work.extend(
            value
            for value in getattr(code, "co_consts", ()) or ()
            if has_code_shape(value)
        )
    raise ValueError("code object walk exceeded max_objects")


@dataclass(slots=True)
class CodeObjectValidation:
    nested_code_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def checks(self) -> tuple[str, ...]:
        return (f"nested code object constants: {self.nested_code_count}",)


def validate_code_object(code: Any) -> CodeObjectValidation:
    warnings: list[str] = []
    warnings.extend(validate_code_counts(code))
    warnings.extend(validate_code_field_shapes(code))

    return CodeObjectValidation(
        nested_code_count=count_nested_code_constants(code),
        warnings=warnings,
    )


def validate_code_counts(code: Any) -> list[str]:
    warnings: list[str] = []
    argcount = read_int_attr(code, "co_argcount")
    posonlyargcount = read_int_attr(code, "co_posonlyargcount")
    kwonlyargcount = read_int_attr(code, "co_kwonlyargcount")
    nlocals = read_int_attr(code, "co_nlocals")
    stacksize = read_int_attr(code, "co_stacksize")
    firstlineno = read_int_attr(code, "co_firstlineno")

    for name, value in (
        ("co_argcount", argcount),
        ("co_posonlyargcount", posonlyargcount),
        ("co_kwonlyargcount", kwonlyargcount),
        ("co_nlocals", nlocals),
        ("co_stacksize", stacksize),
        ("co_firstlineno", firstlineno),
    ):
        if value < 0:
            warnings.append(f"{name} must not be negative: {value}")
    if posonlyargcount > argcount:
        warnings.append(
            f"co_posonlyargcount exceeds co_argcount: {posonlyargcount} > {argcount}"
        )

    varnames = tuple(getattr(code, "co_varnames", ()) or ())
    minimum_locals = argcount + kwonlyargcount
    if nlocals and nlocals < minimum_locals:
        warnings.append(
            f"co_nlocals is smaller than argument locals: {nlocals} < {minimum_locals}"
        )
    if varnames and nlocals and len(varnames) < minimum_locals:
        warnings.append(
            f"co_varnames is smaller than argument locals: {len(varnames)} < {minimum_locals}"
        )

    return warnings


def validate_code_field_shapes(code: Any) -> list[str]:
    warnings: list[str] = []
    code_bytes = read_bytes_attr(code, "co_code")
    if len(code_bytes) > MAX_CODE_SIZE:
        warnings.append(f"co_code exceeds local size limit: {len(code_bytes)}")

    for field_name in (
        "co_consts",
        "co_names",
        "co_varnames",
        "co_freevars",
        "co_cellvars",
    ):
        value = getattr(code, field_name, ()) or ()
        if not isinstance(value, tuple):
            warnings.append(f"{field_name} is not a tuple: {type(value).__name__}")

    for field_name in ("co_filename", "co_name", "co_qualname"):
        value = getattr(code, field_name, "")
        if value is not None and not isinstance(value, str):
            warnings.append(f"{field_name} is not text: {type(value).__name__}")

    return warnings


def count_nested_code_constants(code: Any) -> int:
    count = 0
    for value in getattr(code, "co_consts", ()) or ():
        if has_code_shape(value):
            count += 1
    return count
