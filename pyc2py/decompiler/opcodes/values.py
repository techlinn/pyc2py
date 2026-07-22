import ast
import math
from typing import Any

from pyc2py.astree import constant_sort_key, is_none_constant, make_name
from pyc2py.pyc.objects import (
    FrozenDictValue,
    LegacyByteString,
    LegacyLong,
    MarshalSetValue,
)

EMPTY_SLICE_STEP_NAME = "__pyc2py_empty_slice_step__"


def is_code_constant(value: Any) -> bool:
    return isinstance(value, ast.Constant) and hasattr(value.value, "co_code")


def is_zero_constant(value: Any) -> bool:
    return isinstance(value, ast.Constant) and value.value == 0


def make_constant(
    value: Any,
    *,
    legacy_strings_as_bytes: bool = False,
    negative_nan: bool | None = None,
) -> ast.expr:
    if hasattr(value, "co_code"):
        return ast.Constant(value=value)

    special = make_special_constant(value, legacy_strings_as_bytes, negative_nan)
    if special is not None:
        return special

    collection = make_collection_constant(value, legacy_strings_as_bytes)
    if collection is not None:
        return collection

    return ast.Constant(value=value)


def make_special_constant(
    value: Any,
    legacy_strings_as_bytes: bool,
    negative_nan: bool | None,
) -> ast.expr | None:
    if isinstance(value, LegacyByteString):
        return make_legacy_byte_string_constant(value, legacy_strings_as_bytes)
    if isinstance(value, LegacyLong):
        return make_legacy_long_constant(value)
    if isinstance(value, float) and math.isnan(value):
        return make_nan_constant(value, negative_nan)
    if isinstance(value, int) and not isinstance(value, bool):
        return ast.Constant(value=int(value))
    return None


def make_collection_constant(
    value: Any,
    legacy_strings_as_bytes: bool,
) -> ast.expr | None:
    if isinstance(value, (tuple, list)):
        return make_sequence_constant(value, legacy_strings_as_bytes)
    if isinstance(value, MarshalSetValue):
        return make_set_constant(value.items, legacy_strings_as_bytes)
    if isinstance(value, frozenset):
        return make_set_constant(
            sorted(value, key=constant_sort_key),
            legacy_strings_as_bytes,
        )
    if isinstance(value, FrozenDictValue):
        return make_frozen_dict_constant(value)
    if isinstance(value, slice):
        return make_slice_constant(value)
    return None


def make_legacy_byte_string_constant(
    value: LegacyByteString,
    legacy_strings_as_bytes: bool,
) -> ast.Constant:
    if legacy_strings_as_bytes:
        return ast.Constant(value=value.encode("latin-1", errors="surrogateescape"))
    return ast.Constant(value=str(value))


def make_legacy_long_constant(value: LegacyLong) -> ast.Call:
    return ast.Call(
        func=make_name("__pyc2py_long__", ast.Load()),
        args=[ast.Constant(value=int(value))],
        keywords=[],
    )


def make_nan_constant(value: float, negative_nan: bool | None) -> ast.expr:
    is_negative_nan = math.copysign(1.0, value) < 0

    if negative_nan is not None:
        is_negative_nan = negative_nan
    if is_negative_nan:
        return ast.UnaryOp(
            op=ast.USub(),
            operand=ast.Constant(value=float("nan")),
        )
    return ast.Constant(value=float("nan"))


def make_sequence_constant(
    value: tuple[Any, ...] | list[Any],
    legacy_strings_as_bytes: bool,
) -> ast.Tuple | ast.List:
    elements = make_constant_elements(value, legacy_strings_as_bytes)
    if isinstance(value, tuple):
        return ast.Tuple(elts=elements, ctx=ast.Load())
    return ast.List(elts=elements, ctx=ast.Load())


def make_set_constant(
    values: Any,
    legacy_strings_as_bytes: bool,
) -> ast.Set:
    return ast.Set(elts=make_constant_elements(values, legacy_strings_as_bytes))


def make_constant_elements(
    values: Any,
    legacy_strings_as_bytes: bool,
) -> list[ast.expr]:
    return [
        make_constant(item, legacy_strings_as_bytes=legacy_strings_as_bytes)
        for item in values
    ]


def make_frozen_dict_constant(value: FrozenDictValue) -> ast.Call:
    return ast.Call(
        func=make_name("frozendict", ast.Load()),
        args=[
            ast.Dict(
                keys=[make_constant(key) for key, _ in value.items],
                values=[make_constant(value_) for _, value_ in value.items],
            )
        ],
        keywords=[],
    )


def make_slice_constant(value: slice) -> ast.Call:
    return ast.Call(
        func=make_name("slice", ast.Load()),
        args=[
            make_constant(value.start),
            make_constant(value.stop),
            make_constant(value.step),
        ],
        keywords=[],
    )


def none_to_empty(value: ast.expr) -> ast.expr | None:
    if is_none_constant(value):
        return None
    return value


BINARY_OPS: dict[str, type[ast.operator]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "@": ast.MatMult,
    "/": ast.Div,
    "//": ast.FloorDiv,
    "%": ast.Mod,
    "**": ast.Pow,
    "<<": ast.LShift,
    ">>": ast.RShift,
    "&": ast.BitAnd,
    "|": ast.BitOr,
    "^": ast.BitXor,
}

UNARY_OPS: dict[str, type[ast.unaryop]] = {
    "UNARY_POSITIVE": ast.UAdd,
    "UNARY_NEGATIVE": ast.USub,
    "UNARY_NOT": ast.Not,
    "UNARY_INVERT": ast.Invert,
}

LEGACY_BINARY_OPS: dict[str, type[ast.operator]] = {
    "BINARY_ADD": ast.Add,
    "BINARY_SUBTRACT": ast.Sub,
    "BINARY_MULTIPLY": ast.Mult,
    "BINARY_MATRIX_MULTIPLY": ast.MatMult,
    "BINARY_DIVIDE": ast.Div,
    "BINARY_TRUE_DIVIDE": ast.Div,
    "BINARY_FLOOR_DIVIDE": ast.FloorDiv,
    "BINARY_MODULO": ast.Mod,
    "BINARY_POWER": ast.Pow,
    "BINARY_LSHIFT": ast.LShift,
    "BINARY_RSHIFT": ast.RShift,
    "BINARY_AND": ast.BitAnd,
    "BINARY_OR": ast.BitOr,
    "BINARY_XOR": ast.BitXor,
    "INPLACE_ADD": ast.Add,
    "INPLACE_SUBTRACT": ast.Sub,
    "INPLACE_MULTIPLY": ast.Mult,
    "INPLACE_MATRIX_MULTIPLY": ast.MatMult,
    "INPLACE_DIVIDE": ast.Div,
    "INPLACE_TRUE_DIVIDE": ast.Div,
    "INPLACE_FLOOR_DIVIDE": ast.FloorDiv,
    "INPLACE_MODULO": ast.Mod,
    "INPLACE_POWER": ast.Pow,
    "INPLACE_LSHIFT": ast.LShift,
    "INPLACE_RSHIFT": ast.RShift,
    "INPLACE_AND": ast.BitAnd,
    "INPLACE_OR": ast.BitOr,
    "INPLACE_XOR": ast.BitXor,
}

COMPARE_OPS: dict[str, type[ast.cmpop]] = {
    "<": ast.Lt,
    "<=": ast.LtE,
    "==": ast.Eq,
    "!=": ast.NotEq,
    ">": ast.Gt,
    ">=": ast.GtE,
    "in": ast.In,
    "not in": ast.NotIn,
    "is": ast.Is,
    "is not": ast.IsNot,
}


def conversion_for_format_flags(flags: int) -> int:
    conversion = flags & 0x03

    if conversion == 1:
        return ord("s")
    if conversion == 2:
        return ord("r")
    if conversion == 3:
        return ord("a")
    return -1


def conversion_for_convert_value(arg: int) -> int:
    if arg == 1:
        return ord("s")
    if arg == 2:
        return ord("r")
    if arg == 3:
        return ord("a")
    return -1


def make_joined_formatted_value(
    value: ast.expr,
    conversion: int,
    format_spec: ast.JoinedStr | None,
) -> ast.JoinedStr:
    return ast.JoinedStr(
        values=[
            ast.FormattedValue(
                value=value,
                conversion=conversion,
                format_spec=format_spec,
            )
        ]
    )


def make_converted_formatted_value(value: ast.expr, arg: int) -> ast.FormattedValue:
    return ast.FormattedValue(
        value=value,
        conversion=conversion_for_convert_value(arg),
        format_spec=None,
    )


def make_formatted_with_spec(
    value: ast.expr,
    format_spec: ast.JoinedStr,
) -> ast.JoinedStr:
    if isinstance(value, ast.FormattedValue):
        value.format_spec = format_spec
        return ast.JoinedStr(values=[value])
    return make_joined_formatted_value(value, -1, format_spec)


def make_slice(
    lower: ast.expr | None,
    upper: ast.expr | None,
    step: ast.expr | None = None,
) -> ast.Slice:
    return ast.Slice(lower=lower, upper=upper, step=step)


def make_empty_slice_step() -> ast.Name:
    return make_name(EMPTY_SLICE_STEP_NAME, ast.Load())


def make_subscript(
    value: ast.expr,
    slice_value: ast.expr | ast.Slice,
    ctx: ast.expr_context,
) -> ast.Subscript:
    return ast.Subscript(value=value, slice=slice_value, ctx=ctx)


def make_binary_subscript(value: ast.expr, index: ast.expr) -> ast.Subscript:
    return make_subscript(value, index, ast.Load())


def make_binary_slice(
    value: ast.expr,
    start: ast.expr | None,
    stop: ast.expr | None,
) -> ast.Subscript:
    return make_subscript(value, make_slice(start, stop), ast.Load())


def make_slice_operand(
    count: int,
    start: ast.expr | None,
    stop: ast.expr | None,
    step: ast.expr | None = None,
) -> ast.Slice | None:
    if count == 2:
        return make_slice(start, stop)
    if count == 3:
        return make_slice(start, stop, step)
    return None


def make_legacy_slice_operand(
    mode: str,
    target_value: ast.expr,
    lower: ast.expr | None = None,
    upper: ast.expr | None = None,
) -> tuple[ast.expr, ast.Slice]:
    if mode == "0":
        return target_value, make_slice(None, None)
    if mode == "1":
        return target_value, make_slice(lower, None)
    if mode == "2":
        return target_value, make_slice(None, upper)
    return target_value, make_slice(lower, upper)


def make_attr_load(name: str, target_value: ast.expr) -> ast.Attribute:
    return ast.Attribute(value=target_value, attr=name, ctx=ast.Load())


def make_subscript_store(
    target_value: ast.expr,
    index: ast.expr | ast.Slice,
    value: ast.expr,
) -> ast.Assign:
    target = make_subscript(target_value, index, ast.Store())
    return ast.Assign(targets=[target], value=value)


def make_attr_store(name: str, target_value: ast.expr, value: ast.expr) -> ast.Assign:
    target = ast.Attribute(value=target_value, attr=name, ctx=ast.Store())
    return ast.Assign(targets=[target], value=value)


def make_subscript_delete(
    target_value: ast.expr, index: ast.expr | ast.Slice
) -> ast.Delete:
    target = make_subscript(target_value, index, ast.Del())
    return ast.Delete(targets=[target])


def make_attr_delete(name: str, target_value: ast.expr) -> ast.Delete:
    target = ast.Attribute(value=target_value, attr=name, ctx=ast.Del())
    return ast.Delete(targets=[target])
