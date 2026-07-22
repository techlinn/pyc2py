from typing import Any

from pyc2py.bytecode.opcode_table import normalized_opcode_name

ARG_NOT_RESOLVED = object()


def is_jump_instruction(opcode_table: Any, opcode: int) -> bool:
    return opcode in getattr(opcode_table, "hasjrel", ()) or opcode in getattr(
        opcode_table, "hasjabs", ()
    )


def instruction_size(version: tuple[int, ...], has_arg: bool = True) -> int:
    if version >= (3, 6):
        return 2

    return 3 if has_arg else 1


COMPARE_SYMBOLS = (
    "<",
    "<=",
    "==",
    "!=",
    ">",
    ">=",
    "in",
    "not in",
    "is",
    "is not",
    "exception-match",
)

BINARY_OP_SYMBOLS = (
    "+",
    "&",
    "//",
    "<<",
    "@",
    "*",
    "%",
    "|",
    "**",
    ">>",
    "-",
    "/",
    "^",
    "+=",
    "&=",
    "//=",
    "<<=",
    "@=",
    "*=",
    "%=",
    "|=",
    "**=",
    ">>=",
    "-=",
    "/=",
    "^=",
    "[]",
)

CONVERT_VALUE_NAMES = ("", "str", "repr", "ascii")
COMMON_CONSTANTS_3_14 = (
    "AssertionError",
    "NotImplementedError",
    "tuple",
    "all",
    "any",
)
COMMON_CONSTANTS_3_15 = (
    *COMMON_CONSTANTS_3_14,
    "list",
    "set",
    None,
    "",
    True,
    False,
    -1,
)
INTRINSIC_1_NAMES = (
    "INTRINSIC_1_INVALID",
    "INTRINSIC_PRINT",
    "INTRINSIC_IMPORT_STAR",
    "INTRINSIC_STOPITERATION_ERROR",
    "INTRINSIC_ASYNC_GEN_WRAP",
    "INTRINSIC_UNARY_POSITIVE",
    "INTRINSIC_LIST_TO_TUPLE",
    "INTRINSIC_TYPEVAR",
    "INTRINSIC_PARAMSPEC",
    "INTRINSIC_TYPEVARTUPLE",
    "INTRINSIC_SUBSCRIPT_GENERIC",
    "INTRINSIC_TYPEALIAS",
)
INTRINSIC_2_NAMES = (
    "INTRINSIC_2_INVALID",
    "INTRINSIC_PREP_RERAISE_STAR",
    "INTRINSIC_TYPEVAR_WITH_BOUND",
    "INTRINSIC_TYPEVAR_WITH_CONSTRAINTS",
    "INTRINSIC_SET_FUNCTION_TYPE_PARAMS",
    "INTRINSIC_SET_TYPEPARAM_DEFAULT",
)
SPECIAL_INDEXED_ARG_VALUES = {
    "BINARY_OP": BINARY_OP_SYMBOLS,
    "CONVERT_VALUE": CONVERT_VALUE_NAMES,
    "CALL_INTRINSIC_1": INTRINSIC_1_NAMES,
    "CALL_INTRINSIC_2": INTRINSIC_2_NAMES,
}


def resolve_arg_value(
    code: Any,
    opcode_table: Any,
    opcode: int,
    opname: str,
    arg: int | None,
    offset: int,
    version: tuple[int, ...],
) -> Any:

    if arg is None:
        return None

    normalized_opname = normalized_opcode_name(opname)
    value = resolve_indexed_arg(
        code, opcode_table, opcode, normalized_opname, arg, version
    )
    if value is not ARG_NOT_RESOLVED:
        return value

    value = resolve_special_arg(code, normalized_opname, opname, arg, offset, version)
    if value is not ARG_NOT_RESOLVED:
        return value

    return resolve_jump_arg(opcode_table, opcode, opname, arg, offset, version)


def resolve_indexed_arg(
    code: Any,
    opcode_table: Any,
    opcode: int,
    opname: str,
    arg: int,
    version: tuple[int, ...],
) -> Any:
    if opcode in getattr(opcode_table, "hasconst", ()):
        return read_indexed(getattr(code, "co_consts", ()), arg)
    if opcode in getattr(opcode_table, "hasname", ()):
        return read_indexed(
            getattr(code, "co_names", ()), name_index(opname, arg, version)
        )
    if opcode in getattr(opcode_table, "haslocal", ()):
        return read_indexed(getattr(code, "co_varnames", ()), arg)
    if opcode in getattr(opcode_table, "hasfree", ()):
        return read_indexed(free_names(code, version), arg)
    if opcode in getattr(opcode_table, "hascompare", ()):
        return read_indexed(COMPARE_SYMBOLS, compare_index(arg, version))
    return ARG_NOT_RESOLVED


def resolve_special_arg(
    code: Any,
    normalized_opname: str,
    raw_opname: str,
    arg: int,
    offset: int,
    version: tuple[int, ...],
) -> Any:
    if version[:2] == (3, 12) and raw_opname in WORDCODE_SUPERINSTRUCTIONS:
        return superinstruction_values(code, offset, raw_opname, arg)
    if normalized_opname in PACKED_LOCAL_OPS:
        return packed_local_names(getattr(code, "co_varnames", ()), arg)
    if normalized_opname == "LOAD_COMMON_CONSTANT":
        return read_indexed(common_constants(version), arg)
    if normalized_opname == "LOAD_SMALL_INT":
        return arg
    values = SPECIAL_INDEXED_ARG_VALUES.get(normalized_opname)
    if values is None:
        return ARG_NOT_RESOLVED
    return read_indexed(values, arg)


def resolve_jump_arg(
    opcode_table: Any,
    opcode: int,
    opname: str,
    arg: int,
    offset: int,
    version: tuple[int, ...],
) -> Any:
    if opcode in getattr(opcode_table, "hasjrel", ()):
        return relative_target(opname, offset, arg, version)
    if opcode in getattr(opcode_table, "hasjabs", ()):
        return absolute_target(arg, version)
    return arg


def read_indexed(values: Any, index: int) -> Any:
    if index < 0:
        return index
    try:
        return values[index]
    except (IndexError, TypeError):
        return index


WORDCODE_SUPERINSTRUCTIONS = frozenset(
    {
        "LOAD_CONST__LOAD_FAST",
        "LOAD_FAST__LOAD_CONST",
        "LOAD_FAST__LOAD_FAST",
        "STORE_FAST__LOAD_FAST",
        "STORE_FAST__STORE_FAST",
    }
)

PACKED_LOCAL_OPS = frozenset(
    {
        "LOAD_FAST_LOAD_FAST",
        "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
        "STORE_FAST_LOAD_FAST",
        "STORE_FAST_STORE_FAST",
    }
)


def superinstruction_values(
    code: Any,
    offset: int,
    opname: str,
    first_arg: int,
) -> tuple[Any, Any]:
    second_arg = following_wordcode_arg(code, offset)
    if second_arg is None:
        return first_arg, first_arg

    consts = getattr(code, "co_consts", ())
    varnames = getattr(code, "co_varnames", ())
    if opname == "LOAD_CONST__LOAD_FAST":
        return read_indexed(consts, first_arg), read_indexed(varnames, second_arg)
    if opname == "LOAD_FAST__LOAD_CONST":
        return read_indexed(varnames, first_arg), read_indexed(consts, second_arg)
    return read_indexed(varnames, first_arg), read_indexed(varnames, second_arg)


def following_wordcode_arg(code: Any, offset: int) -> int | None:
    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    next_arg_offset = offset + 3
    if next_arg_offset >= len(code_bytes):
        return None
    return code_bytes[next_arg_offset]


def packed_local_names(values: Any, arg: int) -> tuple[Any, Any]:
    high_index, low_index = packed_local_indexes(arg)
    return (
        read_indexed(values, high_index),
        read_indexed(values, low_index),
    )


def packed_local_indexes(arg: int) -> tuple[int, int]:
    return arg >> 4, arg & 0x0F


def common_constants(version: tuple[int, ...]) -> tuple[Any, ...]:
    if version >= (3, 15):
        return COMMON_CONSTANTS_3_15
    return COMMON_CONSTANTS_3_14


def free_names(
    code: Any,
    version: tuple[int, ...] | None = None,
) -> tuple[Any, ...]:
    if version is not None and version >= (3, 11):
        names = tuple(getattr(code, "co_localsplusnames", ()) or ())
        if names:
            return names
        return (
            *tuple(getattr(code, "co_varnames", ()) or ()),
            *tuple(getattr(code, "co_cellvars", ()) or ()),
            *tuple(getattr(code, "co_freevars", ()) or ()),
        )
    return tuple(getattr(code, "co_cellvars", ()) or ()) + tuple(
        getattr(code, "co_freevars", ()) or ()
    )


def name_index(opname: str, arg: int, version: tuple[int, ...]) -> int:
    opname = normalized_opcode_name(opname)

    if version >= (3, 11) and opname == "LOAD_GLOBAL":
        return arg >> 1

    if version >= (3, 12) and opname == "LOAD_ATTR":
        return arg >> 1

    if version >= (3, 12) and opname == "LOAD_SUPER_ATTR":
        return arg >> 2

    if version >= (3, 15) and opname == "IMPORT_NAME":
        return arg >> 2
    return arg


def compare_index(arg: int, version: tuple[int, ...]) -> int:
    if version >= (3, 13):
        return arg >> 5
    if version >= (3, 12):
        return arg >> 4
    return arg


def absolute_target(arg: int, version: tuple[int, ...]) -> int:
    if version >= (3, 10):
        return arg * 2
    return arg


def relative_target(
    opname: str, offset: int, arg: int, version: tuple[int, ...]
) -> int:
    normalized_opname = normalized_opcode_name(opname)
    cache_width = inline_cache_entries(normalized_opname, version) * instruction_size(
        version
    )
    base_offset = offset + instruction_size(version) + cache_width

    if version >= (3, 11) and "BACKWARD" in opname:
        return base_offset - arg * instruction_size(version)

    if version >= (3, 10):
        return base_offset + arg * instruction_size(version)

    if version >= (3, 6):
        return offset + 2 + arg
    return offset + 3 + arg


def inline_cache_entries(opname: str, version: tuple[int, ...]) -> int:
    if version < (3, 12):
        return 0

    normalized_opname = normalized_opcode_name(opname)
    if normalized_opname in {"FOR_ITER", "SEND"}:
        return 1

    if version >= (3, 13) and normalized_opname in {
        "JUMP_BACKWARD",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_NONE",
        "POP_JUMP_IF_NOT_NONE",
        "POP_JUMP_IF_TRUE",
    }:
        return 1
    return 0


def format_argrepr(opname: str, argval: Any) -> str:
    if argval is None:
        return ""

    normalized_opname = normalized_opcode_name(opname)

    if normalized_opname in {"LOAD_CONST", "RETURN_CONST"}:
        return repr(argval)

    if normalized_opname in {
        "LOAD_CONST_LOAD_FAST",
        "LOAD_FAST_LOAD_CONST",
        "LOAD_FAST_LOAD_FAST",
        "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
        "STORE_FAST_LOAD_FAST",
        "STORE_FAST_STORE_FAST",
    }:
        if isinstance(argval, tuple) and len(argval) == 2:
            return f"{argval[0]}, {argval[1]}"
        return str(argval)

    if "JUMP" in normalized_opname or normalized_opname in {
        "FOR_ITER",
        "SETUP_LOOP",
        "SETUP_EXCEPT",
        "SETUP_FINALLY",
    }:
        return f"to {argval}"

    return str(argval)
