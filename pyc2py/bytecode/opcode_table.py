from dataclasses import dataclass, replace
from functools import cache

from pyc2py.bytecode.formats import pypy_format, releases

HAVE_ARGUMENT = 90
SUPPORTED_VERSIONS = tuple(releases())

CONST_OPS = {"LOAD_CONST", "RETURN_CONST", "KW_NAMES"}
NAME_OPS = {
    "DELETE_ATTR",
    "DELETE_GLOBAL",
    "DELETE_NAME",
    "IMPORT_FROM",
    "IMPORT_NAME",
    "LOAD_ATTR",
    "LOAD_FROM_DICT_OR_GLOBALS",
    "LOAD_GLOBAL",
    "LOAD_METHOD",
    "LOAD_SUPER_ATTR",
    "LOAD_NAME",
    # PyPy's
    "LOOKUP_METHOD",
    "STORE_ANNOTATION",
    "STORE_ATTR",
    "STORE_GLOBAL",
    "STORE_NAME",
}
LOCAL_OPS = {
    "DELETE_FAST",
    "LOAD_FAST",
    "LOAD_FAST_AND_CLEAR",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "STORE_FAST",
}
FREE_OPS = {
    "DELETE_DEREF",
    "LOAD_CLASSDEREF",
    "LOAD_CLOSURE",
    "LOAD_DEREF",
    "LOAD_FROM_DICT_OR_DEREF",
    "MAKE_CELL",
    "STORE_DEREF",
}
COMPARE_OPS = {"COMPARE_OP"}
ARG_OPS = {
    "BINARY_OP",
    "BUILD_CONST_KEY_MAP",
    "BUILD_LIST",
    "BUILD_MAP",
    "BUILD_SET",
    "BUILD_SLICE",
    "BUILD_STRING",
    "BUILD_INTERPOLATION",
    "BUILD_TUPLE",
    "CALL",
    "CALL_FUNCTION_EX",
    "CALL_INTRINSIC_1",
    "CALL_INTRINSIC_2",
    "CALL_KW",
    "COMPARE_OP",
    "CONTAINS_OP",
    "CONVERT_VALUE",
    "COPY",
    "COPY_FREE_VARS",
    "DELETE_ATTR",
    "DELETE_DEREF",
    "DELETE_FAST",
    "DELETE_GLOBAL",
    "DELETE_NAME",
    "DICT_MERGE",
    "DICT_UPDATE",
    "EXTENDED_ARG",
    "FOR_ITER",
    "GEN_START",
    "GET_AWAITABLE",
    "IMPORT_FROM",
    "IMPORT_NAME",
    "IS_OP",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
    "LIST_APPEND",
    "LIST_EXTEND",
    "LOAD_ATTR",
    "LOAD_CONST",
    "LOAD_CONST_LOAD_FAST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_FAST_AND_CLEAR",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "LOAD_FAST_LOAD_CONST",
    "LOAD_FAST_LOAD_FAST",
    "LOAD_COMMON_CONSTANT",
    "LOAD_FROM_DICT_OR_DEREF",
    "LOAD_FROM_DICT_OR_GLOBALS",
    "LOAD_GLOBAL",
    "LOAD_NAME",
    "LOAD_SMALL_INT",
    "LOAD_SPECIAL",
    "LOAD_SUPER_ATTR",
    "MAKE_FUNCTION",
    "MAKE_CELL",
    "MAP_ADD",
    "MATCH_CLASS",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_IF_TRUE",
    "RAISE_VARARGS",
    "RERAISE",
    "ROT_N",
    "RETURN_CONST",
    "SEND",
    "SET_ADD",
    "SET_FUNCTION_ATTRIBUTE",
    "SET_UPDATE",
    "STORE_ATTR",
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_FAST_LOAD_FAST",
    "STORE_FAST_STORE_FAST",
    "STORE_GLOBAL",
    "STORE_NAME",
    "SWAP",
    "UNPACK_EX",
    "UNPACK_SEQUENCE",
    "YIELD_VALUE",
}
A_SUFFIX_OPS = ARG_OPS | {"RETURN_GENERATOR"}
PSEUDO_OPCODE_ALIASES = {
    "JUMP": "JUMP_FORWARD",
    "JUMP_NO_INTERRUPT": "JUMP_BACKWARD_NO_INTERRUPT",
    "LOAD_SUPER_METHOD": "LOAD_SUPER_ATTR",
    "LOAD_ZERO_SUPER_ATTR": "LOAD_SUPER_ATTR",
    "LOAD_ZERO_SUPER_METHOD": "LOAD_SUPER_ATTR",
    "SETUP_CLEANUP": "SETUP_FINALLY",
    "STORE_FAST_MAYBE_NULL": "STORE_FAST",
}
RELATIVE_JUMPS = {
    "CALL_FINALLY",
    "FOR_ITER",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
    "JUMP_IF_FALSE",
    "JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_NONE",
    "POP_JUMP_FORWARD_IF_NOT_NONE",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_NONE",
    "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE",
    "SEND",
    "SETUP_ASYNC_WITH",
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
    "SETUP_LOOP",
    "SETUP_WITH",
}
ABSOLUTE_JUMPS = {
    "CONTINUE_LOOP",
    "JUMP_ABSOLUTE",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_NOT_EXC_MATCH",
    "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
}


@dataclass(frozen=True, slots=True)
class OpcodeTable:
    opname: tuple[str, ...]
    HAVE_ARGUMENT: int
    hasconst: frozenset[int]
    hasname: frozenset[int]
    haslocal: frozenset[int]
    hasfree: frozenset[int]
    hascompare: frozenset[int]
    hasarg: frozenset[int]
    hasjrel: frozenset[int]
    hasjabs: frozenset[int]
    # the CPython version whose argument and jump rules the bytes follow
    layout: tuple[int, int]


def get_opcode_table(version: tuple[int, ...], opcodes: str = "") -> OpcodeTable:
    # opcodes: the module of a compiler that is not CPython's, such as PyPy's
    if opcodes:
        return cached_format_table(opcodes)

    return cached_opcode_table(resolve_supported_version(version))


@cache
def cached_format_table(opcodes: str) -> OpcodeTable:
    found = pypy_format(opcodes)
    table = build_opcode_table(found.opmap, found.layout)
    return replace(
        table,
        hasarg=table.hasarg | opcodes_named(found.opmap, set(found.argument_ops)),
        hasjrel=opcodes_named(found.opmap, set(found.relative_jumps)),
        hasjabs=opcodes_named(found.opmap, set(found.absolute_jumps)),
    )


@cache
def cached_opcode_table(resolved_version: tuple[int, int]) -> OpcodeTable:
    return build_opcode_table(releases()[resolved_version].opmap, resolved_version)


def build_opcode_table(
    opmap: dict[int, str],
    version: tuple[int, int] | None = None,
) -> OpcodeTable:
    resolved_version = version or (0, 0)
    return OpcodeTable(
        opname=make_opname(opmap),
        HAVE_ARGUMENT=HAVE_ARGUMENT,
        hasconst=opcodes_named(opmap, CONST_OPS),
        hasname=opcodes_named(opmap, NAME_OPS),
        haslocal=opcodes_named(opmap, LOCAL_OPS),
        hasfree=opcodes_named(opmap, FREE_OPS),
        hascompare=opcodes_named(opmap, COMPARE_OPS),
        hasarg=argument_opcodes(opmap, resolved_version),
        hasjrel=opcodes_named(opmap, relative_jump_names(resolved_version)),
        hasjabs=opcodes_named(opmap, absolute_jump_names(resolved_version)),
        layout=resolved_version,
    )


def resolve_supported_version(version: tuple[int, ...]) -> tuple[int, int]:
    major_minor = normalize_version(version)
    if major_minor in SUPPORTED_VERSIONS:
        return major_minor

    supported = [item for item in SUPPORTED_VERSIONS if item <= major_minor]
    if supported:
        return supported[-1]

    return SUPPORTED_VERSIONS[0]


def version_resolution_warning(version: tuple[int, ...]) -> str | None:
    requested = normalize_version(version)
    resolved = resolve_supported_version(requested)
    if requested == resolved:
        return None

    return (
        f"bytecode version {format_version(requested)} is not directly supported; "
        f"using opcode table {format_version(resolved)}"
    )


def normalize_version(version: tuple[int, ...]) -> tuple[int, int]:
    if len(version) < 2:
        raise ValueError("bytecode version must include major and minor")

    return int(version[0]), int(version[1])


def format_version(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def make_opname(opmap: dict[int, str]) -> tuple[str, ...]:
    result = [f"<{opcode}>" for opcode in range(256)]

    for opcode, name in opmap.items():
        if 0 <= opcode < len(result):
            result[opcode] = name

    return tuple(result)


def opcodes_named(opmap: dict[int, str], names: set[str]) -> frozenset[int]:
    return frozenset(
        opcode
        for opcode, name in opmap.items()
        if normalized_opcode_name(name) in names
    )


def argument_opcodes(opmap: dict[int, str], version: tuple[int, int]) -> frozenset[int]:
    if version < (3, 12):
        return frozenset(opcode for opcode in opmap if opcode >= HAVE_ARGUMENT)

    arg_ops = argument_opcode_names(version)

    if version < (3, 13):
        return frozenset(
            opcode
            for opcode, name in opmap.items()
            if opcode >= HAVE_ARGUMENT or normalized_opcode_name(name) in arg_ops
        )

    return frozenset(
        opcode
        for opcode, name in opmap.items()
        if normalized_opcode_name(name) in arg_ops
    )


def argument_opcode_names(version: tuple[int, int]) -> set[str]:
    names = set(ARG_OPS)

    if version >= (3, 13):
        names.add("RESUME")
        names.discard("MAKE_FUNCTION")

    if version >= (3, 14):
        names.add("END_ASYNC_FOR")
        names.discard("CALL_FUNCTION_EX")

    if version >= (3, 15):
        names.add("GET_ITER")

    return names


@cache
def normalized_opcode_name(name: str) -> str:
    alias = PSEUDO_OPCODE_ALIASES.get(name)
    if alias is not None:
        return normalized_opcode_name(alias)

    if name.endswith("_A"):
        base_name = name[:-2]
        normalized_base = normalized_opcode_name(base_name)
        if base_name in A_SUFFIX_OPS or normalized_base in A_SUFFIX_OPS:
            return normalized_base

    if name.startswith("POP_JUMP_FORWARD_"):
        return "POP_JUMP_" + name.removeprefix("POP_JUMP_FORWARD_")
    if name.startswith("POP_JUMP_BACKWARD_"):
        return "POP_JUMP_" + name.removeprefix("POP_JUMP_BACKWARD_")

    return name


def relative_jump_names(version: tuple[int, int]) -> set[str]:
    names = set(RELATIVE_JUMPS)

    if version >= (3, 14):
        names.add("END_ASYNC_FOR")
    if version >= (3, 11):
        names.update(
            {
                "JUMP_IF_FALSE_OR_POP",
                "JUMP_IF_TRUE_OR_POP",
                "POP_JUMP_IF_FALSE",
                "POP_JUMP_IF_NONE",
                "POP_JUMP_IF_NOT_NONE",
                "POP_JUMP_IF_TRUE",
            }
        )

    return names


def absolute_jump_names(version: tuple[int, int]) -> set[str]:
    names = set(ABSOLUTE_JUMPS)

    if version >= (3, 11):
        names.discard("JUMP_IF_FALSE_OR_POP")
        names.discard("JUMP_IF_TRUE_OR_POP")
        names.discard("POP_JUMP_IF_FALSE")
        names.discard("POP_JUMP_IF_TRUE")

    return names


def is_jump_opname(opname: str) -> bool:
    return "JUMP" in opname or opname in {"FOR_ITER", "CONTINUE_LOOP"}


def is_unknown_opname(opname: str) -> bool:
    return opname.startswith("<") and opname.endswith(">")


# unconditional jumps by the way they can go
FORWARD_JUMPS = frozenset({"JUMP", "JUMP_FORWARD"})
BACKWARD_JUMPS = frozenset({"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"})
# before 3.11 a jump back to a loop's head is JUMP_ABSOLUTE
LOOP_BACK_JUMPS = frozenset({"JUMP_BACKWARD", "JUMP_ABSOLUTE"})
