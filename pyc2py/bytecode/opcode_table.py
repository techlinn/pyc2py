from dataclasses import dataclass
from importlib import import_module

HAVE_ARGUMENT = 90
SUPPORTED_VERSIONS = (
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 15),
)

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
    "ENTER_EXECUTOR",
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
CANONICAL_CALL_OPS = {
    "CALL",
    "CALL_FUNCTION",
    "CALL_FUNCTION_EX",
    "CALL_FUNCTION_KW",
    "CALL_FUNCTION_VAR",
    "CALL_FUNCTION_VAR_KW",
    "CALL_KW",
    "CALL_METHOD",
}
NORMALIZED_OPCODE_NAMES = {
    "INSTRUMENTED_INSTRUCTION": "NOP",
    "INSTRUMENTED_LINE": "NOP",
    "LOAD_CONST__LOAD_FAST": "LOAD_CONST_LOAD_FAST",
    "LOAD_FAST__LOAD_CONST": "LOAD_FAST_LOAD_CONST",
    "LOAD_FAST__LOAD_FAST": "LOAD_FAST_LOAD_FAST",
    "STORE_FAST__LOAD_FAST": "STORE_FAST_LOAD_FAST",
    "STORE_FAST__STORE_FAST": "STORE_FAST_STORE_FAST",
}
NORMALIZED_OPCODE_PREFIXES = (
    ("BINARY_OP_SUBSCR_", "BINARY_SUBSCR"),
    ("BINARY_OP_", "BINARY_OP"),
    ("BINARY_SUBSCR_", "BINARY_SUBSCR"),
    ("CALL_EX_", "CALL_FUNCTION_EX"),
    ("CALL_KW_", "CALL_KW"),
    ("CALL_", "CALL"),
    ("COMPARE_OP_", "COMPARE_OP"),
    ("CONTAINS_OP_", "CONTAINS_OP"),
    ("FOR_ITER_", "FOR_ITER"),
    ("GET_ITER_", "GET_ITER"),
    ("JUMP_BACKWARD_", "JUMP_BACKWARD"),
    ("LOAD_ATTR_", "LOAD_ATTR"),
    ("LOAD_CONST_", "LOAD_CONST"),
    ("LOAD_GLOBAL_", "LOAD_GLOBAL"),
    ("LOAD_SUPER_ATTR_", "LOAD_SUPER_ATTR"),
    ("RESUME_CHECK", "RESUME"),
    ("SEND_", "SEND"),
    ("STORE_ATTR_", "STORE_ATTR"),
    ("STORE_SUBSCR_", "STORE_SUBSCR"),
    ("TO_BOOL_", "TO_BOOL"),
    ("UNPACK_SEQUENCE_", "UNPACK_SEQUENCE"),
)
RELATIVE_JUMPS = {
    "FOR_ITER",
    "FOR_LOOP",
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
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
    "SETUP_LOOP",
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


def get_opcode_table(version: tuple[int, ...]) -> OpcodeTable:
    resolved_version = resolve_supported_version(version)
    module = import_module(version_module_name(resolved_version))

    return build_opcode_table(module.OPMAP, resolved_version)


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


def version_module_name(version: tuple[int, int]) -> str:
    return f"pyc2py.bytecode.versions.python_{version[0]}_{version[1]}"


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
    if version >= (3, 15):
        names.add("GET_ITER")

    return names


def normalized_opcode_name(name: str) -> str:
    normalized_name = NORMALIZED_OPCODE_NAMES.get(name)
    alias = PSEUDO_OPCODE_ALIASES.get(name)
    if normalized_name is None and alias is not None:
        normalized_name = normalized_opcode_name(alias)
    elif normalized_name is None and name.startswith("INSTRUMENTED_"):
        normalized_name = normalized_opcode_name(name[len("INSTRUMENTED_") :])
    elif normalized_name is None and name.endswith("_A"):
        base_name = name[:-2]
        normalized_base = normalized_opcode_name(base_name)
        if base_name in A_SUFFIX_OPS or normalized_base in A_SUFFIX_OPS:
            normalized_name = normalized_base
    elif normalized_name is None and (
        name.startswith("CALL_INTRINSIC_") or name in CANONICAL_CALL_OPS
    ):
        normalized_name = name
    elif normalized_name is None and name.startswith("POP_JUMP_FORWARD_"):
        normalized_name = "POP_JUMP_" + name.removeprefix("POP_JUMP_FORWARD_")
    elif normalized_name is None and name.startswith("POP_JUMP_BACKWARD_"):
        normalized_name = "POP_JUMP_" + name.removeprefix("POP_JUMP_BACKWARD_")

    if normalized_name is None:
        normalized_name = prefixed_opcode_name(name)

    return normalized_name


def prefixed_opcode_name(name: str) -> str:
    for prefix, opcode_name in NORMALIZED_OPCODE_PREFIXES:
        if name.startswith(prefix):
            return opcode_name
    return name


def relative_jump_names(version: tuple[int, int]) -> set[str]:
    names = set(RELATIVE_JUMPS)
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
