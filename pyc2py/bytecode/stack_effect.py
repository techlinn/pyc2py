from dataclasses import dataclass
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name

@dataclass(frozen=True, slots=True)
class BytecodeStackEffect:
    pops: int
    pushes: int

    @property
    def net(self) -> int:
        return self.pushes - self.pops

LOAD_EFFECT = BytecodeStackEffect(pops=0, pushes=1)
STORE_EFFECT = BytecodeStackEffect(pops=1, pushes=0)
UNARY_EFFECT = BytecodeStackEffect(pops=1, pushes=1)
BINARY_EFFECT = BytecodeStackEffect(pops=2, pushes=1)
NO_NET_EFFECT = BytecodeStackEffect(pops=0, pushes=0)

LOAD_OPS = (
    "LOAD_CLASSDEREF",
    "LOAD_CLOSURE",
    "LOAD_CONST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_FAST_AND_CLEAR",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "LOAD_COMMON_CONSTANT",
    "LOAD_GLOBAL",
    "LOAD_NAME",
    "LOAD_SMALL_INT",
)

STORE_OPS = (
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_GLOBAL",
    "STORE_NAME",
)

UNARY_OPS = (
    "UNARY_CONVERT",
    "UNARY_INVERT",
    "UNARY_NEGATIVE",
    "UNARY_NOT",
    "UNARY_POSITIVE",
    "TO_BOOL",
)

BINARY_OPS = (
    "BINARY_ADD",
    "BINARY_AND",
    "BINARY_DIVIDE",
    "BINARY_FLOOR_DIVIDE",
    "BINARY_LSHIFT",
    "BINARY_MATRIX_MULTIPLY",
    "BINARY_MODULO",
    "BINARY_MULTIPLY",
    "BINARY_OP",
    "BINARY_OR",
    "BINARY_POWER",
    "BINARY_RSHIFT",
    "BINARY_SUBSCR",
    "BINARY_SUBTRACT",
    "BINARY_TRUE_DIVIDE",
    "BINARY_XOR",
    "COMPARE_OP",
    "CONTAINS_OP",
    "INPLACE_ADD",
    "INPLACE_AND",
    "INPLACE_DIVIDE",
    "INPLACE_FLOOR_DIVIDE",
    "INPLACE_LSHIFT",
    "INPLACE_MATRIX_MULTIPLY",
    "INPLACE_MODULO",
    "INPLACE_MULTIPLY",
    "INPLACE_OR",
    "INPLACE_POWER",
    "INPLACE_RSHIFT",
    "INPLACE_SUBTRACT",
    "INPLACE_TRUE_DIVIDE",
    "INPLACE_XOR",
    "IS_OP",
)

FIXED_STACK_EFFECTS: dict[str, BytecodeStackEffect] = {
    **dict.fromkeys(LOAD_OPS, LOAD_EFFECT),
    **dict.fromkeys(STORE_OPS, STORE_EFFECT),
    **dict.fromkeys(UNARY_OPS, UNARY_EFFECT),
    **dict.fromkeys(BINARY_OPS, BINARY_EFFECT),
    "ASYNC_GEN_WRAP": UNARY_EFFECT,
    "BINARY_SLICE": BytecodeStackEffect(pops=3, pushes=1),
    "BEFORE_ASYNC_WITH": BytecodeStackEffect(pops=1, pushes=2),
    "BEFORE_WITH": BytecodeStackEffect(pops=1, pushes=2),
    "BUILD_CLASS": BytecodeStackEffect(pops=3, pushes=1),
    "CACHE": NO_NET_EFFECT,
    "CALL_INTRINSIC_1": BytecodeStackEffect(pops=1, pushes=1),
    "CALL_INTRINSIC_2": BytecodeStackEffect(pops=2, pushes=1),
    "CHECK_EG_MATCH": BytecodeStackEffect(pops=2, pushes=2),
    "CHECK_EXC_MATCH": BytecodeStackEffect(pops=2, pushes=2),
    "CLEANUP_THROW": BytecodeStackEffect(pops=2, pushes=1),
    "CONVERT_VALUE": UNARY_EFFECT,
    "COPY": BytecodeStackEffect(pops=0, pushes=1),
    "COPY_DICT_WITHOUT_KEYS": BytecodeStackEffect(pops=1, pushes=1),
    "DUP_TOP": BytecodeStackEffect(pops=1, pushes=2),
    "DUP_TOP_TWO": BytecodeStackEffect(pops=2, pushes=4),
    "DELETE_ATTR": BytecodeStackEffect(pops=1, pushes=0),
    "DELETE_DEREF": BytecodeStackEffect(pops=0, pushes=0),
    "DELETE_FAST": BytecodeStackEffect(pops=0, pushes=0),
    "DELETE_GLOBAL": BytecodeStackEffect(pops=0, pushes=0),
    "DELETE_NAME": BytecodeStackEffect(pops=0, pushes=0),
    "DELETE_SUBSCR": BytecodeStackEffect(pops=2, pushes=0),
    "END_FOR": BytecodeStackEffect(pops=2, pushes=0),
    "END_ASYNC_FOR": BytecodeStackEffect(pops=2, pushes=0),
    "END_SEND": BytecodeStackEffect(pops=1, pushes=0),
    "ENTER_EXECUTOR": NO_NET_EFFECT,
    "EXIT_INIT_CHECK": NO_NET_EFFECT,
    "EXTENDED_ARG": NO_NET_EFFECT,
    "FORMAT_SIMPLE": BytecodeStackEffect(pops=1, pushes=1),
    "FORMAT_WITH_SPEC": BytecodeStackEffect(pops=2, pushes=1),
    "FOR_ITER": BytecodeStackEffect(pops=0, pushes=1),
    "GEN_START": BytecodeStackEffect(pops=1, pushes=0),
    "GET_AWAITABLE": BytecodeStackEffect(pops=1, pushes=1),
    "GET_AITER": BytecodeStackEffect(pops=1, pushes=1),
    "GET_ANEXT": BytecodeStackEffect(pops=1, pushes=2),
    "GET_ITER": BytecodeStackEffect(pops=1, pushes=1),
    "GET_LEN": BytecodeStackEffect(pops=0, pushes=1),
    "GET_YIELD_FROM_ITER": BytecodeStackEffect(pops=1, pushes=1),
    "EXEC_STMT": BytecodeStackEffect(pops=3, pushes=0),
    "IMPORT_FROM": BytecodeStackEffect(pops=0, pushes=1),
    "IMPORT_STAR": BytecodeStackEffect(pops=1, pushes=0),
    "INTERPRETER_EXIT": BytecodeStackEffect(pops=1, pushes=0),
    "JUMP_ABSOLUTE": NO_NET_EFFECT,
    "JUMP_BACKWARD": NO_NET_EFFECT,
    "JUMP_BACKWARD_NO_INTERRUPT": NO_NET_EFFECT,
    "JUMP_FORWARD": NO_NET_EFFECT,
    "JUMP_IF_NOT_EXC_MATCH": BytecodeStackEffect(pops=2, pushes=0),
    "KW_NAMES": NO_NET_EFFECT,
    "LIST_APPEND": BytecodeStackEffect(pops=1, pushes=0),
    "LIST_EXTEND": BytecodeStackEffect(pops=1, pushes=0),
    "LIST_TO_TUPLE": UNARY_EFFECT,
    "LOAD_ASSERTION_ERROR": BytecodeStackEffect(pops=0, pushes=1),
    "LOAD_ATTR": BytecodeStackEffect(pops=1, pushes=1),
    "LOAD_BUILD_CLASS": BytecodeStackEffect(pops=0, pushes=1),
    "LOAD_CONST_LOAD_FAST": BytecodeStackEffect(pops=0, pushes=2),
    "LOAD_FAST_LOAD_CONST": BytecodeStackEffect(pops=0, pushes=2),
    "LOAD_FAST_LOAD_FAST": BytecodeStackEffect(pops=0, pushes=2),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": BytecodeStackEffect(pops=0, pushes=2),
    "LOAD_FROM_DICT_OR_DEREF": BytecodeStackEffect(pops=1, pushes=1),
    "LOAD_FROM_DICT_OR_GLOBALS": BytecodeStackEffect(pops=1, pushes=1),
    "LOAD_LOCALS": BytecodeStackEffect(pops=0, pushes=1),
    "LOAD_SPECIAL": BytecodeStackEffect(pops=1, pushes=2),
    "LOAD_SUPER_ATTR": BytecodeStackEffect(pops=3, pushes=1),
    "MATCH_KEYS": BytecodeStackEffect(pops=0, pushes=1),
    "MATCH_MAPPING": BytecodeStackEffect(pops=0, pushes=1),
    "MATCH_SEQUENCE": BytecodeStackEffect(pops=0, pushes=1),
    "MATCH_CLASS": BytecodeStackEffect(pops=3, pushes=1),
    "MAKE_CELL": NO_NET_EFFECT,
    "MAP_ADD": BytecodeStackEffect(pops=2, pushes=0),
    "NOP": NO_NET_EFFECT,
    "NOT_TAKEN": NO_NET_EFFECT,
    "PRECALL": NO_NET_EFFECT,
    "PRINT_ITEM": BytecodeStackEffect(pops=1, pushes=0),
    "PRINT_ITEM_TO": BytecodeStackEffect(pops=2, pushes=0),
    "PRINT_NEWLINE": NO_NET_EFFECT,
    "PRINT_NEWLINE_TO": BytecodeStackEffect(pops=1, pushes=0),
    "POP_BLOCK": NO_NET_EFFECT,
    "POP_EXCEPT": BytecodeStackEffect(pops=1, pushes=0),
    "POP_ITER": BytecodeStackEffect(pops=1, pushes=0),
    "POP_TOP": BytecodeStackEffect(pops=1, pushes=0),
    "PREP_RERAISE_STAR": BytecodeStackEffect(pops=2, pushes=1),
    "PUSH_EXC_INFO": BytecodeStackEffect(pops=1, pushes=2),
    "PUSH_NULL": BytecodeStackEffect(pops=0, pushes=1),
    "RETURN_CONST": BytecodeStackEffect(pops=0, pushes=0),
    "RETURN_GENERATOR": NO_NET_EFFECT,
    "RESUME": NO_NET_EFFECT,
    "SEND": BytecodeStackEffect(pops=1, pushes=1),
    "ROT_FOUR": NO_NET_EFFECT,
    "ROT_N": NO_NET_EFFECT,
    "ROT_THREE": NO_NET_EFFECT,
    "ROT_TWO": NO_NET_EFFECT,
    "SET_ADD": BytecodeStackEffect(pops=1, pushes=0),
    "SET_UPDATE": BytecodeStackEffect(pops=1, pushes=0),
    "SETUP_ANNOTATIONS": NO_NET_EFFECT,
    "WITH_EXCEPT_START": BytecodeStackEffect(pops=0, pushes=1),
    "COPY_FREE_VARS": NO_NET_EFFECT,
    "STORE_ATTR": BytecodeStackEffect(pops=2, pushes=0),
    "STORE_FAST_LOAD_FAST": BytecodeStackEffect(pops=1, pushes=1),
    "STORE_FAST_STORE_FAST": BytecodeStackEffect(pops=2, pushes=0),
    "STORE_SLICE": BytecodeStackEffect(pops=4, pushes=0),
    "STORE_SUBSCR": BytecodeStackEffect(pops=3, pushes=0),
    "SWAP": NO_NET_EFFECT,
    "TRACE_RECORD": NO_NET_EFFECT,
    "YIELD_VALUE": BytecodeStackEffect(pops=1, pushes=1),
    "RETURN_VALUE": BytecodeStackEffect(pops=1, pushes=0),
    "DICT_MERGE": BytecodeStackEffect(pops=1, pushes=0),
    "DICT_UPDATE": BytecodeStackEffect(pops=1, pushes=0),
}

def fixed_stack_effect(opname: str) -> BytecodeStackEffect | None:
    return FIXED_STACK_EFFECTS.get(opname)

def instruction_stack_effect(
    instruction: Instruction,
    version: tuple[int, ...] | None = None,
) -> BytecodeStackEffect | None:
    arg = int(instruction.arg or 0)
    original_opname = instruction.opname
    opname = stack_effect_opname(original_opname, version)
    effect = attribute_stack_effect(opname, original_opname, arg, version)
    if effect is not None:
        return effect
    effect = versioned_stack_effect(opname, version)
    if effect is not None:
        return effect
    effect = jump_stack_effect(opname)
    if effect is not None:
        return effect
    for reader in STACK_EFFECT_READERS:
        effect = reader(opname, arg)
        if effect is not None:
            return effect
    effect = call_stack_effect(opname, arg, version)
    if effect is not None:
        return effect
    return fixed_stack_effect(opname)

def stack_effect_opname(
    opname: str,
    version: tuple[int, ...] | None = None,
) -> str:
    if opname == "UNPACK_SEQUENCE_TWO_TUPLE":
        return opname
    if (
        version is not None
        and version >= (3, 13)
        and opname.startswith("CALL_")
        and opname.endswith("_WITH_KEYWORDS")
    ):
        return "CALL_KW"
    return normalized_opcode_name(opname)

def attribute_stack_effect(
    opname: str,
    original_opname: str,
    arg: int,
    version: tuple[int, ...] | None,
) -> BytecodeStackEffect | None:
    if opname == "LOAD_ATTR":
        arg = load_attr_stack_arg(original_opname, arg)
        return BytecodeStackEffect(
            pops=1,
            pushes=1 + int(supports_load_attr_call_shape(version) and arg & 1),
        )

    if opname == "LOAD_GLOBAL":
        return BytecodeStackEffect(
            pops=0,
            pushes=1 + int(supports_load_global_null(version) and arg & 1),
        )

    if opname == "LOAD_SUPER_ATTR":
        arg = load_super_attr_stack_arg(original_opname, arg)
        return BytecodeStackEffect(pops=3, pushes=1 + int(arg & 1))
    return None

def load_attr_stack_arg(opname: str, arg: int) -> int:
    if opname.startswith("LOAD_ATTR_METHOD_"):
        return arg | 1
    return arg

def load_super_attr_stack_arg(opname: str, arg: int) -> int:
    if opname == "LOAD_SUPER_ATTR_METHOD":
        return arg | 1
    return arg

def supports_load_attr_call_shape(version: tuple[int, ...] | None) -> bool:
    return version is None or version >= (3, 12)

def supports_load_global_null(version: tuple[int, ...] | None) -> bool:
    return version is None or version >= (3, 11)

def versioned_stack_effect(
    opname: str,
    version: tuple[int, ...] | None,
) -> BytecodeStackEffect | None:
    if version is None:
        return None

    if opname == "END_FOR" and version >= (3, 14):
        return BytecodeStackEffect(pops=1, pushes=0)

    return None

def jump_stack_effect(opname: str) -> BytecodeStackEffect | None:
    if opname.startswith("POP_JUMP"):
        return BytecodeStackEffect(pops=1, pushes=0)
    return None

BUILD_ARG_EFFECT_OPS = {
    "BUILD_LIST",
    "BUILD_SET",
    "BUILD_STRING",
    "BUILD_TUPLE",
    "BUILD_LIST_UNPACK",
    "BUILD_MAP_UNPACK",
    "BUILD_MAP_UNPACK_WITH_CALL",
    "BUILD_SET_UNPACK",
    "BUILD_TUPLE_UNPACK",
    "BUILD_TUPLE_UNPACK_WITH_CALL",
}

def builder_stack_effect(opname: str, arg: int) -> BytecodeStackEffect | None:
    pops: int | None = None
    if opname in BUILD_ARG_EFFECT_OPS:
        pops = arg
    elif opname == "BUILD_MAP":
        pops = arg * 2
    elif opname == "BUILD_CONST_KEY_MAP":
        pops = arg + 1
    elif opname == "BUILD_INTERPOLATION":
        pops = 2 + int(arg & 1)
    elif opname == "BUILD_TEMPLATE":
        pops = 2
    elif opname == "BUILD_SLICE":
        return build_slice_effect(arg)

    if pops is None:
        return None
    return BytecodeStackEffect(pops=pops, pushes=1)

def format_stack_effect(opname: str, arg: int) -> BytecodeStackEffect | None:
    if opname == "FORMAT_VALUE":
        return BytecodeStackEffect(pops=1 + int(bool(arg & 0x04)), pushes=1)
    return None

def unpack_stack_effect(opname: str, arg: int) -> BytecodeStackEffect | None:
    if opname == "UNPACK_SEQUENCE_TWO_TUPLE":
        return BytecodeStackEffect(pops=1, pushes=2)

    if opname in {"UNPACK_SEQUENCE", "UNPACK_TUPLE"}:
        return BytecodeStackEffect(pops=1, pushes=arg)

    if opname == "UNPACK_EX":
        before_count = arg & 0xFF
        after_count = (arg >> 8) & 0xFF
        return BytecodeStackEffect(pops=1, pushes=before_count + after_count + 1)
    return None

def call_stack_effect(
    opname: str,
    arg: int,
    version: tuple[int, ...] | None = None,
) -> BytecodeStackEffect | None:
    pops: int | None = None
    if opname == "CALL_FUNCTION":
        pops = arg + 1
    elif opname in {"CALL", "CALL_METHOD"}:
        pops = arg + 2
    elif opname == "CALL_FUNCTION_EX":
        pops = 2 + int(arg & 1) + int(version is not None and version >= (3, 11))
    elif opname == "CALL_KW":
        pops = arg + 3
    elif opname in {"CALL_FUNCTION_VAR", "CALL_FUNCTION_KW", "CALL_FUNCTION_VAR_KW"}:
        return legacy_call_effect(arg, opname)
    elif opname == "MAKE_FUNCTION":
        pops = 1 + make_function_extra_operand_count(arg)
    elif opname == "SET_FUNCTION_ATTRIBUTE":
        pops = 2

    if pops is None:
        return None
    return BytecodeStackEffect(pops=pops, pushes=1)

def exception_stack_effect(opname: str, arg: int) -> BytecodeStackEffect | None:
    if opname == "RAISE_VARARGS":
        return BytecodeStackEffect(pops=max(0, arg), pushes=0)

    if opname == "RERAISE":
        return BytecodeStackEffect(pops=1 + int(arg != 0), pushes=0)

    return None

STACK_EFFECT_READERS = (
    builder_stack_effect,
    format_stack_effect,
    unpack_stack_effect,
    exception_stack_effect,
)

def build_slice_effect(arg: int) -> BytecodeStackEffect | None:
    if arg not in {2, 3}:
        return None
    return BytecodeStackEffect(pops=arg, pushes=1)

def legacy_call_effect(arg: int, opname: str) -> BytecodeStackEffect:
    positional_count = arg & 0xFF
    keyword_count = (arg >> 8) & 0xFF
    star_count = int("VAR" in opname) + int("KW" in opname)
    return BytecodeStackEffect(
        pops=positional_count + keyword_count * 2 + star_count + 1,
        pushes=1,
    )

def make_function_extra_operand_count(flags: int) -> int:
    operand_flags = flags & 0x0F
    return operand_flags.bit_count()
