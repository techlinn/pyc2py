import ast
import math
from typing import Any

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.decompiler.opcodes.flow import is_jump_op
from pyc2py.decompiler.opcodes.stack_names import (
    LOAD_OPS,
    NO_VALUE_OPS,
    STORE_OPS,
    make_name,
    rotation_count,
)
from pyc2py.decompiler.opcodes.values import (
    BINARY_OPS,
    LEGACY_BINARY_OPS,
    UNARY_OPS,
    make_constant,
)


def dispatch_instruction(decompiler: Any, instruction: Instruction) -> None:
    for handler in DISPATCH_HANDLERS:
        if handler(decompiler, instruction):
            return

    opname = normalized_opcode_name(instruction.opname)
    if is_jump_op(opname):
        if instruction.argval in decompiler.loop_continue_offsets:
            decompiler.statements.append(ast.Continue())
            decompiler.stack.clear()
            return
        if instruction.argval in decompiler.loop_break_offsets:
            decompiler.statements.append(ast.Break())
            decompiler.stack.clear()
            return
        decompiler.warnings.append(
            skipped_opcode_message("control-flow", instruction, opname)
        )
        decompiler.stack.clear()
        return
    decompiler.warnings.append(
        skipped_opcode_message("unsupported", instruction, opname)
    )


def skipped_opcode_message(
    kind: str,
    instruction: Instruction,
    runtime_opname: str,
) -> str:
    parts = [
        f"{kind} opcode skipped at {instruction.offset}",
        f"opcode={instruction.opname}",
    ]
    if runtime_opname != instruction.opname:
        parts.append(f"runtime={runtime_opname}")
    if instruction.arg is not None:
        parts.append(f"arg={instruction.arg}")
    if instruction.argrepr:
        parts.append(f"argrepr={instruction.argrepr}")
    return ": ".join((parts[0], ", ".join(parts[1:])))


def dispatch_stack_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if opname in NO_VALUE_OPS:
        return True
    if dispatch_stack_value_instruction(decompiler, instruction, opname):
        return True
    if dispatch_noarg_method(decompiler, opname, STACK_NOARG_METHODS):
        return True
    if dispatch_int_arg_method(decompiler, opname, instruction.arg, STACK_INT_METHODS):
        return True
    if opname in {"ROT_TWO", "ROT_THREE", "ROT_FOUR"}:
        decompiler.rotate_stack(rotation_count(opname))
        return True
    return dispatch_end_for(decompiler, opname)


def dispatch_stack_value_instruction(
    decompiler: Any,
    instruction: Instruction,
    opname: str,
) -> bool:
    if opname == "LOAD_CONST":
        decompiler.stack.append(
            make_constant(
                instruction.argval,
                legacy_strings_as_bytes=decompiler.uses_unicode_literals(),
                negative_nan=constant_negative_nan_from_previous_inf(
                    decompiler,
                    instruction,
                ),
            )
        )
        return True
    if opname == "LOAD_SMALL_INT":
        decompiler.stack.append(make_constant(int(instruction.argval or 0)))
        return True
    if opname == "LOAD_COMMON_CONSTANT":
        decompiler.load_common_constant(instruction.argval)
        return True
    if opname == "PUSH_NULL":
        decompiler.stack.append(make_name("NULL", ast.Load()))
        return True
    return False


def dispatch_end_for(decompiler: Any, opname: str) -> bool:
    if opname != "END_FOR":
        return False
    decompiler.pop_or_none()
    if decompiler.version is None or decompiler.version < (3, 14):
        decompiler.pop_or_none()
    return True


def constant_negative_nan_from_previous_inf(
    decompiler: Any,
    instruction: Instruction,
) -> bool | None:
    if not isinstance(instruction.argval, float) or not math.isnan(instruction.argval):
        return None

    if not isinstance(instruction.arg, int) or instruction.arg < 1:
        return None

    constants = getattr(decompiler.code, "co_consts", ())
    if instruction.arg >= len(constants):
        return None

    previous = constants[instruction.arg - 1]
    if not isinstance(previous, float) or not math.isinf(previous):
        return None
    return math.copysign(1.0, previous) < 0


def dispatch_name_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_name_load_instruction(decompiler, instruction, opname):
        return True

    if dispatch_str_arg_method(
        decompiler, opname, instruction.argval, NAME_STR_METHODS
    ):
        return True
    if dispatch_value_arg_method(
        decompiler, opname, instruction.argval, NAME_VALUE_METHODS
    ):
        return True
    if dispatch_noarg_method(decompiler, opname, NAME_NOARG_METHODS):
        return True

    return dispatch_name_store_delete(decompiler, instruction, opname)


def dispatch_name_load_instruction(
    decompiler: Any,
    instruction: Instruction,
    opname: str,
) -> bool:
    if opname == "LOAD_GLOBAL":
        load_global_name(decompiler, instruction)
        return True
    if opname in LOAD_OPS:
        load_name(decompiler, instruction.argval)
        return True
    if opname == "LOAD_ASSERTION_ERROR":
        decompiler.stack.append(make_name("AssertionError", ast.Load()))
        return True
    if opname == "LOAD_BUILD_CLASS":
        decompiler.stack.append(decompiler.build_class_marker())
        return True
    return False


def dispatch_name_store_delete(
    decompiler: Any,
    instruction: Instruction,
    opname: str,
) -> bool:
    if opname in STORE_OPS:
        decompiler.store_name(str(instruction.argval), opname)
        return True
    if opname in {"DELETE_NAME", "DELETE_GLOBAL", "DELETE_FAST", "DELETE_DEREF"}:
        decompiler.delete_name(str(instruction.argval), opname)
        return True
    return False


def dispatch_statement_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_noarg_method(decompiler, opname, STATEMENT_NOARG_METHODS):
        return True
    if dispatch_statement_node(decompiler, opname, STATEMENT_NODES):
        return True

    if opname == "POP_TOP":
        decompiler.pop_expression_statement()
        return True

    if opname in {"INTERPRETER_EXIT", "RETURN_VALUE", "RETURN_CONST"}:
        decompiler.return_value(instruction)
        return True

    if opname == "RAISE_VARARGS":
        decompiler.raise_varargs(int(instruction.arg or 0))
        return True

    return False


def dispatch_builder_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_instruction_method(
        decompiler, opname, instruction, BUILDER_INSTRUCTION_METHODS
    ):
        return True

    if opname in {"BUILD_TUPLE", "BUILD_LIST", "BUILD_SET"}:
        decompiler.build_sequence(opname, int(instruction.arg or 0))
        return True

    if dispatch_opname_int_arg_method(
        decompiler, opname, instruction.arg, BUILDER_OPNAME_INT_METHODS
    ):
        return True
    if dispatch_int_arg_method(
        decompiler, opname, instruction.arg, BUILDER_ZERO_DEFAULT_INT_METHODS
    ):
        return True
    if dispatch_int_arg_method(
        decompiler, opname, instruction.arg, BUILDER_ONE_DEFAULT_INT_METHODS, default=1
    ):
        return True
    return dispatch_noarg_method(decompiler, opname, BUILDER_NOARG_METHODS)


def dispatch_operator_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if opname == "BINARY_OP":
        dispatch_binary_op(decompiler, instruction)
        return True

    if opname in LEGACY_BINARY_OPS:
        decompiler.legacy_binary_op(opname)
        return True

    if opname in UNARY_OPS:
        decompiler.unary_op(UNARY_OPS[opname]())
        return True

    if dispatch_noarg_method(decompiler, opname, OPERATOR_NOARG_METHODS):
        return True

    if opname == "COMPARE_OP":
        decompiler.compare_op(
            str(instruction.argrepr),
            force_bool=compare_op_forces_bool(instruction, decompiler.version),
        )
        return True

    return dispatch_compare_arg_op(decompiler, opname, instruction.arg)


def dispatch_binary_op(decompiler: Any, instruction: Instruction) -> None:
    symbol = binary_operator_symbol(instruction)
    if symbol == "[]":
        decompiler.binary_subscript()
        return
    decompiler.binary_op(symbol)


def binary_operator_symbol(instruction: Instruction) -> str:
    specialized = specialized_binary_operator_symbol(instruction.opname)
    if specialized is not None:
        return specialized

    argrepr = str(instruction.argrepr)
    if argrepr in BINARY_OPS or argrepr.endswith("=") or argrepr == "[]":
        return argrepr
    return argrepr


def specialized_binary_operator_symbol(opname: str) -> str | None:
    if opname == "BINARY_OP_EXTEND":
        return "+="
    if opname.startswith("BINARY_OP_INPLACE_ADD_"):
        return "+="
    if opname.startswith("BINARY_OP_ADD_"):
        return "+"
    if opname.startswith("BINARY_OP_MULTIPLY_"):
        return "*"
    if opname.startswith("BINARY_OP_SUBTRACT_"):
        return "-"
    return None


def compare_op_forces_bool(
    instruction: Instruction,
    version: tuple[int, ...] | None,
) -> bool:
    return (
        version is not None
        and version >= (3, 13)
        and bool(int(instruction.arg or 0) & 0x10)
    )


def load_global_name(decompiler: Any, instruction: Instruction) -> None:
    pushes_null = (
        decompiler.version is not None
        and decompiler.version >= (3, 11)
        and int(instruction.arg or 0) & 1
    )
    if pushes_null:
        decompiler.stack.append(make_name("NULL", ast.Load()))
    load_name(decompiler, instruction.argval)


def load_name(decompiler: Any, value: object) -> None:
    if value == "None":
        decompiler.stack.append(ast.Constant(value=None))
        return
    decompiler.stack.append(make_name(str(value), ast.Load()))


def dispatch_compare_arg_op(
    decompiler: Any,
    opname: str,
    arg: object,
) -> bool:
    symbols = COMPARE_ARG_SYMBOLS.get(opname)
    if symbols is None:
        return False
    decompiler.compare_op(symbols[int(arg == 1)])
    return True


def dispatch_format_call_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_noarg_method(decompiler, opname, FORMAT_CALL_NOARG_METHODS):
        return True
    if dispatch_int_arg_method(
        decompiler, opname, instruction.arg, FORMAT_CALL_INT_METHODS
    ):
        return True
    if dispatch_instruction_method(
        decompiler, opname, instruction, FORMAT_CALL_INSTRUCTION_METHODS
    ):
        return True

    if opname in CALL_OPS:
        decompiler.call_function(instruction)
        return True

    if opname == "KW_NAMES":
        decompiler.set_kw_names(instruction.argval)
        return True

    return False


def dispatch_import_attr_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_noarg_method(decompiler, opname, IMPORT_ATTR_NOARG_METHODS):
        return True
    if dispatch_str_arg_method(
        decompiler, opname, instruction.argval, IMPORT_ATTR_STR_METHODS
    ):
        return True
    if dispatch_int_arg_method(
        decompiler, opname, instruction.arg, IMPORT_ATTR_INT_METHODS
    ):
        return True

    if opname == "LOAD_ATTR":
        decompiler.load_attr(
            str(instruction.argval),
            load_attr_arg_from_instruction(instruction),
        )
        return True

    if opname == "LOAD_SUPER_ATTR":
        decompiler.load_super_attr(
            str(instruction.argval),
            super_attr_arg_from_instruction(instruction),
        )
        return True

    return False


def dispatch_subscript_instruction(decompiler: Any, instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_noarg_method(decompiler, opname, SUBSCRIPT_NOARG_METHODS):
        return True
    if dispatch_str_arg_method(
        decompiler, opname, instruction.argval, SUBSCRIPT_STR_METHODS
    ):
        return True
    if dispatch_opname_prefix_method(decompiler, opname, SUBSCRIPT_PREFIX_METHODS):
        return True

    if opname == "BUILD_SLICE":
        decompiler.build_slice(int(instruction.arg or 0))
        return True

    return False


def dispatch_async_yield_print_instruction(
    decompiler: Any,
    instruction: Instruction,
) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if dispatch_noarg_method(decompiler, opname, ASYNC_YIELD_PRINT_NOARG_METHODS):
        return True

    if opname == "GET_ITER":
        return True

    if opname in {"GET_AWAITABLE", "GET_YIELD_FROM_ITER"}:
        return True

    return opname == "ASYNC_GEN_WRAP"


def dispatch_noarg_method(
    decompiler: Any,
    opname: str,
    methods: dict[str, str],
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)()
    return True


def dispatch_str_arg_method(
    decompiler: Any,
    opname: str,
    value: object,
    methods: dict[str, str],
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)(str(value))
    return True


def dispatch_int_arg_method(
    decompiler: Any,
    opname: str,
    value: object,
    methods: dict[str, str],
    *,
    default: int = 0,
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)(read_int_arg(value, default))
    return True


def dispatch_opname_int_arg_method(
    decompiler: Any,
    opname: str,
    value: object,
    methods: dict[str, str],
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)(opname, read_int_arg(value, 0))
    return True


def read_int_arg(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise TypeError(
            f"opcode argument must be an integer, got {type(value).__name__}"
        )
    return value


def dispatch_value_arg_method(
    decompiler: Any,
    opname: str,
    value: object,
    methods: dict[str, str],
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)(value)
    return True


def dispatch_instruction_method(
    decompiler: Any,
    opname: str,
    instruction: Instruction,
    methods: dict[str, str],
) -> bool:
    method_name = methods.get(opname)
    if method_name is None:
        return False
    getattr(decompiler, method_name)(instruction)
    return True


def dispatch_opname_prefix_method(
    decompiler: Any,
    opname: str,
    methods: dict[str, str],
) -> bool:
    for prefix, method_name in methods.items():
        if opname.startswith(prefix):
            getattr(decompiler, method_name)(opname)
            return True
    return False


def dispatch_statement_node(
    decompiler: Any,
    opname: str,
    nodes: dict[str, type[ast.stmt]],
) -> bool:
    node_type = nodes.get(opname)
    if node_type is None:
        return False
    decompiler.statements.append(node_type())
    return True


def super_attr_arg_from_instruction(instruction: Instruction) -> int:
    arg = int(instruction.arg or 0)
    if instruction.opname in {
        "LOAD_SUPER_METHOD",
        "LOAD_ZERO_SUPER_METHOD",
        "LOAD_SUPER_ATTR_METHOD",
    }:
        arg |= 1
    if instruction.opname == "LOAD_SUPER_METHOD":
        arg |= 0x02
    if instruction.opname in {"LOAD_ZERO_SUPER_ATTR", "LOAD_ZERO_SUPER_METHOD"}:
        arg &= ~0x02

    return arg


def load_attr_arg_from_instruction(instruction: Instruction) -> int:
    arg = int(instruction.arg or 0)
    if instruction.opname.startswith("LOAD_ATTR_METHOD_"):
        return arg | 1
    return arg


CALL_OPS = frozenset(
    {
        "CALL",
        "CALL_KW",
        "CALL_FUNCTION",
        "CALL_FUNCTION_VAR",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_VAR_KW",
        "CALL_FUNCTION_EX",
        "CALL_METHOD",
    }
)

STACK_NOARG_METHODS = {
    "LOAD_LOCALS": "load_locals",
    "DUP_TOP": "duplicate_top",
    "DUP_TOP_TWO": "duplicate_top_two",
    "GEN_START": "pop_or_none",
}
STACK_INT_METHODS = {
    "COPY": "copy_stack_item",
    "SWAP": "swap_stack_item",
    "ROT_N": "rotate_stack",
}

NAME_STR_METHODS = {
    "LOAD_FROM_DICT_OR_GLOBALS": "load_from_dict_or_name",
    "LOAD_FROM_DICT_OR_DEREF": "load_from_dict_or_name",
}
NAME_VALUE_METHODS = {
    "LOAD_FAST_LOAD_FAST": "load_fast_load_fast",
    "LOAD_CONST_LOAD_FAST": "load_const_load_fast",
    "LOAD_FAST_LOAD_CONST": "load_fast_load_const",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": "load_fast_load_fast",
    "STORE_FAST_LOAD_FAST": "store_fast_load_fast",
    "STORE_FAST_STORE_FAST": "store_fast_store_fast",
}
NAME_NOARG_METHODS = {
    "BUILD_CLASS": "build_legacy_class",
}

STATEMENT_NOARG_METHODS = {
    "RERAISE": "reraise",
    "PUSH_EXC_INFO": "push_exc_info",
    "CHECK_EXC_MATCH": "check_exc_match",
    "CHECK_EG_MATCH": "check_eg_match",
    "PREP_RERAISE_STAR": "prep_reraise_star",
    "BEFORE_WITH": "before_with",
    "WITH_EXCEPT_START": "with_except_start",
    "EXEC_STMT": "exec_stmt",
}
STATEMENT_NODES = {
    "CONTINUE_LOOP": ast.Continue,
    "BREAK_LOOP": ast.Break,
}

BUILDER_INSTRUCTION_METHODS = {
    "UNPACK_EX": "unpack_sequence",
    "UNPACK_SEQUENCE": "unpack_sequence",
    "UNPACK_TUPLE": "unpack_sequence",
}
BUILDER_OPNAME_INT_METHODS = {
    "BUILD_LIST_UNPACK": "build_sequence_unpack",
    "BUILD_SET_UNPACK": "build_sequence_unpack",
    "BUILD_TUPLE_UNPACK": "build_sequence_unpack",
    "BUILD_TUPLE_UNPACK_WITH_CALL": "build_sequence_unpack",
    "LIST_EXTEND": "extend_container",
    "SET_UPDATE": "extend_container",
    "DICT_MERGE": "update_dict",
    "DICT_UPDATE": "update_dict",
}
BUILDER_ZERO_DEFAULT_INT_METHODS = {
    "BUILD_STRING": "build_string",
    "BUILD_MAP": "build_map",
    "BUILD_MAP_UNPACK": "build_map_unpack",
    "BUILD_MAP_UNPACK_WITH_CALL": "build_map_unpack",
    "BUILD_CONST_KEY_MAP": "build_const_key_map",
}
BUILDER_ONE_DEFAULT_INT_METHODS = {
    "LIST_APPEND": "list_append",
    "SET_ADD": "set_add",
    "MAP_ADD": "map_add",
}
BUILDER_NOARG_METHODS = {
    "STORE_MAP": "store_map",
    "LIST_TO_TUPLE": "list_to_tuple",
    "POP_ITER": "pop_or_none",
}

OPERATOR_NOARG_METHODS = {
    "TO_BOOL": "to_bool",
    "UNARY_CONVERT": "unary_convert",
}
COMPARE_ARG_SYMBOLS = {
    "CONTAINS_OP": ("in", "not in"),
    "IS_OP": ("is", "is not"),
}

IMPORT_ATTR_NOARG_METHODS = {
    "IMPORT_STAR": "import_star",
    "GET_LEN": "get_len",
    "MATCH_MAPPING": "match_mapping",
    "MATCH_SEQUENCE": "match_sequence",
    "MATCH_KEYS": "match_keys",
    "COPY_DICT_WITHOUT_KEYS": "copy_dict_without_keys",
}
IMPORT_ATTR_STR_METHODS = {
    "IMPORT_NAME": "import_name",
    "IMPORT_FROM": "import_from",
    "LOAD_METHOD": "load_method",
}
IMPORT_ATTR_INT_METHODS = {
    "LOAD_SPECIAL": "load_special",
    "MATCH_CLASS": "match_class",
}

SUBSCRIPT_NOARG_METHODS = {
    "BINARY_SUBSCR": "binary_subscript",
    "BINARY_SLICE": "binary_slice",
    "STORE_SUBSCR": "store_subscript",
    "STORE_SLICE": "store_slice",
    "DELETE_SUBSCR": "delete_subscript",
}
SUBSCRIPT_STR_METHODS = {
    "STORE_ATTR": "store_attr",
    "DELETE_ATTR": "delete_attr",
}
SUBSCRIPT_PREFIX_METHODS = {
    "SLICE+": "legacy_slice",
    "STORE_SLICE+": "store_legacy_slice",
    "DELETE_SLICE+": "delete_slice",
}

ASYNC_YIELD_PRINT_NOARG_METHODS = {
    "GET_AITER": "get_aiter",
    "GET_ANEXT": "get_anext",
    "BEFORE_ASYNC_WITH": "before_async_with",
    "END_ASYNC_FOR": "end_async_for",
    "END_SEND": "end_send",
    "CLEANUP_THROW": "cleanup_throw",
    "YIELD_VALUE": "yield_value",
    "YIELD_FROM": "yield_from",
    "PRINT_ITEM": "print_item",
    "PRINT_NEWLINE": "print_newline",
    "PRINT_ITEM_TO": "print_item_to",
    "PRINT_NEWLINE_TO": "print_newline_to",
}

FORMAT_CALL_NOARG_METHODS = {
    "FORMAT_SIMPLE": "format_simple",
    "FORMAT_WITH_SPEC": "format_with_spec",
    "BUILD_TEMPLATE": "build_template",
    "MAKE_CLOSURE": "make_closure",
}
FORMAT_CALL_INT_METHODS = {
    "FORMAT_VALUE": "format_value",
    "CONVERT_VALUE": "convert_value",
    "BUILD_INTERPOLATION": "build_interpolation",
    "MAKE_FUNCTION": "make_function",
    "SET_FUNCTION_ATTRIBUTE": "set_function_attribute",
}
FORMAT_CALL_INSTRUCTION_METHODS = {
    "CALL_INTRINSIC_1": "call_intrinsic_1",
    "CALL_INTRINSIC_2": "call_intrinsic_2",
}

DISPATCH_HANDLERS = (
    dispatch_stack_instruction,
    dispatch_name_instruction,
    dispatch_statement_instruction,
    dispatch_builder_instruction,
    dispatch_operator_instruction,
    dispatch_format_call_instruction,
    dispatch_import_attr_instruction,
    dispatch_subscript_instruction,
    dispatch_async_yield_print_instruction,
)
