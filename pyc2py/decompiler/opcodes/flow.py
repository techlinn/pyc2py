import ast

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name

TERMINAL_OPS = frozenset(
    {"INTERPRETER_EXIT", "RETURN_CONST", "RETURN_VALUE", "RAISE_VARARGS", "RERAISE"}
)


def is_jump_op(opname: str) -> bool:
    opname = normalized_opcode_name(opname)
    return "JUMP" in opname or opname in {"FOR_ITER", "SEND"}


def is_terminal_op(opname: str) -> bool:
    opname = normalized_opcode_name(opname)
    return opname in TERMINAL_OPS


def terminal_tail_end_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if is_terminal_op(instructions[index].opname):
            return index + 1
    return None


def make_raise(values: list[ast.expr]) -> ast.Raise:
    if not values:
        return ast.Raise(exc=None, cause=None)

    exc = values[0]
    cause = values[1] if len(values) > 1 else None
    return ast.Raise(exc=exc, cause=cause)


LEGACY_SLICE_PREFIXES = ("SLICE+", "STORE_SLICE+", "DELETE_SLICE+")


def is_legacy_slice_op(opname: str) -> bool:
    return opname.startswith(LEGACY_SLICE_PREFIXES)


def legacy_slice_mode(opname: str) -> str:
    if not is_legacy_slice_op(opname):
        raise ValueError(f"not a legacy slice opcode: {opname}")

    return opname.rsplit("+", 1)[1]
