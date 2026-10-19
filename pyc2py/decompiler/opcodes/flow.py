import ast

from pyc2py.astree import make_name
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


def dropped_under_value_end(
    instructions: list[Instruction], index: int, end_index: int
) -> int | None:
    swap = instructions[index]
    if swap.opname == "ROT_TWO":
        dropped = 1
    elif swap.opname == "SWAP":
        dropped = int(swap.arg or 0) - 1
    else:
        return None

    after = index + 1 + dropped
    if dropped < 1 or after > end_index:
        return None
    if any(item.opname != "POP_TOP" for item in instructions[index + 1 : after]):
        return None

    return after


# Python 2's raise E, V, tb until 2.x source is written; raise E, V is from V
LEGACY_RAISE_ARGUMENTS = "__pyc2py_raise_arguments__"


def make_raise(values: list[ast.expr], version: tuple[int, ...] | None) -> ast.Raise:
    if not values:
        return ast.Raise(exc=None, cause=None)

    cause = values[1] if len(values) > 1 else None

    if len(values) == 3 and version is not None and version < (3, 0):
        holder = make_name(LEGACY_RAISE_ARGUMENTS, ast.Load())
        cause = ast.Call(func=holder, args=values[1:], keywords=[])

    return ast.Raise(exc=values[0], cause=cause)


LEGACY_SLICE_PREFIXES = ("SLICE+", "STORE_SLICE+", "DELETE_SLICE+")


def is_legacy_slice_op(opname: str) -> bool:
    return opname.startswith(LEGACY_SLICE_PREFIXES)


def legacy_slice_mode(opname: str) -> str:
    if not is_legacy_slice_op(opname):
        raise ValueError(f"not a legacy slice opcode: {opname}")

    return opname.rsplit("+", 1)[1]
