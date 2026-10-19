import ast
from dataclasses import dataclass

from pyc2py.bytecode.instruction import Instruction
from pyc2py.decompiler.block_try import same_instructions
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.flow import is_terminal_op
from pyc2py.decompiler.runtime import coerce_expr
from pyc2py.decompiler.structures import INLINE_EXIT_CALL, make_with_statement

CLEANUPS = (
    ("WITH_CLEANUP", "END_FINALLY"),
    ("WITH_CLEANUP_START", "WITH_CLEANUP_FINISH", "END_FINALLY"),
)
BODY_EXITS = ({"POP_BLOCK"}, {"LOAD_CONST", "BEGIN_FINALLY"})
INLINE_EXIT = ("POP_BLOCK", *INLINE_EXIT_CALL, "POP_TOP")
# PyPy's handler leaves one item of the exception, CPython's three; both are
# tried, as a file with CPython's magic may be PyPy's and the shapes never meet
EXCEPT_PATHS = (
    ("WITH_EXCEPT_START", None, "RERAISE", *["POP_TOP"] * 3, "POP_EXCEPT", "POP_TOP"),
    ("WITH_EXCEPT_START", None, "RERAISE", "POP_TOP", "POP_EXCEPT"),
)
SPELLED_ENTRY = ("DUP_TOP", "LOAD_ATTR", None, "LOAD_ATTR", "CALL_FUNCTION")
UNWIND_38 = (
    "POP_BLOCK",
    "BEGIN_FINALLY",
    "WITH_CLEANUP_START",
    "WITH_CLEANUP_FINISH",
    "POP_FINALLY",
)


@dataclass(frozen=True, slots=True)
class BlockWith:
    target_start: int | None
    body_start: int
    body_end: int
    end_index: int


def try_translate_block_with(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    if not decompiler.stack:
        return None

    shape = read_block_with(instructions, offset_to_index, cursor, end_index)
    if shape is None:
        return None

    target: ast.expr | None = None
    body_start = shape.body_start

    if shape.target_start is not None:
        target, body_start = decompiler.read_for_loop_target(
            instructions, shape.target_start, shape.body_end
        )
        if target is None:
            return None

    context = coerce_expr(decompiler.pop_or_none())
    bounds = (
        int(instructions[cursor].offset),
        int(instructions[shape.body_end].offset),
    )
    for start, end in unwind_spans(instructions, body_start, shape.body_end, bounds):
        decompiler.claimed_ranges[start] = end

    body = decompiler.translate_child_statements(
        instructions, body_start, shape.body_end
    )
    decompiler.statements.append(make_with_statement(context, body, target))
    return shape.end_index


def read_block_with(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> BlockWith | None:
    if instructions[cursor].opname == "SETUP_WITH":
        cleanup = offset_to_index.get(int(instructions[cursor].argval))
        if cleanup is None or not cursor + 1 < cleanup < end_index:
            return None

        head: tuple[int | None, int]

        if instructions[cursor + 1].opname == "POP_TOP":
            head = (None, cursor + 2)
        else:
            head = (cursor + 1, cursor + 1)

        return read_with_body(instructions, head, cleanup, end_index)

    if instructions[cursor].opname == "DUP_TOP":
        return read_spelled_with(instructions, offset_to_index, cursor, end_index)

    return None


def read_with_body(
    instructions: list[Instruction],
    head: tuple[int | None, int],
    cleanup: int,
    end_index: int,
) -> BlockWith | None:
    target_start, body_start = head

    if instructions[cleanup].opname == "WITH_EXCEPT_START":
        return read_inline_exit_with(instructions, head, cleanup, end_index)

    after = matched_cleanup_end(instructions, cleanup, end_index)
    if after is None:
        return None
    if cleanup - 2 >= body_start and all(
        instructions[cleanup - 2 + position].opname in names
        for position, names in enumerate(BODY_EXITS)
    ):
        return BlockWith(target_start, body_start, cleanup - 2, after)
    if is_terminal_op(instructions[cleanup - 1].opname):
        return BlockWith(target_start, body_start, cleanup, after)

    return None


def matched_cleanup_end(
    instructions: list[Instruction], cleanup: int, end_index: int
) -> int | None:
    for names in CLEANUPS:
        found = tuple(
            item.opname for item in instructions[cleanup : cleanup + len(names)]
        )
        if found == names and cleanup + len(names) <= end_index:
            return cleanup + len(names)

    return None


def read_inline_exit_with(
    instructions: list[Instruction],
    head: tuple[int | None, int],
    cleanup: int,
    end_index: int,
) -> BlockWith | None:
    target_start, body_start = head
    after = except_path_end(instructions, cleanup)
    if after is None or after > end_index:
        return None

    for exit_start in range(cleanup - len(INLINE_EXIT), body_start - 1, -1):
        exit_end = inline_exit_end(instructions, exit_start)
        if exit_end is None or exit_end > cleanup:
            continue
        if leaves_to(instructions, exit_end, cleanup, after):
            return BlockWith(target_start, body_start, exit_start, after)
        break

    if is_terminal_op(instructions[cleanup - 1].opname):
        return BlockWith(target_start, body_start, cleanup, after)

    return None


def except_path_end(instructions: list[Instruction], cleanup: int) -> int | None:
    for path in EXCEPT_PATHS:
        found = [item.opname for item in instructions[cleanup : cleanup + len(path)]]
        if len(found) == len(path) and all(
            name is None or name == opname
            for name, opname in zip(path, found, strict=True)
        ):
            return cleanup + len(path)

    return None


def inline_exit_end(instructions: list[Instruction], index: int) -> int | None:
    cursor = index

    for position, name in enumerate(INLINE_EXIT):
        if cursor >= len(instructions):
            return None
        if position == 1 and instructions[cursor].opname == "ROT_TWO":
            cursor += 1

        instruction = instructions[cursor]
        if instruction.opname != name:
            return None
        if name == "LOAD_CONST" and instruction.argval is not None:
            return None
        if name == "CALL_FUNCTION" and instruction.arg != 3:
            return None

        cursor += 1

    return cursor


def unwind_end(instructions: list[Instruction], index: int) -> int | None:
    found = inline_exit_end(instructions, index)
    if found is not None:
        return found

    cursor = index

    for position, name in enumerate(UNWIND_38):
        if cursor >= len(instructions):
            return None
        if position == 1 and instructions[cursor].opname == "ROT_TWO":
            cursor += 1
        if instructions[cursor].opname != name:
            return None

        cursor += 1

    return cursor


def unwind_spans(
    instructions: list[Instruction],
    start: int,
    end: int,
    bounds: tuple[int, int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    for index in range(start, end):
        span_end = unwind_end(instructions, index)
        if span_end is None:
            continue

        pops_end = iterator_pops_end(instructions, span_end)
        if pops_end is not None:
            spans.append((index, pops_end))
        elif leaves(instructions, span_end, bounds):
            spans.append((index, span_end))

    return spans


def leaves(
    instructions: list[Instruction], index: int, bounds: tuple[int, int]
) -> bool:
    while index < len(instructions):
        instruction = instructions[index]

        if returns_at(instructions, index):
            return True
        if iterator_pops_end(instructions, index) is not None:
            return True

        if instruction.opname in {"JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            first, last = bounds
            return not first <= int(instruction.argval) <= last

        further = unwind_end(instructions, index)
        if further is None:
            return False

        index = further

    return False


def returns_at(instructions: list[Instruction], index: int) -> bool:
    if index >= len(instructions):
        return False
    if instructions[index].opname == "RETURN_VALUE":
        return True

    return (
        instructions[index].opname == "LOAD_CONST"
        and index + 1 < len(instructions)
        and instructions[index + 1].opname == "RETURN_VALUE"
    )


def iterator_pops_end(instructions: list[Instruction], index: int) -> int | None:
    cursor = index
    while cursor < len(instructions):
        if (
            instructions[cursor].opname == "ROT_TWO"
            and cursor + 1 < len(instructions)
            and instructions[cursor + 1].opname == "POP_TOP"
        ):
            cursor += 2
        elif instructions[cursor].opname == "POP_TOP":
            cursor += 1
        else:
            break

    if cursor == index or not returns_at(instructions, cursor):
        return None

    return cursor


def leaves_to(
    instructions: list[Instruction], start: int, cleanup: int, after: int
) -> bool:
    jump = instructions[start]
    # PyPy jumps forward with JUMP_ABSOLUTE as well
    forward = jump.opname == "JUMP_FORWARD" or (
        jump.opname == "JUMP_ABSOLUTE" and jump.argval > jump.offset
    )
    if cleanup - start == 1 and forward:
        return jump.argval == instructions[after].offset

    length = cleanup - start
    return (
        length > 0
        and after + length <= len(instructions)
        and same_instructions(instructions, start, after, length)
    )


def read_spelled_with(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> BlockWith | None:
    entry = instructions[cursor : cursor + 7]
    if len(entry) < 7 or any(
        name is not None and name != item.opname
        for name, item in zip(SPELLED_ENTRY, entry, strict=False)
    ):
        return None
    if entry[1].argval != "__exit__" or entry[3].argval != "__enter__":
        return None

    exit_store, entered, setup = entry[2], entry[5], entry[6]
    if not exit_store.opname.startswith("STORE_") or setup.opname != "SETUP_FINALLY":
        return None

    cleanup = offset_to_index.get(int(setup.argval))
    if cleanup is None or not cursor < cleanup < end_index:
        return None

    load, delete, finish, close = instructions[cleanup : cleanup + 4]
    if (
        load.argval != exit_store.argval
        or not delete.opname.startswith("DELETE_")
        or (finish.opname, close.opname) != ("WITH_CLEANUP", "END_FINALLY")
    ):
        return None

    head = spelled_head(instructions, entered, cursor + 6)
    if head is None:
        return None

    target_start, body_start = head

    if [item.opname for item in instructions[cleanup - 2 : cleanup]] == [
        "POP_BLOCK",
        "LOAD_CONST",
    ]:
        return BlockWith(target_start, body_start, cleanup - 2, cleanup + 4)
    if is_terminal_op(instructions[cleanup - 1].opname):
        return BlockWith(target_start, body_start, cleanup, cleanup + 4)

    return None


def spelled_head(
    instructions: list[Instruction], entered: Instruction, setup_index: int
) -> tuple[int | None, int] | None:
    if entered.opname == "POP_TOP":
        return None, setup_index + 1
    if not entered.opname.startswith("STORE_"):
        return None

    hidden_load, hidden_delete = instructions[setup_index + 1 : setup_index + 3]
    if hidden_load.argval != entered.argval or hidden_delete.argval != entered.argval:
        return None

    return setup_index + 3, setup_index + 3
