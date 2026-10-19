# PyPy writes bytecode with its own compiler; this reads it as CPython's
from dataclasses import replace
from functools import cache
from typing import Any

from pyc2py.bytecode.formats import pypy_format, pypy_formats
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import encode_exception_table, parse_exception_table
from pyc2py.bytecode.opcode_table import is_jump_opname
from pyc2py.bytecode.stack_effect import expression_start
from pyc2py.pyc.code import code_objects


@cache
def pypy_magics() -> dict[int, str]:
    return {
        magic: name for name, found in pypy_formats().items() for magic in found.magics
    }


@cache
def pypy_shared_magics() -> dict[int, str]:
    return {
        magic: name
        for name, found in pypy_formats().items()
        for magic in found.cpython_magics
    }


def written_by_pypy(code: Any, opcodes: str, cpython: dict[int, str]) -> bool | None:
    # None when nothing tells the two apart and an opcode differs between them
    pypy = pypy_format(opcodes).opmap
    codes = list(code_objects(code))
    used = {op for item in codes for op in bytes(item.co_code)[::2]}
    only_pypy = used & (pypy.keys() - cpython.keys())
    only_cpython = used & (cpython.keys() - pypy.keys())
    if only_pypy and not only_cpython:
        return True
    if only_cpython:
        return False
    # CPython marks each cell-free code object CO_NOFREE; PyPy not a class body
    if any(
        not item.co_freevars and not item.co_cellvars and not item.co_flags & CO_NOFREE
        for item in codes
    ):
        return True

    differ = {op for op in pypy if cpython.get(op, pypy[op]) != pypy[op]}
    return None if used & differ else False


CO_NOFREE = 0x40


def pypy_line_starts(code: Any) -> dict[int, int]:
    table = bytes(getattr(code, "co_linetable", b"") or b"")
    first_line = int(getattr(code, "co_firstlineno", 1) or 1)
    starts: dict[int, int] = {}
    last: int | None = None
    position = 0
    offset = 0

    while position < len(table):
        delta, position = read_varint(table, position)
        if delta:
            # one byte when the columns are unknown, three when they are known
            position += 1 if position < len(table) and table[position] == 0 else 3
            line = first_line + delta - 1
            if line != last:
                starts[offset] = line
                last = line

        offset += 2

    return starts


def read_varint(table: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0

    while position < len(table):
        byte = table[position]
        position += 1
        value |= (byte & 0x7F) << shift

        if not byte & 0x80:
            break

        shift += 7

    return value, position


def pypy_in_cpython_form(
    instructions: list[Instruction], version: tuple[int, ...], code: Any
) -> list[Instruction]:
    # PyPy-only operations as the CPython operations that do the same
    found = pypy_format(code.opcodes)
    # a format with exception tables names its jumps as CPython 3.11 does
    exception_table = "exceptiontable" in found.code_extras
    renamed = [
        cpython_form(instruction, version, exception_table)
        for instruction in instructions
    ]

    if exception_table:
        renamed = without_unreachable(renamed, handler_starts(code))
    elif version < (3, 0):
        # measured: the 3.x readers read the unthreaded tests better (N57)
        renamed = tests_threaded_through_jumps(renamed)

    built = list_built_before_its_iterable(renamed, version)
    return without_handler_nops(reraise_unmatched_directly(built))


def table_in_cpython_form(code: bytes, table: bytes, opcodes: str) -> bytes:
    # PyPy routes a handler's exception through its own RERAISE; CPython skips it
    try:
        entries = list(parse_exception_table(table))
    except ValueError:
        return table

    reraise = {
        op for op, name in pypy_format(opcodes).opmap.items() if name == "RERAISE"
    }

    for _round in range(len(entries)):
        changed = False

        for index, entry in enumerate(entries):
            target = entry.target_offset
            if (
                target + 1 >= len(code)
                or code[target] not in reraise
                or code[target + 1] != 1
            ):
                continue

            onward = next(
                (
                    other
                    for other in entries
                    if other.start_offset <= target < other.end_offset
                    and other.target_offset != target
                ),
                None,
            )
            if onward is not None:
                entries[index] = replace(entry, target_offset=onward.target_offset)
                changed = True

        if not changed:
            break

    return encode_exception_table(tuple(entries))


def handler_starts(code: Any) -> set[int]:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")

    try:
        return {entry.target_offset for entry in parse_exception_table(table)}
    except ValueError:
        return set()


def without_unreachable(
    instructions: list[Instruction], handlers: set[int]
) -> list[Instruction]:
    # PyPy keeps unreachable code after a jump or raise; CPython 3.11+ drops it
    kept: list[Instruction] = []

    for item in instructions:
        reached = item.is_jump_target or item.offset in handlers
        if kept and kept[-1].opname in ENDS_FLOW and not reached:
            continue

        kept.append(item)

    return kept


ENDS_FLOW = frozenset(
    {
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "RETURN_VALUE",
        "RETURN_CONST",
        "RERAISE",
        "RAISE_VARARGS",
    }
)


def without_handler_nops(instructions: list[Instruction]) -> list[Instruction]:
    # PyPy starts a cleanup handler with a NOP that CPython does not write
    index_of = {item.offset: index for index, item in enumerate(instructions)}
    moved: dict[int, int] = {}

    for item in instructions:
        start = index_of.get(item.argval) if item.opname in HANDLER_SETUPS else None
        if (
            start is not None
            and start + 1 < len(instructions)
            and instructions[start].opname == "NOP"
        ):
            moved[instructions[start].offset] = instructions[start + 1].offset

    return retargeted(instructions, moved, set(moved), HANDLER_SETUPS)


HANDLER_SETUPS = frozenset({"SETUP_FINALLY", "SETUP_EXCEPT", "SETUP_WITH"})


def retargeted(
    instructions: list[Instruction],
    moved: dict[int, int],
    dropped: set[int],
    jumps: frozenset[str],
) -> list[Instruction]:
    # without the dropped instructions, the given jumps aimed at the moved targets
    if not moved:
        return instructions

    kept: list[Instruction] = []

    for item in instructions:
        if item.offset in dropped:
            continue

        if item.opname in jumps and item.argval in moved:
            target = moved[item.argval]
            item = replace(item, argval=target, argrepr=f"to {target}")

        if item.offset in moved.values():
            item = replace(item, is_jump_target=True)

        kept.append(item)

    return kept


def tests_threaded_through_jumps(instructions: list[Instruction]) -> list[Instruction]:
    # CPython sends a test that lands on a jump straight to where that jump goes
    landing = {
        item.offset: item.argval
        for item in instructions
        if item.opname in ("JUMP_ABSOLUTE", "JUMP_FORWARD")
    }
    threaded = [
        replace(item, argval=landing[item.argval], argrepr=f"to {landing[item.argval]}")
        if item.opname.startswith("POP_JUMP_IF_") and item.argval in landing
        else item
        for item in instructions
    ]
    targets = {
        item.argval
        for item in threaded
        if is_jump_opname(item.opname) or item.opname.startswith("SETUP_")
    }
    return [
        replace(item, is_jump_target=False)
        if item.offset in landing and item.is_jump_target and item.offset not in targets
        else item
        for item in threaded
    ]


def list_built_before_its_iterable(
    instructions: list[Instruction], version: tuple[int, ...]
) -> list[Instruction]:
    # PyPy builds a comprehension's list after loading what it iterates
    result = list(instructions)

    for index, built in enumerate(instructions):
        if built.opname != "BUILD_LIST_FROM_ARG":
            continue

        start = expression_start(result, index, version)
        if start is None:
            continue

        moved = [
            replace(built, opname="BUILD_LIST", arg=0, argval=0),
            *result[start:index],
        ]
        result[start : index + 1] = [
            replace(
                item,
                offset=slot.offset,
                starts_line=slot.starts_line,
                is_jump_target=slot.is_jump_target,
            )
            for item, slot in zip(moved, result[start : index + 1], strict=True)
        ]

    return result


def reraise_unmatched_directly(instructions: list[Instruction]) -> list[Instruction]:
    # PyPy drops an unmatched exception's copies before re-raising; CPython not
    index_of = {item.offset: index for index, item in enumerate(instructions)}
    moved: dict[int, int] = {}
    dropped: set[int] = set()

    for item in instructions:
        if item.opname != "JUMP_IF_NOT_EXC_MATCH" or item.argval not in index_of:
            continue

        start = end = index_of[item.argval]
        while end < len(instructions) and instructions[end].opname == "POP_TOP":
            end += 1

        if start < end < len(instructions) and instructions[end].opname == "RERAISE":
            moved[instructions[start].offset] = instructions[end].offset
            dropped.update(drop.offset for drop in instructions[start:end])

    return retargeted(instructions, moved, dropped, EXCEPTION_TESTS)


EXCEPTION_TESTS = frozenset({"JUMP_IF_NOT_EXC_MATCH"})


def cpython_form(
    instruction: Instruction, version: tuple[int, ...], exception_table: bool
) -> Instruction:
    opname = instruction.opname
    if opname == "LOOKUP_METHOD":
        # before 3.7 CPython has no method calls, and PyPy's carry 2.7's argument
        renamed = "LOAD_METHOD" if version >= (3, 7) else "LOAD_ATTR"
        return replace(instruction, opname=renamed)

    if opname == "CALL_METHOD" and version < (3, 7):
        return replace(instruction, opname="CALL_FUNCTION")
    if opname == "SETUP_EXCEPT" and version >= (3, 8):
        return replace(instruction, opname="SETUP_FINALLY")

    if opname in ("DICT_MERGE", "DICT_UPDATE") and instruction.arg is not None:
        # PyPy counts the dict's depth from 0, CPython from 1
        depth = instruction.arg + 1
        return replace(instruction, arg=depth, argval=depth, argrepr=str(depth))

    if opname == "JUMP_ABSOLUTE" and exception_table:
        # 3.11 names a jump by its direction
        backward = instruction.argval <= instruction.offset
        return replace(
            instruction, opname="JUMP_BACKWARD" if backward else "JUMP_FORWARD"
        )

    if opname == "JUMP_IF_NOT_DEBUG":
        # skips an assert only under -O
        return replace(instruction, opname="NOP", arg=None, argval=None, argrepr="")

    return instruction
