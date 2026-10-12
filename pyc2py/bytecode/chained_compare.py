from dataclasses import replace

from pyc2py.bytecode.instruction import Instruction

CHAINED_COMPARE_LINK = "CHAINED_COMPARE_LINK"

COMPARE_OPNAMES = frozenset({"COMPARE_OP", "CONTAINS_OP", "IS_OP"})
PLAIN_JUMP_OPNAMES = frozenset({"JUMP_BACKWARD", "JUMP_FORWARD"})
EXIT_OPNAMES = frozenset({"RAISE_VARARGS", "RETURN_CONST", "RETURN_VALUE"})


def fold_chained_comparison_tests(instructions: list[Instruction]) -> list[Instruction]:
    offset_to_index = {
        instruction.offset: index for index, instruction in enumerate(instructions)
    }
    folded = list(instructions)
    for cleanup_index, links in chained_compare_links(
        instructions, offset_to_index
    ).items():
        found = chained_compare_cleanup_end(
            instructions,
            offset_to_index,
            links[-1],
            cleanup_index,
        )
        if found is None:
            continue
        cleanup_end, last_link_op = found
        for link in links:
            folded[link] = replace(
                instructions[link], opname=CHAINED_COMPARE_LINK, arg=None, argval=None
            )
        folded[links[-1]] = replace(folded[links[-1]], argval=last_link_op)
        skip_index = previous_real_index(instructions, cleanup_index - 1)
        skip = instructions[skip_index]
        if skip.opname == "JUMP_FORWARD" and skip.argval == end_offset(
            instructions, cleanup_end
        ):
            folded[skip_index] = make_nop(skip)
        for index in range(cleanup_index, cleanup_end):
            folded[index] = make_nop(instructions[index])
    return folded


def chained_compare_links(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
) -> dict[int, list[int]]:
    links: dict[int, list[int]] = {}
    for index, instruction in enumerate(instructions):
        if undirected(instruction.opname) != "POP_JUMP_IF_FALSE":
            continue
        if not compares_copied_operand(instructions, index):
            continue
        cleanup_index = offset_to_index.get(instruction.argval)
        if cleanup_index is None or instructions[cleanup_index].opname != "POP_TOP":
            continue
        links.setdefault(cleanup_index, []).append(index)
    return links


def compares_copied_operand(instructions: list[Instruction], jump_index: int) -> bool:
    compare_index = previous_real_index(instructions, jump_index - 1)
    if instructions[compare_index].opname == "TO_BOOL":
        compare_index = previous_real_index(instructions, compare_index - 1)
    if instructions[compare_index].opname not in COMPARE_OPNAMES:
        return False
    copy = instructions[previous_real_index(instructions, compare_index - 1)]
    return copy.opname == "COPY" and copy.arg == 2


def chained_compare_cleanup_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    last_link: int,
    cleanup_index: int,
) -> tuple[int, str | None] | None:
    final_jump = next_jump_index(instructions, last_link + 1, cleanup_index)
    if final_jump is None:
        return None
    jump = instructions[final_jump]
    exits = (
        offset_to_index.get(jump.argval),
        next_real_index(instructions, final_jump + 1),
    )
    for false_exit in false_exit_choices(jump.opname):
        false_index = exits[false_exit]
        if false_index is None:
            continue
        cleanup_end = cleanup_end_for_exit(
            instructions,
            offset_to_index,
            cleanup_index,
            false_index,
        )
        if cleanup_end is None:
            continue
        if undirected(jump.opname) == NONE_JUMPS[false_exit]:
            return cleanup_end, "is not"
        return cleanup_end, None
    return None


NONE_JUMPS = ("POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE")


def undirected(opname: str) -> str:
    if not opname.startswith("POP_JUMP_"):
        return opname
    return opname.replace("_FORWARD_", "_").replace("_BACKWARD_", "_")


def false_exit_choices(opname: str) -> tuple[int, ...]:
    opname = undirected(opname)
    if opname == "POP_JUMP_IF_FALSE":
        return (0,)
    if opname == "POP_JUMP_IF_TRUE":
        return (1,)
    if opname in NONE_JUMPS:
        # either exit can be the false one, depending on the last link
        return (0, 1)
    return ()


def cleanup_end_for_exit(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cleanup_index: int,
    false_index: int,
) -> int | None:
    after_pop = cleanup_index + 1
    false_index = follow_plain_jumps(instructions, offset_to_index, false_index)
    if follow_plain_jumps(instructions, offset_to_index, after_pop) == false_index:
        if instructions[after_pop].opname in PLAIN_JUMP_OPNAMES:
            return after_pop + 1
        return after_pop
    return copied_exit_end(instructions, after_pop, false_index)


def copied_exit_end(
    instructions: list[Instruction],
    copy_index: int,
    exit_index: int,
) -> int | None:
    while instructions[exit_index].opname == "NOP":
        exit_index += 1
    for step in range(len(instructions) - max(copy_index, exit_index)):
        copy = instructions[copy_index + step]
        original = instructions[exit_index + step]
        if copy.opname != original.opname or copy.argval != original.argval:
            return None
        if "JUMP" in copy.opname:
            return None
        if copy.opname in EXIT_OPNAMES:
            break
    else:
        return None
    if instructions[copy_index].is_jump_target:
        return copy_index
    return copy_index + step + 1


def follow_plain_jumps(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    index: int,
) -> int:
    for _step in range(8):
        instruction = instructions[index]
        if instruction.opname not in PLAIN_JUMP_OPNAMES:
            break
        target_index = offset_to_index.get(instruction.argval)
        if target_index is None:
            break
        index = target_index
    return index


def next_jump_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if "JUMP" in instructions[index].opname:
            return index
    return None


def previous_real_index(instructions: list[Instruction], index: int) -> int:
    while index > 0 and instructions[index].opname == "CACHE":
        index -= 1
    return index


def next_real_index(instructions: list[Instruction], index: int) -> int | None:
    while index < len(instructions) and instructions[index].opname == "CACHE":
        index += 1
    return index if index < len(instructions) else None


def end_offset(instructions: list[Instruction], index: int) -> int | None:
    return instructions[index].offset if index < len(instructions) else None


def make_nop(instruction: Instruction) -> Instruction:
    return replace(
        instruction,
        opname="NOP",
        arg=None,
        argval=None,
        argrepr="",
        starts_line=None,
        is_jump_target=False,
    )
