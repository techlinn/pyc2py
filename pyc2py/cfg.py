from dataclasses import dataclass, field
from typing import Literal

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import ExceptionTableEntry
from pyc2py.constants import MAX_CFG_BLOCKS, MAX_CFG_EDGES

EdgeKind = Literal["fallthrough", "jump", "conditional", "exception"]


@dataclass(frozen=True, slots=True)
class CFGEdge:
    source: int
    target: int
    kind: EdgeKind


@dataclass(frozen=True, slots=True)
class BasicBlock:
    start_offset: int
    instructions: tuple[Instruction, ...]

    @property
    def end_offset(self) -> int:
        if not self.instructions:
            return self.start_offset

        return self.instructions[-1].offset

    @property
    def last_instruction(self) -> Instruction | None:
        if not self.instructions:
            return None

        return self.instructions[-1]


CONDITIONAL_JUMPS = {
    "FOR_ITER",
    "JUMP_IF_FALSE",
    "JUMP_IF_TRUE",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_FORWARD_IF_NONE",
    "POP_JUMP_FORWARD_IF_NOT_NONE",
    "POP_JUMP_BACKWARD_IF_NONE",
    "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "SEND",
}

UNCONDITIONAL_JUMPS = {
    "CONTINUE_LOOP",
    "JUMP",
    "JUMP_ABSOLUTE",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
}

TERMINATORS = {
    "RAISE_VARARGS",
    "RERAISE",
    "RETURN_CONST",
    "RETURN_VALUE",
}

EXCEPTION_SETUP_OPS = {
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
    "SETUP_WITH",
    "SETUP_ASYNC_WITH",
}


@dataclass(frozen=True, slots=True)
class ControlFlowGraph:
    blocks: tuple[BasicBlock, ...]
    edges: tuple[CFGEdge, ...]
    entry_offset: int | None
    # each block's edges, by its start offset
    outgoing: dict[int, tuple[CFGEdge, ...]]

    def block_by_offset(self) -> dict[int, BasicBlock]:
        return {block.start_offset: block for block in self.blocks}

    def successors(self, offset: int) -> tuple[CFGEdge, ...]:
        return self.outgoing.get(offset, ())


def build_cfg(
    instructions: list[Instruction],
    exception_entries: tuple[ExceptionTableEntry, ...] = (),
) -> ControlFlowGraph:
    if not instructions:
        return ControlFlowGraph(blocks=(), edges=(), entry_offset=None, outgoing={})

    offset_to_index = {
        instruction.offset: index for index, instruction in enumerate(instructions)
    }
    leaders = find_leaders(instructions, offset_to_index, exception_entries)
    blocks = make_blocks(instructions, leaders)
    block_start_by_index = make_block_start_by_index(instructions, blocks)
    edges = make_edges(
        blocks,
        instructions,
        offset_to_index,
        block_start_by_index,
        exception_entries,
    )
    outgoing: dict[int, list[CFGEdge]] = {}

    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)

    return ControlFlowGraph(
        blocks=tuple(blocks),
        edges=tuple(edges),
        entry_offset=instructions[0].offset,
        outgoing={source: tuple(found) for source, found in outgoing.items()},
    )


def find_leaders(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    exception_entries: tuple[ExceptionTableEntry, ...] = (),
) -> set[int]:
    leaders = {instructions[0].offset}

    for index, instruction in enumerate(instructions):
        target = jump_target(instruction, offset_to_index)
        if target is not None:
            leaders.add(target)
        if ends_block(instruction) and index + 1 < len(instructions):
            leaders.add(instructions[index + 1].offset)

    for entry in exception_entries:
        add_exception_leaders(leaders, entry, offset_to_index)

    return leaders


def add_exception_leaders(
    leaders: set[int],
    entry: ExceptionTableEntry,
    offset_to_index: dict[int, int],
) -> None:
    for offset in (entry.start_offset, entry.end_offset, entry.target_offset):
        if offset in offset_to_index:
            leaders.add(offset)


def make_blocks(instructions: list[Instruction], leaders: set[int]) -> list[BasicBlock]:
    blocks: list[BasicBlock] = []
    current: list[Instruction] = []

    for instruction in instructions:
        if instruction.offset in leaders and current:
            blocks.append(make_block(current))
            current = []

        current.append(instruction)

    if current:
        blocks.append(make_block(current))

    return blocks


def make_block(instructions: list[Instruction]) -> BasicBlock:
    return BasicBlock(
        start_offset=instructions[0].offset,
        instructions=tuple(instructions),
    )


def make_block_start_by_index(
    instructions: list[Instruction],
    blocks: list[BasicBlock],
) -> dict[int, int]:
    index_by_offset = {
        instruction.offset: index for index, instruction in enumerate(instructions)
    }
    result: dict[int, int] = {}

    for block in blocks:
        start_index = index_by_offset[block.start_offset]
        for index in range(start_index, start_index + len(block.instructions)):
            result[index] = block.start_offset

    return result


def make_edges(
    blocks: list[BasicBlock],
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    block_start_by_index: dict[int, int],
    exception_entries: tuple[ExceptionTableEntry, ...] = (),
) -> list[CFGEdge]:
    edges: list[CFGEdge] = []

    for block in blocks:
        instruction = block.last_instruction
        if instruction is None:
            continue

        add_jump_edge(edges, block.start_offset, instruction, offset_to_index)
        add_fallthrough_edge(
            edges, block, instruction, offset_to_index, block_start_by_index
        )

    add_exception_table_edges(edges, blocks, exception_entries)
    return edges


def add_exception_table_edges(
    edges: list[CFGEdge],
    blocks: list[BasicBlock],
    exception_entries: tuple[ExceptionTableEntry, ...],
) -> None:
    block_offsets = {block.start_offset for block in blocks}

    for entry in exception_entries:
        if entry.target_offset not in block_offsets:
            continue

        for block in blocks:
            if not block_overlaps_exception_entry(block, entry):
                continue

            edges.append(
                CFGEdge(
                    source=block.start_offset,
                    target=entry.target_offset,
                    kind="exception",
                )
            )


def block_overlaps_exception_entry(
    block: BasicBlock, entry: ExceptionTableEntry
) -> bool:
    return (
        block.start_offset < entry.end_offset and block.end_offset >= entry.start_offset
    )


def add_jump_edge(
    edges: list[CFGEdge],
    source: int,
    instruction: Instruction,
    offset_to_index: dict[int, int],
) -> None:
    target = jump_target(instruction, offset_to_index)
    if target is None:
        return

    kind: EdgeKind = (
        "conditional" if is_conditional_jump(instruction.opname) else "jump"
    )

    if instruction.opname in EXCEPTION_SETUP_OPS:
        kind = "exception"

    edges.append(CFGEdge(source=source, target=target, kind=kind))


def add_fallthrough_edge(
    edges: list[CFGEdge],
    block: BasicBlock,
    instruction: Instruction,
    offset_to_index: dict[int, int],
    block_start_by_index: dict[int, int],
) -> None:
    if not can_fall_through(instruction):
        return

    next_index = offset_to_index[instruction.offset] + 1
    target = block_start_by_index.get(next_index)
    if target is None or target == block.start_offset:
        return

    edges.append(CFGEdge(source=block.start_offset, target=target, kind="fallthrough"))


def jump_target(
    instruction: Instruction, offset_to_index: dict[int, int]
) -> int | None:
    if not is_jump_like(instruction.opname):
        return None
    if not isinstance(instruction.argval, int):
        return None
    if instruction.argval not in offset_to_index:
        return None

    return int(instruction.argval)


def is_jump_like(opname: str) -> bool:
    return (
        is_conditional_jump(opname)
        or opname in UNCONDITIONAL_JUMPS
        or opname in EXCEPTION_SETUP_OPS
    )


def is_conditional_jump(opname: str) -> bool:
    if opname in CONDITIONAL_JUMPS:
        return True

    return "IF_FALSE" in opname or "IF_TRUE" in opname


def can_fall_through(instruction: Instruction) -> bool:
    if instruction.opname in TERMINATORS:
        return False

    return instruction.opname not in UNCONDITIONAL_JUMPS


def ends_block(instruction: Instruction) -> bool:
    return is_jump_like(instruction.opname) or instruction.opname in TERMINATORS


@dataclass(slots=True)
class CFGValidation:
    block_count: int
    edge_count: int
    unreachable: tuple[int, ...]
    warnings: list[str] = field(default_factory=list[str])

    @property
    def passed(self) -> bool:
        return not self.warnings

    @property
    def checks(self) -> tuple[str, ...]:
        checks = [
            f"CFG block count: {self.block_count}",
            f"CFG edge count: {self.edge_count}",
        ]

        if self.unreachable:
            checks.append(f"CFG unreachable blocks: {len(self.unreachable)}")
        else:
            checks.append("CFG all blocks reachable")

        return tuple(checks)


def reachable_offsets(graph: ControlFlowGraph) -> set[int]:
    if graph.entry_offset is None:
        return set()

    reachable = {graph.entry_offset}
    work = [graph.entry_offset]
    max_visits = max(1, len(graph.blocks))
    for _visit_index in range(max_visits):
        if not work:
            break

        offset = work.pop()
        for edge in graph.successors(offset):
            if edge.target in reachable:
                continue

            reachable.add(edge.target)
            work.append(edge.target)

    return reachable


def unreachable_offsets(graph: ControlFlowGraph) -> set[int]:
    all_offsets = {block.start_offset for block in graph.blocks}
    return all_offsets - reachable_offsets(graph)


def validate_cfg(graph: ControlFlowGraph) -> CFGValidation:
    warnings: list[str] = []
    block_offsets = {block.start_offset for block in graph.blocks}

    if len(graph.blocks) > MAX_CFG_BLOCKS:
        warnings.append("CFG block count exceeds local limit")
    if len(graph.edges) > MAX_CFG_EDGES:
        warnings.append("CFG edge count exceeds local limit")
    if graph.entry_offset is not None and graph.entry_offset not in block_offsets:
        warnings.append("CFG entry offset does not reference a block")

    for edge in graph.edges:
        if edge.source not in block_offsets:
            warnings.append(f"CFG edge source is missing: {edge.source}")
        if edge.target not in block_offsets:
            warnings.append(f"CFG edge target is missing: {edge.target}")

    unreachable = tuple(sorted(unreachable_offsets(graph)))
    return CFGValidation(
        block_count=len(graph.blocks),
        edge_count=len(graph.edges),
        unreachable=unreachable,
        warnings=warnings,
    )
