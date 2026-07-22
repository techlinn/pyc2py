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
    def first_instruction(self) -> Instruction | None:
        if not self.instructions:
            return None
        return self.instructions[0]

    @property
    def last_instruction(self) -> Instruction | None:
        if not self.instructions:
            return None
        return self.instructions[-1]


CONDITIONAL_JUMPS = {
    "FOR_ITER",
    "FOR_LOOP",
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
}

UNCONDITIONAL_JUMPS = {
    "CONTINUE_LOOP",
    "JUMP",
    "JUMP_ABSOLUTE",
    "JUMP_BACKWARD",
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

    def block_by_offset(self) -> dict[int, BasicBlock]:
        return {block.start_offset: block for block in self.blocks}

    def successors(self, offset: int) -> tuple[CFGEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source == offset)

    def predecessors(self, offset: int) -> tuple[CFGEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target == offset)


def build_cfg(
    instructions: list[Instruction],
    exception_entries: tuple[ExceptionTableEntry, ...] = (),
) -> ControlFlowGraph:
    if not instructions:
        return ControlFlowGraph(blocks=(), edges=(), entry_offset=None)

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

    return ControlFlowGraph(
        blocks=tuple(blocks),
        edges=tuple(edges),
        entry_offset=instructions[0].offset,
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
            edges, block, instruction, instructions, block_start_by_index
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
    instructions: list[Instruction],
    block_start_by_index: dict[int, int],
) -> None:
    if not can_fall_through(instruction):
        return

    next_index = find_instruction_index(instructions, instruction.offset) + 1
    target = block_start_by_index.get(next_index)
    if target is None or target == block.start_offset:
        return
    edges.append(CFGEdge(source=block.start_offset, target=target, kind="fallthrough"))


def find_instruction_index(instructions: list[Instruction], offset: int) -> int:
    for index, instruction in enumerate(instructions):
        if instruction.offset == offset:
            return index
    raise ValueError(f"instruction offset is not present: {offset}")


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


def compute_dominators(graph: ControlFlowGraph) -> dict[int, set[int]]:
    if graph.entry_offset is None:
        return {}

    block_offsets = {block.start_offset for block in graph.blocks}
    reachable = reachable_block_offsets(graph)
    dominators = {
        offset: set(reachable) if offset in reachable else {offset}
        for offset in block_offsets
    }
    dominators[graph.entry_offset] = {graph.entry_offset}

    max_passes = max(1, len(block_offsets))
    for _pass_index in range(max_passes):
        changed = False
        for offset in block_offsets:
            if offset == graph.entry_offset:
                continue
            if offset not in reachable:
                continue

            predecessor_offsets = {
                edge.source
                for edge in graph.predecessors(offset)
                if edge.source in reachable
            }
            if not predecessor_offsets:
                new_dominators = {offset}
            else:
                new_dominators = intersect_dominators(dominators, predecessor_offsets)
                new_dominators.add(offset)

            if new_dominators != dominators[offset]:
                dominators[offset] = new_dominators
                changed = True
        if not changed:
            return dominators

    return dominators


def reachable_block_offsets(graph: ControlFlowGraph) -> set[int]:
    if graph.entry_offset is None:
        return set()

    reachable = {graph.entry_offset}
    work = [graph.entry_offset]
    max_visits = max(1, len(graph.blocks))
    for _visit_index in range(max_visits):
        if not work:
            return reachable
        offset = work.pop()
        for edge in graph.successors(offset):
            if edge.target in reachable:
                continue
            reachable.add(edge.target)
            work.append(edge.target)

    return reachable


def intersect_dominators(
    dominators: dict[int, set[int]],
    offsets: set[int],
) -> set[int]:
    iterator = iter(offsets)
    first = next(iterator)
    result = set(dominators[first])
    for offset in iterator:
        result.intersection_update(dominators[offset])
    return result


def immediate_dominator(
    dominators: dict[int, set[int]],
    offset: int,
) -> int | None:
    # the immediate dominator is the unique strict dominator that every other
    # strict dominator also dominates
    strict = dominators.get(offset, set()) - {offset}
    for candidate in strict:
        others = strict - {candidate}
        if all(candidate in dominators.get(other, set()) for other in others):
            return candidate
    return None


# synthetic sink used so post-dominators have a single root even when a function
# has several exit blocks (multiple return / raise sites)
VIRTUAL_EXIT_OFFSET = -1


def exit_offsets(graph: ControlFlowGraph) -> set[int]:
    sources = {edge.source for edge in graph.edges}
    return {
        block.start_offset
        for block in graph.blocks
        if block.start_offset not in sources
    }


def reversed_graph_with_virtual_exit(graph: ControlFlowGraph) -> ControlFlowGraph:
    edges = [
        CFGEdge(source=edge.target, target=edge.source, kind=edge.kind)
        for edge in graph.edges
    ]
    edges.extend(
        (CFGEdge(source=VIRTUAL_EXIT_OFFSET, target=offset, kind="jump"))
        for offset in exit_offsets(graph)
    )
    blocks = (
        *graph.blocks,
        BasicBlock(start_offset=VIRTUAL_EXIT_OFFSET, instructions=()),
    )
    return ControlFlowGraph(
        blocks=blocks,
        edges=tuple(edges),
        entry_offset=VIRTUAL_EXIT_OFFSET,
    )


def compute_post_dominators(graph: ControlFlowGraph) -> dict[int, set[int]]:
    if graph.entry_offset is None or not graph.blocks:
        return {}

    reverse_dominators = compute_dominators(reversed_graph_with_virtual_exit(graph))
    post_dominators: dict[int, set[int]] = {}
    for offset, values in reverse_dominators.items():
        if offset == VIRTUAL_EXIT_OFFSET:
            continue
        post_dominators[offset] = values - {VIRTUAL_EXIT_OFFSET}
    return post_dominators


def common_post_dominator(
    post_dominators: dict[int, set[int]],
    offsets: set[int],
) -> int | None:
    # the nearest block both branch arms reach: the immediate post-dominator
    # shared by every offset in the set
    candidates: set[int] | None = None
    for offset in offsets:
        values = post_dominators.get(offset, set()) - {offset}
        candidates = set(values) if candidates is None else (candidates & values)
    if not candidates:
        return None

    for candidate in candidates:
        others = candidates - {candidate}
        if all(candidate in post_dominators.get(other, set()) for other in others):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class LoopRegion:
    header: int
    latch: int
    blocks: frozenset[int]


def find_loop_regions(graph: ControlFlowGraph) -> tuple[LoopRegion, ...]:
    dominators = compute_dominators(graph)
    regions = []
    for edge in graph.edges:
        if edge.target not in dominators.get(edge.source, set()):
            continue
        regions.append(make_loop_region(graph, edge))

    return tuple(regions)


def make_loop_region(graph: ControlFlowGraph, edge: CFGEdge) -> LoopRegion:
    blocks = {edge.target, edge.source}
    work = [edge.source]

    max_visits = max(1, len(graph.blocks))
    for _visit_index in range(max_visits):
        if not work:
            break
        offset = work.pop()
        for predecessor in graph.predecessors(offset):
            if predecessor.source in blocks:
                continue
            blocks.add(predecessor.source)
            work.append(predecessor.source)

    return LoopRegion(
        header=edge.target,
        latch=edge.source,
        blocks=frozenset(blocks),
    )


@dataclass(frozen=True, slots=True)
class ExceptionRegion:
    start_offset: int
    end_offset: int
    handler_offset: int
    kind: str


@dataclass(slots=True)
class CFGValidation:
    block_count: int
    edge_count: int
    unreachable: tuple[int, ...]
    warnings: list[str] = field(default_factory=list)

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


@dataclass(slots=True)
class CFGAnalysisValidation:
    dominator_count: int
    loop_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.warnings

    @property
    def checks(self) -> tuple[str, ...]:
        return (
            f"CFG dominator sets: {self.dominator_count}",
            f"CFG natural loops: {self.loop_count}",
        )


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


def validate_cfg_analysis(graph: ControlFlowGraph) -> CFGAnalysisValidation:
    dominators = compute_dominators(graph)
    loops = find_loop_regions(graph)
    warnings: list[str] = []
    block_offsets = {block.start_offset for block in graph.blocks}
    reachable = reachable_offsets(graph)

    warnings.extend(
        validate_dominator_sets(graph, dominators, block_offsets, reachable)
    )
    warnings.extend(validate_loop_regions(loops, dominators, block_offsets))

    return CFGAnalysisValidation(
        dominator_count=len(dominators),
        loop_count=len(loops),
        warnings=warnings,
    )


def validate_dominator_sets(
    graph: ControlFlowGraph,
    dominators: dict[int, set[int]],
    block_offsets: set[int],
    reachable: set[int],
) -> list[str]:
    warnings: list[str] = []
    if set(dominators) != block_offsets:
        missing = sorted(block_offsets - set(dominators))
        extra = sorted(set(dominators) - block_offsets)
        if missing:
            warnings.append(f"CFG dominators missing blocks: {missing[:8]}")
        if extra:
            warnings.append(f"CFG dominators reference unknown blocks: {extra[:8]}")

    for offset, values in dominators.items():
        unknown = values - block_offsets
        if unknown:
            warnings.append(
                f"CFG dominator set has unknown offsets at {offset}: {sorted(unknown)[:8]}"
            )
        if offset not in values:
            warnings.append(f"CFG block does not dominate itself: {offset}")
        if (
            graph.entry_offset is not None
            and offset in reachable
            and graph.entry_offset not in values
        ):
            warnings.append(f"CFG entry does not dominate reachable block: {offset}")

    return warnings


def validate_loop_regions(
    loops: tuple[LoopRegion, ...],
    dominators: dict[int, set[int]],
    block_offsets: set[int],
) -> list[str]:
    warnings: list[str] = []
    for loop in loops:
        header = loop.header
        latch = loop.latch
        blocks = loop.blocks

        if header not in block_offsets:
            warnings.append(f"CFG loop header is missing: {header}")
        if latch not in block_offsets:
            warnings.append(f"CFG loop latch is missing: {latch}")
        if header not in blocks or latch not in blocks:
            warnings.append(
                f"CFG loop does not contain header and latch: {header}->{latch}"
            )

        unknown_blocks = blocks - block_offsets
        if unknown_blocks:
            warnings.append(
                f"CFG loop has unknown blocks: {sorted(unknown_blocks)[:8]}"
            )
        if header not in dominators.get(latch, set()):
            warnings.append(
                f"CFG loop header does not dominate latch: {header}->{latch}"
            )

    return warnings


def to_dot(graph: ControlFlowGraph) -> str:
    lines = ["digraph cfg {"]
    for block in graph.blocks:
        label = f"{block.start_offset}:{block.end_offset}"
        lines.append(f'  "{block.start_offset}" [label="{label}"];')
    lines.extend(
        f'  "{edge.source}" -> "{edge.target}" [label="{edge.kind}"];'
        for edge in graph.edges
    )
    lines.append("}")
    return "\n".join(lines)
