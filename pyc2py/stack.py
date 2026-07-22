from dataclasses import dataclass, field
from typing import Generic, TypeVar, overload
from collections.abc import Iterable, Iterator, MutableSequence
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.stack_effect import instruction_stack_effect
from pyc2py.cfg import BasicBlock, CFGEdge, ControlFlowGraph, build_cfg

T = TypeVar("T")

class StackUnderflowError(IndexError):
    pass

class FastStack(Generic[T], MutableSequence[T]):
    def __init__(self, values: Iterable[T] = (), max_depth: int | None = None) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be positive")

        self.max_depth = max_depth
        self.values = list(values)
        self.check_depth()

    def __bool__(self) -> bool:
        return bool(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self.values[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            self.values[index] = list(value)  # type: ignore[arg-type]
        else:
            self.values[index] = value  # type: ignore[assignment]
        self.check_depth()

    def __delitem__(self, index: int | slice) -> None:
        del self.values[index]

    def insert(self, index: int, value: T) -> None:
        self.values.insert(index, value)
        self.check_depth()

    def append(self, value: T) -> None:
        self.values.append(value)
        self.check_depth()

    def pop(self, index: int = -1) -> T:
        if not self.values:
            raise StackUnderflowError("cannot pop from an empty stack")
        return self.values.pop(index)

    def clear(self) -> None:
        self.values.clear()

    def copy(self) -> "FastStack[T]":
        return FastStack(self.values, max_depth=self.max_depth)

    def to_list(self) -> list[T]:
        return list(self.values)

    def peek(self, depth: int = 0) -> T:
        if depth < 0:
            raise ValueError("depth must not be negative")

        index = len(self.values) - depth - 1
        if index < 0:
            raise StackUnderflowError("cannot peek beyond the stack depth")

        return self.values[index]

    def pop_many(self, count: int) -> list[T]:
        if count < 0:
            raise ValueError("count must not be negative")
        if count > len(self.values):
            raise StackUnderflowError("cannot pop more values than the stack contains")
        if count == 0:
            return []

        start = len(self.values) - count
        result = self.values[start:]
        del self.values[start:]
        return result

    def duplicate_top(self) -> T:
        value = self.peek()
        self.append(value)
        return value

    def copy_from_top(self, depth: int) -> T:
        value = self.peek(depth)
        self.append(value)
        return value

    def rotate_top(self, count: int) -> None:
        if count < 2:
            return
        if count > len(self.values):
            raise StackUnderflowError("cannot rotate beyond the stack depth")

        items = self.values[-count:]
        self.values[-count:] = [items[-1], *items[:-1]]

    def swap_top(self, depth: int) -> None:
        if depth < 0:
            raise ValueError("depth must not be negative")

        target_index = len(self.values) - depth - 1
        if target_index < 0:
            raise StackUnderflowError("cannot swap beyond the stack depth")
        self.values[-1], self.values[target_index] = (
            self.values[target_index],
            self.values[-1],
        )

    def check_depth(self) -> None:
        if self.max_depth is not None and len(self.values) > self.max_depth:
            raise ValueError("stack exceeded max_depth")

MAX_CFG_STACK_STATES = 4096

@dataclass(slots=True)
class StackValidation:
    instruction_count: int
    known_effects: int
    unknown_effects: int
    max_depth: int
    final_depth: int
    underflow_risks: int
    cfg_state_count: int | None = None
    cfg_max_depth: int | None = None
    cfg_terminal_depths: tuple[int, ...] = ()
    cfg_underflow_risks: int | None = None
    cfg_depth_conflicts: int = 0
    cfg_exception_edges_skipped: int = 0
    declared_stacksize: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def checks(self) -> tuple[str, ...]:
        checks = [
            f"stack effects known: {self.known_effects}/{self.instruction_count}",
            f"stack effects unknown: {self.unknown_effects}",
            f"linear stack max depth: {self.max_depth}",
            f"linear stack final depth: {self.final_depth}",
            f"linear stack underflow risks: {self.underflow_risks}",
        ]
        if self.declared_stacksize is not None:
            checks.append(f"declared stack size: {self.declared_stacksize}")
        if self.cfg_state_count is not None and self.cfg_max_depth is not None:
            checks.extend(
                [
                    f"CFG stack states visited: {self.cfg_state_count}",
                    f"CFG stack max depth: {self.cfg_max_depth}",
                    f"CFG stack terminal depths: {format_depths(self.cfg_terminal_depths)}",
                ]
            )
        if self.cfg_underflow_risks is not None:
            checks.append(f"CFG stack underflow risks: {self.cfg_underflow_risks}")
        if self.cfg_depth_conflicts:
            checks.append(
                f"CFG stack entry-depth conflicts: {self.cfg_depth_conflicts}"
            )
        if self.cfg_exception_edges_skipped:
            checks.append(
                f"CFG stack exception edges skipped: {self.cfg_exception_edges_skipped}"
            )
        return tuple(checks)

def validate_linear_stack_effects(
    instructions: list[Instruction],
    expected_stacksize: int | None = None,
    version: tuple[int, ...] | None = None,
) -> StackValidation:
    depth = 0
    max_depth = 0
    known_effects = 0
    unknown_effects = 0
    underflow_risks = 0

    for instruction in instructions:
        step = apply_instruction_effect(depth, instruction, version)
        if not step.known:
            unknown_effects += 1
            continue

        known_effects += 1
        underflow_risks += int(step.underflow)
        depth = step.depth
        max_depth = max(max_depth, depth)

    cfg_validation = validate_cfg_stack_depths(instructions, version)
    declared_stacksize = None
    if expected_stacksize is not None and expected_stacksize >= 0:
        declared_stacksize = expected_stacksize

    return StackValidation(
        instruction_count=len(instructions),
        known_effects=known_effects,
        unknown_effects=unknown_effects,
        max_depth=max_depth,
        final_depth=depth,
        underflow_risks=underflow_risks,
        cfg_state_count=cfg_validation.state_count,
        cfg_max_depth=cfg_validation.max_depth,
        cfg_terminal_depths=cfg_validation.terminal_depths,
        cfg_underflow_risks=cfg_validation.underflow_risks,
        cfg_depth_conflicts=cfg_validation.depth_conflicts,
        cfg_exception_edges_skipped=cfg_validation.exception_edges_skipped,
        declared_stacksize=declared_stacksize,
        warnings=cfg_validation.warnings,
    )

@dataclass(frozen=True, slots=True)
class StackStep:
    depth: int
    known: bool
    underflow: bool

@dataclass(frozen=True, slots=True)
class CFGStackValidation:
    state_count: int
    max_depth: int
    terminal_depths: tuple[int, ...]
    underflow_risks: int
    depth_conflicts: int
    exception_edges_skipped: int
    warnings: list[str] = field(default_factory=list)

def validate_cfg_stack_depths(
    instructions: list[Instruction],
    version: tuple[int, ...] | None = None,
) -> CFGStackValidation:
    graph = build_cfg(instructions)
    blocks = graph.block_by_offset()
    if graph.entry_offset is None:
        return make_empty_cfg_stack_validation()

    state = CFGStackWalkState()
    terminal_depths: set[int] = set()
    entry_depth_by_block: dict[int, int] = {}
    work: list[tuple[int, int]] = [(graph.entry_offset, 0)]
    warnings: list[str] = []

    for _index in range(MAX_CFG_STACK_STATES):
        if not work:
            break
        block_offset, entry_depth = work.pop()
        previous_depth = entry_depth_by_block.get(block_offset)
        if previous_depth is not None:
            state.depth_conflicts += int(previous_depth != entry_depth)
            continue
        entry_depth_by_block[block_offset] = entry_depth
        state.state_count += 1

        block = blocks.get(block_offset)
        if block is None:
            continue

        exit_depth = apply_block_stack_effects(
            state,
            block.instructions,
            entry_depth,
            version,
        )
        successors = non_exception_successors(graph, block_offset)
        state.exception_edges_skipped += len(graph.successors(block_offset)) - len(
            successors
        )
        if not successors:
            terminal_depths.add(exit_depth)
            continue

        append_successor_depths(work, successors, blocks, exit_depth)
    else:
        warnings.append("CFG stack state limit reached")

    return CFGStackValidation(
        state_count=state.state_count,
        max_depth=state.max_depth,
        terminal_depths=tuple(sorted(terminal_depths)),
        underflow_risks=state.underflow_risks,
        depth_conflicts=state.depth_conflicts,
        exception_edges_skipped=state.exception_edges_skipped,
        warnings=warnings,
    )

@dataclass(slots=True)
class CFGStackWalkState:
    state_count: int = 0
    max_depth: int = 0
    underflow_risks: int = 0
    depth_conflicts: int = 0
    exception_edges_skipped: int = 0

def make_empty_cfg_stack_validation() -> CFGStackValidation:
    return CFGStackValidation(
        state_count=0,
        max_depth=0,
        terminal_depths=(),
        underflow_risks=0,
        depth_conflicts=0,
        exception_edges_skipped=0,
    )

def apply_block_stack_effects(
    state: CFGStackWalkState,
    instructions: tuple[Instruction, ...],
    entry_depth: int,
    version: tuple[int, ...] | None = None,
) -> int:
    depth = entry_depth
    for instruction in instructions:
        step = apply_instruction_effect(depth, instruction, version)
        if not step.known:
            continue

        state.underflow_risks += int(step.underflow)
        depth = step.depth
        state.max_depth = max(state.max_depth, depth)
    return depth

def non_exception_successors(
    graph: ControlFlowGraph, block_offset: int
) -> tuple[CFGEdge, ...]:
    return tuple(
        edge for edge in graph.successors(block_offset) if edge.kind != "exception"
    )

def append_successor_depths(
    work: list[tuple[int, int]],
    successors: tuple[CFGEdge, ...],
    blocks: dict[int, BasicBlock],
    exit_depth: int,
) -> None:
    work.extend((edge.target, exit_depth) for edge in successors if edge.target in blocks)

def apply_instruction_effect(
    depth: int,
    instruction: Instruction,
    version: tuple[int, ...] | None = None,
) -> StackStep:
    effect = instruction_stack_effect(instruction, version)
    if effect is None:
        return StackStep(depth=depth, known=False, underflow=False)
    if effect.pops > depth:
        return StackStep(depth=effect.pushes, known=True, underflow=True)
    return StackStep(
        depth=depth - effect.pops + effect.pushes, known=True, underflow=False
    )

def format_depths(depths: tuple[int, ...]) -> str:
    if not depths:
        return "none"
    if len(depths) <= 8:
        return ", ".join(str(depth) for depth in depths)
    prefix = ", ".join(str(depth) for depth in depths[:8])
    return f"{prefix}, ..."
