import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pyc2py.bytecode.instruction import Instruction
from pyc2py.stack import FastStack


@dataclass(slots=True)
class ReadBudget:
    steps: int = 0


@dataclass(slots=True)
class NativeResult:
    source: str
    warnings: tuple[str, ...]
    body: tuple[ast.stmt, ...] | None = None


@dataclass(frozen=True, slots=True)
class UnpackSlot:
    group: "UnpackGroup"
    index: int


@dataclass(slots=True)
class UnpackGroup:
    value: ast.expr
    targets: list[ast.expr | None]
    starred_index: int | None = None
    parent: UnpackSlot | None = None


class DecompilerContext(Protocol):
    code: object = None
    version: tuple[int, ...] | None = None
    is_module: bool = True
    stack: FastStack[object] = FastStack()
    statements: list[ast.stmt] = []
    warnings: list[str] = []
    kw_names: tuple[str, ...] = ()
    unpack_groups: list[UnpackGroup] = []
    pending_print_items: list[ast.expr] = []
    pending_print_target: ast.expr | None = None
    global_names: list[str] = []
    pending_simultaneous_store_count: int = 0
    simultaneous_store_group: int = 0
    simultaneous_store_reversed: bool = False
    nop_lines: list[int] = []
    saw_unicode_literals_future: bool = False
    saw_print_function_future: bool = False
    loop_continue_offsets: frozenset[int] = frozenset()
    loop_break_offsets: frozenset[int] = frozenset()
    loop_none_return_is_break: bool = False
    implicit_return_offsets: frozenset[int] = frozenset()
    written_return_offsets: frozenset[int] = frozenset()
    line_break_offsets: dict[int, tuple[str, ...]] = {}
    claimed_ranges: dict[int, int] = {}
    active_handlers: frozenset[int] = frozenset()
    detached_resume_offsets: frozenset[int] = frozenset()
    evaluating_test_operand: bool = False
    handler_exit_indexes: set[int] = set()
    compare_chain_links: dict[int, tuple[ast.expr, ast.Compare]] = {}
    compare_chain_cleanups: set[int] = set()
    budget: ReadBudget = ReadBudget()

    add_global_name: Callable[[str], None]
    decompile: Callable[[], NativeResult | None]
    join_compare_chain: Callable[[ast.Compare], ast.Compare]
    make_child: Callable[[], "DecompilerContext"]
    make_async_function_comprehension: Callable[
        [list[Instruction], dict[int, int], int, ast.expr], ast.expr | None
    ]
    make_function_comprehension: Callable[
        [list[Instruction], dict[int, int], int, ast.expr], ast.expr | None
    ]
    new_child: Callable[[object], "DecompilerContext"]
    build_looped_condition: Callable[
        [list[Instruction], dict[int, int], int, list[int], int], ast.expr | None
    ]
    evaluate_expression_range: Callable[[list[Instruction], int, int], ast.expr | None]
    expression_from_stack_value: Callable[[object], ast.expr]
    flush_legacy_line_boundary: Callable[[], None]
    flush_print_items: Callable[..., None]
    flush_yield_expressions: Callable[[], None]
    pop_binary: Callable[[], tuple[ast.expr, ast.expr]]
    pop_many: Callable[[int], list[ast.expr]]
    pop_many_raw: Callable[[int], list[object]]
    pop_or_none: Callable[[], object]
    prep_reraise_star_from_values: Callable[[ast.expr, ast.expr], None]
    previous_non_ignorable_index: Callable[[list[Instruction], int, int], int | None]
    read_for_loop_target: Callable[
        [list[Instruction], int, int], tuple[ast.expr | None, int]
    ]
    run_instruction: Callable[[Instruction], None]
    store_unpack_target: Callable[[object, ast.expr], None]
    translate_child_statements: Callable[[list[Instruction], int, int], list[ast.stmt]]
    translate_isolated_child_statements: Callable[
        [list[Instruction], int, int], list[ast.stmt]
    ]
    translate_loop_child_statements: Callable[..., list[ast.stmt]]
    translate_range: Callable[[list[Instruction], int, int], list[ast.stmt]]
    try_translate_for_loop: Callable[
        [list[Instruction], dict[int, int], int, int], int | None
    ]
    try_translate_legacy_list_comprehension: Callable[..., int | None]
