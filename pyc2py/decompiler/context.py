import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyc2py.bytecode.instruction import Instruction
from pyc2py.stack import FastStack


@dataclass(slots=True)
class ReadBudget:
    steps: int = 0


class DecompilerContext:
    code: Any
    version: tuple[int, ...] | None
    is_module: bool
    stack: FastStack[Any]
    statements: list[ast.stmt]
    warnings: list[str]
    kw_names: tuple[str, ...]
    unpack_groups: list[Any]
    pending_print_items: list[ast.expr]
    pending_print_target: ast.expr | None
    global_names: list[str]
    pending_simultaneous_store_count: int
    simultaneous_store_group: int
    simultaneous_store_reversed: bool
    nop_lines: list[int]
    saw_unicode_literals_future: bool
    saw_print_function_future: bool
    loop_continue_offsets: frozenset[int]
    loop_break_offsets: frozenset[int]
    loop_none_return_is_break: bool
    implicit_return_offsets: frozenset[int]
    written_return_offsets: frozenset[int]
    line_break_offsets: dict[int, tuple[str, ...]]
    claimed_ranges: dict[int, int]
    active_handlers: frozenset[int]
    detached_resume_offsets: frozenset[int]
    evaluating_test_operand: bool
    handler_exit_indexes: set[int]
    compare_chain_links: dict[int, tuple[ast.expr, ast.Compare]]
    compare_chain_cleanups: set[int]
    budget: ReadBudget

    add_global_name: Callable[[str], None]
    join_compare_chain: Callable[[ast.Compare], ast.Compare]
    build_looped_condition: Callable[
        [list[Instruction], dict[int, int], int, list[int], int], ast.expr | None
    ]
    evaluate_expression_range: Callable[[list[Instruction], int, int], ast.expr | None]
    expression_from_stack_value: Callable[[Any], ast.expr]
    flush_legacy_line_boundary: Callable[[], None]
    flush_print_items: Callable[..., None]
    flush_yield_expressions: Callable[[], None]
    make_child: Callable[[], Any]
    pop_binary: Callable[[], tuple[ast.expr, ast.expr]]
    pop_many: Callable[[int], list[ast.expr]]
    pop_many_raw: Callable[[int], list[Any]]
    pop_or_none: Callable[[], Any]
    prep_reraise_star_from_values: Callable[[ast.expr, ast.expr], None]
    previous_non_ignorable_index: Callable[[list[Instruction], int, int], int | None]
    read_for_loop_target: Callable[
        [list[Instruction], int, int], tuple[ast.expr | None, int]
    ]
    store_unpack_target: Callable[[Any, ast.expr], None]
    translate_child_statements: Callable[[list[Instruction], int, int], list[ast.stmt]]
    translate_isolated_child_statements: Callable[
        [list[Instruction], int, int], list[ast.stmt]
    ]
    translate_loop_child_statements: Callable[..., list[ast.stmt]]
    try_translate_legacy_list_comprehension: Callable[..., int | None]
