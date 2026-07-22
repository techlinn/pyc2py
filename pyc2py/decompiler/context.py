import ast
from collections.abc import Callable
from typing import Any

from pyc2py.bytecode.instruction import Instruction
from pyc2py.stack import FastStack


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
    saw_unicode_literals_future: bool
    saw_print_function_future: bool
    loop_continue_offsets: frozenset[int]
    loop_break_offsets: frozenset[int]
    loop_none_return_is_break: bool

    add_global_name: Callable[[str], None]
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
