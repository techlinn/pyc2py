import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pyc2py.astree import (
    LINE_BREAK_ALWAYS,
    LINE_BREAK_IN_COMPREHENSION,
    add_global_declarations,
    add_module_global_declarations,
    add_nonlocal_declarations,
    clean_decompiled_body,
    merge_copied_return_branches,
    name_line_breaks,
    unparse_keeping_lines,
)
from pyc2py.bytecode.decoder import decode_instructions
from pyc2py.bytecode.instruction import IGNORED_BEHAVIOR_OPNAMES, Instruction
from pyc2py.bytecode.metadata import make_offset_index
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.decompiler.context import NativeResult, ReadBudget, UnpackGroup
from pyc2py.decompiler.control import ControlRecoveryMixin
from pyc2py.decompiler.control_comprehension import INLINE_CLEANUP_OPS
from pyc2py.decompiler.exception_structures import (
    make_handler_exit_marker,
    try_translate_except,
    try_translate_exception_table_except,
    try_translate_exception_table_except_finally,
    try_translate_exception_table_except_star,
    try_translate_exception_table_finally,
    try_translate_finally,
    try_translate_out_of_line_try,
)
from pyc2py.decompiler.opcode_runtime import OpcodeRuntimeMixin
from pyc2py.decompiler.opcodes.flow import (
    is_jump_op,
    is_terminal_op,
    terminal_tail_end_index,
)
from pyc2py.decompiler.print_runtime import (
    PrintRuntimeMixin,
    render_legacy_print_source,
)
from pyc2py.decompiler.recover import (
    CO_ASYNC_GENERATOR,
    CO_VARARGS,
    CO_VARKEYWORDS,
)
from pyc2py.decompiler.runtime import take_scope_warnings
from pyc2py.decompiler.stack_runtime import StackRuntimeMixin
from pyc2py.decompiler.structures import (
    is_false_jump,
    is_forward_conditional_jump,
    skip_returning_with_exception_handler,
)
from pyc2py.decompiler.variable_annotations import restore_variable_annotations
from pyc2py.stack import FastStack
from pyc2py.types import LiveWarningList, ProgressCallback

READ_STEPS_PER_INSTRUCTION = 100
READ_STEPS_ALLOWANCE = 10_000


class ReadBudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class NativeDecompiler(
    ControlRecoveryMixin,
    PrintRuntimeMixin,
    StackRuntimeMixin,
    OpcodeRuntimeMixin,
):
    code: object
    version: tuple[int, ...] | None
    is_module: bool = True
    progress: ProgressCallback | None = None
    stack: FastStack[object] = field(default_factory=FastStack[object])
    statements: list[ast.stmt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kw_names: tuple[str, ...] = ()
    unpack_groups: list[UnpackGroup] = field(default_factory=list)
    pending_print_items: list[ast.expr] = field(default_factory=list)
    pending_print_target: ast.expr | None = None
    global_names: list[str] = field(default_factory=list)
    pending_simultaneous_store_count: int = 0
    simultaneous_store_group: int = 0
    simultaneous_store_reversed: bool = False
    nop_lines: list[int] = field(default_factory=list)
    saw_unicode_literals_future: bool = False
    saw_print_function_future: bool = False
    loop_continue_offsets: frozenset[int] = field(default_factory=frozenset)
    loop_break_offsets: frozenset[int] = field(default_factory=frozenset)
    loop_none_return_is_break: bool = False
    compare_chain_links: dict[int, tuple[ast.expr, ast.Compare]] = field(
        default_factory=dict
    )
    compare_chain_cleanups: set[int] = field(default_factory=set)
    handler_exit_indexes: set[int] = field(default_factory=set)
    detached_resume_offsets: frozenset[int] = field(default_factory=frozenset)
    evaluating_test_operand: bool = False
    claimed_ranges: dict[int, int] = field(default_factory=dict)
    active_handlers: frozenset[int] = field(default_factory=frozenset)
    implicit_return_offsets: frozenset[int] = field(default_factory=frozenset)
    written_return_offsets: frozenset[int] = field(default_factory=frozenset)
    line_break_offsets: dict[int, tuple[str, ...]] = field(default_factory=dict)
    budget: ReadBudget = field(default_factory=ReadBudget)

    def __post_init__(self) -> None:
        if self.progress is not None and type(self.warnings) is list:
            self.warnings = LiveWarningList(self.progress, prefix="warning: native: ")

    def new_child(self, code: object) -> "NativeDecompiler":
        return NativeDecompiler(code=code, version=self.version, is_module=False)

    def make_child(self) -> "NativeDecompiler":
        return NativeDecompiler(
            code=self.code,
            version=self.version,
            is_module=False,
            stack=self.stack.copy(),
            kw_names=self.kw_names,
            global_names=self.global_names,
            loop_continue_offsets=self.loop_continue_offsets,
            loop_break_offsets=self.loop_break_offsets,
            loop_none_return_is_break=self.loop_none_return_is_break,
            claimed_ranges=self.claimed_ranges,
            active_handlers=self.active_handlers,
            implicit_return_offsets=self.implicit_return_offsets,
            written_return_offsets=self.written_return_offsets,
            line_break_offsets=self.line_break_offsets,
            detached_resume_offsets=self.detached_resume_offsets,
            handler_exit_indexes=self.handler_exit_indexes,
            compare_chain_links=self.compare_chain_links,
            compare_chain_cleanups=self.compare_chain_cleanups,
            budget=self.budget,
        )

    def uses_unicode_literals(self) -> bool:
        flags = int(getattr(self.code, "co_flags", 0) or 0)
        return self.saw_unicode_literals_future or bool(
            flags & CO_FUTURE_UNICODE_LITERALS
        )

    def decompile(self) -> NativeResult | None:
        self.emit_progress(f"decoding bytecode for {self.code_name()}")
        instructions = decode_instructions(self.code, self.version)
        if not instructions:
            self.emit_progress("no instructions found")
            return None

        self.emit_progress(f"decoded {len(instructions)} instruction(s)")
        self.emit_progress("translating bytecode instructions")
        self.implicit_return_offsets = find_implicit_return_offsets(
            instructions, has_return_const(self.version)
        )
        if not has_return_const(self.version):
            self.written_return_offsets = find_written_return_offsets(instructions)
        self.line_break_offsets = find_line_break_offsets(instructions, self.version)
        self.statements = self.translate_range(instructions, 0, len(instructions))
        self.emit_progress("flushing pending print statements")
        self.flush_print_items(newline=True)
        if not self.is_module:
            self.emit_progress("recovering function global declarations")
            self.add_global_names(find_loaded_global_store_sources(instructions))

        self.emit_progress("cleaning recovered AST")
        if self.version is not None and self.version >= (3, 14):
            self.statements = merge_copied_return_branches(self.statements)
            self.statements = restore_variable_annotations(
                self.statements, instructions, self.version, self.new_child
            )
        body = clean_decompiled_body(self.statements, self.is_module)
        if self.is_module:
            self.emit_progress("recovering module global declarations")
            body = add_module_global_declarations(
                body,
                [
                    *find_module_explicit_global_names(instructions),
                    *find_undeclared_module_globals(body, instructions),
                ],
            )
        else:
            body = restore_else_before_shared_return(body, self.code)
            body = add_missing_local_initializers(body, instructions, self.code)
            body = add_dead_yield(body, self.code)
            spell_while_one(body, instructions, self.code)
            body = add_global_declarations(body, self.global_names)
            body = add_nonlocal_declarations(
                body, find_explicit_nonlocal_names(self.code, instructions)
            )
        if not body and not self.is_module:
            body.append(ast.Pass())
        self.warnings.extend(take_scope_warnings(body))
        self.emit_progress("fixing AST locations")
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        self.emit_progress("unparsing AST to source")
        if self.is_module:
            name_line_breaks(module)
        source = unparse_keeping_lines(module) + "\n"
        if (
            self.should_render_legacy_print_source()
            and not self.saw_print_function_future
        ):
            self.emit_progress("rendering legacy print statements")
            source = render_legacy_print_source(source)
            return NativeResult(source=source, warnings=tuple(self.warnings))
        return NativeResult(
            source=source, warnings=tuple(self.warnings), body=tuple(body)
        )

    def spend_read_step(self, size: int) -> None:
        self.budget.steps += 1
        limit = READ_STEPS_PER_INSTRUCTION * size + READ_STEPS_ALLOWANCE
        if self.budget.steps > limit:
            raise ReadBudgetExceededError(
                f"{self.code_name()}: gave up after {limit} reading steps"
                f" for {size} instructions"
            )

    def emit_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def code_name(self) -> str:
        return str(
            getattr(self.code, "co_qualname", None)
            or getattr(self.code, "co_name", "code")
        )

    def should_render_legacy_print_source(self) -> bool:
        if self.version is None or self.version >= (3, 0):
            return False
        return self.is_module

    def should_skip_loop_break_cleanup_pop(
        self,
        instructions: list[Instruction],
        cursor: int,
    ) -> bool:
        if instructions[cursor].opname != "POP_TOP":
            return False
        if self.stack:
            return False

        next_index = next_non_metadata_index(instructions, cursor + 1)
        if next_index is None:
            return False

        next_instruction = instructions[next_index]
        if next_instruction.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return False
        return next_instruction.argval in self.loop_break_offsets

    def skip_orphan_call_suffix(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if cursor >= end_index or instructions[cursor].opname != "CACHE":
            return None
        if not instructions[cursor].is_jump_target:
            return None

        call_index = next_non_metadata_index(instructions, cursor)
        if call_index is None or call_index >= end_index:
            return None
        call = instructions[call_index]

        precall_index = previous_non_metadata_index(instructions, call_index - 1)
        if not self.is_orphan_call_suffix(call, precall_index, instructions):
            return None

        cursor = call_index + 1
        while cursor < end_index and instructions[cursor].opname == "CACHE":
            cursor += 1
        if cursor < end_index and instructions[cursor].opname == "POP_TOP":
            return cursor + 1
        return cursor

    def is_orphan_call_suffix(
        self,
        call: Instruction,
        precall_index: int | None,
        instructions: list[Instruction],
    ) -> bool:
        return (
            call.opname in {"CALL", "CALL_FUNCTION"}
            and precall_index is not None
            and instructions[precall_index].opname == "PRECALL"
            and len(self.stack) < int(call.arg or 0) + 1
        )

    def add_global_name(self, name: str) -> None:
        if name not in self.global_names:
            self.global_names.append(name)

    def add_global_names(self, names: list[str]) -> None:
        for name in names:
            self.add_global_name(name)

    def translate_range(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> list[ast.stmt]:
        saved_statements = self.statements
        self.statements = []
        offset_to_index = make_offset_index(instructions)

        cursor = start_index
        while cursor < end_index:
            self.spend_read_step(len(instructions))
            if cursor in self.claimed_ranges:
                if cursor in self.handler_exit_indexes:
                    self.append_handler_exit_marker(instructions, cursor)
                cursor = self.claimed_ranges[cursor]
                continue

            skipped_cursor = self.skip_unstructured_instruction(
                instructions,
                cursor,
                start_index,
                end_index,
            )
            if skipped_cursor is not None:
                cursor = skipped_cursor
                continue

            if self.is_module and (
                is_after_implicit_module_return(
                    instructions, cursor, start_index, self.version
                )
                or is_trailing_inline_cleanup(instructions, cursor, start_index)
            ):
                break

            advanced = self.try_translate_structured_range(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
            if advanced is None:
                self.run_instruction(instructions[cursor])
                self.mark_shared_none_return(instructions, cursor)
                cursor += 1
            else:
                cursor = advanced

        self.flush_yield_expressions()
        self.flush_print_items(newline=False)
        result = self.statements
        self.statements = saved_statements
        return result

    def append_handler_exit_marker(
        self,
        instructions: list[Instruction],
        cursor: int,
    ) -> None:
        replaced = self.translate_child_statements(
            instructions, cursor + 1, self.claimed_ranges[cursor]
        )
        self.statements.append(make_handler_exit_marker(replaced))

    def mark_shared_none_return(
        self,
        instructions: list[Instruction],
        cursor: int,
    ) -> None:
        start_index = none_return_start(instructions, cursor)
        if start_index is None or not instructions[start_index].is_jump_target:
            return
        if self.version is not None and self.version < (3, 11):
            return
        previous_index = previous_non_metadata_index(instructions, start_index - 1)
        if previous_index is None:
            return
        previous = instructions[previous_index].opname
        if is_terminal_op(previous) or is_jump_op(previous):
            return
        if self.statements and isinstance(self.statements[-1], ast.Return):
            setattr(self.statements[-1], "_pyc2py_written_return", True)

    def try_translate_compare_chain(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
    ) -> int | None:
        link_end = self.read_compare_chain_link(instructions, offset_to_index, cursor)
        if link_end is None:
            link_end = self.read_compare_chain_or_pop_link(
                instructions, offset_to_index, cursor
            )
        if link_end is None and self.evaluating_test_operand:
            link_end = self.read_compare_chain_test_link(
                instructions, offset_to_index, cursor
            )
        if link_end is not None:
            return link_end
        return self.skip_compare_chain_cleanup(instructions, offset_to_index, cursor)

    def read_compare_chain_link(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
    ) -> int | None:
        if len(self.stack) < 2 or not is_copy_top(instructions[cursor]):
            return None
        compare, middle = self.stack[-1], self.stack[-2]
        if (
            not isinstance(compare, ast.Compare)
            or compare.comparators[-1] is not middle
        ):
            return None
        middle = compare.comparators[-1]
        jump_index = next_non_metadata_index(instructions, cursor + 1)
        if jump_index is not None and instructions[jump_index].opname == "TO_BOOL":
            jump_index = next_non_metadata_index(instructions, jump_index + 1)
        if jump_index is None or instructions[jump_index].opname != "POP_JUMP_IF_FALSE":
            return None
        pop_index = next_non_metadata_index(instructions, jump_index + 1)
        cleanup_index = offset_to_index.get(instructions[jump_index].argval)
        if pop_index is None or instructions[pop_index].opname != "POP_TOP":
            return None
        if cleanup_index is None or instructions[cleanup_index].opname != "SWAP":
            return None

        self.stack.pop()
        self.compare_chain_links[id(middle)] = (middle, compare)
        self.compare_chain_cleanups.add(instructions[cleanup_index].offset)
        return pop_index + 1

    def read_compare_chain_or_pop_link(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
    ) -> int | None:
        jump = instructions[cursor]
        if len(self.stack) < 2 or jump.opname != "JUMP_IF_FALSE_OR_POP":
            return None
        compare, middle = self.stack[-1], self.stack[-2]
        if (
            not isinstance(compare, ast.Compare)
            or compare.comparators[-1] is not middle
        ):
            return None
        middle = compare.comparators[-1]
        cleanup_index = offset_to_index.get(jump.argval)
        if cleanup_index is None or instructions[cleanup_index].opname != "SWAP":
            return None
        self.stack.pop()
        self.compare_chain_links[id(middle)] = (middle, compare)
        self.compare_chain_cleanups.add(instructions[cleanup_index].offset)
        return cursor + 1

    def read_compare_chain_test_link(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
    ) -> int | None:
        jump = instructions[cursor]
        if len(self.stack) < 2 or jump.opname != "POP_JUMP_IF_FALSE":
            return None
        compare, middle = self.stack[-1], self.stack[-2]
        if (
            not isinstance(compare, ast.Compare)
            or compare.comparators[-1] is not middle
        ):
            return None
        middle = compare.comparators[-1]
        cleanup_index = offset_to_index.get(jump.argval)
        if cleanup_index is None or instructions[cleanup_index].opname != "POP_TOP":
            return None
        self.stack.pop()
        self.compare_chain_links[id(middle)] = (middle, compare)
        return cursor + 1

    def skip_compare_chain_cleanup(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
    ) -> int | None:
        instruction = instructions[cursor]
        if instruction.offset in self.compare_chain_cleanups:
            return terminal_tail_end_index(instructions, cursor, len(instructions))
        if instruction.opname != "JUMP_FORWARD" or cursor + 1 >= len(instructions):
            return None
        if instructions[cursor + 1].offset not in self.compare_chain_cleanups:
            return None
        return offset_to_index.get(instruction.argval)

    def join_compare_chain(self, compare: ast.Compare) -> ast.Compare:
        link = self.compare_chain_links.pop(id(compare.left), None)
        if link is None or link[0] is not compare.left:
            return compare
        chain = link[1]
        return ast.Compare(
            left=chain.left,
            ops=[*chain.ops, *compare.ops],
            comparators=[*chain.comparators, *compare.comparators],
        )

    def skip_unstructured_instruction(
        self,
        instructions: list[Instruction],
        cursor: int,
        start_index: int,
        end_index: int,
    ) -> int | None:
        orphan_call_suffix = self.skip_orphan_call_suffix(
            instructions,
            cursor,
            end_index,
        )
        if orphan_call_suffix is not None:
            return orphan_call_suffix

        skipped_with_cleanup = should_skip_returning_with_cleanup_after_terminal(
            instructions,
            cursor,
            start_index,
            end_index,
        )
        if skipped_with_cleanup is not None:
            return skipped_with_cleanup
        if should_skip_range_cursor(self, instructions, cursor, start_index):
            return cursor + 1
        return None

    def try_translate_structured_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        compare_chain = self.try_translate_compare_chain(
            instructions,
            offset_to_index,
            cursor,
        )
        if compare_chain is not None:
            return compare_chain
        if instruction.opname in OR_POP_AFTER_TEST:
            mixed = self.try_translate_mixed_short_circuit(
                instructions, offset_to_index, cursor, end_index
            )
            if mixed is not None:
                return mixed

        retry_loop = self.try_translate_backward_retry_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if retry_loop is not None:
            return retry_loop

        advanced = self.try_translate_exception_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        advanced = self.try_translate_statement_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        return self.try_translate_loop_or_branch(
            instruction,
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_exception_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        opname = normalized_opcode_name(instruction.opname)
        is_try_line = opname == "NOP" and not instruction.is_jump_target
        if is_try_line and cursor + 1 < end_index:
            advanced = self.try_translate_exception_range(
                instructions,
                offset_to_index,
                cursor + 1,
                end_index,
            )
            if advanced is not None:
                return advanced

        advanced = self.try_translate_exception_table_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        if opname == "SETUP_FINALLY":
            return try_translate_finally(
                self,
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname == "SETUP_EXCEPT":
            return try_translate_except(
                self,
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        return None

    def try_translate_exception_table_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        for handler in (
            try_translate_out_of_line_try,
            try_translate_exception_table_finally,
            try_translate_exception_table_except_finally,
            try_translate_exception_table_except,
            try_translate_exception_table_except_star,
        ):
            advanced = handler(self, instructions, offset_to_index, cursor, end_index)
            if advanced is not None:
                return advanced
        return None

    def try_translate_statement_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        opname = normalized_opcode_name(instruction.opname)
        inlined_break = self.try_translate_inlined_break(
            instructions,
            cursor,
            end_index,
        )
        if inlined_break is not None:
            return inlined_break
        handler_break = self.try_translate_break_out_of_handler(
            instructions, cursor, end_index
        )
        if handler_break is not None:
            return handler_break
        short_circuit = self.try_translate_short_circuit_expression(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if short_circuit is not None:
            return short_circuit
        cleanup_return = self.try_translate_loop_cleanup_return(
            instructions,
            cursor,
            end_index,
        )
        if cleanup_return is not None:
            return cleanup_return
        if opname in {"BEFORE_ASYNC_WITH", "BEFORE_WITH"}:
            return self.try_translate_with(instructions, cursor, end_index)
        if opname in {"GET_AWAITABLE", "GET_YIELD_FROM_ITER"}:
            return self.try_translate_send_value(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname in {
            "COPY",
            "LOAD_DEREF",
            "LOAD_FAST",
            "LOAD_FAST_BORROW",
            "LOAD_FAST_CHECK",
            "LOAD_GLOBAL",
            "LOAD_NAME",
            "MATCH_MAPPING",
            "MATCH_SEQUENCE",
        }:
            return self.try_translate_match_or_with_statement(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname == "SWAP":
            self.prepare_simultaneous_store_group(instructions, cursor, end_index)
        if opname in LOCAL_STORE_RUN_OPS:
            self.prepare_local_store_run(instructions, cursor, end_index)
        return None

    def try_translate_match_or_with_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        with_region = self.try_translate_with(instructions, cursor, end_index)
        if with_region is not None:
            return with_region
        return self.try_translate_simple_match(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_short_circuit_expression(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        if instruction.opname not in {"JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"}:
            return None
        if not self.stack:
            return None
        if not isinstance(instruction.argval, int):
            return None

        target_index = offset_to_index.get(instruction.argval)
        if (
            target_index is None
            or target_index <= cursor + 1
            or target_index > end_index
        ):
            return None

        child = self.make_child()
        child.pop_or_none()
        child.translate_range(instructions, cursor + 1, target_index)
        if not child.stack:
            return None

        left = self.expression_from_stack_value(self.pop_or_none())
        right = child.expression_from_stack_value(child.stack[-1])
        self.warnings.extend(child.warnings)
        self.stack.append(
            ast.BoolOp(
                op=(
                    ast.Or()
                    if instruction.opname == "JUMP_IF_TRUE_OR_POP"
                    else ast.And()
                ),
                values=[left, right],
            )
        )
        return target_index

    def try_translate_mixed_short_circuit(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        jump = instructions[cursor]
        other_index = offset_to_index.get(jump.argval)
        if not self.stack or other_index is None or other_index >= end_index:
            return None
        last_index = previous_non_metadata_index(instructions, other_index - 1)
        if last_index is None or last_index <= cursor:
            return None
        last = instructions[last_index]
        if last.opname != OR_POP_AFTER_TEST[jump.opname]:
            return None
        join_index = offset_to_index.get(last.argval)
        if join_index is None or not other_index < join_index <= end_index:
            return None

        group = self.evaluate_short_circuit_group(instructions, cursor, last_index)
        other = self.evaluate_value_behind_test(instructions, other_index, join_index)
        if group is None or other is None:
            return None
        group.insert(0, self.expression_from_stack_value(self.pop_or_none()))
        inner: ast.boolop = ast.And() if is_false_jump(jump.opname) else ast.Or()
        outer: ast.boolop = ast.Or() if is_false_jump(jump.opname) else ast.And()
        self.stack.append(
            ast.BoolOp(op=outer, values=[ast.BoolOp(op=inner, values=group), other])
        )
        return join_index

    def evaluate_short_circuit_group(
        self,
        instructions: list[Instruction],
        test_index: int,
        last_index: int,
    ) -> list[ast.expr] | None:
        test = instructions[test_index]
        operands: list[ast.expr] = []
        start_index = test_index + 1
        for index in range(start_index, last_index + 1):
            instruction = instructions[index]
            same_test = (
                instruction.opname == test.opname and instruction.argval == test.argval
            )
            if not same_test and index != last_index:
                continue
            operand = self.evaluate_short_circuit_operand(
                instructions, start_index, index, instructions[last_index]
            )
            if operand is None:
                return None
            operands.append(operand)
            start_index = index + 1
        return operands

    def evaluate_short_circuit_operand(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        last: Instruction,
    ) -> ast.expr | None:
        values: list[ast.expr] = []
        for index in range(start_index, end_index + 1):
            instruction = instructions[index]
            leaves = instruction.opname == last.opname and instruction.argval == (
                last.argval
            )
            if not leaves and index != end_index:
                continue
            value = self.evaluate_value_behind_test(instructions, start_index, index)
            if value is None:
                return None
            values.append(value)
            start_index = index + 1
        if len(values) == 1:
            return values[0]
        operator: ast.boolop = ast.Or() if "TRUE" in last.opname else ast.And()
        return ast.BoolOp(op=operator, values=values)

    def evaluate_value_behind_test(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> ast.expr | None:
        child = self.make_child()
        child.pop_or_none()
        depth = len(child.stack)
        statements = child.translate_range(instructions, start_index, end_index)
        if statements or len(child.stack) != depth + 1:
            return None
        self.warnings.extend(child.warnings)
        return child.expression_from_stack_value(child.stack[-1])

    def prepare_simultaneous_store_group(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> None:
        count = int(instructions[cursor].arg or 0)
        if count < 2:
            return
        if not has_following_stores(instructions, cursor + 1, end_index, count):
            return
        self.simultaneous_store_group += 1
        self.pending_simultaneous_store_count = count
        self.simultaneous_store_reversed = False

    def prepare_local_store_run(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> None:
        if self.pending_simultaneous_store_count > 0:
            return
        count = count_local_store_run(instructions, cursor, end_index)
        if count < 2 or len(self.stack) < count:
            return
        values = self.stack[-count:]
        if not all(isinstance(value, ast.expr) for value in values):
            return
        if len({id(value) for value in values}) != count:
            return
        self.simultaneous_store_group += 1
        self.pending_simultaneous_store_count = count
        self.simultaneous_store_reversed = True

    def try_translate_loop_or_branch(
        self,
        instruction: Instruction,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        opname = normalized_opcode_name(instruction.opname)
        infinite_loop = self.try_translate_modern_infinite_while_loop(
            instructions,
            cursor,
            end_index,
        )
        if infinite_loop is not None:
            return infinite_loop

        loop_handler = self.loop_branch_handler(opname)
        if loop_handler is not None:
            return loop_handler(instructions, offset_to_index, cursor, end_index)
        if is_forward_conditional_jump(instruction):
            return self.try_translate_conditional(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        return None

    def loop_branch_handler(
        self, opname: str
    ) -> Callable[[list[Instruction], dict[int, int], int, int], int | None] | None:
        return {
            "GET_ITER": self.try_translate_iter_loop,
            "GET_AITER": self.try_translate_async_iter_loop,
            "SETUP_LOOP": self.try_translate_while_loop,
            "FOR_LOOP": self.try_translate_legacy_for_loop,
        }.get(opname)

    def try_translate_iter_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        advanced = self.try_translate_inlined_comprehension(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced
        return self.try_translate_for_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_async_iter_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        advanced = self.try_translate_async_inlined_comprehension(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced
        return self.try_translate_async_for_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )


def decompile_native_source(
    code: Any,
    version: tuple[int, ...] | None,
    progress: ProgressCallback | None = None,
) -> NativeResult | None:
    return NativeDecompiler(
        code=code,
        version=version,
        is_module=True,
        progress=progress,
    ).decompile()


def restore_else_before_shared_return(
    body: list[ast.stmt],
    code: Any,
) -> list[ast.stmt]:
    numbers = constant_numbers(code)
    for index in range(len(body) - 2, -1, -1):
        statement = body[index]
        if not isinstance(statement, ast.If) or not is_constant_return(body[-1]):
            continue
        statement = merged_nested_if(statement)
        if not ends_in_same_return(statement, body[-1], numbers):
            continue
        earlier = statement_constant_numbers([*body[:index], statement], numbers)
        between = statement_constant_numbers(body[index + 1 : -1], numbers) - earlier
        returned = statement_constant_numbers([body[-1]], numbers)
        if not between or not returned or min(returned) < max(between):
            continue
        statement.body = statement.body[:-1]
        statement.orelse = body[index + 1 : -1]
        body = [*body[:index], statement, body[-1]]
    return body


def merged_nested_if(statement: ast.If) -> ast.If:
    while len(statement.body) == 1 and not statement.orelse:
        inner = statement.body[0]
        if not isinstance(inner, ast.If) or inner.orelse:
            break
        test = ast.BoolOp(op=ast.And(), values=[statement.test, inner.test])
        statement = ast.If(test=test, body=inner.body, orelse=[])
    return statement


def constant_numbers(code: Any) -> dict[tuple[type, Any], int]:
    numbers: dict[tuple[type, Any], int] = {}
    for number, value in enumerate(getattr(code, "co_consts", ())):
        if isinstance(value, (str, bytes, int, float, bool)):
            numbers.setdefault((type(value), value), number)
    return numbers


def is_constant_return(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Return) and isinstance(
        statement.value, ast.Constant
    )


def ends_in_same_return(
    statement: ast.If,
    last: ast.stmt,
    numbers: dict[tuple[type, Any], int],
) -> bool:
    if statement.orelse or len(statement.body) < 2:
        return False
    tail = statement.body[-1]
    if not is_constant_return(tail):
        return False
    return statement_constant_numbers([tail], numbers) == statement_constant_numbers(
        [last], numbers
    )


def statement_constant_numbers(
    statements: list[ast.stmt],
    numbers: dict[tuple[type, Any], int],
) -> set[int]:
    found: set[int] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Constant):
                continue
            number = numbers.get((type(node.value), node.value))
            if number is not None:
                found.add(number)
    return found


CO_GENERATOR = 0x20


def add_dead_yield(body: list[ast.stmt], code: Any) -> list[ast.stmt]:
    flags = int(getattr(code, "co_flags", 0) or 0)
    if not flags & (CO_GENERATOR | CO_ASYNC_GENERATOR):
        return body
    if str(getattr(code, "co_name", "")).startswith("<"):
        return body
    if any(
        isinstance(node, (ast.Yield, ast.YieldFrom))
        for node in walk_current_scope_nodes(body)
    ):
        return body
    dead = ast.If(
        test=ast.Constant(value=False),
        body=[ast.Expr(value=ast.Yield(value=None))],
        orelse=[],
    )
    return [dead, *body]


def add_missing_local_initializers(
    body: list[ast.stmt],
    instructions: list[Instruction],
    code: Any,
) -> list[ast.stmt]:
    missing = missing_loaded_stored_locals(body, instructions, code)
    if not missing:
        return body

    initializers = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Constant(value=None),
        )
        for name in sorted(missing)
    ]
    return [*initializers, *body]


def missing_loaded_stored_locals(
    body: list[ast.stmt],
    instructions: list[Instruction],
    code: Any,
) -> set[str]:
    local_names = {
        str(name)
        for name in getattr(code, "co_varnames", ())
        if isinstance(name, str) and name.isidentifier()
    }
    if not local_names:
        return set()

    bytecode_stores = {
        str(instruction.argval)
        for instruction in instructions
        if instruction.opname in {"STORE_FAST", "STORE_NAME"}
        and isinstance(instruction.argval, str)
    }
    if not bytecode_stores:
        return set()

    loaded: set[str] = set()
    assigned: set[str] = set()
    for node in walk_current_scope_nodes(body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
        else:
            assigned.update(names_bound_by(node))
    missing = (loaded & bytecode_stores & local_names) - assigned
    return missing - parameter_names(code)


def parameter_names(code: Any) -> set[str]:
    count = int(getattr(code, "co_argcount", 0) or 0)
    count += int(getattr(code, "co_kwonlyargcount", 0) or 0)
    flags = int(getattr(code, "co_flags", 0) or 0)
    count += bool(flags & CO_VARARGS) + bool(flags & CO_VARKEYWORDS)
    return {str(name) for name in getattr(code, "co_varnames", ())[:count]}


def names_bound_by(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.alias):
        return (node.asname or node.name.partition(".")[0],)
    if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)):
        return (node.name,) if node.name is not None else ()
    if isinstance(node, ast.MatchMapping) and node.rest is not None:
        return (node.rest,)
    return ()


def walk_current_scope_nodes(body: list[ast.stmt]) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    work: list[ast.AST] = list(reversed(body))
    while work:
        node = work.pop()
        nodes.append(node)
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.FunctionDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            continue
        work.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


CO_FUTURE_UNICODE_LITERALS = 0x20000

STORE_TARGET_PREFIX_OPS = {
    "BINARY_OP",
    "BINARY_SUBSCR",
    "CACHE",
    "LOAD_ATTR",
    "LOAD_CONST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_FAST_LOAD_FAST",
    "LOAD_GLOBAL",
    "LOAD_NAME",
}


LOCAL_STORE_RUN_OPS = {"STORE_FAST": 1, "STORE_FAST_STORE_FAST": 2}
SIMULTANEOUS_STORE_OPS = {
    "STORE_ATTR": 1,
    "STORE_DEREF": 1,
    "STORE_GLOBAL": 1,
    "STORE_NAME": 1,
    "STORE_SLICE": 1,
    "STORE_SUBSCR": 1,
    **LOCAL_STORE_RUN_OPS,
}


def has_following_stores(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    count: int,
) -> bool:
    stores = 0
    for index in range(start_index, end_index):
        opname = instructions[index].opname
        if opname in SIMULTANEOUS_STORE_OPS:
            stores += SIMULTANEOUS_STORE_OPS[opname]
            if stores == count:
                return True
            continue
        if opname in STORE_TARGET_PREFIX_OPS:
            continue
        return False
    return False


def count_local_store_run(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    stores = 0
    for index in range(start_index, end_index):
        opname = instructions[index].opname
        if opname in LOCAL_STORE_RUN_OPS:
            stores += LOCAL_STORE_RUN_OPS[opname]
        elif opname != "CACHE":
            break
    return stores


GLOBAL_CHAIN_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP", "SET_LINENO"}
STORE_CHAIN_OPS = {
    "STORE_ATTR",
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_GLOBAL",
    "STORE_NAME",
    "STORE_SUBSCR",
}


def none_return_start(instructions: list[Instruction], cursor: int) -> int | None:
    instruction = instructions[cursor]
    if instruction.opname == "RETURN_CONST":
        return cursor if instruction.argval is None else None
    if instruction.opname != "RETURN_VALUE" or cursor == 0:
        return None
    load = instructions[cursor - 1]
    if load.opname == "LOAD_CONST" and load.argval is None:
        return cursor - 1
    return None


OR_POP_AFTER_TEST = {
    "POP_JUMP_FORWARD_IF_FALSE": "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_FORWARD_IF_TRUE": "JUMP_IF_FALSE_OR_POP",
    "POP_JUMP_IF_FALSE": "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_IF_TRUE": "JUMP_IF_FALSE_OR_POP",
}


def is_copy_top(instruction: Instruction) -> bool:
    return instruction.opname == "COPY" and instruction.arg == 1


def should_skip_dead_jump_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index:
        return False

    instruction = instructions[cursor]
    if instruction.is_jump_target:
        return False
    if not is_terminal_op(instructions[cursor - 1].opname):
        return False
    return is_jump_op(instruction.opname)


def should_skip_generator_prologue_pop(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor != start_index + 1:
        return False
    if instructions[cursor].opname != "POP_TOP":
        return False
    return instructions[start_index].opname == "RETURN_GENERATOR"


def should_skip_iterator_cleanup_before_return(
    instructions: list[Instruction],
    cursor: int,
    claimed_ranges: dict[int, int],
) -> bool:
    instruction = instructions[cursor]
    if instruction.opname == "SWAP":
        return should_skip_swap_iterator_cleanup(instructions, cursor, claimed_ranges)

    if instruction.opname != "POP_TOP":
        return False

    swap_index = previous_non_metadata_index(instructions, cursor - 1)
    if swap_index is None:
        return False

    swap = instructions[swap_index]
    if swap.opname != "SWAP" or int(swap.arg or 0) != 2:
        return False

    return returns_after_handler_cleanup(instructions, cursor + 1, claimed_ranges)


def returns_after_handler_cleanup(
    instructions: list[Instruction],
    cursor: int,
    claimed_ranges: dict[int, int],
) -> bool:
    return_index = next_non_metadata_index(instructions, cursor)
    if return_index is None:
        return False
    return_index = claimed_ranges.get(return_index, return_index)
    if instructions[return_index].opname == "SWAP":
        if return_index + 2 >= len(instructions):
            return False
        if instructions[return_index + 1].opname != "POP_EXCEPT":
            return False
        return_index += 2
    return instructions[return_index].opname in {"RETURN_VALUE", "RETURN_CONST"}


def should_skip_swap_iterator_cleanup(
    instructions: list[Instruction],
    cursor: int,
    claimed_ranges: dict[int, int],
) -> bool:
    instruction = instructions[cursor]
    if int(instruction.arg or 0) != 2:
        return False

    pop_index = next_non_metadata_index(instructions, cursor + 1)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return False

    return returns_after_handler_cleanup(instructions, pop_index + 1, claimed_ranges)


def should_skip_range_cursor(
    decompiler: NativeDecompiler,
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    return (
        (
            not decompiler.is_module
            and should_skip_unreachable_after_terminal(
                instructions,
                cursor,
                start_index,
            )
        )
        or should_skip_dead_jump_after_terminal(instructions, cursor, start_index)
        or should_skip_generator_prologue_pop(instructions, cursor, start_index)
        or should_skip_iterator_cleanup_before_return(
            instructions, cursor, decompiler.claimed_ranges
        )
        or decompiler.should_skip_loop_break_cleanup_pop(instructions, cursor)
    )


def should_skip_unreachable_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index or instructions[cursor].is_jump_target:
        return False

    for index in range(cursor - 1, start_index - 1, -1):
        instruction = instructions[index]
        if instruction.is_jump_target or instruction.opname == "END_ASYNC_FOR":
            return False
        if is_terminal_op(instruction.opname):
            return True
    return False


def is_after_implicit_module_return(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
    version: tuple[int, ...] | None,
) -> bool:
    if cursor <= start_index:
        return False
    previous_index = previous_non_metadata_index(instructions, cursor - 1)
    if previous_index is None:
        return False
    previous = instructions[previous_index]
    if version is not None and version >= (3, 14) and previous_index > start_index:
        loaded = instructions[previous_index - 1]
        return (
            previous.opname == "RETURN_VALUE"
            and loaded.opname == "LOAD_CONST"
            and loaded.argval is None
            and loaded.arg is not None
        )
    return previous.opname == "RETURN_CONST" and previous.argval is None


def is_trailing_inline_cleanup(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index or instructions[cursor - 1].opname != "RERAISE":
        return False
    return instructions[-1].opname == "RERAISE" and all(
        instruction.opname in INLINE_CLEANUP_OPS
        for instruction in instructions[cursor:]
    )


def should_skip_returning_with_cleanup_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
    end_index: int,
) -> int | None:
    if cursor <= start_index:
        return None
    previous_index = previous_non_metadata_index(instructions, cursor - 1)
    if previous_index is None:
        return None
    if not is_terminal_op(instructions[previous_index].opname):
        return None

    skipped = skip_returning_with_exception_handler(instructions, cursor, end_index)
    if skipped == cursor:
        return None
    return skipped


def previous_non_metadata_index(
    instructions: list[Instruction],
    cursor: int,
) -> int | None:
    while cursor >= 0:
        if instructions[cursor].opname not in {"CACHE", "SET_LINENO", "NOP"}:
            return cursor
        cursor -= 1
    return None


def find_loaded_global_store_sources(instructions: list[Instruction]) -> list[str]:
    names: list[str] = []
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_GLOBAL":
            continue
        next_index = next_non_metadata_index(instructions, index + 1)
        if next_index is None:
            continue
        if starts_global_store_chain(instructions, next_index):
            name = str(instruction.argval)
            if name not in names:
                names.append(name)
    return names


def spell_while_one(
    body: list[ast.stmt],
    instructions: list[Instruction],
    code: Any,
) -> None:
    constants = list(getattr(code, "co_consts", ()) or ())
    one = next(
        (
            index
            for index, value in enumerate(constants)
            if type(value) is int and value == 1
        ),
        None,
    )
    if one is None:
        return
    first_use = {
        index: offset
        for index, offset in first_constant_uses(instructions).items()
        if constants[index] is not None
    }
    used_at = first_use.get(one, instructions[-1].offset + 1)
    loops = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.While) and hasattr(node, "_pyc2py_loop_offset")
    ]
    for loop in sorted(loops, key=lambda node: getattr(node, "_pyc2py_loop_offset")):
        start = getattr(loop, "_pyc2py_loop_offset")
        if start >= used_at:
            return
        earlier = [use for index, use in first_use.items() if index < one]
        later = [use for index, use in first_use.items() if index > one]
        if all(use < start for use in earlier) and any(
            start <= use < used_at for use in later
        ):
            loop.test = ast.Constant(value=1)
            return


def first_constant_uses(instructions: list[Instruction]) -> dict[int, int]:
    uses: dict[int, int] = {}
    for instruction in instructions:
        if instruction.opname in {"LOAD_CONST", "RETURN_CONST"} and (
            instruction.arg is not None
        ):
            uses.setdefault(instruction.arg, instruction.offset)
    return uses


def find_line_break_offsets(
    instructions: list[Instruction],
    version: tuple[int, ...] | None,
) -> dict[int, tuple[str, ...]]:
    # where the source ended a line in the middle of a statement, as far as
    # the bytecode of python 3.13 shows it: per instruction, the kind of break
    # before each value it pushes
    if version is None or version < (3, 13):
        return {}
    breaks: dict[int, tuple[str, ...]] = {}
    for index, instruction in enumerate(instructions[1:], start=1):
        if instruction.opname == "LOAD_FAST":
            kind = local_load_line_break(instructions[index - 1], instruction)
            if kind:
                breaks[instruction.offset] = (kind,)
        elif instruction.opname == "LOAD_CONST":
            kinds = default_line_breaks(instructions, index)
            if any(kinds):
                breaks[instruction.offset] = kinds
    return breaks


def local_load_line_break(first: Instruction, second: Instruction) -> str:
    if second.is_jump_target or (second.arg or 0) >= 16:
        return ""
    if not is_unjoined_local(first):
        return ""
    if first.opname == "LOAD_FAST":
        return LINE_BREAK_ALWAYS
    return LINE_BREAK_IN_COMPREHENSION


def default_line_breaks(
    instructions: list[Instruction],
    index: int,
) -> tuple[str, ...]:
    values = instructions[index].argval
    if not isinstance(values, tuple) or not feeds_make_function(instructions, index):
        return ()
    first = index
    while first > 0 and instructions[first - 1].opname == "NOP":
        first -= 1
    lines = index - first
    if not 0 < lines <= len(values):
        return ()
    def_line = instructions[index].starts_line or instructions[index - 1].starts_line
    starts_below = instructions[first].starts_line not in {None, def_line}
    kinds = [LINE_BREAK_ALWAYS if starts_below else ""]
    kinds += [LINE_BREAK_ALWAYS] * (lines - 1)
    return tuple(kinds)


FUNCTION_PART_OPS = frozenset(
    {
        "BUILD_CONST_KEY_MAP",
        "BUILD_MAP",
        "BUILD_TUPLE",
        "LOAD_ATTR",
        "LOAD_CONST",
        "LOAD_FAST",
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "NOP",
    }
)


def feeds_make_function(instructions: list[Instruction], index: int) -> bool:
    for instruction in instructions[index + 1 :]:
        if instruction.opname == "MAKE_FUNCTION":
            return True
        if instruction.opname not in FUNCTION_PART_OPS:
            return False
    return False


def is_unjoined_local(instruction: Instruction) -> bool:
    if instruction.opname not in {"LOAD_FAST", "STORE_FAST"}:
        return False
    return instruction.arg is not None and instruction.arg < 16


def has_return_const(version: tuple[int, ...] | None) -> bool:
    # only python 3.12 and 3.13 return a constant with one instruction
    return version is None or (3, 12) <= version[:2] <= (3, 13)


def find_implicit_return_offsets(
    instructions: list[Instruction],
    return_const: bool = True,
) -> frozenset[int]:
    line: int | None = None
    shared_lines: set[int | None] = set()
    returns: list[tuple[int, int | None]] = []
    for index, instruction in enumerate(instructions):
        if instruction.starts_line is not None:
            line = instruction.starts_line
        if instruction.opname == "RETURN_CONST" and instruction.argval is None:
            returns.append((instruction.offset, line))
        elif not return_const and is_none_return_load(instructions, index):
            returns.append((instructions[index + 1].offset, line))
        elif instruction.opname == "PUSH_EXC_INFO":
            continue
        elif instruction.opname not in IGNORED_BEHAVIOR_OPNAMES and not (
            not return_const
            and index > 0
            and is_none_return_load(instructions, index - 1)
        ):
            shared_lines.add(line)
    return frozenset(offset for offset, line in returns if line in shared_lines)


def is_none_return_load(instructions: list[Instruction], index: int) -> bool:
    if index + 1 >= len(instructions):
        return False
    load = instructions[index]
    return (
        load.opname == "LOAD_CONST"
        and load.argval is None
        and instructions[index + 1].opname == "RETURN_VALUE"
    )


def find_written_return_offsets(instructions: list[Instruction]) -> frozenset[int]:
    return frozenset(
        instructions[index + 1].offset
        for index, instruction in enumerate(instructions)
        if instruction.is_jump_target and is_none_return_load(instructions, index)
    )


def find_explicit_nonlocal_names(
    code: Any,
    instructions: list[Instruction],
) -> list[str]:
    freevars = set(getattr(code, "co_freevars", ()) or ())
    if not freevars:
        return []

    names: list[str] = []
    for instruction in instructions:
        if instruction.opname not in {"DELETE_DEREF", "STORE_DEREF"}:
            continue
        name = str(instruction.argval)
        if name in freevars and name not in names:
            names.append(name)
    return names + unused_free_names(code, instructions)


def unused_free_names(code: Any, instructions: list[Instruction]) -> list[str]:
    used: set[object] = set()
    for instruction in instructions:
        argval = instruction.argval
        used.update(argval if isinstance(argval, tuple) else (argval,))
    return [
        str(name) for name in getattr(code, "co_freevars", ()) or () if name not in used
    ]


def starts_global_store_chain(
    instructions: list[Instruction],
    start_index: int,
) -> bool:
    for index in range(start_index, len(instructions)):
        opname = instructions[index].opname
        if opname in GLOBAL_CHAIN_PREFIX_OPS:
            continue
        if opname == "DUP_TOP":
            continue
        return opname == "STORE_GLOBAL"
    return False


def next_non_metadata_index(
    instructions: list[Instruction],
    start_index: int,
) -> int | None:
    for index in range(start_index, len(instructions)):
        if instructions[index].opname not in GLOBAL_CHAIN_PREFIX_OPS:
            return index
    return None


def find_undeclared_module_globals(
    body: list[ast.stmt],
    instructions: list[Instruction],
) -> list[str]:
    declared = {
        name
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Global)
        for name in node.names
    }
    names: list[str] = []
    for instruction in instructions:
        name = str(instruction.argval)
        if instruction.opname != "STORE_GLOBAL" or name in declared:
            continue
        if name not in names:
            names.append(name)
    return names


def find_module_explicit_global_names(instructions: list[Instruction]) -> list[str]:
    names: list[str] = []
    chain: list[Instruction] = []
    for instruction in instructions:
        if instruction.opname in GLOBAL_CHAIN_PREFIX_OPS:
            continue
        if instruction.opname == "DUP_TOP":
            continue
        if instruction.opname in STORE_CHAIN_OPS:
            chain.append(instruction)
            continue
        add_mixed_store_global_names(names, chain)
        chain = []
    add_mixed_store_global_names(names, chain)
    return names


def add_mixed_store_global_names(
    names: list[str],
    chain: list[Instruction],
) -> None:
    if not chain:
        return
    has_name_store = any(instruction.opname == "STORE_NAME" for instruction in chain)
    if not has_name_store:
        return
    for instruction in chain:
        if instruction.opname != "STORE_GLOBAL":
            continue
        name = str(instruction.argval)
        if name not in names:
            names.append(name)
