import ast
from abc import ABC
from dataclasses import dataclass

from pyc2py.astree import make_name, mark_line
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import (
    line_in_effect,
    make_offset_index,
    skip_ignorable_instructions,
)
from pyc2py.decompiler.async_structures import (
    AsyncForLoopPattern,
    find_async_for_loop_pattern,
    make_async_for_statement,
)
from pyc2py.decompiler.control_comprehension import (
    INLINE_CLEANUP_OPS,
    ControlComprehensionRecoveryMixin,
)
from pyc2py.decompiler.exception_structures import (
    handler_break_return_end,
    handler_break_target,
    out_of_line_handlers_end,
    out_of_line_handlers_start,
    read_table_spans,
    same_instruction_behavior,
    while_handler_break_target,
)
from pyc2py.decompiler.opcodes.flow import is_jump_op, is_terminal_op
from pyc2py.decompiler.opcodes.stack_names import STORE_OPS
from pyc2py.decompiler.opcodes.values import is_zero_constant
from pyc2py.decompiler.runtime import coerce_expr, unpack_counts
from pyc2py.decompiler.structures import (
    ForLoopPattern,
    find_detached_for_tail,
    find_for_loop_pattern,
    find_legacy_for_loop_pattern,
    invert_condition,
    is_false_jump,
    is_forward_conditional_jump,
    loop_entry_offsets,
    skip_loop_cleanup,
)


@dataclass(frozen=True, slots=True)
class LoopBreakExit:
    else_end_index: int | None = None
    inlined_tails: tuple[tuple[int, int], ...] = ()
    tail_is_detached: bool = False


class ControlLoopRecoveryMixin(ControlComprehensionRecoveryMixin, ABC):
    def try_translate_modern_infinite_while_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if (
            self.version is not None
            and self.version >= (3, 14)
            and instructions[cursor].opname != "NOP"
        ):
            return None
        loop_end_index = endless_loop_end(self.code, instructions, cursor, end_index)
        back_jump_index = previous_non_metadata_index(instructions, loop_end_index - 1)
        if back_jump_index is None or back_jump_index <= cursor:
            return None

        back_jump = instructions[back_jump_index]
        inner_loop_end = self.body_end_behind_inner_loop(
            instructions, cursor, back_jump_index
        )
        if inner_loop_end is not None:
            back_jump_index = inner_loop_end
            if inner_loop_end < loop_end_index:
                end_index = inner_loop_end
        elif (
            back_jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_BACKWARD"}
            or back_jump.argval != instructions[cursor].offset
        ):
            if continued_only_from_handlers(
                self.code, instructions, cursor, loop_end_index, end_index
            ):
                body = self.translate_loop_child_statements(
                    instructions,
                    cursor,
                    loop_end_index,
                    continue_offset=int(instructions[cursor].offset),
                    break_index=end_index,
                )
                self.statements.append(
                    make_endless_while(body, instructions[cursor].offset)
                )
                return end_index
            return self.try_translate_infinite_while_before_code(
                instructions,
                cursor,
                loop_end_index,
            )

        break_tails = find_inlined_break_tails(
            instructions,
            cursor,
            back_jump_index,
            "NOP",
        )
        continue_index = cursor
        if cursor > 0 and instructions[cursor - 1].opname == "NOP":
            continue_index = cursor - 1
        body = self.translate_loop_child_statements(
            instructions,
            self.infinite_body_start(instructions, cursor),
            back_jump_index,
            continue_offset=int(instructions[continue_index].offset),
            break_index=end_index,
            inlined_break_tails=break_tails,
        )
        if continue_index != cursor and body and isinstance(body[-1], ast.Continue):
            body.pop()
        self.statements.append(
            make_endless_while(body or [ast.Pass()], instructions[cursor].offset)
        )
        self.append_detached_break_tail(
            instructions,
            LoopBreakExit(
                inlined_tails=break_tails, tail_is_detached=bool(break_tails)
            ),
        )
        return end_index

    def leaves_to_enclosing_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        back_jump_index: int,
    ) -> bool:
        if self.version is None or self.version < (3, 14):
            return False
        return any(
            "JUMP" in instruction.opname
            and instruction.argval in self.loop_continue_offsets
            for instruction in instructions[cursor:back_jump_index]
        )

    def body_end_behind_inner_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        last_index: int,
    ) -> int | None:
        # python 3.14: "while True" whose last statement is another loop. the
        # last jump back goes to that loop, and one further up (where the
        # inner loop is left) goes back to this one. behind the last jump
        # there can be the end of the inner body, when that returns
        if self.version is None or self.version < (3, 14):
            return None
        start = instructions[cursor].offset
        inner_jump = next(
            (
                index
                for index in range(last_index, cursor, -1)
                if instructions[index].opname == "JUMP_BACKWARD"
            ),
            None,
        )
        if inner_jump is None or instructions[inner_jump].argval <= start:
            return None
        if not any(
            instruction.opname == "JUMP_BACKWARD"
            and instruction.argval == start
            and instruction.offset > instructions[inner_jump].argval
            for instruction in instructions[cursor:inner_jump]
        ):
            return None
        behind = {
            item.offset: index
            for index, item in enumerate(instructions[: last_index + 1])
            if index > inner_jump
        }
        inner_start = instructions[inner_jump].argval
        body_end = min(
            (
                behind[instruction.argval]
                for instruction in instructions[cursor:inner_jump]
                if "JUMP" in instruction.opname
                and instruction.offset < inner_start
                and instruction.argval in behind
            ),
            default=last_index + 1,
        )
        if body_end > inner_jump + 1 and not is_terminal_op(
            instructions[body_end - 1].opname
        ):
            return None
        return body_end

    def infinite_body_start(self, instructions: list[Instruction], cursor: int) -> int:
        if (
            self.version is not None
            and self.version >= (3, 14)
            and instructions[cursor].opname == "NOP"
        ):
            return cursor + 1
        return cursor

    def try_translate_infinite_while_before_code(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        back_jump_index = find_closing_back_jump(instructions, cursor, end_index)
        if back_jump_index is None:
            return None
        if starts_with_leaving_test(instructions, cursor, back_jump_index):
            return None
        tail_end_index = tail_behind_written_continue(
            instructions, cursor, back_jump_index, end_index
        )
        if tail_end_index is not None:
            body = self.translate_loop_child_statements(
                instructions,
                cursor + 1,
                tail_end_index,
                continue_offset=int(instructions[cursor].offset),
                break_index=end_index,
            )
            self.statements.append(
                make_endless_while(body, instructions[cursor].offset)
            )
            return tail_end_index
        exit_index = out_of_line_handlers_end(
            self.code, instructions, back_jump_index + 1
        )
        if self.leaves_to_enclosing_loop(instructions, cursor, back_jump_index):
            exit_index = back_jump_index = end_index
        elif not loop_exit_is_break_target(
            instructions, cursor, back_jump_index, exit_index, end_index
        ):
            return None
        continue_index = cursor
        if cursor > 0 and instructions[cursor - 1].opname == "NOP":
            continue_index = cursor - 1
        body = self.translate_loop_child_statements(
            instructions,
            self.infinite_body_start(instructions, cursor),
            back_jump_index,
            continue_offset=int(instructions[continue_index].offset),
            break_index=exit_index,
        )
        if continue_index != cursor and body and isinstance(body[-1], ast.Continue):
            body.pop()
        self.statements.append(
            make_endless_while(body or [ast.Pass()], instructions[cursor].offset)
        )
        return exit_index

    def try_translate_while_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        if instruction.opname != "SETUP_LOOP":
            return None

        wrapped_for = self.try_translate_setup_loop_for(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if wrapped_for is not None:
            return wrapped_for

        loop_bounds = self.find_while_loop_bounds(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if loop_bounds is None:
            return None

        loop_start_index, condition_jump_index, false_index, back_jump_index = (
            loop_bounds
        )
        condition = self.evaluate_expression_range(
            instructions,
            loop_start_index,
            condition_jump_index,
        )
        if condition is None:
            return None

        condition_jump = instructions[condition_jump_index]
        test = (
            condition
            if is_false_jump(condition_jump.opname)
            else invert_condition(condition)
        )
        body = self.translate_child_statements(
            instructions,
            condition_jump_index + 1,
            back_jump_index,
        )
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=[])
        )
        return skip_loop_cleanup(instructions, false_index, end_index)

    def find_while_loop_bounds(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> tuple[int, int, int, int] | None:
        loop_start_index = cursor + 1
        if loop_start_index >= end_index:
            return None

        condition_jump_index = self.find_while_condition_jump(
            instructions,
            loop_start_index,
            end_index,
        )
        if condition_jump_index is None:
            return None

        condition_jump = instructions[condition_jump_index]
        false_index = offset_to_index.get(int(condition_jump.argval))
        if false_index is None or false_index <= condition_jump_index:
            return None

        back_jump_index = self.find_while_back_jump(
            instructions,
            condition_jump_index + 1,
            false_index,
            int(instructions[loop_start_index].offset),
        )
        if back_jump_index is None:
            return None

        return loop_start_index, condition_jump_index, false_index, back_jump_index

    def try_translate_setup_loop_for(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        setup_end_index = offset_to_index.get(int(instructions[cursor].argval))
        if (
            setup_end_index is None
            or setup_end_index <= cursor
            or setup_end_index > end_index
        ):
            return None

        get_iter_index = find_setup_loop_get_iter(
            instructions, cursor + 1, setup_end_index
        )
        if get_iter_index is None:
            return None

        child = self.make_child()
        iterable = self.evaluate_expression_range(
            instructions, cursor + 1, get_iter_index
        )
        child.stack.append(iterable)
        advanced = child.try_translate_for_loop(
            instructions,
            offset_to_index,
            get_iter_index,
            setup_end_index,
        )
        if advanced is None:
            return None
        if skip_loop_cleanup(instructions, advanced, end_index) != setup_end_index:
            return None

        self.stack.clear()
        self.statements.extend(child.statements)
        self.warnings.extend(child.warnings)
        return skip_loop_cleanup(instructions, setup_end_index, end_index)

    def find_while_condition_jump(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> int | None:
        for index in range(start_index, end_index):
            instruction = instructions[index]
            if instruction.opname in {"FOR_ITER", "FOR_LOOP"}:
                return None
            if is_forward_conditional_jump(instruction):
                return index
        return None

    def find_while_back_jump(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        loop_start_offset: int,
    ) -> int | None:
        for index in range(end_index - 1, start_index - 1, -1):
            instruction = instructions[index]
            if instruction.opname not in {"JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP"}:
                continue
            if instruction.argval == loop_start_offset:
                return index
        return None

    def try_translate_for_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        pattern = find_for_loop_pattern(
            instructions, offset_to_index, cursor, end_index
        )
        if pattern is None:
            return None

        target, body_start = self.read_for_loop_target(
            instructions,
            pattern.for_iter_index + 1,
            pattern.body_end_index,
        )
        if target is None:
            return None

        iterable = coerce_expr(self.pop_or_none())
        advanced = self.try_translate_legacy_list_comprehension(
            instructions,
            offset_to_index,
            cursor,
            end_index,
            target,
            body_start,
            iterable,
        )
        if advanced is not None:
            return advanced

        break_exit = find_loop_break_exit(
            instructions,
            body_start,
            pattern.body_end_index,
            pattern.after_index,
            end_index,
            "POP_TOP",
        )
        else_end_index = break_exit.else_end_index
        body: list[ast.stmt] = []
        outer_resume_offsets = self.detached_resume_offsets
        body_ranges = self.for_body_ranges(
            instructions, offset_to_index, pattern, body_start
        )
        for start, end in body_ranges:
            body.extend(
                self.translate_loop_child_statements(
                    instructions,
                    start,
                    end,
                    continue_offset=int(instructions[pattern.for_iter_index].offset),
                    break_index=else_end_index or pattern.after_index,
                    extra_continue_offsets=frozenset(
                        loop_entry_offsets(instructions, cursor, pattern.for_iter_index)
                    ),
                    inlined_break_tails=break_exit.inlined_tails,
                )
            )
        self.detached_resume_offsets = outer_resume_offsets
        if len(body_ranges) > 1:
            self.claimed_ranges[body_ranges[1][0]] = body_ranges[1][1] + 1
        if else_end_index is None:
            orelse, after_index = self.translate_for_orelse(
                instructions,
                offset_to_index,
                cursor,
                pattern.after_index,
                end_index,
            )
        else:
            after_index = else_end_index
            orelse = self.translate_child_statements(
                instructions,
                pattern.after_index,
                else_end_index,
            )
        self.statements.append(
            ast.For(
                target=target,
                iter=iterable,
                body=body or [ast.Pass()],
                orelse=orelse,
                type_comment=None,
            )
        )
        self.append_detached_break_tail(instructions, break_exit)
        return after_index

    def for_body_ranges(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        pattern: ForLoopPattern,
        body_start: int,
    ) -> list[tuple[int, int]]:
        tail = find_detached_for_tail(
            instructions, offset_to_index, pattern.for_iter_index
        )
        if tail is None:
            return [(body_start, pattern.body_end_index)]
        tail_start, back_jump_index = tail
        self.detached_resume_offsets |= {int(instructions[tail_start].offset)}
        return [(body_start, pattern.body_end_index + 1), (tail_start, back_jump_index)]

    def append_detached_break_tail(
        self,
        instructions: list[Instruction],
        break_exit: LoopBreakExit,
    ) -> None:
        if not break_exit.tail_is_detached:
            return
        self.statements.extend(
            self.translate_child_statements(instructions, *break_exit.inlined_tails[0])
        )

    def try_translate_legacy_for_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        pattern = find_legacy_for_loop_pattern(
            instructions, offset_to_index, cursor, end_index
        )
        if pattern is None:
            return None

        target, body_start = self.read_for_loop_target(
            instructions,
            cursor + 1,
            pattern.body_end_index,
        )
        if target is None:
            return None

        iterable = self.pop_legacy_for_iterable()
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            pattern.body_end_index,
            continue_offset=int(instructions[cursor].offset),
            break_index=pattern.after_index,
        )
        orelse, after_index = self.translate_for_orelse(
            instructions,
            offset_to_index,
            cursor,
            pattern.after_index,
            end_index,
        )
        self.statements.append(
            ast.For(
                target=target,
                iter=iterable,
                body=body or [ast.Pass()],
                orelse=orelse,
                type_comment=None,
            )
        )
        return after_index

    def translate_for_orelse(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        loop_index: int,
        after_loop_index: int,
        end_index: int,
    ) -> tuple[list[ast.stmt], int]:
        orelse_end_index = self.find_wrapping_setup_loop_end(
            instructions,
            offset_to_index,
            loop_index,
            after_loop_index,
            end_index,
        )
        if orelse_end_index is None:
            return [], after_loop_index

        orelse = self.translate_child_statements(
            instructions,
            after_loop_index,
            orelse_end_index,
        )
        return orelse, orelse_end_index

    def translate_loop_child_statements(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        continue_offset: int,
        break_index: int,
        extra_continue_offsets: frozenset[int] = frozenset(),
        inlined_break_tails: tuple[tuple[int, int], ...] = (),
    ) -> list[ast.stmt]:
        child = self.make_child()
        child.loop_continue_offsets = (
            frozenset({continue_offset}) | extra_continue_offsets
        )
        child.loop_none_return_is_break = loop_exit_is_none_return(
            instructions,
            break_index,
        )
        child.loop_break_offsets = (
            self.loop_break_offsets
            | self.loop_continue_offsets
            | loop_break_offsets_at(instructions, break_index)
            | self.break_line_offsets(instructions, end_index, break_index)
            | frozenset(
                int(instructions[start].offset) for start, _ in inlined_break_tails
            )
        )

        for index in range(start_index, end_index):
            if index in self.handler_exit_indexes:
                self.handler_exit_indexes.discard(index)
                del self.claimed_ranges[index]
        return child.translate_range(instructions, start_index, end_index)

    def break_line_offsets(
        self,
        instructions: list[Instruction],
        body_end_index: int,
        break_index: int,
    ) -> frozenset[int]:
        # python 3.14 keeps the line of a break that ends the loop body as a
        # nop between the jump back and the code behind the loop, and the
        # test in front of the break jumps to that nop
        if self.version is None or self.version < (3, 14):
            return frozenset()
        offsets: set[int] = set()
        index = min(break_index, len(instructions)) - 1
        while index > body_end_index and instructions[index].opname == "NOP":
            offsets.add(int(instructions[index].offset))
            index -= 1
        index = break_index
        while index < len(instructions) and instructions[index].opname == "NOP":
            index += 1
        if break_index < index < len(instructions):
            offsets.add(int(instructions[index].offset))
        return frozenset(offsets)

    def try_translate_inlined_break(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if self.stack or not self.loop_break_offsets:
            return None
        marker = instructions[cursor]
        if marker.opname not in {"NOP", "POP_TOP"}:
            return None
        tail_end = inlined_break_tail_end(
            instructions,
            cursor,
            end_index,
            marker.opname,
        )
        if tail_end is None:
            return None
        if instructions[cursor + 1].offset not in self.loop_break_offsets:
            return None
        self.statements.append(ast.Break())
        return tail_end

    def try_translate_break_out_of_handler(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if self.stack or not self.loop_continue_offsets:
            return None
        copied_return_end = handler_break_return_end(instructions, cursor, end_index)
        if copied_return_end is not None:
            self.statements.append(ast.Break())
            return copied_return_end
        length = 3
        target = handler_break_target(instructions, cursor, end_index)
        if target is None:
            length = 2
            target = while_handler_break_target(instructions, cursor, end_index)
        if target is None:
            return None
        if target in self.loop_continue_offsets | self.detached_resume_offsets:
            return None
        self.statements.append(ast.Break())
        return cursor + length

    def find_wrapping_setup_loop_end(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        loop_index: int,
        after_loop_index: int,
        end_index: int,
    ) -> int | None:
        for index in range(loop_index - 1, -1, -1):
            instruction = instructions[index]
            if instruction.opname != "SETUP_LOOP":
                continue
            setup_end_index = offset_to_index.get(int(instruction.argval))
            if setup_end_index is None:
                return None
            if setup_end_index <= after_loop_index:
                return None
            if setup_end_index > end_index:
                return None
            return setup_end_index
        return None

    def try_translate_async_for_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        pattern = find_async_for_loop_pattern(
            instructions, offset_to_index, cursor, end_index
        )
        if pattern is None:
            return None

        target, body_start = self.read_for_loop_target(
            instructions,
            pattern.target_start_index,
            pattern.body_start_index,
        )
        if target is None:
            return None

        iterable = coerce_expr(self.pop_or_none())
        body = self.translate_async_for_body(instructions, body_start, pattern)
        self.statements.append(
            make_async_for_statement(
                target,
                iterable,
                body,
                empty_body_is_continue=pattern.empty_body_is_continue,
            )
        )
        return pattern.after_index

    def translate_async_for_body(
        self,
        instructions: list[Instruction],
        body_start: int,
        pattern: AsyncForLoopPattern,
    ) -> list[ast.stmt]:
        back_jump = instructions[pattern.body_end_index]
        if back_jump.opname != "JUMP_BACKWARD" or not isinstance(back_jump.argval, int):
            return self.translate_child_statements(
                instructions,
                body_start,
                pattern.body_end_index,
            )
        return self.translate_loop_child_statements(
            instructions,
            body_start,
            pattern.body_end_index,
            continue_offset=back_jump.argval,
            break_index=pattern.after_index,
        )

    def pop_legacy_for_iterable(self) -> ast.expr:
        index_value = self.pop_or_none()
        if not is_zero_constant(index_value):
            self.warnings.append("FOR_LOOP index seed was not zero")
        return coerce_expr(self.pop_or_none())

    def read_for_loop_target(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> tuple[ast.expr | None, int]:
        start_index = skip_ignorable_instructions(instructions, start_index, end_index)
        if start_index >= end_index:
            return None, start_index

        instruction = instructions[start_index]
        if instruction.opname in STORE_OPS:
            name = make_name(str(instruction.argval), ast.Store())
            mark_line(name, line_in_effect(instructions, start_index))
            return name, start_index + 1
        target: ast.expr | None = None
        if instruction.opname in {"UNPACK_EX", "UNPACK_SEQUENCE", "UNPACK_TUPLE"}:
            target, cursor = self.read_unpack_loop_target(
                instructions, start_index, end_index
            )
            if target is not None:
                return target, cursor
        return self.read_stored_loop_target(instructions, start_index, end_index)

    def read_stored_loop_target(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> tuple[ast.expr | None, int]:
        item = make_name("__pyc2py_loop_item__", ast.Load())
        child = self.make_child()
        child.stack.clear()
        child.stack.append(item)
        child.statements = []
        for cursor in range(start_index, end_index):
            if is_jump_op(instructions[cursor].opname):
                break
            child.run_instruction(instructions[cursor])
            if not child.statements:
                continue
            statement = child.statements[0]
            if (
                isinstance(statement, ast.Assign)
                and statement.value is item
                and not child.stack
            ):
                return statement.targets[0], cursor + 1
            break
        return None, start_index

    def read_unpack_loop_target(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> tuple[ast.expr | None, int]:
        count, starred_index = unpack_counts(instructions[start_index])
        target_end = start_index + 1 + count
        if target_end > end_index:
            return None, start_index
        if count == 0:
            return ast.List(elts=[], ctx=ast.Store()), target_end

        targets: list[ast.expr] = []
        cursor = start_index + 1
        for index in range(count):
            target, next_cursor = self.read_for_loop_target(
                instructions, cursor, end_index
            )
            if target is None:
                return None, start_index
            if index == starred_index:
                target = ast.Starred(value=target, ctx=ast.Store())
            targets.append(target)
            cursor = next_cursor
        return ast.Tuple(elts=targets, ctx=ast.Store()), cursor


def break_target_behind_range(
    instructions: list[Instruction],
    body: tuple[int, int],
    exit_index: int,
    end_index: int,
) -> int | None:
    if not exit_index < end_index < len(instructions):
        return None
    end_offset = instructions[end_index].offset
    targets = {
        instruction.argval
        for instruction in instructions[body[0] : body[1]]
        if instruction.opname == "JUMP_FORWARD" and instruction.argval > end_offset
    }
    if len(targets) != 1:
        return None
    target = targets.pop()
    for index in range(end_index + 1, len(instructions)):
        if instructions[index].offset == target:
            return index
    return None


def find_loop_else_end(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    exit_index: int,
    end_index: int,
) -> int | None:
    if exit_index >= end_index:
        return None
    exit_offset = instructions[exit_index].offset
    targets = {
        instruction.argval
        for instruction in instructions[body_start:body_end]
        if instruction.opname in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}
        and isinstance(instruction.argval, int)
        and instruction.argval > exit_offset
    }
    if not targets and breaks_to_enclosing_loop(
        instructions, body_start, body_end, end_index
    ):
        return end_index
    if len(targets) != 1:
        return None
    else_end_offset = targets.pop()
    for index in range(exit_index + 1, min(end_index + 1, len(instructions))):
        if instructions[index].offset == else_end_offset:
            return index
    if else_end_offset in loop_break_offsets_at(instructions, end_index):
        return end_index
    return None


def breaks_to_enclosing_loop(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    end_index: int,
) -> bool:
    if end_index >= len(instructions):
        return False
    closing = instructions[end_index]
    outer_offset = closing.argval
    if closing.opname == "END_FOR":
        outer_offset = next(
            (
                instruction.offset
                for instruction in instructions[:body_start]
                if instruction.opname == "FOR_ITER"
                and instruction.argval == closing.offset
            ),
            None,
        )
    elif closing.opname != "JUMP_BACKWARD":
        return False
    return any(
        instruction.opname == "JUMP_BACKWARD"
        and instruction.argval == outer_offset
        and instructions[index - 1].opname == "POP_TOP"
        for index, instruction in enumerate(
            instructions[body_start:body_end], body_start
        )
    )


STATEMENT_OP_PREFIXES = ("DELETE", "IMPORT", "POP_TOP", "RAISE", "RETURN", "STORE")


def starts_with_leaving_test(
    instructions: list[Instruction],
    cursor: int,
    back_jump_index: int,
) -> bool:
    for index in range(cursor, back_jump_index):
        instruction = instructions[index]
        if instruction.opname.startswith(STATEMENT_OP_PREFIXES):
            return False
        if "JUMP" in instruction.opname:
            return (
                "IF_" in instruction.opname
                and instruction.argval > instructions[back_jump_index + 1].offset
            )
    return False


def make_endless_while(body: list[ast.stmt], offset: int) -> ast.While:
    loop = ast.While(test=ast.Constant(value=True), body=body, orelse=[])
    setattr(loop, "_pyc2py_loop_offset", offset)
    return loop


def tail_behind_written_continue(
    instructions: list[Instruction],
    cursor: int,
    back_jump_index: int,
    end_index: int,
) -> int | None:
    back_jump = instructions[back_jump_index]
    if instructions[cursor].opname != "NOP" or back_jump.starts_line in {
        None,
        instructions[cursor].starts_line,
    }:
        return None
    test = instructions[back_jump_index - 1]
    if not is_forward_conditional_jump(test) or back_jump_index + 1 >= end_index:
        return None
    if test.argval != instructions[back_jump_index + 1].offset:
        return None
    if not is_terminal_op(instructions[end_index - 1].opname):
        return None
    inside = range(
        int(instructions[cursor].offset), int(instructions[end_index - 1].offset) + 1
    )
    if any(
        "JUMP" in instruction.opname and instruction.argval not in inside
        for instruction in instructions[back_jump_index + 1 : end_index]
    ):
        return None
    return end_index


def continued_only_from_handlers(
    code: object,
    instructions: list[Instruction],
    cursor: int,
    loop_end_index: int,
    end_index: int,
) -> bool:
    if instructions[cursor].opname != "NOP" or loop_end_index >= end_index:
        return False
    if instructions[loop_end_index].opname != "PUSH_EXC_INFO":
        return False
    if not is_terminal_op(instructions[loop_end_index - 1].opname):
        return False
    loop_offset = instructions[cursor].offset
    handlers = [
        span.target_index
        for span in read_table_spans(
            code, instructions, make_offset_index(instructions)
        )
        if cursor <= span.start_index < loop_end_index <= span.target_index
    ]
    return any(
        instruction.opname == "JUMP_BACKWARD"
        and instruction.argval == loop_offset
        and any(handler <= index for handler in handlers)
        for index, instruction in enumerate(
            instructions[loop_end_index:end_index], loop_end_index
        )
    )


def endless_loop_end(
    code: object,
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    handlers_start = out_of_line_handlers_start(code, instructions, end_index)
    while (
        instructions[handlers_start - 2].opname == "CLEANUP_THROW"
        and instructions[handlers_start - 1].opname == "JUMP_BACKWARD_NO_INTERRUPT"
    ):
        handlers_start -= 2
    if handlers_start != end_index:
        return handlers_start
    cleanup_start = end_index
    while instructions[cleanup_start - 1].opname in INLINE_CLEANUP_OPS:
        cleanup_start -= 1
    if (
        instructions[end_index - 1].opname == "RERAISE"
        and instructions[cleanup_start - 1].opname == "JUMP_BACKWARD"
    ):
        return cleanup_start
    back_jump_index = find_closing_back_jump(instructions, cursor, end_index)
    if back_jump_index is None or back_jump_index + 1 >= end_index:
        return end_index
    if instructions[back_jump_index + 1].opname != "PUSH_EXC_INFO":
        return end_index
    last = instructions[end_index - 1]
    if last.opname not in {"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"}:
        return end_index
    if (
        instructions[cursor].offset
        <= last.argval
        <= instructions[back_jump_index].offset
    ):
        return back_jump_index + 1
    return end_index


def find_closing_back_jump(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int | None:
    if not instructions[cursor].is_jump_target:
        return None
    start_offset = instructions[cursor].offset
    closing = [
        index
        for index in range(cursor + 1, end_index)
        if instructions[index].opname == "JUMP_BACKWARD"
        and instructions[index].argval == start_offset
    ]
    return closing[-1] if closing else None


def loop_exit_is_break_target(
    instructions: list[Instruction],
    cursor: int,
    back_jump_index: int,
    exit_index: int,
    end_index: int,
) -> bool:
    if exit_index >= end_index or not instructions[exit_index].is_jump_target:
        return False
    start_offset = instructions[cursor].offset
    end_offset = instructions[back_jump_index].offset
    return not any(
        instructions[index].opname == "JUMP_BACKWARD"
        and start_offset < instructions[index].argval <= end_offset
        for index in range(exit_index, end_index)
    )


def find_loop_break_exit(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    exit_index: int,
    end_index: int,
    marker: str,
) -> LoopBreakExit:
    else_end_index = find_loop_else_end(
        instructions,
        body_start,
        body_end,
        exit_index,
        end_index,
    )
    if else_end_index is None:
        else_end_index = handler_break_else_end(
            instructions, body_start, body_end, exit_index, end_index, marker
        )
    if else_end_index is not None:
        return LoopBreakExit(else_end_index)
    outer_index = break_target_behind_range(
        instructions, (body_start, body_end), exit_index, end_index
    )
    if outer_index is not None:
        return LoopBreakExit(end_index, ((outer_index, outer_index),))

    tails = find_inlined_break_tails(instructions, body_start, body_end, marker)
    if not tails:
        return LoopBreakExit()
    start, end = tails[0]
    behind = [end_index] if end_index + end - start <= len(instructions) else []
    for index in [*range(exit_index, end_index - (end - start) + 1), *behind]:
        if not same_instruction_run(instructions, start, index, end - start):
            continue
        if instructions[index - 2].opname == "POP_EXCEPT":
            continue
        if index == exit_index:
            return LoopBreakExit(None, tails)
        return LoopBreakExit(index, tails)
    if exit_index < end_index and is_terminal_op(instructions[end_index - 1].opname):
        return LoopBreakExit(end_index, tails, tail_is_detached=True)
    return LoopBreakExit()


def handler_break_else_end(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    exit_index: int,
    end_index: int,
    marker: str,
) -> int | None:
    if exit_index >= min(end_index, len(instructions) - 1):
        return None
    read_target = (
        handler_break_target if marker == "POP_TOP" else while_handler_break_target
    )
    targets = {
        target
        for index in range(body_start, body_end)
        if (target := read_target(instructions, index, body_end)) is not None
        and not stays_in_loop(instructions, body_start, exit_index, target)
    }
    if not targets and any(
        handler_break_return_end(instructions, index, body_end) is not None
        for index in range(body_start, body_end)
    ):
        return end_index
    if len(targets) != 1:
        return None
    for index in range(exit_index, end_index - 1):
        if (
            instructions[index].opname == "POP_EXCEPT"
            and instructions[index + 1].argval in targets
        ):
            return index
    return end_index


def stays_in_loop(
    instructions: list[Instruction],
    body_start: int,
    exit_index: int,
    target: int,
) -> bool:
    if target >= instructions[exit_index].offset:
        return False
    return not any(
        instruction.opname == "PUSH_EXC_INFO" and target <= instruction.offset
        for instruction in instructions[:body_start]
    )


def find_inlined_break_tails(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    marker: str,
) -> tuple[tuple[int, int], ...]:
    tails: list[tuple[int, int]] = []
    for index in range(body_start, body_end):
        tail_end = inlined_break_tail_end(instructions, index, body_end, marker)
        if tail_end is not None:
            tails.append((index + 1, tail_end))
    if any(
        end - start != tails[0][1] - tails[0][0]
        or not same_instruction_run(instructions, tails[0][0], start, end - start)
        for start, end in tails
    ):
        return ()
    return tuple(tails)


def inlined_break_tail_end(
    instructions: list[Instruction],
    marker_index: int,
    end_index: int,
    marker: str,
) -> int | None:
    start_index = marker_index + 1
    if start_index >= end_index:
        return None
    if instructions[marker_index].opname != marker:
        return None
    if instructions[marker_index].starts_line is None:
        return None
    if instructions[start_index].starts_line is None:
        return None
    for index in range(start_index, end_index):
        opname = instructions[index].opname
        if opname in {"RETURN_CONST", "RETURN_VALUE"}:
            return index + 1
        if "JUMP" in opname or opname in {"FOR_ITER", "POP_TOP"}:
            return None
    return None


def same_instruction_run(
    instructions: list[Instruction],
    left_start: int,
    right_start: int,
    length: int,
) -> bool:
    return all(
        same_instruction_behavior(
            instructions[left_start + step], instructions[right_start + step]
        )
        for step in range(length)
    )


def loop_break_offsets_at(
    instructions: list[Instruction],
    break_index: int,
) -> frozenset[int]:
    if break_index >= len(instructions):
        return frozenset()
    instruction = instructions[break_index]
    offsets = {int(instruction.offset)}
    if instruction.opname in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"} and isinstance(
        instruction.argval, int
    ):
        offsets.add(instruction.argval)
    return frozenset(offsets)


def loop_exit_is_none_return(
    instructions: list[Instruction],
    break_index: int,
) -> bool:
    if break_index >= len(instructions):
        return True
    cursor = skip_ignorable_instructions(instructions, break_index, len(instructions))
    if cursor >= len(instructions):
        return True
    instruction = instructions[cursor]
    if instruction.opname == "RETURN_CONST":
        return instruction.argval is None
    if instruction.opname != "LOAD_CONST" or instruction.argval is not None:
        return False
    cursor = skip_ignorable_instructions(instructions, cursor + 1, len(instructions))
    return cursor < len(instructions) and instructions[cursor].opname == "RETURN_VALUE"


def previous_non_metadata_index(
    instructions: list[Instruction],
    cursor: int,
) -> int | None:
    while cursor >= 0:
        if instructions[cursor].opname not in {"CACHE", "EXTENDED_ARG", "NOP"}:
            return cursor
        cursor -= 1
    return None


def find_setup_loop_get_iter(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname == "GET_ITER":
            return index
        if instruction.opname in {"FOR_ITER", "FOR_LOOP"}:
            return None
    return None
