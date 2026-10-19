import ast
from abc import ABC
from dataclasses import dataclass

from pyc2py.astree import make_name, mark_line
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import (
    line_in_effect,
    make_offset_index,
    opnames_in,
    previous_non_metadata_index,
    skip_ignorable_instructions,
)
from pyc2py.bytecode.opcode_table import (
    BACKWARD_JUMPS,
    LOOP_BACK_JUMPS,
)
from pyc2py.bytecode.stack_effect import STORE_OPS
from pyc2py.decompiler.async_structures import (
    AsyncForLoopPattern,
    find_async_for_loop_pattern,
    make_async_for_statement,
)
from pyc2py.decompiler.control_comprehension import (
    INLINE_CLEANUP_OPS,
    ControlComprehensionRecoveryMixin,
    is_loop_back_jump,
)
from pyc2py.decompiler.exception_structures import (
    enclosing_handler,
    handler_break_return_end,
    handler_break_target,
    handler_regions,
    out_of_line_handlers_end,
    out_of_line_handlers_start,
    read_table_spans,
    same_instruction_run,
    while_handler_break_target,
)
from pyc2py.decompiler.opcodes.flow import is_jump_op, is_terminal_op
from pyc2py.decompiler.runtime import coerce_expr, unpack_counts
from pyc2py.decompiler.structures import (
    ForLoopPattern,
    find_detached_for_tail,
    find_for_loop_pattern,
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
        # under SETUP_LOOP a nested loop has its own SETUP_LOOP
        if (
            "SETUP_LOOP" in opnames_in(instructions)
            and instructions[cursor].offset in self.loop_continue_offsets
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
                tails = tails_below_loop(
                    instructions,
                    cursor,
                    end_index,
                    find_inlined_break_tails(
                        instructions, cursor, loop_end_index, "NOP"
                    ),
                )
                body = self.translate_loop_child_statements(
                    instructions,
                    cursor,
                    loop_end_index,
                    continue_offset=int(instructions[cursor].offset),
                    break_index=end_index,
                    inlined_break_tails=tails,
                )
                self.statements.append(
                    make_endless_while(body, instructions[cursor].offset)
                )
                self.append_detached_break_tail(
                    instructions,
                    LoopBreakExit(inlined_tails=tails, tail_is_detached=bool(tails)),
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
        # 3.14 while True ending in a loop: its last jump back goes to that loop
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
        if resumes_behind_earlier_try(self.code, instructions, cursor, back_jump_index):
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

        loop_end_index = offset_to_index.get(int(instruction.argval))
        read = self.read_block_while(
            instructions, offset_to_index, cursor, loop_end_index or end_index
        ) or self.read_first_test_while(
            instructions, offset_to_index, cursor, end_index
        )
        if read is None:
            if loop_end_index is None or loop_end_index > end_index:
                return None

            return self.try_translate_endless_block_while(
                instructions, cursor + 1, loop_end_index
            )

        test, body_start, false_index = read
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            false_index - 1,
            continue_offset=int(instructions[cursor + 1].offset),
            break_index=loop_end_index or false_index,
        )
        after_index = skip_loop_cleanup(instructions, false_index, end_index)
        if (
            instructions[false_index].opname == "POP_BLOCK"
            and loop_end_index is not None
            and after_index < loop_end_index <= end_index
        ):
            self.append_while(instructions, test, body, (after_index, loop_end_index))
            return loop_end_index

        self.append_while(instructions, test, body, None)
        return after_index

    def append_while(
        self,
        instructions: list[Instruction],
        test: ast.expr,
        body: list[ast.stmt],
        else_range: tuple[int, int] | None,
    ) -> None:
        orelse: list[ast.stmt] = []

        if else_range is not None:
            orelse = self.translate_child_statements(instructions, *else_range)

        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=orelse)
        )

    def try_translate_classic_while_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if self.version is None or not (3, 8) <= self.version[:2] <= (3, 9):
            return None
        if not instructions[cursor].is_jump_target:
            return None

        loop_offset = int(instructions[cursor].offset)
        if loop_offset in self.loop_continue_offsets:
            return None

        back_jump_index = self.find_while_back_jump(
            instructions, cursor + 1, end_index, loop_offset
        )
        if back_jump_index is None or back_jump_index + 1 > end_index:
            return None

        exit_index = back_jump_index + 1
        read = self.block_loop_test(instructions, offset_to_index, cursor, exit_index)
        if read is None:
            # `while True:` has no test since 3.0
            body = self.translate_loop_child_statements(
                instructions,
                cursor,
                back_jump_index,
                continue_offset=loop_offset,
                break_index=exit_index,
            )
            self.statements.append(
                make_endless_while(body or [ast.Pass()], loop_offset)
            )
            return exit_index

        test, body_start = read
        else_end_index = find_loop_else_end(
            instructions, body_start, back_jump_index, exit_index, end_index
        )
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            back_jump_index,
            continue_offset=loop_offset,
            break_index=else_end_index or exit_index,
        )
        else_range = None if else_end_index is None else (exit_index, else_end_index)
        self.append_while(instructions, test, body, else_range)
        return else_end_index or exit_index

    def try_translate_endless_block_while(
        self,
        instructions: list[Instruction],
        loop_start: int,
        loop_end_index: int,
    ) -> int | None:
        loop_offset = int(instructions[loop_start].offset)
        if is_loop_back_jump(instructions[loop_end_index - 1], loop_offset):
            back_jump_index = loop_end_index - 1
        else:
            found = next(
                (
                    index
                    for index in range(loop_end_index - 1, loop_start, -1)
                    if instructions[index].opname == "POP_BLOCK"
                    and is_loop_back_jump(instructions[index - 1], loop_offset)
                ),
                None,
            )
            if found is None:
                return None

            back_jump_index = found - 1

        body = self.translate_loop_child_statements(
            instructions,
            loop_start,
            back_jump_index,
            continue_offset=loop_offset,
            break_index=loop_end_index,
        )
        loop = make_endless_while(body or [ast.Pass()], loop_offset)

        if back_jump_index + 2 < loop_end_index:
            loop.orelse = self.translate_child_statements(
                instructions, back_jump_index + 2, loop_end_index
            )

        self.statements.append(loop)
        return loop_end_index

    def read_block_while(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        loop_end_index: int,
    ) -> tuple[ast.expr, int, int] | None:
        loop_offset = int(instructions[cursor + 1].offset)

        for index in range(cursor + 1, loop_end_index):
            if not is_forward_conditional_jump(instructions[index]):
                continue

            exit_index = offset_to_index.get(int(instructions[index].argval))
            if (
                exit_index is not None
                and exit_index < loop_end_index
                and instructions[exit_index].opname == "POP_BLOCK"
                and is_loop_back_jump(instructions[exit_index - 1], loop_offset)
            ):
                read = self.block_loop_test(
                    instructions, offset_to_index, cursor + 1, exit_index
                )
                return None if read is None else (*read, exit_index)

        return None

    def read_first_test_while(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> tuple[ast.expr, int, int] | None:
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
        if condition is None or back_jump_index != false_index - 1:
            return None

        condition_jump = instructions[condition_jump_index]
        test = (
            condition
            if is_false_jump(condition_jump.opname)
            else invert_condition(condition)
        )
        return test, condition_jump_index + 1, false_index

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
        iterable = child.evaluate_expression_range(
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
            if instruction.opname == "FOR_ITER":
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
        # 3.14 keeps a loop-ending break's line as a NOP the test jumps to
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

        tail_start = break_marker_end(instructions, cursor, end_index)
        if instructions[tail_start].offset not in self.loop_break_offsets:
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

        if instruction.opname in {"UNPACK_EX", "UNPACK_SEQUENCE"}:
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
    elif closing.opname not in LOOP_BACK_JUMPS:
        return False

    # a 3.8/3.9 break the optimizer sent to the head
    return any(
        instruction.opname in LOOP_BACK_JUMPS
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
    if last.opname not in BACKWARD_JUMPS:
        return end_index
    if (
        instructions[cursor].offset
        <= last.argval
        <= instructions[back_jump_index].offset
    ):
        return back_jump_index + 1

    return end_index


def resumes_behind_earlier_try(
    code: object, instructions: list[Instruction], cursor: int, jump_index: int
) -> bool:
    spans = read_table_spans(code, instructions, make_offset_index(instructions))
    handler = enclosing_handler(handler_regions(instructions, spans), jump_index)
    if handler is None:
        return False

    starts = [span.start_index for span in spans if span.target_index == handler]
    return bool(starts) and max(starts) < cursor


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


def tails_below_loop(
    instructions: list[Instruction],
    loop_start: int,
    loop_end: int,
    tails: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    inside = {index for start, end in tails for index in range(start, end)}
    lines = [
        item.starts_line
        for index, item in enumerate(instructions[loop_start:loop_end], loop_start)
        if index not in inside and item.starts_line is not None
    ]
    if not lines:
        return tails

    last_line = max(lines)
    if all(
        (line := line_in_effect(instructions, start)) is not None and line > last_line
        for start, _ in tails
    ):
        return tails

    return ()


def find_inlined_break_tails(
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
    marker: str,
) -> tuple[tuple[int, int], ...]:
    found: list[tuple[int, int, int]] = []

    for index in range(body_start, body_end):
        tail_end = inlined_break_tail_end(instructions, index, body_end, marker)
        if tail_end is not None:
            found.append(
                (index, break_marker_end(instructions, index, body_end), tail_end)
            )

    markers = {index for index, _, _ in found}
    tails = [(start, end) for _, start, end in found if start not in markers]
    if any(
        end - start != tails[0][1] - tails[0][0]
        or not same_instruction_run(instructions, tails[0][0], start, end - start)
        for start, end in tails
    ):
        return ()

    return tuple(tails)


def break_marker_end(
    instructions: list[Instruction], marker_index: int, end_index: int
) -> int:
    index = marker_index + 1
    if (
        instructions[marker_index].opname == "POP_TOP"
        and index < end_index
        and instructions[index].opname == "POP_TOP"
        and instructions[index].starts_line is None
    ):
        index += 1

    return index


def inlined_break_tail_end(
    instructions: list[Instruction],
    marker_index: int,
    end_index: int,
    marker: str,
) -> int | None:
    if instructions[marker_index].opname != marker:
        return None

    start_index = break_marker_end(instructions, marker_index, end_index)
    if start_index >= end_index:
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


def find_setup_loop_get_iter(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname == "GET_ITER":
            return index
        if instruction.opname == "FOR_ITER":
            return None

    return None
