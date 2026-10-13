import ast
from abc import ABC
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

from pyc2py.astree import is_none_constant, make_constant, make_name
from pyc2py.bytecode.instruction import IGNORED_BEHAVIOR_OPNAMES, Instruction
from pyc2py.bytecode.metadata import make_offset_index
from pyc2py.bytecode.stack_effect import instruction_stack_effect
from pyc2py.decompiler.control_comprehension import (
    is_loop_back_jump,
    is_loop_back_test,
)
from pyc2py.decompiler.control_loop import (
    ControlLoopRecoveryMixin,
    find_loop_break_exit,
    find_loop_else_end,
    same_instruction_run,
)
from pyc2py.decompiler.exception_structures import (
    handler_regions,
    out_of_line_handlers_end,
    read_table_spans,
    same_instruction_behavior,
)
from pyc2py.decompiler.opcodes.flow import is_terminal_op, terminal_tail_end_index
from pyc2py.decompiler.runtime import AWAITED_BY_CALLER_MARK, coerce_expr
from pyc2py.decompiler.structures import (
    MatchCaseSpec,
    WithRegion,
    condition_from_jump,
    find_protected_with_region,
    find_simple_match_region,
    find_with_region,
    handler_join_index,
    invert_condition,
    is_false_jump,
    is_forward_conditional_jump,
    is_none_jump,
    line_of_instruction,
    make_async_with_statement,
    make_match,
    make_with_statement,
    unwrap_truth_test,
)


@dataclass(frozen=True, slots=True)
class ConditionTest:
    start_index: int
    end_index: int
    expression: ast.expr
    when_true: int
    when_false: int


@dataclass(frozen=True, slots=True)
class TargetedIfExpressionShape:
    jump_index: int
    false_index: int
    join_jump_index: int
    store_index: int


class ControlRecoveryMixin(ControlLoopRecoveryMixin, ABC):
    def try_translate_send_value(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        end_send_index = find_send_value_end(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if end_send_index is None:
            return None

        value = make_send_value_expression(
            instructions[cursor].opname,
            coerce_expr(self.pop_or_none()),
        )
        after_index = end_send_index + 1
        if after_index < end_index and instructions[after_index].opname == "POP_TOP":
            self.statements.append(ast.Expr(value=value))
            return after_index + 1

        self.stack.append(value)
        return after_index

    def try_translate_simple_match(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        region = find_simple_match_region(
            instructions, offset_to_index, cursor, end_index
        )
        if region is None:
            return None

        subject = coerce_expr(self.pop_or_none())
        cases: list[MatchCaseSpec] = []
        for case in region.cases:
            body = self.translate_isolated_child_statements(
                instructions,
                case.body_start_index,
                case.body_end_index,
            )
            guard = None
            if case.guard_start_index is not None and case.guard_end_index is not None:
                guard = self.evaluate_expression_range(
                    instructions,
                    case.guard_start_index,
                    case.guard_end_index,
                )
                if guard is not None:
                    guard = condition_from_jump(
                        instructions[case.guard_end_index].opname,
                        unwrap_truth_test(guard),
                    )
                    if not is_false_jump(instructions[case.guard_end_index].opname):
                        guard = invert_condition(guard)
            cases.append(
                MatchCaseSpec(pattern=case.pattern, guard=guard, body=tuple(body))
            )

        self.statements.append(make_match(subject, cases))
        return region.after_index

    def try_translate_with(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        region = find_protected_with_region(
            self.code,
            instructions,
            cursor,
            end_index,
            self.claimed_ranges,
        )
        if region is None:
            region = find_with_region(instructions, cursor, end_index)
        if region is None:
            return None

        optional_vars: ast.expr | None = None
        if region.optional_var_index is not None:
            target_instruction = instructions[region.optional_var_index]
            optional_vars = make_name(str(target_instruction.argval), ast.Store())
        elif region.target_end_index is not None:
            optional_vars = self.with_target(instructions, cursor + 1, region)
            if optional_vars is None:
                return None

        context_expr = self.with_context_expr(instructions, region)
        body = self.with_body(instructions, region)
        trailing_return = self.with_trailing_return(instructions, region)
        if trailing_return is not None and self.with_return_is_in_body(
            instructions, region, trailing_return
        ):
            body.append(trailing_return)
            trailing_return = None

        statement: ast.With | ast.AsyncWith
        if region.is_async:
            statement = make_async_with_statement(context_expr, body, optional_vars)
        else:
            statement = make_with_statement(context_expr, body, optional_vars)
        if line_of_instruction(instructions, cursor) == line_of_instruction(
            instructions, region.body_start_index
        ):
            join_with_items(statement)
        self.statements.append(statement)
        if trailing_return is not None:
            self.statements.append(trailing_return)
        return region.after_index

    def with_body(
        self,
        instructions: list[Instruction],
        region: WithRegion,
    ) -> list[ast.stmt]:
        child = self.make_child()
        child.stack.clear()
        body = child.translate_range(
            instructions, region.body_start_index, region.body_end_index
        )
        self.warnings.extend(child.warnings)
        if child.stack:
            body.append(ast.Return(value=coerce_expr(child.stack[-1])))
        elif region.returns_value and body and isinstance(body[-1], ast.Expr):
            body[-1] = ast.Return(value=body[-1].value)
        return body

    def with_trailing_return(
        self,
        instructions: list[Instruction],
        region: WithRegion,
    ) -> ast.stmt | None:
        if region.trailing_return_index is None:
            return None
        return self.return_statement_from_instruction(
            instructions[region.trailing_return_index]
        )

    def with_return_is_in_body(
        self,
        instructions: list[Instruction],
        region: WithRegion,
        trailing_return: ast.stmt,
    ) -> bool:
        if region.trailing_return_index is None:
            return False
        implicit_none = (
            self.version is not None
            and (
                self.version >= (3, 13)
                or (
                    self.version[:2] == (3, 12)
                    and instructions[region.body_end_index - 1].opname != "NOP"
                )
            )
            and isinstance(trailing_return, ast.Return)
            and trailing_return.value is not None
            and is_none_constant(trailing_return.value)
            and not ends_in_return_line(instructions, region)
        )
        return (
            instructions[region.trailing_return_index].starts_line is None
            and not implicit_none
        )

    def with_target(
        self,
        instructions: list[Instruction],
        start_index: int,
        region: WithRegion,
    ) -> ast.expr | None:
        child = self.make_child()
        child.stack.clear()
        child.stack.append(make_name("__pyc2py_with_value__", ast.Load()))
        statements = child.translate_range(
            instructions, start_index, region.target_end_index or start_index
        )
        if len(statements) != 1 or not isinstance(statements[0], ast.Assign):
            return None
        if len(statements[0].targets) != 1:
            return None
        return statements[0].targets[0]

    def return_statement_from_instruction(self, instruction: Instruction) -> ast.Return:
        if instruction.opname == "RETURN_CONST":
            return ast.Return(value=make_constant(instruction.argval))
        return ast.Return(value=coerce_expr(self.pop_or_none()))

    def with_context_expr(
        self,
        instructions: list[Instruction],
        region: WithRegion,
    ) -> ast.expr:
        if region.context_index is None:
            return coerce_expr(self.pop_or_none())
        instruction = instructions[region.context_index]
        return make_name(str(instruction.argval), ast.Load())

    def try_translate_backward_retry_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        loop_start = retry_loop_start_index(instructions, cursor, end_index)
        if loop_start is None:
            return None
        if self.protection_starts_between(instructions, cursor, loop_start):
            return None

        pattern = find_backward_retry_loop(
            instructions,
            offset_to_index,
            loop_start,
            end_index,
        )
        if pattern is None:
            return None
        condition_index, _, after_index = pattern

        child = self.make_child()
        if after_index < len(instructions):
            child.loop_break_offsets = self.loop_break_offsets | frozenset(
                {int(instructions[after_index].offset)}
            )
        body = child.translate_range(instructions, loop_start, condition_index)
        if not child.stack:
            return None

        condition = condition_from_jump(
            instructions[condition_index].opname,
            coerce_expr(child.stack[-1]),
        )
        if not is_false_jump(instructions[condition_index].opname):
            condition = invert_condition(condition)

        self.warnings.extend(child.warnings)
        self.statements.append(
            ast.While(
                test=ast.Constant(value=True),
                body=[*body, ast.If(test=condition, body=[ast.Break()], orelse=[])],
                orelse=[],
            )
        )
        return after_index

    def protection_starts_between(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> bool:
        if start_index == end_index:
            return False
        spans = read_table_spans(
            self.code, instructions, make_offset_index(instructions)
        )
        return any(start_index <= span.start_index < end_index for span in spans)

    def try_translate_conditional(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        target_index = offset_to_index.get(int(instruction.argval))
        if target_index is None or target_index <= cursor:
            return None
        target_index = fold_exit_jump_copy(instructions, target_index, end_index)
        if target_index > end_index and self.version_threads_jumps():
            target_index = threaded_operand_end(
                instructions, cursor, target_index, end_index
            )
        if target_index > end_index:
            return self.try_translate_conditional_break(
                instructions, cursor, end_index
            ) or self.try_translate_test_leaving_branch(instructions, cursor, end_index)

        short_circuit = self.try_translate_copied_short_circuit_value(
            instructions,
            cursor,
            target_index,
        )
        if short_circuit is not None:
            return short_circuit

        condition = condition_from_jump(
            instruction.opname, coerce_expr(self.pop_or_none())
        )
        translated = self.try_translate_conditional_pattern(
            instructions,
            offset_to_index,
            cursor,
            target_index,
            end_index,
            condition,
        )
        if translated is not None:
            return translated

        body_end = simple_if_body_end(instructions, cursor + 1, target_index, end_index)
        body = self.translate_child_statements(instructions, cursor + 1, body_end)
        test = (
            condition
            if is_false_jump(instruction.opname)
            else invert_condition(condition)
        )
        self.statements.append(ast.If(test=test, body=body or [ast.Pass()], orelse=[]))
        return target_index

    def try_translate_copied_short_circuit_value(
        self,
        instructions: list[Instruction],
        cursor: int,
        target_index: int,
    ) -> int | None:
        jump = instructions[cursor]
        if is_none_jump(jump.opname) or len(self.stack) < 2:
            return None
        pop_index = copied_value_pop_index(instructions, cursor, target_index)
        if pop_index is None:
            return None

        outer_test = None
        if self.version_threads_jumps():
            outer_test = threaded_outer_test(instructions, cursor, target_index)
        if outer_test is not None:
            target_index = outer_test

        child = self.make_child()
        child.pop_or_none()
        child.pop_or_none()
        operand_end = short_circuit_operand_end(
            instructions, pop_index + 1, target_index
        )
        if operand_end < target_index:
            self.claimed_ranges[operand_end] = target_index
        child.translate_range(instructions, pop_index + 1, target_index)
        if not child.stack:
            return None
        right = coerce_expr(child.stack[-1])
        self.warnings.extend(child.warnings)

        self.pop_or_none()
        left = coerce_expr(self.pop_or_none())
        operator: ast.boolop = ast.And() if is_false_jump(jump.opname) else ast.Or()
        values = [left]
        if isinstance(right, ast.BoolOp) and type(right.op) is type(operator):
            values.extend(right.values)
        else:
            values.append(right)
        self.stack.append(ast.BoolOp(op=operator, values=values))
        return target_index

    def version_threads_jumps(self) -> bool:
        # from python 3.14 a jump to the test of an enclosing "and" or "or"
        # goes on to where that test would send it
        return self.version is not None and self.version >= (3, 14)

    def try_translate_conditional_break(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        # python 3.14 lets a test jump straight to the code behind the loop
        # where older versions jump over a break. they do the same for a test
        # that ends the body in front of a written continue:
        # "if c: continue" and then a return
        jump = instructions[cursor]
        if self.version is None or (self.version < (3, 14) and cursor + 1 != end_index):
            return None
        if jump.argval not in self.loop_break_offsets or not self.stack:
            return None
        condition = condition_from_jump(jump.opname, coerce_expr(self.pop_or_none()))
        test = terminal_guard_for_jump(jump, condition, jumps_to_terminal=True)
        if self.version >= (3, 14) and self.translate_while_closed_by_handler(
            instructions, cursor, cursor + 1, end_index, invert_condition(test)
        ):
            loop = self.statements[-1]
            if isinstance(loop, ast.While):
                loop.orelse = [ast.Break()]
            return end_index
        self.statements.append(ast.If(test=test, body=[ast.Break()], orelse=[]))
        return cursor + 1

    def try_translate_test_leaving_branch(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        # up to 3.12 a failed test does not stop at the jump over an else
        # branch but goes where that jump goes, out of the range of the
        # branch it stands in. when the rest of the range leaves by itself
        # (it raises or returns), that rest is what the test guards
        jump = instructions[cursor]
        if self.version is None or self.version >= (3, 13) or not self.stack:
            return None
        if cursor + 1 >= end_index or jump.argval in (
            self.loop_break_offsets | self.loop_continue_offsets
        ):
            return None
        if not is_terminal_op(instructions[end_index - 1].opname):
            return None
        condition = condition_from_jump(jump.opname, coerce_expr(self.pop_or_none()))
        test = condition if is_false_jump(jump.opname) else invert_condition(condition)
        body = self.translate_child_statements(instructions, cursor + 1, end_index)
        self.statements.append(ast.If(test=test, body=body, orelse=[]))
        return end_index

    def try_translate_conditional_pattern(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        for handler_name in CONDITIONAL_HANDLER_ORDER:
            translated = call_conditional_handler(
                self,
                handler_name,
                instructions,
                offset_to_index,
                cursor,
                target_index,
                end_index,
                condition,
            )
            if translated is not None:
                return translated
        return None

    def try_translate_disjunctive_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        body_start: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        jump_result = find_disjunctive_guard_jumps(
            instructions,
            offset_to_index,
            cursor,
            body_start,
            end_index,
        )
        if jump_result is None:
            return None
        jump_indexes, skip_index = jump_result
        if skip_index <= body_start or skip_index > end_index:
            return None
        guard = self.build_disjunctive_guard(
            instructions,
            offset_to_index,
            jump_indexes,
            cursor,
            body_start,
            condition,
        )
        if guard is None or is_false_constant(guard):
            return None
        return self.translate_guarded_body(
            instructions,
            offset_to_index,
            body_start,
            skip_index,
            end_index,
            guard,
        )

    def try_translate_compound_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        if is_chained_compare_test(instructions, cursor):
            return None
        compound = self.multi_line_while_condition(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        ) or compound_condition_jumps(
            instructions,
            offset_to_index,
            cursor,
            end_index,
            self.version,
        )
        if compound is None:
            return None
        jumps, skip_index = compound
        if (
            self.version is not None
            and self.version >= (3, 14)
            and (loop_starts_between(instructions, cursor, jumps[-1] + 1, skip_index))
        ):
            return None
        if tests_in_front_of_while(instructions, cursor, jumps[-1] + 1, skip_index):
            return None
        guard = self.build_compound_condition(
            instructions,
            offset_to_index,
            jumps,
            skip_index,
            condition,
        )
        if guard is None:
            return None

        expression_end = self.translate_if_expression_branches(
            instructions,
            offset_to_index,
            jumps[-1] + 1,
            skip_index,
            end_index,
            guard,
        )
        if expression_end is not None:
            return expression_end
        if self.stack:
            return None

        loop_end = self.translate_modern_while(
            instructions,
            cursor,
            jumps[-1] + 1,
            skip_index,
            end_index,
            guard,
        )
        if loop_end is not None:
            return loop_end
        return self.translate_guarded_body(
            instructions,
            offset_to_index,
            jumps[-1] + 1,
            skip_index,
            end_index,
            guard,
        )

    def multi_line_while_condition(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> tuple[list[int], int] | None:
        compound = compound_condition_jumps(
            instructions,
            offset_to_index,
            cursor,
            end_index,
            self.version,
            on_any_line=True,
        )
        if compound is None:
            return None
        jumps, skip_index = compound
        back_jump_index = self.find_modern_while_back_jump(
            instructions,
            jumps[-1] + 1,
            skip_index,
        )
        if back_jump_index is None:
            return None
        copy_start = repeated_condition_copy_start(
            instructions,
            jumps[-1],
            back_jump_index - 1,
            find_condition_line(instructions, cursor),
        )
        return None if copy_start is None else compound

    def build_compound_condition(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        jumps: list[int],
        skip_index: int,
        condition: ast.expr,
    ) -> ast.expr | None:
        tests: list[ConditionTest] = []
        start_index = jumps[0]
        for jump_index in jumps:
            jump = instructions[jump_index]
            expression = condition
            if jump_index != jumps[0]:
                operand = self.evaluate_expression_range(
                    instructions,
                    start_index,
                    jump_index,
                )
                if operand is None:
                    return None
                expression = condition_from_jump(jump.opname, operand)
            target_index = offset_to_index[int(jump.argval)]
            if target_index > skip_index:
                target_index = skip_index
            target_index = body_entry_index(instructions, jumps[-1] + 1, target_index)
            if is_false_jump(jump.opname):
                successors = (jump_index + 1, target_index)
            else:
                successors = (target_index, jump_index + 1)
            tests.append(
                ConditionTest(start_index, jump_index + 1, expression, *successors)
            )
            start_index = jump_index + 1
        return reduce_condition_tests(tests, jumps[-1] + 1, skip_index)

    def build_looped_condition(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        start_index: int,
        jumps: list[int],
        loop_offset: int,
    ) -> ast.expr | None:
        tests: list[ConditionTest] = []
        for jump_index in jumps:
            jump = instructions[jump_index]
            operand = self.evaluate_expression_range(
                instructions,
                start_index,
                jump_index,
            )
            target_index = offset_to_index.get(int(jump.argval))
            if operand is None or target_index is None:
                return None
            next_index = jump_index + 1
            if is_loop_back_jump(
                instructions[target_index], loop_offset
            ) or is_loop_back_test(jump, loop_offset):
                target_index = LOOP_BACK
            if is_loop_back_jump(instructions[next_index], loop_offset):
                successors = (target_index, LOOP_BACK)
                next_index += 1
            else:
                successors = (target_index, next_index)
            if is_false_jump(jump.opname):
                successors = (successors[1], successors[0])
            expression = condition_from_jump(jump.opname, operand)
            tests.append(
                ConditionTest(start_index, next_index, expression, *successors)
            )
            start_index = next_index
        return reduce_condition_tests(tests, start_index, LOOP_BACK)

    def translate_guarded_body(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        body_start: int,
        skip_index: int,
        end_index: int,
        guard: ast.expr,
    ) -> int | None:
        body_end = skip_index
        after_index = skip_index
        join_offset: int | None = None
        else_jump = else_jump_index(self, instructions, body_start, skip_index)
        if else_jump is None and skip_index < len(instructions):
            inner_jump = jump_over_else_inside(
                self, instructions, body_start - 1, skip_index
            )
            if inner_jump is not None:
                join_offset = int(instructions[inner_jump].argval)
                join_index = offset_to_index.get(join_offset)
                after_index = min(join_index or skip_index, end_index)
            elif self.ends_before_module_return(
                instructions, skip_index - self.module_return_length()
            ):
                after_index = end_index
        if else_jump is not None:
            join_index = offset_to_index.get(int(instructions[else_jump].argval))
            if join_index is None:
                return None
            if any(
                is_forward_conditional_jump(instructions[index])
                and instructions[index].argval == instructions[skip_index].offset
                for index in range(body_start, else_jump)
            ):
                return None
            body_end = else_jump
            after_index = min(join_index, end_index)

        if join_offset is None:
            body = self.translate_child_statements(instructions, body_start, body_end)
        else:
            body = self.translate_first_branch(
                instructions, body_start, body_end, join_offset
            )
        if not body:
            return None
        orelse = self.translate_child_statements(instructions, skip_index, after_index)
        self.statements.append(ast.If(test=guard, body=body, orelse=orelse))
        return after_index

    def build_disjunctive_guard(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        jump_indexes: frozenset[int],
        cursor: int,
        body_start: int,
        condition: ast.expr,
    ) -> ast.expr | None:
        results: dict[int, ast.expr | None] = {body_start: ast.Constant(value=True)}
        for start_index in range(body_start - 1, cursor - 1, -1):
            jump_index = next_condition_jump(
                instructions, jump_indexes, start_index, body_start
            )
            if jump_index is None:
                results[start_index] = ignored_guard_result(
                    instructions, start_index, body_start
                )
                continue

            expression = self.read_disjunctive_guard_expression(
                instructions, start_index, jump_index, cursor, condition
            )
            jump = instructions[jump_index]
            target_index = offset_to_index.get(int(jump.argval))
            taken = guard_result_at(results, target_index, body_start)
            fallthrough = guard_result_at(results, jump_index + 1, body_start)
            if expression is None or taken is None or fallthrough is None:
                results[start_index] = None
                continue

            jump_condition = condition_from_jump(jump.opname, expression)
            results[start_index] = combine_branch_success(
                terminal_guard_for_jump(jump, jump_condition, jumps_to_terminal=True),
                taken,
                terminal_guard_for_jump(jump, jump_condition, jumps_to_terminal=False),
                fallthrough,
            )
        return results.get(cursor)

    def read_disjunctive_guard_expression(
        self,
        instructions: list[Instruction],
        start_index: int,
        jump_index: int,
        cursor: int,
        initial_condition: ast.expr,
    ) -> ast.expr | None:
        if jump_index == cursor:
            return initial_condition
        if not simple_condition_range(
            instructions, start_index, jump_index, self.version
        ):
            return None
        return self.evaluate_expression_range(instructions, start_index, jump_index)

    def try_translate_targeted_prefixed_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        body_start: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        prefix = targeted_prefix_jumps(
            instructions,
            offset_to_index,
            cursor,
            body_start,
            end_index,
            self.version,
        )
        if prefix is None:
            return None
        prefix_jumps, skip_index = prefix
        if else_jump_index(self, instructions, body_start, skip_index) is not None:
            return None

        body_offset = instructions[body_start].offset
        skip_offset = instructions[skip_index].offset
        current = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        scan_start = cursor + 1
        pieces: list[tuple[Instruction, ast.expr]] = []
        for index in prefix_jumps:
            jump = instructions[index]
            expr = self.evaluate_expression_range(instructions, scan_start, index)
            if expr is None:
                return None
            expr = condition_from_jump(jump.opname, expr)
            if jump.argval not in {skip_offset, body_offset}:
                return None
            pieces.append((jump, expr))
            scan_start = index + 1

        suffix = make_statement_condition_suffix(
            pieces,
            skip_offset,
            body_offset,
        )
        body = self.translate_child_statements(
            instructions,
            body_start,
            skip_index,
        )
        if not body:
            return None
        self.statements.append(
            ast.If(test=make_bool_or([current, suffix]), body=body, orelse=[])
        )
        return skip_index

    def try_translate_prefixed_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        condition: ast.expr,
    ) -> int | None:
        prefix_jumps = statement_condition_prefix_jumps(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            false_index,
            self.version,
        )
        if prefix_jumps is None:
            return None

        false_offset = instructions[false_index].offset
        body_start = statement_condition_body_start(
            instructions,
            offset_to_index,
            prefix_jumps,
            false_offset,
        )
        if body_start is None or body_start <= cursor or body_start >= false_index:
            return None
        if else_jump_index(
            self, instructions, body_start, false_index
        ) is not None or tests_in_front_of_while(
            instructions, cursor, body_start, false_index
        ):
            return None
        if (
            self.version is not None
            and self.version >= (3, 14)
            and (loop_starts_between(instructions, cursor, body_start, false_index))
        ):
            return None

        current = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=False,
        )
        scan_start = cursor + 1
        body_offset = instructions[body_start].offset
        pieces: list[tuple[Instruction, ast.expr]] = []
        for index in prefix_jumps:
            jump = instructions[index]
            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
                return None
            expr = self.evaluate_expression_range(instructions, scan_start, index)
            if expr is None:
                return None
            expr = condition_from_jump(jump.opname, expr)
            if jump.argval not in {false_offset, body_offset}:
                return None
            pieces.append((jump, expr))
            scan_start = index + 1

        suffix = make_statement_condition_suffix(
            pieces,
            false_offset,
            body_offset,
        )
        body = self.translate_child_statements(
            instructions,
            body_start,
            false_index,
        )
        if not body:
            return None
        self.statements.append(
            ast.If(test=make_bool_and(current, suffix), body=body, orelse=[])
        )
        return false_index

    def try_translate_targeted_if_expression_assignment(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        true_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        shape = targeted_if_expression_shape(
            instructions,
            offset_to_index,
            cursor,
            true_index,
            end_index,
        )
        if shape is None:
            return None
        jump = instructions[shape.jump_index]
        store = instructions[shape.store_index]

        right = self.evaluate_expression_range(
            instructions,
            cursor + 1,
            shape.jump_index,
        )
        true_value = self.evaluate_expression_range(
            instructions,
            true_index,
            shape.join_jump_index,
        )
        false_value = self.evaluate_expression_range(
            instructions,
            shape.false_index,
            shape.store_index,
        )
        if right is None or true_value is None or false_value is None:
            return None

        current = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        extra = terminal_guard_for_jump(
            jump,
            condition_from_jump(jump.opname, right),
            jumps_to_terminal=False,
        )
        expression_condition = make_bool_or([current, extra])

        value = ast.IfExp(
            test=expression_condition,
            body=true_value,
            orelse=false_value,
        )
        target = make_name(str(store.argval), ast.Store())
        if store.opname == "STORE_GLOBAL":
            self.add_global_name(str(store.argval))
        self.statements.append(ast.Assign(targets=[target], value=value))
        return shape.store_index + 1

    def try_translate_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        guard = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        scan_start = cursor + 1
        after_index: int | None = None
        found_skip_jump = False
        for index in range(cursor + 1, target_index):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue

            skip_index = offset_to_index.get(int(instruction.argval))
            if (
                skip_index is None
                or skip_index <= target_index
                or skip_index > end_index
            ):
                return None
            if after_index is None:
                after_index = skip_index
            elif after_index != skip_index:
                return None

            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
                return None
            guard_expr = self.evaluate_expression_range(
                instructions,
                scan_start,
                index,
            )
            if guard_expr is None:
                return None
            guard = ast.BoolOp(
                op=ast.Or(),
                values=[
                    guard,
                    terminal_guard_for_jump(
                        instruction,
                        condition_from_jump(instruction.opname, guard_expr),
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1
            found_skip_jump = True

        if (
            not found_skip_jump
            or after_index is None
            or else_jump_index(self, instructions, target_index, after_index)
            is not None
            or not has_only_ignored_instructions(instructions, scan_start, target_index)
        ):
            return None

        body = self.translate_child_statements(
            instructions,
            target_index,
            after_index,
        )
        if not body:
            return None
        self.statements.append(ast.If(test=guard, body=body, orelse=[]))
        return after_index

    def try_translate_clipped_if_expression_return(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        if false_index != end_index:
            return None
        if false_index + 1 >= len(instructions):
            return None

        jump_index = previous_non_ignored_index(
            instructions,
            false_index - 1,
            cursor + 1,
        )
        if jump_index is None:
            return None
        jump = instructions[jump_index]
        if jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return None

        join_index = offset_to_index.get(int(jump.argval))
        if join_index is None or join_index != false_index + 1:
            return None
        if instructions[join_index].opname not in {"RETURN_CONST", "RETURN_VALUE"}:
            return None

        true_value = self.evaluate_expression_range(
            instructions,
            cursor + 1,
            jump_index,
        )
        false_value = self.evaluate_expression_range(
            instructions,
            false_index,
            join_index,
        )
        if true_value is None or false_value is None:
            return None

        if not is_false_jump(instructions[cursor].opname):
            condition = invert_condition(condition)
        expression = ast.IfExp(test=condition, body=true_value, orelse=false_value)
        self.statements.append(ast.Return(value=expression))
        return join_index + 1

    def try_translate_loop_continue_guard_statement(
        self,
        instructions: list[Instruction],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        if not target_is_loop_continue(instructions, cursor, target_index, end_index):
            return None
        if find_split_while_tail(instructions, cursor + 1, target_index) is not None:
            return None
        closing = instructions[target_index - 1]
        if closing.opname.startswith("POP_JUMP_BACKWARD_IF_") and (
            closing.argval == instructions[cursor + 1].offset
        ):
            return None

        after_index = target_index + 1
        after_offset = None
        if after_index < len(instructions):
            after_offset = instructions[after_index].offset

        guard = self.read_loop_continue_guard(
            instructions,
            cursor,
            target_index,
            after_offset,
            condition,
        )
        if guard is not None:
            self.statements.append(ast.If(test=guard, body=[ast.Continue()], orelse=[]))
            return after_index

        body = self.translate_child_statements(
            instructions,
            cursor + 1,
            target_index,
        )
        if body:
            test = (
                condition
                if is_false_jump(instructions[cursor].opname)
                else invert_condition(condition)
            )
            self.statements.append(ast.If(test=test, body=body, orelse=[]))
            if target_index >= end_index:
                return end_index
            if target_index < end_index and (
                not body_ends_with_break(body)
                or jumps_back_again(instructions, target_index)
            ):
                self.statements.append(ast.Continue())
            return after_index

        guard = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        self.statements.append(ast.If(test=guard, body=[ast.Continue()], orelse=[]))
        return after_index

    def read_loop_continue_guard(
        self,
        instructions: list[Instruction],
        cursor: int,
        target_index: int,
        after_offset: int | None,
        condition: ast.expr,
    ) -> ast.expr | None:
        guard = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        found_skip_jump = False
        scan_start = cursor + 1
        for index in range(cursor + 1, target_index):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue
            if instruction.argval != after_offset:
                continue
            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
                continue
            found_skip_jump = True
            guard_expr = self.evaluate_expression_range(
                instructions,
                scan_start,
                index,
            )
            if guard_expr is None:
                return None
            guard = ast.BoolOp(
                op=ast.Or(),
                values=[
                    guard,
                    terminal_guard_for_jump(
                        instruction,
                        condition_from_jump(instruction.opname, guard_expr),
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1
        return guard if found_skip_jump else None

    def try_translate_for_tail_if_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        if target_index != cursor + 2 or not jumps_over_for_continue(
            instructions,
            offset_to_index,
            cursor,
        ):
            return None
        if ends_in_written_continue(instructions, end_index):
            return None
        if rest_runs_on_behind(instructions, target_index, end_index):
            return None
        body = self.translate_child_statements(instructions, target_index, end_index)
        if not body:
            return None
        self.statements.append(ast.If(test=condition, body=body, orelse=[]))
        return end_index

    def trailing_if_test(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
        condition: ast.expr,
        test: ast.expr,
    ) -> ast.expr | None:
        if self.is_flagged_trailing_if(instructions, cursor, end_index, condition):
            return condition
        if self.is_trailing_none_test(instructions, cursor, end_index):
            return invert_condition(test)
        return None

    def try_translate_loop_continue_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        terminal_index = loop_continue_terminal_start(
            instructions,
            offset_to_index,
            cursor + 1,
            target_index,
            end_index,
            self.loop_continue_offsets,
        )
        if terminal_index is None or (
            jump_over_else_inside(self, instructions, cursor, target_index) is not None
        ):
            return None

        after_offset = instructions[target_index].offset
        if has_effectful_skip_to_offset(
            instructions,
            cursor + 1,
            terminal_index,
            after_offset,
            self.version,
        ):
            return None

        test = (
            condition
            if is_false_jump(instructions[cursor].opname)
            else invert_condition(condition)
        )
        guard = test
        found_skip_jump = False
        scan_start = cursor + 1
        for index in range(cursor + 1, terminal_index):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue
            if instruction.argval != after_offset:
                continue
            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
                continue
            guard_expr = self.evaluate_expression_range(
                instructions,
                scan_start,
                index,
            )
            if guard_expr is None:
                return None
            found_skip_jump = True
            guard = ast.BoolOp(
                op=ast.And(),
                values=[
                    guard,
                    terminal_guard_for_jump(
                        instruction,
                        condition_from_jump(instruction.opname, guard_expr),
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1

        if found_skip_jump:
            body = self.translate_child_statements(
                instructions, scan_start, terminal_index
            )
            self.statements.append(
                ast.If(test=guard, body=[*body, ast.Continue()], orelse=[])
            )
            return target_index

        trailing_test = (
            self.trailing_if_test(instructions, cursor, end_index, condition, test)
            if terminal_index == cursor + 1
            else None
        )
        if trailing_test is not None:
            rest = self.translate_child_statements(
                instructions, target_index, end_index
            )
            self.statements.append(
                ast.If(test=trailing_test, body=rest or [ast.Pass()], orelse=[])
            )
            return end_index

        body = self.translate_child_statements(
            instructions,
            cursor + 1,
            terminal_index,
        )
        self.statements.append(
            ast.If(test=test, body=[*body, ast.Continue()], orelse=[])
        )
        return target_index

    def is_trailing_none_test(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> bool:
        # "if x is not None: rest" at the end of a loop body. up to python
        # 3.13 it compiles to the same code as "if x is None: continue"; from
        # 3.14 a continue that was written starts a line and is covered by
        # the exception table of a generator, the jump python adds is neither
        if self.version is None or self.version < (3, 14):
            return False
        if not is_none_jump(instructions[cursor].opname):
            return False
        if instructions[cursor + 1].starts_line is not None:
            return False
        if end_index >= len(instructions):
            return False
        after = instructions[end_index]
        return (
            after.opname == "JUMP_BACKWARD"
            and after.argval in self.loop_continue_offsets
        )

    def is_flagged_trailing_if(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
        condition: ast.expr,
    ) -> bool:
        opname = instructions[cursor].opname
        if is_false_jump(opname) or is_none_jump(opname):
            return False
        if is_none_jump(instructions[cursor].opname):
            return False
        if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
            return False
        if not isinstance(condition.ops[0], ast.In | ast.NotIn | ast.Is | ast.IsNot):
            return False
        if end_index >= len(instructions):
            return False
        after = instructions[end_index]
        return (
            after.opname == "JUMP_BACKWARD"
            and after.argval in self.loop_continue_offsets
        )

    def try_translate_terminal_guard_statement(
        self,
        instructions: list[Instruction],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        terminal_end = terminal_block_end_index(instructions, target_index, end_index)
        if terminal_end is None:
            return None

        after_offset = None
        if terminal_end < len(instructions):
            after_offset = instructions[terminal_end].offset
        found_skip_jump = False
        guard = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=True,
        )
        scan_start = cursor + 1
        terminal_offset = instructions[target_index].offset
        inner_target = 0
        for index in range(cursor + 1, target_index):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue
            if instruction.argval not in {after_offset, terminal_offset}:
                inner_target = max(inner_target, int(instruction.argval))
                continue
            if instruction.offset < inner_target:
                return None
            found_skip_jump = instruction.argval == after_offset
            guard_expr = self.evaluate_expression_range(
                instructions,
                scan_start,
                index,
            )
            if guard_expr is None:
                return None
            guard = ast.BoolOp(
                op=ast.Or(),
                values=[
                    guard,
                    terminal_guard_for_jump(
                        instruction,
                        condition_from_jump(instruction.opname, guard_expr),
                        jumps_to_terminal=instruction.argval == terminal_offset,
                    ),
                ],
            )
            scan_start = index + 1

        between = [
            (instruction.opname, instruction.argval)
            for instruction in instructions[scan_start:target_index]
            if instruction.opname != "NOP"
        ]
        block = [
            (instruction.opname, instruction.argval)
            for instruction in instructions[target_index:terminal_end]
        ]
        if not found_skip_jump or (between and between != block):
            return None

        body = self.translate_child_statements(
            instructions,
            target_index,
            terminal_end,
        )
        if not body:
            return None
        self.statements.append(ast.If(test=guard, body=body, orelse=[]))
        return terminal_end

    def scan_until_chained_comparison_target_change(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        scan_start: int,
        false_index: int,
        end_index: int,
        comparison: ast.Compare,
        before_else: bool = False,
    ) -> tuple[Any, ast.Compare, int, int] | None:
        child = self.make_child()
        for _scan_step in range(len(instructions)):
            jump_index = find_chained_comparison_jump(
                instructions, scan_start, false_index
            )
            if jump_index is None:
                return None
            target_index = offset_to_index.get(int(instructions[jump_index].argval))
            if target_index is None or target_index <= jump_index:
                return None
            if target_index > end_index and not (
                before_else
                and is_branch_end_jump(instructions, end_index, target_index)
            ):
                return None

            merged = extend_chained_comparison(
                child, instructions, scan_start, jump_index, comparison
            )
            if merged is None:
                return None
            comparison = merged
            if target_index != false_index:
                return child, comparison, jump_index, target_index
            scan_start = jump_index + 1
        return None

    def scan_fixed_target_chained_comparison(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        scan_start: int,
        false_index: int,
        comparison: ast.Compare,
    ) -> tuple[Any, ast.Compare, int] | None:
        child = self.make_child()
        for _scan_step in range(len(instructions)):
            jump_index = find_chained_comparison_jump(
                instructions, scan_start, false_index
            )
            if jump_index is None:
                return child, comparison, scan_start
            if offset_to_index.get(int(instructions[jump_index].argval)) != false_index:
                return None

            merged = extend_chained_comparison(
                child, instructions, scan_start, jump_index, comparison
            )
            if merged is None:
                return None
            comparison = merged
            scan_start = jump_index + 1
        return None

    def try_translate_chained_comparison(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        first_compare = single_compare(condition)
        if first_compare is None or not is_false_jump(instructions[cursor].opname):
            return None

        return_statement = self.try_translate_chained_comparison_return(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            end_index,
            first_compare,
        )
        if return_statement is not None:
            return return_statement

        assignment = self.try_translate_chained_comparison_assignment(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            first_compare,
        )
        if assignment is not None:
            return assignment

        value = self.try_translate_chained_comparison_value(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            first_compare,
        )
        if value is not None:
            return value

        negated = self.try_translate_negated_chained_comparison(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            end_index,
            first_compare,
        )
        if negated is not None:
            return negated

        body_return = self.try_translate_chained_comparison_body_return(
            instructions,
            offset_to_index,
            cursor,
            false_index,
            end_index,
            first_compare,
        )
        if body_return is not None:
            return body_return

        scan = self.scan_until_chained_comparison_target_change(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            end_index,
            first_compare,
        )
        if scan is None or not drops_chain_operand(instructions, false_index):
            return None
        child, combined, next_jump_index, next_false_index = scan
        self.warnings.extend(child.warnings)

        body_end = min(
            false_index,
            simple_if_body_end(
                instructions,
                next_jump_index + 1,
                next_false_index,
                end_index,
            ),
        )
        true_start = chained_compare_true_body_start(
            instructions,
            false_index,
            next_false_index,
        )
        if true_start is not None:
            guarded = self.translate_guarded_body(
                instructions,
                offset_to_index,
                true_start,
                next_false_index,
                end_index,
                combined,
            )
            if guarded is not None:
                return guarded
            body_end = next_false_index
        body = self.translate_child_statements(
            instructions,
            next_jump_index + 1 if true_start is None else true_start,
            body_end,
        )
        self.statements.append(
            ast.If(test=combined, body=body or [ast.Pass()], orelse=[])
        )
        return next_false_index

    def try_translate_chained_comparison_before_else(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        first_compare = single_compare(condition)
        if first_compare is None or not is_false_jump(instructions[cursor].opname):
            return None
        scan = self.scan_until_chained_comparison_target_change(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            end_index,
            first_compare,
            before_else=True,
        )
        if scan is None or scan[3] <= end_index:
            return None
        child, combined, _, next_false_index = scan
        true_start = chained_compare_true_body_start(
            instructions, false_index, next_false_index
        )
        if true_start is None:
            return None
        self.warnings.extend(child.warnings)
        body = self.translate_child_statements(instructions, true_start, end_index)
        self.statements.append(
            ast.If(test=combined, body=body or [ast.Pass()], orelse=[])
        )
        return end_index

    def try_translate_negated_chained_comparison(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        scan = self.scan_fixed_target_chained_comparison(
            instructions, offset_to_index, cursor + 1, false_index, first_compare
        )
        if scan is None:
            return None
        child, comparison, link_start = scan
        exits = negated_chain_exits(
            instructions, offset_to_index, link_start, false_index
        )
        if exits is None or exits[2] > end_index:
            return None
        jump_index, body_start, after_index = exits
        merged = extend_chained_comparison(
            child, instructions, link_start, jump_index, comparison
        )
        if merged is None:
            return None

        guard = ast.UnaryOp(op=ast.Not(), operand=merged)
        translated = self.translate_guarded_body(
            instructions, offset_to_index, body_start, after_index, end_index, guard
        )
        if translated is not None:
            self.warnings.extend(child.warnings)
        return translated

    def try_translate_chained_comparison_body_return(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        scan = self.scan_until_chained_comparison_target_change(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            end_index,
            first_compare,
        )
        if scan is None:
            return None
        child, combined, _, next_false_index = scan

        true_start = chained_compare_true_body_start(
            instructions,
            false_index,
            next_false_index,
        )
        if true_start is None:
            return None

        true_return_index = previous_non_ignored_index(
            instructions,
            next_false_index - 1,
            true_start,
        )
        if true_return_index is None:
            return None
        if instructions[true_return_index].opname not in {
            "RETURN_CONST",
            "RETURN_VALUE",
        }:
            return None

        false_return_end = chained_compare_none_return_end(
            instructions,
            next_false_index,
            end_index,
        )
        if false_return_end is None:
            return None

        body = self.translate_child_statements(
            instructions,
            true_start,
            true_return_index + 1,
        )
        if not body:
            return None
        self.warnings.extend(child.warnings)
        self.statements.append(ast.If(test=combined, body=body, orelse=[]))
        self.statements.append(ast.Return(value=ast.Constant(value=None)))
        return false_return_end

    def try_translate_chained_comparison_return(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        true_start = skip_chained_compare_true_cleanup(
            instructions, cursor + 1, false_index
        )
        if true_start is None:
            return None

        true_return_index = previous_non_ignored_index(
            instructions,
            false_index - 1,
            true_start,
        )
        if true_return_index is None:
            return None
        if instructions[true_return_index].opname not in {
            "RETURN_CONST",
            "RETURN_VALUE",
        }:
            return None

        scan = self.scan_fixed_target_chained_comparison(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            first_compare,
        )
        if scan is None:
            return None
        child, combined, scan_start = scan

        child.translate_range(instructions, scan_start, true_return_index)
        if not child.stack:
            return None
        final_compare = coerce_expr(child.stack[-1])
        merged = merge_chained_compare(combined, final_compare)
        if merged is None:
            return None
        combined = merged
        self.warnings.extend(child.warnings)

        after_false_return = chained_compare_false_return_end(
            instructions,
            false_index,
            end_index,
        )
        if after_false_return is None:
            return None

        self.statements.append(ast.Return(value=combined))
        return after_false_return

    def try_translate_chained_comparison_value(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        join_jump_index = previous_non_ignored_index(
            instructions,
            false_index - 1,
            cursor + 1,
        )
        if join_jump_index is None:
            return None
        join_jump = instructions[join_jump_index]
        if join_jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return None
        join_index = offset_to_index.get(int(join_jump.argval))
        if join_index is None or join_index != chained_compare_cleanup_end(
            instructions,
            false_index,
            join_index,
        ):
            return None

        scan = self.scan_fixed_target_chained_comparison(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            first_compare,
        )
        if scan is None:
            return None
        child, combined, scan_start = scan
        child.translate_range(instructions, scan_start, join_jump_index)
        if not child.stack:
            return None
        merged = merge_chained_compare(combined, coerce_expr(child.stack[-1]))
        if merged is None or len(self.stack) < 2:
            return None

        self.warnings.extend(child.warnings)
        self.pop_or_none()
        self.pop_or_none()
        self.stack.append(merged)
        return join_index

    def try_translate_chained_comparison_assignment(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        true_start = skip_chained_compare_true_cleanup(
            instructions, cursor + 1, false_index
        )
        if true_start is None:
            return None

        true_store_index = find_chained_compare_true_store(
            instructions,
            true_start,
            false_index,
        )
        if true_store_index is None:
            return None

        true_store = instructions[true_store_index]
        scan = self.scan_fixed_target_chained_comparison(
            instructions,
            offset_to_index,
            cursor + 1,
            false_index,
            first_compare,
        )
        if scan is None:
            return None
        child, combined, scan_start = scan

        child.translate_range(instructions, scan_start, true_store_index)
        if not child.stack:
            return None
        final_compare = coerce_expr(child.stack[-1])
        merged = merge_chained_compare(combined, final_compare)
        if merged is None:
            return None
        combined = merged

        after_false_store = chained_compare_false_store_end(
            instructions,
            false_index,
            len(instructions),
            true_store,
        )
        if after_false_store is None:
            return None

        if true_store.opname == "STORE_GLOBAL":
            self.add_global_name(str(true_store.argval))
        self.warnings.extend(child.warnings)
        self.statements.append(
            ast.Assign(
                targets=[make_name(str(true_store.argval), ast.Store())],
                value=combined,
            )
        )
        return after_false_store

    def try_translate_modern_while_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        exit_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        test = (
            condition
            if is_false_jump(instructions[cursor].opname)
            else invert_condition(condition)
        )
        return self.translate_modern_while(
            instructions,
            cursor,
            cursor + 1,
            exit_index,
            end_index,
            test,
        )

    def try_translate_continue_exit_while_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        target_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        tests = continue_exit_tests(
            instructions,
            cursor,
            end_index,
            self.loop_continue_offsets,
            self.version,
        )
        if not tests:
            return None
        body_start = tests[-1] + 2
        back_jump_index = self.continue_exit_back_jump(
            instructions, cursor, body_start, end_index
        )
        if back_jump_index is None:
            return None

        test = condition
        for previous_jump, jump_index in pairwise(tests):
            operand = self.evaluate_expression_range(
                instructions,
                previous_jump + 2,
                jump_index,
            )
            if operand is None:
                return None
            test = make_bool_and(
                test,
                condition_from_jump(instructions[jump_index].opname, operand),
            )
        if (
            self.version is not None
            and self.version >= (3, 14)
            and back_jump_index + 1 < end_index
            and is_terminal_op(instructions[end_index - 1].opname)
        ):
            return self.finish_modern_while(
                instructions,
                cursor,
                (body_start, end_index, end_index, end_index),
                test,
            )
        exit_index = self.translate_modern_while(
            instructions,
            cursor,
            body_start,
            back_jump_index + 1,
            end_index,
            test,
        )
        if exit_index is None or exit_index >= end_index:
            return exit_index
        if instructions[exit_index].argval in self.loop_continue_offsets:
            return exit_index + 1
        return exit_index

    def continue_exit_back_jump(
        self,
        instructions: list[Instruction],
        cursor: int,
        body_start: int,
        end_index: int,
    ) -> int | None:
        if self.version is not None and self.version >= (3, 14):
            test_offsets = condition_prefix_offsets(instructions, cursor)
            return next(
                (
                    index
                    for index in range(end_index - 1, body_start - 1, -1)
                    if instructions[index].opname == "JUMP_BACKWARD"
                    and instructions[index].argval in test_offsets
                ),
                None,
            )
        back_jump_index = self.find_modern_while_back_jump(
            instructions,
            body_start,
            end_index,
        )
        if back_jump_index is None:
            return None
        repeated_start = find_repeated_condition_start(
            instructions,
            body_start - 1,
            back_jump_index - 1,
        )
        if repeated_start is None or instructions[
            repeated_start
        ].starts_line != find_condition_line(instructions, cursor):
            return None
        return back_jump_index

    def translate_modern_while(
        self,
        instructions: list[Instruction],
        cursor: int,
        body_start: int,
        exit_index: int,
        end_index: int,
        test: ast.expr,
    ) -> int | None:
        # cursor is the first jump of the loop test, body_start follows its last
        if self.version is not None and self.version >= (3, 14):
            return self.translate_top_tested_while(
                instructions, cursor, body_start, exit_index, end_index, test
            )
        back_jump_index = self.find_modern_while_back_jump(
            instructions,
            body_start,
            exit_index,
        )
        if back_jump_index is not None and is_inner_loop_jump(
            instructions, body_start, exit_index, back_jump_index
        ):
            back_jump_index = None
        if back_jump_index is None:
            return self.translate_split_body_while(
                instructions,
                cursor,
                (body_start, exit_index, end_index),
                test,
            )

        if is_backward_conditional_jump(instructions[back_jump_index]):
            repeated_jump_index = back_jump_index
        else:
            previous_jump_index = self.previous_non_ignorable_index(
                instructions,
                back_jump_index - 1,
                body_start - 1,
            )
            if previous_jump_index is None:
                return None
            if closes_another_loop(
                instructions, body_start, back_jump_index, previous_jump_index
            ):
                return None
            repeated_jump_index = previous_jump_index

        repeated_condition_start = find_repeated_condition_start(
            instructions,
            body_start - 1,
            repeated_jump_index,
            find_condition_line(instructions, cursor),
        )
        if repeated_condition_start is None:
            return None
        return self.finish_modern_while(
            instructions,
            cursor,
            (body_start, repeated_condition_start, exit_index, end_index),
            test,
        )

    def translate_top_tested_while(
        self,
        instructions: list[Instruction],
        cursor: int,
        body_start: int,
        exit_index: int,
        end_index: int,
        test: ast.expr,
    ) -> int | None:
        loop_end = self.translate_while_with_detached_tail(
            instructions, cursor, body_start, exit_index, test
        )
        if loop_end is not None:
            return loop_end
        back_jump_index = self.previous_non_ignorable_index(
            instructions,
            exit_index - 1,
            body_start - 1,
        )
        if back_jump_index is None:
            return None
        back_jump = instructions[back_jump_index]
        if ends_in_inner_loop(instructions, cursor, body_start, back_jump_index):
            return self.finish_modern_while(
                instructions,
                cursor,
                (body_start, back_jump_index + 1, exit_index, end_index),
                test,
            )
        if back_jump.opname != "JUMP_BACKWARD" or (
            back_jump.argval
            not in condition_prefix_offsets(instructions, cursor)
            - self.loop_continue_offsets
        ):
            return (
                self.translate_while_with_break_block(
                    instructions, cursor, (body_start, exit_index, end_index), test
                )
                or self.translate_while_closed_by_handler(
                    instructions, cursor, body_start, exit_index, test
                )
                or self.translate_while_closed_inside(
                    instructions, cursor, (body_start, exit_index, end_index), test
                )
            )
        return self.finish_modern_while(
            instructions,
            cursor,
            (body_start, back_jump_index, exit_index, end_index),
            test,
        )

    def translate_while_with_detached_tail(
        self,
        instructions: list[Instruction],
        cursor: int,
        body_start: int,
        exit_index: int,
        test: ast.expr,
    ) -> int | None:
        test_offsets = condition_prefix_offsets(instructions, cursor) - (
            self.loop_continue_offsets
        )
        handlers_start = first_handler_of_range(
            self, instructions, body_start, exit_index
        )
        tail_end = next(
            (
                index
                for index in range(len(instructions) - 1, handlers_start, -1)
                if instructions[index].opname == "JUMP_BACKWARD"
                and instructions[index].argval in test_offsets
            ),
            None,
        )
        if tail_end is None or instructions[tail_end - 1].opname == "POP_EXCEPT":
            return None
        tail_start = handler_join_index(instructions, handlers_start, tail_end)
        if tail_start == tail_end:
            return None
        body: list[ast.stmt] = []
        outer_resume_offsets = self.detached_resume_offsets
        self.detached_resume_offsets |= {int(instructions[tail_start].offset)}
        for start, end in ((body_start, exit_index), (tail_start, tail_end)):
            body.extend(
                self.translate_loop_child_statements(
                    instructions,
                    start,
                    end,
                    continue_offset=int(instructions[cursor].offset),
                    break_index=exit_index,
                    extra_continue_offsets=test_offsets,
                )
            )
        self.detached_resume_offsets = outer_resume_offsets
        self.claimed_ranges[tail_start] = tail_end + 1
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=[])
        )
        return exit_index

    def translate_while_closed_inside(
        self,
        instructions: list[Instruction],
        cursor: int,
        bounds: tuple[int, int, int],
        test: ast.expr,
    ) -> int | None:
        body_start, exit_index, end_index = bounds
        test_offsets = condition_prefix_offsets(instructions, cursor) - (
            self.loop_continue_offsets
        )
        if not is_terminal_op(instructions[exit_index - 1].opname) or not any(
            instruction.opname == "JUMP_BACKWARD" and instruction.argval in test_offsets
            for instruction in instructions[body_start:exit_index]
        ):
            return None
        return self.finish_modern_while(
            instructions, cursor, (body_start, exit_index, exit_index, end_index), test
        )

    def translate_while_closed_by_handler(
        self,
        instructions: list[Instruction],
        cursor: int,
        body_start: int,
        exit_index: int,
        test: ast.expr,
    ) -> int | None:
        test_offsets = condition_prefix_offsets(instructions, cursor) - (
            self.loop_continue_offsets
        )
        closing = [
            index
            for index in range(body_start, len(instructions))
            if instructions[index].opname == "JUMP_BACKWARD"
            and instructions[index].argval in test_offsets
        ]
        if not closing or closing[0] < exit_index:
            return None
        if any(instructions[index - 1].opname != "POP_EXCEPT" for index in closing):
            return None
        spans = read_table_spans(
            self.code, instructions, make_offset_index(instructions)
        )
        if not any(
            body_start <= span.start_index < exit_index <= span.target_index
            and span.target_index <= closing[0]
            for span in spans
        ):
            return None
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            exit_index,
            continue_offset=int(instructions[cursor].offset),
            break_index=exit_index,
            extra_continue_offsets=test_offsets,
        )
        if body and isinstance(body[-1], ast.Try):
            for handler in body[-1].handlers:
                if isinstance(handler.body[-1], ast.Continue):
                    handler.body[-1:] = [] if len(handler.body) > 1 else [ast.Pass()]
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=[])
        )
        return exit_index

    def translate_while_with_break_block(
        self,
        instructions: list[Instruction],
        cursor: int,
        bounds: tuple[int, int, int],
        test: ast.expr,
    ) -> int | None:
        body_start, exit_index, end_index = bounds
        test_offsets = condition_prefix_offsets(instructions, cursor)
        back_jump_index = next(
            (
                index
                for index in range(exit_index - 1, body_start - 1, -1)
                if instructions[index].opname == "JUMP_BACKWARD"
                and instructions[index].argval in test_offsets
            ),
            None,
        )
        if back_jump_index is None:
            return None
        else_end_index = break_block_else_end(
            instructions, back_jump_index + 1, exit_index, end_index
        )
        if else_end_index is None:
            return None
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            back_jump_index,
            continue_offset=int(instructions[cursor].offset),
            break_index=back_jump_index + 1,
            extra_continue_offsets=test_offsets,
        )
        orelse = self.translate_child_statements(
            instructions, exit_index, else_end_index
        )
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=orelse)
        )
        return else_end_index

    def finish_modern_while(
        self,
        instructions: list[Instruction],
        cursor: int,
        bounds: tuple[int, int, int, int],
        test: ast.expr,
    ) -> int:
        body_start, repeated_condition_start, exit_index, end_index = bounds
        break_exit = find_loop_break_exit(
            instructions,
            body_start,
            repeated_condition_start,
            exit_index,
            end_index,
            "NOP",
        )
        else_end_index = break_exit.else_end_index
        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            repeated_condition_start,
            continue_offset=int(instructions[cursor].offset),
            break_index=else_end_index or exit_index,
            extra_continue_offsets=condition_prefix_offsets(instructions, cursor),
            inlined_break_tails=break_exit.inlined_tails,
        )
        orelse: list[ast.stmt] = []
        if else_end_index is None:
            body = self.simplify_modern_loop_break_tail(
                instructions,
                body,
                exit_index,
                end_index,
            )
        else:
            orelse = self.translate_child_statements(
                instructions,
                exit_index,
                else_end_index,
            )
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=orelse)
        )
        self.append_detached_break_tail(instructions, break_exit)
        return else_end_index or exit_index

    def try_translate_while_without_repeated_test(
        self,
        instructions: list[Instruction],
        cursor: int,
        exit_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        test_offsets = condition_prefix_offsets(instructions, cursor)
        back_jumps = [
            index
            for index in range(cursor + 1, min(exit_index, end_index))
            if instructions[index].opname == "JUMP_BACKWARD"
            and instructions[index].argval in test_offsets
            and instructions[index].argval not in self.loop_continue_offsets
        ]
        if not back_jumps or back_jumps[-1] + 1 >= exit_index:
            return None
        if not is_false_jump(instructions[cursor].opname):
            return None
        if self.version is not None and self.version >= (3, 14):
            if ends_in_inner_loop(instructions, cursor, cursor + 1, exit_index - 1):
                return None
            loop_end = self.translate_while_with_break_block(
                instructions, cursor, (cursor + 1, exit_index, end_index), condition
            )
            if loop_end is not None:
                return loop_end
        repeated = self.find_modern_while_back_jump(
            instructions, cursor + 1, exit_index
        )
        if repeated is not None:
            return None
        body = self.translate_loop_child_statements(
            instructions,
            cursor + 1,
            exit_index,
            continue_offset=int(instructions[back_jumps[-1]].argval),
            break_index=exit_index,
        )
        if not body:
            return None
        self.statements.append(
            ast.While(test=condition, body=[*body, ast.Break()], orelse=[])
        )
        return exit_index

    def translate_split_body_while(
        self,
        instructions: list[Instruction],
        cursor: int,
        loop_range: tuple[int, int, int],
        test: ast.expr,
    ) -> int | None:
        body_start, exit_index, end_index = loop_range
        handlers_start = first_handler_of_range(
            self, instructions, body_start, exit_index
        )
        tail = find_split_while_tail(
            instructions, body_start, exit_index, handlers_start
        )
        if tail is None:
            return None
        handlers_end = out_of_line_handlers_end(self.code, instructions, handlers_start)
        if handlers_start < handlers_end <= tail.test_start:
            tail = replace(tail, start=handlers_end)
        after_index = (
            find_loop_else_end(
                instructions, body_start, exit_index, exit_index, end_index
            )
            or exit_index
        )
        body: list[ast.stmt] = []
        outer_resume_offsets = self.detached_resume_offsets
        self.detached_resume_offsets |= {int(instructions[tail.start].offset)}
        for start, end in ((body_start, exit_index), (tail.start, tail.test_start)):
            body.extend(
                self.translate_loop_child_statements(
                    instructions,
                    start,
                    end,
                    continue_offset=int(instructions[cursor].offset),
                    break_index=after_index,
                    extra_continue_offsets=condition_prefix_offsets(
                        instructions, cursor
                    ),
                )
            )
        self.detached_resume_offsets = outer_resume_offsets
        self.claimed_ranges[tail.start] = tail.end
        orelse = self.translate_child_statements(instructions, exit_index, after_index)
        self.statements.append(
            ast.While(test=test, body=body or [ast.Pass()], orelse=orelse)
        )
        return after_index

    def simplify_modern_loop_break_tail(
        self,
        instructions: list[Instruction],
        body: list[ast.stmt],
        exit_index: int,
        end_index: int,
    ) -> list[ast.stmt]:
        tail_end_index = terminal_tail_end_index(instructions, exit_index, end_index)
        if tail_end_index is None:
            return body

        tail = self.translate_child_statements(instructions, exit_index, tail_end_index)
        if not tail:
            return body
        return replace_matching_terminal_if_tail(body, tail)

    def find_modern_while_back_jump(
        self,
        instructions: list[Instruction],
        start_index: int,
        exit_index: int,
    ) -> int | None:
        if start_index >= exit_index:
            return None

        body_index = start_index
        while body_index + 1 < exit_index and instructions[body_index].opname == "NOP":
            body_index += 1
        loop_body_offsets = {
            int(instruction.offset)
            for instruction in instructions[start_index : body_index + 1]
        }
        for index in range(exit_index - 1, start_index - 1, -1):
            instruction = instructions[index]
            if instruction.opname not in {
                "JUMP_ABSOLUTE",
                "JUMP_BACKWARD",
                "JUMP",
            } and not is_backward_conditional_jump(instruction):
                continue
            if int(instruction.argval) in loop_body_offsets:
                return index
        return None

    def try_translate_copied_tail_if_expression(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        test = terminal_guard_for_jump(
            instructions[cursor],
            condition,
            jumps_to_terminal=False,
        )
        return self.translate_copied_tail_if_expression(
            instructions,
            cursor + 1,
            false_index,
            end_index,
            test,
        )

    def translate_copied_tail_if_expression(
        self,
        instructions: list[Instruction],
        true_start: int,
        false_index: int,
        end_index: int,
        test: ast.expr,
    ) -> int | None:
        # python 3.12+ copies a short tail that ends in a return into both
        # branches of a conditional expression instead of joining them. older
        # versions join them with a jump, so there two returns are two statements
        if self.version is not None and self.version < (3, 12):
            return None
        ends = copied_tail_branch_ends(
            instructions,
            true_start - 1,
            false_index,
            end_index,
        )
        if ends is None:
            return None
        true_end, false_end = ends
        if (
            instructions[false_end - 1].offset in self.implicit_return_offsets
            and next_non_ignored_index(instructions, false_index, end_index)
            == false_end - 2
        ):
            return None

        tail_limit = common_tail_length(instructions, true_end, false_end)
        for tail_length in range(1, tail_limit + 1):
            true_value = self.evaluate_single_value_range(
                instructions,
                true_start,
                true_end - tail_length,
            )
            if true_value is None:
                continue
            false_value = self.evaluate_single_value_range(
                instructions,
                false_index,
                false_end - tail_length,
            )
            if false_value is None:
                continue
            self.stack.append(ast.IfExp(test=test, body=true_value, orelse=false_value))
            return false_end - tail_length
        return None

    def translate_if_expression_branches(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        true_start: int,
        false_index: int,
        end_index: int,
        test: ast.expr,
    ) -> int | None:
        jump_index = previous_non_ignored_index(
            instructions,
            false_index - 1,
            true_start,
        )
        if jump_index is None:
            return None
        jump = instructions[jump_index]
        if jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return self.translate_copied_tail_if_expression(
                instructions,
                true_start,
                false_index,
                end_index,
                test,
            )

        join_index = offset_to_index.get(int(jump.argval))
        if join_index is None or not false_index < join_index <= end_index:
            return None
        true_value = self.evaluate_single_value_range(
            instructions,
            true_start,
            jump_index,
        )
        false_value = self.evaluate_single_value_range(
            instructions,
            false_index,
            join_index,
        )
        if true_value is None or false_value is None:
            return None
        self.stack.append(ast.IfExp(test=test, body=true_value, orelse=false_value))
        return join_index

    def evaluate_single_value_range(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> ast.expr | None:
        if not is_closed_expression_range(instructions, start_index, end_index):
            return None
        child = self.make_child()
        statements = child.translate_range(instructions, start_index, end_index)
        if statements or len(child.stack) != len(self.stack) + 1:
            return None
        if any(kept is not own for kept, own in zip(child.stack, self.stack)):
            return None
        self.warnings.extend(child.warnings)
        return coerce_expr(child.stack[-1])

    def try_translate_if_expression(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        jump_index = false_index - 1
        if jump_index <= cursor:
            return None
        jump = instructions[jump_index]
        if jump.opname not in {"JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP"}:
            inner_jump_index = jump_over_else_inside(
                self, instructions, cursor, false_index
            )
            if inner_jump_index is None:
                return None
            jump = instructions[inner_jump_index]
            jump_index = false_index
        join_index = offset_to_index.get(int(jump.argval))
        if join_index is None or join_index <= false_index or join_index > end_index:
            return None

        condition, true_start = self.combine_short_circuit_condition(
            instructions,
            cursor,
            jump_index,
            false_index,
            condition,
        )
        true_value = self.evaluate_expression_range(
            instructions, true_start, jump_index
        )
        if true_value is None:
            return None
        false_value = self.evaluate_expression_range(
            instructions, false_index, join_index
        )
        if false_value is None:
            return None

        if not is_false_jump(instructions[cursor].opname):
            condition = invert_condition(condition)
        expression = ast.IfExp(test=condition, body=true_value, orelse=false_value)
        self.stack.append(expression)
        return join_index

    def combine_short_circuit_condition(
        self,
        instructions: list[Instruction],
        cursor: int,
        jump_index: int,
        false_index: int,
        condition: ast.expr,
    ) -> tuple[ast.expr, int]:
        for index in range(jump_index - 1, cursor, -1):
            candidate = instructions[index]
            if not is_forward_conditional_jump(candidate):
                continue
            if int(candidate.argval) != instructions[false_index].offset:
                continue
            if not simple_condition_range(
                instructions,
                cursor + 1,
                index,
                self.version,
            ):
                continue
            right = self.evaluate_expression_range(instructions, cursor + 1, index)
            if right is None:
                return condition, cursor + 1
            right = condition_from_jump(candidate.opname, right)
            first_on_false = is_false_jump(instructions[cursor].opname)
            if first_on_false != is_false_jump(candidate.opname):
                right = invert_condition(right)
            if first_on_false:
                return ast.BoolOp(op=ast.And(), values=[condition, right]), index + 1
            return ast.BoolOp(op=ast.Or(), values=[condition, right]), index + 1
        return condition, cursor + 1

    def jump_over_else(
        self,
        instructions: list[Instruction],
        cursor: int,
        false_index: int,
    ) -> tuple[Instruction, int] | None:
        jump_index = false_index - 1
        if jump_index <= cursor:
            return None
        jump = instructions[jump_index]
        loop_exits = self.loop_break_offsets | self.loop_continue_offsets
        if jump.opname in {"JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP"} and (
            jump.argval not in loop_exits
        ):
            return jump, jump_index
        inner_jump_index = jump_over_else_inside(
            self, instructions, cursor, false_index
        )
        if inner_jump_index is None:
            return None
        return instructions[inner_jump_index], false_index

    def translate_first_branch(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        join_offset: int,
    ) -> list[ast.stmt]:
        child = self.make_child()
        last = instructions[end_index - 1]
        if last.opname == "JUMP_BACKWARD" and (
            last.argval not in self.loop_continue_offsets
        ):
            child.loop_break_offsets = self.loop_break_offsets | {join_offset}
        return child.translate_range(instructions, start_index, end_index)

    def try_translate_if_else_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        found = self.jump_over_else(instructions, cursor, false_index)
        if found is None:
            return None
        jump, jump_index = found
        join_index = offset_to_index.get(int(jump.argval))
        if join_index is None or join_index <= false_index:
            return None
        if jump.argval in self.loop_break_offsets:
            return None
        if join_index > end_index:
            join_index = end_index

        statement_condition = self.combine_simple_statement_condition_prefix(
            instructions,
            offset_to_index,
            cursor,
            jump_index,
            false_index,
            condition,
        )
        if statement_condition is None:
            condition, true_start = self.combine_short_circuit_condition(
                instructions,
                cursor,
                jump_index,
                false_index,
                condition,
            )
            test = (
                condition
                if is_false_jump(instructions[cursor].opname)
                else invert_condition(condition)
            )
        else:
            test, true_start = statement_condition

        true_body = self.translate_first_branch(
            instructions, true_start, jump_index, int(jump.argval)
        )
        if has_test_failing_into(instructions, true_start, jump_index, false_index):
            folded = fold_test_failing_into_else(test, true_body)
            if folded is None:
                return None
            test, true_body = folded
        false_body = self.translate_child_statements(
            instructions, false_index, join_index
        )
        if not true_body and not false_body:
            return None

        self.statements.append(
            ast.If(
                test=test,
                body=true_body or [ast.Pass()],
                orelse=false_body,
            )
        )
        return join_index

    def combine_simple_statement_condition_prefix(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        jump_index: int,
        false_index: int,
        condition: ast.expr,
    ) -> tuple[ast.expr, int] | None:
        prefix_jumps = statement_condition_prefix_jumps(
            instructions,
            offset_to_index,
            cursor,
            jump_index,
            false_index,
            self.version,
        )
        if prefix_jumps is None:
            return None

        false_offset = instructions[false_index].offset
        body_start = statement_condition_body_start(
            instructions,
            offset_to_index,
            prefix_jumps,
            false_offset,
        )
        if body_start is None:
            return None

        current = (
            condition
            if is_false_jump(instructions[cursor].opname)
            else invert_condition(condition)
        )
        scan_start = cursor + 1
        body_offset = instructions[body_start].offset
        pieces: list[tuple[Instruction, ast.expr]] = []
        for index in prefix_jumps:
            jump = instructions[index]
            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
                return None
            expr = self.evaluate_expression_range(instructions, scan_start, index)
            if expr is None:
                return None
            expr = condition_from_jump(jump.opname, expr)
            if jump.argval not in {false_offset, body_offset}:
                return None
            pieces.append((jump, expr))
            scan_start = index + 1

        suffix = make_statement_condition_suffix(
            pieces,
            false_offset,
            body_offset,
        )
        return make_bool_and(current, suffix), body_start

    def try_translate_terminal_if_else_statement(
        self,
        instructions: list[Instruction],
        cursor: int,
        false_index: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        if self.loop_none_return_is_break:
            return None
        true_return_start = generated_none_return_region_ending_at(
            instructions,
            false_index,
            cursor + 1,
        )
        if true_return_start is None or self.loop_continue_offsets:
            return None

        false_return_start = none_return_region_ending_at(
            instructions,
            end_index,
            false_index,
        )
        if false_return_start is None and self.ends_before_module_return(
            instructions, end_index
        ):
            false_return_start = end_index
        if false_return_start is None or false_return_start <= false_index:
            return None

        true_body = self.translate_child_statements(
            instructions,
            cursor + 1,
            true_return_start,
        )
        false_body = self.translate_child_statements(
            instructions,
            false_index,
            false_return_start,
        )
        if not true_body and not false_body:
            return None

        test = (
            condition
            if is_false_jump(instructions[cursor].opname)
            else invert_condition(condition)
        )
        if not true_body:
            self.statements.append(
                ast.If(test=test, body=[ast.Return(value=None)], orelse=[])
            )
            return false_index
        self.statements.append(
            ast.If(
                test=test,
                body=true_body,
                orelse=false_body or [ast.Pass()],
            )
        )
        return end_index

    def module_return_length(self) -> int:
        # RETURN_CONST None, or LOAD_CONST None and RETURN_VALUE on python 3.14
        return 2 if self.version is not None and self.version >= (3, 14) else 1

    def ends_before_module_return(
        self,
        instructions: list[Instruction],
        end_index: int,
    ) -> bool:
        if end_index >= len(instructions):
            return False
        if getattr(self.code, "co_name", "") != "<module>":
            return False
        last = instructions[end_index]
        if self.version is not None and self.version >= (3, 14):
            return (
                end_index + 1 < len(instructions)
                and last.opname == "LOAD_CONST"
                and last.argval is None
                and last.arg is not None
                and instructions[end_index + 1].opname == "RETURN_VALUE"
            )
        return last.opname == "RETURN_CONST" and last.argval is None

    def translate_child_statements(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> list[ast.stmt]:
        child = self.make_child()
        return child.translate_range(instructions, start_index, end_index)

    def evaluate_expression_range(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> ast.expr | None:
        child = self.make_child()
        child.evaluating_test_operand = True
        child.translate_range(instructions, start_index, end_index)
        if not child.stack:
            return None
        self.warnings.extend(child.warnings)
        return coerce_expr(child.stack[-1])

    def try_translate_loop_cleanup_return(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        cleanup_start = find_loop_cleanup_return_start(
            instructions,
            cursor,
            end_index,
            self.claimed_ranges,
        )
        if cleanup_start is None:
            return None

        return_index = loop_cleanup_return_end(
            instructions, cleanup_start, end_index, self.claimed_ranges
        )
        if return_index is None:
            return None

        value = self.evaluate_single_value_range(instructions, cursor, cleanup_start)
        if value is None:
            return None

        self.statements.append(ast.Return(value=value))
        return return_index + 1


def ends_in_return_line(instructions: list[Instruction], region: WithRegion) -> bool:
    last = instructions[region.body_end_index - 1]
    return last.opname == "NOP" and last.starts_line is not None


def tests_in_front_of_while(
    instructions: list[Instruction],
    cursor: int,
    body_start: int,
    exit_index: int,
) -> bool:
    closing = instructions[exit_index - 1]
    if not closing.opname.startswith("POP_JUMP_BACKWARD_IF_") or (
        closing.argval != instructions[body_start].offset
    ):
        return False
    head_index, copy_index = body_start - 1, exit_index - 1
    while head_index >= cursor:
        head, copy = instructions[head_index], instructions[copy_index]
        both_test = "POP_JUMP" in head.opname and "POP_JUMP" in copy.opname
        if not both_test and (head.opname, head.argval) != (copy.opname, copy.argval):
            return True
        head_index -= 1
        copy_index -= 1
    return False


def jump_over_else_inside(
    decompiler: ControlRecoveryMixin,
    instructions: list[Instruction],
    cursor: int,
    false_index: int,
) -> int | None:
    last = instructions[false_index - 1].opname
    loop_exits = decompiler.loop_break_offsets | decompiler.loop_continue_offsets
    if last == "JUMP_FORWARD":
        if instructions[false_index - 1].argval not in loop_exits:
            return None
    elif not is_terminal_op(last) and last != "JUMP_BACKWARD":
        return None
    false_offset = instructions[false_index].offset
    # up to 3.12 a test in the body that would arrive at such a jump goes
    # where the jump goes, and the jump itself is gone
    threaded = decompiler.version is not None and decompiler.version < (3, 13)
    for index in range(false_index - 2, cursor, -1):
        jump = instructions[index]
        if jump.argval in loop_exits or not (
            jump.opname == "JUMP_FORWARD"
            or (threaded and is_forward_conditional_jump(jump))
        ):
            continue
        if jump.argval > false_offset:
            return index
    if last != "JUMP_BACKWARD":
        return None
    return handler_jump_over_else(decompiler, instructions, cursor, false_index)


def handler_jump_over_else(
    decompiler: ControlRecoveryMixin,
    instructions: list[Instruction],
    cursor: int,
    false_index: int,
) -> int | None:
    spans = read_table_spans(
        decompiler.code, instructions, make_offset_index(instructions)
    )
    regions = handler_regions(instructions, spans)
    false_offset = instructions[false_index].offset
    loop_exits = decompiler.loop_break_offsets | decompiler.loop_continue_offsets
    for span in spans:
        handler_index = span.target_index
        if not cursor < span.start_index < false_index <= handler_index:
            continue
        for index in range(handler_index, regions.get(handler_index, handler_index)):
            jump = instructions[index]
            if jump.opname not in {"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"}:
                continue
            if instructions[index - 1].opname != "POP_EXCEPT":
                continue
            if false_offset < jump.argval and jump.argval not in loop_exits:
                return index
    return None


def else_jump_index(
    decompiler: ControlRecoveryMixin,
    instructions: list[Instruction],
    body_start: int,
    body_end: int,
) -> int | None:
    if body_end >= len(instructions):
        return None
    index = previous_non_ignored_index(instructions, body_end - 1, body_start)
    if index is None:
        return None
    jump = instructions[index]
    if jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
        return None
    if not isinstance(jump.argval, int):
        return None
    if jump.argval in decompiler.loop_break_offsets:
        return None
    if jump.argval in decompiler.loop_continue_offsets:
        return None
    if jump.argval <= instructions[body_end].offset:
        return None
    return index


CONDITIONAL_HANDLER_ORDER = (
    "try_translate_while_without_repeated_test",
    "try_translate_copied_tail_if_expression",
    "try_translate_continue_exit_while_loop",
    "try_translate_compound_guard_body_statement",
    "try_translate_for_tail_if_statement",
    "try_translate_loop_continue_guard_statement",
    "try_translate_loop_continue_body_statement",
    "try_translate_chained_comparison",
    "try_translate_chained_comparison_before_else",
    "try_translate_modern_while_loop",
    "try_translate_terminal_guard_statement",
    "try_translate_prefixed_guard_body_statement",
    "try_translate_disjunctive_guard_body_statement",
    "try_translate_targeted_prefixed_guard_body_statement",
    "try_translate_targeted_if_expression_assignment",
    "try_translate_guard_body_statement",
    "try_translate_if_expression",
    "try_translate_clipped_if_expression_return",
    "try_translate_if_else_statement",
    "try_translate_terminal_if_else_statement",
)

CONDITIONAL_HANDLERS_WITHOUT_OFFSET = frozenset(
    {
        "try_translate_while_without_repeated_test",
        "try_translate_continue_exit_while_loop",
        "try_translate_loop_continue_guard_statement",
        "try_translate_modern_while_loop",
        "try_translate_terminal_guard_statement",
        "try_translate_terminal_if_else_statement",
    }
)
CONDITIONAL_HANDLERS_WITHOUT_END = frozenset(
    {"try_translate_prefixed_guard_body_statement"}
)


def call_conditional_handler(
    decompiler: ControlRecoveryMixin,
    handler_name: str,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    target_index: int,
    end_index: int,
    condition: ast.expr,
) -> int | None:
    handler = getattr(decompiler, handler_name)
    if handler_name in CONDITIONAL_HANDLERS_WITHOUT_OFFSET:
        return handler(instructions, cursor, target_index, end_index, condition)
    if handler_name in CONDITIONAL_HANDLERS_WITHOUT_END:
        return handler(instructions, offset_to_index, cursor, target_index, condition)
    return handler(
        instructions,
        offset_to_index,
        cursor,
        target_index,
        end_index,
        condition,
    )


def find_send_value_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    if cursor + 5 >= end_index:
        return None
    if instructions[cursor].opname not in {"GET_AWAITABLE", "GET_YIELD_FROM_ITER"}:
        return None

    load_none_index = skip_yield_from_prefix(instructions, cursor + 1, end_index)
    if load_none_index >= end_index or not is_none_load(instructions[load_none_index]):
        return None

    send_index = next_prefixed_opcode(
        instructions, load_none_index + 1, end_index, "SEND"
    )
    if send_index is None:
        return None

    jump_index = find_send_value_jump(instructions, send_index, end_index)
    if jump_index is None:
        return None
    return send_value_end_index(
        instructions,
        offset_to_index,
        send_index,
        jump_index,
        end_index,
    )


def next_prefixed_opcode(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
    opname: str,
) -> int | None:
    index = skip_yield_from_prefix(instructions, cursor, end_index)
    if index >= end_index or instructions[index].opname != opname:
        return None
    return index


def find_send_value_jump(
    instructions: list[Instruction],
    send_index: int,
    end_index: int,
) -> int | None:
    yield_index = next_prefixed_opcode(
        instructions, send_index + 1, end_index, "YIELD_VALUE"
    )
    if yield_index is None:
        return None

    resume_index = next_prefixed_opcode(
        instructions, yield_index + 1, end_index, "RESUME"
    )
    if resume_index is None:
        return None
    return next_prefixed_opcode(
        instructions,
        resume_index + 1,
        end_index,
        "JUMP_BACKWARD_NO_INTERRUPT",
    )


def send_value_end_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    send_index: int,
    jump_index: int,
    end_index: int,
) -> int | None:
    jump = instructions[jump_index]
    if int(jump.argval) != instructions[send_index].offset:
        return None

    end_send_index = offset_to_index.get(int(instructions[send_index].argval))
    if end_send_index is None or end_send_index <= jump_index:
        return None
    if is_send_loop_without_end(instructions, jump_index, end_send_index):
        return jump_index
    if end_send_index >= end_index:
        return None
    if instructions[end_send_index].opname != "END_SEND":
        return None
    return end_send_index


def is_send_loop_without_end(
    instructions: list[Instruction],
    jump_index: int,
    target_index: int,
) -> bool:
    if target_index != jump_index + 1:
        return False
    if target_index >= len(instructions):
        return True
    return instructions[target_index].opname != "END_SEND"


def skip_yield_from_prefix(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    while cursor < end_index and instructions[cursor].opname in {
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
    }:
        cursor += 1
    return cursor


def find_loop_cleanup_return_start(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
    claimed_ranges: dict[int, int],
) -> int | None:
    if is_expression_range_boundary_op(instructions[cursor].opname):
        return None

    for index in range(cursor + 1, end_index):
        opname = instructions[index].opname
        if opname != "SWAP":
            if is_expression_range_boundary_op(opname):
                return None
            continue
        cleanup_end = loop_cleanup_return_end(
            instructions, index, end_index, claimed_ranges
        )
        return None if cleanup_end is None else index
    return None


def is_expression_range_boundary_op(opname: str) -> bool:
    return "JUMP" in opname or opname in {
        "FOR_ITER",
        "GET_ITER",
        "GET_AITER",
        "SEND",
        "RETURN_CONST",
        "RETURN_VALUE",
        "RAISE_VARARGS",
        "RERAISE",
    }


def retry_loop_start_index(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int | None:
    loop_start = cursor
    while loop_start < end_index and instructions[loop_start].opname in {
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
    }:
        loop_start += 1
    if loop_start >= end_index:
        return None
    if not instructions[loop_start].is_jump_target:
        return None
    return loop_start


def find_backward_retry_loop(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    loop_start: int,
    end_index: int,
) -> tuple[int, int, int] | None:
    loop_start_offset = int(instructions[loop_start].offset)
    scan_end = min(end_index, loop_start + 256)
    for condition_index in range(loop_start + 1, scan_end):
        condition = instructions[condition_index]
        if not is_forward_conditional_jump(condition):
            continue
        retry_jump_index = offset_to_index.get(int(condition.argval))
        if retry_jump_index is None or retry_jump_index <= condition_index:
            continue
        retry_jump = instructions[retry_jump_index]
        if retry_jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_BACKWARD"}:
            continue
        if int(retry_jump.argval) != loop_start_offset:
            continue
        if has_jump_out_of_range(instructions, loop_start, condition_index):
            continue
        after_index = retry_loop_after_index(
            instructions,
            offset_to_index,
            condition_index,
            retry_jump_index,
            end_index,
        )
        if after_index is None:
            continue
        return condition_index, retry_jump_index, after_index
    return None


def has_jump_out_of_range(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    end_offset = instructions[end_index].offset
    return any(
        is_forward_conditional_jump(instruction) and instruction.argval > end_offset
        for instruction in instructions[start_index:end_index]
    )


def retry_loop_after_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    condition_index: int,
    retry_jump_index: int,
    end_index: int,
) -> int | None:
    after_index = retry_jump_index + 1
    fallthrough_index = condition_index + 1
    if fallthrough_index < retry_jump_index:
        fallthrough = instructions[fallthrough_index]
        if fallthrough.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return None
        after_index = offset_to_index.get(int(fallthrough.argval), after_index)
    while after_index < end_index and instructions[after_index].opname in {
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
    }:
        after_index += 1
    if after_index <= retry_jump_index or after_index > end_index:
        return None
    return after_index


def loop_cleanup_return_end(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
    claimed_ranges: dict[int, int],
) -> int | None:
    saw_cleanup = False
    while cursor + 1 < end_index:
        if instructions[cursor].opname != "SWAP":
            break
        if instructions[cursor + 1].opname != "POP_TOP":
            break
        saw_cleanup = True
        cursor += 2
    cursor = claimed_ranges.get(cursor, cursor)

    if not saw_cleanup or cursor >= end_index:
        return None
    if instructions[cursor].opname != "RETURN_VALUE":
        return None
    return cursor


def make_send_value_expression(opname: str, value: ast.expr) -> ast.expr:
    if getattr(value, AWAITED_BY_CALLER_MARK, False):
        return value
    if opname == "GET_AWAITABLE":
        return ast.Await(value=value)
    return ast.YieldFrom(value=value)


def terminal_block_end_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname in {
            "RETURN_CONST",
            "RETURN_VALUE",
            "RAISE_VARARGS",
        }:
            return index + 1
    return None


def targeted_prefix_jumps(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    body_start: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> tuple[list[int], int] | None:
    body_offset = instructions[body_start].offset
    skip_index: int | None = None
    prefix_jumps: list[int] = []
    scan_start = cursor + 1
    for index in range(cursor + 1, body_start):
        instruction = instructions[index]
        if not is_forward_conditional_jump(instruction):
            continue
        if not simple_condition_range(instructions, scan_start, index, version):
            return None
        target_index = offset_to_index.get(int(instruction.argval))
        if target_index is None:
            return None
        if instruction.argval != body_offset:
            if target_index <= body_start or target_index > end_index:
                return None
            if skip_index is None:
                skip_index = target_index
            elif skip_index != target_index:
                return None
        prefix_jumps.append(index)
        scan_start = index + 1

    if (
        not prefix_jumps
        or skip_index is None
        or not has_only_ignored_instructions(instructions, scan_start, body_start)
    ):
        return None
    return prefix_jumps, skip_index


def targeted_if_expression_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    true_index: int,
    end_index: int,
) -> TargetedIfExpressionShape | None:
    jump_index = previous_non_ignored_index(
        instructions,
        true_index - 1,
        cursor + 1,
    )
    if jump_index is None:
        return None
    jump = instructions[jump_index]
    if not is_forward_conditional_jump(jump):
        return None

    false_index = offset_to_index.get(int(jump.argval))
    if false_index is None or false_index <= true_index or false_index >= end_index:
        return None
    join_jump_index = previous_non_ignored_index(
        instructions,
        false_index - 1,
        true_index,
    )
    if join_jump_index is None:
        return None
    join_jump = instructions[join_jump_index]
    if join_jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
        return None

    store_index = offset_to_index.get(int(join_jump.argval))
    if (
        store_index is None
        or store_index != false_index + 1
        or store_index >= end_index
    ):
        return None
    if instructions[store_index].opname not in {
        "STORE_DEREF",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_NAME",
    }:
        return None
    return TargetedIfExpressionShape(
        jump_index,
        false_index,
        join_jump_index,
        store_index,
    )


def terminal_guard_for_jump(
    instruction: Instruction,
    condition: ast.expr,
    jumps_to_terminal: bool,
) -> ast.expr:
    if jumps_to_terminal:
        if is_false_jump(instruction.opname):
            return invert_condition(condition)
        return condition

    if is_false_jump(instruction.opname):
        return condition
    return invert_condition(condition)


def statement_condition_prefix_jumps(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    jump_index: int,
    false_index: int,
    version: tuple[int, ...] | None,
) -> list[int] | None:
    jumps: list[int] = []
    false_offset = instructions[false_index].offset
    body_start: int | None = None
    scan_start = cursor + 1
    for index in range(cursor + 1, jump_index):
        instruction = instructions[index]
        if not is_forward_conditional_jump(instruction):
            continue
        if not simple_condition_range(instructions, scan_start, index, version):
            return None

        target_index = offset_to_index.get(int(instruction.argval))
        target = next_statement_condition_body_start(
            instructions,
            cursor,
            jump_index,
            instruction,
            target_index,
            false_offset,
            body_start,
        )
        if not target[0]:
            return None
        body_start = target[1]
        jumps.append(index)
        scan_start = index + 1

    if not jumps:
        return None
    if body_start is None:
        return jumps
    return [index for index in jumps if index < body_start]


def next_statement_condition_body_start(
    instructions: list[Instruction],
    cursor: int,
    jump_index: int,
    instruction: Instruction,
    target_index: int | None,
    false_offset: int,
    body_start: int | None,
) -> tuple[bool, int | None]:
    if target_index is None:
        return False, body_start
    if instruction.argval == false_offset:
        return True, body_start
    if target_index <= cursor or target_index >= jump_index:
        return False, body_start
    if target_follows_branch_body(instructions, cursor, target_index):
        return False, body_start
    if body_start is None or target_index < body_start:
        body_start = target_index
    return True, body_start


def target_follows_branch_body(
    instructions: list[Instruction],
    cursor: int,
    target_index: int,
) -> bool:
    previous_index = previous_non_ignored_index(instructions, target_index - 1, cursor)
    if previous_index is None:
        return False
    return not is_forward_conditional_jump(instructions[previous_index])


def statement_condition_body_start(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    jumps: list[int],
    false_offset: int,
) -> int | None:
    body_start: int | None = None
    for index in jumps:
        jump = instructions[index]
        if jump.argval == false_offset:
            continue
        target_index = offset_to_index.get(int(jump.argval))
        if target_index is None:
            return None
        if body_start is None or target_index < body_start:
            body_start = target_index
    if body_start is not None:
        return body_start
    return jumps[-1] + 1


SIMPLE_CONDITION_OPS = {
    "CACHE",
    "COPY",
    "LOAD_ATTR",
    "LOAD_CONST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "LOAD_GLOBAL",
    "LOAD_NAME",
    "LOAD_NULL",
    "LOAD_SMALL_INT",
    "BINARY_OP",
    "BINARY_SLICE",
    "BUILD_LIST",
    "BUILD_SLICE",
    "BUILD_STRING",
    "BUILD_TUPLE",
    "CALL",
    "CALL_KW",
    "CHAINED_COMPARE_LINK",
    "COMPARE_OP",
    "CONTAINS_OP",
    "CONVERT_VALUE",
    "FORMAT_SIMPLE",
    "FORMAT_VALUE",
    "FORMAT_WITH_SPEC",
    "IS_OP",
    "KW_NAMES",
    "LOAD_METHOD",
    "PRECALL",
    "PUSH_NULL",
    "SWAP",
    "TO_BOOL",
    "END_SEND",
    "GET_AWAITABLE",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "SEND",
    "YIELD_VALUE",
}

MAX_SIMPLE_CONDITION_OPS = 16


def simple_condition_range(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> bool:
    if end_index <= start_index:
        return False
    meaningful = 0
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
        if not is_simple_condition_op(
            instruction.opname
        ) and not stores_named_expression(instructions, index, start_index):
            return False
        if instruction.opname not in {"CACHE", "EXTENDED_ARG"}:
            meaningful += 1
    if not 0 < meaningful <= MAX_SIMPLE_CONDITION_OPS:
        return False
    return condition_range_has_clean_stack_effect(
        instructions,
        start_index,
        end_index,
        version,
    )


NAMED_EXPRESSION_STORE_OPS = frozenset(
    {"STORE_DEREF", "STORE_FAST", "STORE_GLOBAL", "STORE_NAME"}
)


def stores_named_expression(
    instructions: list[Instruction],
    index: int,
    start_index: int,
) -> bool:
    if instructions[index].opname not in NAMED_EXPRESSION_STORE_OPS:
        return False
    if index == start_index:
        return False
    copy = instructions[index - 1]
    return copy.opname == "COPY" and copy.arg == 1


def is_simple_condition_op(opname: str) -> bool:
    return opname in SIMPLE_CONDITION_OPS or opname.startswith(("BINARY_", "UNARY_"))


def condition_range_has_clean_stack_effect(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> bool:
    depth = 0
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
        effect = instruction_stack_effect(instruction, version)
        if effect is None:
            return False
        if depth < effect.pops:
            return False
        depth += effect.net
    return depth == 1


def make_bool_and(left: ast.expr, right: ast.expr) -> ast.expr:
    if isinstance(left, ast.Constant) and left.value is True:
        return right
    if isinstance(right, ast.Constant) and right.value is True:
        return left
    values: list[ast.expr] = []
    if isinstance(left, ast.BoolOp) and isinstance(left.op, ast.And):
        values.extend(left.values)
    else:
        values.append(left)
    if isinstance(right, ast.BoolOp) and isinstance(right.op, ast.And):
        values.extend(right.values)
    else:
        values.append(right)
    return ast.BoolOp(op=ast.And(), values=values)


def make_statement_condition_suffix(
    pieces: list[tuple[Instruction, ast.expr]],
    false_offset: int,
    body_offset: int,
) -> ast.expr:
    condition: ast.expr = ast.Constant(value=True)
    for jump, expr in reversed(pieces):
        taken = terminal_guard_for_jump(jump, expr, jumps_to_terminal=True)
        fallthrough = terminal_guard_for_jump(jump, expr, jumps_to_terminal=False)
        if jump.argval == false_offset:
            condition = make_bool_and(fallthrough, condition)
        elif jump.argval == body_offset:
            condition = make_bool_or([taken, make_bool_and(fallthrough, condition)])
    return condition


def combine_branch_success(
    taken_condition: ast.expr,
    taken_success: ast.expr,
    fallthrough_condition: ast.expr,
    fallthrough_success: ast.expr,
) -> ast.expr:
    before_shared = values_before_shared_tail(taken_success, fallthrough_success)
    if before_shared:
        other = before_shared[0]
        if len(before_shared) > 1:
            other = ast.BoolOp(op=ast.And(), values=before_shared)
        return make_bool_and(make_bool_or([taken_condition, other]), taken_success)
    taken = make_success_branch(taken_condition, taken_success)
    fallthrough = make_success_branch(fallthrough_condition, fallthrough_success)
    if is_false_constant(taken):
        return fallthrough
    if is_false_constant(fallthrough):
        return taken
    if is_true_constant(taken) or is_true_constant(fallthrough):
        return ast.Constant(value=True)
    return make_bool_or([taken, fallthrough])


def values_before_shared_tail(
    taken_success: ast.expr,
    fallthrough_success: ast.expr,
) -> list[ast.expr]:
    if isinstance(taken_success, ast.Constant):
        return []
    tail = and_values(taken_success)
    values = and_values(fallthrough_success)
    if len(values) <= len(tail):
        return []
    if any(
        left is not right
        for left, right in zip(values[-len(tail) :], tail, strict=True)
    ):
        return []
    return values[: -len(tail)]


def and_values(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
        return value.values
    return [value]


def make_success_branch(condition: ast.expr, success: ast.expr) -> ast.expr:
    if is_true_constant(success):
        return condition
    if is_false_constant(success):
        return ast.Constant(value=False)
    return make_bool_and(condition, success)


def is_true_constant(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def is_false_constant(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is False


def make_bool_or(values: list[ast.expr]) -> ast.expr:
    if not values:
        return ast.Constant(value=True)
    if len(values) == 1:
        return values[0]
    merged: list[ast.expr] = []
    for value in values:
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
            merged.extend(value.values)
        else:
            merged.append(value)
    return ast.BoolOp(op=ast.Or(), values=merged)


def find_disjunctive_guard_jumps(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    body_start: int,
    end_index: int,
) -> tuple[frozenset[int], int] | None:
    conditional_jumps = [cursor]
    skip_index: int | None = None
    for index in range(cursor + 1, body_start):
        instruction = instructions[index]
        if not is_forward_conditional_jump(instruction):
            continue

        target_index = offset_to_index.get(int(instruction.argval))
        if target_index is None:
            return None
        if target_index == body_start or index < target_index < body_start:
            conditional_jumps.append(index)
            continue
        if target_index > body_start:
            if target_index > end_index:
                return None
            if skip_index is not None and skip_index != target_index:
                return None
            skip_index = target_index
            conditional_jumps.append(index)
            continue
        if "OR_POP" not in instruction.opname:
            return None

    if len(conditional_jumps) < 2 or skip_index is None:
        return None
    return frozenset(conditional_jumps), skip_index


def compound_condition_jumps(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
    version: tuple[int, ...] | None,
    on_any_line: bool = False,
) -> tuple[list[int], int] | None:
    jumps = compound_condition_jump_run(
        instructions, cursor, end_index, version, on_any_line
    )
    for count in range(len(jumps), 1, -1):
        candidate = jumps[:count]
        skip_index = compound_condition_skip_index(
            instructions,
            offset_to_index,
            candidate,
            end_index,
            returns_are_one=on_any_line,
        )
        if skip_index is not None:
            return candidate, skip_index
    return None


LOOP_BACK = -1


def reduce_condition_tests(
    tests: list[ConditionTest],
    body_start: int,
    skip_index: int,
) -> ast.expr | None:
    while len(tests) > 1:
        for position in range(len(tests) - 1):
            merged = merge_condition_tests(tests, position)
            if merged is not None:
                tests[position : position + 2] = [merged]
                break
        else:
            return None

    test = tests[0]
    if (test.when_true, test.when_false) == (body_start, skip_index):
        return test.expression
    if (test.when_true, test.when_false) == (skip_index, body_start):
        return invert_condition(test.expression)
    return None


def merge_condition_tests(
    tests: list[ConditionTest],
    position: int,
) -> ConditionTest | None:
    first = tests[position]
    second = tests[position + 1]
    if any(
        second.start_index in (test.when_true, test.when_false)
        for test in tests
        if test is not first
    ):
        return None

    if first.when_true == first.end_index:
        if second.when_false == first.when_false:
            values = [first.expression, second.expression]
            exits = (second.when_true, first.when_false)
        elif second.when_true == first.when_false:
            values = [first.expression, invert_condition(second.expression)]
            exits = (second.when_false, first.when_false)
        else:
            return None
        expression = make_bool_and(*values)
    elif first.when_false == first.end_index:
        if second.when_true == first.when_true:
            values = [first.expression, second.expression]
            exits = (first.when_true, second.when_false)
        elif second.when_false == first.when_true:
            values = [first.expression, invert_condition(second.expression)]
            exits = (first.when_true, second.when_true)
        else:
            return None
        expression = make_bool_or(values)
    else:
        return None
    return ConditionTest(first.start_index, second.end_index, expression, *exits)


def jumps_to_body(instructions: list[Instruction], cursor: int, end_index: int) -> bool:
    target = instructions[cursor].argval
    return any(
        instructions[index].offset == target
        and is_forward_conditional_jump(instructions[index - 1])
        for index in range(cursor + 2, end_index)
    )


def is_await_jump(instructions: list[Instruction], index: int) -> bool:
    return (
        instructions[index].opname == "JUMP_BACKWARD_NO_INTERRUPT"
        and index + 1 < len(instructions)
        and instructions[index + 1].opname == "END_SEND"
    )


def compound_condition_jump_run(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
    version: tuple[int, ...] | None,
    on_any_line: bool = False,
) -> list[int]:
    condition_line = find_condition_line(instructions, cursor)
    jumps = [cursor]
    scan_start = cursor + 1
    while True:
        jump_index = next(
            (
                index
                for index in range(scan_start, end_index)
                if "JUMP" in instructions[index].opname
                and not is_await_jump(instructions, index)
            ),
            None,
        )
        if jump_index is None:
            return jumps
        if not is_forward_conditional_jump(instructions[jump_index]):
            return jumps
        if not simple_condition_range(instructions, scan_start, jump_index, version):
            return jumps
        line = instructions[scan_start].starts_line
        same_exit = instructions[jump_index].argval == instructions[cursor].argval
        on_other_line = line is not None and line != condition_line
        if (
            on_other_line
            and same_exit
            and not on_any_line
            and not jumps_to_body(instructions, cursor, end_index)
        ):
            return jumps
        jumps.append(jump_index)
        scan_start = jump_index + 1


def body_entry_index(
    instructions: list[Instruction],
    body_start: int,
    target_index: int,
) -> int:
    if target_index > body_start and all(
        instruction.opname == "NOP"
        for instruction in instructions[body_start:target_index]
    ):
        return body_start
    return target_index


def compound_condition_skip_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    jumps: list[int],
    end_index: int,
    returns_are_one: bool = False,
) -> int | None:
    body_start = jumps[-1] + 1
    test_starts = {jump_index + 1 for jump_index in jumps[:-1]}
    skip_index: int | None = None
    for jump_index in jumps:
        target_index = offset_to_index.get(int(instructions[jump_index].argval))
        if target_index is None:
            return None
        target_index = fold_exit_jump_copy(instructions, target_index, end_index)
        target_index = body_entry_index(instructions, body_start, target_index)
        if target_index == body_start:
            continue
        if target_index in test_starts and target_index > jump_index:
            continue
        if target_index < body_start or target_index > end_index:
            return None
        if returns_are_one:
            target_index = first_return_copy(instructions, target_index)
        if skip_index is not None and skip_index != target_index:
            return None
        skip_index = target_index
    return skip_index


def ignored_guard_result(
    instructions: list[Instruction], start_index: int, body_start: int
) -> ast.expr | None:
    if has_only_ignored_instructions(instructions, start_index, body_start):
        return ast.Constant(value=True)
    return None


def guard_result_at(
    results: dict[int, ast.expr | None],
    index: int | None,
    body_start: int,
) -> ast.expr | None:
    if index is None:
        return None
    if index > body_start:
        return ast.Constant(value=False)
    return results.get(index)


def next_condition_jump(
    instructions: list[Instruction],
    jump_indexes: frozenset[int],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if index in jump_indexes:
            return index
        if instructions[index].opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
    return None


def join_with_items(statement: ast.With | ast.AsyncWith) -> None:
    inner = statement.body[0]
    if len(statement.body) == 1 and type(inner) is type(statement):
        statement.items.extend(inner.items)
        statement.body = inner.body


def body_ends_with_break(body: list[ast.stmt]) -> bool:
    if not body:
        return False
    tail = body[-1]
    if isinstance(tail, ast.Break):
        return True
    if isinstance(tail, ast.If) and not tail.orelse:
        return body_ends_with_break(tail.body)
    return False


def has_effectful_skip_to_offset(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    target_offset: int,
    version: tuple[int, ...] | None,
) -> bool:
    scan_start = start_index
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if not is_forward_conditional_jump(instruction):
            continue
        if instruction.argval != target_offset:
            continue
        if not simple_condition_range(instructions, scan_start, index, version):
            return True
        scan_start = index + 1
    return False


def jumps_back_again(instructions: list[Instruction], jump_index: int) -> bool:
    jump = instructions[jump_index]
    return any(
        "JUMP_BACKWARD" in instruction.opname
        and jump.argval <= instruction.argval <= jump.offset
        for instruction in instructions[jump_index + 1 :]
    )


def target_is_loop_continue(
    instructions: list[Instruction],
    cursor: int,
    target_index: int,
    end_index: int,
) -> bool:
    if target_index > end_index or target_index >= len(instructions):
        return False

    target = instructions[target_index]
    if target.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_BACKWARD"}:
        return False
    if not isinstance(target.argval, int):
        return False
    return target.argval <= instructions[cursor].offset


def is_backward_conditional_jump(instruction: Instruction) -> bool:
    if "JUMP_BACKWARD" not in instruction.opname:
        return False
    if not isinstance(instruction.argval, int):
        return False
    return (
        "IF_FALSE" in instruction.opname
        or "IF_TRUE" in instruction.opname
        or "IF_NONE" in instruction.opname
        or "IF_NOT_NONE" in instruction.opname
    )


def ends_in_inner_loop(
    instructions: list[Instruction],
    cursor: int,
    body_start: int,
    back_jump_index: int,
) -> bool:
    back_jump = instructions[back_jump_index]
    if back_jump.opname != "JUMP_BACKWARD":
        return False
    if back_jump.argval < instructions[body_start].offset:
        return False
    test_offsets = condition_prefix_offsets(instructions, cursor)
    return any(
        instruction.opname == "JUMP_BACKWARD" and instruction.argval in test_offsets
        for instruction in instructions[body_start:back_jump_index]
    )


def loop_starts_between(
    instructions: list[Instruction],
    cursor: int,
    body_start: int,
    end_index: int,
) -> bool:
    first = instructions[cursor].offset
    start = instructions[body_start].offset
    return any(
        instruction.opname == "JUMP_BACKWARD" and first < instruction.argval < start
        for instruction in instructions[body_start:end_index]
    )


def break_block_else_end(
    instructions: list[Instruction],
    block_start: int,
    exit_index: int,
    end_index: int,
) -> int | None:
    first = instructions[block_start]
    if first.starts_line is None or first.opname not in {"NOP", "JUMP_FORWARD"}:
        return None
    start = block_start
    while start < exit_index and instructions[start].opname == "NOP":
        start += 1
    length = exit_index - start
    last = instructions[exit_index - 1]
    if length == 1 and last.opname == "JUMP_FORWARD":
        return next(
            (
                index
                for index in range(exit_index, end_index + 1)
                if index < len(instructions)
                and instructions[index].offset == last.argval
            ),
            None,
        )
    if length == 0 or not is_terminal_op(last.opname):
        return None
    if any("JUMP" in item.opname for item in instructions[start:exit_index]):
        return None
    return next(
        (
            index
            for index in range(exit_index, end_index - length + 1)
            if same_instruction_run(instructions, start, index, length)
        ),
        None,
    )


def condition_prefix_offsets(
    instructions: list[Instruction],
    jump_index: int,
) -> frozenset[int]:
    condition_line = find_condition_line(instructions, jump_index)
    start_index = jump_index
    for index in range(jump_index - 1, -1, -1):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            start_index = index
            continue
        if condition_line is not None:
            line = instruction.starts_line
            if line is not None and line != condition_line:
                break
        start_index = index
        if instruction.starts_line == condition_line:
            break

    return frozenset(
        int(instruction.offset)
        for instruction in instructions[start_index : jump_index + 1]
        if instruction.opname not in IGNORED_BEHAVIOR_OPNAMES
    )


def first_return_copy(instructions: list[Instruction], index: int) -> int:
    while (
        index > 0
        and instructions[index].opname == "RETURN_CONST"
        and instructions[index - 1].opname == "RETURN_CONST"
        and instructions[index - 1].argval == instructions[index].argval
    ):
        index -= 1
    return index


def fold_exit_jump_copy(
    instructions: list[Instruction],
    target_index: int,
    end_index: int,
) -> int:
    if not end_index < target_index < len(instructions):
        return target_index
    exit_jump = instructions[end_index]
    if exit_jump.opname not in {"JUMP_BACKWARD", "JUMP_FORWARD", "RETURN_CONST"}:
        return target_index
    if exit_jump.argval == instructions[target_index].offset and (
        exit_jump.opname != "RETURN_CONST"
    ):
        return end_index
    for index in range(end_index + 1, target_index + 1):
        copy = instructions[index]
        if copy.opname != exit_jump.opname or copy.argval != exit_jump.argval:
            return target_index
    return end_index


def rest_runs_on_behind(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    if end_index >= len(instructions):
        return False
    behind = instructions[end_index]
    if behind.opname in {"JUMP_BACKWARD", "END_FOR"}:
        return False
    return any(
        is_forward_conditional_jump(instruction) and instruction.argval == behind.offset
        for instruction in instructions[start_index:end_index]
    )


def jumps_over_for_continue(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
) -> bool:
    if is_false_jump(instructions[cursor].opname):
        return False
    back_jump = instructions[cursor + 1]
    if back_jump.opname != "JUMP_BACKWARD":
        return False
    loop_index = offset_to_index.get(int(back_jump.argval))
    return loop_index is not None and instructions[loop_index].opname == "FOR_ITER"


def jumps_between(
    instructions: list[Instruction],
    index_range: tuple[int, int],
    offsets: tuple[int, int],
) -> bool:
    return any(
        is_forward_conditional_jump(instruction)
        and offsets[0] < instruction.argval < offsets[1]
        for instruction in instructions[index_range[0] : index_range[1]]
    )


def loop_continue_terminal_start(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    end_index: int,
    terminal_limit_index: int,
    loop_continue_offsets: frozenset[int],
) -> int | None:
    if not loop_continue_offsets:
        return None

    terminal_index = previous_non_ignored_index(
        instructions, end_index - 1, start_index
    )
    if terminal_index is None:
        return None
    if jump_is_skipped_from(instructions, start_index, terminal_index, end_index):
        return None
    if end_index < len(instructions) and jumps_between(
        instructions,
        (start_index, terminal_index),
        (instructions[terminal_index].offset, instructions[end_index].offset),
    ):
        return None
    if is_loop_continue_terminal(
        instructions,
        offset_to_index,
        terminal_index,
        terminal_limit_index,
        loop_continue_offsets,
    ):
        return terminal_index
    return None


def jump_is_skipped_from(
    instructions: list[Instruction],
    start_index: int,
    jump_index: int,
    end_index: int,
) -> bool:
    skipped_from = instructions[jump_index].offset
    end_offset = instructions[end_index].offset
    return any(
        is_forward_conditional_jump(instruction)
        and skipped_from < instruction.argval <= end_offset
        for instruction in instructions[start_index:jump_index]
    )


def is_loop_continue_terminal(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    index: int,
    end_index: int,
    loop_continue_offsets: frozenset[int],
) -> bool:
    instruction = instructions[index]
    if instruction.opname in {"JUMP", "JUMP_ABSOLUTE", "JUMP_BACKWARD"}:
        return instruction.argval in loop_continue_offsets

    if instruction.opname != "JUMP_FORWARD":
        return False
    if not isinstance(instruction.argval, int):
        return False
    target_index = offset_to_index.get(instruction.argval)
    if target_index is None or target_index >= end_index:
        return False
    if target_index >= len(instructions):
        return False
    return is_loop_continue_terminal(
        instructions,
        offset_to_index,
        target_index,
        end_index,
        loop_continue_offsets,
    )


def replace_matching_terminal_if_tail(
    body: list[ast.stmt],
    tail: list[ast.stmt],
) -> list[ast.stmt]:
    return [
        replace_matching_terminal_if_tail_statement(statement, tail)
        for statement in body
    ]


def replace_matching_terminal_if_tail_statement(
    statement: ast.stmt,
    tail: list[ast.stmt],
) -> ast.stmt:
    if not isinstance(statement, ast.If):
        return statement
    if statement.orelse:
        return statement
    if not ast_statement_lists_equal(statement.body, tail):
        return statement
    return ast.If(test=statement.test, body=[ast.Break()], orelse=[])


def ast_statement_lists_equal(
    left: list[ast.stmt],
    right: list[ast.stmt],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        ast.dump(left_item) == ast.dump(right_item)
        for left_item, right_item in zip(left, right, strict=True)
    )


def simple_if_body_end(
    instructions: list[Instruction],
    body_start_index: int,
    target_index: int,
    end_index: int,
) -> int:
    true_return_start = generated_none_return_region_ending_at(
        instructions, target_index, body_start_index
    )
    if true_return_start is None:
        return target_index

    false_return_end = none_return_region_starting_at(
        instructions, target_index, end_index
    )
    if false_return_end is None:
        return target_index

    if not has_only_ignored_instructions(
        instructions, false_return_end, len(instructions)
    ):
        return target_index
    return true_return_start


def generated_none_return_region_ending_at(
    instructions: list[Instruction],
    end_index: int,
    lower_bound: int,
) -> int | None:
    start_index = none_return_region_ending_at(instructions, end_index, lower_bound)
    if start_index is None:
        return None
    pop = instructions[start_index - 1]
    if pop.opname == "POP_TOP" and pop.starts_line is not None:
        return None

    if any(
        instruction.starts_line is not None
        for instruction in instructions[start_index:end_index]
    ):
        return None
    return start_index


def none_return_region_ending_at(
    instructions: list[Instruction],
    end_index: int,
    lower_bound: int,
) -> int | None:
    return_index = end_index - 1
    if return_index < lower_bound:
        return None

    instruction = instructions[return_index]
    if instruction.opname == "RETURN_CONST" and instruction.argval is None:
        return return_index

    value_index = return_index - 1
    if instruction.opname != "RETURN_VALUE" or value_index < lower_bound:
        return None
    if is_none_load(instructions[value_index]):
        return value_index
    return None


def none_return_region_starting_at(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    if start_index >= end_index:
        return None

    instruction = instructions[start_index]
    if instruction.opname == "RETURN_CONST" and instruction.argval is None:
        return start_index + 1

    return_index = start_index + 1
    if not is_none_load(instruction) or return_index >= end_index:
        return None
    if instructions[return_index].opname == "RETURN_VALUE":
        return return_index + 1
    return None


def has_only_ignored_instructions(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    for index in range(start_index, end_index):
        if instructions[index].opname not in IGNORED_BEHAVIOR_OPNAMES:
            return False
    return True


def single_compare(value: ast.expr) -> ast.Compare | None:
    if not isinstance(value, ast.Compare):
        return None
    if len(value.ops) != 1 or len(value.comparators) != 1:
        return None
    return value


def merge_chained_compare(
    first: ast.Compare,
    next_value: ast.expr | None,
) -> ast.Compare | None:
    next_compare = single_compare(next_value) if next_value is not None else None
    if next_compare is None:
        return None
    if not same_expression(first.comparators[-1], next_compare.left):
        return None
    return ast.Compare(
        left=first.left,
        ops=[*first.ops, *next_compare.ops],
        comparators=[*first.comparators, *next_compare.comparators],
    )


def extend_chained_comparison(
    child: Any,
    instructions: list[Instruction],
    start_index: int,
    jump_index: int,
    comparison: ast.Compare,
) -> ast.Compare | None:
    child.translate_range(instructions, start_index, jump_index)
    if not child.stack:
        return None
    link = unwrap_truth_test(coerce_expr(child.stack[-1]))
    merged = merge_chained_compare(comparison, link)
    if merged is not None:
        child.pop_or_none()
    return merged


def same_expression(left: ast.expr, right: ast.expr) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def copied_value_pop_index(
    instructions: list[Instruction],
    cursor: int,
    target_index: int,
) -> int | None:
    copy_index = previous_non_ignored_index(instructions, cursor - 1, 0)
    if copy_index is not None and instructions[copy_index].opname == "TO_BOOL":
        copy_index = previous_non_ignored_index(instructions, copy_index - 1, 0)
    if copy_index is None:
        return None
    copy = instructions[copy_index]
    if copy.opname != "COPY" or copy.arg != 1:
        return None
    pop_index = next_non_ignored_index(instructions, cursor + 1, target_index)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return None
    if (
        target_index + 1 < len(instructions)
        and instructions[target_index].opname == "SWAP"
        and instructions[target_index + 1].opname != "POP_EXCEPT"
    ):
        return None
    return pop_index


def fold_test_failing_into_else(
    test: ast.expr,
    true_body: list[ast.stmt],
) -> tuple[ast.expr, list[ast.stmt]] | None:
    inner = true_body[0] if len(true_body) == 1 else None
    if not isinstance(inner, ast.If) or inner.orelse:
        return None
    return ast.BoolOp(op=ast.And(), values=[test, inner.test]), inner.body


def has_test_failing_into(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    target_index: int,
) -> bool:
    target_offset = instructions[target_index].offset
    return any(
        is_forward_conditional_jump(instruction) and instruction.argval == target_offset
        for instruction in instructions[start_index:end_index]
    )


def is_branch_end_jump(
    instructions: list[Instruction],
    end_index: int,
    target_index: int,
) -> bool:
    if target_index != end_index + 1 or end_index >= len(instructions):
        return False
    return instructions[end_index].opname == "JUMP_FORWARD"


def find_chained_comparison_jump(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
        if is_forward_conditional_jump(instruction) and is_false_jump(
            instruction.opname
        ):
            return index
        if instruction.opname == "POP_TOP":
            continue
        if comparison_expression_op(instruction.opname):
            continue
        return None
    return None


def comparison_expression_op(opname: str) -> bool:
    return opname in {
        "BINARY_OP",
        "BINARY_SUBSCR",
        "CACHE",
        "CALL",
        "COMPARE_OP",
        "COPY",
        "IS_OP",
        "LOAD_ATTR",
        "LOAD_CONST",
        "LOAD_DEREF",
        "LOAD_FAST",
        "LOAD_FAST_BORROW",
        "LOAD_FAST_CHECK",
        "LOAD_FAST_LOAD_FAST",
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "PUSH_NULL",
        "SWAP",
        "TO_BOOL",
    }


def skip_chained_compare_true_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = next_non_ignored_index(instructions, start_index, end_index)
    if cursor is None:
        return None
    if instructions[cursor].opname == "POP_TOP":
        cursor = next_non_ignored_index(instructions, cursor + 1, end_index)
    return cursor


def copied_test_start(instructions: list[Instruction], jump_index: int) -> int | None:
    if not 0 < jump_index < len(instructions):
        return None
    jump = instructions[jump_index]
    if not is_forward_conditional_jump(jump) or is_none_jump(jump.opname):
        return None
    copy_index = jump_index - 1
    if instructions[copy_index].opname == "TO_BOOL":
        copy_index -= 1
    copy = instructions[copy_index]
    return copy_index if copy.opname == "COPY" and copy.arg == 1 else None


def threaded_outer_test(
    instructions: list[Instruction],
    cursor: int,
    target_index: int,
) -> int | None:
    if instructions[target_index].opname != "POP_TOP":
        return None
    copy_index = copied_test_start(instructions, target_index - 1)
    if copy_index is None or copy_index <= cursor:
        return None
    outer, inner = instructions[target_index - 1], instructions[cursor]
    if is_false_jump(outer.opname) == is_false_jump(inner.opname):
        return None
    return copy_index


def threaded_operand_end(
    instructions: list[Instruction],
    cursor: int,
    target_index: int,
    end_index: int,
) -> int:
    if copied_test_start(instructions, cursor) is None:
        return target_index
    jump = instructions[cursor]
    for outer_index in (end_index + 1, end_index + 2):
        if copied_test_start(instructions, outer_index) != end_index:
            continue
        outer = instructions[outer_index]
        if is_false_jump(outer.opname) != is_false_jump(jump.opname):
            return end_index if target_index == outer_index + 1 else target_index
        return end_index if outer.argval == jump.argval else target_index
    return target_index


def short_circuit_operand_end(
    instructions: list[Instruction],
    start_index: int,
    target_index: int,
) -> int:
    swap_index = target_index - 2
    if swap_index <= start_index or instructions[swap_index].opname != "SWAP":
        return target_index
    if instructions[swap_index + 1].opname != "POP_TOP":
        return target_index
    if instructions[swap_index - 1].opname == "JUMP_FORWARD":
        return swap_index - 1
    tail_end = terminal_tail_end_index(instructions, target_index, len(instructions))
    if tail_end is None:
        return target_index
    return max(start_index, swap_index - (tail_end - target_index))


def chained_compare_false_return_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    swap_index = next_non_ignored_index(instructions, start_index, end_index)
    if swap_index is None:
        return None
    swap = instructions[swap_index]
    if swap.opname != "SWAP" or int(swap.arg or 0) != 2:
        return None

    pop_index = next_non_ignored_index(instructions, swap_index + 1, end_index)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return None

    return_index = next_non_ignored_index(instructions, pop_index + 1, end_index)
    if return_index is None:
        return None
    if instructions[return_index].opname not in {"RETURN_CONST", "RETURN_VALUE"}:
        return None
    return return_index + 1


def is_chained_compare_test(instructions: list[Instruction], cursor: int) -> bool:
    compare_index = previous_non_ignored_index(instructions, cursor - 1, 0)
    if compare_index is not None and instructions[compare_index].opname == "TO_BOOL":
        compare_index = previous_non_ignored_index(instructions, compare_index - 1, 0)
    if compare_index is None:
        return False
    if instructions[compare_index].opname not in {"COMPARE_OP", "CONTAINS_OP", "IS_OP"}:
        return False
    copy_index = previous_non_ignored_index(instructions, compare_index - 1, 0)
    if copy_index is None:
        return False
    copy = instructions[copy_index]
    return copy.opname == "COPY" and copy.arg == 2


def chained_compare_cleanup_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    swap_index = next_non_ignored_index(instructions, start_index, end_index)
    if swap_index is None:
        return None
    swap = instructions[swap_index]
    if swap.opname != "SWAP" or int(swap.arg or 0) != 2:
        return None
    pop_index = next_non_ignored_index(instructions, swap_index + 1, end_index)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return None
    return pop_index + 1


def negated_chain_exits(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    link_start: int,
    false_index: int,
) -> tuple[int, int, int] | None:
    jump_index = next(
        (
            index
            for index in range(link_start, false_index)
            if is_forward_conditional_jump(instructions[index])
        ),
        None,
    )
    if jump_index is None or is_false_jump(instructions[jump_index].opname):
        return None
    after_index = offset_to_index.get(int(instructions[jump_index].argval))
    if after_index is None or after_index <= false_index:
        return None

    skip_index = next_non_ignored_index(instructions, jump_index + 1, false_index)
    body_start = next_non_ignored_index(instructions, false_index + 1, after_index)
    if skip_index is None or body_start is None:
        return None
    if instructions[false_index].opname != "POP_TOP":
        return None
    skip = instructions[skip_index]
    if skip.opname == "JUMP_FORWARD" and skip.argval == instructions[body_start].offset:
        return jump_index, body_start, after_index
    length = false_index - skip_index
    if body_start + length == after_index and same_instruction_run(
        instructions, skip_index, body_start, length
    ):
        return jump_index, body_start, after_index
    return None


def drops_chain_operand(instructions: list[Instruction], false_index: int) -> bool:
    pop_index = next_non_ignored_index(instructions, false_index, len(instructions))
    return pop_index is not None and instructions[pop_index].opname == "POP_TOP"


def chained_compare_true_body_start(
    instructions: list[Instruction],
    false_index: int,
    next_false_index: int,
) -> int | None:
    pop_index = next_non_ignored_index(instructions, false_index, next_false_index)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return None

    jump_index = next_non_ignored_index(instructions, pop_index + 1, next_false_index)
    if jump_index is None:
        return None
    jump = instructions[jump_index]
    if jump.opname != "JUMP_FORWARD":
        return repeated_false_path_end(instructions, jump_index, next_false_index)
    if jump.argval != instructions[next_false_index].offset:
        return None

    return next_non_ignored_index(instructions, jump_index + 1, next_false_index)


def repeated_false_path_end(
    instructions: list[Instruction],
    start_index: int,
    false_index: int,
) -> int | None:
    copy_end = terminal_tail_end_index(instructions, start_index, false_index)
    if copy_end is None:
        return None
    length = copy_end - start_index
    if false_index + length > len(instructions) or not same_instruction_run(
        instructions, start_index, false_index, length
    ):
        return None
    return next_non_ignored_index(instructions, copy_end, false_index)


def chained_compare_none_return_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    load_index = next_non_ignored_index(instructions, start_index, end_index)
    if load_index is None or not is_none_load(instructions[load_index]):
        return None

    return_index = next_non_ignored_index(instructions, load_index + 1, end_index)
    if return_index is None or instructions[return_index].opname != "RETURN_VALUE":
        return None
    return return_index + 1


def chained_compare_false_store_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    true_store: Instruction,
) -> int | None:
    swap_index = next_non_ignored_index(instructions, start_index, end_index)
    if swap_index is None:
        return None
    swap = instructions[swap_index]
    if swap.opname != "SWAP" or int(swap.arg or 0) != 2:
        return None

    pop_index = next_non_ignored_index(instructions, swap_index + 1, end_index)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return None

    store_index = next_non_ignored_index(instructions, pop_index + 1, end_index)
    if store_index is None:
        return None
    false_store = instructions[store_index]
    if false_store.opname != true_store.opname:
        return None
    if false_store.argval != true_store.argval:
        return None
    return store_index + 1


def is_name_store_op(opname: str) -> bool:
    return opname in {
        "STORE_DEREF",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_NAME",
    }


def find_chained_compare_true_store(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    store_index = None
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
        if is_name_store_op(instruction.opname):
            store_index = index
            continue
        if store_index is None:
            continue
        if instruction.opname in {
            "LOAD_DEREF",
            "LOAD_FAST",
            "LOAD_GLOBAL",
            "LOAD_NAME",
        }:
            continue
        if instruction.opname in {"RETURN_CONST", "RETURN_VALUE"}:
            continue
        return None
    return store_index


def next_non_ignored_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname not in IGNORED_BEHAVIOR_OPNAMES:
            return index
    return None


def previous_non_ignored_index(
    instructions: list[Instruction],
    start_index: int,
    lower_bound: int,
) -> int | None:
    for index in range(start_index, lower_bound - 1, -1):
        if instructions[index].opname not in IGNORED_BEHAVIOR_OPNAMES:
            return index
    return None


def continue_exit_tests(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
    continue_offsets: frozenset[int],
    version: tuple[int, ...] | None,
) -> list[int]:
    tests: list[int] = []
    start_index = cursor
    while True:
        jump_index = next(
            (
                index
                for index in range(start_index, end_index - 2)
                if "JUMP" in instructions[index].opname
            ),
            None,
        )
        if jump_index is None:
            return tests
        jump = instructions[jump_index]
        exit_jump = instructions[jump_index + 1]
        if (
            not is_forward_conditional_jump(jump)
            or is_false_jump(jump.opname)
            or jump.argval != instructions[jump_index + 2].offset
            or exit_jump.opname != "JUMP_BACKWARD"
            or exit_jump.argval not in continue_offsets
        ):
            return tests
        if tests and not simple_condition_range(
            instructions, start_index, jump_index, version
        ):
            return tests
        tests.append(jump_index)
        start_index = jump_index + 2


@dataclass(frozen=True, slots=True)
class SplitWhileTail:
    start: int
    test_start: int
    end: int


def closes_another_loop(
    instructions: list[Instruction],
    body_start: int,
    back_jump_index: int,
    previous_index: int,
) -> bool:
    if instructions[back_jump_index].argval == instructions[body_start].offset:
        return False
    return not is_forward_conditional_jump(instructions[previous_index])


def is_inner_loop_jump(
    instructions: list[Instruction],
    body_start: int,
    exit_index: int,
    back_jump_index: int,
) -> bool:
    if instructions[back_jump_index].argval == instructions[body_start].offset:
        return False
    return find_split_while_tail(instructions, body_start, exit_index) is not None


def first_handler_of_range(
    decompiler: ControlRecoveryMixin,
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    spans = read_table_spans(
        decompiler.code, instructions, make_offset_index(instructions)
    )
    return min(
        (
            span.target_index
            for span in spans
            if start_index <= span.start_index < end_index <= span.target_index
        ),
        default=end_index,
    )


def find_split_while_tail(
    instructions: list[Instruction],
    body_start: int,
    exit_index: int,
    handlers_start: int | None = None,
) -> SplitWhileTail | None:
    loop_offset = int(instructions[body_start].offset)
    back_jump_index = next(
        (
            index
            for index in range(exit_index, len(instructions))
            if instructions[index].opname == "JUMP_BACKWARD"
            and instructions[index].argval == loop_offset
        ),
        None,
    )
    if back_jump_index is None:
        return None
    test_jump_index = previous_non_ignored_index(
        instructions, back_jump_index - 1, exit_index
    )
    if test_jump_index is None:
        return None
    if not is_forward_conditional_jump(instructions[test_jump_index]):
        return None
    test_start = find_repeated_condition_start(
        instructions, body_start - 1, test_jump_index
    )
    if test_start is None or test_start <= exit_index:
        return None
    return SplitWhileTail(
        start=handler_join_index(
            instructions, handlers_start or exit_index, test_start
        ),
        test_start=test_start,
        end=split_while_tail_end(instructions, back_jump_index + 1),
    )


def split_while_tail_end(instructions: list[Instruction], exit_index: int) -> int:
    if exit_index >= len(instructions):
        return exit_index
    if instructions[exit_index].opname == "JUMP_BACKWARD_NO_INTERRUPT":
        return exit_index + 1
    tail_end = terminal_tail_end_index(instructions, exit_index, len(instructions))
    return exit_index if tail_end is None else tail_end


def ends_in_written_continue(instructions: list[Instruction], end_index: int) -> bool:
    for index in (end_index - 1, end_index):
        if index >= len(instructions):
            continue
        instruction = instructions[index]
        if instruction.opname == "JUMP_BACKWARD":
            return instruction.starts_line is not None
    return False


def find_repeated_condition_start(
    instructions: list[Instruction],
    cursor: int,
    repeated_jump_index: int,
    condition_line: int | None = None,
) -> int | None:
    if repeated_jump_index <= cursor + 1:
        return None

    if condition_line is None:
        condition_line = find_condition_line(instructions, cursor)
    copy_start = repeated_condition_copy_start(
        instructions,
        cursor,
        repeated_jump_index,
        condition_line,
    )
    if copy_start is not None:
        return copy_start
    if condition_line is not None:
        for index in range(repeated_jump_index - 1, cursor, -1):
            if instructions[index].starts_line == condition_line:
                return index

    previous_index = repeated_jump_index - 1
    if previous_index <= cursor:
        return None
    return previous_index


def repeated_condition_copy_start(
    instructions: list[Instruction],
    cursor: int,
    repeated_jump_index: int,
    condition_line: int | None,
) -> int | None:
    test_start = cursor
    while test_start > 0 and instructions[test_start].starts_line != condition_line:
        test_start -= 1
    remaining = [
        instruction
        for instruction in instructions[test_start:cursor]
        if "JUMP" not in instruction.opname
    ]
    copy_start = repeated_jump_index
    while remaining and copy_start > cursor + 1:
        copy_start -= 1
        copy = instructions[copy_start]
        if "JUMP" in copy.opname:
            continue
        original = remaining.pop()
        if original.opname != copy.opname or original.argval != copy.argval:
            return None
    return None if remaining else copy_start


COPIED_TAIL_END_OPS = frozenset({"RAISE_VARARGS", "RETURN_CONST", "RETURN_VALUE"})


def copied_tail_branch_ends(
    instructions: list[Instruction],
    cursor: int,
    false_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    true_last = previous_non_ignored_index(instructions, false_index - 1, cursor + 1)
    if true_last is None or instructions[true_last].opname not in COPIED_TAIL_END_OPS:
        return None

    false_start = next_non_ignored_index(instructions, false_index, end_index)
    if false_start is None:
        return None
    false_line = instructions[false_start].starts_line
    if false_line is not None and false_line != find_condition_line(
        instructions, cursor
    ):
        return None

    for false_last in range(false_start, end_index):
        if instructions[false_last].opname in COPIED_TAIL_END_OPS:
            return true_last + 1, false_last + 1
    return None


def common_tail_length(
    instructions: list[Instruction],
    true_end: int,
    false_end: int,
) -> int:
    length = 0
    while true_end - length > 0 and same_instruction_behavior(
        instructions[true_end - length - 1],
        instructions[false_end - length - 1],
    ):
        length += 1
    return length


def is_closed_expression_range(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    if next_non_ignored_index(instructions, start_index, end_index) is None:
        return False
    if end_index >= len(instructions):
        return False
    start_offset = instructions[start_index].offset
    end_offset = instructions[end_index].offset
    for instruction in instructions[start_index:end_index]:
        if "JUMP" not in instruction.opname:
            continue
        if not isinstance(instruction.argval, int):
            return False
        if not start_offset <= instruction.argval <= end_offset:
            return False
    return True


def find_condition_line(
    instructions: list[Instruction],
    cursor: int,
) -> int | None:
    for index in range(cursor, -1, -1):
        line = instructions[index].starts_line
        if line is not None:
            return line
    return None


def is_none_load(instruction: Instruction) -> bool:
    if instruction.opname in {"LOAD_CONST", "RETURN_CONST"}:
        return instruction.argval is None
    return False
