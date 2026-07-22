import ast
from typing import Any
from pyc2py.astree import make_constant, make_name
from pyc2py.bytecode.instruction import IGNORED_BEHAVIOR_OPNAMES
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.stack_effect import instruction_stack_effect
from pyc2py.decompiler.control_loop import ControlLoopRecoveryMixin
from pyc2py.decompiler.opcodes.flow import terminal_tail_end_index
from pyc2py.decompiler.runtime import coerce_expr
from pyc2py.decompiler.structures import (
    MatchCaseSpec,
    WithRegion,
    condition_from_jump,
    find_simple_match_region,
    find_with_region,
    invert_condition,
    is_false_jump,
    is_forward_conditional_jump,
    make_async_with_statement,
    make_match,
    make_with_statement,
)

class ControlRecoveryMixin(ControlLoopRecoveryMixin):
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
        region = find_with_region(instructions, cursor, end_index)
        if region is None:
            return None

        context_expr = self.with_context_expr(instructions, region)
        optional_vars = None
        if region.optional_var_index is not None:
            target_instruction = instructions[region.optional_var_index]
            optional_vars = make_name(str(target_instruction.argval), ast.Store())

        child = self.make_child()
        child.stack.clear()
        body = child.translate_range(
            instructions, region.body_start_index, region.body_end_index
        )
        self.warnings.extend(child.warnings)
        if child.stack:
            body.append(ast.Return(value=coerce_expr(child.stack[-1])))

        trailing_return = None
        if region.trailing_return_index is not None:
            trailing_return = self.return_statement_from_instruction(
                instructions[region.trailing_return_index]
            )
            if instructions[region.trailing_return_index].starts_line is None:
                body.append(trailing_return)
                trailing_return = None

        if region.is_async:
            self.statements.append(
                make_async_with_statement(context_expr, body, optional_vars)
            )
        else:
            self.statements.append(make_with_statement(context_expr, body, optional_vars))
        if trailing_return is not None:
            self.statements.append(trailing_return)
        return region.after_index

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

    def try_translate_conditional(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        target_index = offset_to_index.get(int(instruction.argval))
        if target_index is None or target_index <= cursor or target_index > end_index:
            return None

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
        conditional_jumps = [cursor]
        skip_index: int | None = None
        for index in range(cursor + 1, body_start):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue

            target_index = offset_to_index.get(int(instruction.argval))
            if target_index is None:
                return None
            if target_index == body_start:
                conditional_jumps.append(index)
            elif target_index > body_start:
                if target_index > end_index:
                    return None
                if skip_index is None:
                    skip_index = target_index
                elif skip_index != target_index:
                    return None
                conditional_jumps.append(index)
            elif "OR_POP" in instruction.opname:
                continue
            elif target_index > index:
                conditional_jumps.append(index)
            else:
                return None

        if len(conditional_jumps) < 2 or skip_index is None:
            return None
        if skip_index <= body_start or skip_index > end_index:
            return None
        if not has_only_ignored_instructions(instructions, body_start, skip_index):
            body = self.translate_child_statements(
                instructions,
                body_start,
                skip_index,
            )
        else:
            body = []
        if not body:
            body = self.translate_child_statements(
                instructions,
                body_start,
                skip_index,
            )
        if not body:
            return None

        jump_indexes = frozenset(conditional_jumps)
        expression_cache: dict[tuple[int, int], ast.expr] = {}
        condition_cache: dict[int, ast.expr | None] = {}

        def expression_between(start_index: int, jump_index: int) -> ast.expr | None:
            cache_key = (start_index, jump_index)
            if cache_key in expression_cache:
                return expression_cache[cache_key]
            if jump_index == cursor:
                expression = condition
            else:
                if not simple_condition_range(
                    instructions,
                    start_index,
                    jump_index,
                    self.version,
                ):
                    return None
                expression = self.evaluate_expression_range(
                    instructions,
                    start_index,
                    jump_index,
                )
                if expression is None:
                    return None
            expression_cache[cache_key] = expression
            return expression

        def success_from(start_index: int) -> ast.expr | None:
            if start_index == body_start:
                return ast.Constant(value=True)
            if start_index == skip_index:
                return ast.Constant(value=False)
            if start_index > body_start:
                return ast.Constant(value=False)
            if start_index in condition_cache:
                return condition_cache[start_index]

            jump_index = next_condition_jump(
                instructions,
                jump_indexes,
                start_index,
                body_start,
            )
            if jump_index is None:
                if has_only_ignored_instructions(instructions, start_index, body_start):
                    return ast.Constant(value=True)
                return None

            expression = expression_between(start_index, jump_index)
            if expression is None:
                condition_cache[start_index] = None
                return None

            jump = instructions[jump_index]
            target_index = offset_to_index.get(int(jump.argval))
            if target_index is None:
                condition_cache[start_index] = None
                return None
            jump_condition = condition_from_jump(jump.opname, expression)
            taken_success = success_from(target_index)
            fallthrough_success = success_from(jump_index + 1)
            if taken_success is None or fallthrough_success is None:
                condition_cache[start_index] = None
                return None

            result = combine_branch_success(
                terminal_guard_for_jump(jump, jump_condition, jumps_to_terminal=True),
                taken_success,
                terminal_guard_for_jump(jump, jump_condition, jumps_to_terminal=False),
                fallthrough_success,
            )
            condition_cache[start_index] = result
            return result

        guard = success_from(cursor)
        if guard is None or is_false_constant(guard):
            return None
        self.statements.append(ast.If(test=guard, body=body, orelse=[]))
        return skip_index

    def try_translate_targeted_prefixed_guard_body_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        body_start: int,
        end_index: int,
        condition: ast.expr,
    ) -> int | None:
        body_offset = instructions[body_start].offset
        skip_index: int | None = None
        prefix_jumps: list[int] = []
        scan_start = cursor + 1
        for index in range(cursor + 1, body_start):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue
            if not simple_condition_range(
                instructions,
                scan_start,
                index,
                self.version,
            ):
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

        if not prefix_jumps or skip_index is None:
            return None

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
        if store_index is None or store_index != false_index + 1:
            return None
        if store_index >= end_index:
            return None

        store = instructions[store_index]
        if store.opname not in {"STORE_DEREF", "STORE_FAST", "STORE_GLOBAL", "STORE_NAME"}:
            return None

        right = self.evaluate_expression_range(
            instructions,
            cursor + 1,
            jump_index,
        )
        true_value = self.evaluate_expression_range(
            instructions,
            true_index,
            join_jump_index,
        )
        false_value = self.evaluate_expression_range(
            instructions,
            false_index,
            store_index,
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
        return store_index + 1

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
            if skip_index is None or skip_index <= target_index or skip_index > end_index:
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
                        guard_expr,
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1
            found_skip_jump = True

        if not found_skip_jump or after_index is None:
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

        if is_false_jump(instructions[cursor].opname):
            expression = ast.IfExp(test=condition, body=true_value, orelse=false_value)
        else:
            expression = ast.IfExp(test=condition, body=false_value, orelse=true_value)
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

        after_index = target_index + 1
        after_offset = None
        if after_index < len(instructions):
            after_offset = instructions[after_index].offset

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
                        guard_expr,
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1

        if found_skip_jump:
            self.statements.append(
                ast.If(test=guard, body=[ast.Continue()], orelse=[])
            )
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
            if not body_ends_with_break(body):
                self.statements.append(ast.Continue())
            return after_index

        self.statements.append(ast.If(test=guard, body=[ast.Continue()], orelse=[]))
        return after_index

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
        if terminal_index is None:
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
                        guard_expr,
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1

        if found_skip_jump:
            self.statements.append(
                ast.If(test=guard, body=[ast.Continue()], orelse=[])
            )
            return target_index

        body = self.translate_child_statements(
            instructions,
            cursor + 1,
            terminal_index,
        )
        self.statements.append(
            ast.If(test=test, body=[*body, ast.Continue()], orelse=[])
        )
        return target_index

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
        for index in range(cursor + 1, target_index):
            instruction = instructions[index]
            if not is_forward_conditional_jump(instruction):
                continue
            if instruction.argval != after_offset:
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
                        guard_expr,
                        jumps_to_terminal=False,
                    ),
                ],
            )
            scan_start = index + 1

        if not found_skip_jump:
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

        child = self.make_child()
        combined = first_compare
        scan_start = cursor + 1
        while True:
            next_jump_index = find_chained_comparison_jump(
                instructions, scan_start, false_index
            )
            if next_jump_index is None:
                return None

            next_jump = instructions[next_jump_index]
            next_false_index = offset_to_index.get(int(next_jump.argval))
            if next_false_index is None or next_false_index <= next_jump_index:
                return None
            if next_false_index > end_index:
                return None

            child.translate_range(instructions, scan_start, next_jump_index)
            if not child.stack:
                return None
            next_compare = coerce_expr(child.stack[-1])
            combined = merge_chained_compare(combined, next_compare)
            if combined is None:
                return None
            child.pop_or_none()
            if next_false_index != false_index:
                break
            scan_start = next_jump_index + 1
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
        body = self.translate_child_statements(
            instructions, next_jump_index + 1, body_end
        )
        self.statements.append(
            ast.If(test=combined, body=body or [ast.Pass()], orelse=[])
        )
        return next_false_index

    def try_translate_chained_comparison_body_return(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        false_index: int,
        end_index: int,
        first_compare: ast.Compare,
    ) -> int | None:
        child = self.make_child()
        combined = first_compare
        scan_start = cursor + 1
        next_jump_index = None
        next_false_index = false_index
        while True:
            next_jump_index = find_chained_comparison_jump(
                instructions,
                scan_start,
                next_false_index,
            )
            if next_jump_index is None:
                return None

            next_jump_false_index = offset_to_index.get(
                int(instructions[next_jump_index].argval)
            )
            if next_jump_false_index is None or next_jump_false_index <= next_jump_index:
                return None
            if next_jump_false_index > end_index:
                return None

            child.translate_range(instructions, scan_start, next_jump_index)
            if not child.stack:
                return None
            next_compare = coerce_expr(child.stack[-1])
            combined = merge_chained_compare(combined, next_compare)
            if combined is None:
                return None
            child.pop_or_none()
            if next_jump_false_index != next_false_index:
                next_false_index = next_jump_false_index
                break
            scan_start = next_jump_index + 1

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

        child = self.make_child()
        combined = first_compare
        scan_start = cursor + 1
        while True:
            next_jump_index = find_chained_comparison_jump(
                instructions, scan_start, false_index
            )
            if next_jump_index is None:
                break

            next_jump_false_index = offset_to_index.get(
                int(instructions[next_jump_index].argval)
            )
            if next_jump_false_index is None or next_jump_false_index != false_index:
                return None

            child.translate_range(instructions, scan_start, next_jump_index)
            if not child.stack:
                return None
            next_compare = coerce_expr(child.stack[-1])
            combined = merge_chained_compare(combined, next_compare)
            if combined is None:
                return None
            child.pop_or_none()
            scan_start = next_jump_index + 1

        child.translate_range(instructions, scan_start, true_return_index)
        if not child.stack:
            return None
        final_compare = coerce_expr(child.stack[-1])
        combined = merge_chained_compare(combined, final_compare)
        if combined is None:
            return None
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
        child = self.make_child()
        combined = first_compare
        scan_start = cursor + 1
        while True:
            next_jump_index = find_chained_comparison_jump(
                instructions, scan_start, false_index
            )
            if next_jump_index is None:
                break

            next_jump_false_index = offset_to_index.get(
                int(instructions[next_jump_index].argval)
            )
            if next_jump_false_index is None or next_jump_false_index != false_index:
                return None

            child.translate_range(instructions, scan_start, next_jump_index)
            if not child.stack:
                return None
            next_compare = coerce_expr(child.stack[-1])
            combined = merge_chained_compare(combined, next_compare)
            if combined is None:
                return None
            child.pop_or_none()
            scan_start = next_jump_index + 1

        child.translate_range(instructions, scan_start, true_store_index)
        if not child.stack:
            return None
        final_compare = coerce_expr(child.stack[-1])
        combined = merge_chained_compare(combined, final_compare)
        if combined is None:
            return None

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
        back_jump_index = self.find_modern_while_back_jump(
            instructions,
            cursor + 1,
            exit_index,
        )
        if back_jump_index is None:
            return None

        if is_backward_conditional_jump(instructions[back_jump_index]):
            repeated_jump_index = back_jump_index
        else:
            previous_jump_index = self.previous_non_ignorable_index(
                instructions,
                back_jump_index - 1,
                cursor,
            )
            if previous_jump_index is None:
                return None
            repeated_jump_index = previous_jump_index

        repeated_condition_start = find_repeated_condition_start(
            instructions,
            cursor,
            repeated_jump_index,
        )
        if repeated_condition_start is None:
            return None

        body = self.translate_loop_child_statements(
            instructions,
            cursor + 1,
            repeated_condition_start,
            continue_offset=int(instructions[cursor].offset),
            break_index=exit_index,
            extra_continue_offsets=condition_prefix_offsets(instructions, cursor),
        )
        body = self.simplify_modern_loop_break_tail(
            instructions,
            body,
            exit_index,
            end_index,
        )
        test = (
            condition
            if is_false_jump(instructions[cursor].opname)
            else invert_condition(condition)
        )
        self.statements.append(ast.While(test=test, body=body or [ast.Pass()], orelse=[]))
        return exit_index

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

        loop_body_offset = int(instructions[start_index].offset)
        for index in range(exit_index - 1, start_index - 1, -1):
            instruction = instructions[index]
            if (
                instruction.opname not in {"JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP"}
                and not is_backward_conditional_jump(instruction)
            ):
                continue
            if int(instruction.argval) == loop_body_offset:
                return index
        return None

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
            return None
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
        false_value = self.evaluate_expression_range(
            instructions, false_index, join_index
        )
        if true_value is None or false_value is None:
            return None

        if is_false_jump(instructions[cursor].opname):
            expression = ast.IfExp(test=condition, body=true_value, orelse=false_value)
        else:
            expression = ast.IfExp(test=condition, body=false_value, orelse=true_value)
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
            if is_false_jump(instructions[cursor].opname) and is_false_jump(
                candidate.opname
            ):
                return ast.BoolOp(op=ast.And(), values=[condition, right]), index + 1
            if not is_false_jump(instructions[cursor].opname) and not is_false_jump(
                candidate.opname
            ):
                return ast.BoolOp(op=ast.Or(), values=[condition, right]), index + 1
        return condition, cursor + 1

    def try_translate_if_else_statement(
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
            return None
        join_index = offset_to_index.get(int(jump.argval))
        if join_index is None or join_index <= false_index:
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

        true_body = self.translate_child_statements(
            instructions, true_start, jump_index
        )
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
        true_return_start = generated_none_return_region_ending_at(
            instructions,
            false_index,
            cursor + 1,
        )
        if true_return_start is None:
            return None

        false_return_start = none_return_region_ending_at(
            instructions,
            end_index,
            false_index,
        )
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
        self.statements.append(
            ast.If(
                test=test,
                body=true_body or [ast.Pass()],
                orelse=false_body or [ast.Pass()],
            )
        )
        return end_index

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
        )
        if cleanup_start is None:
            return None

        return_index = loop_cleanup_return_end(instructions, cleanup_start, end_index)
        if return_index is None:
            return None

        value = self.evaluate_expression_range(instructions, cursor, cleanup_start)
        if value is None:
            return None

        self.statements.append(ast.Return(value=value))
        return return_index + 1

    def make_child(self) -> Any:
        from pyc2py.decompiler.engine import NativeDecompiler

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
        )

CONDITIONAL_HANDLER_ORDER = (
    "try_translate_loop_continue_guard_statement",
    "try_translate_loop_continue_body_statement",
    "try_translate_chained_comparison",
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
    if end_send_index >= end_index:
        return None
    if instructions[end_send_index].opname != "END_SEND":
        return None
    return end_send_index

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
) -> int | None:
    if is_expression_range_boundary_op(instructions[cursor].opname):
        return None

    for index in range(cursor + 1, end_index):
        opname = instructions[index].opname
        if opname != "SWAP":
            if is_expression_range_boundary_op(opname):
                return None
            continue
        if loop_cleanup_return_end(instructions, index, end_index) is not None:
            return index
        return None
    return None

def is_expression_range_boundary_op(opname: str) -> bool:
    return (
        "JUMP" in opname
        or opname in {
            "FOR_ITER",
            "GET_ITER",
            "GET_AITER",
            "SEND",
            "RETURN_CONST",
            "RETURN_VALUE",
            "RAISE_VARARGS",
            "RERAISE",
        }
    )

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
) -> int | None:
    saw_cleanup = False
    while cursor + 1 < end_index:
        if instructions[cursor].opname != "SWAP":
            break
        if instructions[cursor + 1].opname != "POP_TOP":
            break
        saw_cleanup = True
        cursor += 2

    if not saw_cleanup or cursor >= end_index:
        return None
    if instructions[cursor].opname != "RETURN_VALUE":
        return None
    return cursor

def make_send_value_expression(opname: str, value: ast.expr) -> ast.expr:
    if opname == "GET_AWAITABLE":
        return ast.Await(value=value)
    return ast.YieldFrom(value=value)

def terminal_block_end_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname in {"RETURN_CONST", "RETURN_VALUE", "RAISE_VARARGS"}:
            return index + 1
    return None

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
        if target_index is None:
            return None
        if instruction.argval != false_offset:
            if target_index <= cursor or target_index >= jump_index:
                return None
            if target_follows_branch_body(instructions, cursor, target_index):
                return None
            if body_start is None or target_index < body_start:
                body_start = target_index
        jumps.append(index)
        scan_start = index + 1

    if not jumps:
        return None
    if body_start is None:
        return jumps
    return [index for index in jumps if index < body_start]

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
    "CALL",
    "COMPARE_OP",
    "CONTAINS_OP",
    "IS_OP",
    "KW_NAMES",
    "LOAD_METHOD",
    "PRECALL",
    "PUSH_NULL",
    "SWAP",
    "TO_BOOL",
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
        if not is_simple_condition_op(instruction.opname):
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

def is_simple_condition_op(opname: str) -> bool:
    return (
        opname in SIMPLE_CONDITION_OPS
        or opname.startswith(("BINARY_", "UNARY_"))
    )

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
    condition = ast.Constant(value=True)
    for jump, expr in reversed(pieces):
        taken = terminal_guard_for_jump(jump, expr, jumps_to_terminal=True)
        fallthrough = terminal_guard_for_jump(jump, expr, jumps_to_terminal=False)
        if jump.argval == false_offset:
            condition = make_bool_and(fallthrough, condition)
        elif jump.argval == body_offset:
            condition = make_bool_or(
                [taken, make_bool_and(fallthrough, condition)]
            )
    return condition

def combine_branch_success(
    taken_condition: ast.expr,
    taken_success: ast.expr,
    fallthrough_condition: ast.expr,
    fallthrough_success: ast.expr,
) -> ast.expr:
    taken = make_success_branch(taken_condition, taken_success)
    fallthrough = make_success_branch(fallthrough_condition, fallthrough_success)
    if is_false_constant(taken):
        return fallthrough
    if is_false_constant(fallthrough):
        return taken
    if is_true_constant(taken) or is_true_constant(fallthrough):
        return ast.Constant(value=True)
    return make_bool_or([taken, fallthrough])

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
            continue

    return frozenset(
        int(instruction.offset)
        for instruction in instructions[start_index : jump_index + 1]
        if instruction.opname not in IGNORED_BEHAVIOR_OPNAMES
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

    terminal_index = previous_non_ignored_index(instructions, end_index - 1, start_index)
    if terminal_index is None:
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
    if target_index is None or target_index > end_index:
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

    if not has_only_ignored_instructions(instructions, false_return_end, end_index):
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

    return_index = start_index
    if instructions[start_index].opname != "RETURN_CONST":
        return_index = start_index + 1
    if instructions[return_index].starts_line is not None:
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

def same_expression(left: ast.expr, right: ast.expr) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )

def find_chained_comparison_jump(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in IGNORED_BEHAVIOR_OPNAMES:
            continue
        if is_forward_conditional_jump(instruction) and is_false_jump(instruction.opname):
            return index
        if instruction.opname == "POP_TOP":
            continue
        if comparison_expression_op(instruction.opname):
            continue
        return None
    return None

def comparison_expression_op(opname: str) -> bool:
    return opname in {
        "CACHE",
        "COMPARE_OP",
        "COPY",
        "IS_OP",
        "LOAD_CONST",
        "LOAD_DEREF",
        "LOAD_FAST",
        "LOAD_FAST_BORROW",
        "LOAD_FAST_CHECK",
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "SWAP",
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
        return None
    if jump.argval != instructions[next_false_index].offset:
        return None

    return next_non_ignored_index(instructions, jump_index + 1, next_false_index)

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
        if instruction.opname in {"LOAD_DEREF", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME"}:
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

def find_repeated_condition_start(
    instructions: list[Instruction],
    cursor: int,
    repeated_jump_index: int,
) -> int | None:
    if repeated_jump_index <= cursor + 1:
        return None

    condition_line = find_condition_line(instructions, cursor)
    if condition_line is not None:
        for index in range(repeated_jump_index - 1, cursor, -1):
            if instructions[index].starts_line == condition_line:
                return index

    previous_index = repeated_jump_index - 1
    if previous_index <= cursor:
        return None
    return previous_index

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
