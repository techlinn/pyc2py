import ast
from typing import TypeGuard

from pyc2py.astree import make_name
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import skip_ignorable_instructions
from pyc2py.decompiler.async_structures import (
    find_async_for_loop_pattern,
    make_async_for_statement,
)
from pyc2py.decompiler.control_comprehension import ControlComprehensionRecoveryMixin
from pyc2py.decompiler.opcodes.stack_names import STORE_OPS
from pyc2py.decompiler.opcodes.values import is_zero_constant
from pyc2py.decompiler.runtime import coerce_expr
from pyc2py.decompiler.structures import (
    find_for_loop_pattern,
    find_legacy_for_loop_pattern,
    invert_condition,
    is_false_jump,
    is_forward_conditional_jump,
    loop_entry_offsets,
    skip_loop_cleanup,
)


class ControlLoopRecoveryMixin(ControlComprehensionRecoveryMixin):
    def try_translate_modern_infinite_while_loop(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        back_jump_index = previous_non_metadata_index(instructions, end_index - 1)
        if back_jump_index is None or back_jump_index <= cursor:
            return None

        back_jump = instructions[back_jump_index]
        if back_jump.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_BACKWARD"}:
            return None
        if back_jump.argval != instructions[cursor].offset:
            return None

        body = self.translate_loop_child_statements(
            instructions,
            cursor,
            back_jump_index,
            continue_offset=int(instructions[cursor].offset),
            break_index=end_index,
        )
        self.statements.append(
            ast.While(
                test=ast.Constant(value=True),
                body=body or [ast.Pass()],
                orelse=[],
            )
        )
        return end_index

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

        body = self.translate_loop_child_statements(
            instructions,
            body_start,
            pattern.body_end_index,
            continue_offset=int(instructions[pattern.for_iter_index].offset),
            break_index=pattern.after_index,
            extra_continue_offsets=frozenset(
                loop_entry_offsets(instructions, cursor, pattern.for_iter_index)
            ),
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
    ) -> list[ast.stmt]:
        child = self.make_child()
        child.loop_continue_offsets = (
            self.loop_continue_offsets
            | frozenset({continue_offset})
            | extra_continue_offsets
        )
        child.loop_none_return_is_break = loop_exit_is_none_return(
            instructions,
            break_index,
        )
        if break_index < len(instructions):
            child.loop_break_offsets = self.loop_break_offsets | frozenset(
                {int(instructions[break_index].offset)}
            )

        body = child.translate_range(instructions, start_index, end_index)
        return simplify_loop_guard_continue(body)

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
        body = self.translate_child_statements(
            instructions,
            body_start,
            pattern.body_end_index,
        )
        self.statements.append(
            make_async_for_statement(
                target,
                iterable,
                body,
                empty_body_is_continue=pattern.empty_body_is_continue,
            )
        )
        return pattern.after_index

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
            return make_name(str(instruction.argval), ast.Store()), start_index + 1
        if instruction.opname in {"UNPACK_SEQUENCE", "UNPACK_TUPLE"}:
            return self.read_unpack_loop_target(instructions, start_index, end_index)
        return None, start_index

    def read_unpack_loop_target(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> tuple[ast.expr | None, int]:
        count = int(instructions[start_index].arg or 0)
        target_end = start_index + 1 + count
        if target_end > end_index:
            return None, start_index
        if count == 0:
            return ast.List(elts=[], ctx=ast.Store()), target_end

        targets: list[ast.expr] = []
        cursor = start_index + 1
        for _index in range(count):
            cursor = skip_ignorable_instructions(instructions, cursor, end_index)
            if cursor >= end_index:
                return None, start_index
            instruction = instructions[cursor]
            if instruction.opname not in STORE_OPS:
                return None, start_index
            targets.append(make_name(str(instruction.argval), ast.Store()))
            cursor += 1
        return ast.Tuple(elts=targets, ctx=ast.Store()), cursor


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


def simplify_loop_guard_continue(statements: list[ast.stmt]) -> list[ast.stmt]:
    for index, statement in enumerate(statements):
        if not is_guard_continue(statement):
            continue
        if index + 1 >= len(statements):
            return statements
        return [
            *statements[:index],
            ast.If(
                test=invert_condition(statement.test),
                body=statements[index + 1 :] or [ast.Pass()],
                orelse=[],
            ),
        ]
    return statements


def is_guard_continue(statement: ast.stmt) -> TypeGuard[ast.If]:
    if not isinstance(statement, ast.If):
        return False
    if statement.orelse:
        return False
    if len(statement.body) != 1:
        return False
    return isinstance(statement.body[0], ast.Continue)
