import ast
from abc import ABC
from dataclasses import dataclass
from typing import Any

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import skip_ignorable_instructions
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.stack_names import NO_VALUE_OPS
from pyc2py.decompiler.recover import (
    make_dict_comp,
    make_generator,
    make_generator_exp,
    make_list_comp,
    make_set_comp,
)
from pyc2py.decompiler.runtime import HiddenLocalRestore, coerce_expr
from pyc2py.decompiler.structures import (
    find_for_loop_pattern,
    find_loop_back_jump,
    invert_condition,
    is_forward_conditional_jump,
)

COMPREHENSION_BUILD_OPS = frozenset(
    {
        "BUILD_DICT",
        "BUILD_LIST",
        "BUILD_MAP",
        "BUILD_SET",
    }
)


@dataclass(frozen=True, slots=True)
class InlinedComprehensionShape:
    save_instructions: tuple[Instruction, ...]
    build_instruction: Instruction
    for_iter: Instruction
    for_iter_index: int
    loop_end_index: int


INLINE_CLEANUP_OPS = frozenset({"SWAP", "POP_TOP", "STORE_FAST", "RERAISE"})


def skip_inline_cleanup_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    after_index: int,
) -> int:
    if after_index >= len(instructions):
        return after_index
    jump = instructions[after_index]
    if jump.opname != "JUMP_FORWARD":
        return after_index
    join_index = offset_to_index.get(int(jump.argval))
    if join_index is None or instructions[join_index - 1].opname != "RERAISE":
        return after_index
    if any(
        instruction.opname not in INLINE_CLEANUP_OPS
        for instruction in instructions[after_index + 1 : join_index]
    ):
        return after_index
    return join_index


@dataclass(frozen=True, slots=True)
class AsyncInlinedComprehensionShape:
    save_instructions: tuple[Instruction, ...]
    build_instruction: Instruction
    get_anext: Instruction
    get_anext_index: int
    body_start_index: int
    loop_end_index: int


@dataclass(frozen=True, slots=True)
class InlinedComprehensionHeader:
    save_instructions: tuple[Instruction, ...]
    build_instruction: Instruction
    cursor: int


@dataclass(frozen=True, slots=True)
class InlinedComprehensionBody:
    target: ast.expr
    body_start: int
    back_jump_index: int
    update_index: int | None


def is_valid_inlined_first_swap(
    instruction: Instruction,
    save_count: int,
) -> bool:
    if instruction.opname != "SWAP":
        return False
    return instruction.arg in {2, save_count + 1}


class ControlComprehensionRecoveryMixin(DecompilerContext, ABC):
    def try_translate_legacy_list_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
        target: ast.expr,
        body_start: int,
        iterable: ast.expr,
    ) -> int | None:
        if not self.stack:
            return None
        container = self.stack[-1]
        if not is_empty_list_literal(container):
            return None

        pattern = find_for_loop_pattern(
            instructions, offset_to_index, cursor, end_index
        )
        if pattern is None:
            return None

        generator = make_generator(target=target, iterator=iterable)
        comprehension = self.make_legacy_list_comprehension(
            instructions,
            offset_to_index,
            body_start,
            pattern.body_end_index,
            int(instructions[pattern.for_iter_index].offset),
            [generator],
        )
        if comprehension is None:
            return None

        self.stack.pop()
        self.stack.append(comprehension)
        return pattern.after_index

    def make_legacy_list_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        start_index: int,
        end_index: int,
        loop_offset: int,
        generators: list[ast.comprehension],
    ) -> ast.ListComp | None:
        cursor = self.add_legacy_comprehension_conditions(
            instructions,
            start_index,
            end_index,
            loop_offset,
            generators[-1],
        )
        nested = self.find_nested_legacy_comprehension_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if nested is not None:
            nested_get_iter, nested_body_start, nested_after, nested_generator = nested
            if nested_after != end_index:
                return None
            return self.make_legacy_list_comprehension(
                instructions,
                offset_to_index,
                nested_body_start,
                end_index - 1,
                int(instructions[nested_get_iter + 1].offset),
                [*generators, nested_generator],
            )

        update_index = self.find_legacy_list_append_update(
            instructions,
            cursor,
            end_index,
            len(generators),
        )
        if update_index is None:
            return None

        element = self.evaluate_expression_range(instructions, cursor, update_index)
        if element is None:
            return None
        return make_list_comp(element, generators)

    def find_legacy_list_append_update(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
        generator_count: int,
    ) -> int | None:
        update_index = self.previous_non_ignorable_index(
            instructions,
            end_index - 1,
            cursor - 1,
        )
        if update_index is None:
            return None

        update = instructions[update_index]
        if update.opname != "LIST_APPEND":
            return None
        if int(update.arg or 0) != generator_count + 1:
            return None
        return update_index

    def add_legacy_comprehension_conditions(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        loop_offset: int,
        generator: ast.comprehension,
    ) -> int:
        cursor = start_index
        while cursor < end_index:
            condition_index = self.find_legacy_condition_jump(
                instructions,
                cursor,
                end_index,
                loop_offset,
            )
            if condition_index is None:
                return cursor
            condition = self.evaluate_expression_range(
                instructions,
                cursor,
                condition_index,
            )
            if condition is None:
                return cursor
            if "IF_TRUE" in instructions[condition_index].opname:
                condition = invert_condition(condition)
            generator.ifs.append(condition)
            cursor = condition_index + 1
        return cursor

    def find_legacy_condition_jump(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
        loop_offset: int,
    ) -> int | None:
        for index in range(start_index, end_index):
            instruction = instructions[index]
            if instruction.opname == "LIST_APPEND":
                return None
            if "JUMP" not in instruction.opname:
                continue
            if (
                "IF_FALSE" not in instruction.opname
                and "IF_TRUE" not in instruction.opname
            ):
                continue
            if instruction.argval != loop_offset:
                continue
            return index
        return None

    def find_nested_legacy_comprehension_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        start_index: int,
        end_index: int,
    ) -> tuple[int, int, int, ast.comprehension] | None:
        for get_iter_index in range(start_index, end_index):
            if instructions[get_iter_index].opname != "GET_ITER":
                continue
            pattern = find_for_loop_pattern(
                instructions,
                offset_to_index,
                get_iter_index,
                end_index,
            )
            if pattern is None:
                continue
            iterable = self.evaluate_expression_range(
                instructions,
                start_index,
                get_iter_index,
            )
            if iterable is None:
                return None
            target, body_start = self.read_for_loop_target(
                instructions,
                pattern.for_iter_index + 1,
                pattern.body_end_index,
            )
            if target is None:
                return None
            generator = make_generator(target=target, iterator=iterable)
            return get_iter_index, body_start, pattern.after_index, generator
        return None

    def try_translate_inlined_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        shape = self.read_inlined_comprehension_shape(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if shape is None:
            return None

        body = self.read_inlined_comprehension_body(instructions, shape)
        if body is None:
            return None

        iterable = coerce_expr(self.pop_or_none())
        outer_generator = make_generator(target=body.target, iterator=iterable)
        if body.update_index is None:
            comprehension = self.make_nested_inlined_comprehension(
                instructions,
                offset_to_index,
                shape.build_instruction,
                body.body_start,
                body.back_jump_index,
                [outer_generator],
                int(shape.for_iter.offset),
            )
        else:
            comprehension = self.make_inlined_comprehension(
                instructions,
                offset_to_index,
                shape.build_instruction,
                body.body_start,
                body.update_index,
                [outer_generator],
                int(shape.for_iter.offset),
            )
        if comprehension is None:
            return None

        for save_instruction in reversed(shape.save_instructions):
            self.stack.append(HiddenLocalRestore(name=str(save_instruction.argval)))
        self.stack.append(comprehension)
        after_index = shape.loop_end_index + 1
        if (
            self.version is not None
            and self.version >= (3, 13)
            and after_index < end_index
            and instructions[after_index].opname == "POP_TOP"
        ):
            after_index += 1
        return skip_inline_cleanup_handler(instructions, offset_to_index, after_index)

    def make_function_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        for_iter_index: int,
        iterable: ast.expr,
    ) -> ast.expr | None:
        for_iter = instructions[for_iter_index]
        loop_end_index = offset_to_index.get(int(for_iter.argval))
        build = next(
            (
                instruction
                for instruction in reversed(instructions[:for_iter_index])
                if instruction.opname in {*COMPREHENSION_BUILD_OPS, "RETURN_GENERATOR"}
            ),
            None,
        )
        if build is None or loop_end_index is None:
            return None
        shape = InlinedComprehensionShape(
            save_instructions=(),
            build_instruction=build,
            for_iter=for_iter,
            for_iter_index=for_iter_index,
            loop_end_index=loop_end_index,
        )
        body = self.read_inlined_comprehension_body(instructions, shape)
        if body is None:
            return None
        generators = [make_generator(target=body.target, iterator=iterable)]
        if body.update_index is None:
            return self.make_nested_inlined_comprehension(
                instructions,
                offset_to_index,
                build,
                body.body_start,
                body.back_jump_index,
                generators,
                int(for_iter.offset),
            )
        return self.make_inlined_comprehension(
            instructions,
            offset_to_index,
            build,
            body.body_start,
            body.update_index,
            generators,
            int(for_iter.offset),
        )

    def make_async_function_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        get_anext_index: int,
        iterable: ast.expr,
    ) -> ast.expr | None:
        end_index = len(instructions)
        build = next(
            (
                instruction
                for instruction in instructions[:get_anext_index]
                if instruction.opname in COMPREHENSION_BUILD_OPS
            ),
            None,
        )
        if build is None or not self.is_async_inlined_next_sequence(
            instructions, get_anext_index, end_index
        ):
            return None
        end_send_index = self.async_inlined_end_send_index(
            instructions, offset_to_index, get_anext_index, end_index
        )
        if end_send_index is None:
            return None
        target, body_start = self.read_for_loop_target(
            instructions, end_send_index + 1, end_index
        )
        loop_offset = int(instructions[get_anext_index].offset)
        back_jump_index = find_loop_back_jump(
            instructions, {loop_offset}, body_start, end_index
        )
        if target is None or back_jump_index is None:
            return None
        update_index = self.find_inlined_comprehension_update(
            instructions, build, body_start, back_jump_index
        )
        if update_index is None:
            return None
        return self.make_inlined_comprehension(
            instructions,
            offset_to_index,
            build,
            body_start,
            update_index,
            [make_generator(target=target, iterator=iterable, is_async=True)],
            loop_offset,
        )

    def read_inlined_comprehension_body(
        self,
        instructions: list[Instruction],
        shape: InlinedComprehensionShape,
    ) -> InlinedComprehensionBody | None:
        target, body_start = self.read_for_loop_target(
            instructions,
            shape.for_iter_index + 1,
            shape.loop_end_index,
        )
        if target is None:
            return None

        back_jump_index = self.find_inlined_comprehension_back_jump(
            instructions, shape.for_iter, body_start, shape.loop_end_index
        )
        if back_jump_index is None:
            return None

        update_index = self.find_inlined_comprehension_update(
            instructions, shape.build_instruction, body_start, back_jump_index
        )
        return InlinedComprehensionBody(
            target=target,
            body_start=body_start,
            back_jump_index=back_jump_index,
            update_index=update_index,
        )

    def try_translate_async_inlined_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        shape = self.read_async_inlined_comprehension_shape(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if shape is None:
            return None

        target, body_start = self.read_for_loop_target(
            instructions,
            shape.body_start_index,
            shape.loop_end_index,
        )
        if target is None:
            return None

        back_jump_index = find_loop_back_jump(
            instructions,
            {int(shape.get_anext.offset)},
            body_start,
            shape.loop_end_index,
        )
        if back_jump_index is None:
            return None

        update_index = self.find_inlined_comprehension_update(
            instructions,
            shape.build_instruction,
            body_start,
            back_jump_index,
        )
        if update_index is None:
            return None

        iterable = coerce_expr(self.pop_or_none())
        generator = make_generator(target=target, iterator=iterable, is_async=True)
        comprehension = self.make_inlined_comprehension(
            instructions,
            offset_to_index,
            shape.build_instruction,
            body_start,
            update_index,
            [generator],
            int(shape.get_anext.offset),
        )
        if comprehension is None:
            return None

        for save_instruction in reversed(shape.save_instructions):
            self.stack.append(HiddenLocalRestore(name=str(save_instruction.argval)))
        self.stack.append(comprehension)
        return skip_inline_cleanup_handler(
            instructions, offset_to_index, shape.loop_end_index + 1
        )

    def make_inlined_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        build_instruction: Instruction,
        body_start: int,
        update_index: int,
        generators: list[ast.comprehension],
        loop_offset: int,
    ) -> ast.expr | None:
        update_instruction = instructions[update_index]
        structured = self.read_inlined_body_with_tests(
            instructions, body_start, update_index, loop_offset
        )
        if structured is not None:
            generators[-1].ifs.extend(structured[0])
            return self.make_inlined_comprehension_expr(
                build_instruction.opname,
                update_instruction.opname,
                structured[1],
                generators,
            )
        expression_start, conditions = self.read_inlined_comprehension_conditions(
            instructions,
            offset_to_index,
            body_start,
            update_index,
            loop_offset,
        )
        generators[-1].ifs.extend(conditions)
        expressions = self.evaluate_stack_range(
            instructions, expression_start, update_index
        )
        return self.make_inlined_comprehension_expr(
            build_instruction.opname,
            update_instruction.opname,
            expressions,
            generators,
        )

    def read_inlined_body_with_tests(
        self,
        instructions: list[Instruction],
        body_start: int,
        update_index: int,
        loop_offset: int,
    ) -> tuple[list[ast.expr], list[Any]] | None:
        if not any(
            is_forward_conditional_jump(instructions[index])
            and not is_loop_back_jump(instructions[index + 1], loop_offset)
            for index in range(body_start, update_index)
        ):
            return None
        if any(
            is_loop_back_test(instructions[index], loop_offset)
            or (
                is_forward_conditional_jump(instructions[index])
                and instructions[index].argval > instructions[update_index].offset
            )
            for index in range(body_start, update_index)
        ):
            return None
        child = self.make_child()
        child.stack.clear()
        child.loop_continue_offsets = frozenset({loop_offset})
        body = child.translate_range(instructions, body_start, update_index)
        conditions: list[ast.expr] = []
        for statement in body:
            condition = condition_kept_by(statement)
            if condition is None:
                return None
            conditions.append(condition)
        if not child.stack:
            return None
        self.warnings.extend(child.warnings)
        return conditions, child.stack.to_list()

    def make_nested_inlined_comprehension(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        build_instruction: Instruction,
        body_start: int,
        back_jump_index: int,
        generators: list[ast.comprehension],
        outer_loop_offset: int,
    ) -> ast.expr | None:
        nested = self.find_nested_inlined_comprehension_loop(
            instructions,
            offset_to_index,
            body_start,
            back_jump_index,
            outer_loop_offset,
        )
        if nested is None:
            return None

        (
            nested_get_iter,
            nested_body_start,
            nested_after,
            nested_generator,
            outer_conditions,
        ) = nested
        generators[-1].ifs.extend(outer_conditions)
        next_index = skip_ignorable_instructions(
            instructions,
            nested_after,
            back_jump_index,
        )
        if next_index != back_jump_index:
            return None

        nested_for_iter = instructions[nested_get_iter + 1]
        nested_loop_end = offset_to_index.get(int(nested_for_iter.argval))
        if nested_loop_end is None:
            return None
        nested_back_jump = self.find_inlined_comprehension_back_jump(
            instructions,
            nested_for_iter,
            nested_body_start,
            nested_loop_end,
        )
        if nested_back_jump is None:
            return None

        update_index = self.find_inlined_comprehension_update(
            instructions,
            build_instruction,
            nested_body_start,
            nested_back_jump,
        )
        if update_index is None:
            return self.make_nested_inlined_comprehension(
                instructions,
                offset_to_index,
                build_instruction,
                nested_body_start,
                nested_back_jump,
                [*generators, nested_generator],
                int(instructions[nested_get_iter + 1].offset),
            )

        return self.make_inlined_comprehension(
            instructions,
            offset_to_index,
            build_instruction,
            nested_body_start,
            update_index,
            [*generators, nested_generator],
            int(instructions[nested_get_iter + 1].offset),
        )

    def find_nested_inlined_comprehension_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        start_index: int,
        end_index: int,
        loop_offset: int,
    ) -> tuple[int, int, int, ast.comprehension, list[ast.expr]] | None:
        for get_iter_index in range(start_index, end_index):
            if instructions[get_iter_index].opname != "GET_ITER":
                continue

            expression_start, conditions = self.read_inlined_comprehension_conditions(
                instructions,
                offset_to_index,
                start_index,
                get_iter_index,
                loop_offset,
            )
            iterable = self.evaluate_expression_range(
                instructions,
                expression_start,
                get_iter_index,
            )
            if iterable is None:
                return None

            pattern = find_for_loop_pattern(
                instructions,
                offset_to_index,
                get_iter_index,
                end_index,
            )
            if pattern is None:
                continue

            target, body_start = self.read_for_loop_target(
                instructions,
                pattern.for_iter_index + 1,
                pattern.body_end_index,
            )
            if target is None:
                return None

            generator = make_generator(target=target, iterator=iterable)
            return (
                get_iter_index,
                body_start,
                pattern.after_index,
                generator,
                conditions,
            )
        return None

    def read_inlined_comprehension_conditions(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        body_start: int,
        update_index: int,
        loop_offset: int,
    ) -> tuple[int, list[ast.expr]]:
        jumps = inlined_condition_jumps(
            instructions,
            body_start,
            update_index,
            loop_offset,
        )
        for count in range(len(jumps), 0, -1):
            condition = self.build_looped_condition(
                instructions,
                offset_to_index,
                body_start,
                jumps[:count],
                loop_offset,
            )
            if condition is None:
                continue
            expression_start = jumps[count - 1] + 1
            if is_loop_back_jump(instructions[expression_start], loop_offset):
                expression_start += 1
            return expression_start, [condition]
        return body_start, []

    def find_inlined_comprehension_back_jump(
        self,
        instructions: list[Instruction],
        for_iter: Instruction,
        body_start: int,
        loop_end_index: int,
    ) -> int | None:
        back_jump_index = self.previous_non_ignorable_index(
            instructions, loop_end_index - 1, body_start - 1
        )
        if back_jump_index is None:
            return None

        back_jump = instructions[back_jump_index]
        if back_jump.opname != "JUMP_BACKWARD":
            return None
        if int(back_jump.argval) != for_iter.offset:
            return None
        return back_jump_index

    def find_inlined_comprehension_update(
        self,
        instructions: list[Instruction],
        build_instruction: Instruction,
        body_start: int,
        back_jump_index: int,
    ) -> int | None:
        update_index = self.previous_non_ignorable_index(
            instructions, back_jump_index - 1, body_start - 1
        )
        if update_index is not None and build_instruction.opname == "RETURN_GENERATOR":
            update_index = find_loop_yield(instructions, body_start, update_index)
        if update_index is None:
            return None

        update_instruction = instructions[update_index]
        if not self.is_matching_comprehension_update(
            build_instruction.opname, update_instruction.opname
        ):
            return None
        return update_index

    def read_inlined_comprehension_shape(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> InlinedComprehensionShape | None:
        if cursor + 5 >= end_index:
            return None

        header = self.read_inlined_comprehension_header(
            instructions,
            cursor + 1,
            end_index,
            min_remaining=3,
        )
        if header is None:
            header = self.read_inlined_comprehension_header_before(
                instructions,
                cursor,
                end_index,
            )
        if header is None:
            return None

        for_iter_index = header.cursor + 3
        if instructions[for_iter_index].opname == "GET_ITER":
            for_iter_index += 1
            if for_iter_index >= end_index:
                return None
        for_iter = instructions[for_iter_index]
        if for_iter.opname != "FOR_ITER":
            return None

        loop_end_index = offset_to_index.get(int(for_iter.argval))
        if loop_end_index is None or loop_end_index >= end_index:
            return None
        if instructions[loop_end_index].opname != "END_FOR":
            return None
        return InlinedComprehensionShape(
            save_instructions=header.save_instructions,
            build_instruction=header.build_instruction,
            for_iter=for_iter,
            for_iter_index=for_iter_index,
            loop_end_index=loop_end_index,
        )

    def read_async_inlined_comprehension_shape(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> AsyncInlinedComprehensionShape | None:
        if not self.starts_async_inlined_comprehension(
            instructions,
            cursor,
            end_index,
        ):
            return None

        header = self.read_inlined_comprehension_header(
            instructions,
            cursor + 1,
            end_index,
            min_remaining=4,
        )
        if header is None:
            return None

        get_anext_index = header.cursor + 3
        get_anext = instructions[get_anext_index]

        if not self.is_async_inlined_next_sequence(
            instructions,
            get_anext_index,
            end_index,
        ):
            return None

        end_send_index = self.async_inlined_end_send_index(
            instructions,
            offset_to_index,
            get_anext_index,
            end_index,
        )
        if end_send_index is None:
            return None

        loop_end_index = self.find_async_inlined_comprehension_end(
            instructions,
            get_anext_index,
            end_index,
        )
        if loop_end_index is None:
            return None

        return AsyncInlinedComprehensionShape(
            save_instructions=header.save_instructions,
            build_instruction=header.build_instruction,
            get_anext=get_anext,
            get_anext_index=get_anext_index,
            body_start_index=end_send_index + 1,
            loop_end_index=loop_end_index,
        )

    def async_inlined_end_send_index(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        get_anext_index: int,
        end_index: int,
    ) -> int | None:
        send_instruction = instructions[get_anext_index + 2]
        end_send_index = offset_to_index.get(int(send_instruction.argval))
        if end_send_index is None or end_send_index >= end_index:
            return None
        if instructions[end_send_index].opname != "END_SEND":
            return None
        return end_send_index

    def starts_async_inlined_comprehension(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> bool:
        return cursor + 8 < end_index and instructions[cursor].opname == "GET_AITER"

    def read_inlined_comprehension_header(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
        min_remaining: int,
    ) -> InlinedComprehensionHeader | None:
        save_instructions, cursor = self.read_inlined_hidden_local_saves(
            instructions,
            cursor,
            end_index,
        )
        if not save_instructions or cursor + min_remaining >= end_index:
            return None

        first_swap = instructions[cursor]
        build_instruction = instructions[cursor + 1]
        second_swap = instructions[cursor + 2]
        if not is_valid_inlined_first_swap(first_swap, len(save_instructions)):
            return None
        if second_swap.opname != "SWAP" or second_swap.arg != 2:
            return None
        if build_instruction.opname not in COMPREHENSION_BUILD_OPS:
            return None
        return InlinedComprehensionHeader(
            save_instructions=tuple(save_instructions),
            build_instruction=build_instruction,
            cursor=cursor,
        )

    def read_inlined_comprehension_header_before(
        self,
        instructions: list[Instruction],
        get_iter_index: int,
        end_index: int,
    ) -> InlinedComprehensionHeader | None:
        if get_iter_index < 4 or get_iter_index > end_index:
            return None
        second_swap = instructions[get_iter_index - 1]
        build_instruction = instructions[get_iter_index - 2]
        first_swap = instructions[get_iter_index - 3]
        if second_swap.opname != "SWAP" or second_swap.arg != 2:
            return None
        if build_instruction.opname not in COMPREHENSION_BUILD_OPS:
            return None
        save_count = 0
        while (
            get_iter_index - 4 - save_count >= 0
            and instructions[get_iter_index - 4 - save_count].opname
            == "LOAD_FAST_AND_CLEAR"
        ):
            save_count += 1
        if save_count == 0:
            return None
        if not is_valid_inlined_first_swap(first_swap, save_count):
            return None
        return InlinedComprehensionHeader(
            save_instructions=tuple(
                instructions[get_iter_index - 3 - save_count : get_iter_index - 3]
            ),
            build_instruction=build_instruction,
            cursor=get_iter_index - 3,
        )

    def is_async_inlined_next_sequence(
        self,
        instructions: list[Instruction],
        get_anext_index: int,
        end_index: int,
    ) -> bool:
        if get_anext_index + 3 >= end_index:
            return False
        if instructions[get_anext_index].opname != "GET_ANEXT":
            return False
        if instructions[get_anext_index + 1].opname != "LOAD_CONST":
            return False
        return instructions[get_anext_index + 2].opname == "SEND"

    def find_async_inlined_comprehension_end(
        self,
        instructions: list[Instruction],
        get_anext_index: int,
        end_index: int,
    ) -> int | None:
        back_jump_index = find_loop_back_jump(
            instructions,
            {int(instructions[get_anext_index].offset)},
            get_anext_index + 1,
            end_index,
        )
        if back_jump_index is None:
            return None

        for index in range(back_jump_index + 1, end_index):
            if instructions[index].opname != "END_ASYNC_FOR":
                continue
            following = instructions[index + 1] if index + 1 < end_index else None
            if following is None or following.opname not in {
                "JUMP_BACKWARD",
                "JUMP_BACKWARD_NO_INTERRUPT",
            }:
                return index
        return None

    def read_inlined_hidden_local_saves(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> tuple[list[Instruction], int]:
        saves: list[Instruction] = []
        for _index in range(32):
            cursor = skip_ignorable_instructions(instructions, cursor, end_index)
            if cursor >= end_index:
                return saves, cursor
            instruction = instructions[cursor]
            if instruction.opname != "LOAD_FAST_AND_CLEAR":
                return saves, cursor
            saves.append(instruction)
            cursor += 1
        return [], end_index

    def previous_non_ignorable_index(
        self,
        instructions: list[Instruction],
        start_index: int,
        floor_index: int,
    ) -> int | None:
        for index in range(start_index, floor_index, -1):
            if instructions[index].opname in NO_VALUE_OPS:
                continue
            return index
        return None

    def is_matching_comprehension_update(
        self, build_opname: str, update_opname: str
    ) -> bool:
        return (
            (build_opname == "BUILD_LIST" and update_opname == "LIST_APPEND")
            or (build_opname == "BUILD_LIST" and update_opname == "LIST_EXTEND")
            or (build_opname == "BUILD_SET" and update_opname == "SET_ADD")
            or (build_opname == "BUILD_SET" and update_opname == "SET_UPDATE")
            or (
                build_opname in {"BUILD_DICT", "BUILD_MAP"}
                and update_opname == "MAP_ADD"
            )
            or (
                build_opname in {"BUILD_DICT", "BUILD_MAP"}
                and update_opname == "DICT_UPDATE"
            )
            or (build_opname == "RETURN_GENERATOR" and update_opname == "YIELD_VALUE")
        )

    def make_inlined_comprehension_expr(
        self,
        build_opname: str,
        update_opname: str,
        expressions: list[Any],
        generators: list[ast.comprehension],
    ) -> ast.expr | None:
        if update_opname in {"LIST_APPEND", "SET_ADD", "YIELD_VALUE"}:
            if not expressions:
                return None
            elt = coerce_expr(expressions[-1])
            if update_opname == "YIELD_VALUE":
                return make_generator_exp(elt, generators)
            if build_opname == "BUILD_LIST":
                return make_list_comp(elt, generators)
            return make_set_comp(elt, generators)

        if update_opname in {"LIST_EXTEND", "SET_UPDATE"}:
            if not expressions:
                return None
            elt = ast.Starred(value=coerce_expr(expressions[-1]), ctx=ast.Load())
            if update_opname == "LIST_EXTEND":
                return make_list_comp(elt, generators)
            return make_set_comp(elt, generators)

        if update_opname == "DICT_UPDATE" and expressions:
            return make_dict_comp(coerce_expr(expressions[-1]), None, generators)

        if update_opname == "MAP_ADD" and len(expressions) >= 2:
            key = coerce_expr(expressions[-2])
            value = coerce_expr(expressions[-1])
            return make_dict_comp(key, value, generators)
        return None

    def evaluate_stack_range(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> list[Any]:
        child = self.make_child()
        child.stack.clear()
        child.translate_range(instructions, start_index, end_index)
        self.warnings.extend(child.warnings)
        return child.stack.to_list()

    def translate_isolated_child_statements(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> list[ast.stmt]:
        child = self.make_child()
        child.stack.clear()
        body = child.translate_range(instructions, start_index, end_index)
        self.warnings.extend(child.warnings)
        return body


def find_loop_yield(
    instructions: list[Instruction],
    body_start: int,
    pop_index: int,
) -> int | None:
    if pop_index - 2 < body_start:
        return None
    names = [item.opname for item in instructions[pop_index - 2 : pop_index + 1]]
    return pop_index - 2 if names == ["YIELD_VALUE", "RESUME", "POP_TOP"] else None


def condition_kept_by(statement: ast.stmt) -> ast.expr | None:
    if not isinstance(statement, ast.If):
        return None
    if not statement.orelse:
        if len(statement.body) == 1 and isinstance(statement.body[0], ast.Continue):
            return invert_condition(statement.test)
        return None
    kept = [conditions_kept_by(statement.body), conditions_kept_by(statement.orelse)]
    if kept[0] is None or kept[1] is None:
        return None
    return ast.IfExp(test=statement.test, body=kept[0], orelse=kept[1])


def conditions_kept_by(statements: list[ast.stmt]) -> ast.expr | None:
    values: list[ast.expr] = []
    for statement in statements:
        value = condition_kept_by(statement)
        if value is None:
            return None
        values.append(value)
    if len(values) == 1:
        return values[0]
    return ast.BoolOp(op=ast.And(), values=values) if values else None


def inlined_condition_jumps(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    loop_offset: int,
) -> list[int]:
    jumps: list[int] = []
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if "JUMP" not in instruction.opname:
            continue
        if is_forward_conditional_jump(instruction) or is_loop_back_test(
            instruction, loop_offset
        ):
            jumps.append(index)
            continue
        if (
            jumps
            and jumps[-1] == index - 1
            and is_loop_back_jump(instruction, loop_offset)
        ):
            continue
        break
    return jumps


def is_loop_back_test(instruction: Instruction, loop_offset: int) -> bool:
    return (
        instruction.opname.startswith("POP_JUMP_BACKWARD_IF_")
        and instruction.argval == loop_offset
    )


def is_loop_back_jump(instruction: Instruction, loop_offset: int) -> bool:
    return instruction.opname == "JUMP_BACKWARD" and instruction.argval == loop_offset


def is_empty_list_literal(value: Any) -> bool:
    return isinstance(value, ast.List) and not value.elts
