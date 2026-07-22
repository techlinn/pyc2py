import ast
from dataclasses import dataclass

from pyc2py.bytecode.instruction import Instruction
from pyc2py.decompiler.structures import find_loop_back_jump


@dataclass(frozen=True, slots=True)
class AsyncForLoopPattern:
    target_start_index: int
    body_start_index: int
    body_end_index: int
    after_index: int
    empty_body_is_continue: bool = False


@dataclass(frozen=True, slots=True)
class AsyncForHeader:
    setup_except_index: int
    get_anext_index: int
    target_start_index: int
    target_end_index: int


ASYNC_FOR_HANDLER_PREFIX = {"CACHE", "EXTENDED_ARG", "NOP"}
ASYNC_FOR_CLEANUP_OPS = {
    "CLEANUP_THROW",
    "END_ASYNC_FOR",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
    "POP_TOP",
    "POP_EXCEPT",
    "POP_BLOCK",
}


def find_async_for_loop_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_aiter_index: int,
    end_index: int,
) -> AsyncForLoopPattern | None:
    modern = find_modern_async_for_loop_pattern(
        instructions,
        offset_to_index,
        get_aiter_index,
        end_index,
    )
    if modern is not None:
        return modern

    return find_legacy_async_for_loop_pattern(
        instructions,
        offset_to_index,
        get_aiter_index,
        end_index,
    )


def find_legacy_async_for_loop_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_aiter_index: int,
    end_index: int,
) -> AsyncForLoopPattern | None:
    header = find_async_for_header(instructions, get_aiter_index, end_index)
    if header is None:
        return None

    handler_start_index = find_async_for_handler_start(
        instructions,
        offset_to_index,
        header,
    )
    if handler_start_index is None:
        return None

    cleanup_index = find_async_for_cleanup_jump(
        instructions,
        offset_to_index,
        handler_start_index,
        end_index,
    )
    if cleanup_index is None:
        return None

    loop_offsets = {
        instructions[header.setup_except_index].offset,
        instructions[header.get_anext_index].offset,
    }
    body_span = find_async_for_body_span(
        instructions,
        offset_to_index,
        header,
        loop_offsets,
        cleanup_index,
    )
    if body_span is None:
        return None

    return make_legacy_async_for_pattern(
        instructions,
        end_index,
        header,
        handler_start_index,
        cleanup_index,
        loop_offsets,
        body_span,
    )


def find_modern_async_for_loop_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_aiter_index: int,
    end_index: int,
) -> AsyncForLoopPattern | None:
    get_anext_index = find_modern_async_for_get_anext_index(
        instructions,
        get_aiter_index,
        end_index,
    )
    if get_anext_index is None:
        return None

    end_send_index = find_modern_async_for_end_send_index(
        instructions,
        offset_to_index,
        get_anext_index,
        end_index,
    )
    if end_send_index is None:
        return None

    target_span = find_modern_async_for_target_span(
        instructions,
        end_send_index,
        end_index,
    )
    if target_span is None:
        return None

    target_start_index, target_end_index = target_span
    loop_offsets = {instructions[get_anext_index].offset}
    body_end_index = find_loop_back_jump(
        instructions,
        loop_offsets,
        target_end_index,
        end_index,
    )
    if body_end_index is None:
        return None

    cleanup_index = body_end_index + 1
    after_index = skip_modern_async_for_cleanup(
        instructions,
        cleanup_index,
        end_index,
    )
    if after_index is None:
        return None

    return AsyncForLoopPattern(
        target_start_index=target_start_index,
        body_start_index=target_end_index,
        body_end_index=body_end_index,
        after_index=after_index,
    )


def find_async_for_header(
    instructions: list[Instruction],
    get_aiter_index: int,
    end_index: int,
) -> AsyncForHeader | None:
    setup_except_index = find_async_for_setup_except_index(
        instructions,
        get_aiter_index,
        end_index,
    )
    if setup_except_index is None:
        return None

    get_anext_index = find_async_for_get_anext_index(
        instructions,
        setup_except_index,
        end_index,
    )
    if get_anext_index is None:
        return None

    target_span = find_async_for_header_target_span(
        instructions,
        get_anext_index,
        end_index,
    )
    if target_span is None:
        return None

    target_start_index, target_end_index = target_span

    return AsyncForHeader(
        setup_except_index=setup_except_index,
        get_anext_index=get_anext_index,
        target_start_index=target_start_index,
        target_end_index=target_end_index,
    )


def find_async_for_handler_start(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    header: AsyncForHeader,
) -> int | None:
    setup_except = instructions[header.setup_except_index]
    return offset_to_index.get(int(setup_except.argval))


def make_legacy_async_for_pattern(
    instructions: list[Instruction],
    end_index: int,
    header: AsyncForHeader,
    handler_start_index: int,
    cleanup_index: int,
    loop_offsets: set[int],
    body_span: tuple[int, int],
) -> AsyncForLoopPattern | None:
    body_start_index, body_end_index = body_span
    if cleanup_index <= body_start_index:
        return None

    return AsyncForLoopPattern(
        target_start_index=header.target_start_index,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        after_index=skip_async_for_cleanup(instructions, cleanup_index, end_index),
        empty_body_is_continue=(
            body_start_index == body_end_index
            and has_empty_async_for_continue_marker(
                instructions, handler_start_index, cleanup_index, loop_offsets
            )
        ),
    )


def find_modern_async_for_get_anext_index(
    instructions: list[Instruction],
    get_aiter_index: int,
    end_index: int,
) -> int | None:
    if get_aiter_index + 4 >= end_index:
        return None
    if instructions[get_aiter_index].opname != "GET_AITER":
        return None

    get_anext_index = skip_async_for_prefix(
        instructions,
        get_aiter_index + 1,
        end_index,
    )
    if not is_async_for_next_sequence(instructions, get_anext_index, end_index):
        return None

    return get_anext_index


def find_modern_async_for_end_send_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_anext_index: int,
    end_index: int,
) -> int | None:
    end_send_index = offset_to_index.get(int(instructions[get_anext_index + 2].argval))
    if end_send_index is None or end_send_index >= end_index:
        return None
    if instructions[end_send_index].opname != "END_SEND":
        return None

    return end_send_index


def find_modern_async_for_target_span(
    instructions: list[Instruction],
    end_send_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    target_start_index = end_send_index + 1
    target_end_index = find_async_for_target_end(
        instructions,
        target_start_index,
        end_index,
    )
    if target_end_index is None or target_end_index <= target_start_index:
        return None

    return target_start_index, target_end_index


def find_async_for_setup_except_index(
    instructions: list[Instruction],
    get_aiter_index: int,
    end_index: int,
) -> int | None:
    if get_aiter_index + 5 >= end_index:
        return None
    if instructions[get_aiter_index].opname != "GET_AITER":
        return None

    setup_except_index = skip_async_for_prefix(
        instructions, get_aiter_index + 1, end_index
    )
    if setup_except_index >= end_index:
        return None
    if instructions[setup_except_index].opname != "SETUP_EXCEPT":
        return None

    return setup_except_index


def find_async_for_get_anext_index(
    instructions: list[Instruction],
    setup_except_index: int,
    end_index: int,
) -> int | None:
    get_anext_index = skip_async_for_prefix(
        instructions, setup_except_index + 1, end_index
    )
    if not is_async_for_next_sequence(instructions, get_anext_index, end_index):
        return None

    return get_anext_index


def find_async_for_header_target_span(
    instructions: list[Instruction],
    get_anext_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    target_start_index = get_anext_index + 3
    target_end_index = find_async_for_target_end(
        instructions, target_start_index, end_index
    )
    if target_end_index is None or target_end_index + 1 >= end_index:
        return None
    if instructions[target_end_index].opname != "POP_BLOCK":
        return None

    return target_start_index, target_end_index


def is_async_for_next_sequence(
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
    return instructions[get_anext_index + 2].opname in {"YIELD_FROM", "SEND"}


def find_async_for_body_span(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    header: AsyncForHeader,
    loop_offsets: set[int],
    cleanup_index: int,
) -> tuple[int, int] | None:
    body_jump = instructions[header.target_end_index + 1]
    if body_jump.opname not in {
        "JUMP_FORWARD",
        "JUMP_ABSOLUTE",
        "JUMP_BACKWARD",
        "JUMP",
    }:
        return None
    if body_jump.argval in loop_offsets:
        body_start_index = header.target_end_index + 1
        return body_start_index, body_start_index

    resolved_body_start = offset_to_index.get(int(body_jump.argval))
    if resolved_body_start is None or resolved_body_start <= header.target_end_index:
        return None
    body_start_index = resolved_body_start

    body_end_index = find_loop_back_jump(
        instructions,
        loop_offsets,
        body_start_index,
        cleanup_index,
    )
    if body_end_index is None:
        return None

    return body_start_index, body_end_index


def skip_async_for_prefix(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index
    while (
        cursor < end_index and instructions[cursor].opname in ASYNC_FOR_HANDLER_PREFIX
    ):
        cursor += 1
    return cursor


def find_async_for_target_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = skip_async_for_prefix(instructions, start_index, end_index)
    if cursor >= end_index:
        return None
    instruction = instructions[cursor]
    if instruction.opname.startswith("STORE_"):
        return cursor + 1
    if instruction.opname not in {"UNPACK_SEQUENCE", "UNPACK_TUPLE"}:
        return None

    count = int(instruction.arg or 0)
    target_end = cursor + 1
    for _index in range(count):
        target_end = skip_async_for_prefix(instructions, target_end, end_index)
        if target_end >= end_index:
            return None
        if not instructions[target_end].opname.startswith("STORE_"):
            return None
        target_end += 1
    return target_end


def find_async_for_cleanup_jump(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    end_index: int,
) -> int | None:
    cursor = handler_start_index
    while cursor < end_index:
        instruction = instructions[cursor]
        if instruction.opname not in {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE"}:
            cursor += 1
            continue
        return offset_to_index.get(int(instruction.argval))
    return None


def skip_async_for_cleanup(
    instructions: list[Instruction],
    cleanup_index: int,
    end_index: int,
) -> int:
    cursor = cleanup_index
    while cursor < end_index and instructions[cursor].opname in ASYNC_FOR_CLEANUP_OPS:
        if instructions[cursor].opname == "POP_BLOCK":
            return cursor + 1
        cursor += 1
    return cursor


def skip_modern_async_for_cleanup(
    instructions: list[Instruction],
    cleanup_index: int,
    end_index: int,
) -> int | None:
    cursor = cleanup_index
    for _index in range(8):
        cursor = skip_async_for_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None
        instruction = instructions[cursor]
        if instruction.opname == "END_ASYNC_FOR":
            return cursor + 1
        if instruction.opname not in ASYNC_FOR_CLEANUP_OPS:
            return None
        cursor += 1
    return None


def has_empty_async_for_continue_marker(
    instructions: list[Instruction],
    handler_start_index: int,
    cleanup_index: int,
    loop_offsets: set[int],
) -> bool:
    loop_jumps = 0
    for index in range(handler_start_index, cleanup_index):
        instruction = instructions[index]
        if instruction.opname not in {
            "JUMP",
            "JUMP_ABSOLUTE",
            "JUMP_BACKWARD",
            "JUMP_FORWARD",
        }:
            continue
        if instruction.argval in loop_offsets:
            loop_jumps += 1

    return loop_jumps > 1


def make_async_for_statement(
    target: ast.expr,
    iterable: ast.expr,
    body: list[ast.stmt],
    *,
    empty_body_is_continue: bool = False,
) -> ast.AsyncFor:
    if not body and empty_body_is_continue:
        body = [ast.Continue()]
    return ast.AsyncFor(
        target=target,
        iter=iterable,
        body=body or [ast.Pass()],
        orelse=[],
        type_comment=None,
    )
