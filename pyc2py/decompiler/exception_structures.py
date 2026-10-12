import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import (
    ExceptionTableEntry,
    make_offset_index,
    parse_exception_table,
)
from pyc2py.bytecode.opcode_table import NAME_OPS
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.flow import (
    is_jump_op,
    is_terminal_op,
    terminal_tail_end_index,
)


@dataclass(frozen=True, slots=True)
class TryExceptPattern:
    body_start_index: int
    body_end_index: int
    handler_start_index: int
    handler_body_start_index: int
    handler_body_end_index: int
    exception_type_index: int
    after_index: int
    name: str | None = None


@dataclass(frozen=True, slots=True)
class LegacyTryExceptShape:
    body_start_index: int
    body_end_index: int
    handler_start_index: int
    after_index: int
    after_offset: int


@dataclass(frozen=True, slots=True)
class TryFinallyPattern:
    body_start_index: int
    body_end_index: int
    final_start_index: int
    final_end_index: int
    after_index: int


@dataclass(frozen=True, slots=True)
class SimpleHandler:
    exception_type_index: int
    exception_type_end_index: int
    body_start_index: int
    body_end_index: int
    name: str | None = None
    miss_index: int | None = None


@dataclass(frozen=True, slots=True)
class LegacyExceptionMatch:
    exception_type_index: int
    miss_jump_index: int
    miss_index: int


@dataclass(frozen=True, slots=True)
class ExceptionTableHandlerMatch:
    exception_type_index: int
    exception_type_end_index: int
    miss_jump_index: int
    miss_index: int


@dataclass(frozen=True, slots=True)
class CommonTrailingBody:
    start_index: int
    end_index: int
    length: int


@dataclass(frozen=True, slots=True)
class ExceptionTableExceptPattern:
    body_start_index: int
    body_end_index: int
    handler_start_index: int
    handler_body_start_index: int
    handler_body_end_index: int
    exception_type_index: int
    exception_type_end_index: int
    after_index: int
    name: str | None = None
    trailing_start_index: int | None = None
    trailing_end_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTableExceptStarPattern:
    body_start_index: int
    body_end_index: int
    handlers: tuple[SimpleHandler, ...]
    after_index: int
    trailing_start_index: int | None = None
    trailing_end_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTableFinallyPattern:
    body_start_index: int
    body_end_index: int
    final_start_index: int
    final_end_index: int
    after_index: int
    handler_index: int
    returns_protected_value: bool = False
    handler_final_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTableRegion:
    entry: ExceptionTableEntry
    body_start_index: int
    body_end_index: int
    handler_start_index: int


@dataclass(frozen=True, slots=True)
class FinallyBodyRange:
    final_start_index: int
    final_end_index: int
    handler_final_end_index: int
    handler_final_start_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExceptFinallyRange:
    final_start_index: int
    final_end_index: int


@dataclass(frozen=True, slots=True)
class ReturningExceptFinallyRange:
    final_start_index: int
    final_end_index: int
    handler_end_index: int


@dataclass(frozen=True, slots=True)
class ExceptionTableExceptFinallyPattern:
    body_start_index: int
    body_end_index: int
    handler_body_start_index: int
    handler_body_end_index: int
    exception_type_index: int
    exception_type_end_index: int
    final_start_index: int
    final_end_index: int
    after_index: int
    name: str | None = None
    returns_protected_values: bool = False


def try_translate_exception_table_except(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_exception_table_except_pattern(
        decompiler.code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if pattern is None:
        return None

    body = decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )
    if not body:
        protected_return = protected_cleanup_return_statement(
            decompiler,
            instructions,
            pattern,
        )
        if protected_return is not None:
            body = [protected_return]

    handler_body = decompiler.translate_child_statements(
        instructions,
        pattern.handler_body_start_index,
        pattern.handler_body_end_index,
    )
    handler_body = remove_named_exception_cleanup(handler_body, pattern.name)

    trailing_body = []
    if (
        pattern.trailing_start_index is not None
        and pattern.trailing_end_index is not None
    ):
        trailing_body = decompiler.translate_child_statements(
            instructions,
            pattern.trailing_start_index,
            pattern.trailing_end_index,
        )

    exception_type = decompiler.evaluate_expression_range(
        instructions,
        pattern.exception_type_index,
        pattern.exception_type_end_index,
    )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[
                ast.ExceptHandler(
                    type=exception_type,
                    name=pattern.name,
                    body=handler_body or [ast.Pass()],
                )
            ],
            orelse=[],
            finalbody=[],
        )
    )
    decompiler.statements.extend(trailing_body)
    return pattern.after_index


def protected_cleanup_return_statement(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: ExceptionTableExceptPattern,
) -> ast.Return | None:
    if not has_cleanup_return_suffix(
        instructions,
        pattern.body_end_index,
        pattern.handler_start_index,
    ):
        return None

    value = decompiler.evaluate_expression_range(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )
    if value is None:
        return None

    return ast.Return(value=value)


def has_cleanup_return_suffix(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> bool:
    if cursor >= end_index:
        return False

    saw_cleanup = False
    while cursor + 1 < end_index:
        if instructions[cursor].opname != "SWAP":
            break
        if instructions[cursor + 1].opname != "POP_TOP":
            break
        saw_cleanup = True
        cursor += 2

    return (
        saw_cleanup
        and cursor < end_index
        and instructions[cursor].opname == "RETURN_VALUE"
    )


def try_translate_exception_table_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_exception_table_finally_pattern(
        decompiler.code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if pattern is None:
        return None

    if pattern.handler_final_range is None:
        claim_finally_copies_before_return(
            decompiler,
            instructions,
            (pattern.body_start_index, pattern.body_end_index),
            (pattern.final_start_index, pattern.final_end_index),
            pattern.handler_index,
        )
    body = exception_table_finally_body(decompiler, instructions, pattern)
    final_body = translate_finally_block(
        decompiler,
        instructions,
        (pattern.final_start_index, pattern.final_end_index),
        pattern.handler_final_range,
    )
    final_body = wrap_suppressed_finally_cleanup(
        decompiler,
        instructions,
        offset_to_index,
        pattern,
        final_body,
    )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[],
            orelse=[],
            finalbody=final_body or [ast.Pass()],
        )
    )
    return pattern.after_index


def inline_finally_copy(
    instructions: list[Instruction],
    pattern: "OutOfLineTryPattern",
) -> tuple[int, int] | None:
    final_end_index = pattern.final_end_index
    if final_end_index is None or pattern.handler_final_range is not None:
        return None
    if pattern.final_start_index is not None:
        return None
    start_index = inline_finally_start(
        instructions, pattern.body_end_index, pattern.handler_index + 1
    )
    return start_index, final_end_index


def inline_finally_start(
    instructions: list[Instruction],
    body_end_index: int,
    copy_start_index: int,
) -> int:
    start_index = body_end_index
    if instructions[copy_start_index].opname == "NOP":
        return start_index
    while instructions[start_index].opname == "NOP":
        start_index += 1
    return start_index


def claim_finally_copies(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: "OutOfLineTryPattern",
) -> None:
    final_range = inline_finally_copy(instructions, pattern)
    body_ranges = [(pattern.body_start_index, pattern.handler_index)]
    if final_range is not None:
        body_ranges = [
            (pattern.body_start_index, pattern.body_end_index),
            (final_range[1], pattern.handler_index),
        ]
    elif pattern.final_start_index is not None and pattern.final_end_index is not None:
        final_range = (pattern.final_start_index, pattern.final_end_index)
    if final_range is None:
        claim_branching_copies_before_return(decompiler, instructions, pattern)
        return
    for body_range in body_ranges:
        claim_finally_copies_before_return(
            decompiler, instructions, body_range, final_range, pattern.handler_index
        )


def claim_branching_copies_before_return(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: "OutOfLineTryPattern",
) -> None:
    if pattern.handler_final_range is None:
        return
    start_index, end_index = pattern.handler_final_range
    length = end_index - start_index
    for index in range(pattern.body_start_index, pattern.body_end_index - length + 1):
        if all(
            (
                instructions[index + step].opname in {"RETURN_CONST", "RETURN_VALUE"}
                if is_reraise_zero(instructions[start_index + step])
                else same_instruction_behavior(
                    instructions[index + step], instructions[start_index + step]
                )
            )
            for step in range(length)
        ):
            claim_finally_copy(
                decompiler,
                instructions,
                pattern.body_start_index,
                (index, index + length - 1),
            )


def claim_finally_copies_before_return(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    body_range: tuple[int, int],
    final_range: tuple[int, int],
    handler_index: int,
) -> None:
    final_start, final_end = final_range
    length = final_end - final_start
    if length < 1:
        return
    protected = indexes_inside_handler(decompiler, instructions, handler_index)
    for index in range(body_range[0], body_range[1] - length):
        return_index = unprotected_return_end(
            instructions, index + length, body_range[1] + 1
        )
        returns = (
            return_index != index + length
            or returns_constant(decompiler, instructions, index + length)
            or (bool(protected) and index not in protected)
        )
        if returns and same_instruction_range_behavior(
            instructions, index, index + length, final_start, final_end
        ):
            claim_finally_copy(
                decompiler, instructions, body_range[0], (index, index + length)
            )


def indexes_inside_handler(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    handler_index: int,
) -> set[int]:
    spans = read_table_spans(
        decompiler.code, instructions, make_offset_index(instructions)
    )
    return {
        index
        for span in spans
        if span.target_index == handler_index
        for index in range(span.start_index, span.end_index)
    }


def returns_constant(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    index: int,
) -> bool:
    if instructions[index].opname == "RETURN_CONST":
        return True
    # only 3.12 and 3.13 have RETURN_CONST; the others load the constant first
    if decompiler.version is None or decompiler.version[:2] in {(3, 12), (3, 13)}:
        return False
    return (
        instructions[index].opname == "LOAD_CONST"
        and instructions[index + 1].opname == "RETURN_VALUE"
    )


def claim_finally_copy(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    lower_bound: int,
    copy_range: tuple[int, int],
) -> None:
    start_index, end_index = copy_range
    spans = read_table_spans(
        decompiler.code, instructions, make_offset_index(instructions)
    )
    regions = handler_regions(instructions, spans)
    for span in spans:
        handler_index = span.target_index
        if handler_index not in regions or handler_index < end_index:
            continue
        if all(
            start_index <= other.start_index < end_index
            for other in spans
            if other.target_index == handler_index
        ):
            decompiler.claimed_ranges[handler_index] = regions[handler_index]
    while start_index > lower_bound and instructions[start_index - 1].opname == "NOP":
        start_index -= 1
    for index in range(start_index, end_index):
        decompiler.claimed_ranges[index] = end_index
        if instructions[index].opname != "NOP":
            break


def translate_finally_block(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    inline_range: tuple[int, int],
    handler_range: tuple[int, int] | None,
) -> list[ast.stmt]:
    if handler_range is None:
        return decompiler.translate_child_statements(instructions, *inline_range)
    start_index, end_index = handler_range
    for index in range(start_index, end_index):
        if is_reraise_zero(instructions[index]):
            decompiler.claimed_ranges[index] = index + 1
            decompiler.handler_exit_indexes.add(index)
    statements = decompiler.translate_child_statements(
        instructions, start_index, end_index
    )
    return restore_handler_exits(fold_handler_exits(statements))


def wrap_suppressed_finally_cleanup(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    pattern: ExceptionTableFinallyPattern,
    final_body: list[ast.stmt],
) -> list[ast.stmt]:
    handler_start_index = find_suppressed_finally_cleanup_handler(
        decompiler.code,
        instructions,
        offset_to_index,
        pattern.final_start_index,
        pattern.final_end_index,
    )
    if handler_start_index is None:
        return final_body
    if not final_body:
        return final_body

    return [
        ast.Try(
            body=final_body,
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="Exception", ctx=ast.Load()),
                    name=None,
                    body=[ast.Pass()],
                )
            ],
            orelse=[],
            finalbody=[],
        )
    ]


def find_suppressed_finally_cleanup_handler(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    final_start_index: int,
    final_end_index: int,
) -> int | None:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if not table:
        return None

    for entry in parse_exception_table(table):
        start_index = offset_to_index.get(entry.start_offset)
        end_index = offset_to_index.get(entry.end_offset)
        handler_start_index = offset_to_index.get(entry.target_offset)
        if start_index is None or end_index is None or handler_start_index is None:
            continue
        if start_index < final_start_index or start_index >= final_end_index:
            continue
        if end_index != final_end_index:
            continue
        if is_suppressed_exception_return_handler(
            instructions,
            handler_start_index,
            len(instructions),
        ):
            return handler_start_index

    return None


def is_suppressed_exception_return_handler(
    instructions: list[Instruction],
    handler_start_index: int,
    end_index: int,
) -> bool:
    cursor = read_exception_opcode_index(
        instructions,
        handler_start_index,
        end_index,
        lambda instruction: instruction.opname == "PUSH_EXC_INFO",
    )
    if cursor is None:
        return False

    exception_type_index = read_exception_opcode_index(
        instructions,
        cursor + 1,
        end_index,
        is_exception_load,
    )
    if exception_type_index is None:
        return False

    match_index = read_exception_opcode_index(
        instructions,
        exception_type_index + 1,
        end_index,
        lambda instruction: instruction.opname == "CHECK_EXC_MATCH",
    )
    if match_index is None:
        return False

    jump_index = read_exception_opcode_index(
        instructions,
        match_index + 1,
        end_index,
        is_exception_false_jump,
    )
    if jump_index is None:
        return False

    return has_suppressed_exception_return_tail(
        instructions,
        jump_index + 1,
        end_index,
    )


ExceptionInstructionPredicate = Callable[[Instruction], bool]


def read_exception_opcode_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    predicate: ExceptionInstructionPredicate,
) -> int | None:
    cursor = skip_exception_match_prefix(instructions, start_index, end_index)
    if cursor >= end_index or not predicate(instructions[cursor]):
        return None
    return cursor


def is_exception_false_jump(instruction: Instruction) -> bool:
    return "POP_JUMP" in instruction.opname and "IF_FALSE" in instruction.opname


def has_suppressed_exception_return_tail(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    required = (
        lambda instruction: instruction.opname == "POP_TOP",
        lambda instruction: instruction.opname == "POP_EXCEPT",
        is_none_load,
        lambda instruction: instruction.opname == "RETURN_VALUE",
    )
    cursor = start_index
    for predicate in required:
        opcode_index = read_exception_opcode_index(
            instructions,
            cursor,
            end_index,
            predicate,
        )
        if opcode_index is None:
            return False
        cursor = opcode_index + 1
    return True


def is_exception_load(instruction: Instruction) -> bool:
    return instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"} and (
        instruction.argval == "Exception"
    )


def try_translate_exception_table_except_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_exception_table_except_finally_pattern(
        decompiler.code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if pattern is None:
        return None

    body: list[ast.stmt]
    handler_body: list[ast.stmt]
    if pattern.returns_protected_values:
        body = [
            ast.Return(
                value=decompiler.evaluate_expression_range(
                    instructions,
                    pattern.body_start_index,
                    pattern.body_end_index,
                )
            )
        ]
        handler_body = [
            ast.Return(
                value=decompiler.evaluate_expression_range(
                    instructions,
                    pattern.handler_body_start_index,
                    pattern.handler_body_end_index,
                )
            )
        ]
    else:
        body = decompiler.translate_child_statements(
            instructions,
            pattern.body_start_index,
            pattern.body_end_index,
        )
        handler_body = decompiler.translate_child_statements(
            instructions,
            pattern.handler_body_start_index,
            pattern.handler_body_end_index,
        )

    final_body = decompiler.translate_child_statements(
        instructions,
        pattern.final_start_index,
        pattern.final_end_index,
    )
    exception_type = decompiler.evaluate_expression_range(
        instructions,
        pattern.exception_type_index,
        pattern.exception_type_end_index,
    )
    inner_try = ast.Try(
        body=body or [ast.Pass()],
        handlers=[
            ast.ExceptHandler(
                type=exception_type,
                name=pattern.name,
                body=handler_body or [ast.Pass()],
            )
        ],
        orelse=[],
        finalbody=[],
    )
    decompiler.statements.append(
        ast.Try(
            body=[inner_try],
            handlers=[],
            orelse=[],
            finalbody=final_body or [ast.Pass()],
        )
    )
    return pattern.after_index


def exception_table_finally_body(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: ExceptionTableFinallyPattern,
) -> list[ast.stmt]:
    if pattern.returns_protected_value:
        child = decompiler.make_child()
        statements = child.translate_range(
            instructions,
            pattern.body_start_index,
            pattern.body_end_index,
        )
        decompiler.warnings.extend(child.warnings)
        value = (
            decompiler.expression_from_stack_value(child.stack[-1])
            if child.stack
            else None
        )
        return [*statements, ast.Return(value=value)]

    return decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )


def try_translate_exception_table_except_star(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_exception_table_except_star_pattern(
        decompiler.code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if pattern is None:
        return None

    body = decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )
    handlers = [
        ast.ExceptHandler(
            type=decompiler.evaluate_expression_range(
                instructions,
                handler.exception_type_index,
                handler.exception_type_end_index,
            ),
            name=handler.name,
            body=decompiler.translate_child_statements(
                instructions,
                handler.body_start_index,
                handler.body_end_index,
            )
            or [ast.Pass()],
        )
        for handler in pattern.handlers
    ]
    try_star = getattr(ast, "TryStar", None)
    if try_star is None:
        raise RuntimeError("exception groups require Python 3.11 or newer")
    decompiler.statements.append(
        try_star(
            body=body or [ast.Pass()],
            handlers=handlers,
            orelse=[],
            finalbody=[],
        )
    )
    if pattern.trailing_start_index is not None:
        decompiler.statements.extend(
            decompiler.translate_child_statements(
                instructions,
                pattern.trailing_start_index,
                pattern.trailing_end_index or pattern.trailing_start_index,
            )
        )
    return pattern.after_index


def try_translate_except(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_try_except_pattern(instructions, offset_to_index, cursor, end_index)
    if pattern is None:
        return None

    body = decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )
    handler_body = decompiler.translate_child_statements(
        instructions,
        pattern.handler_body_start_index,
        pattern.handler_body_end_index,
    )

    exception_type = None
    if pattern.exception_type_index >= 0:
        exception_type = decompiler.evaluate_expression_range(
            instructions,
            pattern.exception_type_index,
            pattern.exception_type_index + 1,
        )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[
                ast.ExceptHandler(
                    type=exception_type,
                    name=pattern.name,
                    body=handler_body or [ast.Pass()],
                )
            ],
            orelse=[],
            finalbody=[],
        )
    )
    return pattern.after_index


def find_exception_table_except_pattern(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> ExceptionTableExceptPattern | None:
    region = read_exception_table_region(
        code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
        use_protected_end=True,
    )
    if region is None:
        return None

    after_index = find_exception_table_after_index(
        code,
        instructions,
        offset_to_index,
        region.entry,
        region.body_end_index,
        end_index,
    )
    if after_index is None:
        return None

    handler = read_exception_table_handler(
        instructions,
        offset_to_index,
        region.handler_start_index,
        after_index,
    )
    if handler is None:
        return None

    success_jump_index = find_exception_table_success_jump_before_handler(
        instructions,
        region.body_end_index,
        region.handler_start_index,
        offset_to_index,
    )
    render_body_end_index = region.body_end_index
    resume_index = after_index
    if success_jump_index is not None:
        render_body_end_index = success_jump_index
    elif is_exception_table_fallthrough_body(
        instructions,
        region.body_end_index,
        region.handler_start_index,
    ):
        resume_index = region.body_end_index

    render_body_end_index = exception_table_body_end_index(
        instructions,
        render_body_end_index,
        region.handler_start_index,
    )
    trailing = find_exception_table_common_trailing_body(
        instructions,
        render_body_end_index,
        region.handler_start_index,
        handler.body_start_index,
        handler.body_end_index,
    )
    handler_body_end_index = handler.body_end_index
    trailing_start_index = None
    trailing_end_index = None
    if trailing is not None:
        handler_body_end_index -= trailing.length
        trailing_start_index = trailing.start_index
        trailing_end_index = trailing.end_index

    return ExceptionTableExceptPattern(
        body_start_index=region.body_start_index,
        body_end_index=render_body_end_index,
        handler_start_index=region.handler_start_index,
        handler_body_start_index=handler.body_start_index,
        handler_body_end_index=handler_body_end_index,
        exception_type_index=handler.exception_type_index,
        exception_type_end_index=handler.exception_type_end_index,
        after_index=resume_index,
        name=handler.name,
        trailing_start_index=trailing_start_index,
        trailing_end_index=trailing_end_index,
    )


def read_exception_table_region(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
    use_protected_end: bool,
) -> ExceptionTableRegion | None:
    entry = find_exception_table_entry(code, instructions[cursor].offset)
    if entry is None:
        return None

    body_start_index = offset_to_index.get(entry.start_offset)
    body_end_index = read_exception_table_body_end_index(
        code,
        offset_to_index,
        entry,
        end_index,
        use_protected_end,
    )
    handler_start_index = offset_to_index.get(entry.target_offset)
    if (
        body_start_index is None
        or body_end_index is None
        or handler_start_index is None
    ):
        return None
    if body_start_index != cursor or body_end_index <= body_start_index:
        return None
    if handler_start_index <= body_start_index or handler_start_index >= end_index:
        return None

    return ExceptionTableRegion(
        entry=entry,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        handler_start_index=handler_start_index,
    )


def read_exception_table_body_end_index(
    code: object,
    offset_to_index: dict[int, int],
    entry: ExceptionTableEntry,
    end_index: int,
    use_protected_end: bool,
) -> int | None:
    if use_protected_end:
        return exception_table_protected_end_index(
            code,
            offset_to_index,
            entry,
            end_index,
        )
    return offset_to_index.get(entry.end_offset)


def is_exception_table_fallthrough_body(
    instructions: list[Instruction],
    body_end_index: int,
    handler_start_index: int,
) -> bool:
    if body_end_index >= handler_start_index:
        return False
    instruction = instructions[body_end_index]
    if is_jump_op(instruction.opname) or is_terminal_op(instruction.opname):
        return False
    return instruction.opname != "POP_EXCEPT"


def find_exception_table_success_jump_before_handler(
    instructions: list[Instruction],
    body_end_index: int,
    handler_start_index: int,
    offset_to_index: dict[int, int],
) -> int | None:
    for index in range(body_end_index, handler_start_index):
        instruction = instructions[index]
        if not is_jump_op(instruction.opname) or not isinstance(
            instruction.argval,
            int,
        ):
            continue
        target_index = offset_to_index.get(instruction.argval)
        if target_index is not None and target_index > handler_start_index:
            return index
    return None


def find_exception_table_finally_pattern(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> ExceptionTableFinallyPattern | None:
    region = read_exception_table_region(
        code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
        use_protected_end=False,
    )
    if region is None:
        return None

    final_range = read_exception_table_finally_body_range(
        instructions,
        region,
        end_index,
    )
    if final_range is None:
        return None

    cleanup_after_index = find_finally_exception_cleanup_after(
        instructions,
        final_range.handler_final_end_index + 1,
        end_index,
    )
    returns_protected_value = is_return_after_finally(
        instructions,
        final_range.final_end_index,
        region.handler_start_index,
    )
    after_index = (
        cleanup_after_index if returns_protected_value else final_range.final_end_index
    )

    handler_final_range = None
    if final_range.handler_final_start_index is not None:
        handler_final_range = (
            final_range.handler_final_start_index,
            final_range.handler_final_end_index + 1,
        )
    return ExceptionTableFinallyPattern(
        body_start_index=region.body_start_index,
        body_end_index=region.body_end_index,
        final_start_index=final_range.final_start_index,
        final_end_index=final_range.final_end_index,
        after_index=after_index,
        handler_index=region.handler_start_index,
        returns_protected_value=returns_protected_value,
        handler_final_range=handler_final_range,
    )


def read_exception_table_finally_body_range(
    instructions: list[Instruction],
    region: ExceptionTableRegion,
    end_index: int,
) -> FinallyBodyRange | None:
    if not is_valid_finally_region_start(instructions, region, end_index):
        return None

    handler_final_start_index = region.handler_start_index + 1
    several_ends = read_branching_finally_range(
        instructions, region.body_end_index, region.handler_start_index
    )
    if several_ends is not None:
        return several_ends
    handler_final_end_index = find_reraise_zero(
        instructions,
        handler_final_start_index,
        end_index,
    )
    if handler_final_end_index is None:
        return None
    if (
        find_check_exception_match(
            instructions,
            handler_final_start_index,
            handler_final_end_index,
        )
        is not None
    ):
        return None

    final_start_index = region.body_end_index
    final_end_index = final_start_index + (
        handler_final_end_index - handler_final_start_index
    )
    if final_end_index > region.handler_start_index:
        return None
    if has_terminal_instruction(instructions, final_start_index, final_end_index):
        return None

    return FinallyBodyRange(
        final_start_index=final_start_index,
        final_end_index=final_end_index,
        handler_final_end_index=handler_final_end_index,
    )


def has_jump_out_of_range(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    start_offset = instructions[start_index].offset
    end_offset = instructions[end_index].offset
    return any(
        is_jump_op(instruction.opname)
        and not start_offset <= int(instruction.argval) <= end_offset
        for instruction in instructions[start_index:end_index]
    )


def read_branching_finally_range(
    instructions: list[Instruction],
    body_end_index: int,
    handler_index: int,
) -> FinallyBodyRange | None:
    start_index = handler_index + 1
    cleanup_index = next(
        (
            index
            for index in range(start_index, len(instructions) - 2)
            if is_handler_cleanup(instructions, index)
        ),
        None,
    )
    if cleanup_index is None:
        return None
    copy = instructions[start_index:cleanup_index]
    ends = [instruction for instruction in copy if is_reraise_zero(instruction)]
    nested = any(i.opname in {"PUSH_EXC_INFO", "CHECK_EXC_MATCH"} for i in copy)
    if len(ends) < 2 or nested or not is_reraise_zero(copy[-1]):
        return None
    final_end_index = inline_finally_end(
        instructions,
        inline_finally_start(instructions, body_end_index, start_index),
        start_index,
        cleanup_index,
    )
    if final_end_index is None or final_end_index > handler_index:
        return None
    return FinallyBodyRange(
        final_start_index=body_end_index,
        final_end_index=final_end_index,
        handler_final_end_index=cleanup_index - 1,
        handler_final_start_index=start_index,
    )


def is_reraise_zero(instruction: Instruction) -> bool:
    return instruction.opname == "RERAISE" and int(instruction.arg or 0) == 0


def is_handler_cleanup(instructions: list[Instruction], index: int) -> bool:
    return (
        instructions[index].opname == "COPY"
        and instructions[index + 1].opname == "POP_EXCEPT"
        and instructions[index + 2].opname == "RERAISE"
    )


def inline_finally_end(
    instructions: list[Instruction],
    inline_index: int,
    handler_index: int,
    handler_end_index: int,
) -> int | None:
    for index in range(handler_index, handler_end_index - 1):
        instruction = instructions[index]
        if not is_reraise_zero(instruction):
            if instructions[inline_index].opname != instruction.opname:
                return None
            inline_index += 1
            continue
        resumed = instructions[index + 1]
        while not (
            instructions[inline_index].is_jump_target
            and (
                is_reraise_zero(resumed)
                or instructions[inline_index].opname == resumed.opname
            )
        ):
            inline_index += 1
            if inline_index >= handler_index:
                return None
    return inline_index


def is_valid_finally_region_start(
    instructions: list[Instruction],
    region: ExceptionTableRegion,
    end_index: int,
) -> bool:
    if has_jump_out_of_range(
        instructions,
        region.body_start_index,
        region.body_end_index,
    ):
        return False
    if region.handler_start_index <= region.body_end_index:
        return False
    if region.handler_start_index >= end_index:
        return False
    return instructions[region.handler_start_index].opname == "PUSH_EXC_INFO"


def find_exception_table_except_finally_pattern(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> ExceptionTableExceptFinallyPattern | None:
    except_pattern = find_exception_table_except_pattern(
        code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if except_pattern is None:
        return None
    if except_pattern.trailing_start_index is not None:
        return find_exception_table_returning_except_finally_pattern(
            code,
            instructions,
            offset_to_index,
            except_pattern,
        )
    if except_pattern.body_end_index >= except_pattern.handler_start_index:
        return None

    final_range = read_exception_table_except_finally_range(
        code,
        instructions,
        offset_to_index,
        except_pattern,
    )
    if final_range is None:
        return None

    return ExceptionTableExceptFinallyPattern(
        body_start_index=except_pattern.body_start_index,
        body_end_index=except_pattern.body_end_index,
        handler_body_start_index=except_pattern.handler_body_start_index,
        handler_body_end_index=except_pattern.handler_body_end_index,
        exception_type_index=except_pattern.exception_type_index,
        exception_type_end_index=except_pattern.exception_type_end_index,
        final_start_index=final_range.final_start_index,
        final_end_index=final_range.final_end_index,
        after_index=except_pattern.after_index,
        name=except_pattern.name,
    )


def read_exception_table_except_finally_range(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    except_pattern: ExceptionTableExceptPattern,
) -> ExceptFinallyRange | None:
    final_start_index = skip_finally_body_prefix(
        instructions,
        except_pattern.body_end_index,
        except_pattern.handler_start_index,
    )
    final_end_index = find_finally_body_end_before_cleanup(
        instructions,
        final_start_index,
        except_pattern.handler_start_index,
    )
    if final_end_index is None or final_end_index <= final_start_index:
        return None

    cleanup_range = find_duplicated_exception_finally_range(
        code,
        instructions,
        offset_to_index,
        except_pattern.handler_start_index,
        except_pattern.after_index,
    )
    if cleanup_range is None:
        return None
    cleanup_start_index, cleanup_end_index = cleanup_range
    if not same_instruction_range_behavior(
        instructions,
        final_start_index,
        final_end_index,
        cleanup_start_index,
        cleanup_end_index,
    ):
        return None

    return ExceptFinallyRange(final_start_index, final_end_index)


def find_exception_table_returning_except_finally_pattern(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    except_pattern: ExceptionTableExceptPattern,
) -> ExceptionTableExceptFinallyPattern | None:
    final_range = read_returning_except_finally_range(
        code,
        instructions,
        offset_to_index,
        except_pattern,
    )
    if final_range is None:
        return None

    return ExceptionTableExceptFinallyPattern(
        body_start_index=except_pattern.body_start_index,
        body_end_index=except_pattern.body_end_index,
        handler_body_start_index=except_pattern.handler_body_start_index,
        handler_body_end_index=final_range.handler_end_index,
        exception_type_index=except_pattern.exception_type_index,
        exception_type_end_index=except_pattern.exception_type_end_index,
        final_start_index=final_range.final_start_index,
        final_end_index=final_range.final_end_index,
        after_index=except_pattern.after_index,
        name=except_pattern.name,
        returns_protected_values=True,
    )


def read_returning_except_finally_range(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    except_pattern: ExceptionTableExceptPattern,
) -> ReturningExceptFinallyRange | None:
    trailing_range = read_returning_finally_trailing_range(
        instructions,
        except_pattern,
    )
    if trailing_range is None:
        return None
    final_start_index, final_end_index = trailing_range

    handler_end_index = trim_saved_return_handler_cleanup(
        instructions,
        except_pattern.handler_body_start_index,
        except_pattern.handler_body_end_index,
    )
    if handler_end_index <= except_pattern.handler_body_start_index:
        return None
    if not has_duplicated_exception_finally_range(
        code,
        instructions,
        offset_to_index,
        except_pattern,
        final_start_index,
        final_end_index,
    ):
        return None

    return ReturningExceptFinallyRange(
        final_start_index=final_start_index,
        final_end_index=final_end_index,
        handler_end_index=handler_end_index,
    )


def read_returning_finally_trailing_range(
    instructions: list[Instruction],
    except_pattern: ExceptionTableExceptPattern,
) -> tuple[int, int] | None:
    if (
        except_pattern.trailing_start_index is None
        or except_pattern.trailing_end_index is None
    ):
        return None
    if except_pattern.trailing_end_index <= except_pattern.trailing_start_index:
        return None
    if instructions[except_pattern.trailing_end_index - 1].opname != "RETURN_VALUE":
        return None

    final_start_index = except_pattern.trailing_start_index
    final_end_index = except_pattern.trailing_end_index - 1
    if final_end_index <= final_start_index:
        return None

    return final_start_index, final_end_index


def has_duplicated_exception_finally_range(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    except_pattern: ExceptionTableExceptPattern,
    final_start_index: int,
    final_end_index: int,
) -> bool:
    cleanup_range = find_duplicated_exception_finally_range(
        code,
        instructions,
        offset_to_index,
        except_pattern.handler_start_index,
        except_pattern.after_index,
    )
    if cleanup_range is None:
        return False
    cleanup_start_index, cleanup_end_index = cleanup_range

    return same_instruction_range_behavior(
        instructions,
        final_start_index,
        final_end_index,
        cleanup_start_index,
        cleanup_end_index,
    )


def trim_saved_return_handler_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = end_index
    while cursor > start_index and instructions[cursor - 1].opname in {
        "POP_EXCEPT",
        "SWAP",
    }:
        cursor -= 1
    return cursor


def skip_finally_body_prefix(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index
    while cursor < end_index and instructions[cursor].opname in {
        "NOP",
        "PUSH_EXC_INFO",
    }:
        cursor += 1
    return cursor


def find_finally_body_end_before_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if is_terminal_op(instructions[index].opname):
            return index
    return None


def find_duplicated_exception_finally_range(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if not table:
        return None

    for entry in parse_exception_table(table):
        if entry.depth != 0:
            continue
        start_index = offset_to_index.get(entry.start_offset)
        target_index = offset_to_index.get(entry.target_offset)
        if start_index is None or target_index is None:
            continue
        if start_index <= handler_start_index or target_index <= start_index:
            continue
        if target_index >= end_index:
            continue
        body_start_index = skip_finally_body_prefix(
            instructions,
            target_index,
            end_index,
        )
        body_end_index = find_finally_body_end_before_cleanup(
            instructions,
            body_start_index,
            end_index,
        )
        if body_end_index is not None:
            return body_start_index, body_end_index

    return None


def same_instruction_range_behavior(
    instructions: list[Instruction],
    left_start_index: int,
    left_end_index: int,
    right_start_index: int,
    right_end_index: int,
) -> bool:
    left_length = left_end_index - left_start_index
    if left_length != right_end_index - right_start_index:
        return False

    for offset in range(left_length):
        left = instructions[left_start_index + offset]
        right = instructions[right_start_index + offset]
        if not same_instruction_behavior(left, right):
            return False
    return True


def has_terminal_instruction(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    for index in range(start_index, end_index):
        if is_terminal_op(instructions[index].opname):
            return True
    return False


def find_reraise_zero(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname == "RERAISE" and int(instruction.arg or 0) == 0:
            return index
    return None


def find_finally_exception_cleanup_after(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname == "RERAISE" and int(instruction.arg or 0) != 0:
            return index + 1
    return end_index


def is_return_after_finally(
    instructions: list[Instruction],
    final_end_index: int,
    handler_start_index: int,
) -> bool:
    if final_end_index >= handler_start_index:
        return False
    return instructions[final_end_index].opname == "RETURN_VALUE"


def find_exception_table_except_star_pattern(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> ExceptionTableExceptStarPattern | None:
    region = read_exception_table_region(
        code,
        instructions,
        offset_to_index,
        cursor,
        end_index,
        use_protected_end=False,
    )
    if region is None:
        return None

    handler = read_exception_group_handler(
        instructions,
        offset_to_index,
        region.handler_start_index,
        end_index,
    )
    if handler is None:
        return None

    handlers = read_exception_group_handlers(
        instructions,
        offset_to_index,
        handler,
        end_index,
    )
    if not handlers:
        return None

    after_region = find_exception_group_after_region(
        instructions,
        handlers[-1].body_end_index,
        end_index,
    )
    if after_region is None:
        return None
    after_index, trailing_start_index, trailing_end_index = after_region

    return ExceptionTableExceptStarPattern(
        body_start_index=region.body_start_index,
        body_end_index=exception_table_body_end_index(
            instructions,
            region.body_end_index,
            region.handler_start_index,
        ),
        handlers=handlers,
        after_index=after_index,
        trailing_start_index=trailing_start_index,
        trailing_end_index=trailing_end_index,
    )


def exception_table_body_end_index(
    instructions: list[Instruction],
    body_end_index: int,
    handler_start_index: int,
) -> int:
    if body_end_index >= handler_start_index:
        return body_end_index

    instruction = instructions[body_end_index]
    if is_implicit_none_return(instructions, body_end_index):
        return body_end_index
    if is_terminal_op(instruction.opname):
        return body_end_index + 1
    if instruction.opname == "POP_EXCEPT":
        return exception_table_pop_except_body_end(
            instructions,
            body_end_index,
            handler_start_index,
        )
    if is_none_load(instruction) and body_end_index + 1 < handler_start_index:
        next_instruction = instructions[body_end_index + 1]
        if next_instruction.opname == "RETURN_VALUE":
            return body_end_index + 2
    return body_end_index


def exception_table_pop_except_body_end(
    instructions: list[Instruction],
    body_end_index: int,
    handler_start_index: int,
) -> int:
    next_index = body_end_index + 1
    if next_index >= handler_start_index:
        return body_end_index

    next_instruction = instructions[next_index]
    if is_terminal_op(next_instruction.opname):
        return next_index + 1
    if is_none_load(next_instruction) and next_index + 1 < handler_start_index:
        return_instruction = instructions[next_index + 1]
        if return_instruction.opname == "RETURN_VALUE":
            return next_index + 2
    return body_end_index


def find_exception_table_common_trailing_body(
    instructions: list[Instruction],
    success_start_index: int,
    handler_start_index: int,
    handler_body_start_index: int,
    handler_body_end_index: int,
) -> CommonTrailingBody | None:
    if success_start_index >= handler_start_index:
        return None
    if handler_body_start_index >= handler_body_end_index:
        return None
    if is_implicit_none_return(instructions, success_start_index):
        return None

    success_length = handler_start_index - success_start_index
    handler_length = handler_body_end_index - handler_body_start_index
    if success_length > handler_length:
        return None

    handler_tail_start = handler_body_end_index - success_length
    for offset in range(success_length):
        success_instruction = instructions[success_start_index + offset]
        handler_instruction = instructions[handler_tail_start + offset]
        if not same_instruction_behavior(success_instruction, handler_instruction):
            return None

    return CommonTrailingBody(
        start_index=success_start_index,
        end_index=handler_start_index,
        length=success_length,
    )


def same_instruction_behavior(left: Instruction, right: Instruction) -> bool:
    if common_trailing_opcode_name(left.opname) != common_trailing_opcode_name(
        right.opname
    ):
        return False
    if left.argrepr == right.argrepr and left.argval == right.argval:
        return True
    return (
        left.argrepr.startswith("to ")
        and right.argrepr.startswith("to ")
        and left.argval - left.offset == right.argval - right.offset
    )


def common_trailing_opcode_name(opname: str) -> str:
    if opname == "LOAD_FAST_CHECK":
        return "LOAD_FAST"
    return opname


def is_implicit_none_return(
    instructions: list[Instruction],
    start_index: int,
) -> bool:
    instruction = instructions[start_index]
    if instruction.starts_line is not None:
        return False
    if instruction.opname == "RETURN_CONST" and instruction.argval is None:
        return True
    if not is_none_load(instruction):
        return False

    next_index = start_index + 1
    if next_index >= len(instructions):
        return False
    next_instruction = instructions[next_index]
    return (
        next_instruction.opname == "RETURN_VALUE"
        and next_instruction.starts_line is None
    )


def find_exception_table_entry(
    code: object,
    start_offset: int,
) -> ExceptionTableEntry | None:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if not table:
        return None

    match: ExceptionTableEntry | None = None
    for entry in parse_exception_table(table):
        if entry.start_offset != start_offset:
            continue
        if match is None or entry.depth < match.depth:
            match = entry
    return match


def exception_table_protected_end_index(
    code: object,
    offset_to_index: dict[int, int],
    entry: ExceptionTableEntry,
    end_index: int,
) -> int | None:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if not table:
        return offset_to_index.get(entry.end_offset)

    end_offset = entry.end_offset
    for candidate in parse_exception_table(table):
        if candidate.depth != entry.depth:
            continue
        if candidate.target_offset != entry.target_offset:
            continue
        if candidate.start_offset < entry.start_offset:
            continue
        if candidate.start_offset >= entry.target_offset:
            continue
        end_offset = max(end_offset, candidate.end_offset)

    end_index_value = offset_to_index.get(end_offset)
    if end_index_value is None or end_index_value > end_index:
        return None
    return end_index_value


def find_exception_table_after_index(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    entry: ExceptionTableEntry,
    body_end_index: int,
    end_index: int,
) -> int | None:
    if body_end_index >= end_index:
        return None

    jump = instructions[body_end_index]
    if is_jump_op(jump.opname):
        target = offset_to_index.get(int(jump.argval))
        if target is None or target <= body_end_index or target > end_index:
            return None
        return target

    next_index = body_end_index + 1
    handler_index = offset_to_index.get(entry.target_offset)
    if (
        is_terminal_op(jump.opname)
        and handler_index is not None
        and next_index < handler_index
    ):
        next_instruction = instructions[next_index]
        if is_jump_op(next_instruction.opname) and isinstance(
            next_instruction.argval,
            int,
        ):
            target = offset_to_index.get(next_instruction.argval)
            if target is not None and target > next_index and target <= end_index:
                return target

    if is_terminal_op(jump.opname):
        return find_exception_table_terminal_after_index(
            code,
            instructions,
            offset_to_index,
            entry,
            body_end_index,
            end_index,
        )
    return find_exception_table_terminal_after_index(
        code,
        instructions,
        offset_to_index,
        entry,
        body_end_index,
        end_index,
    )


def find_exception_table_terminal_after_index(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    entry: ExceptionTableEntry,
    body_end_index: int,
    end_index: int,
) -> int | None:
    cleanup_index = offset_to_index.get(entry.target_offset)
    if cleanup_index is not None and body_end_index < cleanup_index < end_index:
        matched_after = exception_handler_matched_after_index(
            instructions,
            cleanup_index,
            end_index,
        )
        if matched_after is not None:
            return matched_after

    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if not table:
        return None

    cleanup_indices: list[int] = []
    for item in parse_exception_table(table):
        if item.depth < 1:
            continue
        target_index = offset_to_index.get(item.target_offset)
        if (
            target_index is None
            or target_index <= body_end_index
            or target_index >= end_index
        ):
            continue
        cleanup_indices.append(target_index)

    if not cleanup_indices:
        return None

    return skip_exception_cleanup_tail(
        instructions,
        max(cleanup_indices),
        end_index,
    )


def exception_handler_matched_after_index(
    instructions: list[Instruction],
    cleanup_index: int,
    end_index: int,
) -> int | None:
    miss_offset = exception_handler_miss_offset(instructions, cleanup_index, end_index)
    for index in range(cleanup_index, end_index):
        instruction = instructions[index]
        if instruction.opname == "JUMP_FORWARD" and isinstance(instruction.argval, int):
            if miss_offset is not None and instruction.argval <= miss_offset:
                continue
            return next_index_at_or_after_offset(
                instructions,
                instruction.argval,
                end_index,
            )
        if is_terminal_op(instruction.opname):
            return None
    return None


def exception_handler_miss_offset(
    instructions: list[Instruction],
    cleanup_index: int,
    end_index: int,
) -> int | None:
    match_index = find_check_exception_match(instructions, cleanup_index, end_index)
    if match_index is None:
        return None

    jump_index = skip_exception_match_prefix(instructions, match_index + 1, end_index)
    if jump_index >= end_index:
        return None
    jump = instructions[jump_index]
    if "POP_JUMP" not in jump.opname or not isinstance(jump.argval, int):
        return None
    return jump.argval


def skip_exception_cleanup_tail(
    instructions: list[Instruction],
    cleanup_index: int,
    end_index: int,
) -> int | None:
    for index in range(cleanup_index, end_index):
        if is_terminal_op(instructions[index].opname):
            return index + 1
    return None


def next_index_at_or_after_offset(
    instructions: list[Instruction],
    offset: int,
    end_index: int,
) -> int | None:
    for index in range(end_index):
        if instructions[index].offset >= offset:
            return index
    return None


def read_exception_table_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
) -> SimpleHandler | None:
    match = read_exception_table_handler_match(
        instructions,
        offset_to_index,
        handler_start_index,
        after_index,
    )
    if match is None:
        return None

    body_start_index = skip_exception_stack_pops(
        instructions,
        match.miss_jump_index + 1,
        match.miss_index,
    )
    name = exception_handler_name(instructions, body_start_index, match.miss_index)
    if name is not None:
        body_start_index += 1

    body_end_index = find_exception_table_handler_body_end(
        instructions,
        body_start_index,
        match.miss_index,
        name,
    )
    if body_end_index is None:
        return None

    return SimpleHandler(
        exception_type_index=match.exception_type_index,
        exception_type_end_index=match.exception_type_end_index,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        name=name,
    )


def read_exception_table_handler_match(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
) -> ExceptionTableHandlerMatch | None:
    if instructions[handler_start_index].opname != "PUSH_EXC_INFO":
        return None

    exception_type_index = skip_exception_match_prefix(
        instructions,
        handler_start_index + 1,
        after_index,
    )
    match_index = find_check_exception_match(
        instructions, exception_type_index, after_index
    )
    if match_index is None:
        return None

    miss_jump_index = skip_exception_match_prefix(
        instructions, match_index + 1, after_index
    )
    miss_index = read_exception_handler_miss_index(
        instructions,
        offset_to_index,
        handler_start_index,
        miss_jump_index,
        after_index,
    )
    if miss_index is None:
        return None

    return ExceptionTableHandlerMatch(
        exception_type_index=exception_type_index,
        exception_type_end_index=match_index,
        miss_jump_index=miss_jump_index,
        miss_index=miss_index,
    )


def read_exception_handler_miss_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    miss_jump_index: int,
    after_index: int,
) -> int | None:
    if miss_jump_index >= after_index:
        return None

    miss_jump = instructions[miss_jump_index]
    if "IF_FALSE" not in miss_jump.opname and "IF_TRUE" not in miss_jump.opname:
        return None

    miss_index = offset_to_index.get(int(miss_jump.argval))
    if (
        miss_index is None
        or miss_index <= handler_start_index
        or miss_index > after_index
    ):
        return None
    return miss_index


EXCEPTION_NAME_STORES = {"STORE_DEREF", "STORE_FAST", "STORE_GLOBAL", "STORE_NAME"}
EXCEPTION_NAME_DELETES = {
    "DELETE_DEREF",
    "DELETE_FAST",
    "DELETE_GLOBAL",
    "DELETE_NAME",
}


def exception_handler_name(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
) -> str | None:
    if body_start_index >= miss_index:
        return None
    instruction = instructions[body_start_index]
    if instruction.opname not in EXCEPTION_NAME_STORES:
        return None
    if not isinstance(instruction.argval, str):
        return None
    return instruction.argval


def find_check_exception_match(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname == "CHECK_EXC_MATCH":
            return index
    return None


def find_exception_table_handler_body_end(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    name: str | None,
) -> int | None:
    if miss_index <= body_start_index:
        return None

    success_cleanup_index = find_exception_handler_success_cleanup(
        instructions,
        body_start_index,
        miss_index,
    )
    if success_cleanup_index is not None:
        return success_cleanup_index

    if name is None:
        return miss_index

    named_cleanup_end = find_named_exception_cleanup_end(
        instructions,
        body_start_index,
        miss_index,
        name,
    )
    if named_cleanup_end is not None:
        return named_cleanup_end

    for index in range(body_start_index, miss_index):
        if is_terminal_op(instructions[index].opname):
            return index + 1
    cursor = miss_index - 1
    while cursor >= body_start_index and instructions[cursor].opname == "POP_EXCEPT":
        cursor -= 1
    return cursor + 1


def find_named_exception_cleanup_end(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    name: str,
) -> int | None:
    cleanup_index = find_named_exception_cleanup(
        instructions,
        body_start_index,
        miss_index,
        name,
    )
    if cleanup_index is None:
        return None

    returning_end_index = named_exception_cleanup_return_end(
        instructions,
        cleanup_index,
        miss_index,
    )
    if returning_end_index is not None:
        return returning_end_index
    return cleanup_index


def find_exception_handler_success_cleanup(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
) -> int | None:
    cleanup_index = None
    for index in range(body_start_index, miss_index):
        if instructions[index].opname != "POP_EXCEPT":
            continue
        jump_index = skip_extended_args(instructions, index + 1, miss_index)
        if jump_index >= miss_index:
            continue
        if is_jump_op(instructions[jump_index].opname):
            cleanup_index = index
    return cleanup_index


def skip_extended_args(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    while cursor < end_index and instructions[cursor].opname == "EXTENDED_ARG":
        cursor += 1
    return cursor


REST_OF_GROUP_OPS = frozenset({"CALL_INTRINSIC_2", "PREP_RERAISE_STAR"})


def read_exception_group_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    end_index: int,
) -> SimpleHandler | None:
    if instructions[handler_start_index].opname != "PUSH_EXC_INFO":
        return None

    exception_type_index = read_exception_group_type_start(
        instructions,
        handler_start_index + 1,
        end_index,
    )
    if exception_type_index is None:
        return None

    return read_exception_group_handler_from_type(
        instructions,
        offset_to_index,
        exception_type_index,
        end_index,
    )


def read_exception_group_handlers(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    first_handler: SimpleHandler,
    end_index: int,
) -> tuple[SimpleHandler, ...]:
    handlers = [first_handler]
    cursor = exception_group_next_handler_cursor(first_handler)

    for _index in range(16):
        next_type_index = find_next_exception_group_type_start(
            instructions,
            cursor,
            end_index,
        )
        if next_type_index is None:
            break

        handler = read_exception_group_handler_from_type(
            instructions,
            offset_to_index,
            next_type_index,
            end_index,
        )
        if handler is None:
            break

        handlers.append(handler)
        cursor = exception_group_next_handler_cursor(handler)

    return tuple(handlers)


def exception_group_next_handler_cursor(handler: SimpleHandler) -> int:
    if handler.miss_index is not None:
        return handler.miss_index
    return handler.body_end_index


def find_next_exception_group_type_start(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = start_index
    for _index in range(64):
        cursor = skip_exception_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None

        instruction = instructions[cursor]
        if instruction.opname in REST_OF_GROUP_OPS:
            return None
        if instruction.opname == "POP_TOP":
            return skip_exception_match_prefix(instructions, cursor + 1, end_index)
        cursor += 1
    return None


def read_exception_group_handler_from_type(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    exception_type_index: int,
    end_index: int,
) -> SimpleHandler | None:
    match_index = find_check_exception_group_match(
        instructions,
        exception_type_index,
        end_index,
    )
    if match_index is None:
        return None

    miss_jump_index = skip_exception_match_prefix(
        instructions,
        match_index + 1,
        end_index,
    )
    if miss_jump_index < end_index and instructions[miss_jump_index].opname == "COPY":
        miss_jump_index = skip_exception_match_prefix(
            instructions,
            miss_jump_index + 1,
            end_index,
        )
    if miss_jump_index >= end_index:
        return None

    miss_jump = instructions[miss_jump_index]
    if "POP_JUMP" not in miss_jump.opname or "IF_NONE" not in miss_jump.opname:
        return None

    miss_index = offset_to_index.get(int(miss_jump.argval))
    if miss_index is None or miss_index <= miss_jump_index:
        return None

    body_start_index = skip_exception_stack_pops(
        instructions,
        miss_jump_index + 1,
        miss_index,
    )
    name = exception_handler_name(instructions, body_start_index, miss_index)
    if name is not None:
        body_start_index = skip_exception_stack_pops(
            instructions,
            body_start_index + 1,
            miss_index,
        )

    body_end_index = find_exception_group_handler_body_end(
        instructions,
        body_start_index,
        miss_index,
        name,
    )
    if body_end_index is None:
        return None

    return SimpleHandler(
        exception_type_index=exception_type_index,
        exception_type_end_index=match_index,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        name=name,
        miss_index=miss_index,
    )


def read_exception_group_type_start(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = skip_exception_match_prefix(instructions, start_index, end_index)
    for opname in ("COPY", "BUILD_LIST", "COPY", "SWAP"):
        if cursor < end_index and instructions[cursor].opname == opname:
            cursor = skip_exception_match_prefix(instructions, cursor + 1, end_index)
    if cursor >= end_index:
        return None
    return cursor


def find_check_exception_group_match(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname == "CHECK_EG_MATCH":
            return index
    return None


def find_exception_group_handler_body_end(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    name: str | None,
) -> int | None:
    if miss_index <= body_start_index:
        return None

    if name is not None:
        cleanup_index = find_named_exception_cleanup(
            instructions,
            body_start_index,
            miss_index,
            name,
        )
        if cleanup_index is not None:
            return cleanup_index

    for index in range(body_start_index, miss_index):
        if instructions[index].opname in {"JUMP", "JUMP_FORWARD"}:
            return index
    return miss_index


def find_named_exception_cleanup(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    name: str,
) -> int | None:
    for index in range(body_start_index, miss_index - 2):
        if not is_none_load(instructions[index]):
            continue
        store = instructions[index + 1]
        delete = instructions[index + 2]
        if store.opname not in EXCEPTION_NAME_STORES:
            continue
        if delete.opname not in EXCEPTION_NAME_DELETES:
            continue
        if store.argval == name and delete.argval == name:
            return index
    return None


def named_exception_cleanup_return_end(
    instructions: list[Instruction],
    cleanup_index: int,
    miss_index: int,
) -> int | None:
    for index in range(cleanup_index + 3, miss_index):
        if instructions[index].opname in {"CACHE", "EXTENDED_ARG", "NOP"}:
            continue
        if instructions[index].opname in {"RETURN_CONST", "RETURN_VALUE"}:
            return index + 1
        if instructions[index].opname in {"RERAISE", "RAISE_VARARGS"}:
            return None
    return None


def remove_named_exception_cleanup(
    body: list[ast.stmt],
    name: str | None,
) -> list[ast.stmt]:
    if name is None:
        return body

    cleaned: list[ast.stmt] = []
    cursor = 0
    while cursor < len(body):
        if is_named_exception_cleanup_statement_pair(body, cursor, name):
            cursor += 2
            continue
        statement = body[cursor]
        if isinstance(statement, ast.If):
            statement.body = remove_named_exception_cleanup(statement.body, name)
            statement.orelse = remove_named_exception_cleanup(statement.orelse, name)
        if isinstance(statement, ast.Try):
            statement.body = remove_named_exception_cleanup(statement.body, name)
        cleaned.append(statement)
        cursor += 1
    return cleaned


def is_named_exception_cleanup_statement_pair(
    body: list[ast.stmt],
    index: int,
    name: str,
) -> bool:
    if index + 1 >= len(body):
        return False
    return is_named_exception_none_assignment(
        body[index],
        name,
    ) and is_named_exception_delete(body[index + 1], name)


def is_named_exception_none_assignment(statement: ast.stmt, name: str) -> bool:
    if not isinstance(statement, ast.Assign):
        return False
    if len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name) or target.id != name:
        return False
    return isinstance(statement.value, ast.Constant) and statement.value.value is None


def is_named_exception_delete(statement: ast.stmt, name: str) -> bool:
    if not isinstance(statement, ast.Delete):
        return False
    if len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return isinstance(target, ast.Name) and target.id == name


def is_none_load(instruction: Instruction) -> bool:
    return instruction.opname in {"LOAD_CONST", "RETURN_CONST"} and (
        instruction.argval is None
    )


def find_exception_group_after_region(
    instructions: list[Instruction],
    body_end_index: int,
    end_index: int,
) -> tuple[int, int | None, int | None] | None:
    cursor = find_exception_group_reraise_star_end(
        instructions,
        body_end_index,
        end_index,
    )
    if cursor is None:
        return None
    cursor = skip_exception_match_prefix(instructions, cursor, end_index)
    if cursor < end_index and instructions[cursor].opname == "COPY":
        cursor = skip_exception_match_prefix(instructions, cursor + 1, end_index)

    jump_index = read_exception_opcode_index(
        instructions,
        cursor,
        end_index,
        is_exception_group_not_none_jump,
    )
    if jump_index is None:
        return None

    cursor = skip_exception_match_prefix(instructions, jump_index + 1, end_index)
    if cursor < end_index and instructions[cursor].opname == "POP_TOP":
        cursor = skip_exception_match_prefix(instructions, cursor + 1, end_index)
    return read_exception_group_after_tail(instructions, cursor, end_index)


def find_exception_group_reraise_star_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = start_index
    for _index in range(32):
        cursor = skip_exception_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None
        instruction = instructions[cursor]
        if instruction.opname == "PREP_RERAISE_STAR":
            return cursor + 1
        if instruction.opname == "CALL_INTRINSIC_2":
            if instruction.argrepr != "INTRINSIC_PREP_RERAISE_STAR":
                return None
            return cursor + 1
        cursor += 1
    return None


def is_exception_group_not_none_jump(instruction: Instruction) -> bool:
    return "POP_JUMP" in instruction.opname and "IF_NOT_NONE" in instruction.opname


def read_exception_group_after_tail(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> tuple[int, int | None, int | None] | None:
    if cursor >= end_index or instructions[cursor].opname != "POP_EXCEPT":
        return None
    cursor += 1

    trailing_end_index = terminal_tail_end_index(
        instructions,
        cursor,
        end_index,
    )
    if trailing_end_index is None:
        return cursor, None, None
    return (
        find_exception_group_cleanup_end(
            instructions,
            trailing_end_index,
            end_index,
        ),
        cursor,
        trailing_end_index,
    )


def find_exception_group_cleanup_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cleanup_end = start_index
    for index in range(start_index, end_index):
        if is_terminal_op(instructions[index].opname):
            cleanup_end = index + 1
    return cleanup_end


def try_translate_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_try_finally_pattern(instructions, offset_to_index, cursor, end_index)
    if pattern is None:
        return None

    body = decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )
    final_body = decompiler.translate_child_statements(
        instructions,
        pattern.final_start_index,
        pattern.final_end_index,
    )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[],
            orelse=[],
            finalbody=final_body or [ast.Pass()],
        )
    )
    return pattern.after_index


def find_try_except_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
) -> TryExceptPattern | None:
    shape = read_legacy_try_except_shape(
        instructions,
        offset_to_index,
        setup_index,
        end_index,
    )
    if shape is None:
        return None

    handler = read_simple_handler(
        instructions,
        offset_to_index,
        shape.handler_start_index,
        shape.after_index,
        shape.after_offset,
    )
    if handler is None:
        return None

    return TryExceptPattern(
        body_start_index=shape.body_start_index,
        body_end_index=shape.body_end_index,
        handler_start_index=shape.handler_start_index,
        handler_body_start_index=handler.body_start_index,
        handler_body_end_index=handler.body_end_index,
        exception_type_index=handler.exception_type_index,
        after_index=shape.after_index,
        name=handler.name,
    )


def read_legacy_try_except_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
) -> LegacyTryExceptShape | None:
    setup = instructions[setup_index]
    if setup.opname != "SETUP_EXCEPT":
        return None

    handler_start_index = offset_to_index.get(int(setup.argval))
    if handler_start_index is None or handler_start_index <= setup_index:
        return None
    if handler_start_index + 4 >= end_index:
        return None

    body_jump = read_legacy_try_except_body_jump(
        instructions,
        setup_index,
        handler_start_index,
    )
    if body_jump is None:
        return None
    body_jump_index, after_offset = body_jump

    after_index = jump_target_index_or_end(
        instructions,
        offset_to_index,
        after_offset,
    )
    if (
        after_index is None
        or after_index <= handler_start_index
        or after_index > end_index
    ):
        return None

    return LegacyTryExceptShape(
        body_start_index=setup_index + 1,
        body_end_index=body_jump_index - 1,
        handler_start_index=handler_start_index,
        after_index=after_index,
        after_offset=after_offset,
    )


def read_legacy_try_except_body_jump(
    instructions: list[Instruction],
    setup_index: int,
    handler_start_index: int,
) -> tuple[int, int] | None:
    body_jump_index = handler_start_index - 1
    if body_jump_index <= setup_index:
        return None
    body_jump = instructions[body_jump_index]
    if body_jump.opname not in {"JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP"}:
        return None
    return body_jump_index, int(body_jump.argval)


def find_try_finally_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
) -> TryFinallyPattern | None:
    setup = instructions[setup_index]
    if setup.opname != "SETUP_FINALLY":
        return None

    final_start_index = offset_to_index.get(int(setup.argval))
    if final_start_index is None or final_start_index <= setup_index:
        return None

    body_end_index = find_finally_body_end(
        instructions, setup_index + 1, final_start_index
    )
    if body_end_index is None:
        return None

    final_end_index = find_end_finally(instructions, final_start_index, end_index)
    if final_end_index is None:
        return None

    return TryFinallyPattern(
        body_start_index=setup_index + 1,
        body_end_index=body_end_index,
        final_start_index=final_start_index,
        final_end_index=final_end_index,
        after_index=final_end_index + 1,
    )


def jump_target_index_or_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    target_offset: int,
) -> int | None:
    target_index = offset_to_index.get(target_offset)
    if target_index is not None:
        return target_index
    if instructions and target_offset > instructions[-1].offset:
        return len(instructions)
    return None


def read_simple_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
    after_offset: int | None = None,
) -> SimpleHandler | None:
    handler_start_index = skip_exception_match_prefix(
        instructions,
        handler_start_index,
        after_index,
    )
    if handler_start_index >= after_index:
        return None

    bare_handler = read_bare_handler(instructions, handler_start_index, after_index)
    if bare_handler is not None:
        return bare_handler

    return read_legacy_exception_match_handler(
        instructions,
        offset_to_index,
        handler_start_index,
        after_index,
        after_offset,
    )


def read_legacy_exception_match_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
    after_offset: int | None,
) -> SimpleHandler | None:
    jump_handler = read_legacy_jump_if_not_exception_match_handler(
        instructions,
        offset_to_index,
        handler_start_index,
        after_index,
        after_offset,
    )
    if jump_handler is not None:
        return jump_handler

    match = read_legacy_exception_match(
        instructions,
        offset_to_index,
        handler_start_index,
        after_index,
    )
    if match is None:
        return None

    handler_body = read_named_handler_body(
        instructions,
        match.miss_jump_index + 1,
        match.miss_index,
        after_index,
        after_offset,
    )
    if handler_body is None:
        return None
    body_start_index, body_end_index, name = handler_body

    return SimpleHandler(
        exception_type_index=match.exception_type_index,
        exception_type_end_index=match.exception_type_index + 1,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        name=name,
    )


def read_legacy_jump_if_not_exception_match_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
    after_offset: int | None,
) -> SimpleHandler | None:
    if instructions[handler_start_index].opname != "DUP_TOP":
        return None
    exception_type_index = handler_start_index + 1
    compare_index = skip_exception_match_prefix(
        instructions,
        exception_type_index + 1,
        after_index,
    )
    if compare_index >= after_index:
        return None
    if instructions[compare_index].opname != "JUMP_IF_NOT_EXC_MATCH":
        return None

    return read_jump_if_not_exception_match_handler(
        instructions,
        offset_to_index,
        exception_type_index,
        compare_index,
        after_index,
        after_offset,
    )


def read_legacy_exception_match(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_start_index: int,
    after_index: int,
) -> LegacyExceptionMatch | None:
    if instructions[handler_start_index].opname != "DUP_TOP":
        return None
    exception_type_index = handler_start_index + 1
    compare_index = skip_exception_match_prefix(
        instructions,
        exception_type_index + 1,
        after_index,
    )
    if not is_legacy_compare_exception_match(instructions, compare_index, after_index):
        return None

    miss_jump_index = skip_exception_match_prefix(
        instructions,
        compare_index + 1,
        after_index,
    )
    if miss_jump_index >= after_index:
        return None
    miss_jump = instructions[miss_jump_index]
    if "IF_FALSE" not in miss_jump.opname and "IF_TRUE" not in miss_jump.opname:
        return None

    miss_index = offset_to_index.get(int(miss_jump.argval))
    if miss_index is None or miss_index <= handler_start_index:
        return None

    return LegacyExceptionMatch(
        exception_type_index=exception_type_index,
        miss_jump_index=miss_jump_index,
        miss_index=miss_index,
    )


def is_legacy_compare_exception_match(
    instructions: list[Instruction],
    compare_index: int,
    after_index: int,
) -> bool:
    if compare_index >= after_index:
        return False
    instruction = instructions[compare_index]
    return instruction.opname == "COMPARE_OP" and instruction.argrepr == (
        "exception-match"
    )


def read_named_handler_body(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    after_index: int,
    after_offset: int | None,
) -> tuple[int, int, str | None] | None:
    body_start_index = skip_exception_stack_pops(
        instructions, body_start_index, miss_index
    )
    name = exception_handler_name(instructions, body_start_index, miss_index)
    if name is not None:
        body_start_index = skip_exception_stack_pops(
            instructions, body_start_index + 1, miss_index
        )

    body_end_index = find_handler_body_end(
        instructions, body_start_index, miss_index, after_index, after_offset
    )
    if body_end_index is None:
        return None
    return body_start_index, body_end_index, name


def read_jump_if_not_exception_match_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    exception_type_index: int,
    jump_index: int,
    after_index: int,
    after_offset: int | None = None,
) -> SimpleHandler | None:
    miss_jump = instructions[jump_index]
    miss_index = offset_to_index.get(int(miss_jump.argval))
    if miss_index is None or miss_index <= jump_index:
        return None

    body_start_index = skip_exception_stack_pops(
        instructions, jump_index + 1, miss_index
    )
    name = exception_handler_name(instructions, body_start_index, miss_index)
    if name is not None:
        body_start_index = skip_exception_stack_pops(
            instructions, body_start_index + 1, miss_index
        )

    body_end_index = find_handler_body_end(
        instructions, body_start_index, miss_index, after_index, after_offset
    )
    if body_end_index is None:
        return None

    return SimpleHandler(
        exception_type_index=exception_type_index,
        exception_type_end_index=jump_index,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        name=name,
    )


def read_bare_handler(
    instructions: list[Instruction],
    handler_start_index: int,
    after_index: int,
) -> SimpleHandler | None:
    if handler_start_index + 2 >= after_index:
        return None
    for index in range(handler_start_index, handler_start_index + 3):
        if instructions[index].opname != "POP_TOP":
            return None

    body_start_index = handler_start_index + 3
    body_end_index = find_handler_body_end(
        instructions,
        body_start_index,
        after_index,
        after_index,
        None,
    )
    if body_end_index is None:
        return None

    return SimpleHandler(
        exception_type_index=-1,
        exception_type_end_index=-1,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
    )


def skip_exception_match_prefix(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index
    while cursor < end_index and instructions[cursor].opname in {
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
        "SET_LINENO",
    }:
        cursor += 1
    return cursor


def skip_exception_stack_pops(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index
    while cursor < end_index and instructions[cursor].opname in {
        "POP_EXCEPT",
        "POP_TOP",
    }:
        cursor += 1
    return cursor


def find_handler_body_end(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
    after_index: int,
    after_offset: int | None = None,
) -> int | None:
    if after_offset is None and after_index < len(instructions):
        after_offset = instructions[after_index].offset
    for index in range(body_start_index, miss_index):
        instruction = instructions[index]
        if instruction.opname not in {"JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP"}:
            continue
        if instruction.argval == after_offset:
            return index
    if miss_index > body_start_index:
        return miss_index
    return None


def find_finally_body_end(
    instructions: list[Instruction],
    start_index: int,
    final_start_index: int,
) -> int | None:
    for index in range(final_start_index - 1, start_index - 1, -1):
        if instructions[index].opname == "POP_BLOCK":
            return index
    return final_start_index


def find_end_finally(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname == "END_FINALLY":
            return index
    return None


@dataclass(frozen=True, slots=True)
class TableSpan:
    start_index: int
    end_index: int
    target_index: int


@dataclass(frozen=True, slots=True)
class OutOfLineClause:
    type_start_index: int | None
    type_end_index: int
    name: str | None
    body_start_index: int
    body_end_index: int
    exit_jump_index: int | None
    resume_index: int | None


@dataclass(frozen=True, slots=True)
class OutOfLineTryPattern:
    handler_index: int
    handler_end_index: int
    body_start_index: int
    body_end_index: int
    clauses: tuple[OutOfLineClause, ...]
    else_end_index: int
    final_end_index: int | None = None
    is_star: bool = False
    final_start_index: int | None = None
    final_after_index: int | None = None
    handler_final_range: tuple[int, int] | None = None
    break_index: int | None = None


HANDLER_EXIT_JUMPS = {"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD"}
NOT_EXCEPTION_TYPE_OPS = {
    "CHECK_EG_MATCH",
    "POP_TOP",
    "PUSH_EXC_INFO",
    "WITH_EXCEPT_START",
}


def try_translate_out_of_line_try(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    pattern = find_out_of_line_try_pattern(
        decompiler,
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if pattern is None:
        return None

    decompiler.claimed_ranges[pattern.handler_index] = pattern.handler_end_index
    claim_finally_copies(decompiler, instructions, pattern)
    claim_try_body_exits(decompiler, instructions, offset_to_index, pattern)
    body = translate_guarded_statements(
        decompiler,
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
        pattern.handler_index,
        pattern.break_index,
    )
    body = restore_handler_exits(fold_handler_exits(body))
    ends_loop_body = closes_loop_body(decompiler, instructions, pattern, end_index)
    handlers = [
        make_out_of_line_handler(
            decompiler,
            instructions,
            clause,
            pattern.is_star,
            ends_loop_body,
            pattern.else_end_index,
        )
        for clause in pattern.clauses
    ]
    orelse = decompiler.translate_child_statements(
        instructions,
        pattern.body_end_index,
        pattern.else_end_index,
    )
    if pattern.final_end_index is not None:
        final_body = translate_finally_block(
            decompiler,
            instructions,
            (
                pattern.final_start_index or pattern.body_end_index,
                pattern.final_end_index,
            ),
            pattern.handler_final_range,
        )
        if pattern.final_after_index is not None:
            final_body = without_closing_returns(final_body)
        decompiler.statements.append(make_try_finally(body, final_body))
        return finally_after_index(instructions, pattern)

    break_end = break_closing_try_body(decompiler, instructions, pattern, end_index)
    if break_end is not None:
        body.append(ast.Break())
    make_try = getattr(ast, "TryStar", ast.Try) if pattern.is_star else ast.Try
    if pattern.is_star and make_try is ast.Try:
        decompiler.warnings.append(
            "except* cannot be written by this Python (needs 3.11); "
            "the clauses are written as plain except"
        )
    decompiler.statements.append(
        make_try(
            body=body or [ast.Pass()],
            handlers=handlers,
            orelse=orelse,
            finalbody=[],
        )
    )
    append_statement_behind_clause(decompiler, instructions, pattern)
    return (
        break_end
        or end_behind_closing_jump(decompiler, instructions, pattern, end_index)
        or end_behind_inline_handler(instructions, pattern)
    )


def end_behind_closing_jump(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
    end_index: int,
) -> int | None:
    if not stands_in_line(decompiler) or pattern.handler_end_index != end_index:
        return None
    if pattern.else_end_index + 1 != pattern.handler_index:
        return None
    jump = instructions[pattern.else_end_index]
    if jump.opname != "JUMP_BACKWARD":
        return None
    if jump.argval not in decompiler.loop_continue_offsets:
        return None
    return end_index


def end_behind_inline_handler(
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
) -> int:
    jump = instructions[pattern.else_end_index]
    if pattern.else_end_index + 1 != pattern.handler_index or (
        pattern.else_end_index == pattern.body_end_index
    ):
        return pattern.else_end_index
    if pattern.handler_end_index >= len(instructions) or (
        jump.opname != "JUMP_FORWARD"
        or jump.argval != instructions[pattern.handler_end_index].offset
    ):
        return pattern.else_end_index
    return pattern.handler_end_index


def handler_break_target(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> int | None:
    if index + 2 >= end_index:
        return None
    pop, leave, jump = instructions[index : index + 3]
    if pop.opname != "POP_TOP" or pop.starts_line is None:
        return None
    if leave.opname != "POP_EXCEPT":
        return None
    if jump.opname not in {"JUMP_FORWARD", "JUMP_BACKWARD"}:
        return None
    return int(jump.argval)


def handler_break_return_end(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> int | None:
    if index + 3 >= end_index:
        return None
    pop, leave, first = instructions[index : index + 3]
    if pop.opname != "POP_TOP" or pop.starts_line is None:
        return None
    if leave.opname != "POP_EXCEPT" or first.starts_line is None:
        return None
    for cursor in range(index + 2, end_index):
        opname = instructions[cursor].opname
        if opname == "RETURN_VALUE":
            return cursor + 1
        if "JUMP" in opname or opname in {"FOR_ITER", "POP_TOP", "RETURN_CONST"}:
            return None
    return None


def while_handler_break_target(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> int | None:
    if index + 1 >= end_index:
        return None
    leave, jump = instructions[index : index + 2]
    if leave.opname != "POP_EXCEPT" or leave.starts_line is None:
        return None
    if jump.opname not in {"JUMP_FORWARD", "JUMP_BACKWARD"}:
        return None
    return int(jump.argval)


def break_closing_try_body(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
    end_index: int,
) -> int | None:
    index = pattern.body_end_index
    if pattern.else_end_index != index or not decompiler.loop_continue_offsets:
        return None
    if instructions[index].is_jump_target:
        return None
    if is_backward_break_314(decompiler, instructions[index]):
        return index + 1
    copied_return_end = handler_break_return_end(instructions, index, end_index)
    if copied_return_end is not None:
        return copied_return_end
    target = handler_break_target(instructions, index, end_index)
    if target is None:
        return plain_break_end(decompiler, instructions, index)
    if target in decompiler.loop_continue_offsets | decompiler.detached_resume_offsets:
        return None
    return index + 3


def plain_break_end(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    index: int,
) -> int | None:
    if instructions[index].opname != "POP_TOP":
        return None
    jump = instructions[index + 1]
    if (
        jump.opname != "JUMP_FORWARD"
        or jump.argval not in decompiler.loop_break_offsets
    ):
        return None
    return index + 2


def is_backward_break_314(decompiler: DecompilerContext, jump: Instruction) -> bool:
    # python 3.14: a break of a loop that ends the body of another loop
    # jumps back to where that loop goes on. at the end of a try body it is
    # not the jump over the handlers, which would go forward
    if decompiler.version is None or decompiler.version < (3, 14):
        return False
    return (
        jump.opname == "JUMP_BACKWARD"
        and jump.argval in decompiler.loop_break_offsets
        and jump.argval not in decompiler.loop_continue_offsets
    )


def append_statement_behind_clause(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
) -> None:
    for clause in pattern.clauses:
        resume_index = clause.resume_index
        if resume_index is None or resume_index <= pattern.handler_index:
            continue
        tail_end = terminal_tail_end_index(
            instructions, resume_index, pattern.handler_end_index
        )
        if tail_end is not None:
            decompiler.statements.extend(
                decompiler.translate_child_statements(
                    instructions, resume_index, tail_end
                )
            )
        return


def finally_after_index(
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
) -> int:
    if pattern.final_after_index is not None:
        return pattern.final_after_index
    if pattern.final_start_index is not None:
        return pattern.body_end_index
    final_end_index = pattern.final_end_index or pattern.body_end_index
    return_index = final_end_index
    while (
        instructions[return_index].opname == "SWAP"
        and instructions[return_index + 1].opname == "POP_TOP"
    ):
        return_index += 2
    if is_return_after_finally(instructions, return_index, pattern.handler_index):
        return return_index + 1
    return final_end_index


def claim_try_body_exits(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    pattern: OutOfLineTryPattern,
) -> None:
    after_index = pattern.body_end_index
    while instructions[after_index].opname == "NOP":
        after_index += 1
    spans = sorted(
        (span.start_index, span.end_index)
        for span in read_table_spans(decompiler.code, instructions, offset_to_index)
        if span.target_index == pattern.handler_index
    )
    for (_, hole_start), (hole_end, _) in pairwise(spans):
        length = hole_end - hole_start
        if length < 1 or not is_terminal_op(instructions[hole_end - 1].opname):
            continue
        if hole_end > pattern.body_end_index or after_index + length > len(
            instructions
        ):
            continue
        if same_instruction_range_behavior(
            instructions, hole_start, hole_end, after_index, after_index + length
        ):
            decompiler.claimed_ranges[hole_start] = hole_end
            decompiler.handler_exit_indexes.add(hole_start)


def make_try_finally(body: list[ast.stmt], final_body: list[ast.stmt]) -> ast.Try:
    return ast.Try(
        body=body or [ast.Pass()],
        handlers=[],
        orelse=[],
        finalbody=final_body or [ast.Pass()],
    )


def translate_guarded_statements(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    handler_index: int,
    break_index: int | None = None,
) -> list[ast.stmt]:
    child = decompiler.make_child()
    child.active_handlers = decompiler.active_handlers | {handler_index}
    if break_index is not None:
        child.loop_break_offsets = decompiler.loop_break_offsets | {
            int(instructions[break_index].offset)
        }
    statements = child.translate_range(instructions, start_index, end_index)
    if len(child.stack) > len(decompiler.stack):
        value = child.expression_from_stack_value(child.stack[-1])
        statements.append(ast.Return(value=value))
    return statements


def closes_loop_body(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    pattern: OutOfLineTryPattern,
    end_index: int,
) -> bool:
    if pattern.else_end_index < end_index or end_index >= len(instructions):
        return False
    jump = instructions[end_index]
    return (
        jump.opname == "JUMP_BACKWARD"
        and jump.argval in decompiler.loop_continue_offsets
    )


def make_out_of_line_handler(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
    is_star: bool,
    ends_loop_body: bool,
    after_index: int,
) -> ast.ExceptHandler:
    exception_type = None
    if clause.type_start_index is not None:
        exception_type = decompiler.evaluate_expression_range(
            instructions,
            clause.type_start_index,
            clause.type_end_index,
        )
    claim_handler_exits(decompiler, instructions, clause, after_index)
    falls_out_to_continue = ends_loop_body and any(
        index in decompiler.handler_exit_indexes
        for index in range(clause.body_start_index, clause.body_end_index)
    )
    body = decompiler.translate_child_statements(
        instructions,
        clause.body_start_index,
        clause.body_end_index,
    )
    body = restore_handler_exits(fold_handler_exits(body))
    body = remove_named_exception_cleanup(body, clause.name)
    loop_exit = handler_loop_exit(decompiler, instructions, clause)
    if loop_exit is not None and not is_star and not falls_out_to_continue:
        body.append(loop_exit)
    return ast.ExceptHandler(
        type=exception_type,
        name=clause.name,
        body=body or [ast.Pass()],
    )


HANDLER_EXIT_MARK = "_pyc2py_handler_exit"


def make_handler_exit_marker(replaced: list[ast.stmt]) -> ast.stmt:
    marker = ast.Pass()
    setattr(marker, HANDLER_EXIT_MARK, replaced)
    return marker


def is_handler_exit_marker(statement: ast.stmt) -> bool:
    return hasattr(statement, HANDLER_EXIT_MARK)


def restore_handler_exits(statements: list[ast.stmt]) -> list[ast.stmt]:
    restored: list[ast.stmt] = []
    for statement in statements:
        if is_handler_exit_marker(statement):
            restored.extend(getattr(statement, HANDLER_EXIT_MARK))
            continue
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(statement, field, None)
            if isinstance(nested, list) and nested:
                setattr(statement, field, restore_handler_exits(nested) or [ast.Pass()])
        for handler in getattr(statement, "handlers", []):
            handler.body = restore_handler_exits(handler.body) or [ast.Pass()]
        restored.append(statement)
    return restored


def claim_handler_exits(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
    after_index: int,
) -> None:
    for index in range(clause.body_start_index, clause.body_end_index):
        if instructions[index].opname != "POP_EXCEPT":
            continue
        exit_end = exit_to_statement_end(instructions, clause, index + 1, after_index)
        if exit_end is None and clause.name is None:
            exit_end = handler_exit_end(instructions, clause, index + 1)
        if exit_end is not None:
            decompiler.claimed_ranges[index] = exit_end
            decompiler.handler_exit_indexes.add(index)


def exit_to_statement_end(
    instructions: list[Instruction],
    clause: OutOfLineClause,
    start_index: int,
    after_index: int,
) -> int | None:
    if clause.exit_jump_index is not None or clause.resume_index is not None:
        return None
    if clause.name is not None:
        if (
            find_named_exception_cleanup(
                instructions, start_index, start_index + 3, clause.name
            )
            != start_index
        ):
            return None
        start_index += 3
    jump = instructions[start_index]
    if jump.opname not in HANDLER_EXIT_JUMPS or after_index >= len(instructions):
        return None
    if jump.argval != instructions[after_index].offset:
        return None
    return start_index + 1


def handler_exit_end(
    instructions: list[Instruction],
    clause: OutOfLineClause,
    start_index: int,
) -> int | None:
    if clause.exit_jump_index is not None:
        jump = instructions[start_index]
        same_target = jump.argval == instructions[clause.exit_jump_index].argval
        if jump.opname in HANDLER_EXIT_JUMPS and same_target:
            return start_index + 1
        return None
    if clause.resume_index is None:
        return None
    tail_end = terminal_tail_end_index(
        instructions, clause.resume_index, len(instructions)
    )
    if tail_end is None:
        return None
    length = tail_end - clause.resume_index
    if same_instruction_range_behavior(
        instructions,
        start_index,
        start_index + length,
        clause.resume_index,
        tail_end,
    ):
        return start_index + length
    return None


def fold_handler_exits(statements: list[ast.stmt]) -> list[ast.stmt]:
    folded: list[ast.stmt] = []
    for index, statement in enumerate(statements):
        if is_handler_exit_marker(statement):
            break
        folded.append(statement)
        if not isinstance(statement, ast.If):
            continue
        leaves = bool(statement.body) and is_handler_exit_marker(statement.body[-1])
        statement.body = fold_handler_exits(statement.body) or [ast.Pass()]
        statement.orelse = fold_else_handler_exit(statement.orelse)
        if leaves and not statement.orelse:
            statement.orelse = fold_handler_exits(statements[index + 1 :])
            break
    return folded


def fold_else_handler_exit(statements: list[ast.stmt]) -> list[ast.stmt]:
    if statements and is_handler_exit_marker(statements[-1]):
        exit_code = getattr(statements[-1], HANDLER_EXIT_MARK)
        if exit_code:
            return [*fold_handler_exits(statements[:-1]), *exit_code]
    return fold_handler_exits(statements)


def handler_loop_exit(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> ast.stmt | None:
    if clause.exit_jump_index is None:
        if decompiler.loop_none_return_is_break and clause_returns_none(
            instructions, clause
        ):
            return ast.Break()
        return handler_copied_break(decompiler, instructions, clause)
    target = instructions[clause.exit_jump_index].argval
    if target in decompiler.loop_continue_offsets:
        return ast.Continue()
    if target in decompiler.loop_break_offsets:
        return ast.Break()
    return None


def clause_returns_none(
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> bool:
    cursor = clause.body_end_index
    if instructions[cursor].opname != "POP_EXCEPT":
        return False
    while instructions[cursor].opname == "POP_EXCEPT":
        cursor += 1
    if clause.name is not None:
        cursor += 3
    last = instructions[cursor]
    return last.opname == "RETURN_CONST" and last.argval is None


def handler_copied_break(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> ast.stmt | None:
    resume_index = clause.resume_index
    if resume_index is None:
        return None
    if instructions[resume_index].offset in decompiler.loop_break_offsets:
        return ast.Break()
    if (
        0 < resume_index < len(instructions) - 1
        and instructions[resume_index].opname == "POP_TOP"
        and (
            instructions[resume_index - 1].opname == "END_FOR"
            or instructions[resume_index].starts_line is not None
        )
        and instructions[resume_index + 1].offset in decompiler.loop_break_offsets
    ):
        return ast.Break()
    return None


def find_try_handler(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    cursor: int,
    end_index: int,
) -> tuple[int, int] | None:
    if any(span.start_index == cursor for span in spans):
        return find_out_of_line_handler(
            instructions,
            spans,
            cursor,
            end_index,
            decompiler.active_handlers,
        )
    return find_handler_of_empty_try(
        instructions, spans, cursor, decompiler.active_handlers
    )


def find_out_of_line_try_pattern(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> OutOfLineTryPattern | None:
    spans = read_table_spans(decompiler.code, instructions, offset_to_index)
    found = find_try_handler(decompiler, instructions, spans, cursor, end_index)
    if found is None:
        return None
    handler_index, body_end_index = found
    cleanup_index = handler_cleanup_index(instructions, spans, handler_index)
    if cleanup_index is None:
        return None
    body_end_index = unprotected_return_end(instructions, body_end_index, end_index)
    body_end_index = unprotected_branch_end(
        instructions, offset_to_index, cursor, body_end_index, end_index
    )

    clauses = read_out_of_line_clauses(
        instructions,
        offset_to_index,
        spans,
        handler_index,
        cleanup_index,
        body_end_index,
    )
    clauses = clauses_returning_through_finally(decompiler, instructions, clauses)
    if has_later_handler(spans, body_end_index, handler_index, clauses) or any(
        leaves_enclosing_handler(instructions, clause) for clause in clauses
    ):
        clauses = tuple(
            clause_with_own_return(decompiler, instructions, clause)
            for clause in clauses
        )
    is_star = False
    if not clauses:
        clauses = read_out_of_line_star_clauses(
            instructions,
            offset_to_index,
            handler_index,
            cleanup_index,
            body_end_index,
        )
        is_star = bool(clauses)
    if not clauses:
        final = find_out_of_line_finally_pattern(
            instructions,
            spans,
            (cursor, body_end_index, end_index),
            handler_index,
            cleanup_index,
        )
        return with_inline_handlers_of_final_block(
            decompiler, instructions, spans, final
        )

    else_end_index = out_of_line_else_end(
        decompiler,
        instructions,
        offset_to_index,
        clauses,
        handler_index,
        body_end_index,
        end_index,
    )
    if else_end_index is None:
        return None
    else_end_index = else_end_before_handler(
        instructions,
        clauses,
        handler_index,
        (body_end_index, else_end_index),
        end_index,
    )
    else_end_index = else_end_from_name_numbers(
        decompiler,
        instructions,
        spans,
        clauses,
        (body_end_index, else_end_index),
        (handler_index, cleanup_index + 3, end_index),
    )
    if decompiler.version is not None and decompiler.version < (3, 13):
        else_end_index = else_end_by_numbering(
            decompiler.code,
            instructions,
            (body_end_index, else_end_index, handler_index, cleanup_index + 3),
            all(clause_leaves_on_its_own(clause) for clause in clauses),
        )
    elif decompiler.version is not None and decompiler.version >= (3, 14):
        else_end_index = else_end_by_borrowed_loads(
            instructions,
            (body_end_index, else_end_index, handler_index),
            all(clause_leaves_on_its_own(clause) for clause in clauses),
        )
    if not is_star:
        clauses = with_unshared_returns(
            decompiler, instructions, clauses, handler_index, else_end_index
        )
    return OutOfLineTryPattern(
        handler_index=handler_index,
        handler_end_index=cleanup_index + 3,
        body_start_index=cursor,
        body_end_index=body_end_index,
        clauses=clauses,
        else_end_index=else_end_index,
        is_star=is_star,
    )


def with_inline_handlers_of_final_block(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    pattern: OutOfLineTryPattern | None,
) -> OutOfLineTryPattern | None:
    if pattern is None or not stands_in_line(decompiler):
        return pattern
    end_index = pattern.final_end_index
    if end_index is None or pattern.final_start_index is not None:
        return pattern
    if instructions[end_index].opname != "JUMP_FORWARD":
        return pattern
    regions = handler_regions(instructions, spans)
    inside = range(pattern.body_end_index, end_index)
    index = end_index + 1
    while (
        index in regions
        and regions[index] <= pattern.handler_index
        and all(
            span.start_index in inside for span in spans if span.target_index == index
        )
    ):
        index = regions[index]
    if index == end_index + 1:
        return pattern
    if instructions[index].opname == "JUMP_FORWARD":
        index += 1
    if index != pattern.handler_index:
        return pattern
    return replace(pattern, final_end_index=index)


def else_end_by_numbering(
    code: object,
    instructions: list[Instruction],
    bounds: tuple[int, int, int, int],
    handlers_leave: bool,
) -> int:
    body_end_index, else_end_index, handler_index, handler_end_index = bounds
    if body_end_index >= handler_index:
        return else_end_index
    if not is_terminal_op(instructions[handler_index - 1].opname):
        return else_end_index
    block_end_index = else_end_index
    if else_end_index == body_end_index:
        block_end_index = handler_index
    comes_first = numbered_before(
        code,
        instructions,
        (body_end_index, block_end_index),
        (handler_index, handler_end_index),
    )
    if comes_first and else_end_index == body_end_index and handlers_leave:
        return handler_index
    if comes_first is False and else_end_index > body_end_index:
        return body_end_index
    return else_end_index


def else_end_by_borrowed_loads(
    instructions: list[Instruction],
    bounds: tuple[int, int, int],
    handlers_leave: bool,
) -> int:
    body_end_index, else_end_index, handler_index = bounds
    if body_end_index >= handler_index:
        return else_end_index
    if not is_terminal_op(instructions[handler_index - 1].opname):
        return else_end_index
    block_end_index = else_end_index
    if else_end_index == body_end_index:
        block_end_index = handler_index
    loads = [
        instruction.borrowed
        for instruction in instructions[body_end_index:block_end_index]
        if instruction.opname == "LOAD_FAST"
    ]
    if loads and all(loads) and else_end_index == body_end_index and handlers_leave:
        return handler_index
    if loads and not any(loads) and else_end_index > body_end_index:
        return body_end_index
    return else_end_index


def clause_leaves_on_its_own(clause: OutOfLineClause) -> bool:
    return clause.resume_index is None and clause.exit_jump_index is None


def numbered_before(
    code: object,
    instructions: list[Instruction],
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool | None:
    seen = set(numbered_operands(code, instructions[: first[0]]))
    for kind in ("name", "const"):
        new_first = [
            number
            for operand in numbered_operands(code, instructions[first[0] : first[1]])
            if operand not in seen and operand[0] == kind
            for number in [operand[1]]
        ]
        new_second = [
            number
            for operand in numbered_operands(code, instructions[second[0] : second[1]])
            if operand not in seen and operand[0] == kind
            for number in [operand[1]]
        ]
        if new_first and new_second and min(new_first) != min(new_second):
            return min(new_first) < min(new_second)
        if new_first and new_second:
            return order_behind_shared_operand(set(new_first), set(new_second))
    return None


def order_behind_shared_operand(first: set[int], second: set[int]) -> bool | None:
    first_leads = not second - first or max(first) < min(second - first)
    second_leads = not first - second or max(second) < min(first - second)
    if first_leads == second_leads:
        return None
    return first_leads


# the flag of code that keeps its locals in slots: functions and lambdas
CO_OPTIMIZED = 0x1


def numbered_operands(
    code: object,
    instructions: list[Instruction],
) -> list[tuple[str, int]]:
    names = list(getattr(code, "co_names", ()))
    first_met = 1 if int(getattr(code, "co_flags", 0) or 0) & CO_OPTIMIZED else 0
    operands: list[tuple[str, int]] = []
    for instruction in instructions:
        opname = instruction.opname
        if opname in {"LOAD_CONST", "RETURN_CONST"} and instruction.arg is not None:
            if instruction.arg >= first_met:
                operands.append(("const", instruction.arg))
        elif (
            isinstance(instruction.argval, str)
            and instruction.argval in names
            and any(part in opname for part in ("NAME", "GLOBAL", "ATTR", "METHOD"))
        ):
            operands.append(("name", names.index(instruction.argval)))
    return operands


def find_out_of_line_finally_pattern(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    body: tuple[int, int, int],
    handler_index: int,
    cleanup_index: int,
) -> OutOfLineTryPattern | None:
    cursor, body_end_index, end_index = body
    if ends_in_endless_loop(instructions, body_end_index):
        return find_finally_around_endless_loop(
            instructions, spans, body, handler_index, cleanup_index
        )
    branching = read_branching_finally_range(
        instructions, body_end_index, handler_index
    )
    if branching is not None and (
        branching.final_end_index <= end_index + 1
        and instructions[branching.final_end_index].opname != "RETURN_VALUE"
    ):
        return OutOfLineTryPattern(
            handler_index=handler_index,
            handler_end_index=cleanup_index + 3,
            body_start_index=cursor,
            body_end_index=body_end_index,
            clauses=(),
            else_end_index=body_end_index,
            final_end_index=min(branching.final_end_index, end_index),
            handler_final_range=(handler_index + 1, cleanup_index),
        )
    final_range = out_of_line_final_end(
        instructions,
        spans,
        handler_index,
        cleanup_index,
        body_end_index,
        end_index,
    )
    if final_range is None:
        if branching is None or handler_index > end_index:
            return None
        if instructions[branching.final_end_index].opname != "RETURN_VALUE":
            return None
    if final_range is None or (
        handler_index <= end_index
        and returns_value_from_several_ends(instructions, body_end_index, final_range)
    ):
        return OutOfLineTryPattern(
            handler_index=handler_index,
            handler_end_index=cleanup_index + 3,
            body_start_index=cursor,
            body_end_index=body_end_index,
            clauses=(),
            else_end_index=body_end_index,
            final_end_index=handler_index,
            handler_final_range=(handler_index + 1, cleanup_index),
        )
    return OutOfLineTryPattern(
        handler_index=handler_index,
        handler_end_index=cleanup_index + 3,
        body_start_index=cursor,
        body_end_index=body_end_index,
        clauses=(),
        else_end_index=body_end_index,
        final_start_index=final_range[0],
        final_end_index=final_range[1],
        final_after_index=final_range[2],
    )


def returns_value_from_several_ends(
    instructions: list[Instruction],
    body_end_index: int,
    final_range: tuple[int | None, int, int | None],
) -> bool:
    final_start_index, final_end_index, final_after_index = final_range
    if final_start_index is not None or final_after_index is not None:
        return False
    last = instructions[final_end_index]
    if last.opname != "RETURN_VALUE":
        return False
    return any(
        is_jump_op(instruction.opname)
        and isinstance(instruction.argval, int)
        and instruction.argval > last.offset
        for instruction in instructions[body_end_index:final_end_index]
    )


def find_finally_around_endless_loop(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    body: tuple[int, int, int],
    handler_index: int,
    cleanup_index: int,
) -> OutOfLineTryPattern | None:
    cursor, body_end_index, end_index = body
    regions = handler_regions(instructions, spans)
    copy_index = body_end_index
    while copy_index in regions and copy_index != handler_index:
        copy_index = regions[copy_index]
    final_end_index: int | None = body_end_index
    if copy_index != handler_index:
        final_end_index = finally_copy_end(instructions, copy_index, handler_index)
    if final_end_index is None or final_end_index > end_index:
        return None
    return OutOfLineTryPattern(
        handler_index=handler_index,
        handler_end_index=cleanup_index + 3,
        body_start_index=cursor,
        body_end_index=body_end_index,
        clauses=(),
        else_end_index=body_end_index,
        final_end_index=final_end_index,
        handler_final_range=(handler_index + 1, cleanup_index),
        break_index=None if copy_index == handler_index else copy_index,
    )


def finally_copy_end(
    instructions: list[Instruction],
    copy_index: int,
    handler_index: int,
) -> int | None:
    branching = read_branching_finally_range(instructions, copy_index, handler_index)
    if branching is not None:
        return branching.final_end_index
    start_index = handler_index + 1
    length = handler_index - copy_index
    for step in range(length):
        left = instructions[start_index + step]
        if left.opname == "RERAISE":
            return copy_index + step
        if left.opname != instructions[copy_index + step].opname:
            return None
    return None


def with_unshared_returns(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clauses: tuple[OutOfLineClause, ...],
    handler_index: int,
    else_end_index: int,
) -> tuple[OutOfLineClause, ...]:
    settled: list[OutOfLineClause] = []
    for clause in clauses:
        resume_index = clause.resume_index
        code_end_index = None
        if (
            resume_index is not None
            and else_end_index != resume_index < handler_index
            and not resumes_behind_enclosing_exit(instructions, clause, else_end_index)
            and handler_loop_exit(decompiler, instructions, clause) is None
        ):
            code_end_index = terminal_tail_end_index(
                instructions, clause.body_end_index, len(instructions)
            )
        if code_end_index is not None:
            clause = replace(clause, body_end_index=code_end_index, resume_index=None)
        settled.append(clause)
    return tuple(settled)


def resumes_behind_enclosing_exit(
    instructions: list[Instruction],
    clause: OutOfLineClause,
    else_end_index: int,
) -> bool:
    if clause.resume_index != else_end_index + 1:
        return False
    if instructions[else_end_index].opname != "POP_EXCEPT":
        return False
    return instructions[clause.body_end_index + 1].opname == "POP_EXCEPT"


def has_later_handler(
    spans: tuple[TableSpan, ...],
    body_end_index: int,
    handler_index: int,
    clauses: tuple[OutOfLineClause, ...],
) -> bool:
    resumes = [
        clause.resume_index
        for clause in clauses
        if clause.exit_jump_index is None
        and clause.resume_index is not None
        and clause.resume_index < handler_index
    ]
    if not resumes:
        return False
    enclosing = {
        span.target_index for span in spans if span.start_index < body_end_index
    }
    return any(
        body_end_index <= span.start_index < max(resumes)
        and span.target_index > handler_index
        and span.target_index not in enclosing
        for span in spans
    )


def clause_with_own_return(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> OutOfLineClause:
    if clause.resume_index is None or clause.exit_jump_index is not None:
        return clause
    if clause.resume_index > clause.body_start_index:
        return clause
    exit_index = clause.body_end_index
    while instructions[exit_index].opname == "POP_EXCEPT":
        exit_index += 1
    if clause.name is not None:
        exit_index += 3
    return_index = exit_index
    while not is_terminal_op(instructions[return_index].opname):
        return_index += 1
    decompiler.claimed_ranges[clause.body_end_index] = exit_index
    return replace(clause, body_end_index=return_index + 1, resume_index=None)


def clauses_returning_through_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clauses: tuple[OutOfLineClause, ...],
) -> tuple[OutOfLineClause, ...]:
    return tuple(
        clause_returning_through_finally(decompiler, instructions, clause)
        for clause in clauses
    )


def clause_returning_through_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> OutOfLineClause:
    resume_index = clause.resume_index
    if resume_index is None or resume_index not in decompiler.claimed_ranges:
        return clause
    tail_end = terminal_tail_end_index(instructions, resume_index, len(instructions))
    if tail_end is None:
        return clause
    return replace(clause, body_end_index=tail_end, resume_index=None)


def unprotected_return_end(
    instructions: list[Instruction],
    body_end_index: int,
    end_index: int,
) -> int:
    index = body_end_index
    while (
        index + 1 < end_index
        and instructions[index].opname == "SWAP"
        and instructions[index + 1].opname == "POP_TOP"
    ):
        index += 2
    if index < end_index and instructions[index].opname == "RETURN_VALUE":
        return index + 1
    return branch_returns_end(instructions, body_end_index, end_index)


def branch_returns_end(
    instructions: list[Instruction],
    body_end_index: int,
    end_index: int,
) -> int:
    index = body_end_index
    while index < end_index and is_none_return(instructions[index]):
        index += 1
    return index if index - body_end_index > 1 else body_end_index


def is_none_return(instruction: Instruction) -> bool:
    return instruction.opname == "RETURN_CONST" and instruction.argval is None


def unprotected_branch_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    body_start_index: int,
    body_end_index: int,
    end_index: int,
) -> int:
    for index in range(body_start_index, body_end_index):
        jump = instructions[index]
        if not is_jump_op(jump.opname) or not isinstance(jump.argval, int):
            continue
        target_index = offset_to_index.get(jump.argval)
        if target_index is None or not body_end_index < target_index <= end_index:
            continue
        skipped = instructions[target_index - 1].opname
        if is_terminal_op(skipped) or skipped in {"JUMP_BACKWARD", "JUMP_FORWARD"}:
            body_end_index = target_index
    return body_end_index


def out_of_line_final_end(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    handler_index: int,
    cleanup_index: int,
    body_end_index: int,
    end_index: int,
) -> tuple[int | None, int, int | None] | None:
    copy_start_index = handler_index + 1
    if instructions[copy_start_index].opname == "WITH_EXCEPT_START":
        return None
    copy_end_index = cleanup_index
    for span in spans:
        if copy_start_index < span.target_index < copy_end_index and (
            instructions[span.target_index].opname != "CLEANUP_THROW"
        ):
            copy_end_index = span.target_index
    copy_end_index -= 1
    if copy_end_index < copy_start_index:
        return None
    if instructions[copy_end_index].opname != "RERAISE":
        return final_block_ending_function(
            instructions, spans, handler_index, cleanup_index, body_end_index, end_index
        )
    if instructions[body_end_index - 1].opname in {
        "RAISE_VARARGS",
        "RERAISE",
    } and not jumps_over_inline_handler(instructions, body_end_index):
        return copy_start_index, copy_end_index, None

    inline_start_index = inline_finally_start(
        instructions, body_end_index, copy_start_index
    )
    copy = [
        instruction.opname
        for instruction in instructions[copy_start_index:copy_end_index]
        if instruction.opname != "CLEANUP_THROW"
    ]
    final_end_index = inline_start_index + len(copy)
    if final_end_index > end_index:
        return None
    for offset, opname in enumerate(copy):
        if opname != instructions[inline_start_index + offset].opname:
            return None
    return None, final_end_index, None


def jumps_over_inline_handler(
    instructions: list[Instruction],
    body_end_index: int,
) -> bool:
    end_offset = instructions[body_end_index].offset
    return any(
        instruction.opname == "JUMP_FORWARD" and instruction.argval == end_offset
        for instruction in instructions[:body_end_index]
    )


def ends_in_endless_loop(instructions: list[Instruction], body_end_index: int) -> bool:
    return (
        instructions[body_end_index - 1].opname == "JUMP_BACKWARD"
        and instructions[body_end_index].opname == "PUSH_EXC_INFO"
    )


def final_block_ending_function(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    handler_index: int,
    cleanup_index: int,
    body_end_index: int,
    end_index: int,
) -> tuple[int | None, int, int | None] | None:
    copy_start_index = handler_index + 1
    if instructions[cleanup_index - 1].opname != "RERAISE":
        return None
    regions = handler_regions(instructions, spans)
    after_index = min(
        (index for index in regions if index > body_end_index), default=end_index + 1
    )
    if after_index > end_index:
        return None
    last = instructions[after_index - 1]
    if (last.opname != "RETURN_CONST" or last.argval is not None) and not (
        last.opname == "RETURN_VALUE" and is_none_load(instructions[after_index - 2])
    ):
        return None
    for offset in range(min(3, cleanup_index - 1 - copy_start_index)):
        left = instructions[copy_start_index + offset]
        right = instructions[body_end_index + offset]
        if left.opname != right.opname:
            return None
    while last.opname == "RETURN_VALUE" and (
        regions.get(after_index, handler_index + 1) <= handler_index
    ):
        after_index = regions[after_index]
    return None, after_index, after_index


def without_closing_returns(statements: list[ast.stmt]) -> list[ast.stmt]:
    if not statements:
        return statements
    last = statements[-1]
    if isinstance(last, ast.Return) and (
        last.value is None
        or (isinstance(last.value, ast.Constant) and last.value.value is None)
    ):
        return without_closing_returns(statements[:-1])
    if isinstance(last, ast.If):
        last.body = without_closing_returns(last.body) or [ast.Pass()]
        last.orelse = without_closing_returns(last.orelse)
    return statements


def read_table_spans(
    code: object,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
) -> tuple[TableSpan, ...]:
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    spans: list[TableSpan] = []
    for entry in parse_exception_table(table) if table else ():
        start_index = offset_to_index.get(entry.start_offset)
        end_index = offset_to_index.get(entry.end_offset, len(instructions))
        target_index = offset_to_index.get(entry.target_offset)
        if start_index is None or target_index is None:
            continue
        spans.append(TableSpan(start_index, end_index, target_index))
    return tuple(spans)


def handler_cleanup_index(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    handler_index: int,
) -> int | None:
    if instructions[handler_index].opname != "PUSH_EXC_INFO":
        return None
    for span in spans:
        if span.start_index != handler_index:
            continue
        cleanup_index = span.target_index
        if cleanup_index + 2 >= len(instructions):
            return None
        cleanup = instructions[cleanup_index : cleanup_index + 3]
        if [item.opname for item in cleanup] == ["COPY", "POP_EXCEPT", "RERAISE"]:
            return cleanup_index
    return None


def handler_regions(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
) -> dict[int, int]:
    regions: dict[int, int] = {}
    for span in spans:
        handler_index = span.target_index
        if handler_index in regions:
            continue
        cleanup_index = handler_cleanup_index(instructions, spans, handler_index)
        if cleanup_index is not None:
            regions[handler_index] = cleanup_index + 3
    return regions


def out_of_line_handlers_start(
    code: object,
    instructions: list[Instruction],
    end_index: int,
) -> int:
    if end_index < 1 or instructions[end_index - 1].opname != "RERAISE":
        return end_index
    spans = read_table_spans(code, instructions, make_offset_index(instructions))
    regions = handler_regions(instructions, spans)
    start_index = end_index
    if end_index > 1 and instructions[end_index - 2].opname == "CALL_INTRINSIC_1":
        start_index -= 2
    while True:
        starts = [start for start, end in regions.items() if end == start_index]
        if not starts:
            return start_index
        start_index = min(starts)


def out_of_line_handlers_end(
    code: object,
    instructions: list[Instruction],
    start_index: int,
) -> int:
    if start_index >= len(instructions) or (
        instructions[start_index].opname != "PUSH_EXC_INFO"
    ):
        return start_index
    spans = read_table_spans(code, instructions, make_offset_index(instructions))
    regions = handler_regions(instructions, spans)
    while start_index in regions:
        start_index = regions[start_index]
    return start_index


def enclosing_handler(
    regions: dict[int, int],
    index: int,
    skip: int | None = None,
) -> int | None:
    best: int | None = None
    for handler_index, region_end in regions.items():
        if handler_index == skip or not handler_index <= index < region_end:
            continue
        if best is None or handler_index > best:
            best = handler_index
    return best


def find_out_of_line_handler(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    cursor: int,
    end_index: int,
    active_handlers: frozenset[int],
) -> tuple[int, int] | None:
    regions = handler_regions(instructions, spans)
    owner = enclosing_handler(regions, cursor)
    best: tuple[int, int] | None = None
    for handler_index in regions:
        if handler_index in active_handlers:
            continue
        if enclosing_handler(regions, handler_index, handler_index) != owner:
            continue
        extent = protected_extent(instructions, spans, regions, handler_index, owner)
        if extent is None or extent[0] != cursor or extent[1] > end_index:
            continue
        candidate = (extent[1], handler_index)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return best[1], best[0]


def find_handler_of_empty_try(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    cursor: int,
    active_handlers: frozenset[int],
) -> tuple[int, int] | None:
    line = instructions[cursor].starts_line
    if instructions[cursor].opname != "NOP" or line is None:
        return None
    handler_index = first_unreferenced_handler(instructions, spans)
    if handler_index is None or handler_index in active_handlers:
        return None
    handler_line = instruction_line_after(instructions, handler_index)
    lines = [
        item.starts_line
        for item in instructions[:handler_index]
        if item.opname == "NOP" and item.starts_line is not None
    ]
    if handler_line is None or handler_line <= line or max(lines) != line:
        return None
    return handler_index, cursor + 1


def first_unreferenced_handler(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
) -> int | None:
    targets = {span.target_index for span in spans}
    for span in spans:
        if span.start_index in targets:
            continue
        if handler_cleanup_index(instructions, spans, span.start_index) is not None:
            return span.start_index
    return None


def instruction_line_after(instructions: list[Instruction], index: int) -> int | None:
    for instruction in instructions[index:]:
        if instruction.starts_line is not None:
            return instruction.starts_line
    return None


def nested_protected_extent(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    regions: dict[int, int],
    level: tuple[int, int | None],
    known: dict[int, tuple[int, int] | None],
) -> tuple[int, int] | None:
    handler_index, owner = level
    if handler_index not in known:
        known[handler_index] = protected_extent(
            instructions, spans, regions, handler_index, owner
        )
    return known[handler_index]


def protected_extent(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    regions: dict[int, int],
    handler_index: int,
    owner: int | None,
) -> tuple[int, int] | None:
    start_index: int | None = None
    end_index = 0
    nested_extents: dict[int, tuple[int, int] | None] = {}
    for span in spans:
        if span.target_index != handler_index:
            continue
        if instructions[span.start_index].opname == "CLEANUP_THROW":
            continue
        if is_comprehension_cleanup(instructions, spans, span.start_index):
            continue
        level = enclosing_handler(regions, span.start_index)
        if level == owner:
            extent: tuple[int, int] | None = (span.start_index, span.end_index)
        else:
            nested = nested_handler_at_level(regions, level, owner)
            if nested is None or nested == handler_index:
                return None
            extent = nested_protected_extent(
                instructions, spans, regions, (nested, owner), nested_extents
            )
        if extent is None:
            return None
        if start_index is None or extent[0] < start_index:
            start_index = extent[0]
        end_index = max(end_index, extent[1])
    if start_index is None:
        return None
    return start_index, end_index


def is_comprehension_cleanup(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    index: int,
) -> bool:
    return instructions[index].opname == "SWAP" and any(
        span.target_index == index for span in spans
    )


def nested_handler_at_level(
    regions: dict[int, int],
    handler_index: int | None,
    owner: int | None,
) -> int | None:
    while handler_index is not None:
        parent = enclosing_handler(regions, handler_index, handler_index)
        if parent == owner:
            return handler_index
        handler_index = parent
    return None


def read_out_of_line_clauses(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    spans: tuple[TableSpan, ...],
    handler_index: int,
    cleanup_index: int,
    after_index: int,
) -> tuple[OutOfLineClause, ...]:
    clauses: list[OutOfLineClause] = []
    cursor = handler_index + 1
    while cursor < cleanup_index:
        instruction = instructions[cursor]
        if instruction.opname == "RERAISE" and cursor + 1 == cleanup_index:
            return tuple(clauses)
        if instruction.opname == "POP_TOP":
            clause_end_index = cleanup_index
            type_start_index = None
            type_end_index = cursor
            body_start_index = cursor + 1
            name = None
        else:
            match = read_out_of_line_clause_match(
                instructions,
                offset_to_index,
                cursor,
                cleanup_index,
            )
            if match is None:
                return ()
            type_start_index = cursor
            type_end_index, body_start_index, clause_end_index = match
            name = exception_handler_name(
                instructions,
                body_start_index,
                clause_end_index,
            )
            if name is None and instructions[body_start_index].opname != "POP_TOP":
                return ()
            body_start_index += 1

        tail = read_out_of_line_clause_tail(
            instructions,
            spans,
            body_start_index,
            clause_end_index,
            name,
            after_index,
            handler_index,
        )
        body_end_index, exit_jump_index, resume_index = tail
        clauses.append(
            OutOfLineClause(
                type_start_index=type_start_index,
                type_end_index=type_end_index,
                name=name,
                body_start_index=body_start_index,
                body_end_index=body_end_index,
                exit_jump_index=exit_jump_index,
                resume_index=resume_index,
            )
        )
        if type_start_index is None:
            return tuple(clauses)
        cursor = clause_end_index
    return ()


def read_out_of_line_clause_match(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    type_start_index: int,
    cleanup_index: int,
) -> tuple[int, int, int] | None:
    match_index = type_start_index
    while match_index < cleanup_index:
        opname = instructions[match_index].opname
        if opname == "CHECK_EXC_MATCH":
            break
        if (
            is_jump_op(opname)
            or is_terminal_op(opname)
            or opname in NOT_EXCEPTION_TYPE_OPS
            or opname.startswith("STORE_")
        ):
            return None
        match_index += 1
    else:
        return None

    jump_index = match_index + 1
    jump = instructions[jump_index]
    if not is_exception_false_jump(jump) or not isinstance(jump.argval, int):
        return None
    miss_index = offset_to_index.get(jump.argval)
    if miss_index is None or miss_index <= jump_index or miss_index >= cleanup_index:
        return None
    return match_index, jump_index + 1, miss_index


def read_out_of_line_clause_tail(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    body_start_index: int,
    clause_end_index: int,
    name: str | None,
    after_index: int,
    handler_index: int,
) -> tuple[int, int | None, int | None]:
    end_index = out_of_line_clause_code_end(
        instructions,
        spans,
        body_start_index,
        clause_end_index,
        name,
    )

    exit_index = end_index
    while exit_index > body_start_index and (
        instructions[exit_index - 1].opname != "POP_EXCEPT"
    ):
        exit_index -= 1
    body_end_index = exit_index
    while body_end_index > body_start_index and (
        instructions[body_end_index - 1].opname == "POP_EXCEPT"
    ):
        body_end_index -= 1
    if exit_index == body_start_index or (
        instructions[body_end_index - 1].opname == "SWAP"
    ):
        return end_index, None, None
    if name is not None:
        unbind_index = find_named_exception_cleanup(
            instructions,
            exit_index,
            exit_index + 5,
            name,
        )
        if unbind_index != exit_index:
            return end_index, None, None
        exit_index += 3

    jump_index = skip_extended_args(instructions, exit_index, end_index - 1)
    if jump_index == end_index - 1 and (
        instructions[jump_index].opname in HANDLER_EXIT_JUMPS
    ):
        return body_end_index, jump_index, None
    resume_index = inlined_tail_resume_index(
        instructions,
        exit_index,
        end_index,
        after_index,
        handler_index,
    ) or break_copy_resume_index(
        instructions, (exit_index, end_index), after_index, handler_index
    )
    if resume_index is not None:
        return body_end_index, None, resume_index
    if (
        end_index - exit_index == 1
        and is_implicit_none_return(instructions, exit_index)
        and is_terminal_op(instructions[after_index - 1].opname)
    ):
        return body_end_index, None, None
    if is_statement_behind_clause(
        instructions, exit_index, end_index
    ) and not leaves_through_nested_handler(
        instructions, body_start_index, end_index, clause_end_index
    ):
        return body_end_index, None, exit_index
    return end_index, None, None


def break_copy_resume_index(
    instructions: list[Instruction],
    tail: tuple[int, int],
    after_index: int,
    handler_index: int,
) -> int | None:
    exit_index, end_index = tail
    if instructions[exit_index].opname != "POP_TOP":
        return None
    resume_index = inlined_tail_resume_index(
        instructions, exit_index + 1, end_index, after_index, handler_index
    )
    if resume_index is None or instructions[resume_index - 1].opname != "END_FOR":
        return None
    return resume_index


def leaves_through_nested_handler(
    instructions: list[Instruction],
    body_start_index: int,
    code_end_index: int,
    clause_end_index: int,
) -> bool:
    clause = range(
        int(instructions[body_start_index].offset),
        int(instructions[clause_end_index].offset),
    )
    return any(
        jump.opname in HANDLER_EXIT_JUMPS
        and previous.opname == "POP_EXCEPT"
        and jump.argval not in clause
        for previous, jump in pairwise(instructions[code_end_index:clause_end_index])
    )


def is_statement_behind_clause(
    instructions: list[Instruction],
    exit_index: int,
    end_index: int,
) -> bool:
    if end_index - exit_index < 2:
        return False
    if end_index - exit_index == 2 and instructions[exit_index].opname == "LOAD_CONST":
        return False
    if end_index - exit_index == 2 and is_none_load(instructions[exit_index]):
        return False
    if not is_terminal_op(instructions[end_index - 1].opname):
        return False
    if all(
        instruction.opname == "POP_TOP"
        for instruction in instructions[exit_index : end_index - 1]
    ):
        return False
    return not any(
        instruction.is_jump_target for instruction in instructions[exit_index:end_index]
    )


def out_of_line_clause_code_end(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    body_start_index: int,
    clause_end_index: int,
    name: str | None,
) -> int:
    regions = handler_regions(instructions, spans)
    end_index = clause_end_index
    while end_index > body_start_index:
        if (
            name is not None
            and instructions[end_index - 1].opname == "RERAISE"
            and find_named_exception_cleanup(
                instructions, end_index - 4, end_index, name
            )
            == end_index - 4
        ):
            end_index -= 4
            continue
        starts = [start for start, end in regions.items() if end == end_index]
        if not starts or min(starts) <= body_start_index:
            break
        end_index = min(starts)
    return end_index


def inlined_tail_resume_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    after_index: int,
    handler_index: int,
) -> int | None:
    length = end_index - start_index
    if length < 1 or not is_terminal_op(instructions[end_index - 1].opname):
        return None
    for resume_index in range(after_index, handler_index - length + 1):
        if not same_instruction_range_behavior(
            instructions,
            start_index,
            end_index,
            resume_index,
            resume_index + length,
        ):
            continue
        if not jumps_past(instructions, after_index, resume_index):
            return resume_index
    return None


def jumps_past(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    end_offset = instructions[end_index].offset
    return any(
        ("JUMP" in instruction.opname or instruction.opname == "FOR_ITER")
        and isinstance(instruction.argval, int)
        and instruction.argval > end_offset
        for instruction in instructions[start_index:end_index]
    )


def read_out_of_line_star_clauses(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_index: int,
    cleanup_index: int,
    after_index: int,
) -> tuple[OutOfLineClause, ...]:
    first = read_exception_group_handler(
        instructions,
        offset_to_index,
        handler_index,
        cleanup_index,
    )
    if first is None:
        return ()
    handlers = read_exception_group_handlers(
        instructions,
        offset_to_index,
        first,
        cleanup_index,
    )
    leave = read_out_of_line_star_exit(
        instructions,
        offset_to_index,
        handlers[-1].body_end_index,
        cleanup_index,
        after_index,
        handler_index,
    )
    if leave is None:
        return ()

    exit_jump_index, resume_index = leave
    return tuple(
        OutOfLineClause(
            type_start_index=handler.exception_type_index,
            type_end_index=handler.exception_type_end_index,
            name=handler.name,
            body_start_index=handler.body_start_index,
            body_end_index=star_clause_body_end(instructions, handler),
            exit_jump_index=exit_jump_index,
            resume_index=resume_index,
        )
        for handler in handlers
    )


def read_out_of_line_star_exit(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    cleanup_index: int,
    after_index: int,
    handler_index: int,
) -> tuple[int | None, int | None] | None:
    reraise_index = start_index
    while instructions[reraise_index].opname not in REST_OF_GROUP_OPS:
        reraise_index += 1
        if reraise_index >= cleanup_index:
            return None

    jump = instructions[reraise_index + 2]
    if not is_exception_group_not_none_jump(jump):
        return None
    exit_index = reraise_index + 5
    end_index = offset_to_index.get(jump.argval)
    if end_index is None or end_index <= exit_index:
        return None
    if instructions[exit_index - 1].opname != "POP_EXCEPT":
        return None

    if end_index == exit_index + 1 and (
        instructions[exit_index].opname in HANDLER_EXIT_JUMPS
    ):
        return exit_index, None
    resume_index = inlined_tail_resume_index(
        instructions,
        exit_index,
        end_index,
        after_index,
        handler_index,
    )
    return None, resume_index


def star_clause_body_end(
    instructions: list[Instruction],
    handler: SimpleHandler,
) -> int:
    if handler.miss_index is None:
        return handler.body_end_index
    end_index = handler.miss_index - 1
    while end_index > handler.body_start_index and not (
        instructions[end_index].opname == "LIST_APPEND"
        and instructions[end_index + 1].opname == "POP_TOP"
    ):
        end_index -= 1
    if handler.name is not None:
        end_index -= 3
    if instructions[end_index - 1].opname not in HANDLER_EXIT_JUMPS:
        return max(end_index, handler.body_start_index)

    end_index -= 1
    if instructions[end_index - 1].opname == "LIST_APPEND":
        end_index -= 1
    if handler.name is not None:
        end_index -= 3
    return max(end_index, handler.body_start_index)


def out_of_line_else_end(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    clauses: tuple[OutOfLineClause, ...],
    handler_index: int,
    body_end_index: int,
    end_index: int,
) -> int | None:
    else_end_index = body_end_index
    for clause in clauses:
        if handler_loop_exit(decompiler, instructions, clause) is not None:
            continue
        target_index = clause.resume_index
        if clause.exit_jump_index is not None:
            target_index = offset_to_index.get(
                instructions[clause.exit_jump_index].argval
            )
            if target_index is None:
                return None
        if target_index is None:
            continue
        if target_index > handler_index:
            forward_end = forward_resume_else_end(
                decompiler,
                instructions,
                clause,
                target_index,
                (body_end_index, handler_index, end_index),
            )
            if forward_end is None:
                return None
            else_end_index = forward_end or else_end_index
            continue
        if body_end_index < target_index <= end_index or skips_rest_of_branch(
            instructions, target_index, end_index
        ):
            target_index = min(target_index, end_index)
            if else_end_index not in {body_end_index, target_index}:
                return None
            else_end_index = target_index
    return else_end_index


def skips_rest_of_branch(
    instructions: list[Instruction], target_index: int, end_index: int
) -> bool:
    if not end_index < target_index or end_index >= len(instructions):
        return False
    jump = instructions[end_index]
    return (
        jump.opname == "JUMP_FORWARD"
        and jump.argval == instructions[target_index].offset
    )


def else_end_before_handler(
    instructions: list[Instruction],
    clauses: tuple[OutOfLineClause, ...],
    handler_index: int,
    else_range: tuple[int, int],
    end_index: int,
) -> int:
    body_end_index, else_end_index = else_range
    if else_end_index != body_end_index or not body_end_index < handler_index:
        return else_end_index
    if handler_index <= end_index and any(
        leaves_enclosing_handler(instructions, clause) for clause in clauses
    ):
        return handler_index
    if any(clause.resume_index == body_end_index for clause in clauses):
        return else_end_index
    if is_terminal_op(instructions[handler_index - 1].opname) and any(
        instruction.opname == "PUSH_EXC_INFO"
        for instruction in instructions[:body_end_index]
    ):
        return min(handler_index, end_index)
    return else_end_index


def else_end_from_name_numbers(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    clauses: tuple[OutOfLineClause, ...],
    else_range: tuple[int, int],
    limits: tuple[int, int, int],
) -> int:
    body_end_index, else_end_index = else_range
    handler_index, handler_end_index, end_index = limits
    range_end_index = min(handler_index, end_index)
    if else_end_index != body_end_index or body_end_index >= range_end_index:
        return else_end_index
    if not all(
        handler_loop_exit(decompiler, instructions, clause) is not None
        or (clause.exit_jump_index is None and clause.resume_index is None)
        for clause in clauses
    ):
        return else_end_index
    regions = handler_regions(instructions, spans)
    earlier = {
        index
        for start, end in regions.items()
        if body_end_index <= start < handler_index
        for index in range(start, end)
    }
    names = tuple(getattr(decompiler.code, "co_names", ()))
    known = name_numbers(names, instructions, {*range(body_end_index), *earlier})
    behind = name_numbers(
        names, instructions, set(range(body_end_index, range_end_index)) - earlier
    )
    handled = name_numbers(
        names, instructions, set(range(handler_index, handler_end_index))
    )
    behind, handled = behind - handled - known, handled - behind - known
    if behind and handled and max(behind) < min(handled):
        return range_end_index
    return else_end_index


def name_numbers(
    names: tuple[str, ...],
    instructions: list[Instruction],
    indexes: set[int],
) -> set[int]:
    return {
        names.index(instructions[index].argval)
        for index in indexes
        if instructions[index].opname in NAME_OPS
        and instructions[index].argval in names
    }


def leaves_enclosing_handler(
    instructions: list[Instruction],
    clause: OutOfLineClause,
) -> bool:
    jump_index = clause.exit_jump_index or clause.body_end_index - 1
    if "JUMP" not in instructions[jump_index].opname:
        return False
    if instructions[jump_index - 1].opname != "POP_EXCEPT":
        return False
    inner_exit = jump_index - (5 if clause.name is not None else 2)
    return instructions[inner_exit].opname == "POP_EXCEPT"


def follows_a_handler(instructions: list[Instruction], index: int) -> bool:
    return any(
        instruction.opname == "PUSH_EXC_INFO" for instruction in instructions[:index]
    )


def stands_in_line(decompiler: DecompilerContext) -> bool:
    # before 3.12 every handler follows its try body, with a jump over it
    return decompiler.version is not None and decompiler.version < (3, 12)


def resumes_at_finally_copy(
    instructions: list[Instruction], final_index: int, target_index: int
) -> bool:
    while instructions[target_index].opname == "NOP":
        target_index += 1
    for step in range(len(instructions) - target_index):
        if instructions[final_index + step].opname in {"RETURN_VALUE", "SWAP"}:
            return step > 0
        if not same_instruction_behavior(
            instructions[final_index + step], instructions[target_index + step]
        ):
            return False
    return False


def forward_resume_else_end(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    clause: OutOfLineClause,
    target_index: int,
    bounds: tuple[int, int, int],
) -> int | None:
    body_end_index, handler_index, end_index = bounds
    if clause.exit_jump_index is None:
        return handler_index if handler_index <= end_index else 0
    resume_offset = instructions[target_index].offset
    if resume_offset not in decompiler.detached_resume_offsets:
        before_handler = instructions[handler_index - 1].opname
        if handler_index <= end_index and (
            is_terminal_op(before_handler) or before_handler == "JUMP_BACKWARD"
        ):
            return handler_index
        if end_index < handler_index and resumes_at_finally_copy(
            instructions, end_index, target_index
        ):
            return 0
        jump_index = handler_index - 1
        if (
            jump_index == body_end_index
            and not stands_in_line(decompiler)
            and not (
                target_index > end_index
                and follows_a_handler(instructions, body_end_index)
            )
        ):
            return None
        if body_end_index <= jump_index < end_index and (
            instructions[jump_index].opname == "JUMP_FORWARD"
            and instructions[jump_index].argval == resume_offset
        ):
            return jump_index
        return None
    return end_index
