import ast
from collections.abc import Callable
from dataclasses import dataclass

from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import ExceptionTableEntry, parse_exception_table
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
    returns_protected_value: bool = False

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
    decompiler,
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
    decompiler,
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
    decompiler,
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

    body = exception_table_finally_body(decompiler, instructions, pattern)
    final_body = decompiler.translate_child_statements(
        instructions,
        pattern.final_start_index,
        pattern.final_end_index,
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

def wrap_suppressed_finally_cleanup(
    decompiler,
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
        if (
            start_index is None
            or end_index is None
            or handler_start_index is None
        ):
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
        cursor = read_exception_opcode_index(
            instructions,
            cursor,
            end_index,
            predicate,
        )
        if cursor is None:
            return False
        cursor += 1
    return True

def is_exception_load(instruction: Instruction) -> bool:
    return instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"} and (
        instruction.argval == "Exception"
    )

def try_translate_exception_table_except_finally(
    decompiler,
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
    decompiler,
    instructions: list[Instruction],
    pattern: ExceptionTableFinallyPattern,
) -> list[ast.stmt]:
    if pattern.returns_protected_value:
        value = decompiler.evaluate_expression_range(
            instructions,
            pattern.body_start_index,
            pattern.body_end_index,
        )
        return [ast.Return(value=value)]

    return decompiler.translate_child_statements(
        instructions,
        pattern.body_start_index,
        pattern.body_end_index,
    )

def try_translate_exception_table_except_star(
    decompiler,
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
    decompiler.statements.append(
        ast.TryStar(
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
    decompiler,
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
        cleanup_after_index
        if returns_protected_value
        else final_range.final_end_index
    )

    return ExceptionTableFinallyPattern(
        body_start_index=region.body_start_index,
        body_end_index=region.body_end_index,
        final_start_index=final_range.final_start_index,
        final_end_index=final_range.final_end_index,
        after_index=after_index,
        returns_protected_value=returns_protected_value,
    )

def read_exception_table_finally_body_range(
    instructions: list[Instruction],
    region: ExceptionTableRegion,
    end_index: int,
) -> FinallyBodyRange | None:
    if not is_valid_finally_region_start(instructions, region, end_index):
        return None

    handler_final_start_index = region.handler_start_index + 1
    handler_final_end_index = find_reraise_zero(
        instructions,
        handler_final_start_index,
        end_index,
    )
    if handler_final_end_index is None:
        return None
    if find_check_exception_match(
        instructions,
        handler_final_start_index,
        handler_final_end_index,
    ) is not None:
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

def is_valid_finally_region_start(
    instructions: list[Instruction],
    region: ExceptionTableRegion,
    end_index: int,
) -> bool:
    if has_jump_instruction(
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

def has_jump_instruction(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    for index in range(start_index, end_index):
        if is_jump_op(instructions[index].opname):
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
    return (
        common_trailing_opcode_name(left.opname)
        == common_trailing_opcode_name(right.opname)
        and left.argrepr == right.argrepr
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

def exception_handler_name(
    instructions: list[Instruction],
    body_start_index: int,
    miss_index: int,
) -> str | None:
    if body_start_index >= miss_index:
        return None
    instruction = instructions[body_start_index]
    if instruction.opname not in {"STORE_FAST", "STORE_NAME"}:
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
        if instruction.opname == "CALL_INTRINSIC_2":
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
    if (
        miss_jump_index < end_index
        and instructions[miss_jump_index].opname == "COPY"
    ):
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
    if cursor < end_index and instructions[cursor].opname == "BUILD_LIST":
        cursor = skip_exception_match_prefix(instructions, cursor + 1, end_index)
    if cursor < end_index and instructions[cursor].opname == "COPY":
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
        if store.opname not in {"STORE_FAST", "STORE_NAME"}:
            continue
        if delete.opname not in {"DELETE_FAST", "DELETE_NAME"}:
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
        cleaned.append(body[cursor])
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
    decompiler,
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
    while (
        cursor < end_index
        and instructions[cursor].opname in {"POP_EXCEPT", "POP_TOP"}
    ):
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
