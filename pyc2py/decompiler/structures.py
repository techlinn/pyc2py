import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

from pyc2py.astree import invert_compare_expression, make_name, safe_identifier
from pyc2py.bytecode.instruction import Instruction, is_none_load, is_none_return
from pyc2py.bytecode.metadata import (
    backward_jump_indexes,
    line_in_effect,
    make_offset_index,
)
from pyc2py.bytecode.opcode_table import (
    FORWARD_JUMPS,
    normalized_opcode_name,
)
from pyc2py.bytecode.stack_effect import instruction_stack_effect
from pyc2py.decompiler.exception_structures import (
    TableSpan,
    enclosing_handler,
    handler_regions,
    protected_extent,
    read_table_spans,
    same_instruction_behavior,
    same_instruction_range_behavior,
)
from pyc2py.decompiler.match_patterns import SimpleMatchTest, read_nested_match
from pyc2py.decompiler.opcodes.flow import dropped_under_value_end, is_terminal_op


def is_forward_conditional_jump(instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    if "JUMP" not in opname:
        return False
    if not is_conditional_jump_name(opname):
        return False

    return (
        isinstance(instruction.argval, int) and instruction.argval > instruction.offset
    )


def is_false_jump(opname: str) -> bool:
    opname = normalized_opcode_name(opname)
    return "IF_FALSE" in opname or "IF_NOT_NONE" in opname


def is_none_jump(opname: str) -> bool:
    opname = normalized_opcode_name(opname)
    return "IF_NONE" in opname or "IF_NOT_NONE" in opname


def is_conditional_jump_name(opname: str) -> bool:
    opname = normalized_opcode_name(opname)
    return (
        "IF_FALSE" in opname
        or "IF_TRUE" in opname
        or "IF_NONE" in opname
        or "IF_NOT_NONE" in opname
    )


def invert_condition(condition: ast.expr) -> ast.expr:
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        return condition.operand

    return invert_compare_expression(condition) or ast.UnaryOp(
        op=ast.Not(), operand=condition
    )


def condition_from_jump(opname: str, value: ast.expr) -> ast.expr:
    value = unwrap_truth_test(value)
    if not is_none_jump(opname) or getattr(value, "_pyc2py_none_jump_test", False):
        return value

    last_link_op = getattr(value, "_pyc2py_chain", None)
    if isinstance(value, ast.Compare) and last_link_op is not None:
        chain = ast.Compare(
            left=value.left,
            ops=[*value.ops, ast.IsNot() if last_link_op == "is not" else ast.Is()],
            comparators=[*value.comparators, ast.Constant(value=None)],
        )

        if last_link_op == "is not":
            return ast.UnaryOp(op=ast.Not(), operand=chain)

        return chain

    test = ast.Compare(
        left=value,
        ops=[ast.Is()],
        comparators=[ast.Constant(value=None)],
    )
    setattr(test, "_pyc2py_none_jump_test", True)
    return test


def unwrap_truth_test(value: ast.expr) -> ast.expr:
    if not bool(getattr(value, "_pyc2py_truth_test", False)):
        return value
    if not isinstance(value, ast.Call):
        return value
    if not isinstance(value.func, ast.Name) or value.func.id != "bool":
        return value
    if len(value.args) != 1 or value.keywords:
        return value

    return value.args[0]


@dataclass(frozen=True, slots=True)
class ForLoopPattern:
    for_iter_index: int
    body_end_index: int
    after_index: int


LOOP_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP"}


def find_for_loop_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_iter_index: int,
    end_index: int,
) -> ForLoopPattern | None:
    for_iter_index = find_for_iter_index(instructions, get_iter_index + 1, end_index)
    if for_iter_index is None:
        return None

    for_iter = instructions[for_iter_index]
    exit_index = offset_to_index.get(int(for_iter.argval))
    if exit_index is None or exit_index <= for_iter_index:
        return None

    search_end_index = min(exit_index, end_index)
    back_jump_index = find_loop_back_jump(
        instructions,
        loop_entry_offsets(instructions, get_iter_index, for_iter_index),
        for_iter_index + 1,
        search_end_index,
    )
    if back_jump_index is None:
        # no jump back after a body that only breaks where dead code is dropped
        return ForLoopPattern(
            for_iter_index=for_iter_index,
            body_end_index=exit_index,
            after_index=skip_loop_cleanup(instructions, exit_index, end_index),
        )

    return ForLoopPattern(
        for_iter_index=for_iter_index,
        body_end_index=for_loop_body_end_index(
            instructions,
            loop_entry_offsets(instructions, get_iter_index, for_iter_index),
            back_jump_index,
            search_end_index,
        ),
        after_index=skip_loop_cleanup(instructions, exit_index, end_index),
    )


def find_for_iter_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = start_index
    while cursor < end_index:
        instruction = instructions[cursor]
        if instruction.opname == "FOR_ITER":
            return cursor
        if instruction.opname not in LOOP_PREFIX_OPS:
            return None

        cursor += 1

    return None


def loop_entry_offsets(
    instructions: list[Instruction],
    get_iter_index: int,
    for_iter_index: int,
) -> set[int]:
    return {
        instruction.offset
        for instruction in instructions[get_iter_index + 1 : for_iter_index + 1]
    }


def find_loop_back_jump(
    instructions: list[Instruction],
    loop_offsets: set[int],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(end_index - 1, start_index - 1, -1):
        instruction = instructions[index]
        if instruction.opname not in {"JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP"}:
            continue
        if instruction.argval in loop_offsets:
            return index

    return None


def handler_join_index(
    instructions: list[Instruction],
    start_index: int,
    limit_index: int,
) -> int:
    joins = [
        int(instruction.argval)
        for previous, instruction in pairwise(
            instructions[start_index:limit_index],
        )
        if previous.opname == "POP_EXCEPT" and instruction.opname == "JUMP_FORWARD"
    ]

    for index in range(start_index, limit_index):
        if instructions[index].offset in joins:
            return index

    return limit_index


def find_detached_for_tail(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    for_iter_index: int,
) -> tuple[int, int] | None:
    for_iter = instructions[for_iter_index]
    exit_index = offset_to_index.get(int(for_iter.argval))
    if exit_index is None:
        return None

    back_jump_index = max(
        backward_jump_indexes(instructions, (for_iter.offset,), exit_index + 1),
        default=None,
    )
    if back_jump_index is None:
        return None

    start_index = handler_join_index(instructions, exit_index, back_jump_index)
    if start_index >= back_jump_index:
        return None

    return start_index, back_jump_index


def for_loop_body_end_index(
    instructions: list[Instruction],
    loop_offsets: set[int],
    back_jump_index: int,
    exit_index: int,
) -> int:
    tail_index = exit_index - 1
    if tail_index <= back_jump_index:
        return first_back_jump_copy(instructions, back_jump_index)

    tail = instructions[tail_index]
    if (
        tail.opname in {"JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP"}
        and tail.argval in loop_offsets
    ):
        return first_back_jump_copy(instructions, tail_index)

    return exit_index


def first_back_jump_copy(instructions: list[Instruction], back_jump_index: int) -> int:
    back_jump = instructions[back_jump_index]

    while back_jump_index > 0:
        previous = instructions[back_jump_index - 1]
        if previous.opname != back_jump.opname or previous.argval != back_jump.argval:
            break

        back_jump_index -= 1

    return back_jump_index


def skip_loop_cleanup(
    instructions: list[Instruction],
    exit_index: int,
    end_index: int,
) -> int:
    if exit_index >= end_index:
        return exit_index

    if instructions[exit_index].opname == "END_FOR":
        after_index = exit_index + 1
        if after_index < end_index and instructions[after_index].opname == "POP_TOP":
            return after_index + 1

        return after_index

    if instructions[exit_index].opname == "POP_BLOCK":
        return exit_index + 1

    return exit_index


def make_async_with_statement(
    context_expr: ast.expr,
    body: list[ast.stmt],
    optional_vars: ast.expr | None = None,
) -> ast.AsyncWith:
    return ast.AsyncWith(
        items=[ast.withitem(context_expr=context_expr, optional_vars=optional_vars)],
        body=body or [ast.Pass()],
        type_comment=None,
    )


def is_yield_expression(value: ast.expr) -> bool:
    return isinstance(value, (ast.Yield, ast.YieldFrom, ast.Await))


@dataclass(frozen=True, slots=True)
class MatchCaseSpec:
    pattern: ast.pattern
    body: tuple[ast.stmt, ...]
    guard: ast.expr | None = None


@dataclass(frozen=True, slots=True)
class SimpleMatchCaseRegion:
    pattern: ast.pattern
    body_start_index: int
    body_end_index: int
    copied_subject: bool = False
    guard_start_index: int | None = None
    guard_end_index: int | None = None
    exit_index: int | None = None


@dataclass(frozen=True, slots=True)
class MatchJumpTarget:
    jump_index: int
    miss_index: int


@dataclass(frozen=True, slots=True)
class SequenceMatchHeader:
    match_jump_index: int
    miss_index: int
    get_len_index: int
    load_count_index: int
    compare_index: int
    length_jump_index: int
    count_value: int


@dataclass(frozen=True, slots=True)
class SimpleMatchRegion:
    cases: tuple[SimpleMatchCaseRegion, ...]
    after_index: int


MATCH_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP"}


def make_match_case(spec: MatchCaseSpec) -> ast.match_case:
    return ast.match_case(
        pattern=spec.pattern,
        guard=spec.guard,
        body=list(spec.body) or [ast.Pass()],
    )


def make_match(subject: ast.expr, cases: list[MatchCaseSpec]) -> ast.Match:
    return ast.Match(
        subject=subject,
        cases=[make_match_case(case) for case in cases],
    )


def find_simple_match_region(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    end_index: int,
) -> SimpleMatchRegion | None:
    cases: list[SimpleMatchCaseRegion] = []
    cursor = start_index
    last_test_index = start_index

    while cursor < end_index:
        miss_cursor = cursor
        case_cursor = read_match_case_cursor(instructions, cursor, end_index)
        if case_cursor is None:
            return finish_simple_match_region(
                instructions, cases, miss_cursor, end_index, last_test_index
            )

        cursor, copied_subject = case_cursor
        in_match = bool(copied_subject or cases)
        parsed = read_nested_match(
            instructions, offset_to_index, cursor, end_index, in_match
        )
        if parsed is None and in_match:
            # the last case in a loop jumps to its body when the value matches
            parsed = read_dotted_value_match_test(
                instructions, offset_to_index, cursor, end_index
            )
        if parsed is None and cases:
            parsed = read_guarded_capture_match_test(
                instructions, offset_to_index, cursor, end_index
            )
        if parsed is None:
            return finish_simple_match_region(
                instructions, cases, miss_cursor, end_index, last_test_index
            )

        last_test_index = cursor
        region = make_simple_match_case_region(
            instructions,
            offset_to_index,
            parsed,
            copied_subject,
        )
        if region is None:
            return None

        cases.append(region)

        if parsed.miss_index <= cursor:
            return None

        cursor = parsed.miss_index

    if cursor == end_index and any(case.copied_subject for case in cases):
        return SimpleMatchRegion(
            cases=tuple(collapse_simple_or_cases(instructions, cases)),
            after_index=end_index,
        )

    return None


def read_match_case_cursor(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> tuple[int, bool] | None:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor >= end_index:
        return None

    if instructions[cursor].opname == "POP_TOP":
        cursor = skip_match_prefix(instructions, cursor + 1, end_index)
        if cursor >= end_index:
            return None

    instruction = instructions[cursor]
    if instruction.opname == "COPY" and int(instruction.arg or 0) == 1:
        return cursor + 1, True

    return cursor, False


def read_simple_match_default_region(
    instructions: list[Instruction],
    cases: list[SimpleMatchCaseRegion],
    cursor: int,
    end_index: int,
) -> SimpleMatchRegion | None:
    if not cases:
        return None

    default_region = read_default_match_case(instructions, cursor, end_index)
    if default_region is None:
        return None

    cases.append(default_region)
    return SimpleMatchRegion(
        cases=tuple(collapse_simple_or_cases(instructions, cases)),
        after_index=default_region.body_end_index,
    )


def finish_simple_match_region(
    instructions: list[Instruction],
    cases: list[SimpleMatchCaseRegion],
    cursor: int,
    end_index: int,
    last_test_index: int,
) -> SimpleMatchRegion | None:
    if not cases:
        return None

    exits = {case.exit_index for case in cases if case.exit_index is not None}
    if len(exits) > 1:
        return None

    default_start_index = skip_match_success_cleanup(instructions, cursor, end_index)
    if default_start_index >= end_index and not exits:
        # the last miss only drops what its test left, at the end of the region
        return SimpleMatchRegion(
            cases=tuple(collapse_simple_or_cases(instructions, cases)),
            after_index=end_index,
        )

    if exits:
        after_index = exits.pop()
        if after_index < default_start_index or after_index > end_index:
            return None
        if after_index > default_start_index:
            cases.append(
                SimpleMatchCaseRegion(
                    pattern=ast.MatchAs(pattern=None, name=None),
                    body_start_index=default_start_index,
                    body_end_index=after_index,
                )
            )

        return SimpleMatchRegion(
            cases=tuple(collapse_simple_or_cases(instructions, cases)),
            after_index=after_index,
        )

    tail_end_index = repeated_match_tail_end(
        instructions, cases, cursor, default_start_index, end_index, last_test_index
    )
    captures_subject = (
        default_start_index < end_index
        and instructions[default_start_index].opname in MATCH_ALIAS_STORE_OPS
    )
    if tail_end_index is None or captures_subject:
        return read_simple_match_default_region(instructions, cases, cursor, end_index)

    cases = [
        without_repeated_match_tail(
            instructions, case, default_start_index, tail_end_index
        )
        for case in cases
    ]
    return SimpleMatchRegion(
        cases=tuple(collapse_simple_or_cases(instructions, cases)),
        after_index=default_start_index,
    )


def repeated_match_tail_end(
    instructions: list[Instruction],
    cases: list[SimpleMatchCaseRegion],
    cursor: int,
    tail_start_index: int,
    end_index: int,
    last_test_index: int,
) -> int | None:
    for index in range(cursor, tail_start_index + 1):
        if index < end_index and instructions[index].opname == "NOP":
            return None

    tail_end_index = find_terminal_body_end(instructions, tail_start_index, end_index)
    if tail_end_index is None:
        return None

    tail = instructions[tail_start_index]
    if is_none_return(tail) and tail.starts_line in {
        None,
        line_in_effect(instructions, last_test_index),
    }:
        return tail_end_index

    length = tail_end_index - tail_start_index

    for case in cases:
        if case.body_end_index - length >= case.body_start_index and (
            same_instruction_range_behavior(
                instructions,
                case.body_end_index - length,
                case.body_end_index,
                tail_start_index,
                tail_end_index,
            )
        ):
            continue
        if not is_terminal_op(instructions[case.body_end_index - 1].opname):
            return None

    return tail_end_index


def without_repeated_match_tail(
    instructions: list[Instruction],
    case: SimpleMatchCaseRegion,
    tail_start_index: int,
    tail_end_index: int,
) -> SimpleMatchCaseRegion:
    body_end_index = case.body_end_index - (tail_end_index - tail_start_index)
    if body_end_index < case.body_start_index:
        return case
    if not same_instruction_range_behavior(
        instructions,
        body_end_index,
        case.body_end_index,
        tail_start_index,
        tail_end_index,
    ):
        return case

    return replace(case, body_end_index=body_end_index)


def make_simple_match_case_region(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    parsed: SimpleMatchTest,
    copied_subject: bool,
) -> SimpleMatchCaseRegion | None:
    pattern = parsed.pattern
    body_start_index = (
        parsed.jump_index + 1
        if parsed.body_start_index is None
        else parsed.body_start_index
    )
    body_end_index = (
        parsed.miss_index if parsed.body_end_index is None else parsed.body_end_index
    )
    test_end_index = body_start_index

    if copied_subject:
        body_start_index = skip_match_subject_cleanup(
            instructions, body_start_index, body_end_index
        )

    alias = read_match_alias(instructions, body_start_index, body_end_index)
    if alias is not None:
        pattern = ast.MatchAs(pattern=pattern, name=alias.name)
        body_start_index = alias.next_index

    guard_start_index = None
    guard_end_index = None
    guard_jump_index = find_match_guard_jump(
        instructions,
        offset_to_index,
        body_start_index,
        body_end_index,
    )
    if guard_jump_index is not None:
        guard_start_index = body_start_index
        guard_end_index = guard_jump_index
        body_start_index = guard_jump_index + 1

    exit_index = None

    if not leaves_to_enclosing_loop(
        instructions, parsed, body_start_index, body_end_index
    ):
        settled = settle_case_body_end(
            instructions,
            offset_to_index,
            (body_start_index, body_end_index),
            copied_subject and body_start_index == test_end_index,
        )
        if settled is None:
            return None

        body_end_index, exit_index = settled

    return SimpleMatchCaseRegion(
        pattern=pattern,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        copied_subject=copied_subject,
        guard_start_index=guard_start_index,
        guard_end_index=guard_end_index,
        exit_index=exit_index,
    )


def settle_case_body_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    body_range: tuple[int, int],
    is_bare_alternative: bool,
) -> tuple[int, int | None] | None:
    body_start_index, body_end_index = body_range
    terminal_body_end = closing_terminal_end(
        instructions, body_start_index, body_end_index
    )
    if terminal_body_end is None:
        if is_bare_alternative:
            return None

        exit_index: int | None = body_end_index
        last = instructions[body_end_index - 1]
        if last.opname == "JUMP_FORWARD":
            exit_index = offset_to_index.get(int(last.argval))
            body_end_index -= 1

        if exit_index is None or body_end_index < body_start_index:
            return None

        return body_end_index, exit_index

    if terminal_body_end != body_end_index and not is_match_cleanup_range(
        instructions, terminal_body_end, body_end_index
    ):
        return None

    return terminal_body_end, None


def closing_terminal_end(
    instructions: list[Instruction],
    body_start_index: int,
    body_end_index: int,
) -> int | None:
    index = body_end_index
    while index > body_start_index and (
        instructions[index - 1].opname in MATCH_PREFIX_OPS
        or instructions[index - 1].opname == "POP_TOP"
    ):
        index -= 1

    if index > body_start_index and is_match_terminal(instructions[index - 1]):
        return index

    return None


def leaves_to_enclosing_loop(
    instructions: list[Instruction],
    parsed: SimpleMatchTest,
    body_start_index: int,
    body_end_index: int,
) -> bool:
    if instructions[parsed.jump_index].opname == "JUMP_BACKWARD":
        return True
    if body_end_index <= body_start_index:
        return False

    last = instructions[body_end_index - 1]
    return (
        last.opname == "JUMP_BACKWARD"
        and last.argval < instructions[body_start_index].offset
    )


@dataclass(frozen=True, slots=True)
class MatchAlias:
    name: str
    next_index: int


MATCH_ALIAS_STORE_OPS = {
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_GLOBAL",
    "STORE_NAME",
}


def read_match_alias(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> MatchAlias | None:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor >= end_index:
        return None

    instruction = instructions[cursor]
    if instruction.opname not in MATCH_ALIAS_STORE_OPS:
        return None

    name = safe_identifier(str(instruction.argval))
    if name == "_":
        return None

    return MatchAlias(name=name, next_index=cursor + 1)


def find_match_guard_jump(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    body_start_index: int,
    body_end_index: int,
) -> int | None:
    for index in range(body_start_index, body_end_index):
        instruction = instructions[index]
        if not is_forward_conditional_jump(instruction):
            continue

        target_index = offset_to_index.get(int(instruction.argval))
        if target_index is None or target_index < body_end_index:
            continue
        if index == body_start_index:
            return None

        return index

    return None


def collapse_simple_or_cases(
    instructions: list[Instruction],
    cases: list[SimpleMatchCaseRegion],
) -> list[SimpleMatchCaseRegion]:
    collapsed: list[SimpleMatchCaseRegion] = []
    cursor = 0
    while cursor < len(cases):
        case = cases[cursor]
        alternatives = [case.pattern]
        next_cursor = cursor + 1
        while (
            next_cursor < len(cases)
            and cases[next_cursor].copied_subject
            and case.copied_subject
            and same_match_body(instructions, case, cases[next_cursor])
        ):
            alternatives.append(cases[next_cursor].pattern)
            next_cursor += 1

        if len(alternatives) == 1:
            collapsed.append(case)
        else:
            collapsed.append(
                SimpleMatchCaseRegion(
                    pattern=or_pattern(alternatives),
                    body_start_index=case.body_start_index,
                    body_end_index=case.body_end_index,
                    copied_subject=case.copied_subject,
                    guard_start_index=case.guard_start_index,
                    guard_end_index=case.guard_end_index,
                )
            )

        cursor = next_cursor

    return collapsed


def or_pattern(alternatives: list[ast.pattern]) -> ast.pattern:
    # `A | B as x` is compiled as `A as x` and `B as x`, each with its body
    names = {
        item.name if isinstance(item, ast.MatchAs) and item.pattern else None
        for item in alternatives
    }
    if len(names) != 1 or None in names:
        return ast.MatchOr(patterns=alternatives)

    inner = [item.pattern for item in alternatives if isinstance(item, ast.MatchAs)]
    return ast.MatchAs(
        pattern=ast.MatchOr(patterns=[item for item in inner if item is not None]),
        name=names.pop(),
    )


def same_match_body(
    instructions: list[Instruction],
    left: SimpleMatchCaseRegion,
    right: SimpleMatchCaseRegion,
) -> bool:
    left_length = left.body_end_index - left.body_start_index
    right_length = right.body_end_index - right.body_start_index
    if left_length != right_length:
        return False

    for offset in range(left_length):
        left_instruction = instructions[left.body_start_index + offset]
        right_instruction = instructions[right.body_start_index + offset]
        if not same_instruction_behavior(left_instruction, right_instruction):
            return False

    return True


def read_guarded_capture_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    alias = read_match_alias(instructions, cursor, end_index)
    if alias is not None and alias.next_index == cursor + 1:
        name, guard_start = alias.name, cursor + 1
    elif is_wildcard_guard_start(instructions, cursor):
        name, guard_start = None, cursor
    else:
        return None

    miss_index = guard_miss_index(instructions, offset_to_index, guard_start, end_index)
    if miss_index is None:
        return None

    return SimpleMatchTest(
        pattern=ast.MatchAs(pattern=None, name=name),
        jump_index=guard_start - 1,
        miss_index=miss_index,
    )


def is_wildcard_guard_start(instructions: list[Instruction], cursor: int) -> bool:
    # `case _ if test:` tests on the case's line; a body statement has its own
    return (
        cursor > 0
        and instructions[cursor - 1].opname == "POP_TOP"
        and instructions[cursor].starts_line is None
        and line_in_effect(instructions, cursor - 1) is not None
    )


def guard_miss_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    end_index: int,
) -> int | None:
    # an `or` guard jumps into the body first; its miss is the furthest false way
    body_start: int | None = None
    misses: list[int] = []

    for index in range(start_index, end_index):
        if index == body_start:
            break
        if "JUMP" not in instructions[index].opname:
            continue
        if not is_forward_conditional_jump(instructions[index]):
            return None

        target = offset_to_index.get(int(instructions[index].argval))
        if target is None:
            return None

        if is_false_jump(instructions[index].opname):
            misses.append(target)

            if body_start is None:
                break
        elif body_start is None:
            body_start = target

    return max(misses, default=None)


DOTTED_VALUE_START_OPS = frozenset(
    {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_DEREF"}
)


def read_dotted_value_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    if instructions[cursor].opname not in DOTTED_VALUE_START_OPS:
        return None

    value: ast.expr = make_name(str(instructions[cursor].argval))
    index = cursor + 1
    while index < end_index and instructions[index].opname == "LOAD_ATTR":
        value = ast.Attribute(
            value=value, attr=str(instructions[index].argval), ctx=ast.Load()
        )
        index += 1

    if index == cursor + 1 or index >= end_index:
        return None

    compare = instructions[index]
    if compare.opname != "COMPARE_OP" or not compare.argrepr.endswith("=="):
        return None

    jump_target = read_match_jump_target(
        instructions,
        offset_to_index,
        skip_match_prefix(instructions, index + 1, end_index),
        end_index,
    )
    if jump_target is None:
        return None

    return SimpleMatchTest(
        pattern=ast.MatchValue(value=value),
        jump_index=jump_target.jump_index,
        miss_index=jump_target.miss_index,
    )


def read_match_jump_target(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    jump_index: int,
    end_index: int,
) -> MatchJumpTarget | None:
    if jump_index >= end_index:
        return None

    jump = instructions[jump_index]

    if is_miss_to_loop_continue(instructions, jump_index, end_index):
        return MatchJumpTarget(jump_index=jump_index + 1, miss_index=end_index)
    if not is_match_miss_jump(jump):
        return None

    miss_index = offset_to_index.get(int(jump.argval))
    if miss_index is None:
        return None

    return MatchJumpTarget(jump_index=jump_index, miss_index=miss_index)


def is_miss_to_loop_continue(
    instructions: list[Instruction],
    jump_index: int,
    end_index: int,
) -> bool:
    if jump_index + 2 >= end_index:
        return False

    jump, back, body = instructions[jump_index : jump_index + 3]
    return (
        normalized_opcode_name(jump.opname) == "POP_JUMP_IF_TRUE"
        and back.opname == "JUMP_BACKWARD"
        and jump.argval == body.offset
    )


def read_sequence_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    header = read_sequence_match_header(
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if header is None:
        return None

    unpack_index = skip_match_prefix(
        instructions,
        header.length_jump_index + 1,
        end_index,
    )
    if unpack_index >= end_index:
        return None
    if is_fixed_length_sequence_test(
        instructions,
        header.get_len_index,
        header.load_count_index,
        header.compare_index,
        header.length_jump_index,
        header.miss_index,
    ):
        return read_fixed_sequence_match_test(
            instructions,
            offset_to_index,
            unpack_index,
            header.count_value,
            header.length_jump_index,
            header.miss_index,
        )
    if is_min_length_match_test(
        instructions,
        header.get_len_index,
        header.load_count_index,
        header.compare_index,
        header.length_jump_index,
        header.miss_index,
    ):
        return read_starred_sequence_match_test(
            instructions,
            offset_to_index,
            unpack_index,
            header.count_value,
            header.length_jump_index,
            header.miss_index,
        )

    return None


def read_sequence_match_header(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SequenceMatchHeader | None:
    if instructions[cursor].opname != "MATCH_SEQUENCE":
        return None

    match_jump_index = skip_match_prefix(instructions, cursor + 1, end_index)
    jump_target = read_match_jump_target(
        instructions,
        offset_to_index,
        match_jump_index,
        end_index,
    )
    if jump_target is None:
        return None

    indexes = read_sequence_length_test_indexes(
        instructions,
        match_jump_index + 1,
        end_index,
    )
    if indexes is None:
        return None

    get_len_index, load_count_index, compare_index, length_jump_index = indexes
    count_value = instructions[load_count_index].argval
    if not isinstance(count_value, int) or isinstance(count_value, bool):
        return None

    return SequenceMatchHeader(
        match_jump_index=match_jump_index,
        miss_index=jump_target.miss_index,
        get_len_index=get_len_index,
        load_count_index=load_count_index,
        compare_index=compare_index,
        length_jump_index=length_jump_index,
        count_value=count_value,
    )


def read_sequence_length_test_indexes(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> tuple[int, int, int, int] | None:
    get_len_index = skip_match_prefix(instructions, start_index, end_index)
    load_count_index = skip_match_prefix(instructions, get_len_index + 1, end_index)
    compare_index = skip_match_prefix(instructions, load_count_index + 1, end_index)
    length_jump_index = skip_match_prefix(instructions, compare_index + 1, end_index)
    if (
        get_len_index >= end_index
        or load_count_index >= end_index
        or compare_index >= end_index
        or length_jump_index >= end_index
    ):
        return None

    return get_len_index, load_count_index, compare_index, length_jump_index


def read_fixed_sequence_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    unpack_index: int,
    count_value: int,
    jump_index: int,
    miss_index: int,
) -> SimpleMatchTest | None:
    if instructions[unpack_index].opname != "UNPACK_SEQUENCE":
        return None
    if int(instructions[unpack_index].arg or 0) != count_value:
        return None

    captures = read_sequence_patterns(
        instructions, offset_to_index, unpack_index + 1, count_value, miss_index
    )
    if captures is None:
        return None

    patterns, body_start_index = captures
    return SimpleMatchTest(
        pattern=ast.MatchSequence(patterns=patterns),
        jump_index=jump_index,
        miss_index=miss_index,
        body_start_index=body_start_index,
    )


def read_starred_sequence_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    unpack_index: int,
    minimum_count: int,
    jump_index: int,
    miss_index: int,
) -> SimpleMatchTest | None:
    if instructions[unpack_index].opname != "UNPACK_EX":
        return None

    before_count, after_count = unpack_ex_counts(
        int(instructions[unpack_index].arg or 0)
    )
    if before_count + after_count != minimum_count:
        return None

    captures = read_starred_sequence_patterns(
        instructions,
        offset_to_index,
        unpack_index + 1,
        before_count,
        after_count,
        miss_index,
    )
    if captures is None:
        return None

    patterns, body_start_index = captures
    return SimpleMatchTest(
        pattern=ast.MatchSequence(patterns=patterns),
        jump_index=jump_index,
        miss_index=miss_index,
        body_start_index=body_start_index,
    )


def unpack_ex_counts(arg: int) -> tuple[int, int]:
    before_count = arg & 0xFF
    after_count = (arg >> 8) & 0xFF
    return before_count, after_count


def is_min_length_match_test(
    instructions: list[Instruction],
    get_len_index: int,
    load_count_index: int,
    compare_index: int,
    jump_index: int,
    miss_index: int,
) -> bool:
    if instructions[get_len_index].opname != "GET_LEN":
        return False
    if instructions[load_count_index].opname != "LOAD_CONST":
        return False

    compare = instructions[compare_index]
    if compare.opname != "COMPARE_OP" or compare.argrepr != ">=":
        return False

    jump = instructions[jump_index]
    return (
        is_match_miss_jump(jump) and int(jump.argval) == instructions[miss_index].offset
    )


def is_fixed_length_sequence_test(
    instructions: list[Instruction],
    get_len_index: int,
    load_count_index: int,
    compare_index: int,
    jump_index: int,
    miss_index: int,
) -> bool:
    if instructions[get_len_index].opname != "GET_LEN":
        return False
    if instructions[load_count_index].opname != "LOAD_CONST":
        return False

    compare = instructions[compare_index]
    if compare.opname != "COMPARE_OP" or compare.argrepr != "==":
        return False

    jump = instructions[jump_index]
    return (
        is_match_miss_jump(jump) and int(jump.argval) == instructions[miss_index].offset
    )


def read_sequence_patterns(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    count: int,
    end_index: int,
) -> tuple[list[ast.pattern], int] | None:
    patterns: dict[int, ast.pattern] = {}
    stack = list(range(count - 1, -1, -1))
    while stack:
        cursor = skip_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None

        instruction = instructions[cursor]
        if instruction.opname == "SWAP":
            depth = int(instruction.arg or 0)
            if depth < 2 or depth > len(stack):
                return None

            stack[-1], stack[-depth] = stack[-depth], stack[-1]
            cursor += 1
            continue

        parsed = read_sequence_item_pattern(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if parsed is None:
            return None

        patterns[stack.pop()], cursor = parsed

    return (
        [patterns[index] for index in range(count)],
        skip_match_prefix(instructions, cursor, end_index),
    )


def read_sequence_item_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> tuple[ast.pattern, int] | None:
    capture = sequence_capture_pattern(instructions[cursor])
    if capture is not None:
        return capture, cursor + 1

    value = read_class_value_pattern(instructions, cursor, end_index)
    if value is not None:
        return value

    return read_nested_sequence_pattern(
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )


def read_nested_sequence_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> tuple[ast.pattern, int] | None:
    nested = read_sequence_match_test(
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if nested is None:
        return None
    if not isinstance(nested.pattern, ast.MatchSequence):
        return None
    if nested.body_start_index is None:
        return None

    return nested.pattern, nested.body_start_index


def read_class_value_pattern(
    instructions: list[Instruction],
    cursor: int,
    miss_index: int,
) -> tuple[ast.pattern, int] | None:
    if is_none_match_miss_jump(instructions[cursor]):
        return (
            ast.MatchSingleton(value=None),
            skip_match_prefix(instructions, cursor + 1, miss_index),
        )
    if cursor + 2 >= miss_index:
        return None

    value_instruction = instructions[cursor]
    if value_instruction.opname != "LOAD_CONST":
        return None

    compare_index = skip_match_prefix(instructions, cursor + 1, miss_index)
    jump_index = skip_match_prefix(instructions, compare_index + 1, miss_index)
    if compare_index >= miss_index or jump_index >= miss_index:
        return None
    if not class_value_pattern_jump_matches(instructions, compare_index, jump_index):
        return None

    value = value_instruction.argval
    if value in {None, True, False} and instructions[compare_index].opname == "IS_OP":
        pattern: ast.pattern = ast.MatchSingleton(value=value)
    else:
        pattern = ast.MatchValue(value=ast.Constant(value=value))

    return pattern, skip_match_prefix(instructions, jump_index + 1, miss_index)


def class_value_pattern_jump_matches(
    instructions: list[Instruction],
    compare_index: int,
    jump_index: int,
) -> bool:
    compare = instructions[compare_index]
    jump = instructions[jump_index]

    if compare.opname == "COMPARE_OP" and compare.argrepr == "==":
        return is_match_miss_jump(jump)
    if compare.opname == "IS_OP" and int(compare.arg or 0) == 0:
        return is_match_miss_jump(jump)

    return False


def read_starred_sequence_patterns(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    before_count: int,
    after_count: int,
    end_index: int,
) -> tuple[list[ast.pattern], int] | None:
    captures = read_sequence_patterns(
        instructions,
        offset_to_index,
        cursor,
        before_count,
        end_index,
    )
    if captures is None:
        return None

    before_patterns, cursor = captures
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor >= end_index:
        return None

    star_pattern = starred_sequence_capture_pattern(instructions[cursor])
    if star_pattern is None:
        return None

    cursor += 1
    captures = read_sequence_patterns(
        instructions,
        offset_to_index,
        cursor,
        after_count,
        end_index,
    )
    if captures is None:
        return None

    after_patterns, body_start_index = captures
    return [*before_patterns, star_pattern, *after_patterns], body_start_index


def starred_sequence_capture_pattern(instruction: Instruction) -> ast.MatchStar | None:
    if instruction.opname == "POP_TOP":
        return ast.MatchStar(name=None)
    if instruction.opname in {
        "STORE_DEREF",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_NAME",
    }:
        return ast.MatchStar(name=safe_identifier(str(instruction.argval)))

    return None


MAPPING_REST_CLEANUP_OPS = {
    "BUILD_MAP",
    "COPY",
    "COPY_DICT_WITHOUT_KEYS",
    "DELETE_SUBSCR",
    "DICT_UPDATE",
    "SWAP",
    "UNPACK_SEQUENCE",
}


def skip_match_success_cleanup(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    while cursor < end_index and instructions[cursor].opname == "POP_TOP":
        cursor = skip_match_prefix(instructions, cursor + 1, end_index)

    return cursor


def sequence_capture_pattern(instruction: Instruction) -> ast.pattern | None:
    if instruction.opname == "POP_TOP":
        return ast.MatchAs(pattern=None, name=None)
    if instruction.opname in {
        "STORE_DEREF",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_NAME",
    }:
        return ast.MatchAs(
            pattern=None,
            name=safe_identifier(str(instruction.argval)),
        )

    return None


def read_default_match_case(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> SimpleMatchCaseRegion | None:
    body_start_index = skip_match_prefix(instructions, cursor, end_index)
    if (
        body_start_index < end_index
        and instructions[body_start_index].opname == "POP_TOP"
    ):
        body_start_index = skip_match_prefix(
            instructions, body_start_index + 1, end_index
        )

    name = None

    if (
        body_start_index < end_index
        and instructions[body_start_index].opname in MATCH_ALIAS_STORE_OPS
    ):
        # no statement starts with a store: it binds the subject, `case name:`
        name = safe_identifier(str(instructions[body_start_index].argval))
        body_start_index += 1

    body_end_index = find_terminal_body_end(instructions, body_start_index, end_index)
    if body_end_index is None and body_start_index < end_index:
        # no case leaves to code behind the match, so the last one runs to the end
        body_end_index = end_index
    if body_end_index is None:
        return None

    return SimpleMatchCaseRegion(
        pattern=ast.MatchAs(pattern=None, name=name),
        body_start_index=body_start_index,
        body_end_index=body_end_index,
    )


def skip_match_prefix(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    while cursor < end_index and instructions[cursor].opname in MATCH_PREFIX_OPS:
        cursor += 1

    return cursor


def skip_match_subject_cleanup(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor < end_index and instructions[cursor].opname == "POP_TOP":
        return cursor + 1

    return cursor


def is_match_miss_jump(instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    return opname in {
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_IF_NOT_NONE",
        "POP_JUMP_FORWARD_IF_NOT_NONE",
        "POP_JUMP_BACKWARD_IF_NOT_NONE",
    }


def is_none_match_miss_jump(instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    return "POP_JUMP" in opname and "IF_NOT_NONE" in opname


def find_terminal_body_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in MATCH_PREFIX_OPS:
            continue
        if is_match_terminal(instruction):
            return index + 1

    return None


def is_match_cleanup_range(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> bool:
    cursor = start_index
    while cursor < end_index:
        cursor = skip_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return True
        if instructions[cursor].opname != "POP_TOP":
            return False

        cursor += 1

    return True


def is_match_terminal(instruction: Instruction) -> bool:
    return instruction.opname in {
        "INTERPRETER_EXIT",
        "RETURN_CONST",
        "RETURN_VALUE",
        "RAISE_VARARGS",
        "RERAISE",
    }


@dataclass(frozen=True, slots=True)
class WithRegion:
    body_start_index: int
    body_end_index: int
    after_index: int
    context_index: int | None = None
    optional_var_index: int | None = None
    returns_value: bool = False
    trailing_return_index: int | None = None
    is_async: bool = False
    target_end_index: int | None = None


WITH_CLEANUP_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP", "PRECALL"}
WithSetupPredicate = Callable[[list[Instruction], int, int], bool]


def make_with_item(
    context_expr: ast.expr, optional_vars: ast.expr | None = None
) -> ast.withitem:
    return ast.withitem(context_expr=context_expr, optional_vars=optional_vars)


def make_with_statement(
    context_expr: ast.expr,
    body: list[ast.stmt],
    optional_vars: ast.expr | None = None,
) -> ast.With:
    return ast.With(
        items=[make_with_item(context_expr, optional_vars)],
        body=body or [ast.Pass()],
        type_comment=None,
    )


def find_with_region(
    instructions: list[Instruction],
    before_with_index: int,
    end_index: int,
) -> WithRegion | None:
    if instructions[before_with_index].opname == "BEFORE_ASYNC_WITH":
        return find_async_with_region(instructions, before_with_index, end_index)
    if instructions[before_with_index].opname != "BEFORE_WITH":
        return find_load_special_with_region(instructions, before_with_index, end_index)

    optional_var_index, body_start_index = read_with_target(
        instructions, before_with_index + 1, end_index
    )
    if body_start_index is None:
        return None

    return find_sync_with_body_region(
        instructions,
        body_start_index,
        end_index,
        optional_var_index,
        context_index=None,
        is_nested_setup=is_before_with_instruction,
    )


def find_protected_with_region(
    code: object,
    instructions: list[Instruction],
    before_with_index: int,
    end_index: int,
    claimed: dict[int, int],
) -> WithRegion | None:
    is_async = instructions[before_with_index].opname == "BEFORE_ASYNC_WITH"
    if instructions[before_with_index].opname != "BEFORE_WITH" and not is_async:
        return None

    enter_end_index = (
        skip_async_with_await(instructions, before_with_index + 1, end_index)
        if is_async
        else before_with_index + 1
    )
    if enter_end_index is None:
        return None

    optional_var_index, body_start_index = read_with_target(
        instructions, enter_end_index, end_index
    )
    target_end_index = None

    if body_start_index is None:
        target_end_index = with_target_end(instructions, enter_end_index, end_index)
        body_start_index = target_end_index

    if body_start_index is None:
        return None

    spans = read_table_spans(code, instructions, make_offset_index(instructions))
    handler_index = with_handler_index(instructions, spans, enter_end_index)
    regions = handler_regions(instructions, spans)
    if handler_index is None or handler_index not in regions:
        return None

    extent = protected_extent(
        instructions,
        spans,
        regions,
        handler_index,
        enclosing_handler(regions, before_with_index),
    )
    if extent is None or extent[1] > end_index:
        return None

    body_end_index = with_body_end_index(instructions, extent[1], end_index)
    while body_end_index != handler_index and body_end_index in regions:
        body_end_index = regions[body_end_index]

    cleanup: tuple[int, bool, int | None] | None

    if body_end_index == handler_index:
        cleanup = (handler_index, False, None)
    else:
        cleanup = match_with_cleanup(instructions, body_end_index, end_index)

    if cleanup is None:
        return None

    cleanup_end_index, returns_value, trailing_return_index = cleanup
    claimed[handler_index] = with_handler_end(instructions, handler_index, regions)
    claim_early_with_exits(instructions, spans, handler_index, body_end_index, claimed)
    return WithRegion(
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        after_index=cleanup_end_index,
        optional_var_index=optional_var_index,
        returns_value=returns_value,
        trailing_return_index=trailing_return_index,
        is_async=is_async,
        target_end_index=target_end_index,
    )


SUPPRESSED_WITH_EXIT = ("POP_TOP", "POP_EXCEPT", "POP_TOP", "POP_TOP")


def with_handler_end(
    instructions: list[Instruction], handler_index: int, regions: dict[int, int]
) -> int:
    region_end = regions[handler_index]
    # the test of __exit__'s answer, after TO_BOOL from 3.13
    test = next(
        (
            item
            for item in instructions[handler_index + 1 : handler_index + 4]
            if is_forward_conditional_jump(item)
        ),
        None,
    )
    exit_names = tuple(
        item.opname for item in instructions[region_end : region_end + 4]
    )
    if (
        region_end < len(instructions)
        and test is not None
        and test.argval == instructions[region_end].offset
        and exit_names == SUPPRESSED_WITH_EXIT
    ):
        return region_end + 4

    return region_end


WITH_LEAVE_OPS = {"JUMP_BACKWARD", "JUMP_FORWARD", "RETURN_CONST", "RETURN_VALUE"}


def with_body_end_index(
    instructions: list[Instruction],
    protected_end_index: int,
    end_index: int,
) -> int:
    leave_index = read_with_exit_call_end(instructions, protected_end_index, end_index)
    if leave_index is None:
        return protected_end_index

    while (
        after := dropped_under_value_end(instructions, leave_index, end_index)
    ) is not None:
        leave_index = after

    if leave_index < end_index and instructions[leave_index].opname == "POP_TOP":
        leave_index += 1
    if leave_index >= end_index:
        return protected_end_index
    if instructions[leave_index].opname not in WITH_LEAVE_OPS:
        return protected_end_index

    body_end_index = leave_index + 1
    if body_end_index >= end_index or not instructions[body_end_index].is_jump_target:
        return protected_end_index
    if match_with_cleanup(instructions, body_end_index, end_index) is None:
        return protected_end_index

    return body_end_index


def with_target_end(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    depth = 1

    for index in range(start_index, end_index):
        effect = instruction_stack_effect(instructions[index])
        if effect is None:
            break

        depth += effect.net
        if depth == 0:
            return index + 1

    for index in range(start_index + 1, end_index):
        if instructions[index].starts_line is not None:
            return index

    return None


def with_handler_index(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    protected_index: int,
) -> int | None:
    for span in spans:
        if span.start_index != protected_index:
            continue

        handler_index = span.target_index
        if handler_index + 1 >= len(instructions):
            return None
        if instructions[handler_index + 1].opname == "WITH_EXCEPT_START":
            return handler_index

    return None


def claim_early_with_exits(
    instructions: list[Instruction],
    spans: tuple[TableSpan, ...],
    handler_index: int,
    body_end_index: int,
    claimed: dict[int, int],
) -> None:
    for span in spans:
        if span.target_index != handler_index or span.end_index >= handler_index:
            continue
        if span.end_index == body_end_index:
            continue

        exit_end_index = read_with_exit_call_end(
            instructions, span.end_index, handler_index
        )
        if exit_end_index is not None:
            claimed[span.end_index] = exit_end_index


def read_with_exit_call_end(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int | None:
    if cursor < end_index and instructions[cursor].opname == "SWAP":
        cursor += 1

    for _index in range(3):
        if cursor >= end_index or not is_none_load(instructions[cursor]):
            return None

        cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor >= end_index or instructions[cursor].opname != "CALL":
        return None

    awaited = skip_async_with_await(instructions, cursor + 1, end_index)
    cursor = cursor + 1 if awaited is None else awaited
    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor >= end_index or instructions[cursor].opname != "POP_TOP":
        return None

    return cursor + 1


def find_async_with_region(
    instructions: list[Instruction],
    before_with_index: int,
    end_index: int,
) -> WithRegion | None:
    enter_end_index = skip_async_with_await(
        instructions,
        before_with_index + 1,
        end_index,
    )
    if enter_end_index is None:
        return None

    optional_var_index, body_start_index = read_with_target(
        instructions,
        enter_end_index,
        end_index,
    )
    if body_start_index is None:
        return None

    return find_async_with_body_region(
        instructions,
        body_start_index,
        end_index,
        optional_var_index,
        context_index=None,
        is_nested_setup=is_before_async_with_instruction,
    )


LOAD_SPECIAL_WITH_CONTEXT_OPS = {
    "LOAD_NAME",
    "LOAD_GLOBAL",
    "LOAD_FAST",
    "LOAD_FAST_CHECK",
    "LOAD_FAST_BORROW",
    "LOAD_DEREF",
}


def find_load_special_with_region(
    instructions: list[Instruction],
    context_index: int,
    end_index: int,
) -> WithRegion | None:
    async_region = find_load_special_async_with_region(
        instructions,
        context_index,
        end_index,
    )
    if async_region is not None:
        return async_region
    if not is_load_special_with_setup(instructions, context_index, end_index):
        return None

    optional_var_index, body_start_index = read_with_target(
        instructions, context_index + 8, end_index
    )
    if body_start_index is None:
        return None

    return find_sync_with_body_region(
        instructions,
        body_start_index,
        end_index,
        optional_var_index,
        context_index=context_index,
        is_nested_setup=is_load_special_with_setup,
    )


def find_load_special_async_with_region(
    instructions: list[Instruction],
    context_index: int,
    end_index: int,
) -> WithRegion | None:
    setup_with_index = load_special_async_with_setup_end(
        instructions,
        context_index,
        end_index,
    )
    if setup_with_index is None:
        return None

    optional_var_index, body_start_index = read_with_target(
        instructions, setup_with_index + 1, end_index
    )
    if body_start_index is None:
        return None

    return find_async_with_body_region(
        instructions,
        body_start_index,
        end_index,
        optional_var_index,
        context_index=context_index,
        is_nested_setup=is_load_special_async_with_setup,
    )


def is_before_with_instruction(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> bool:
    return index < end_index and instructions[index].opname == "BEFORE_WITH"


def is_before_async_with_instruction(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> bool:
    return index < end_index and instructions[index].opname == "BEFORE_ASYNC_WITH"


def is_load_special_async_with_setup(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> bool:
    return load_special_async_with_setup_end(instructions, index, end_index) is not None


def find_sync_with_body_region(
    instructions: list[Instruction],
    body_start_index: int,
    end_index: int,
    optional_var_index: int | None,
    context_index: int | None,
    is_nested_setup: WithSetupPredicate,
) -> WithRegion | None:
    nested_depth = 0

    for index in range(body_start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in WITH_CLEANUP_PREFIX_OPS:
            continue

        if is_nested_setup(instructions, index, end_index):
            nested_depth += 1
            continue

        region = match_sync_with_body_end(
            instructions,
            index,
            end_index,
            body_start_index,
            optional_var_index,
            context_index,
        )
        if region is None:
            continue

        if nested_depth:
            nested_depth -= 1
            continue

        return region

    return None


def match_sync_with_body_end(
    instructions: list[Instruction],
    index: int,
    end_index: int,
    body_start_index: int,
    optional_var_index: int | None,
    context_index: int | None,
) -> WithRegion | None:
    exceptional_cleanup = match_exceptional_with_cleanup(instructions, index, end_index)
    if exceptional_cleanup is not None:
        return WithRegion(
            body_start_index=body_start_index,
            body_end_index=index + 1,
            after_index=exceptional_cleanup,
            context_index=context_index,
            optional_var_index=optional_var_index,
        )

    cleanup = match_with_cleanup(instructions, index, end_index)
    if cleanup is None:
        return None

    cleanup_end_index, returns_value, trailing_return_index = cleanup
    return WithRegion(
        body_start_index=body_start_index,
        body_end_index=index,
        after_index=cleanup_end_index,
        context_index=context_index,
        optional_var_index=optional_var_index,
        returns_value=returns_value,
        trailing_return_index=trailing_return_index,
    )


def find_async_with_body_region(
    instructions: list[Instruction],
    body_start_index: int,
    end_index: int,
    optional_var_index: int | None,
    context_index: int | None,
    is_nested_setup: WithSetupPredicate,
) -> WithRegion | None:
    nested_depth = 0

    for index in range(body_start_index, end_index):
        instruction = instructions[index]
        if instruction.opname in WITH_CLEANUP_PREFIX_OPS:
            continue

        if is_nested_setup(instructions, index, end_index):
            nested_depth += 1
            continue

        cleanup = match_async_with_cleanup(instructions, index, end_index)
        if cleanup is None:
            continue

        if nested_depth:
            nested_depth -= 1
            continue

        cleanup_end_index, returns_value, trailing_return_index = cleanup
        return WithRegion(
            body_start_index=body_start_index,
            body_end_index=index,
            after_index=cleanup_end_index,
            context_index=context_index,
            optional_var_index=optional_var_index,
            returns_value=returns_value,
            trailing_return_index=trailing_return_index,
            is_async=True,
        )

    return None


def load_special_async_with_setup_end(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> int | None:
    if not is_load_special_async_with_prefix(instructions, index, end_index):
        return None

    setup_with_index = skip_async_with_await(instructions, index + 7, end_index)
    if setup_with_index is None:
        return None
    if (
        setup_with_index >= end_index
        or instructions[setup_with_index].opname != "SETUP_WITH"
    ):
        return None

    return setup_with_index


def is_load_special_async_with_prefix(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> bool:
    if index + 6 >= end_index:
        return False

    return (
        instructions[index].opname in LOAD_SPECIAL_WITH_CONTEXT_OPS
        and instructions[index + 1].opname == "COPY"
        and int(instructions[index + 1].arg or 0) == 1
        and instructions[index + 2].opname == "LOAD_SPECIAL"
        and int(instructions[index + 2].arg or 0) == 3
        and instructions[index + 3].opname == "SWAP"
        and int(instructions[index + 3].arg or 0) == 2
        and instructions[index + 4].opname == "SWAP"
        and int(instructions[index + 4].arg or 0) == 3
        and instructions[index + 5].opname == "LOAD_SPECIAL"
        and int(instructions[index + 5].arg or 0) == 2
        and instructions[index + 6].opname == "CALL"
        and int(instructions[index + 6].arg or 0) == 0
    )


def is_load_special_with_setup(
    instructions: list[Instruction],
    index: int,
    end_index: int,
) -> bool:
    if index + 7 >= end_index:
        return False

    return (
        instructions[index].opname in LOAD_SPECIAL_WITH_CONTEXT_OPS
        and instructions[index + 1].opname == "COPY"
        and int(instructions[index + 1].arg or 0) == 1
        and instructions[index + 2].opname == "LOAD_SPECIAL"
        and int(instructions[index + 2].arg or 0) == 1
        and instructions[index + 3].opname == "SWAP"
        and int(instructions[index + 3].arg or 0) == 2
        and instructions[index + 4].opname == "SWAP"
        and int(instructions[index + 4].arg or 0) == 3
        and instructions[index + 5].opname == "LOAD_SPECIAL"
        and int(instructions[index + 5].arg or 0) == 0
        and instructions[index + 6].opname == "CALL"
        and int(instructions[index + 6].arg or 0) == 0
        and instructions[index + 7].opname == "SETUP_WITH"
    )


def read_with_target(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> tuple[int | None, int | None]:
    cursor = start_index
    while cursor < end_index and instructions[cursor].opname in WITH_CLEANUP_PREFIX_OPS:
        cursor += 1

    if cursor >= end_index:
        return None, None

    instruction = instructions[cursor]
    if instruction.opname == "POP_TOP":
        return None, cursor + 1
    if instruction.opname in {
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_FAST",
        "STORE_DEREF",
    }:
        return cursor, cursor + 1

    return None, None


def match_with_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> tuple[int, bool, int | None] | None:
    cursor = start_index
    returns_value = False
    trailing_return_index: int | None = None

    if (
        cursor < end_index
        and instructions[cursor].opname == "SWAP"
        and instructions[cursor].arg == 2
    ):
        returns_value = True
        cursor += 1

    cursor = skip_optional_pop_block(instructions, cursor, end_index)
    exit_call = with_exit_call_index(instructions, cursor, end_index)
    if exit_call is None:
        return None

    cursor = exit_call
    awaited = skip_async_with_await(instructions, cursor + 1, end_index)
    cursor = cursor + 1 if awaited is None else awaited
    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor >= end_index or instructions[cursor].opname != "POP_TOP":
        return None

    cursor += 1
    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor < end_index and jumps_over_with_handler(instructions, cursor):
        cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)

    if returns_value:
        if cursor >= end_index or instructions[cursor].opname not in {
            "RETURN_CONST",
            "RETURN_VALUE",
        }:
            return None

        cursor += 1
        cursor = skip_returning_with_exception_handler(instructions, cursor, end_index)
        return cursor, returns_value, trailing_return_index

    if cursor < end_index and instructions[cursor].opname in {
        "RETURN_CONST",
        "RETURN_VALUE",
    }:
        trailing_return_index = cursor
        cursor += 1
        cursor = skip_returning_with_exception_handler(instructions, cursor, end_index)

    return cursor, returns_value, trailing_return_index


# 3.9, 3.10 and PyPy call __exit__(None, None, None) with one None copied twice
INLINE_EXIT_CALL = ("LOAD_CONST", "DUP_TOP", "DUP_TOP", "CALL_FUNCTION")


def with_exit_call_index(
    instructions: list[Instruction], cursor: int, end_index: int
) -> int | None:
    # the call of __exit__(None, None, None) as the body ends
    found = instructions[cursor : cursor + len(INLINE_EXIT_CALL)]
    if (
        cursor + len(INLINE_EXIT_CALL) <= end_index
        and tuple(item.opname for item in found) == INLINE_EXIT_CALL
        and is_none_load(found[0])
        and found[-1].arg == 3
    ):
        return cursor + len(INLINE_EXIT_CALL) - 1

    for _index in range(3):
        cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
        if cursor >= end_index or not is_none_load(instructions[cursor]):
            return None

        cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if (
        cursor >= end_index
        or instructions[cursor].opname != "CALL"
        or int(instructions[cursor].arg or 0) not in {2, 3}
    ):
        return None

    return cursor


def jumps_over_with_handler(instructions: list[Instruction], index: int) -> bool:
    if instructions[index].opname not in FORWARD_JUMPS:
        return False
    if instructions[index].opname == "JUMP":
        return True

    return (
        index + 1 < len(instructions)
        and instructions[index + 1].opname == "PUSH_EXC_INFO"
    )


def match_exceptional_with_cleanup(
    instructions: list[Instruction],
    terminal_index: int,
    end_index: int,
) -> int | None:
    if not is_match_terminal(instructions[terminal_index]):
        return None

    cursor = skip_with_cleanup_prefix(instructions, terminal_index + 1, end_index)
    if (
        cursor + 1 >= end_index
        or instructions[cursor].opname != "PUSH_EXC_INFO"
        or instructions[cursor + 1].opname != "WITH_EXCEPT_START"
    ):
        return None

    saw_copy = False
    saw_pop_except = False

    for index in range(cursor + 2, end_index):
        instruction = instructions[index]
        if instruction.opname == "COPY":
            saw_copy = True
            continue

        if saw_copy and instruction.opname == "POP_EXCEPT":
            saw_pop_except = True
            continue

        if saw_pop_except and instruction.opname == "RERAISE":
            return index + 1

    return None


def match_async_with_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> tuple[int, bool, int | None] | None:
    cursor = start_index
    returns_value = False
    trailing_return_index: int | None = None

    if (
        cursor < end_index
        and instructions[cursor].opname == "SWAP"
        and int(instructions[cursor].arg or 0) == 2
    ):
        returns_value = True
        cursor += 1

    call_end = read_async_with_cleanup_call_end(instructions, cursor, end_index)
    if call_end is None:
        return None

    cursor = skip_with_cleanup_prefix(instructions, call_end, end_index)
    if cursor >= end_index or instructions[cursor].opname != "POP_TOP":
        return None

    cursor += 1
    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)

    if returns_value:
        if cursor >= end_index or instructions[cursor].opname not in {
            "RETURN_CONST",
            "RETURN_VALUE",
        }:
            return None

        cursor += 1
        cursor = skip_async_with_exception_handler(instructions, cursor, end_index)
        return cursor, returns_value, trailing_return_index

    if cursor < end_index and instructions[cursor].opname in {
        "RETURN_CONST",
        "RETURN_VALUE",
    }:
        trailing_return_index = cursor
        cursor += 1

    cursor = skip_async_with_exception_handler(instructions, cursor, end_index)
    return cursor, returns_value, trailing_return_index


def read_async_with_cleanup_call_end(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int | None:
    cursor = skip_optional_pop_block(instructions, cursor, end_index)

    for _index in range(3):
        cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
        if cursor >= end_index or not is_none_load(instructions[cursor]):
            return None

        cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if (
        cursor >= end_index
        or instructions[cursor].opname != "CALL"
        or int(instructions[cursor].arg or 0) not in {2, 3}
    ):
        return None

    return skip_async_with_await(instructions, cursor + 1, end_index)


def skip_async_with_exception_handler(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = skip_with_cleanup_prefix(instructions, start_index, end_index)

    for _index in range(8):
        if cursor >= end_index:
            return start_index
        if instructions[cursor].opname == "PUSH_EXC_INFO":
            break
        if instructions[cursor].opname not in {
            "CLEANUP_THROW",
            "JUMP_BACKWARD",
            "JUMP_BACKWARD_NO_INTERRUPT",
            "JUMP_FORWARD",
        }:
            return start_index

        cursor = skip_with_cleanup_prefix(instructions, cursor + 1, end_index)
    else:
        return start_index

    if (
        cursor + 1 >= end_index
        or instructions[cursor + 1].opname != "WITH_EXCEPT_START"
    ):
        return start_index

    saw_copy = False
    saw_pop_except = False

    for index in range(cursor + 2, end_index):
        instruction = instructions[index]
        if instruction.opname == "COPY":
            saw_copy = True
            continue

        if saw_copy and instruction.opname == "POP_EXCEPT":
            saw_pop_except = True
            continue

        if saw_pop_except and instruction.opname == "RERAISE":
            return index + 1

    return start_index


def skip_async_with_await(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = skip_with_cleanup_prefix(instructions, start_index, end_index)
    if cursor >= end_index or instructions[cursor].opname != "GET_AWAITABLE":
        return None

    cursor = skip_with_cleanup_prefix(instructions, cursor + 1, end_index)
    if cursor < end_index and instructions[cursor].opname == "PUSH_NULL":
        cursor = skip_with_cleanup_prefix(instructions, cursor + 1, end_index)
    if cursor >= end_index or not is_none_load(instructions[cursor]):
        return None

    cursor = skip_with_cleanup_prefix(instructions, cursor + 1, end_index)
    if cursor >= end_index or instructions[cursor].opname != "SEND":
        return None

    for index in range(cursor + 1, end_index):
        instruction = instructions[index]
        if instruction.opname in WITH_CLEANUP_PREFIX_OPS:
            continue
        if instruction.opname == "END_SEND":
            return index + 1

    return None


def skip_optional_pop_block(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor < end_index and instructions[cursor].opname == "POP_BLOCK":
        return cursor + 1

    return cursor


def skip_returning_with_exception_handler(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = skip_with_cleanup_prefix(instructions, start_index, end_index)
    if cursor >= end_index or instructions[cursor].opname != "PUSH_EXC_INFO":
        return start_index
    if (
        cursor + 2 >= end_index
        or instructions[cursor + 1].opname != "WITH_EXCEPT_START"
    ):
        return start_index

    for index in range(cursor + 2, end_index):
        instruction = instructions[index]
        if instruction.opname == "RERAISE" and int(instruction.arg or 0) == 1:
            return skip_chained_returning_with_cleanup(
                instructions, index + 1, end_index
            )

    return start_index


def skip_chained_returning_with_cleanup(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index

    for _index in range(8):
        cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
        cleanup = match_with_cleanup(instructions, cursor, end_index)
        if cleanup is not None:
            cursor = cleanup[0]
            cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
            if cursor < end_index and is_none_return(instructions[cursor]):
                cursor += 1
            continue

        skipped = skip_returning_with_exception_handler(instructions, cursor, end_index)
        if skipped == cursor:
            return cursor

        cursor = skipped

    return cursor


def skip_with_cleanup_prefix(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    while cursor < end_index and instructions[cursor].opname in WITH_CLEANUP_PREFIX_OPS:
        cursor += 1

    return cursor
