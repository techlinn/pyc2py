import ast
from collections.abc import Callable
from dataclasses import dataclass

from pyc2py.astree import safe_identifier
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name


@dataclass(frozen=True, slots=True)
class StatementBlock:
    body: tuple[ast.stmt, ...]


def make_block(statements: list[ast.stmt]) -> StatementBlock:
    return StatementBlock(body=tuple(statements))


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
    if isinstance(condition, ast.Compare) and len(condition.ops) == 1:
        if isinstance(condition.ops[0], ast.Is):
            return ast.Compare(
                left=condition.left,
                ops=[ast.IsNot()],
                comparators=condition.comparators,
            )
        if isinstance(condition.ops[0], ast.IsNot):
            return ast.Compare(
                left=condition.left, ops=[ast.Is()], comparators=condition.comparators
            )
        if isinstance(condition.ops[0], ast.In):
            return ast.Compare(
                left=condition.left,
                ops=[ast.NotIn()],
                comparators=condition.comparators,
            )
        if isinstance(condition.ops[0], ast.NotIn):
            return ast.Compare(
                left=condition.left, ops=[ast.In()], comparators=condition.comparators
            )
    return ast.UnaryOp(op=ast.Not(), operand=condition)


def condition_from_jump(opname: str, value: ast.expr) -> ast.expr:
    value = unwrap_truth_test(value)
    if is_none_jump(opname):
        return ast.Compare(
            left=value,
            ops=[ast.Is()],
            comparators=[ast.Constant(value=None)],
        )
    return value


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


@dataclass(frozen=True, slots=True)
class LegacyForLoopPattern:
    body_end_index: int
    after_index: int


LOOP_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "SET_LINENO", "NOP"}


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
        if (
            find_loop_back_jump(
                instructions,
                loop_entry_offsets(instructions, get_iter_index, for_iter_index),
                search_end_index,
                end_index,
            )
            is None
        ):
            return None

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


def find_legacy_for_loop_pattern(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    for_loop_index: int,
    end_index: int,
) -> LegacyForLoopPattern | None:
    instruction = instructions[for_loop_index]
    if instruction.opname != "FOR_LOOP" or instruction.arg is None:
        return None

    body_end_offset = instruction.offset + int(instruction.arg)
    body_end_index = offset_to_index.get(body_end_offset)
    if body_end_index is None:
        return None
    if body_end_index <= for_loop_index or body_end_index >= end_index:
        return None
    if instructions[body_end_index].opname not in {"JUMP_ABSOLUTE", "JUMP"}:
        return None

    return LegacyForLoopPattern(
        body_end_index=body_end_index,
        after_index=skip_loop_cleanup(instructions, body_end_index + 1, end_index),
    )


def find_for_iter_index(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    cursor = start_index
    while cursor < end_index:
        instruction = instructions[cursor]
        if instruction.opname in {"FOR_ITER", "FOR_LOOP"}:
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


def for_loop_body_end_index(
    instructions: list[Instruction],
    loop_offsets: set[int],
    back_jump_index: int,
    exit_index: int,
) -> int:
    tail_index = exit_index - 1
    if tail_index <= back_jump_index:
        return back_jump_index

    tail = instructions[tail_index]
    if (
        tail.opname in {"JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP"}
        and tail.argval in loop_offsets
    ):
        return tail_index
    return exit_index


def skip_loop_cleanup(
    instructions: list[Instruction],
    exit_index: int,
    end_index: int,
) -> int:
    if exit_index >= end_index:
        return exit_index
    if instructions[exit_index].opname in {"POP_BLOCK", "END_FOR"}:
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


@dataclass(frozen=True, slots=True)
class SimpleMatchTest:
    pattern: ast.pattern
    jump_index: int
    miss_index: int
    body_start_index: int | None = None
    body_end_index: int | None = None


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
class MappingMatchHeader:
    match_jump_index: int
    miss_index: int
    keys_index: int


@dataclass(frozen=True, slots=True)
class MappingKeysShape:
    keys: list[ast.expr]
    keys_jump_index: int
    inner_miss_index: int
    unpack_index: int


@dataclass(frozen=True, slots=True)
class MappingCaptureShape:
    patterns: list[ast.pattern]
    rest_name: str | None
    body_start_index: int


@dataclass(frozen=True, slots=True)
class ClassMatchShape:
    class_expr: ast.expr
    keyword_attrs: list[str]
    positional_count: int
    capture_count: int
    jump_index: int
    miss_index: int
    unpack_index: int


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

    for _case_index in range(32):
        case_cursor = read_match_case_cursor(instructions, cursor, end_index)
        if case_cursor is None:
            return None

        cursor, copied_subject = case_cursor

        parsed = read_simple_match_test(
            instructions, offset_to_index, cursor, end_index
        )
        if parsed is None:
            return read_simple_match_default_region(
                instructions, cases, cursor, end_index
            )

        region = make_simple_match_case_region(
            instructions,
            offset_to_index,
            parsed,
            copied_subject,
        )
        if region is None:
            return None

        cases.append(region)
        cursor = parsed.miss_index
        if cursor == case_cursor[0]:
            return None
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

    terminal_body_end = find_terminal_body_end(
        instructions, body_start_index, body_end_index
    )
    if terminal_body_end != body_end_index:
        if terminal_body_end is None or not is_match_cleanup_range(
            instructions, terminal_body_end, body_end_index
        ):
            return None
        body_end_index = terminal_body_end

    return SimpleMatchCaseRegion(
        pattern=pattern,
        body_start_index=body_start_index,
        body_end_index=body_end_index,
        copied_subject=copied_subject,
        guard_start_index=guard_start_index,
        guard_end_index=guard_end_index,
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
        if not is_match_miss_jump(instruction):
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
                    pattern=ast.MatchOr(patterns=alternatives),
                    body_start_index=case.body_start_index,
                    body_end_index=case.body_end_index,
                    copied_subject=case.copied_subject,
                    guard_start_index=case.guard_start_index,
                    guard_end_index=case.guard_end_index,
                )
            )
        cursor = next_cursor
    return collapsed


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
        if left_instruction.opname != right_instruction.opname:
            return False
        if left_instruction.argrepr != right_instruction.argrepr:
            return False
    return True


def read_simple_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor >= end_index:
        return None

    structural_test = read_structural_match_test(
        instructions, offset_to_index, cursor, end_index
    )
    if structural_test is not None:
        return structural_test

    instruction = instructions[cursor]
    none_test = read_none_match_test(instruction, offset_to_index, cursor)
    if none_test is not None:
        return none_test

    return read_literal_match_test(instructions, offset_to_index, cursor, end_index)


def read_structural_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    readers = (
        read_class_match_test,
        read_mapping_match_test,
        read_sequence_match_test,
    )
    for reader in readers:
        parsed = reader(instructions, offset_to_index, cursor, end_index)
        if parsed is not None:
            return parsed
    return None


def read_none_match_test(
    instruction: Instruction,
    offset_to_index: dict[int, int],
    cursor: int,
) -> SimpleMatchTest | None:
    if not is_none_match_miss_jump(instruction):
        return None
    miss_index = offset_to_index.get(int(instruction.argval))
    if miss_index is None:
        return None
    return SimpleMatchTest(
        pattern=ast.MatchSingleton(value=None),
        jump_index=cursor,
        miss_index=miss_index,
    )


def read_literal_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    instruction = instructions[cursor]
    if instruction.opname != "LOAD_CONST" or cursor + 2 >= end_index:
        return None

    compare_index = skip_match_prefix(instructions, cursor + 1, end_index)
    if compare_index >= end_index:
        return None

    jump_target = read_match_jump_target(
        instructions,
        offset_to_index,
        skip_match_prefix(instructions, compare_index + 1, end_index),
        end_index,
    )
    if jump_target is None:
        return None

    return make_literal_match_test(
        instruction,
        instructions[compare_index],
        jump_target,
    )


def make_literal_match_test(
    instruction: Instruction,
    compare: Instruction,
    jump_target: MatchJumpTarget,
) -> SimpleMatchTest | None:
    value = ast.Constant(value=instruction.argval)
    if compare.opname == "COMPARE_OP" and compare.argrepr == "==":
        return SimpleMatchTest(
            pattern=ast.MatchValue(value=value),
            jump_index=jump_target.jump_index,
            miss_index=jump_target.miss_index,
        )
    if (
        compare.opname == "IS_OP"
        and int(compare.arg or 0) == 0
        and instruction.argval in {None, True, False}
    ):
        return SimpleMatchTest(
            pattern=ast.MatchSingleton(value=instruction.argval),
            jump_index=jump_target.jump_index,
            miss_index=jump_target.miss_index,
        )
    return None


def read_match_jump_target(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    jump_index: int,
    end_index: int,
) -> MatchJumpTarget | None:
    if jump_index >= end_index:
        return None
    jump = instructions[jump_index]
    if not is_match_miss_jump(jump):
        return None
    miss_index = offset_to_index.get(int(jump.argval))
    if miss_index is None:
        return None
    return MatchJumpTarget(jump_index=jump_index, miss_index=miss_index)


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


def read_mapping_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    header = read_mapping_match_header(
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if header is None:
        return None

    empty_mapping = read_empty_mapping_match_test(instructions, header)
    if empty_mapping is not None:
        return empty_mapping

    keys_index = normalize_mapping_keys_index(
        instructions,
        header.keys_index,
        header.miss_index,
        end_index,
    )
    if keys_index is None:
        return None

    shape = read_mapping_keys_shape(
        instructions,
        offset_to_index,
        keys_index,
        header.miss_index,
        end_index,
    )
    if shape is None:
        return None

    captures = read_mapping_capture_shape(
        instructions,
        offset_to_index,
        shape.keys,
        shape.unpack_index,
        shape.inner_miss_index,
    )
    if captures is None:
        return None
    return SimpleMatchTest(
        pattern=ast.MatchMapping(
            keys=shape.keys,
            patterns=captures.patterns,
            rest=captures.rest_name,
        ),
        jump_index=shape.keys_jump_index,
        miss_index=header.miss_index,
        body_start_index=captures.body_start_index,
        body_end_index=shape.inner_miss_index,
    )


def read_mapping_match_header(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> MappingMatchHeader | None:
    if instructions[cursor].opname != "MATCH_MAPPING":
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
    keys_index = skip_match_prefix(instructions, match_jump_index + 1, end_index)
    return MappingMatchHeader(
        match_jump_index=match_jump_index,
        miss_index=jump_target.miss_index,
        keys_index=keys_index,
    )


def read_empty_mapping_match_test(
    instructions: list[Instruction],
    header: MappingMatchHeader,
) -> SimpleMatchTest | None:
    if header.keys_index >= header.miss_index:
        return None
    if instructions[header.keys_index].opname != "POP_TOP":
        return None
    body_start_index = skip_match_success_cleanup(
        instructions,
        header.keys_index,
        header.miss_index,
    )
    return SimpleMatchTest(
        pattern=ast.MatchMapping(keys=[], patterns=[], rest=None),
        jump_index=header.match_jump_index,
        miss_index=header.miss_index,
        body_start_index=body_start_index,
        body_end_index=header.miss_index,
    )


def normalize_mapping_keys_index(
    instructions: list[Instruction],
    keys_index: int,
    miss_index: int,
    end_index: int,
) -> int | None:
    if is_mapping_key_tuple(instructions, keys_index):
        return keys_index

    indexes = read_sequence_length_test_indexes(instructions, keys_index, end_index)
    if indexes is None:
        return None
    get_len_index, load_count_index, compare_index, length_jump_index = indexes
    if not is_min_length_match_test(
        instructions,
        get_len_index,
        load_count_index,
        compare_index,
        length_jump_index,
        miss_index,
    ):
        return None
    return skip_match_prefix(instructions, length_jump_index + 1, end_index)


def read_mapping_keys_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    keys_index: int,
    miss_index: int,
    end_index: int,
) -> MappingKeysShape | None:
    indexes = read_mapping_keys_test_indexes(instructions, keys_index, end_index)
    if indexes is None:
        return None
    match_keys_index, copy_index, keys_jump_index = indexes

    keys = read_mapping_keys(instructions, keys_index)
    if keys is None or not is_mapping_keys_test(
        instructions,
        match_keys_index,
        copy_index,
        keys_jump_index,
    ):
        return None

    inner_miss_index = offset_to_index.get(int(instructions[keys_jump_index].argval))
    if inner_miss_index is None or inner_miss_index > miss_index:
        return None

    unpack_index = read_unpack_sequence_index(
        instructions,
        keys_jump_index + 1,
        len(keys),
        end_index,
    )
    if unpack_index is None:
        return None
    return MappingKeysShape(
        keys=keys,
        keys_jump_index=keys_jump_index,
        inner_miss_index=inner_miss_index,
        unpack_index=unpack_index,
    )


def read_mapping_keys(
    instructions: list[Instruction],
    keys_index: int,
) -> list[ast.expr] | None:
    if keys_index >= len(instructions):
        return None
    return mapping_match_keys(instructions[keys_index])


def read_unpack_sequence_index(
    instructions: list[Instruction],
    start_index: int,
    item_count: int,
    end_index: int,
) -> int | None:
    unpack_index = skip_match_prefix(instructions, start_index, end_index)
    if unpack_index >= end_index:
        return None
    if instructions[unpack_index].opname != "UNPACK_SEQUENCE":
        return None
    if int(instructions[unpack_index].arg or 0) != item_count:
        return None
    return unpack_index


def read_mapping_keys_test_indexes(
    instructions: list[Instruction],
    keys_index: int,
    end_index: int,
) -> tuple[int, int, int] | None:
    match_keys_index = skip_match_prefix(instructions, keys_index + 1, end_index)
    copy_index = skip_match_prefix(instructions, match_keys_index + 1, end_index)
    keys_jump_index = skip_match_prefix(instructions, copy_index + 1, end_index)
    if (
        keys_index >= end_index
        or match_keys_index >= end_index
        or copy_index >= end_index
        or keys_jump_index >= end_index
    ):
        return None
    return match_keys_index, copy_index, keys_jump_index


def read_mapping_capture_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    keys: list[ast.expr],
    unpack_index: int,
    inner_miss_index: int,
) -> MappingCaptureShape | None:
    captures = read_mapping_value_capture_patterns(
        instructions,
        offset_to_index,
        keys,
        unpack_index,
        inner_miss_index,
    )
    rest_name = None
    if captures is None:
        rest_captures = read_mapping_rest_capture_patterns(
            instructions, unpack_index + 1, len(keys), inner_miss_index
        )
        if rest_captures is None:
            return None
        patterns, rest_name, body_start_index = rest_captures
        return MappingCaptureShape(patterns, rest_name, body_start_index)

    patterns, body_start_index = captures
    body_start_index = skip_match_success_cleanup(
        instructions, body_start_index, inner_miss_index
    )
    return MappingCaptureShape(patterns, rest_name, body_start_index)


def read_mapping_value_capture_patterns(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    keys: list[ast.expr],
    unpack_index: int,
    inner_miss_index: int,
) -> tuple[list[ast.pattern], int] | None:
    if not keys:
        rest_captures = read_mapping_rest_capture_patterns(
            instructions, unpack_index + 1, len(keys), inner_miss_index
        )
        if rest_captures is not None:
            return None

    return read_sequence_patterns(
        instructions,
        offset_to_index,
        unpack_index + 1,
        len(keys),
        inner_miss_index,
    )


def is_mapping_key_tuple(instructions: list[Instruction], index: int) -> bool:
    if index >= len(instructions):
        return False
    instruction = instructions[index]
    return instruction.opname == "LOAD_CONST" and isinstance(instruction.argval, tuple)


def read_class_match_test(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> SimpleMatchTest | None:
    shape = read_class_match_shape(
        instructions,
        offset_to_index,
        cursor,
        end_index,
    )
    if shape is None:
        return None

    captures = read_class_match_patterns(
        instructions,
        shape.unpack_index + 1,
        shape.capture_count,
        shape.miss_index,
    )
    if captures is None:
        return None
    patterns, body_start_index = captures
    return SimpleMatchTest(
        pattern=ast.MatchClass(
            cls=shape.class_expr,
            patterns=patterns[: shape.positional_count],
            kwd_attrs=shape.keyword_attrs,
            kwd_patterns=patterns[shape.positional_count :],
        ),
        jump_index=shape.jump_index,
        miss_index=shape.miss_index,
        body_start_index=body_start_index,
    )


def read_class_match_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> ClassMatchShape | None:
    class_expr = class_match_expr(instructions[cursor])
    indexes = read_class_match_indexes(instructions, cursor + 1, end_index)
    if class_expr is None or indexes is None:
        return None
    keys_index, match_class_index, copy_index, jump_index = indexes
    if not is_class_match_test(
        instructions,
        keys_index,
        match_class_index,
        copy_index,
        jump_index,
    ):
        return None

    keyword_attrs = class_match_keyword_attrs(instructions[keys_index])
    positional_count = int(instructions[match_class_index].arg or 0)
    miss_index = offset_to_index.get(int(instructions[jump_index].argval))
    if keyword_attrs is None or miss_index is None:
        return None

    capture_count = positional_count + len(keyword_attrs)
    unpack_index = read_unpack_sequence_index(
        instructions,
        jump_index + 1,
        capture_count,
        end_index,
    )
    if unpack_index is None:
        return None
    return ClassMatchShape(
        class_expr=class_expr,
        keyword_attrs=keyword_attrs,
        positional_count=positional_count,
        capture_count=capture_count,
        jump_index=jump_index,
        miss_index=miss_index,
        unpack_index=unpack_index,
    )


def read_class_match_indexes(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> tuple[int, int, int, int] | None:
    keys_index = skip_match_prefix(instructions, start_index, end_index)
    match_class_index = skip_match_prefix(instructions, keys_index + 1, end_index)
    copy_index = skip_match_prefix(instructions, match_class_index + 1, end_index)
    jump_index = skip_match_prefix(instructions, copy_index + 1, end_index)
    if (
        keys_index >= end_index
        or match_class_index >= end_index
        or copy_index >= end_index
        or jump_index >= end_index
    ):
        return None
    return keys_index, match_class_index, copy_index, jump_index


def class_match_expr(instruction: Instruction) -> ast.expr | None:
    if instruction.opname not in {"LOAD_DEREF", "LOAD_GLOBAL", "LOAD_NAME"}:
        return None
    return ast.Name(id=safe_identifier(str(instruction.argval)), ctx=ast.Load())


def class_match_keyword_attrs(instruction: Instruction) -> list[str] | None:
    if instruction.opname != "LOAD_CONST":
        return None
    if not isinstance(instruction.argval, tuple):
        return None

    attrs: list[str] = []
    for value in instruction.argval:
        if not isinstance(value, str):
            return None
        attrs.append(safe_identifier(value))
    return attrs


def is_class_match_test(
    instructions: list[Instruction],
    keys_index: int,
    match_class_index: int,
    copy_index: int,
    jump_index: int,
) -> bool:
    if instructions[keys_index].opname != "LOAD_CONST":
        return False
    if instructions[match_class_index].opname != "MATCH_CLASS":
        return False
    copy = instructions[copy_index]
    if copy.opname != "COPY" or int(copy.arg or 0) != 1:
        return False
    return is_mapping_keys_miss_jump(instructions[jump_index])


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


def mapping_match_keys(instruction: Instruction) -> list[ast.expr] | None:
    if instruction.opname != "LOAD_CONST":
        return None
    if not isinstance(instruction.argval, tuple):
        return None
    return [ast.Constant(value=value) for value in instruction.argval]


def is_mapping_keys_test(
    instructions: list[Instruction],
    match_keys_index: int,
    copy_index: int,
    jump_index: int,
) -> bool:
    if instructions[match_keys_index].opname != "MATCH_KEYS":
        return False
    copy = instructions[copy_index]
    if copy.opname != "COPY" or int(copy.arg or 0) != 1:
        return False
    return is_mapping_keys_miss_jump(instructions[jump_index])


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
    patterns: list[ast.pattern] = []
    for _index in range(count):
        cursor = skip_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None

        capture = sequence_capture_pattern(instructions[cursor])
        if capture is not None:
            patterns.append(capture)
            cursor += 1
            continue

        nested = read_nested_sequence_pattern(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if nested is None:
            return None
        pattern, cursor = nested
        patterns.append(pattern)
    return patterns, skip_match_prefix(instructions, cursor, end_index)


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


def read_class_match_patterns(
    instructions: list[Instruction],
    cursor: int,
    count: int,
    miss_index: int,
) -> tuple[list[ast.pattern], int] | None:
    patterns: list[ast.pattern] = []
    for _index in range(count):
        cursor = skip_match_prefix(instructions, cursor, miss_index)
        if cursor >= miss_index:
            return None

        capture = sequence_capture_pattern(instructions[cursor])
        if capture is not None:
            patterns.append(capture)
            cursor += 1
            continue

        value_pattern = read_class_value_pattern(instructions, cursor, miss_index)
        if value_pattern is None:
            return None
        pattern, cursor = value_pattern
        patterns.append(pattern)
    return patterns, skip_match_prefix(instructions, cursor, miss_index)


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


def read_mapping_rest_capture_patterns(
    instructions: list[Instruction],
    cursor: int,
    key_count: int,
    end_index: int,
) -> tuple[list[ast.pattern], str, int] | None:
    cursor = skip_mapping_rest_cleanup(instructions, cursor, end_index)
    rest_name = rest_capture_name(instructions, cursor, end_index)
    if rest_name is None:
        return None

    cursor += 1
    patterns: list[ast.pattern] = []
    for _index in range(key_count):
        cursor = skip_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return None
        pattern = sequence_capture_pattern(instructions[cursor])
        if pattern is None:
            return None
        patterns.append(pattern)
        cursor += 1
    return patterns, rest_name, skip_match_prefix(instructions, cursor, end_index)


def skip_mapping_rest_cleanup(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> int:
    saw_cleanup = False
    for _index in range(64):
        cursor = skip_match_prefix(instructions, cursor, end_index)
        if cursor >= end_index:
            return cursor
        if instructions[cursor].opname not in MAPPING_REST_CLEANUP_OPS:
            return cursor if saw_cleanup else end_index
        saw_cleanup = True
        cursor += 1
    return end_index


def rest_capture_name(
    instructions: list[Instruction],
    cursor: int,
    end_index: int,
) -> str | None:
    cursor = skip_match_prefix(instructions, cursor, end_index)
    if cursor >= end_index:
        return None
    instruction = instructions[cursor]
    if instruction.opname not in {
        "STORE_DEREF",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_NAME",
    }:
        return None
    return safe_identifier(str(instruction.argval))


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
    body_end_index = find_terminal_body_end(instructions, body_start_index, end_index)
    if body_end_index is None:
        return None
    return SimpleMatchCaseRegion(
        pattern=ast.MatchAs(pattern=None, name=None),
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


def is_mapping_keys_miss_jump(instruction: Instruction) -> bool:
    opname = normalized_opcode_name(instruction.opname)
    return "POP_JUMP" in opname and "IF_NONE" in opname


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


WITH_CLEANUP_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP"}
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
    trailing_return_index = None
    if (
        cursor < end_index
        and instructions[cursor].opname == "SWAP"
        and instructions[cursor].arg == 2
    ):
        returns_value = True
        cursor += 1

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
    cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor >= end_index or instructions[cursor].opname != "POP_TOP":
        return None
    cursor += 1

    cursor = skip_with_cleanup_prefix(instructions, cursor, end_index)
    if cursor < end_index and instructions[cursor].opname in {"JUMP", "JUMP_FORWARD"}:
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
    trailing_return_index = None
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


def is_none_load(instruction: Instruction) -> bool:
    if instruction.opname == "LOAD_CONST":
        return instruction.argval is None
    if instruction.opname == "RETURN_CONST":
        return instruction.argval is None
    return False


def is_none_return(instruction: Instruction) -> bool:
    if instruction.opname == "RETURN_CONST":
        return instruction.argval is None
    return False
