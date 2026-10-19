import ast
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from pyc2py.astree import safe_identifier
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import (
    FORWARD_JUMPS,
    normalized_opcode_name,
)

STORES = frozenset({"STORE_DEREF", "STORE_FAST", "STORE_GLOBAL", "STORE_NAME"})
NAME_LOADS = frozenset({"LOAD_DEREF", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME"})
SKIPPED = frozenset({"CACHE", "EXTENDED_ARG", "NOP", "TO_BOOL"})


@dataclass(eq=False)
class Part:
    kind: str = ""
    pattern: ast.pattern | None = None
    length: int | None = None
    at_least: bool = False
    items: list[Part] = field(default_factory=list["Part"])
    keys: list[ast.expr] = field(default_factory=list[ast.expr])
    rest: str | None = None
    class_expr: ast.expr | None = None
    positional: int = 0
    kwd_attrs: list[str] = field(default_factory=list[str])
    subscripts: dict[int, Part] = field(default_factory=dict[int, "Part"])
    alias: Part | None = None
    star: bool = False
    alternatives: list[Part] = field(default_factory=list["Part"])
    binding: Part | None = None


@dataclass(eq=False)
class Held:
    kind: str
    part: Part | None = None
    value: object = None
    expr: ast.expr | None = None


Stack = list[Part | Held]


@dataclass(frozen=True, slots=True)
class SimpleMatchTest:
    pattern: ast.pattern
    jump_index: int
    miss_index: int
    body_start_index: int | None = None
    body_end_index: int | None = None


Fail = tuple[int, int, int]


def read_nested_match(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
    in_match: bool = False,
) -> SimpleMatchTest | None:
    # in_match: something already shows a match statement here, such as a
    # copied subject, without which a test of the subject is an if
    root = Part()
    fails: list[Fail] = []
    index = run(instructions, offset_to_index, [root], cursor, end_index, fails)
    pattern = None if index is None else finished_pattern(root)
    if index is None or pattern is None or not fails:
        return None

    return read_fail_chain(instructions, fails, pattern, index, in_match)


def run(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    stack: Stack,
    index: int,
    end_index: int,
    fails: list[Fail],
    until_join: bool = False,
) -> int | None:
    while stack and index < end_index:
        instruction = instructions[index]
        if normalized_opcode_name(instruction.opname) in SKIPPED:
            index += 1
            continue

        if (
            until_join
            and only_captures(stack)
            and forward_jump_target(instructions, offset_to_index, index)
        ):
            return index

        if is_fail_jump(instruction):
            target = offset_to_index.get(int(instruction.argval))
            if target is None or not fail_test(stack, instruction):
                return None

            fails.append((index, target, len(stack)))
        elif is_alternatives_start(stack, instruction):
            after = read_alternatives(
                instructions, offset_to_index, stack, index, end_index, fails
            )
            if after is None:
                if not step(stack, instruction):
                    return None
            else:
                index = after
                continue
        elif not step(stack, instruction):
            return None

        index += 1

    return None if stack else index


def only_captures(stack: Stack) -> bool:
    return all(isinstance(item, Part) and is_open(item) for item in stack)


def is_alternatives_start(stack: Stack, instruction: Instruction) -> bool:
    top = stack[-1] if stack else None
    return (
        normalized_opcode_name(instruction.opname) == "COPY"
        and instruction.arg == 1
        and isinstance(top, Part)
        and is_open(top)
    )


Alternatives = tuple[list[Part], list[Part], int, int]


def read_alternatives(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    stack: Stack,
    index: int,
    end_index: int,
    fails: list[Fail],
) -> int | None:
    read = read_each_alternative(instructions, offset_to_index, stack, index, end_index)
    subject = stack[-1] if stack else None
    if read is None or not isinstance(subject, Part):
        return None

    roots, captures, success, index = read
    if instructions[index].opname != "POP_TOP":
        return None

    miss_target = forward_jump_target(instructions, offset_to_index, index + 1)

    if captures:
        subject.kind = "or"
        subject.alternatives = roots

        if miss_target is not None:
            fails.append((index + 1, miss_target, len(stack) - 1))
        else:
            fails.append((index, index, len(stack)))

        stack.extend(captures)
        return success

    patterns = [finished_pattern(root) for root in roots]
    if success != index + 2 or miss_target is None or None in patterns:
        return None
    if instructions[success].opname != "POP_TOP":
        return None

    stack.pop()
    subject.pattern = ast.MatchOr(
        patterns=[pattern for pattern in patterns if pattern is not None]
    )
    fails.append((index + 1, miss_target, len(stack)))
    return success + 1


def read_each_alternative(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    stack: Stack,
    index: int,
    end_index: int,
) -> Alternatives | None:
    roots: list[Part] = []
    captures: list[Part] | None = None
    success: int | None = None

    while index < end_index and is_alternatives_start(stack, instructions[index]):
        part = Part()
        own: list[Fail] = []
        left: Stack = [part]
        leave = run(
            instructions, offset_to_index, left, index + 1, end_index, own, True
        )
        target = (
            None
            if leave is None
            else forward_jump_target(instructions, offset_to_index, leave)
        )
        if leave is None or target is None or not own:
            return None
        if success is not None and target != success:
            return None

        found = [item for item in left if isinstance(item, Part)]
        if captures is not None and len(found) != len(captures):
            return None

        for mine, first in zip(found, captures or [], strict=False):
            mine.binding = first

        captures = found if captures is None else captures
        next_index = after_alternative(instructions, own, leave)
        if next_index is None:
            return None

        success = target
        roots.append(part)
        index = next_index

    if len(roots) < 2 or success is None or captures is None:
        return None

    return roots, captures, success, index


def after_alternative(
    instructions: list[Instruction], own: list[Fail], leave: int
) -> int | None:
    ends = {target + depth for _jump, target, depth in own}
    if len(ends) != 1:
        return None

    next_index = ends.pop()
    if next_index <= leave or any(
        item.opname != "POP_TOP" for item in instructions[leave + 1 : next_index]
    ):
        return None

    return next_index


def forward_jump_target(
    instructions: list[Instruction], offset_to_index: dict[int, int], index: int
) -> int | None:
    instruction = instructions[index]
    if normalized_opcode_name(instruction.opname) not in FORWARD_JUMPS:
        return None

    target = offset_to_index.get(int(instruction.argval))
    return target if target is not None and target > index else None


def read_fail_chain(
    instructions: list[Instruction],
    fails: list[Fail],
    pattern: ast.pattern,
    body_start_index: int,
    in_match: bool,
) -> SimpleMatchTest | None:
    chain_end = fails[0][1] + fails[0][2]

    for _jump, target, depth in fails:
        if target + depth != chain_end or target < body_start_index:
            return None
        if any(item.opname != "POP_TOP" for item in instructions[target:chain_end]):
            return None

    first = min(target for _jump, target, _depth in fails)
    miss_index = chain_end - 1
    if miss_index < first and all(
        (in_match and depth == 0) or consumes_its_test(instructions, jump)
        for jump, _, depth in fails
    ):
        # a test of the subject itself leaves nothing to drop when it misses
        miss_index = first
    if miss_index < first:
        return None

    return SimpleMatchTest(
        pattern=pattern,
        jump_index=fails[0][0],
        miss_index=miss_index,
        body_start_index=body_start_index,
        body_end_index=first if first < miss_index else None,
    )


def consumes_its_test(instructions: list[Instruction], jump: int) -> bool:
    # an or-pattern's miss, or a 3.16 class test, leaves nothing to drop
    return forward_jump_target_name(instructions[jump]) or is_class_isinstance_test(
        instructions[jump - 1]
    )


def forward_jump_target_name(instruction: Instruction) -> bool:
    return normalized_opcode_name(instruction.opname) in FORWARD_JUMPS


def is_class_isinstance_test(instruction: Instruction) -> bool:
    return instruction.argrepr == "INTRINSIC_MATCH_CLASS_ISINSTANCE"


def is_fail_jump(instruction: Instruction) -> bool:
    name = normalized_opcode_name(instruction.opname)
    return name.startswith("POP_JUMP") and name.endswith(
        ("IF_FALSE", "IF_NONE", "IF_NOT_NONE")
    )


def fail_test(stack: Stack, instruction: Instruction) -> bool:
    if not stack:
        return False

    top = stack.pop()
    name = normalized_opcode_name(instruction.opname)
    if name.endswith("IF_NOT_NONE"):
        # `case None`
        if not isinstance(top, Part) or not is_open(top):
            return False

        top.pattern = ast.MatchSingleton(value=None)
        return True

    if name.endswith("IF_NONE"):
        return isinstance(top, Held) and top.kind == "copy"

    return isinstance(top, Held) and top.kind == "test"


def is_open(part: Part) -> bool:
    return not part.kind and part.pattern is None and part.alias is None


def step(stack: Stack, instruction: Instruction) -> bool:
    name = normalized_opcode_name(instruction.opname)
    if name in NAME_LOADS:
        name = "LOAD_NAME"
    elif name in STORES:
        name = "STORE_NAME"

    handler = STEPS.get(name)
    return handler is not None and handler(stack, instruction)


def match_type(stack: Stack, instruction: Instruction) -> bool:
    top = stack[-1] if stack else None
    if not isinstance(top, Part) or not is_open(top):
        return False

    is_sequence = normalized_opcode_name(instruction.opname) == "MATCH_SEQUENCE"
    top.kind = "sequence" if is_sequence else "mapping"
    stack.append(Held("test"))
    return True


def get_len(stack: Stack, instruction: Instruction) -> bool:
    top = stack[-1] if stack else None
    if isinstance(top, Held) and top.kind == "indexed":
        # len(subject) - k
        stack.append(Held("end", part=top.part))
        return True

    if not isinstance(top, Part) or top.kind not in {"sequence", "mapping"}:
        return False

    stack.append(Held("length", part=top))
    return True


def load_const(stack: Stack, instruction: Instruction) -> bool:
    stack.append(Held("value", value=instruction.argval))
    return True


def load_name(stack: Stack, instruction: Instruction) -> bool:
    expr = ast.Name(id=safe_identifier(str(instruction.argval)), ctx=ast.Load())
    stack.append(Held("value", expr=expr))
    return True


def load_attr(stack: Stack, instruction: Instruction) -> bool:
    top = stack[-1] if stack else None
    if not isinstance(top, Held) or top.expr is None:
        return False

    top.expr = ast.Attribute(
        value=top.expr, attr=safe_identifier(str(instruction.argval)), ctx=ast.Load()
    )
    return True


def build_key_tuple(stack: Stack, instruction: Instruction) -> bool:
    count = int(instruction.arg or 0)
    if count < 1 or count > len(stack):
        return False

    keys: list[ast.expr] = []

    for item in stack[-count:]:
        key = value_expr(item) if isinstance(item, Held) else None
        if key is None:
            return False

        keys.append(key)

    del stack[-count:]
    stack.append(Held("key tuple", value=tuple(keys)))
    return True


def value_expr(held: Held) -> ast.expr | None:
    if held.kind != "value":
        return None
    if held.expr is not None:
        return held.expr
    if isinstance(held.value, str | bytes | bool | int | float | complex | None):
        return ast.Constant(value=held.value)

    return None


def compare(stack: Stack, instruction: Instruction) -> bool:
    if len(stack) < 2:
        return False

    subject, value = stack[-2], stack[-1]
    if not isinstance(value, Held) or value.kind != "value":
        return False

    if isinstance(subject, Held) and subject.kind == "length":
        matched = compare_length(subject, value, instruction)
    elif isinstance(subject, Part) and is_open(subject):
        matched = compare_value(subject, value, instruction)
    else:
        return False

    if not matched:
        return False

    del stack[-2:]
    stack.append(Held("test"))
    return True


def compare_length(length: Held, value: Held, instruction: Instruction) -> bool:
    part, count = length.part, value.value
    if part is None or part.length is not None or instruction.opname != "COMPARE_OP":
        return False
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return False
    if instruction.argval not in {"==", ">="}:
        return False

    part.length = count
    part.at_least = instruction.argval == ">="
    return True


def compare_value(subject: Part, value: Held, instruction: Instruction) -> bool:
    if normalized_opcode_name(instruction.opname) == "IS_OP":
        singleton = value.value

        if instruction.arg or value.expr is not None:
            return False
        if not (singleton is None or singleton is True or singleton is False):
            return False

        subject.pattern = ast.MatchSingleton(value=singleton)
        return True

    expr = value_expr(value)
    if instruction.argval != "==" or expr is None or isinstance(expr, ast.Name):
        return False

    subject.pattern = ast.MatchValue(value=expr)
    return True


def swap(stack: Stack, instruction: Instruction) -> bool:
    depth = int(instruction.arg or 0)
    if depth < 2 or depth > len(stack):
        return False

    stack[-1], stack[-depth] = stack[-depth], stack[-1]
    return True


def copy(stack: Stack, instruction: Instruction) -> bool:
    depth = int(instruction.arg or 0)
    if depth < 1 or depth > len(stack):
        return False

    source = stack[-depth]

    if depth > 1:
        if not isinstance(source, Held) or source.kind != "rest":
            return False

        stack.append(source)
        return True

    if isinstance(source, Part):
        if is_open(source):
            # `pattern as name`
            source.alias = Part()
            stack.append(source.alias)
            return True

        if source.kind == "sequence" and source.length is not None and not source.items:
            # `[a, *_, b]`
            stack.append(Held("indexed", part=source))
            return True

        return False

    if source.kind in {"values", "class"}:
        stack.append(Held("copy", part=source.part))
        return True

    return False


def subscript(stack: Stack, instruction: Instruction) -> bool:
    if normalized_opcode_name(instruction.opname) == "BINARY_OP":
        if instruction.argrepr == "-":
            return count_from_end(stack)
        if instruction.argrepr != "[]":
            return False

    if len(stack) < 2:
        return False

    holder, position = stack[-2], stack[-1]
    if not isinstance(holder, Held) or holder.kind != "indexed":
        return False
    if not isinstance(position, Held) or position.kind != "value":
        return False

    sequence = holder.part
    index = position.value
    if sequence is None or not isinstance(index, int) or index in sequence.subscripts:
        return False

    item = Part()
    sequence.subscripts[index] = item
    del stack[-2:]
    stack.append(item)
    return True


def count_from_end(stack: Stack) -> bool:
    if len(stack) < 2:
        return False

    end, count = stack[-2], stack[-1]
    if not isinstance(end, Held) or end.kind != "end":
        return False
    if not isinstance(count, Held) or count.kind != "value":
        return False
    if not isinstance(count.value, int) or isinstance(count.value, bool):
        return False

    del stack[-2:]
    stack.append(Held("value", value=-count.value))
    return True


def match_keys(stack: Stack, instruction: Instruction) -> bool:
    if len(stack) < 2:
        return False

    subject, keys = stack[-2], stack[-1]
    if not isinstance(subject, Part) or subject.kind != "mapping" or subject.keys:
        return False
    if not isinstance(keys, Held) or keys.kind not in {"value", "key tuple"}:
        return False

    held = keys.value
    if not isinstance(held, tuple):
        return False

    items = cast("tuple[object, ...]", held)
    found = [
        item if isinstance(item, ast.expr) else value_expr(Held("value", value=item))
        for item in items
    ]
    if None in found:
        return False

    subject.keys = [item for item in found if item is not None]
    stack[-1] = Held("keys", part=subject)
    stack.append(Held("values", part=subject))
    return True


def match_class(stack: Stack, instruction: Instruction) -> bool:
    positional = int(instruction.arg or 0)

    if len(stack) < 3:
        return False

    subject, cls, names = stack[-3:]
    if not isinstance(subject, Part) or not is_open(subject):
        return False
    if not isinstance(cls, Held) or cls.expr is None:
        return False
    if not isinstance(names, Held):
        return False

    held = names.value
    if not isinstance(held, tuple):
        return False

    attrs = cast("tuple[object, ...]", held)
    if not all(isinstance(item, str) for item in attrs):
        return False

    subject.kind = "class"
    subject.class_expr = cls.expr
    subject.positional = positional
    subject.kwd_attrs = [safe_identifier(str(item)) for item in attrs]
    del stack[-3:]
    stack.append(Held("class", part=subject))
    return True


def match_class_isinstance(stack: Stack, instruction: Instruction) -> bool:
    if not is_class_isinstance_test(instruction) or len(stack) < 2:
        return False

    subject, cls = stack[-2:]
    if not isinstance(subject, Part) or not is_open(subject):
        return False
    if not isinstance(cls, Held) or cls.expr is None:
        return False

    subject.kind = "class"
    subject.class_expr = cls.expr
    subject.positional = 0
    subject.kwd_attrs = []
    del stack[-2:]
    stack.append(Held("test"))
    return True


def unpack(stack: Stack, instruction: Instruction) -> bool:
    name = normalized_opcode_name(instruction.opname)
    arg = int(instruction.arg or 0)

    if not stack:
        return False

    top = stack[-1]
    if isinstance(top, Held) and top.kind == "keys" and name == "UNPACK_SEQUENCE":
        part = top.part
        if part is None or arg != len(part.keys):
            return False

        stack.pop()
        stack.extend(Held("key") for _ in range(arg))
        return True

    items = unpacked_items(top, name, arg)
    if items is None:
        return False

    owner = top if isinstance(top, Part) else top.part
    if owner is None:
        return False

    owner.items = items
    stack.pop()
    stack.extend(reversed(items))
    return True


def unpacked_items(top: Part | Held, name: str, arg: int) -> list[Part] | None:
    if isinstance(top, Part):
        if top.kind != "sequence" or top.length is None or top.items:
            return None

        if name == "UNPACK_SEQUENCE":
            if top.at_least or arg != top.length:
                return None

            return [Part() for _ in range(arg)]

        before, after = arg & 0xFF, arg >> 8
        if not top.at_least or before + after != top.length:
            return None

        return [
            *(Part() for _ in range(before)),
            Part(star=True),
            *(Part() for _ in range(after)),
        ]

    if name != "UNPACK_SEQUENCE" or top.part is None:
        return None
    if top.kind == "values" and arg == len(top.part.keys):
        return [Part() for _ in range(arg)]
    if top.kind == "class" and arg == top.part.positional + len(top.part.kwd_attrs):
        return [Part() for _ in range(arg)]

    return None


def store(stack: Stack, instruction: Instruction) -> bool:
    name = safe_identifier(str(instruction.argval))

    if not stack:
        return False

    top = stack.pop()
    if isinstance(top, Held):
        if top.kind != "rest" or top.part is None or top.part.rest is not None:
            return False

        top.part.rest = name
        return True

    if top.star and is_open(top):
        top.pattern = ast.MatchStar(name=name)
        return True

    if top.alias is not None and top.pattern is None and not top.kind:
        inner = finished_pattern(top.alias)
        if inner is None:
            return False

        top.pattern = ast.MatchAs(pattern=inner, name=name)
        return True

    if not is_open(top):
        return False

    top.pattern = ast.MatchAs(pattern=None, name=name)
    return True


def pop(stack: Stack, instruction: Instruction) -> bool:
    if not stack:
        return False

    top = stack.pop()
    if isinstance(top, Held):
        return top.kind == "keys"

    if top.star and is_open(top):
        top.pattern = ast.MatchStar(name=None)
        return True

    if is_open(top):
        top.pattern = ast.MatchAs(pattern=None, name=None)
        return True

    return top.kind in {"mapping", "or"} or (top.kind == "sequence" and not top.items)


def copy_rest(stack: Stack, instruction: Instruction) -> bool:
    depth = int(instruction.arg or 0)
    if len(stack) < depth + 1:
        return False

    subject, target = stack[-1], stack[-1 - depth]
    if not isinstance(subject, Part) or subject.kind != "mapping":
        return False
    if not isinstance(target, Held) or target.kind != "rest" or target.part:
        return False

    target.part = subject
    stack.pop()
    return True


def delete_rest_key(stack: Stack, instruction: Instruction) -> bool:
    if len(stack) < 2:
        return False

    rest, key = stack[-2], stack[-1]
    if not isinstance(rest, Held) or rest.kind != "rest":
        return False
    if not isinstance(key, Held) or key.kind != "key":
        return False

    del stack[-2:]
    return True


def build_rest(stack: Stack, instruction: Instruction) -> bool:
    if instruction.arg:
        return False

    stack.append(Held("rest"))
    return True


STEPS: dict[str, Callable[[Stack, Instruction], bool]] = {
    "MATCH_SEQUENCE": match_type,
    "MATCH_MAPPING": match_type,
    "GET_LEN": get_len,
    "LOAD_CONST": load_const,
    "LOAD_NAME": load_name,
    "LOAD_ATTR": load_attr,
    "BUILD_TUPLE": build_key_tuple,
    "COMPARE_OP": compare,
    "IS_OP": compare,
    "COPY": copy,
    "SWAP": swap,
    "BINARY_SUBSCR": subscript,
    "BINARY_OP": subscript,
    "MATCH_KEYS": match_keys,
    "MATCH_CLASS": match_class,
    "CALL_INTRINSIC_2": match_class_isinstance,
    "UNPACK_SEQUENCE": unpack,
    "UNPACK_EX": unpack,
    "STORE_NAME": store,
    "POP_TOP": pop,
    "BUILD_MAP": build_rest,
    "DICT_UPDATE": copy_rest,
    "DELETE_SUBSCR": delete_rest_key,
}


def finished_pattern(part: Part) -> ast.pattern | None:
    if part.pattern is not None:
        return part.pattern

    if part.binding is not None:
        bound = finished_pattern(part.binding)
        return None if bound is None else deepcopy(bound)

    if part.kind == "or":
        patterns = [finished_pattern(root) for root in part.alternatives]
        if None in patterns:
            return None

        return ast.MatchOr(patterns=[item for item in patterns if item is not None])

    if part.kind == "sequence":
        return sequence_pattern(part)

    if part.kind == "mapping":
        patterns = [finished_pattern(item) for item in part.items]
        if len(patterns) != len(part.keys) or None in patterns:
            return None

        return ast.MatchMapping(
            keys=part.keys,
            patterns=[item for item in patterns if item is not None],
            rest=part.rest,
        )

    if part.kind == "class" and part.class_expr is not None:
        patterns = [finished_pattern(item) for item in part.items]
        if None in patterns:
            return None

        found = [item for item in patterns if item is not None]
        return ast.MatchClass(
            cls=part.class_expr,
            patterns=found[: part.positional],
            kwd_attrs=part.kwd_attrs,
            kwd_patterns=found[part.positional :],
        )

    return None


def sequence_pattern(part: Part) -> ast.pattern | None:
    if part.items:
        patterns = [finished_pattern(item) for item in part.items]
        if None in patterns:
            return None

        return ast.MatchSequence(patterns=[item for item in patterns if item])

    if part.length is None:
        # `[*_]`
        return ast.MatchSequence(patterns=[ast.MatchStar(name=None)])

    if not part.at_least:
        # `[_, _]`
        if part.subscripts:
            return None

        return ast.MatchSequence(patterns=[wildcard() for _ in range(part.length)])

    # `[a, *_, b]`
    after = max((-index for index in part.subscripts if index < 0), default=0)
    before = part.length - after
    if before < 0 or any(index >= before for index in part.subscripts):
        return None

    found: list[ast.pattern] = []

    for index in [*range(before), None, *range(-after, 0)]:
        if index is None:
            found.append(ast.MatchStar(name=None))
            continue

        item = part.subscripts.get(index)
        pattern = wildcard() if item is None else finished_pattern(item)
        if pattern is None:
            return None

        found.append(pattern)

    return ast.MatchSequence(patterns=found)


def wildcard() -> ast.MatchAs:
    return ast.MatchAs(pattern=None, name=None)
