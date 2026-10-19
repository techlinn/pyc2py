import ast
from dataclasses import dataclass

from pyc2py.bytecode.instruction import Instruction
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.flow import is_terminal_op
from pyc2py.types import has_fact

LEAVING_JUMPS = frozenset({"JUMP_FORWARD", "JUMP_ABSOLUTE"})
# 3.10 back to the loop head
BODY_LEAVING_JUMPS = LEAVING_JUMPS | {"JUMP_BACKWARD"}
NAME_STORES = frozenset({"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"})
BLOCK_SETUPS = frozenset({"SETUP_EXCEPT", "SETUP_FINALLY", "SETUP_WITH"})
# END_FINALLY, RERAISE from 3.9
CHAIN_ENDS = frozenset({"END_FINALLY", "RERAISE"})
NAMED_BODY_EXITS = frozenset({"POP_EXCEPT", "LOAD_CONST", "BEGIN_FINALLY"})
# a copied tail: straight-line code up to a return or raise
TAIL_ENDS = frozenset({"RETURN_VALUE", "RAISE_VARARGS"})
NOT_IN_TAIL = frozenset({"POP_BLOCK", "POP_EXCEPT", "RERAISE", "FOR_ITER"})


@dataclass(frozen=True, slots=True)
class BlockHandler:
    type_range: tuple[int, int] | None
    name: str | None
    body_range: tuple[int, int]
    end_target: int | None


@dataclass(frozen=True, slots=True)
class BlockTry:
    body_range: tuple[int, int]
    handlers: tuple[BlockHandler, ...]
    else_range: tuple[int, int] | None
    end_index: int


def try_translate_block_except(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    shape = read_block_try(
        instructions, offset_to_index, cursor, end_index, decompiler.version
    )
    if shape is None or not append_try(decompiler, instructions, shape):
        return None

    return shape.end_index


def append_try(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    shape: BlockTry,
) -> bool:
    types: list[ast.expr | None] = []

    for handler in shape.handlers:
        if handler.type_range is None:
            types.append(None)
            continue

        exception_type = decompiler.evaluate_expression_range(
            instructions, *handler.type_range
        )
        if exception_type is None:
            return False

        types.append(exception_type)

    body = decompiler.translate_child_statements(instructions, *shape.body_range)
    handlers = [
        ast.ExceptHandler(
            type=exception_type,
            name=handler.name,
            body=decompiler.translate_child_statements(
                instructions, *handler.body_range
            )
            or [ast.Pass()],
        )
        for handler, exception_type in zip(shape.handlers, types, strict=True)
    ]
    orelse: list[ast.stmt] = []

    if shape.else_range is not None:
        orelse = decompiler.translate_child_statements(instructions, *shape.else_range)

    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=handlers,
            orelse=orelse,
            finalbody=[],
        )
    )
    return True


def handler_start(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> int | None:
    setup = instructions[setup_index]
    if setup.opname not in {"SETUP_EXCEPT", "SETUP_FINALLY"}:
        return None

    handler_index = offset_to_index.get(int(setup.argval))
    if handler_index is None or not setup_index < handler_index < end_index:
        return None
    # from 3.8 SETUP_FINALLY also starts a try/except
    if setup.opname == "SETUP_FINALLY" and not (
        instructions[handler_index].opname == "DUP_TOP"
        or is_bare_handler(instructions, handler_index, end_index, version)
    ):
        return None

    return handler_index


def read_block_try(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> BlockTry | None:
    handler_index = handler_start(
        instructions, offset_to_index, setup_index, end_index, version
    )
    if handler_index is None:
        return None

    protected = protected_body(instructions, setup_index, handler_index)
    if protected is None:
        return None

    body_range, leave = protected
    chain = read_handler_chain(
        instructions, offset_to_index, handler_index, end_index, version
    )
    if chain is None:
        return None

    handlers, chain_end = chain
    after_chain = chain_end + 1

    if leave is not None:
        target = offset_to_index.get(int(instructions[leave].argval))
        # the optimizer may thread the jump on to a loop head
        if (
            target is not None
            and after_chain != target <= end_index
            and instructions[leave].argval > instructions[leave].offset
            and not jumps_to(instructions, after_chain, instructions[leave].argval)
        ):
            return None

    statement_end = handlers_end(instructions, offset_to_index, handlers, end_index)
    else_range = None

    if statement_end is not None and statement_end > after_chain:
        else_range = (after_chain, statement_end)

    return BlockTry(
        body_range=body_range,
        handlers=handlers,
        else_range=else_range,
        end_index=max(after_chain, statement_end or after_chain),
    )


def jumps_to(instructions: list[Instruction], index: int, offset: object) -> bool:
    return (
        index < len(instructions)
        and instructions[index].opname in LEAVING_JUMPS
        and instructions[index].argval == offset
    )


def protected_body(
    instructions: list[Instruction],
    setup_index: int,
    handler_index: int,
) -> tuple[tuple[int, int], int | None] | None:
    last = instructions[handler_index - 1]
    if last.opname in BODY_LEAVING_JUMPS and (
        instructions[handler_index - 2].opname == "POP_BLOCK"
    ):
        return (setup_index + 1, handler_index - 2), handler_index - 1
    if last.opname == "POP_BLOCK":
        return (setup_index + 1, handler_index - 1), None
    if is_terminal_op(last.opname):
        return (setup_index + 1, handler_index), None

    return None


def read_handler_chain(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> tuple[tuple[BlockHandler, ...], int] | None:
    handlers: list[BlockHandler] = []
    index = handler_index
    while index < end_index:
        if instructions[index].opname in CHAIN_ENDS:
            return (tuple(handlers), index) if handlers else None

        pops = exception_pops(instructions, index, end_index, version)
        if pops is not None:
            body_start = index + pops
            chain_end = closing_end_finally(instructions, body_start, end_index)
            if chain_end is None:
                # the optimizer drops END_FINALLY behind a returning handler
                body_end = terminal_block_end(instructions, body_start, end_index)
                if body_end is None:
                    return None

                handlers.append(BlockHandler(None, None, (body_start, body_end), None))
                return tuple(handlers), body_end - 1

            handler = read_handler_rest(
                instructions, (body_start, chain_end), None, None
            )
            if handler is None:
                return None

            handlers.append(handler)
            return tuple(handlers), chain_end

        typed = read_typed_handler(
            instructions, offset_to_index, index, end_index, version
        )
        if typed is None:
            return None

        handler, index = typed
        handlers.append(handler)

    return None


def is_bare_handler(
    instructions: list[Instruction],
    index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> bool:
    return exception_pops(instructions, index, end_index, version) is not None


def exception_items(version: tuple[int, ...] | None) -> int:
    return 2 if has_fact(version, "two_item_exceptions") else 3


def exception_pops(
    instructions: list[Instruction],
    index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> int | None:
    pops = 0
    while index + pops < end_index and instructions[index + pops].opname == "POP_TOP":
        pops += 1

    return pops if pops == exception_items(version) else None


@dataclass(frozen=True, slots=True)
class TypedHandlerHead:
    type_range: tuple[int, int]
    # the name the exception is bound to, if any
    name: str | None
    body_start: int
    next_index: int


def typed_handler_head(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> TypedHandlerHead | None:
    # DUP_TOP, the type, the test, then the exception's items popped or bound
    if instructions[index].opname != "DUP_TOP":
        return None

    match = handler_test(instructions, index + 1, end_index)
    if match is None:
        return None

    type_end, test = match
    next_index = offset_to_index.get(int(instructions[test].argval))
    items = exception_items(version)
    if next_index is None or not test + items < next_index < end_index:
        return None

    first, bound, *rest = instructions[test + 1 : test + 1 + items]
    if first.opname != "POP_TOP" or any(item.opname != "POP_TOP" for item in rest):
        return None

    name = None

    if bound.opname in NAME_STORES:
        name = str(bound.argval)
    elif bound.opname != "POP_TOP":
        return None

    return TypedHandlerHead((index + 1, type_end), name, test + 1 + items, next_index)


def read_typed_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    index: int,
    end_index: int,
    version: tuple[int, ...] | None,
) -> tuple[BlockHandler, int] | None:
    head = typed_handler_head(instructions, offset_to_index, index, end_index, version)
    if head is None:
        return None

    handler = read_handler_rest(
        instructions, (head.body_start, head.next_index), head.type_range, head.name
    )
    return None if handler is None else (handler, head.next_index)


def handler_test(
    instructions: list[Instruction], start: int, end_index: int
) -> tuple[int, int] | None:
    for position in range(start, end_index - 3):
        instruction = instructions[position]
        if instruction.opname == "JUMP_IF_NOT_EXC_MATCH":
            return position, position

        if (
            instruction.opname == "COMPARE_OP"
            and instruction.argval == "exception-match"
        ):
            jump = position + 1
            while instructions[jump].opname == "EXTENDED_ARG":
                jump += 1

            if instructions[jump].opname != "POP_JUMP_IF_FALSE":
                return None

            return position, jump

    return None


def read_handler_rest(
    instructions: list[Instruction],
    span: tuple[int, int],
    type_range: tuple[int, int] | None,
    name: str | None,
) -> BlockHandler | None:
    start, end = span
    if start >= end:
        return None
    if name is not None and instructions[start].opname == "SETUP_FINALLY":
        return read_named_handler(instructions, span, type_range, name)

    body_end, end_target = handler_exit(instructions, start, end)
    return BlockHandler(type_range, name, (start, body_end), end_target)


def read_named_handler(
    instructions: list[Instruction],
    span: tuple[int, int],
    type_range: tuple[int, int] | None,
    name: str,
) -> BlockHandler | None:
    start, end = span
    cleanup = next(
        (
            index
            for index in range(start + 1, end)
            if instructions[index].offset == instructions[start].argval
        ),
        None,
    )
    if cleanup is None or cleanup + 4 > end:
        return None
    if not clears_name(instructions, cleanup, name) or (
        instructions[cleanup + 3].opname not in CHAIN_ENDS
    ):
        return None

    inline = inline_name_cleanup(instructions, start, cleanup, name)
    if inline is not None:
        # 3.9 clears the name on the normal way out too
        exit_start, leave_target = inline

        if cleanup + 4 != end:
            return None

        return BlockHandler(type_range, name, (start + 1, exit_start), leave_target)

    body_end = cleanup
    while body_end - 1 > start and instructions[body_end - 1].opname in (
        NAMED_BODY_EXITS
    ):
        body_end -= 1

    if instructions[body_end - 1].opname == "POP_BLOCK":
        body_end -= 1
    elif not is_terminal_op(instructions[cleanup - 1].opname):
        return None

    after_end, end_target = handler_exit(instructions, cleanup + 4, end)
    if after_end != cleanup + 4:
        return None

    return BlockHandler(type_range, name, (start + 1, body_end), end_target)


def clears_name(instructions: list[Instruction], index: int, name: str) -> bool:
    clear, store, delete = instructions[index : index + 3]
    return (
        clear.opname == "LOAD_CONST"
        and clear.argval is None
        and store.opname in NAME_STORES
        and store.argval == name
        and delete.opname.startswith("DELETE_")
        and delete.argval == name
    )


def inline_name_cleanup(
    instructions: list[Instruction], start: int, cleanup: int, name: str
) -> tuple[int, int] | None:
    exit_start = cleanup - 6
    if exit_start <= start or (
        instructions[cleanup - 1].opname not in BODY_LEAVING_JUMPS
    ):
        return None
    if [item.opname for item in instructions[exit_start : exit_start + 2]] != [
        "POP_BLOCK",
        "POP_EXCEPT",
    ]:
        return None
    if not clears_name(instructions, exit_start + 2, name):
        return None

    return exit_start, int(instructions[cleanup - 1].argval)


def handler_exit(
    instructions: list[Instruction], start: int, end: int
) -> tuple[int, int | None]:
    body_end = end
    end_target = None

    if body_end > start and instructions[body_end - 1].opname in LEAVING_JUMPS:
        end_target = int(instructions[body_end - 1].argval)
        body_end -= 1
        if body_end > start and instructions[body_end - 1].opname == "POP_EXCEPT":
            body_end -= 1

    return body_end, end_target


def closing_end_finally(
    instructions: list[Instruction], start: int, end_index: int
) -> int | None:
    depth = 0

    for index in range(start, end_index):
        opname = instructions[index].opname
        if opname in BLOCK_SETUPS:
            depth += 1
        elif opname in CHAIN_ENDS:
            if depth == 0:
                return index

            depth -= 1

    return None


def handlers_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handlers: tuple[BlockHandler, ...],
    end_index: int,
) -> int | None:
    ends: list[int] = []

    for handler in handlers:
        if handler.end_target is None:
            continue

        target = offset_to_index.get(handler.end_target)
        if target is None or target > end_index or target <= handler.body_range[0]:
            ends.append(end_index)
        else:
            ends.append(target)

    return max(ends, default=None)


def try_translate_copied_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    setup = instructions[cursor]
    final_start = offset_to_index.get(int(setup.argval))
    if setup.opname != "SETUP_FINALLY" or final_start is None:
        return None
    if not cursor < final_start < end_index:
        return None

    final_end = closing_end_finally(instructions, final_start, end_index)
    if final_end is None or instructions[final_end].opname != "RERAISE":
        return None

    exit_start, tail = finally_copy_exit(
        decompiler, instructions, offset_to_index, final_start, final_end
    )
    if exit_start is None or exit_start <= cursor:
        return None

    spans = early_exit_copies(
        instructions, (cursor + 1, exit_start), (final_start, final_end)
    )
    if spans is None:
        return None

    for start, end in spans:
        decompiler.claimed_ranges[start] = end

    body = decompiler.translate_child_statements(instructions, cursor + 1, exit_start)
    final_body = decompiler.translate_child_statements(
        instructions, final_start, final_end
    )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[],
            orelse=[],
            finalbody=final_body or [ast.Pass()],
        )
    )

    if tail is not None:
        decompiler.statements.extend(
            decompiler.translate_child_statements(instructions, *tail)
        )

    return final_end + 1


def early_exit_copies(
    instructions: list[Instruction],
    body: tuple[int, int],
    final: tuple[int, int],
) -> list[tuple[int, int]] | None:
    start, end = body
    final_start, final_end = final
    length = final_end - final_start
    spans: list[tuple[int, int]] = []

    for index in range(start, end):
        if instructions[index].opname != "POP_BLOCK":
            continue
        if not same_instructions(instructions, index + 1, final_start, length):
            continue

        copy_end = index + 1 + length
        if instructions[index].is_jump_target or not runs_into_return(
            instructions, copy_end, end
        ):
            return None

        spans.append((index, copy_end))

    return spans


def runs_into_return(instructions: list[Instruction], start: int, end: int) -> bool:
    for index in range(start, end):
        instruction = instructions[index]
        if instruction.opname == "RETURN_VALUE":
            return True
        if (
            "JUMP" in instruction.opname
            or instruction.opname.startswith("SETUP_")
            or instruction.opname in {"FOR_ITER", "RAISE_VARARGS", "RERAISE"}
        ):
            return False

    return False


# 3.10 copies short exit blocks
COPIED_TAIL_LIMIT = 6


def finally_copy_exit(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    final_start: int,
    final_end: int,
) -> tuple[int | None, tuple[int, int] | None]:
    length = final_end - final_start
    leave = final_start - 1
    if instructions[leave].opname == "JUMP_FORWARD":
        exit_start = leave - length - 1
        if (
            offset_to_index.get(int(instructions[leave].argval)) == final_end + 1
            and exit_start >= 0
            and instructions[exit_start].opname == "POP_BLOCK"
            and same_instructions(instructions, exit_start + 1, final_start, length)
        ):
            return exit_start, None

        return None, None

    if instructions[leave].opname == "RETURN_VALUE":
        exit_start = leave - length - 1
        if (
            exit_start >= 0
            and instructions[exit_start].opname == "POP_BLOCK"
            and same_instructions(instructions, exit_start + 1, final_start, length)
        ):
            return final_start, None

    if decompiler.version is None or decompiler.version[:2] != (3, 10):
        return None, None

    for tail_length in range(1, COPIED_TAIL_LIMIT + 1):
        tail_start = final_start - tail_length
        exit_start = tail_start - length - 1
        if exit_start < 0 or instructions[exit_start].opname != "POP_BLOCK":
            continue
        if not is_tail(instructions, tail_start, final_start):
            continue
        if same_instructions(instructions, exit_start + 1, final_start, length):
            return exit_start, (tail_start, final_start)

    return None, None


def is_tail_instruction(instruction: Instruction) -> bool:
    # walking back from a tail's end, an earlier return or raise ends another one
    return instruction.opname not in TAIL_ENDS and is_tail_step(instruction)


def is_tail_step(instruction: Instruction) -> bool:
    name = instruction.opname
    return not ("JUMP" in name or name.startswith("SETUP_") or name in NOT_IN_TAIL)


def is_tail(instructions: list[Instruction], start: int, end: int) -> bool:
    return (
        start < end
        and instructions[end - 1].opname in TAIL_ENDS
        and all(is_tail_step(item) for item in instructions[start : end - 1])
    )


def same_instructions(
    instructions: list[Instruction], left: int, right: int, length: int
) -> bool:
    # a copy's jumps may lead to other exits than the original's
    return length >= 0 and all(
        first.opname == second.opname
        and ("JUMP" in first.opname or first.argval == second.argval)
        for first, second in zip(
            instructions[left : left + length],
            instructions[right : right + length],
            strict=True,
        )
    )


def try_translate_returning_finally(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    # from 3.9 every way out has its own finally copy
    if decompiler.version is None or decompiler.version >= (3, 9):
        return None

    setup = instructions[cursor]
    final_start = offset_to_index.get(int(setup.argval))
    if setup.opname != "SETUP_FINALLY" or final_start is None:
        return None
    if not cursor + 1 < final_start < end_index:
        return None
    if not is_terminal_op(instructions[final_start - 1].opname):
        return None

    final_end = terminal_block_end(instructions, final_start, end_index)
    if final_end is None:
        return None

    body = decompiler.translate_child_statements(instructions, cursor + 1, final_start)
    final_body = decompiler.translate_child_statements(
        instructions, final_start, final_end
    )
    decompiler.statements.append(
        ast.Try(
            body=body or [ast.Pass()],
            handlers=[],
            orelse=[],
            finalbody=final_body or [ast.Pass()],
        )
    )
    return final_end


def terminal_block_end(
    instructions: list[Instruction], start: int, end_index: int
) -> int | None:
    reach = -1

    for index in range(start, end_index):
        instruction = instructions[index]
        names_target = (
            "JUMP" in instruction.opname
            or instruction.opname.startswith("SETUP_")
            or instruction.opname == "FOR_ITER"
        )
        if names_target and isinstance(instruction.argval, int):
            reach = max(reach, instruction.argval)
        if is_terminal_op(instruction.opname) and instruction.offset >= reach:
            return index + 1

    return None


# the one version whose compiler copies a statement's tail into its ways out
COPIED_TAIL_VERSION = (3, 10)


@dataclass(frozen=True, slots=True)
class CopiedTailTry:
    statement: BlockTry
    # the code after the statement, which 3.10 copies into each way out
    tail_range: tuple[int, int]


def try_translate_copied_tail_except(
    decompiler: DecompilerContext,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    cursor: int,
    end_index: int,
) -> int | None:
    if decompiler.version is None or decompiler.version[:2] != COPIED_TAIL_VERSION:
        return None

    shape = read_copied_tail_try(instructions, offset_to_index, cursor, end_index)
    if shape is None or not append_try(decompiler, instructions, shape.statement):
        return None

    decompiler.statements.extend(
        decompiler.translate_child_statements(instructions, *shape.tail_range)
    )
    return shape.statement.end_index


def tail_behind(
    instructions: list[Instruction], markers: frozenset[str], end: int
) -> tuple[int, int] | None:
    if end < 2 or instructions[end - 1].opname not in TAIL_ENDS:
        return None

    start = end - 1
    while start > 0 and is_tail_instruction(instructions[start - 1]):
        start -= 1

    if start == 0 or instructions[start - 1].opname not in markers:
        return None
    if start == end - 1 and instructions[start].opname == "RETURN_VALUE":
        # the value was computed before the marker, so this is not a copy
        return None

    return start - 1, start


def read_copied_tail_try(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    setup_index: int,
    end_index: int,
) -> CopiedTailTry | None:
    if instructions[setup_index].opname != "SETUP_FINALLY":
        return None

    handler_index = handler_start(
        instructions, offset_to_index, setup_index, end_index, COPIED_TAIL_VERSION
    )
    if handler_index is None:
        return None

    tails: list[tuple[int, int]] = []
    body_exit = read_body_exit(instructions, offset_to_index, handler_index, tails)
    if body_exit is None:
        return None

    body_end, else_start = body_exit
    if else_start is not None and else_start > end_index:
        return None

    chain = read_copied_tail_handlers(
        instructions, offset_to_index, handler_index, end_index, tails
    )
    if chain is None or not tails:
        return None

    handlers, chain_end = chain

    if not copies_agree(instructions, tails):
        return None

    else_range = None
    statement_end = chain_end

    if else_start is not None:
        if else_start != chain_end:
            return None

        else_end = copied_tail_else_end(instructions, else_start, end_index, tails[0])
        if else_end is None:
            return None

        else_range = (else_start, else_end - (tails[0][1] - tails[0][0]))
        statement_end = else_end

    statement = BlockTry(
        (setup_index + 1, body_end), handlers, else_range, statement_end
    )
    return CopiedTailTry(statement, tails[0])


def copies_agree(instructions: list[Instruction], tails: list[tuple[int, int]]) -> bool:
    tail_start, tail_end = tails[0]
    length = tail_end - tail_start
    return all(
        end - start == length
        and same_instructions(instructions, start, tail_start, length)
        for start, end in tails
    )


def read_body_exit(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_index: int,
    tails: list[tuple[int, int]],
) -> tuple[int, int | None] | None:
    body_tail = tail_behind(instructions, frozenset({"POP_BLOCK"}), handler_index)
    if body_tail is not None:
        body_end, tail_start = body_tail
        tails.append((tail_start, handler_index))
        return body_end, None

    if (
        instructions[handler_index - 1].opname == "JUMP_FORWARD"
        and instructions[handler_index - 2].opname == "POP_BLOCK"
    ):
        else_start = offset_to_index.get(int(instructions[handler_index - 1].argval))
        return None if else_start is None else (handler_index - 2, else_start)

    return None


def copied_tail_else_end(
    instructions: list[Instruction],
    else_start: int,
    end_index: int,
    tail: tuple[int, int],
) -> int | None:
    tail_start, tail_end = tail
    length = tail_end - tail_start
    else_end = terminal_block_end(instructions, else_start, end_index)
    if else_end is None or else_end - length < else_start:
        return None
    if not same_instructions(instructions, else_end - length, tail_start, length):
        return None

    return else_end


def read_copied_tail_handlers(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    handler_index: int,
    end_index: int,
    tails: list[tuple[int, int]],
) -> tuple[tuple[BlockHandler, ...], int] | None:
    handlers: list[BlockHandler] = []
    index = handler_index
    while index < end_index:
        if instructions[index].opname == "RERAISE":
            return (tuple(handlers), index + 1) if handlers else None

        if is_bare_handler(instructions, index, end_index, COPIED_TAIL_VERSION):
            bare = read_bare_copied_tail_handler(instructions, index, end_index, tails)
            if bare is None:
                return None

            handler, end = bare
            handlers.append(handler)
            return tuple(handlers), end

        typed = read_copied_tail_typed_handler(
            instructions, offset_to_index, index, end_index, tails
        )
        if typed is None:
            return None

        handler, index = typed
        handlers.append(handler)

    return None


def read_copied_tail_typed_handler(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    index: int,
    end_index: int,
    tails: list[tuple[int, int]],
) -> tuple[BlockHandler, int] | None:
    head = typed_handler_head(
        instructions, offset_to_index, index, end_index, COPIED_TAIL_VERSION
    )
    if head is None:
        return None

    start, next_index, type_range = head.body_start, head.next_index, head.type_range

    if head.name is not None:
        body = named_handler_body(instructions, start, next_index, head.name, tails)
        if body is None:
            return None

        return BlockHandler(type_range, head.name, body, None), next_index

    tail = tail_behind(instructions, frozenset({"POP_EXCEPT"}), next_index)
    if tail is None:
        return None

    body_end, tail_start = tail
    tails.append((tail_start, next_index))
    return BlockHandler(type_range, None, (start, body_end), None), next_index


def named_handler_body(
    instructions: list[Instruction],
    start: int,
    next_index: int,
    name: str,
    tails: list[tuple[int, int]],
) -> tuple[int, int] | None:
    cleanup = next_index - 4
    if instructions[start].opname != "SETUP_FINALLY" or cleanup <= start:
        return None
    if instructions[start].argval != instructions[cleanup].offset:
        return None
    if not clears_name(instructions, cleanup, name) or (
        instructions[cleanup + 3].opname != "RERAISE"
    ):
        return None

    for exit_start in range(cleanup - 6, start, -1):
        if [item.opname for item in instructions[exit_start : exit_start + 2]] != [
            "POP_BLOCK",
            "POP_EXCEPT",
        ] or not clears_name(instructions, exit_start + 2, name):
            continue

        tail_start = exit_start + 5
        if not is_tail(instructions, tail_start, cleanup):
            return None

        tails.append((tail_start, cleanup))
        return start + 1, exit_start

    return None


def read_bare_copied_tail_handler(
    instructions: list[Instruction],
    index: int,
    end_index: int,
    tails: list[tuple[int, int]],
) -> tuple[BlockHandler, int] | None:
    if not tails:
        return None

    tail_start, tail_end = tails[0]
    length = tail_end - tail_start
    for position in range(index + 3, end_index - length):
        if instructions[position].opname != "POP_EXCEPT":
            continue
        if not same_instructions(instructions, position + 1, tail_start, length):
            continue

        tails.append((position + 1, position + 1 + length))
        handler = BlockHandler(None, None, (index + 3, position), None)
        return handler, position + 1 + length

    return None
