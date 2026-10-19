import dis
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import CodeType
from typing import Any

from pyc2py.bytecode.chained_compare import fold_chained_comparison_tests
from pyc2py.bytecode.formats import pypy_format
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import location_table_line_starts
from pyc2py.bytecode.opcode_table import (
    OpcodeTable,
    get_opcode_table,
    is_jump_opname,
    is_unknown_opname,
    normalized_opcode_name,
    version_resolution_warning,
)
from pyc2py.bytecode.operand import (
    COMPARE_SYMBOLS,
    PACKED_LOCAL_OPS,
    SPECIAL_INDEXED_ARG_VALUES,
    common_constants,
    compare_index,
    format_argrepr,
    free_names,
    is_jump_instruction,
    local_names,
    name_index,
    packed_local_indexes,
    read_indexed,
    resolve_arg_value,
)
from pyc2py.bytecode.pypy import pypy_in_cpython_form, pypy_line_starts
from pyc2py.bytecode.scanner import scan_code
from pyc2py.bytecode.stack_effect import expression_start
from pyc2py.types import BuiltinName, is_tuple


@dataclass(slots=True)
class BytecodeValidation:
    instruction_count: int
    warnings: list[str] = field(default_factory=list[str])

    @property
    def checks(self) -> tuple[str, ...]:
        return (f"decoded instruction count: {self.instruction_count}",)


def read_native_instructions(
    code: Any, version: tuple[int, ...]
) -> list[Instruction] | None:
    if not isinstance(code, CodeType) or version[:2] != sys.version_info[:2]:
        return None

    try:
        return decode_native_instructions(code)
    except (IndexError, KeyError, ValueError):
        # dis fails on damaged operands; the input checks report them
        return None


# each code object is decoded once per file; a decoded list never changes
DECODED: ContextVar[dict[int, tuple[Any, Any, list[Instruction]]] | None] = ContextVar(
    "DECODED", default=None
)


@contextmanager
def decoding_once() -> Generator[None]:
    token = DECODED.set({})

    try:
        yield
    finally:
        DECODED.reset(token)


def decode_instructions(
    code: Any, version: tuple[int, ...] | None
) -> list[Instruction]:
    decoded = DECODED.get()
    if decoded is None:
        return decode_code_instructions(code, version)

    found = decoded.get(id(code))
    if found is not None and found[0] is code and found[1] == version:
        return found[2]

    instructions = decode_code_instructions(code, version)
    decoded[id(code)] = (code, version, instructions)
    return instructions


def decode_code_instructions(
    code: Any, version: tuple[int, ...] | None
) -> list[Instruction]:
    if code is None:
        return []
    if version is None:
        return []

    native = read_native_instructions(code, version)
    if native is not None:
        instructions = without_caches(merge_extended_arg_prefixes(native))
    else:
        instructions = decode_legacy_instructions(code, version)

        if version >= (3, 13):
            instructions = [
                half
                for instruction in instructions
                for half in split_fast_superinstruction(instruction)
            ]

        instructions = merge_extended_arg_prefixes(instructions)

        if getattr(code, "opcodes", ""):
            instructions = pypy_in_cpython_form(instructions, version, code)

    return in_common_form(instructions, version)


def in_common_form(
    instructions: list[Instruction], version: tuple[int, ...]
) -> list[Instruction]:
    if version < (3, 2):
        instructions = dup_topx_before_32(instructions)
    if version < (3, 5):
        instructions = empty_maps_before_35(instructions)
    if version[:2] == (3, 0):
        instructions = popping_jumps_30(instructions)
    if version < (3, 11):
        instructions = threaded_value_tests(instructions)
    if version[:2] == (3, 10):
        instructions = jumps_by_direction_310(instructions)

    if version[:2] == (3, 11):
        instructions = with_end_send_311(instructions)
        instructions = with_nested_return_copies_311(instructions)

    if version >= (3, 14):
        instructions = plain_instructions_314(instructions, version)
        instructions = fold_with_setup_314(instructions)
        instructions = drop_inlined_builtin_calls_314(instructions)

    if version >= (3, 16):
        instructions = plain_instructions_316(instructions, version)

    return fold_chained_comparison_tests(instructions)


# a kept test value that lands on the same test is that test
THREADED_TESTS = {
    "JUMP_IF_FALSE_OR_POP": "POP_JUMP_IF_FALSE",
    "JUMP_IF_TRUE_OR_POP": "POP_JUMP_IF_TRUE",
}


def threaded_value_tests(instructions: list[Instruction]) -> list[Instruction]:
    index_of = {item.offset: index for index, item in enumerate(instructions)}
    threaded: list[Instruction] = []

    for instruction in instructions:
        test = THREADED_TESTS.get(instruction.opname)
        target = index_of.get(instruction.argval)
        if test is not None and target is not None:
            landing = instructions[target]
            if landing.opname == test:
                instruction = replace(
                    instruction,
                    opname=test,
                    argval=landing.argval,
                    argrepr=f"to {landing.argval}",
                )

        threaded.append(instruction)

    return threaded


def empty_maps_before_35(instructions: list[Instruction]) -> list[Instruction]:
    # BUILD_MAP's argument was only a size hint; STORE_MAP adds the items
    return [
        replace(item, arg=0, argval=0, argrepr="")
        if item.opname == "BUILD_MAP"
        else item
        for item in instructions
    ]


def dup_topx_before_32(instructions: list[Instruction]) -> list[Instruction]:
    named = {1: "DUP_TOP", 2: "DUP_TOP_TWO"}
    return [
        (
            replace(instruction, opname=named[instruction.arg], arg=None, argval=None)
            if instruction.opname == "DUP_TOPX" and instruction.arg in named
            else instruction
        )
        for instruction in instructions
    ]


# 3.1 folded each 3.0 jump and pop into one jump
POPPING_JUMPS_30 = {
    "JUMP_IF_FALSE": ("POP_JUMP_IF_FALSE", "JUMP_IF_FALSE_OR_POP"),
    "JUMP_IF_TRUE": ("POP_JUMP_IF_TRUE", "JUMP_IF_TRUE_OR_POP"),
}
NO_FALLTHROUGH_30 = {
    "BREAK_LOOP",
    "CONTINUE_LOOP",
    "JUMP_ABSOLUTE",
    "JUMP_FORWARD",
    "RAISE_VARARGS",
    "RETURN_VALUE",
}


def popping_jumps_30(instructions: list[Instruction]) -> list[Instruction]:
    index_of = {item.offset: index for index, item in enumerate(instructions)}
    jumps: dict[int, Instruction] = {}
    dropped: set[int] = set()

    for index, jump in enumerate(instructions):
        if jump.opname not in POPPING_JUMPS_30:
            continue

        target = index_of.get(jump.argval)
        if (
            index + 1 >= len(instructions)
            or instructions[index + 1].opname != "POP_TOP"
            or target is None
            or instructions[target].opname in POPPING_JUMPS_30
        ):
            return instructions

        popping, keeping = POPPING_JUMPS_30[jump.opname]
        dropped.add(index + 1)

        if instructions[target].opname == "POP_TOP":
            if target + 1 >= len(instructions):
                return instructions

            if ends_statement_30(instructions, target):
                target = past_shared_pop_30(instructions, index_of, target)
            else:
                dropped.add(target)
                target += 1

            opname = popping
        else:
            opname = keeping

        landing = instructions[target].offset
        jumps[index] = replace(
            jump, opname=opname, argval=landing, argrepr=f"to {landing}"
        )

    if not jumps or not pops_only_reached_by(instructions, jumps, dropped):
        return instructions

    dropped |= jumps_over_dropped_pops(instructions, dropped)
    return without_dropped_pops(instructions, jumps, dropped)


def ends_statement_30(instructions: list[Instruction], pop: int) -> bool:
    before = instructions[pop - 1].opname
    return before not in NO_FALLTHROUGH_30 and before not in POPPING_JUMPS_30


def past_shared_pop_30(
    instructions: list[Instruction], index_of: dict[int, int], pop: int
) -> int:
    after = instructions[pop + 1]
    if after.opname == "JUMP_ABSOLUTE":
        return index_of.get(after.argval, pop + 1)

    return pop + 1


def jumps_over_dropped_pops(
    instructions: list[Instruction], dropped: set[int]
) -> set[int]:
    redundant: set[int] = set()

    for pop in dropped:
        jump, after = instructions[pop - 1], instructions[pop + 1 : pop + 2]
        if (
            pop - 1 in dropped
            or jump.opname not in {"JUMP_ABSOLUTE", "JUMP_FORWARD"}
            or jump.is_jump_target
            or not after
        ):
            continue
        if jump.argval == after[0].offset or (
            after[0].opname in {"JUMP_ABSOLUTE", "JUMP_FORWARD"}
            and after[0].argval == jump.argval
        ):
            redundant.add(pop - 1)

    return redundant


def without_dropped_pops(
    instructions: list[Instruction],
    jumps: dict[int, Instruction],
    dropped: set[int],
) -> list[Instruction]:
    landings = {jump.argval for jump in jumps.values()}
    rewritten: list[Instruction] = []
    line: int | None = None

    for index, instruction in enumerate(instructions):
        if index in dropped:
            line = line if instruction.starts_line is None else instruction.starts_line
            continue

        instruction = jumps.get(index, instruction)
        if line is not None and instruction.starts_line is None:
            instruction = replace(instruction, starts_line=line)
        if instruction.offset in landings:
            instruction = replace(instruction, is_jump_target=True)

        rewritten.append(instruction)
        line = None

    return rewritten


def pops_only_reached_by(
    instructions: list[Instruction],
    jumps: dict[int, Instruction],
    dropped: set[int],
) -> bool:
    for index, instruction in enumerate(instructions):
        if index in jumps:
            continue
        if not (
            is_jump_opname(instruction.opname)
            or instruction.opname.startswith("SETUP_")
        ):
            continue
        if any(instructions[pop].offset == instruction.argval for pop in dropped):
            return False

    return all(
        pop and (pop - 1 in jumps or instructions[pop - 1].opname in NO_FALLTHROUGH_30)
        for pop in dropped
    )


# 3.11 split 3.10's jumps by direction
DIRECTED_JUMPS_310 = {
    "JUMP_ABSOLUTE": ("JUMP_FORWARD", "JUMP_BACKWARD"),
    "POP_JUMP_IF_FALSE": ("POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_FALSE"),
    "POP_JUMP_IF_TRUE": ("POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE"),
}


def jumps_by_direction_310(instructions: list[Instruction]) -> list[Instruction]:
    directed: list[Instruction] = []

    for instruction in instructions:
        names = DIRECTED_JUMPS_310.get(instruction.opname)
        if names is not None:
            forward, backward = names
            opname = forward if instruction.argval > instruction.offset else backward
            instruction = replace(instruction, opname=opname)

        directed.append(instruction)

    return directed


def with_end_send_311(instructions: list[Instruction]) -> list[Instruction]:
    targets = {
        instruction.argval
        for instruction in instructions
        if instruction.opname == "SEND"
    }
    if not targets:
        return instructions

    marked: list[Instruction] = []

    for instruction in instructions:
        if instruction.offset in targets:
            marked.append(
                Instruction(
                    offset=instruction.offset - 1,
                    opname="END_SEND",
                    arg=None,
                    argval=None,
                    argrepr="",
                    starts_line=None,
                    is_jump_target=True,
                )
            )
            instruction = replace(instruction, is_jump_target=False)
        elif instruction.opname == "SEND":
            instruction = replace(instruction, argval=instruction.argval - 1)

        marked.append(instruction)

    return marked


def with_nested_return_copies_311(instructions: list[Instruction]) -> list[Instruction]:
    copies = trailing_return_copies(instructions)
    first_jump: dict[int, int] = {}

    for instruction in instructions:
        if "JUMP" in instruction.opname and instruction.argval in copies:
            first_jump.setdefault(instruction.argval, instruction.offset)

    if len(first_jump) != len(copies) or len(copies) < 2:
        return instructions

    inner_first = sorted(copies, key=lambda offset: -first_jump[offset])
    moved = dict(zip(inner_first, sorted(copies), strict=True))
    lines = {moved[offset]: copies[offset] for offset in copies}
    nested: list[Instruction] = []

    for instruction in instructions:
        if instruction.offset in lines:
            instruction = replace(instruction, starts_line=lines[instruction.offset])
        elif "JUMP" in instruction.opname and instruction.argval in moved:
            target = moved[instruction.argval]
            instruction = replace(instruction, argval=target, argrepr=f"to {target}")

        nested.append(instruction)

    return nested


def trailing_return_copies(instructions: list[Instruction]) -> dict[int, int | None]:
    copies: dict[int, int | None] = {}
    index = len(instructions) - 2
    while index >= 0:
        load, leave = instructions[index : index + 2]
        if leave.opname != "RETURN_VALUE" or not load.is_jump_target:
            break
        if load.opname != "LOAD_CONST" or load.argval is not None:
            break

        copies[load.offset] = load.starts_line
        index -= 2

    return copies


def without_caches(instructions: list[Instruction]) -> list[Instruction]:
    return [
        instruction for instruction in instructions if instruction.opname != "CACHE"
    ]


NULL_STORE_DELETES = {
    "STORE_NAME": "DELETE_NAME",
    "STORE_GLOBAL": "DELETE_GLOBAL",
    "STORE_ATTR": "DELETE_ATTR",
}


def plain_instructions_316(
    instructions: list[Instruction], version: tuple[int, ...]
) -> list[Instruction]:
    result: list[Instruction] = []

    for instruction in instructions:
        if (
            instruction.opname == "CALL_INTRINSIC_2"
            and instruction.argrepr == "INTRINSIC_ADD_CONDITIONAL_ANNOTATION"
        ):
            # 3.14 and 3.15 spell it SET_ADD 1
            instruction = replace(
                instruction, opname="SET_ADD", arg=1, argval=1, argrepr="1"
            )

        # 3.16 spells del x as PUSH_NULL; STORE_NAME x, del a.x as PUSH_NULL; a
        start = len(result)

        if instruction.opname == "STORE_ATTR":
            start = expression_start(result, len(result), version) or 0

        if (
            instruction.opname in NULL_STORE_DELETES
            and start
            and result[start - 1].opname == "PUSH_NULL"
        ):
            null = result.pop(start - 1)
            deleted = NULL_STORE_DELETES[instruction.opname]

            if start - 1 < len(result):
                first = result[start - 1]
                result[start - 1] = replace(
                    first,
                    offset=null.offset,
                    starts_line=null.starts_line,
                    is_jump_target=null.is_jump_target or first.is_jump_target,
                )
                instruction = replace(instruction, opname=deleted)
            else:
                instruction = replace(
                    instruction,
                    offset=null.offset,
                    opname=deleted,
                    starts_line=null.starts_line,
                    is_jump_target=null.is_jump_target,
                )

        result.append(instruction)

    return result


def plain_instructions_314(
    instructions: list[Instruction], version: tuple[int, ...]
) -> list[Instruction]:
    plain: list[Instruction] = []

    for instruction in instructions:
        if instruction.opname == "NOT_TAKEN":
            continue
        if (
            instruction.opname == "PUSH_NULL"
            and plain
            and plain[-1].opname in {"GET_AWAITABLE", "GET_ANEXT"}
        ):
            # 3.15 holds an awaitable in two slots
            continue

        if instruction.opname == "GET_ITER" and instruction.arg == 3:
            # 3.15 spells yield from as GET_ITER 3
            instruction = replace(
                instruction, opname="GET_YIELD_FROM_ITER", arg=None, argval=None
            )
        elif instruction.opname == "LOAD_FAST_BORROW":
            instruction = replace(instruction, opname="LOAD_FAST", borrowed=True)
        elif instruction.opname == "LOAD_SMALL_INT":
            instruction = replace(
                instruction,
                opname="LOAD_CONST",
                arg=None,
                argval=instruction.arg,
                argrepr=repr(instruction.arg),
            )
        elif (
            instruction.opname == "LOAD_COMMON_CONSTANT"
            and instruction.arg is not None
            and instruction.arg < len(common_constants(version))
            and not isinstance(instruction.argval, BuiltinName)
        ):
            # 3.15 loads None, True, False, '' and -1 this way
            instruction = replace(
                instruction,
                opname="LOAD_CONST",
                arg=None,
                argrepr=repr(instruction.argval),
            )
        elif instruction.opname == "POP_ITER":
            instruction = replace(instruction, opname="POP_TOP")
        elif instruction.opname == "FOR_ITER" and plain and plain[-1].argval == ".0":
            plain.append(
                replace(
                    plain[-1],
                    offset=plain[-1].offset + 1,
                    opname="GET_ITER",
                    arg=None,
                    argval=None,
                    argrepr="",
                    starts_line=None,
                    is_jump_target=False,
                    borrowed=False,
                )
            )

        plain.append(instruction)

    return plain


WITH_SETUP_314 = (
    ("COPY", 1),
    ("LOAD_SPECIAL", 1),
    ("SWAP", 2),
    ("SWAP", 3),
    ("LOAD_SPECIAL", 0),
    ("CALL", 0),
)
ASYNC_WITH_SETUP_314 = (
    ("COPY", 1),
    ("LOAD_SPECIAL", 3),
    ("SWAP", 2),
    ("SWAP", 3),
    ("LOAD_SPECIAL", 2),
    ("CALL", 0),
)


def is_with_exit_under_value_314(window: list[Instruction]) -> bool:
    if [item.opname for item in window] != [
        "SWAP",
        "SWAP",
        *["LOAD_CONST"] * 3,
        "CALL",
    ]:
        return False

    return [item.arg for item in (*window[:2], window[5])] == [3, 2, 3] and all(
        item.argval is None for item in window[2:5]
    )


def fold_with_setup_314(instructions: list[Instruction]) -> list[Instruction]:
    folded: list[Instruction] = []
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        window = tuple(
            (item.opname, item.arg) for item in instructions[index : index + 6]
        )
        if window == WITH_SETUP_314:
            instruction = replace(instruction, opname="BEFORE_WITH", arg=None)
            index += 5
        elif window == ASYNC_WITH_SETUP_314:
            instruction = replace(instruction, opname="BEFORE_ASYNC_WITH", arg=None)
            index += 5
        elif is_with_exit_under_value_314(instructions[index : index + 6]):
            instruction = replace(instruction, arg=2, argval=2, argrepr="2")
            index += 1

        folded.append(instruction)
        index += 1

    return folded


def drop_inlined_builtin_calls_314(
    instructions: list[Instruction],
) -> list[Instruction]:
    offsets = {item.offset: index for index, item in enumerate(instructions)}
    kept: list[Instruction] = []
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        kept.append(instruction)
        index += 1
        call_index = inlined_builtin_call_start(instructions, index, offsets)
        if call_index is not None:
            kept.append(replace(instructions[call_index], is_jump_target=False))
            index = call_index + 1

    return kept


def inlined_builtin_call_start(
    instructions: list[Instruction],
    index: int,
    offsets: dict[int, int],
) -> int | None:
    if index + 4 >= len(instructions):
        return None

    name, copy, builtin, same, jump, pop = instructions[index - 1 : index + 5]
    if (copy.opname, copy.arg) != ("COPY", 1) or (same.opname, same.arg) != (
        "IS_OP",
        0,
    ):
        return None
    if builtin.opname != "LOAD_COMMON_CONSTANT" or builtin.argval != name.argval:
        return None
    if jump.opname != "POP_JUMP_IF_FALSE" or pop.opname != "POP_TOP":
        return None

    call_index = offsets.get(jump.argval)
    if call_index is None or instructions[call_index].opname != "PUSH_NULL":
        return None

    return call_index


def merge_extended_arg_prefixes(instructions: list[Instruction]) -> list[Instruction]:
    merged: list[Instruction] = []
    prefix: Instruction | None = None

    for instruction in instructions:
        if instruction.opname == "EXTENDED_ARG":
            prefix = prefix or instruction
            continue

        if prefix is not None:
            instruction = replace(
                instruction,
                offset=prefix.offset,
                starts_line=prefix.starts_line or instruction.starts_line,
                is_jump_target=prefix.is_jump_target or instruction.is_jump_target,
            )
            prefix = None

        merged.append(instruction)

    return merged


def validate_bytecode(code: Any, version: tuple[int, ...] | None) -> BytecodeValidation:
    if code is None or version is None:
        return BytecodeValidation(instruction_count=0)

    opcode_table = code_opcode_table(code, version)
    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    decoded = scan_code(code_bytes, opcode_table, opcode_table.layout)
    warnings: list[str] = []
    version_warning = version_resolution_warning(version)
    if version_warning is not None:
        warnings.append(version_warning)
    if opcode_table.layout >= (3, 6) and len(code_bytes) % 2:
        warnings.append("wordcode length is not aligned to 2-byte instruction units")

    target_offsets = {item.offset for item in decoded}
    target_offsets.add(len(code_bytes))

    for item in decoded:
        warnings.extend(
            validate_decoded_opcode(
                code, opcode_table, item, target_offsets, opcode_table.layout
            )
        )

    return BytecodeValidation(instruction_count=len(decoded), warnings=warnings)


def decode_native_instructions(code: CodeType) -> list[Instruction]:
    decoded = dis.get_instructions(code)
    instructions: list[Instruction] = []

    for item in decoded:
        instruction = Instruction(
            offset=int(item.offset),
            opname=str(item.opname),
            arg=item.arg,
            argval=native_argval(code, item),
            argrepr=native_argrepr(item),
            starts_line=native_starts_line(item),
            is_jump_target=bool(item.is_jump_target),
        )
        instructions.extend(split_fast_superinstruction(instruction))

    return instructions


FAST_SUPERINSTRUCTIONS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST", "LOAD_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
}


def split_fast_superinstruction(instruction: Instruction) -> list[Instruction]:
    halves = FAST_SUPERINSTRUCTIONS.get(instruction.opname)
    argval = instruction.argval
    if halves is None or not is_tuple(argval) or len(argval) != 2:
        return [instruction]

    borrowed = "BORROW" in instruction.opname
    return [
        Instruction(
            offset=instruction.offset,
            opname=halves[0],
            arg=None,
            argval=argval[0],
            argrepr=str(argval[0]),
            starts_line=instruction.starts_line,
            is_jump_target=instruction.is_jump_target,
            borrowed=borrowed,
        ),
        Instruction(
            offset=instruction.offset + 1,
            opname=halves[1],
            arg=None,
            argval=argval[1],
            argrepr=str(argval[1]),
            starts_line=None,
            is_jump_target=False,
            borrowed=borrowed,
        ),
    ]


def native_starts_line(item: dis.Instruction) -> int | None:
    return item.line_number if item.starts_line else None


def native_argval(code: CodeType, item: dis.Instruction) -> object:
    if item.opname == "KW_NAMES" and not is_tuple(item.argval):
        return code.co_consts[item.arg] if item.arg is not None else item.argval
    # dis gives a common constant's index; the decompiler wants its name
    if item.opname == "LOAD_COMMON_CONSTANT" and item.arg is not None:
        return read_indexed(common_constants(sys.version_info[:2]), item.arg)

    return item.argval


def native_argrepr(item: dis.Instruction) -> str:
    # 3.13+ dis writes a truth-only comparison as bool(==); readers want ==
    if item.opname == "COMPARE_OP":
        return str(item.argval)

    return str(item.argrepr)


def decode_legacy_instructions(
    code: Any, version: tuple[int, ...]
) -> list[Instruction]:
    opcode_table = code_opcode_table(code, version)
    layout = opcode_table.layout
    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    decoded = scan_code(code_bytes, opcode_table, layout)
    line_starts = code_line_starts(code, version)
    resolved = [
        resolve_arg_value(
            code, opcode_table, item.opcode, item.opname, item.arg, item.offset, layout
        )
        for item in decoded
    ]
    targets = {
        argval
        for item, argval in zip(decoded, resolved, strict=True)
        if is_jump_instruction(opcode_table, item.opcode)
    }
    instructions: list[Instruction] = []

    for item, argval in zip(decoded, resolved, strict=True):
        instructions.append(
            Instruction(
                offset=item.offset,
                opname=item.opname,
                arg=item.arg,
                argval=argval,
                argrepr=format_argrepr(item.opname, argval),
                starts_line=line_starts.get(item.offset),
                is_jump_target=item.offset in targets,
            )
        )

    return instructions


def code_opcode_table(code: Any, version: tuple[int, ...]) -> OpcodeTable:
    return get_opcode_table(version, getattr(code, "opcodes", ""))


def code_line_starts(code: Any, version: tuple[int, ...]) -> dict[int, int]:
    opcodes = getattr(code, "opcodes", "")
    if opcodes and pypy_format(opcodes).positions:
        return pypy_line_starts(code)
    if version >= (3, 11):
        return scanned_location_line_starts(code)

    return legacy_line_starts(code, version)


def legacy_line_starts(
    code: Any, version: tuple[int, ...] | None = None
) -> dict[int, int]:
    if version is not None and version[:2] == (3, 10):
        return line_starts_310(code)

    lnotab = bytes(getattr(code, "co_lnotab", b"") or b"")
    first_line = int(getattr(code, "co_firstlineno", 1) or 1)
    starts: dict[int, int] = {0: first_line}
    offset = 0
    line = first_line

    for index in range(0, len(lnotab) - 1, 2):
        offset += lnotab[index]

        # the line step only became a signed byte in python 3.6
        if version is not None and version < (3, 6):
            line += lnotab[index + 1]
        else:
            line += signed_byte(lnotab[index + 1])

        starts[offset] = line

    return starts


def line_starts_310(code: Any) -> dict[int, int]:
    table = bytes(getattr(code, "co_linetable", b"") or b"")
    line = int(getattr(code, "co_firstlineno", 1) or 1)
    starts: dict[int, int] = {}
    offset = 0
    last: int | None = None

    for index in range(0, len(table) - 1, 2):
        length, step = table[index], signed_byte(table[index + 1])
        if step != -128:
            line += step
            if line != last and length:
                starts[offset] = line
                last = line

        offset += length

    return starts


def scanned_location_line_starts(code: Any) -> dict[int, int]:
    table = bytes(getattr(code, "co_linetable", b"") or b"")
    first_line = int(getattr(code, "co_firstlineno", 1) or 1)

    try:
        return location_table_line_starts(table, first_line)
    except ValueError:
        return {}


def signed_byte(value: int) -> int:
    if value < 128:
        return value

    return value - 256


def validate_decoded_opcode(
    code: Any,
    opcode_table: Any,
    item: Any,
    target_offsets: set[int],
    version: tuple[int, ...],
) -> list[str]:
    warnings: list[str] = []

    if is_unknown_opname(item.opname):
        warnings.append(f"unknown opcode at {item.offset}: {item.opname}")

    arg = item.arg
    if arg is None:
        return warnings

    warnings.extend(validate_decoded_operand(code, opcode_table, item, arg, version))
    warnings.extend(
        validate_decoded_jump(code, opcode_table, item, arg, target_offsets, version)
    )
    return warnings


def validate_decoded_operand(
    code: Any,
    opcode_table: Any,
    item: Any,
    arg: int,
    version: tuple[int, ...],
) -> list[str]:
    opname = normalized_opcode_name(item.opname)
    warnings = validate_indexed_decoded_operand(
        code, opcode_table, item, opname, arg, version
    )
    if warnings:
        return warnings

    warnings = validate_special_decoded_operand(code, item, opname, arg, version)
    if warnings:
        return warnings

    return []


def validate_indexed_decoded_operand(
    code: Any,
    opcode_table: Any,
    item: Any,
    opname: str,
    arg: int,
    version: tuple[int, ...],
) -> list[str]:
    if item.opcode in getattr(opcode_table, "hasconst", ()):
        return validate_index("const", item.offset, getattr(code, "co_consts", ()), arg)

    if item.opcode in getattr(opcode_table, "hasname", ()):
        index = name_index(opname, arg, version)
        return validate_index("name", item.offset, getattr(code, "co_names", ()), index)

    if item.opcode in getattr(opcode_table, "haslocal", ()):
        return validate_index("local", item.offset, local_names(code, version), arg)
    if item.opcode in getattr(opcode_table, "hasfree", ()):
        return validate_index("free", item.offset, free_names(code, version), arg)
    if item.opcode in getattr(opcode_table, "hascompare", ()):
        return validate_index(
            "compare", item.offset, COMPARE_SYMBOLS, compare_index(arg, version)
        )

    return []


def validate_special_decoded_operand(
    code: Any,
    item: Any,
    opname: str,
    arg: int,
    version: tuple[int, ...],
) -> list[str]:
    if opname in PACKED_LOCAL_OPS:
        return validate_packed_local_operand(code, item.offset, arg)
    if opname == "LOAD_COMMON_CONSTANT":
        return validate_index(
            "common-constant", item.offset, common_constants(version), arg
        )

    values = SPECIAL_INDEXED_ARG_VALUES.get(opname)
    if values is None:
        return []

    return validate_index(special_arg_label(opname), item.offset, values, arg)


def special_arg_label(opname: str) -> str:
    if opname == "BINARY_OP":
        return "binary-op"
    if opname == "CALL_INTRINSIC_1":
        return "intrinsic-1"
    if opname == "CALL_INTRINSIC_2":
        return "intrinsic-2"

    return opname.lower().replace("_", "-")


def validate_packed_local_operand(code: Any, offset: int, arg: int) -> list[str]:
    high_index, low_index = packed_local_indexes(arg)
    varnames = getattr(code, "co_varnames", ())
    return [
        *validate_index("local", offset, varnames, high_index),
        *validate_index("local", offset, varnames, low_index),
    ]


def validate_decoded_jump(
    code: Any,
    opcode_table: Any,
    item: Any,
    arg: int,
    target_offsets: set[int],
    version: tuple[int, ...],
) -> list[str]:
    if not is_jump_instruction(opcode_table, item.opcode):
        return []

    target = resolve_arg_value(
        code, opcode_table, item.opcode, item.opname, arg, item.offset, version
    )
    if isinstance(target, int) and target not in target_offsets:
        return [f"jump target is outside decoded offsets at {item.offset}: {target}"]

    return []


def validate_index(label: str, offset: int, values: Any, index: int) -> list[str]:
    if index < 0:
        return [f"{label} operand index is negative at {offset}: {index}"]

    try:
        count = len(values)
    except TypeError:
        return [f"{label} operand table is not sized at {offset}"]

    if index >= count:
        return [
            f"{label} operand index is out of range at {offset}: {index} >= {count}"
        ]

    return []
