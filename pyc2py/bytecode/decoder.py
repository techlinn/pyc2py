import dis
import sys
from dataclasses import dataclass, field, replace
from types import CodeType
from typing import Any

from pyc2py.bytecode.chained_compare import fold_chained_comparison_tests
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import location_table_line_starts
from pyc2py.bytecode.opcode_table import (
    get_opcode_table,
    normalized_opcode_name,
    version_resolution_warning,
)
from pyc2py.bytecode.operand import (
    COMPARE_SYMBOLS,
    PACKED_LOCAL_OPS,
    SPECIAL_INDEXED_ARG_VALUES,
    common_constants,
    compare_index,
    following_wordcode_arg,
    format_argrepr,
    free_names,
    is_jump_instruction,
    local_names,
    name_index,
    packed_local_indexes,
    read_indexed,
    resolve_arg_value,
)
from pyc2py.bytecode.scanner import scan_code


@dataclass(slots=True)
class BytecodeValidation:
    instruction_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def checks(self) -> tuple[str, ...]:
        return (f"decoded instruction count: {self.instruction_count}",)


def decode_instructions(
    code: Any, version: tuple[int, ...] | None
) -> list[Instruction]:
    if code is None:
        return []

    if version is None:
        return []

    if isinstance(code, CodeType) and version[:2] == sys.version_info[:2]:
        instructions = merge_extended_arg_prefixes(decode_native_instructions(code))
        instructions = without_caches(instructions)
    else:
        instructions = decode_legacy_instructions(code, version)
        if version >= (3, 13):
            instructions = merge_extended_arg_prefixes(
                [
                    half
                    for instruction in instructions
                    for half in split_fast_superinstruction(instruction)
                ]
            )
    # what follows is for the bytecode, whichever python reads it
    if version[:2] == (3, 11):
        instructions = with_end_send_311(instructions)
        instructions = with_nested_return_copies_311(instructions)
    if version >= (3, 14):
        instructions = plain_instructions_314(instructions)
        instructions = fold_with_setup_314(instructions)
        instructions = drop_inlined_builtin_calls_314(instructions)
    return fold_chained_comparison_tests(instructions)


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


def plain_instructions_314(instructions: list[Instruction]) -> list[Instruction]:
    plain: list[Instruction] = []
    for instruction in instructions:
        if instruction.opname == "NOT_TAKEN":
            continue
        if instruction.opname == "LOAD_FAST_BORROW":
            instruction = replace(instruction, opname="LOAD_FAST", borrowed=True)
        elif instruction.opname == "LOAD_SMALL_INT":
            instruction = replace(
                instruction,
                opname="LOAD_CONST",
                arg=None,
                argval=instruction.arg,
                argrepr=repr(instruction.arg),
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
        item.argval is None and item.arg is not None for item in window[2:5]
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

    if isinstance(code, CodeType) and version[:2] == sys.version_info[:2]:
        instructions = decode_native_instructions(code)
        return validate_instruction_objects(instructions)

    opcode_table = get_opcode_table(version)
    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    decoded = scan_code(code_bytes, opcode_table, version)
    warnings: list[str] = []
    version_warning = version_resolution_warning(version)

    if version_warning is not None:
        warnings.append(version_warning)

    if version >= (3, 6) and len(code_bytes) % 2:
        warnings.append("wordcode length is not aligned to 2-byte instruction units")

    target_offsets = {item.offset for item in decoded}
    target_offsets.add(len(code_bytes))
    for item in decoded:
        warnings.extend(
            validate_decoded_opcode(code, opcode_table, item, target_offsets, version)
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
    if halves is None or not isinstance(argval, tuple) or len(argval) != 2:
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
    if isinstance(item.starts_line, bool):
        return getattr(item, "line_number", None) if item.starts_line else None
    return item.starts_line


def native_argval(code: CodeType, item: dis.Instruction) -> object:
    if item.opname == "KW_NAMES" and not isinstance(item.argval, tuple):
        return code.co_consts[item.arg] if item.arg is not None else item.argval
    # dis gives the number of a common constant as the value. the decompiler
    # wants the name, the same one the decoder for other hosts gives: dis
    # writes the builtin "all" as "<built-in function all>"
    if item.opname == "LOAD_COMMON_CONSTANT" and item.arg is not None:
        return read_indexed(common_constants(sys.version_info[:2]), item.arg)
    return item.argval


def native_argrepr(item: dis.Instruction) -> str:
    # python 3.13+ dis prints a comparison whose result is only used as a truth
    # value as bool(==), the rest of the decompiler matches on the bare operator
    if item.opname == "COMPARE_OP":
        return str(item.argval)
    return str(item.argrepr)


def validate_instruction_objects(instructions: list[Instruction]) -> BytecodeValidation:
    warnings: list[str] = []
    offsets = {instruction.offset for instruction in instructions}

    for instruction in instructions:
        if is_unknown_opname(instruction.opname):
            warnings.append(
                f"unknown opcode at {instruction.offset}: {instruction.opname}"
            )

        if (
            is_jump_opname(instruction.opname)
            and isinstance(instruction.argval, int)
            and instruction.argval not in offsets
        ):
            warnings.append(
                "jump target is outside decoded offsets at "
                f"{instruction.offset}: {instruction.argval}"
            )

    return BytecodeValidation(instruction_count=len(instructions), warnings=warnings)


def decode_legacy_instructions(
    code: Any, version: tuple[int, ...]
) -> list[Instruction]:
    opcode_table = get_opcode_table(version)
    code_bytes = bytes(getattr(code, "co_code", b"") or b"")
    decoded = scan_code(code_bytes, opcode_table, version)
    line_starts = legacy_line_starts(code, version)

    resolved = [
        resolve_arg_value(
            code, opcode_table, item.opcode, item.opname, item.arg, item.offset, version
        )
        for item in decoded
    ]
    targets = {
        argval
        for item, argval in zip(decoded, resolved, strict=True)
        if is_jump_instruction(opcode_table, item.opcode)
    }

    if version >= (3, 11):
        line_starts = scanned_location_line_starts(code)

    instructions: list[Instruction] = []
    for item, argval in zip(decoded, resolved, strict=True):
        instructions.append(
            Instruction(
                offset=item.offset,
                opname=item.opname,
                arg=item.arg,
                argval=argval,
                argrepr=format_argrepr(item.opname, argval),
                starts_line=legacy_instruction_line(item, line_starts),
                is_jump_target=item.offset in targets,
            )
        )

    return instructions


def legacy_instruction_line(item: Any, line_starts: dict[int, int]) -> int | None:
    if item.opname == "SET_LINENO":
        return item.arg
    return line_starts.get(item.offset)


def legacy_line_starts(
    code: Any, version: tuple[int, ...] | None = None
) -> dict[int, int]:
    lnotab = bytes(getattr(code, "co_lnotab", b"") or b"")
    first_line = int(getattr(code, "co_firstlineno", 1) or 1)
    if not lnotab:
        return {}

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
    if version[:2] == (3, 12) and item.opname in {
        "LOAD_CONST__LOAD_FAST",
        "LOAD_FAST__LOAD_CONST",
        "LOAD_FAST__LOAD_FAST",
        "STORE_FAST__LOAD_FAST",
        "STORE_FAST__STORE_FAST",
    }:
        return validate_wordcode_superinstruction_operand(
            code, item.offset, item.opname, arg
        )

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


def validate_wordcode_superinstruction_operand(
    code: Any,
    offset: int,
    opname: str,
    first_arg: int,
) -> list[str]:
    second_arg = following_wordcode_arg(code, offset)
    if second_arg is None:
        return [f"superinstruction missing following operand at {offset}: {opname}"]

    consts = getattr(code, "co_consts", ())
    varnames = getattr(code, "co_varnames", ())

    if opname == "LOAD_CONST__LOAD_FAST":
        return [
            *validate_index("const", offset, consts, first_arg),
            *validate_index("local", offset, varnames, second_arg),
        ]
    if opname == "LOAD_FAST__LOAD_CONST":
        return [
            *validate_index("local", offset, varnames, first_arg),
            *validate_index("const", offset, consts, second_arg),
        ]
    return [
        *validate_index("local", offset, varnames, first_arg),
        *validate_index("local", offset, varnames, second_arg),
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


def is_unknown_opname(opname: str) -> bool:
    return opname.startswith("<") and opname.endswith(">")


def is_jump_opname(opname: str) -> bool:
    return "JUMP" in opname or opname in {"FOR_ITER", "CONTINUE_LOOP"}
