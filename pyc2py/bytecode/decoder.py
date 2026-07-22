import dis
import sys
from dataclasses import dataclass, field
from types import CodeType
from typing import Any
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.operand import (
    COMPARE_SYMBOLS,
    PACKED_LOCAL_OPS,
    SPECIAL_INDEXED_ARG_VALUES,
    compare_index,
    common_constants,
    following_wordcode_arg,
    format_argrepr,
    free_names,
    is_jump_instruction,
    name_index,
    packed_local_indexes,
    resolve_arg_value,
)
from pyc2py.bytecode.opcode_table import (
    get_opcode_table,
    normalized_opcode_name,
    version_resolution_warning,
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
        return decode_native_instructions(code)

    return decode_legacy_instructions(code, version)

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
    return [
        Instruction(
            offset=int(item.offset),
            opname=str(item.opname),
            arg=item.arg,
            argval=item.argval,
            argrepr=str(item.argrepr),
            starts_line=item.starts_line,
            is_jump_target=bool(item.is_jump_target),
        )
        for item in dis.get_instructions(code, show_caches=True)
    ]

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
    line_starts = legacy_line_starts(code)

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

def legacy_line_starts(code: Any) -> dict[int, int]:
    lnotab = bytes(getattr(code, "co_lnotab", b"") or b"")
    first_line = int(getattr(code, "co_firstlineno", 1) or 1)
    if not lnotab:
        return {}

    starts: dict[int, int] = {0: first_line}
    offset = 0
    line = first_line

    for index in range(0, len(lnotab) - 1, 2):
        offset += lnotab[index]
        line += signed_byte(lnotab[index + 1])
        starts[offset] = line

    return starts

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
        return validate_index(
            "local", item.offset, getattr(code, "co_varnames", ()), arg
        )

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
