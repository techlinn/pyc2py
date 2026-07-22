from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecodedOpcode:
    offset: int
    opcode: int
    opname: str
    arg: int | None
    next_offset: int


def scan_code(
    code: bytes, opcode_table: Any, version: tuple[int, ...]
) -> list[DecodedOpcode]:
    if version >= (3, 6):
        return scan_wordcode(code, opcode_table, version)
    return scan_bytecode(code, opcode_table)


def scan_wordcode(
    code: bytes,
    opcode_table: Any,
    version: tuple[int, ...],
) -> list[DecodedOpcode]:
    result: list[DecodedOpcode] = []
    extended_arg = 0
    offset = 0

    while offset < len(code) - 1:
        opcode = code[offset]
        raw_arg = code[offset + 1]
        opname = opname_for(opcode_table, opcode)
        arg = None
        next_offset = offset + 2
        if has_opcode_argument(opcode_table, opcode):
            arg = extended_arg | raw_arg
            extended_arg = arg << 8 if opname == "EXTENDED_ARG" else 0
        else:
            extended_arg = 0
        if consumes_next_wordcode(opname, version) and offset + 3 < len(code):
            next_offset += 2
        result.append(DecodedOpcode(offset, opcode, opname, arg, next_offset))
        offset = next_offset

    return result


def consumes_next_wordcode(opname: str, version: tuple[int, ...]) -> bool:
    return version[:2] == (3, 12) and opname in {
        "LOAD_CONST__LOAD_FAST",
        "LOAD_FAST__LOAD_CONST",
        "LOAD_FAST__LOAD_FAST",
        "STORE_FAST__LOAD_FAST",
        "STORE_FAST__STORE_FAST",
    }


def scan_bytecode(code: bytes, opcode_table: Any) -> list[DecodedOpcode]:
    result: list[DecodedOpcode] = []
    extended_arg = 0
    offset = 0
    max_steps = len(code)

    for _step in range(max_steps):
        if offset >= len(code):
            return result

        opcode_offset = offset
        opcode = code[offset]
        offset += 1
        opname = opname_for(opcode_table, opcode)
        arg = None
        if has_opcode_argument(opcode_table, opcode):
            if offset + 1 >= len(code):
                raise ValueError("truncated bytecode argument")
            raw_arg = code[offset] | (code[offset + 1] << 8)
            offset += 2
            arg = extended_arg | raw_arg
            extended_arg = arg << 16 if opname == "EXTENDED_ARG" else 0
        else:
            extended_arg = 0
        result.append(DecodedOpcode(opcode_offset, opcode, opname, arg, offset))
    raise ValueError("bytecode scan exceeded byte length")


def opname_for(opcode_table: Any, opcode: int) -> str:
    opnames = getattr(opcode_table, "opname", ())
    if opcode < len(opnames):
        return str(opnames[opcode])
    return f"<{opcode}>"


def has_opcode_argument(opcode_table: Any, opcode: int) -> bool:
    hasarg = getattr(opcode_table, "hasarg", None)
    if hasarg is not None:
        return opcode in hasarg
    return opcode >= int(getattr(opcode_table, "HAVE_ARGUMENT", 90))
