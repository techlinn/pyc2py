from collections.abc import Sequence
from dataclasses import dataclass, field

from pyc2py.bytecode.instruction import Instruction
from pyc2py.constants import MAX_CFG_EDGES


@dataclass(frozen=True, slots=True)
class ExceptionTableEntry:
    start_offset: int
    end_offset: int
    target_offset: int
    depth: int = 0
    lasti: bool = False


@dataclass(slots=True)
class ExceptionTableValidation:
    entry_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def checks(self) -> tuple[str, ...]:
        return (f"exception table entry count: {self.entry_count}",)


def parse_exception_table(
    data: bytes, max_entries: int = MAX_CFG_EDGES
) -> tuple[ExceptionTableEntry, ...]:
    if max_entries < 1:
        raise ValueError("max_entries must be positive")

    entries: list[ExceptionTableEntry] = []
    offset = 0
    for _index in range(max_entries):
        if offset >= len(data):
            return tuple(entries)

        start, offset = read_exception_varint(data, offset)
        length, offset = read_exception_varint(data, offset)
        target, offset = read_exception_varint(data, offset)
        depth_lasti, offset = read_exception_varint(data, offset)
        start_offset = start * 2
        length_bytes = length * 2
        entries.append(
            ExceptionTableEntry(
                start_offset=start_offset,
                end_offset=start_offset + length_bytes,
                target_offset=target * 2,
                depth=depth_lasti >> 1,
                lasti=bool(depth_lasti & 1),
            )
        )
    raise ValueError("exception table exceeded local entry limit")


def read_exception_varint(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise ValueError("truncated exception table entry")

    value = data[offset] & 0x3F
    has_more = bool(data[offset] & 0x40)
    offset += 1
    while has_more:
        if offset >= len(data):
            raise ValueError("truncated exception table varint")
        value = (value << 6) | (data[offset] & 0x3F)
        has_more = bool(data[offset] & 0x40)
        offset += 1

    return value, offset


def validate_exception_table(
    data: bytes,
    instruction_offsets: set[int],
    code_size: int,
) -> ExceptionTableValidation:
    entries = parse_exception_table(data)
    warnings: list[str] = []
    previous_start = -1

    valid_end_offsets = set(instruction_offsets)
    valid_end_offsets.add(code_size)

    for entry in entries:
        warnings.extend(
            validate_exception_entry(
                entry, instruction_offsets, valid_end_offsets, code_size
            )
        )

        if entry.start_offset < previous_start:
            warnings.append(
                f"exception table entries are not sorted at {entry.start_offset}"
            )

        previous_start = entry.start_offset

    return ExceptionTableValidation(entry_count=len(entries), warnings=warnings)


def validate_exception_entry(
    entry: ExceptionTableEntry,
    instruction_offsets: set[int],
    valid_end_offsets: set[int],
    code_size: int,
) -> list[str]:
    warnings: list[str] = []

    if entry.start_offset >= entry.end_offset:
        warnings.append(
            f"exception range is empty or reversed: {entry.start_offset}-{entry.end_offset}"
        )

    if entry.start_offset not in instruction_offsets:
        warnings.append(f"exception start offset is not decoded: {entry.start_offset}")

    if entry.end_offset not in valid_end_offsets:
        warnings.append(f"exception end offset is not a boundary: {entry.end_offset}")

    if entry.target_offset not in instruction_offsets:
        warnings.append(
            f"exception target offset is not decoded: {entry.target_offset}"
        )

    if entry.end_offset > code_size:
        warnings.append(
            f"exception range extends past bytecode: {entry.end_offset} > {code_size}"
        )

    if entry.depth < 0:
        warnings.append(f"exception stack depth is negative: {entry.depth}")

    return warnings


@dataclass(frozen=True, slots=True)
class LineEntry:
    offset: int
    line: int


@dataclass(slots=True)
class LineValidation:
    entry_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def checks(self) -> tuple[str, ...]:
        return (f"line entry count: {self.entry_count}",)


def line_entries(instructions: list[Instruction]) -> tuple[LineEntry, ...]:
    entries: list[LineEntry] = []
    for instruction in instructions:
        if instruction.starts_line is None:
            continue
        if instruction.starts_line < 0:
            continue
        entries.append(
            LineEntry(offset=instruction.offset, line=instruction.starts_line)
        )
    return tuple(entries)


def validate_line_entries(instructions: list[Instruction]) -> LineValidation:
    entries = line_entries(instructions)
    warnings: list[str] = []
    previous_offset = -1
    offsets = {instruction.offset for instruction in instructions}

    for entry in entries:
        if entry.offset not in offsets:
            warnings.append(f"line entry offset is not decoded: {entry.offset}")

        if entry.offset < previous_offset:
            warnings.append(f"line entries are not offset-sorted at {entry.offset}")

        previous_offset = entry.offset

    return LineValidation(entry_count=len(entries), warnings=warnings)


IGNORABLE_OPS = frozenset({"CACHE", "SET_LINENO", "NOP"})


def make_offset_index(instructions: Sequence[Instruction]) -> dict[int, int]:
    return {instruction.offset: index for index, instruction in enumerate(instructions)}


def skip_ignorable_instructions(
    instructions: Sequence[Instruction],
    start_index: int,
    end_index: int,
) -> int:
    cursor = start_index
    while cursor < end_index and instructions[cursor].opname in IGNORABLE_OPS:
        cursor += 1
    return cursor
