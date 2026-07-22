from dataclasses import dataclass

HASH_BASED_FLAG = 0x01


@dataclass(frozen=True, slots=True)
class PycFlags:
    raw: int

    @property
    def is_hash_based(self) -> bool:
        return bool(self.raw & HASH_BASED_FLAG)


def parse_pyc_flags(raw: int) -> PycFlags:
    if raw < 0:
        raise ValueError("pyc flags must not be negative")
    return PycFlags(raw=raw)


def payload_offset_for_version(version: tuple[int, ...]) -> int:
    if version >= (3, 7):
        return 16
    if version >= (3, 3):
        return 12
    return 8
