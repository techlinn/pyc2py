import struct
from dataclasses import dataclass

from pyc2py.pyc.flags import parse_pyc_flags


@dataclass(frozen=True, slots=True)
class SourceStamp:
    timestamp: int | None = None
    size: int | None = None
    source_hash: bytes | None = None
    checks_source: bool = False


def read_header_metadata(data: bytes, version: tuple[int, ...]) -> SourceStamp:
    if (3, 3) <= version < (3, 7):
        return SourceStamp(read_timestamp(data), read_source_size(data))
    if version >= (3, 7):
        return read_pep552_metadata(data)

    return SourceStamp(read_timestamp(data))


def read_pep552_metadata(data: bytes) -> SourceStamp:
    if len(data) < 16:
        return SourceStamp()

    flags = parse_pyc_flags(struct.unpack("<I", data[4:8])[0])
    if flags.is_hash_based:
        return SourceStamp(source_hash=data[8:16], checks_source=flags.checks_source)

    return SourceStamp(
        struct.unpack("<I", data[8:12])[0], struct.unpack("<I", data[12:16])[0]
    )


def read_timestamp(data: bytes) -> int | None:
    if len(data) < 8:
        return None

    return struct.unpack("<I", data[4:8])[0]


def read_source_size(data: bytes) -> int | None:
    if len(data) < 12:
        return None

    return struct.unpack("<I", data[8:12])[0]
