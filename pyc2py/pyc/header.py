import struct
from pathlib import Path

from pyc2py.pyc.flags import parse_pyc_flags
from pyc2py.pyc.magic import find_version, read_magic_int
from pyc2py.types import PycHeader


def read_header(path: Path) -> PycHeader:
    data = path.read_bytes()
    magic_int = read_magic_int(data)
    version = find_version(magic_int)
    timestamp, source_size = read_header_metadata(data, version)

    return PycHeader(
        path=path,
        version=version,
        magic_int=magic_int,
        timestamp=timestamp,
        source_size=source_size,
        raw_size=len(data),
    )


def read_header_metadata(
    data: bytes,
    version: tuple[int, ...] | None,
) -> tuple[int | None, int | None]:
    if version is None:
        return read_timestamp(data), read_source_size(data)
    if version >= (3, 7):
        return read_pep552_metadata(data)
    if version >= (3, 3):
        return read_timestamp(data), read_source_size(data)
    return read_timestamp(data), None


def read_pep552_metadata(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 8:
        return None, None
    flags = parse_pyc_flags(struct.unpack("<I", data[4:8])[0])
    if flags.is_hash_based:
        return None, None

    if len(data) < 16:
        return None, None
    return struct.unpack("<I", data[8:12])[0], struct.unpack("<I", data[12:16])[0]


def read_timestamp(data: bytes) -> int | None:
    if len(data) < 8:
        return None
    return struct.unpack("<I", data[4:8])[0]


def read_source_size(data: bytes) -> int | None:
    if len(data) < 12:
        return None
    return struct.unpack("<I", data[8:12])[0]
