import importlib.util
import marshal
import struct
import sys
from pathlib import Path
from types import CodeType

from pyc2py.constants import MAX_PYC_SIZE
from pyc2py.pyc.flags import parse_pyc_flags, payload_offset_for_version
from pyc2py.pyc.header import read_header
from pyc2py.pyc.marshal_reader import load_marshal_code
from pyc2py.types import PycHeader, PycModule


def load_pyc(path: Path) -> PycModule:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    if path.stat().st_size > MAX_PYC_SIZE:
        raise ValueError(f"pyc file exceeds local size limit: {path}")

    native = load_native_pyc_with_error(path)
    if native is not None:
        return native
    return load_legacy_pyc_with_error(path)


def load_native_pyc_with_error(path: Path) -> PycModule | None:
    try:
        return load_native_pyc(path)
    except (EOFError, UnicodeDecodeError, ValueError, TypeError, struct.error):
        return None


def load_legacy_pyc_with_error(path: Path) -> PycModule:
    try:
        return load_legacy_pyc(path)
    except (EOFError, UnicodeDecodeError, ValueError, struct.error):
        return make_failed_module(path)


def make_failed_module(path: Path) -> PycModule:
    header = read_header(path)
    return PycModule(header=header, code=None)


def load_native_pyc(path: Path) -> PycModule | None:
    data = path.read_bytes()
    if len(data) < 16:
        return None
    if data[:4] != importlib.util.MAGIC_NUMBER:
        return None

    code = marshal.loads(data[16:])
    if not isinstance(code, CodeType):
        raise ValueError(f"native pyc payload is not a code object: {path}")

    header = PycHeader(
        path=path,
        version=sys.version_info[:2],
        magic_int=struct.unpack("<H", data[:2])[0],
        timestamp=read_native_timestamp(data),
        source_size=read_native_source_size(data),
        raw_size=len(data),
    )
    return PycModule(header=header, code=code)


def read_native_timestamp(data: bytes) -> int | None:
    flags = parse_pyc_flags(struct.unpack("<I", data[4:8])[0])
    if flags.is_hash_based:
        return None
    return struct.unpack("<I", data[8:12])[0]


def read_native_source_size(data: bytes) -> int | None:
    flags = parse_pyc_flags(struct.unpack("<I", data[4:8])[0])
    if flags.is_hash_based:
        return None
    return struct.unpack("<I", data[12:16])[0]


def load_legacy_pyc(path: Path) -> PycModule:
    header = read_header(path)
    if header.version is None:
        raise ValueError(f"unknown pyc magic: {path}")

    data = path.read_bytes()
    payload_offset = legacy_payload_offset(header.version)
    if len(data) <= payload_offset:
        raise ValueError(f"legacy pyc is too short: {path}")

    code = load_marshal_code(data[payload_offset:], header.version)
    return PycModule(header=header, code=code)


def legacy_payload_offset(version: tuple[int, ...]) -> int:
    return payload_offset_for_version(version)
