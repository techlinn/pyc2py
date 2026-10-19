import marshal
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import CodeType

from pyc2py.bytecode.formats import pypy_format
from pyc2py.bytecode.opcode_table import get_opcode_table, is_unknown_opname
from pyc2py.bytecode.pypy import pypy_shared_magics, written_by_pypy
from pyc2py.constants import MAX_PYC_SIZE
from pyc2py.pyc.code import PycCode
from pyc2py.pyc.flags import payload_offset_for_version
from pyc2py.pyc.header import SourceStamp, read_header_metadata
from pyc2py.pyc.magic import describe_magic, find_format, read_magic_int
from pyc2py.pyc.marshal_reader import listed_item, load_marshal_code
from pyc2py.types import (
    CompilerVersion,
    PycHeader,
    PycModule,
    StatedCode,
    VersionTuple,
)

# TYPE_CODE and TYPE_LIST, with or without FLAG_REF
RAW_CODE_MARKS = (ord("c"), ord("c") | 0x80)
RAW_LIST_MARKS = (ord("["), ord("[") | 0x80)


class PycLoadError(ValueError):
    pass


def load_pyc(path: Path, version: VersionTuple | None = None) -> PycModule:
    return load_pyc_data(read_input(path), path, version)


def read_input(path: Path) -> bytes:
    if not path.exists():
        raise PycLoadError(f"cannot read {path}: no such file")
    if not path.is_file():
        raise PycLoadError(f"cannot read {path}: not a file")
    if path.stat().st_size > MAX_PYC_SIZE:
        raise PycLoadError(f"cannot read {path}: larger than pyc2py reads")

    return path.read_bytes()


def load_pyc_data(
    data: bytes,
    path: Path,
    version: VersionTuple | None = None,
    stated: StatedCode | None = None,
) -> PycModule:
    # version is forced by the user; stated is what the file's container says
    stated = stated or StatedCode()
    item = stated.item

    if version is None:
        header, payload = read_pyc_header(data, path, stated.version, item is not None)
        if stated.opcodes and header.magic_int is None:
            header = replace(header, opcodes=stated.opcodes)
    else:
        header, payload = read_pyc_header(data, path, version, item is not None)

    if header.version == sys.version_info[:2] and not header.opcodes:
        native = load_native_code(payload, item)
        if native is not None:
            return PycModule(header=header, code=native)

    code = load_portable_code(payload, header, item)
    opcodes = pypy_shared_magics().get(header.magic_int or 0)
    if opcodes is None or version is not None:
        return PycModule(header=header, code=code)

    return read_shared_magic(payload, header, item, code, opcodes)


def read_shared_magic(
    payload: bytes, header: PycHeader, item: int | None, code: PycCode, opcodes: str
) -> PycModule:
    # early PyPy releases wrote CPython's magic with PyPy's own opcodes
    version = header.version
    assert version is not None
    cpython = get_opcode_table(version).opname
    cpython_map = {
        number: name
        for number, name in enumerate(cpython)
        if not is_unknown_opname(name)
    }
    verdict = written_by_pypy(code, opcodes, cpython_map)
    if verdict:
        header = replace(header, opcodes=opcodes)
        return PycModule(header=header, code=load_portable_code(payload, header, item))

    if verdict is None:
        name = ".".join(map(str, version))
        header = replace(
            header,
            doubt=f"magic {header.magic_int} is written by CPython {name} and by"
            f" early PyPy {name} releases, and nothing in this file tells which;"
            f" it was read as CPython {name}",
        )

    return PycModule(header=header, code=code)


def reading_version(header: PycHeader) -> VersionTuple | None:
    # the version whose readers recover the code; a PyPy format names its own
    if header.opcodes:
        found = pypy_format(header.opcodes)
        return CompilerVersion(found.reads_as, found.facts)

    return header.version


def read_pyc_header(
    data: bytes, path: Path, version: VersionTuple | None, listed: bool = False
) -> tuple[PycHeader, bytes]:
    if len(data) > MAX_PYC_SIZE:
        raise ValueError(f"pyc file exceeds local size limit: {path}")

    magic_int: int | None = None
    opcodes = ""
    stamp = SourceStamp()

    if has_pyc_header(data):
        magic_int = read_magic_int(data)

        if version is None:
            found = find_format(magic_int)
            if found is not None:
                version, opcodes = found

        if version is None:
            reason = describe_magic(magic_int)
            raise PycLoadError(f"cannot read pyc file {path}: {reason}")

        offset = payload_offset_for_version(version)
        if len(data) <= offset:
            raise PycLoadError(f"cannot read pyc file {path}: the header is truncated")

        stamp = read_header_metadata(data, version)

    elif data[:1] and data[0] in (RAW_LIST_MARKS if listed else RAW_CODE_MARKS):
        if version is None:
            raise PycLoadError(
                f"cannot read {path}: it is a code object without a pyc header,"
                " so its Python version is unknown; give it with --python-version"
            )

        offset = 0

    else:
        raise PycLoadError(
            f"cannot read {path}: not a pyc file or code object"
            f" (it starts with {data[:4].hex(' ') or 'nothing'})"
        )

    header = PycHeader(
        path=path,
        version=(version[0], version[1]),
        magic_int=magic_int,
        timestamp=stamp.timestamp,
        source_size=stamp.size,
        raw_size=len(data),
        source_hash=stamp.source_hash,
        checks_source=stamp.checks_source,
        opcodes=opcodes,
    )
    return header, data[offset:]


def has_pyc_header(data: bytes) -> bool:
    return len(data) >= 4 and data[2:4] == b"\r\n"


def load_native_code(payload: bytes, item: int | None = None) -> CodeType | None:
    try:
        code = marshal.loads(payload)
    except (EOFError, ValueError, TypeError, SystemError):
        return None

    if item is not None:
        code = listed_item(code, item)

    return code if isinstance(code, CodeType) else None


def load_portable_code(
    payload: bytes, header: PycHeader, item: int | None = None
) -> PycCode:
    assert header.version is not None

    try:
        return load_marshal_code(payload, header.version, item, header.opcodes)
    except (EOFError, UnicodeDecodeError, ValueError, struct.error) as error:
        reason = str(error) or type(error).__name__
        raise PycLoadError(f"cannot read pyc file {header.path}: {reason}") from error
