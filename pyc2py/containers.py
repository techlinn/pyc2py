import re
import struct
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pyc2py.constants import MAX_PYC_SIZE
from pyc2py.pyc.code import PycCode
from pyc2py.pyc.magic import find_version
from pyc2py.pyc.marshal_reader import MarshalReader
from pyc2py.types import StatedCode, VersionTuple, is_sequence

PYC_SUFFIXES = (".pyc", ".pyo")

# PyInstaller (bootloader/src/pyi_archive.h, PyInstaller/archive/readers.py)
PYINSTALLER_COOKIE = b"MEI\014\013\012\013\016"
PYINSTALLER_COOKIE_LAYOUT = "!8sIIII64s"
PYINSTALLER_ENTRY_LAYOUT = "!IIIIBc"
PYINSTALLER_SCRIPT = "s"
PYINSTALLER_MODULE = "m"
PYINSTALLER_PYZ = "z"
PYZ_MAGIC = b"PYZ\0"
PYZ_CODE_KINDS = {0: "module", 1: "package"}

# py2exe's PYTHONSCRIPT: this header, the zip path, a zero byte, then code
PY2EXE_SCRIPT_MAGIC = 0x78563412
PY2EXE_SCRIPT_LAYOUT = "<IIII"
PY2EXE_RESOURCE = "PYTHONSCRIPT"
PYTHON_DLL = re.compile(rb"python(\d)(\d{1,2})\.dll", re.IGNORECASE)


class ContainerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Member:
    # where the member's output goes, relative to the output folder
    path: PurePosixPath
    data: bytes
    stated: StatedCode = field(default_factory=StatedCode)
    # why this member could not be taken out of the container
    failure: str | None = None


def container_kind(path: Path) -> str | None:
    # an executable may carry a zip, so it is told apart first
    with path.open("rb") as handle:
        head = handle.read(2)
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 4096))
        tail = handle.read()

    if PYINSTALLER_COOKIE in tail:
        return "PyInstaller bundle"
    if head == b"MZ" and find_resource(path.read_bytes(), PY2EXE_RESOURCE):
        return "py2exe executable"
    if zipfile.is_zipfile(path):
        return "zip archive"

    return None


def container_members(
    path: Path, version: VersionTuple | None = None
) -> Iterator[Member]:
    # version: the user's --python-version, for a container that does not say
    kind = container_kind(path)
    if kind == "zip archive":
        yield from zip_members(path)
    elif kind == "PyInstaller bundle":
        yield from pyinstaller_members(path.read_bytes())
    elif kind == "py2exe executable":
        yield from py2exe_members(path.read_bytes(), version)
    else:
        raise ContainerError(f"{path} is not an archive pyc2py can read")


def zip_members(path: Path) -> Iterator[Member]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if member.is_dir() or relative.suffix.lower() not in PYC_SUFFIXES:
                continue
            if member.file_size > MAX_PYC_SIZE:
                raise ContainerError(f"pyc file exceeds local size limit: {relative}")

            yield Member(safe_path(relative, path), archive.read(member))


def safe_path(relative: PurePosixPath, container: object) -> PurePosixPath:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ContainerError(f"unsafe member path in {container}: {relative}")

    return relative


def inflated(data: bytes) -> bytes:
    # a zlib stream larger than any pyc is not inflated
    inflater = zlib.decompressobj()

    try:
        out = inflater.decompress(data, MAX_PYC_SIZE + 1)
    except zlib.error as error:
        raise ContainerError(f"damaged compressed member: {error}") from error

    if len(out) > MAX_PYC_SIZE:
        raise ContainerError("compressed member exceeds local size limit")

    return out


def pyinstaller_members(data: bytes) -> Iterator[Member]:
    cookie_at = data.rfind(PYINSTALLER_COOKIE)
    cookie_size = struct.calcsize(PYINSTALLER_COOKIE_LAYOUT)
    if cookie_at < 0 or cookie_at + cookie_size > len(data):
        raise ContainerError("the PyInstaller cookie is truncated")

    _, length, table_at, table_size, python, _ = struct.unpack_from(
        PYINSTALLER_COOKIE_LAYOUT, data, cookie_at
    )
    start = cookie_at + cookie_size - length
    if start < 0 or start + table_at + table_size > len(data):
        raise ContainerError("the PyInstaller table lies outside the file")

    # PyInstaller 4 and later write 312 for 3.12; older releases wrote 27
    stated = divmod(python, 100) if python >= 100 else divmod(python, 10)

    for name, kind, blob in pyinstaller_entries(data, start, table_at, table_size):
        if kind == PYINSTALLER_PYZ:
            yield from pyz_members(blob)
        elif kind in (PYINSTALLER_SCRIPT, PYINSTALLER_MODULE):
            yield Member(module_path(name, "module"), blob, StatedCode(stated))


def pyinstaller_entries(
    data: bytes, start: int, table_at: int, table_size: int
) -> Iterator[tuple[str, str, bytes]]:
    entry_size = struct.calcsize(PYINSTALLER_ENTRY_LAYOUT)
    position, end = start + table_at, start + table_at + table_size
    while position + entry_size <= end:
        length, offset, stored, _, compressed, kind = struct.unpack_from(
            PYINSTALLER_ENTRY_LAYOUT, data, position
        )
        if length < entry_size:
            raise ContainerError("damaged PyInstaller table entry")

        name = data[position + entry_size : position + length].rstrip(b"\0")
        blob = data[start + offset : start + offset + stored]
        if len(blob) != stored:
            raise ContainerError("a PyInstaller entry lies outside the file")

        yield (
            name.decode("utf-8", "replace"),
            kind.decode("latin-1"),
            inflated(blob) if compressed else blob,
        )
        position += length


def pyz_members(pyz: bytes) -> Iterator[Member]:
    if pyz[:4] != PYZ_MAGIC or len(pyz) < 12:
        raise ContainerError("the PYZ archive has no PYZ header")

    version = find_version(int.from_bytes(pyz[4:6], "little"))
    if version is None:
        raise ContainerError("the PYZ archive's pyc magic is not one pyc2py knows")

    (table_at,) = struct.unpack_from("!I", pyz, 8)
    for name, code_kind, offset, length in pyz_table(pyz[table_at:], version):
        kind = PYZ_CODE_KINDS.get(code_kind)
        if kind is not None:
            yield pyz_member(
                module_path(name, kind), pyz[offset : offset + length], version
            )


def pyz_table(data: bytes, version: VersionTuple) -> list[tuple[str, int, int, int]]:
    try:
        table, _ = MarshalReader(data, version).read_object(0)
    except (EOFError, ValueError, TypeError, struct.error, RecursionError) as error:
        raise ContainerError(f"damaged PYZ table: {error}") from error

    items = (
        list(cast("dict[Any, Any]", table).items())
        if isinstance(table, dict)
        else table
    )
    if not isinstance(items, list):
        raise ContainerError("damaged PYZ table")

    entries: list[tuple[str, int, int, int]] = []

    for item in cast("list[Any]", items):
        if not (is_sequence(item) and len(item) == 2):
            raise ContainerError("damaged PYZ table entry")

        name, value = item
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        if not (
            isinstance(name, str)
            and is_sequence(value)
            and len(value) == 3
            and all(isinstance(number, int) and number >= 0 for number in value)
        ):
            raise ContainerError("damaged PYZ table entry")

        entries.append((name, *value))

    return entries


def pyz_member(path: PurePosixPath, blob: bytes, version: VersionTuple) -> Member:
    try:
        return Member(path, inflated(blob), StatedCode(version))
    except ContainerError as error:
        # PyInstaller before 6.0 could encrypt the PYZ entries (--key)
        reason = f"{error}; the entry may be encrypted (PyInstaller before 6.0 --key)"
        return Member(path, b"", StatedCode(version), reason)


def module_path(name: str, kind: str) -> PurePosixPath:
    if not all(name.split(".")):
        raise ContainerError(f"damaged module name: {name!r}")

    parts = PurePosixPath(*name.split("."))
    path = parts / "__init__.pyc" if kind == "package" else parts.with_suffix(".pyc")
    return safe_path(path, name)


def py2exe_members(data: bytes, version: VersionTuple | None) -> Iterator[Member]:
    script = find_resource(data, PY2EXE_RESOURCE)
    size = struct.calcsize(PY2EXE_SCRIPT_LAYOUT)
    if script is None or len(script) < size:
        raise ContainerError("the PYTHONSCRIPT resource is missing or truncated")

    magic, _, _, length = struct.unpack_from(PY2EXE_SCRIPT_LAYOUT, script)
    if magic != PY2EXE_SCRIPT_MAGIC:
        raise ContainerError("the PYTHONSCRIPT resource has the wrong magic")

    zip_end = script.find(b"\0", size)
    if zip_end < 0:
        raise ContainerError("the PYTHONSCRIPT resource is truncated")

    codes = script[zip_end + 1 : zip_end + 1 + length]
    found = PYTHON_DLL.search(data)
    stated = version or ((int(found[1]), int(found[2])) if found else None)
    if stated is None:
        raise ContainerError(
            "the py2exe executable does not name its python DLL;"
            " give the version with --python-version"
        )

    for index, name in enumerate(script_names(codes, stated)):
        yield Member(PurePosixPath(f"{name}.pyc"), codes, StatedCode(stated, index))


def script_names(codes: bytes, version: VersionTuple) -> list[str]:
    try:
        loaded, _ = MarshalReader(codes, version).read_object(0)
    except (EOFError, ValueError, TypeError, struct.error, RecursionError) as error:
        raise ContainerError(f"damaged PYTHONSCRIPT code: {error}") from error

    if not isinstance(loaded, list):
        raise ContainerError("the PYTHONSCRIPT resource holds no list of code")

    names: list[str] = []

    for index, code in enumerate(cast("list[Any]", loaded)):
        stem = PurePosixPath(code.co_filename).stem if isinstance(code, PycCode) else ""
        names.append(stem if stem.isidentifier() else f"script{index}")

    return names


def find_resource(data: bytes, type_name: str) -> bytes | None:
    # the first resource of a named type in a PE file
    try:
        return resource_data(data, type_name)
    except (struct.error, IndexError, UnicodeDecodeError):
        return None


def resource_data(data: bytes, type_name: str) -> bytes | None:
    sections, directory = pe_resource_directory(data)
    base = None if directory is None else rva_to_offset(directory, sections)
    if base is None:
        return None

    found = resource_entry(data, base, base, type_name)

    # below the type come the name and the language, and then the data
    for _level in range(2):
        if found is None or not found[1]:
            break

        found = resource_entry(data, base, base + found[0], None)

    if found is None or found[1]:
        return None

    rva, size = struct.unpack_from("<II", data, base + found[0])
    offset = rva_to_offset(rva, sections)
    return None if offset is None else data[offset : offset + size]


def pe_resource_directory(data: bytes) -> tuple[list[tuple[int, int, int]], int | None]:
    header = struct.unpack_from("<I", data, 0x3C)[0]
    if data[header : header + 4] != b"PE\0\0":
        return [], None

    sections_count, _, _, _, optional_size = struct.unpack_from(
        "<HIIIH", data, header + 6
    )
    optional = header + 24
    pe32_plus = struct.unpack_from("<H", data, optional)[0] == 0x20B
    directories = optional + (112 if pe32_plus else 96)
    resource_rva = struct.unpack_from("<I", data, directories + 2 * 8)[0]
    section_table = optional + optional_size
    sections: list[tuple[int, int, int]] = []

    for index in range(sections_count):
        virtual_size, virtual_address, raw_size, raw_at = struct.unpack_from(
            "<IIII", data, section_table + index * 40 + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_at))

    return sections, resource_rva or None


def rva_to_offset(rva: int, sections: list[tuple[int, int, int]]) -> int | None:
    for address, size, raw_at in sections:
        if address <= rva < address + size:
            return rva - address + raw_at

    return None


def resource_entry(
    data: bytes, base: int, table: int, name: str | None
) -> tuple[int, bool] | None:
    # (offset from the resource base, whether it is a directory)
    named, numbered = struct.unpack_from("<HH", data, table + 12)
    for index in range(named + numbered):
        key, target = struct.unpack_from("<II", data, table + 16 + index * 8)

        if name is not None:
            if not key & 0x80000000:
                continue

            length = struct.unpack_from("<H", data, base + (key & 0x7FFFFFFF))[0]
            text_at = base + (key & 0x7FFFFFFF) + 2
            if data[text_at : text_at + 2 * length].decode("utf-16-le") != name:
                continue

        return target & 0x7FFFFFFF, bool(target & 0x80000000)

    return None
