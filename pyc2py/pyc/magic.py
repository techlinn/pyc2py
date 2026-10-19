import importlib.util
import struct
import sys

from pyc2py.bytecode.formats import pypy_format, releases
from pyc2py.bytecode.pypy import pypy_magics
from pyc2py.types import VersionTuple

MAGIC_VERSION_MAP: dict[int, VersionTuple] = {
    magic: version
    for version, release in releases().items()
    for magic in release.magics
}

UNSUPPORTED_MAGIC_VERSION_MAP: dict[int, str] = {
    39170: "1.0",
    39171: "1.1 or 1.2",
    11913: "1.3",
    5892: "1.4",
    20121: "1.5",
    50428: "1.6",
    50823: "2.0",
    60202: "2.1",
    60717: "2.2",
    62011: "2.3",
    62061: "2.4",
    62131: "2.5",
    62161: "2.6",
}

# PyPy releases that no checked download exists for, or betas
UNSUPPORTED_PYPY_MAGICS: dict[int, str] = {
    3187: "3.2",
    64: "3.3",
    112: "3.5",
    160: "3.6 (a beta)",
    320: "3.9 (an early release)",
}


MAGIC_RANGES: dict[VersionTuple, tuple[int, int]] = {
    version: release.magic_range for version, release in releases().items()
}


def version_in_development(magic_int: int) -> VersionTuple | None:
    for version, (first, last) in MAGIC_RANGES.items():
        if first <= magic_int <= last:
            return version

    return None


def describe_magic(magic_int: int) -> str:
    pypy = UNSUPPORTED_PYPY_MAGICS.get(magic_int)
    if pypy is not None:
        return (
            f"PyPy {pypy} bytecode is not supported (pyc2py reads PyPy 2.7 and"
            " 3.6 or newer, the releases it can be checked against)"
        )

    unsupported = UNSUPPORTED_MAGIC_VERSION_MAP.get(magic_int)
    if unsupported is not None:
        return (
            f"Python {unsupported} bytecode is not supported"
            " (pyc2py reads Python 2.7 and 3.0 or newer)"
        )

    version = version_in_development(magic_int)
    if version is None:
        return (
            f"unknown pyc magic {magic_int}: not a magic number of CPython 2.7 or"
            " 3.x; if you know the Python version, give it with --python-version"
        )

    name = ".".join(map(str, version))
    released = [magic for magic, known in MAGIC_VERSION_MAP.items() if known == version]
    newer = all(magic_int > magic for magic in released)
    kind = "a newer pre-release" if newer else "a pre-release or patched build"
    return (
        f"magic {magic_int} is from {kind} of Python {name}; pyc2py reads the"
        f" Python {name} format of magic {', '.join(map(str, released))}, and"
        f" --python-version {name} reads this file as that format anyway"
    )


CURRENT_MAGIC_INT = struct.unpack("<H", importlib.util.MAGIC_NUMBER[:2])[0]


def read_magic_int(data: bytes) -> int:
    if len(data) < 2:
        raise ValueError("pyc data is shorter than the magic word")

    return struct.unpack("<H", data[:2])[0]


def find_version(magic_int: int) -> tuple[int, ...] | None:
    if magic_int == CURRENT_MAGIC_INT:
        return sys.version_info[:2]

    return MAGIC_VERSION_MAP.get(magic_int)


def find_format(magic_int: int) -> tuple[tuple[int, ...], str] | None:
    # the language version and, for a compiler that is not CPython's, its opcodes
    opcodes = pypy_magics().get(magic_int)
    if opcodes is not None:
        return pypy_format(opcodes).version, opcodes

    version = find_version(magic_int)
    return None if version is None else (version, "")
