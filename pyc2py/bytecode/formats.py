from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from pyc2py.types import VersionTuple

# versions holds python_X_Y per CPython version and pypy_* per PyPy format
VERSIONS = "pyc2py.bytecode.versions"


@dataclass(frozen=True, slots=True)
class Release:
    version: tuple[int, int]
    # the magic of each release, and the span its pre-releases took
    magics: tuple[int, ...]
    magic_range: tuple[int, int]
    opmap: dict[int, str]
    common_constants: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class PyPyFormat:
    name: str
    # the language version
    version: VersionTuple
    magics: tuple[int, ...]
    # CPython's magic, which some early releases wrote
    cpython_magics: tuple[int, ...]
    # the CPython version whose compiler output its control flow follows
    reads_as: VersionTuple
    # where its code differs from reads_as's, such as legacy_calls
    facts: frozenset[str]
    # opcodes below HAVE_ARGUMENT whose argument byte counts
    argument_ops: frozenset[str]
    # the CPython version whose argument and jump rules the bytes follow
    layout: tuple[int, int]
    # code-object fields marshalled after CPython 3.10's
    code_extras: tuple[str, ...]
    # co_linetable holds PyPy's position table
    positions: bool
    opmap: dict[int, str]
    relative_jumps: frozenset[str]
    absolute_jumps: frozenset[str]


def format_modules(prefix: str) -> dict[str, ModuleType]:
    package = import_module(VERSIONS)
    return {
        found.name: import_module(f"{VERSIONS}.{found.name}")
        for found in iter_modules(package.__path__)
        if found.name.startswith(prefix)
    }


@cache
def releases() -> dict[tuple[int, int], Release]:
    found: dict[tuple[int, int], Release] = {}

    for name, module in format_modules("python_").items():
        _, major, minor = name.split("_")
        version = (int(major), int(minor))
        found[version] = Release(
            version=version,
            magics=module.MAGICS,
            magic_range=module.MAGIC_RANGE,
            opmap=module.OPMAP,
            common_constants=getattr(module, "COMMON_CONSTANTS", ()),
        )

    return dict(sorted(found.items()))


@cache
def pypy_formats() -> dict[str, PyPyFormat]:
    return {
        name: PyPyFormat(
            name=module.NAME,
            version=module.VERSION,
            magics=module.MAGICS,
            cpython_magics=module.CPYTHON_MAGICS,
            reads_as=module.READS_AS,
            facts=module.FACTS,
            argument_ops=module.ARGUMENT_OPS,
            layout=module.LAYOUT,
            code_extras=module.CODE_EXTRAS,
            positions=module.POSITIONS,
            opmap=module.OPMAP,
            relative_jumps=module.RELATIVE_JUMPS,
            absolute_jumps=module.ABSOLUTE_JUMPS,
        )
        for name, module in format_modules("pypy_").items()
    }


def pypy_format(opcodes: str) -> PyPyFormat:
    return pypy_formats()[opcodes]
