from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeIs

VersionTuple = tuple[int, ...]


class CompilerVersion(tuple[int, ...]):
    # a version naming where its compiler differs from CPython's (PyPy 3.6 calls)
    facts: frozenset[str]

    def __new__(cls, version: VersionTuple, facts: frozenset[str]) -> CompilerVersion:
        made = super().__new__(cls, version)
        made.facts = facts
        return made

    def __getnewargs__(self) -> tuple[VersionTuple, frozenset[str]]:
        return tuple(self), self.facts


def has_fact(version: object, fact: str) -> bool:
    return fact in getattr(version, "facts", ())


def is_tuple(value: object) -> TypeIs[tuple[Any, ...]]:
    return isinstance(value, tuple)


def is_sequence(value: object) -> TypeIs[list[Any] | tuple[Any, ...]]:
    return isinstance(value, (list, tuple))


ProgressCallback = Callable[[str], None]


def emit_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


class LiveWarningList(list[str]):
    def __init__(
        self,
        progress: ProgressCallback | None = None,
        prefix: str = "warning: ",
    ) -> None:
        super().__init__()
        self.progress = progress
        self.prefix = prefix

    def append(self, item: str) -> None:
        super().append(item)

        if self.progress is not None:
            self.progress(f"{self.prefix}{item}")

    def extend(self, items: Iterable[str]) -> None:
        for item in items:
            self.append(item)

    def __reduce__(self) -> tuple[type[list[str]], tuple[list[str]]]:
        return list, (list(self),)


@dataclass(frozen=True, slots=True)
class PycHeader:
    path: Path
    version: VersionTuple | None
    magic_int: int | None
    timestamp: int | None
    source_size: int | None
    raw_size: int
    source_hash: bytes | None = None
    checks_source: bool = False
    # the opcode module of a compiler that is not CPython's, such as PyPy's
    opcodes: str = ""
    # what the file does not show, such as which compiler wrote it
    doubt: str = ""


@dataclass(frozen=True, slots=True)
class StatedCode:
    # what a container says about the code object it holds
    version: VersionTuple | None = None
    # which code object of a marshalled list (a py2exe script)
    item: int | None = None
    opcodes: str = ""


@dataclass(frozen=True, slots=True)
class PycModule:
    header: PycHeader
    code: Any | None


@dataclass(frozen=True, slots=True)
class DecompiledSource:
    path: Path
    source: str
    strategy: str
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class UnitVerdict:
    name: str
    # verified, different, missing, damaged, error or unverified
    status: str
    reason: str = ""


class BuiltinName(str):
    # a builtin a code object loads by name from a version's common constants
    pass


FAILED_UNIT_STATUSES = frozenset({"different", "missing", "damaged", "error"})
# a file's statuses, from best to worst
FILE_STATUSES = ("passed", "unverified", "failed")
MISSING_BODY = "function body not recovered"


@dataclass(slots=True)
class VerificationReport:
    source_path: Path
    pyc_path: Path
    compiled: bool
    source_valid: bool
    checks: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])
    units: list[UnitVerdict] = field(default_factory=list[UnitVerdict])

    @property
    def status(self) -> str:
        if not self.source_valid or self.errors:
            return "failed"
        if any(unit.status in FAILED_UNIT_STATUSES for unit in self.units):
            return "failed"
        if not self.units or any(unit.status == "unverified" for unit in self.units):
            return "unverified"

        return "passed"

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True, slots=True)
class OutputOptions:
    # each statement on the line it had, unless --compact
    keep_original_lines: bool = True
    write_disassembly: bool = False
    # the interpreters --verify-with names
    verify_with: tuple[Path, ...] = ()


OUTPUT_OPTIONS: ContextVar[OutputOptions] = ContextVar(
    "OUTPUT_OPTIONS", default=OutputOptions()
)
