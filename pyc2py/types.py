from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VersionTuple = tuple[int, ...]
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


@dataclass(frozen=True, slots=True)
class PycHeader:
    path: Path
    version: VersionTuple | None
    magic_int: int | None
    timestamp: int | None
    source_size: int | None
    raw_size: int


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
class VerificationReport:
    source_path: Path
    pyc_path: Path
    compiled: bool
    source_valid: bool
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.compiled and self.source_valid and not self.errors
