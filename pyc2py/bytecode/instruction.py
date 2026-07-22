from dataclasses import dataclass
from typing import Any

IGNORED_BEHAVIOR_OPNAMES = frozenset(
    {
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
        "RESUME",
    }
)


@dataclass(frozen=True, slots=True)
class Instruction:
    offset: int
    opname: str
    arg: int | None
    argval: Any
    argrepr: str
    starts_line: int | None
    is_jump_target: bool
