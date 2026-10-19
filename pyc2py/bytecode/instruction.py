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
    borrowed: bool = False


def is_none_load(instruction: Instruction) -> bool:
    return instruction.opname in {"LOAD_CONST", "RETURN_CONST"} and (
        instruction.argval is None
    )


def is_none_return(instruction: Instruction) -> bool:
    return instruction.opname == "RETURN_CONST" and instruction.argval is None
