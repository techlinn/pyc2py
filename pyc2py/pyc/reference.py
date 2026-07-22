from dataclasses import dataclass, field
from typing import Any
from pyc2py.constants import MAX_MARSHAL_OBJECTS

FLAG_REF = 0x80
TYPE_REF = ord("r")

@dataclass(slots=True)
class ReferenceTable:
    values: list[Any] = field(default_factory=list)
    max_values: int = MAX_MARSHAL_OBJECTS

    def reserve(self, enabled: bool) -> int | None:
        if not enabled:
            return None

        self.check_capacity()
        self.values.append(None)
        return len(self.values) - 1

    def remember(self, value: Any, enabled: bool) -> Any:
        if enabled:
            self.check_capacity()
            self.values.append(value)
        return value

    def set_reserved(self, index: int | None, value: Any) -> Any:
        if index is not None:
            if index < 0 or index >= len(self.values):
                raise ValueError("marshal reference reservation is invalid")
            self.values[index] = value

        return value

    def get(self, index: int) -> Any:
        if index < 0 or index >= len(self.values):
            raise ValueError("marshal reference index is invalid")

        value = self.values[index]
        if value is None:
            raise ValueError("marshal reference points to unresolved object")

        return value

    def check_capacity(self) -> None:
        if len(self.values) >= self.max_values:
            raise ValueError("marshal reference table exceeded local limit")

def split_type_code(code: int) -> tuple[int, bool]:
    return code & ~FLAG_REF, bool(code & FLAG_REF)
