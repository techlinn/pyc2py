from dataclasses import dataclass
from typing import Any, Protocol
from pyc2py.constants import MAX_MARSHAL_COLLECTION_ITEMS
from pyc2py.pyc.primitives import read_exact, read_int32, read_size, read_uint8

class ObjectReader(Protocol):
    def read_object(self, offset: int, depth: int = 0) -> tuple[Any, int]: ...

class NullObject:
    __slots__ = ()

NULL_OBJECT = NullObject()

class LegacyByteString(str):
    __slots__ = ()

class LegacyLong(int):
    __slots__ = ()

@dataclass(slots=True)
class FrozenDictValue:
    items: tuple[tuple[Any, Any], ...]

@dataclass(frozen=True, slots=True)
class MarshalSetValue:
    items: tuple[Any, ...]
    frozen: bool

def read_dict(
    reader: ObjectReader, offset: int, depth: int
) -> tuple[dict[Any, Any], int]:
    result: dict[Any, Any] = {}
    for _index in range(MAX_MARSHAL_COLLECTION_ITEMS):
        key, offset = reader.read_object(offset, depth + 1)
        if key is NULL_OBJECT:
            return result, offset
        value, offset = reader.read_object(offset, depth + 1)
        if value is NULL_OBJECT:
            raise ValueError("marshal dictionary value cannot be null sentinel")
        result[key] = value
    raise ValueError("marshal dictionary exceeded item limit")

def read_frozen_dict(
    reader: ObjectReader, offset: int, depth: int
) -> tuple[FrozenDictValue, int]:
    items: list[tuple[Any, Any]] = []
    for _index in range(MAX_MARSHAL_COLLECTION_ITEMS):
        key, offset = reader.read_object(offset, depth + 1)
        if key is NULL_OBJECT:
            return FrozenDictValue(tuple(items)), offset
        value, offset = reader.read_object(offset, depth + 1)
        if value is NULL_OBJECT:
            raise ValueError("marshal frozendict value cannot be null sentinel")
        items.append((key, value))
    raise ValueError("marshal frozendict exceeded item limit")

def read_tuple(
    reader: ObjectReader,
    data: bytes,
    offset: int,
    depth: int,
) -> tuple[tuple[Any, ...], int]:
    size, offset = read_size(data, offset)
    return read_tuple_items(reader, offset, size, depth)

def read_small_tuple(
    reader: ObjectReader,
    data: bytes,
    offset: int,
    depth: int,
) -> tuple[tuple[Any, ...], int]:
    size, offset = read_uint8(data, offset)
    return read_tuple_items(reader, offset, size, depth)

def read_list(
    reader: ObjectReader,
    data: bytes,
    offset: int,
    depth: int,
) -> tuple[list[Any], int]:
    size, offset = read_size(data, offset)
    require_collection_size(size)
    values: list[Any] = []

    for _index in range(size):
        value, offset = reader.read_object(offset, depth + 1)
        values.append(value)

    return values, offset

def read_set(
    reader: ObjectReader,
    data: bytes,
    offset: int,
    depth: int,
    frozen: bool,
) -> tuple[MarshalSetValue, int]:
    size, offset = read_size(data, offset)
    require_collection_size(size)
    values: list[Any] = []

    for _index in range(size):
        value, offset = reader.read_object(offset, depth + 1)
        values.append(value)

    return MarshalSetValue(tuple(values), frozen=frozen), offset

def read_slice(
    reader: ObjectReader,
    offset: int,
    depth: int,
) -> tuple[slice, int]:
    start, offset = reader.read_object(offset, depth + 1)
    stop, offset = reader.read_object(offset, depth + 1)
    step, offset = reader.read_object(offset, depth + 1)

    if start is NULL_OBJECT or stop is NULL_OBJECT or step is NULL_OBJECT:
        raise ValueError("marshal slice component cannot be null sentinel")
    return slice(start, stop, step), offset

def read_tuple_items(
    reader: ObjectReader,
    offset: int,
    size: int,
    depth: int,
) -> tuple[tuple[Any, ...], int]:
    require_collection_size(size)
    values: list[Any] = []

    for _index in range(size):
        value, offset = reader.read_object(offset, depth + 1)
        values.append(value)

    return tuple(values), offset

def require_collection_size(size: int) -> None:
    if size < 0:
        raise ValueError("marshal collection size is negative")
    if size > MAX_MARSHAL_COLLECTION_ITEMS:
        raise ValueError("marshal collection exceeded item limit")

def read_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = read_size(data, offset)
    return read_sized_bytes(data, offset, size)

def read_short_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = read_uint8(data, offset)
    return read_sized_bytes(data, offset, size)

def read_text(data: bytes, offset: int) -> tuple[str, int]:
    raw, offset = read_bytes(data, offset)
    return raw.decode("utf-8", errors="surrogatepass"), offset

def read_ascii(data: bytes, offset: int) -> tuple[str, int]:
    raw, offset = read_bytes(data, offset)
    return raw.decode("ascii"), offset

def read_short_ascii(data: bytes, offset: int) -> tuple[str, int]:
    raw, offset = read_short_bytes(data, offset)
    return raw.decode("ascii"), offset

def read_sized_bytes(data: bytes, offset: int, size: int) -> tuple[bytes, int]:
    raw = read_exact(data, offset, size)
    return raw, offset + size

STRING_READERS = {
    ord("u"): read_text,
    ord("A"): read_ascii,
    ord("a"): read_ascii,
    ord("z"): read_short_ascii,
    ord("Z"): read_short_ascii,
}
INTERNED_STRING_CODES = {ord("t"), ord("A"), ord("Z")}

def read_marshal_string(
    data: bytes,
    type_code: int,
    offset: int,
    version: tuple[int, ...],
    interned: list[str | bytes],
) -> tuple[Any, int]:
    if type_code == ord("R"):
        index, offset = read_int32(data, offset)
        if index < 0 or index >= len(interned):
            raise ValueError("marshal interned string reference is invalid")
        return interned[index], offset

    if type_code in {ord("s"), ord("t")}:
        value, offset = read_legacy_sensitive_string(data, type_code, offset, version)
    else:
        reader = STRING_READERS.get(type_code)
        if reader is None:
            raise ValueError(f"unsupported marshal string type: {type_code!r}")
        value, offset = reader(data, offset)

    if type_code in INTERNED_STRING_CODES:
        interned.append(value)
    return value, offset

def read_legacy_sensitive_string(
    data: bytes,
    type_code: int,
    offset: int,
    version: tuple[int, ...],
) -> tuple[Any, int]:
    if type_code == ord("s") or version < (3, 0):
        value, offset = read_bytes(data, offset)
        if version < (3, 0):
            value = LegacyByteString(
                value.decode("latin-1", errors="surrogateescape")
            )
        return value, offset
    return read_text(data, offset)
