from typing import Any

from pyc2py.pyc.code import PycCode
from pyc2py.pyc.marshal_code_reader import MarshalCodeReaderMixin
from pyc2py.pyc.objects import (
    NULL_OBJECT,
    FrozenDictValue,
    LegacyLong,
    read_dict,
    read_frozen_dict,
    read_list,
    read_marshal_string,
    read_set,
    read_slice,
    read_small_tuple,
    read_tuple,
)
from pyc2py.pyc.primitives import (
    read_int32,
    read_int64,
    read_long,
    read_marshal_number,
    read_uint8,
)
from pyc2py.pyc.reference import TYPE_REF, ReferenceTable, split_type_code

MAX_MARSHAL_DEPTH = 2_000
TYPE_CODE_LEGACY = ord("C")
MARSHAL_SCALARS = {
    ord("0"): NULL_OBJECT,
    ord("N"): None,
    ord("F"): False,
    ord("T"): True,
    ord("."): Ellipsis,
    ord("S"): StopIteration,
}
NUMBER_TYPE_CODES = {ord("f"), ord("g"), ord("x"), ord("y")}
STRING_TYPE_CODES = {
    ord("s"),
    ord("t"),
    ord("R"),
    ord("u"),
    ord("a"),
    ord("A"),
    ord("z"),
    ord("Z"),
}
COLLECTION_TYPE_CODES = {ord("("), ord(")"), ord("["), ord("<"), ord(">")}
RECURSIVE_PAYLOAD_METHODS = {
    ord(":"): "read_slice",
    ord("{"): "read_mapping",
    ord("}"): "read_frozen_mapping",
    TYPE_CODE_LEGACY: "read_code_ancient_compact",
    ord("c"): "read_code",
}


class MarshalReader(MarshalCodeReaderMixin):
    def __init__(self, data: bytes, version: tuple[int, ...]) -> None:
        self.data = data
        self.version = version
        self.refs = ReferenceTable()
        self.interned: list[str | bytes] = []

    def read_object(self, offset: int, depth: int = 0) -> tuple[Any, int]:
        if depth > MAX_MARSHAL_DEPTH:
            raise ValueError("marshal object graph is too deep")

        raw_type, offset = read_uint8(self.data, offset)
        type_code, has_ref = split_type_code(raw_type)
        if type_code == TYPE_REF:
            index, offset = read_int32(self.data, offset)
            return self.refs.get(index), offset

        value, offset = self.read_object_payload(type_code, offset, depth, has_ref)
        return value, offset

    def read_object_payload(
        self,
        type_code: int,
        offset: int,
        depth: int,
        has_ref: bool,
    ) -> tuple[Any, int]:
        scalar = self.read_scalar_payload(type_code, offset, has_ref)
        if scalar is not None:
            return scalar
        if type_code == ord("?"):
            return None, offset

        payload = self.read_simple_payload(type_code, offset, has_ref)
        if payload is not None:
            return payload

        payload = self.read_recursive_payload(type_code, offset, depth, has_ref)
        if payload is not None:
            return payload

        raise ValueError(f"unsupported marshal type code: {type_code!r}")

    def read_simple_payload(
        self,
        type_code: int,
        offset: int,
        has_ref: bool,
    ) -> tuple[Any, int] | None:
        if type_code in NUMBER_TYPE_CODES:
            return self.read_number(type_code, offset, has_ref)
        if type_code in STRING_TYPE_CODES:
            return self.read_string(type_code, offset, has_ref)
        return None

    def read_recursive_payload(
        self,
        type_code: int,
        offset: int,
        depth: int,
        has_ref: bool,
    ) -> tuple[Any, int] | None:
        if type_code in COLLECTION_TYPE_CODES:
            return self.read_collection(type_code, offset, depth, has_ref)

        method_name = RECURSIVE_PAYLOAD_METHODS.get(type_code)
        if method_name is None:
            return None
        return getattr(self, method_name)(offset, depth, has_ref)

    def read_scalar_payload(
        self,
        type_code: int,
        offset: int,
        has_ref: bool,
    ) -> tuple[Any, int] | None:
        if type_code in MARSHAL_SCALARS:
            return MARSHAL_SCALARS[type_code], offset

        if type_code == ord("i"):
            value, offset = read_int32(self.data, offset)
            return self.refs.remember(value, has_ref), offset
        if type_code == ord("I"):
            value, offset = read_int64(self.data, offset)
            return self.refs.remember(value, has_ref), offset
        if type_code == ord("l"):
            value, offset = read_long(self.data, offset)
            if self.version < (3, 0):
                value = LegacyLong(value)
            return self.refs.remember(value, has_ref), offset
        return None

    def read_number(
        self, type_code: int, offset: int, has_ref: bool
    ) -> tuple[Any, int]:
        value, offset = read_marshal_number(self.data, type_code, offset)
        return self.refs.remember(value, has_ref), offset

    def read_string(
        self, type_code: int, offset: int, has_ref: bool
    ) -> tuple[Any, int]:
        value, offset = read_marshal_string(
            self.data,
            type_code,
            offset,
            self.version,
            self.interned,
        )
        return self.refs.remember(value, has_ref), offset

    def read_collection(
        self,
        type_code: int,
        offset: int,
        depth: int,
        has_ref: bool,
    ) -> tuple[Any, int]:
        reserved = self.refs.reserve(has_ref)
        value: Any

        if type_code == ord("("):
            value, offset = read_tuple(self, self.data, offset, depth)
        elif type_code == ord(")"):
            value, offset = read_small_tuple(self, self.data, offset, depth)
        elif type_code == ord("["):
            value, offset = read_list(self, self.data, offset, depth)
        elif type_code == ord("<"):
            value, offset = read_set(self, self.data, offset, depth, frozen=False)
        else:
            value, offset = read_set(self, self.data, offset, depth, frozen=True)

        return self.refs.set_reserved(reserved, value), offset

    def read_slice(
        self,
        offset: int,
        depth: int,
        has_ref: bool,
    ) -> tuple[slice, int]:
        reserved = self.refs.reserve(has_ref)
        value, offset = read_slice(self, offset, depth)
        return self.refs.set_reserved(reserved, value), offset

    def read_mapping(
        self, offset: int, depth: int, has_ref: bool
    ) -> tuple[dict[Any, Any], int]:
        reserved = self.refs.reserve(has_ref)
        value, offset = read_dict(self, offset, depth)
        return self.refs.set_reserved(reserved, value), offset

    def read_frozen_mapping(
        self, offset: int, depth: int, has_ref: bool
    ) -> tuple[FrozenDictValue, int]:
        reserved = self.refs.reserve(has_ref)
        value, offset = read_frozen_dict(self, offset, depth)
        return self.refs.set_reserved(reserved, value), offset


def load_marshal_code(data: bytes, version: tuple[int, ...]) -> PycCode:
    reader = MarshalReader(data, version)
    value, offset = reader.read_object(0)

    if not isinstance(value, PycCode):
        raise ValueError("marshal payload did not contain a code object")
    if offset != len(data):
        raise ValueError("marshal payload has trailing data")

    return value
