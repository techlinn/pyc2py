from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class PycCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[Any, ...]
    co_varnames: tuple[Any, ...]
    co_filename: str
    co_name: str
    co_qualname: str
    co_firstlineno: int
    co_lnotab: bytes
    co_linetable: bytes
    co_exceptiontable: bytes
    co_freevars: tuple[Any, ...]
    co_cellvars: tuple[Any, ...]
    co_localsplusnames: tuple[Any, ...]

def normalize_code_object(code: Any) -> PycCode:
    return PycCode(
        co_argcount=read_int_attr(code, "co_argcount"),
        co_posonlyargcount=read_int_attr(code, "co_posonlyargcount"),
        co_kwonlyargcount=read_int_attr(code, "co_kwonlyargcount"),
        co_nlocals=read_int_attr(code, "co_nlocals"),
        co_stacksize=read_int_attr(code, "co_stacksize"),
        co_flags=read_int_attr(code, "co_flags"),
        co_code=read_bytes_attr(code, "co_code"),
        co_consts=normalize_consts(getattr(code, "co_consts", ()) or ()),
        co_names=tuple(getattr(code, "co_names", ()) or ()),
        co_varnames=tuple(getattr(code, "co_varnames", ()) or ()),
        co_filename=str(getattr(code, "co_filename", "") or ""),
        co_name=str(getattr(code, "co_name", "") or ""),
        co_qualname=read_qualname(code),
        co_firstlineno=read_int_attr(code, "co_firstlineno", default=1),
        co_lnotab=read_bytes_attr(code, "co_lnotab"),
        co_linetable=read_bytes_attr(code, "co_linetable"),
        co_exceptiontable=read_bytes_attr(code, "co_exceptiontable"),
        co_freevars=tuple(getattr(code, "co_freevars", ()) or ()),
        co_cellvars=tuple(getattr(code, "co_cellvars", ()) or ()),
        co_localsplusnames=read_localsplusnames(code),
    )

def normalize_consts(values: Any) -> tuple[Any, ...]:
    return tuple(normalize_const(value) for value in values)

def normalize_const(value: Any) -> Any:
    if has_code_shape(value):
        return normalize_code_object(value)
    if isinstance(value, tuple):
        return tuple(normalize_const(item) for item in value)
    if isinstance(value, list):
        return [normalize_const(item) for item in value]
    if isinstance(value, frozenset):
        return frozenset(normalize_const(item) for item in value)
    return value

def has_code_shape(value: object) -> bool:
    return hasattr(value, "co_code") and hasattr(value, "co_consts")

def read_int_attr(code: Any, name: str, default: int = 0) -> int:
    value = getattr(code, name, default)
    if value is None:
        return default

    return int(value)

def read_bytes_attr(code: Any, name: str) -> bytes:
    value = getattr(code, name, b"")
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("latin-1", errors="surrogateescape")

    return bytes(value)

def read_qualname(code: Any) -> str:
    value = getattr(code, "co_qualname", None)
    if value:
        return str(value)
    return str(getattr(code, "co_name", "") or "")

def read_localsplusnames(code: Any) -> tuple[Any, ...]:
    value = getattr(code, "co_localsplusnames", None)
    if value is not None:
        return tuple(value)

    varnames = tuple(getattr(code, "co_varnames", ()) or ())
    cellvars = tuple(getattr(code, "co_cellvars", ()) or ())
    freevars = tuple(getattr(code, "co_freevars", ()) or ())
    return (*varnames, *cellvars, *freevars)

def make_code_object(
    *,
    argcount: int = 0,
    posonlyargcount: int = 0,
    kwonlyargcount: int = 0,
    nlocals: int = 0,
    stacksize: int = 0,
    flags: int = 0,
    code: bytes = b"",
    consts: tuple[Any, ...] = (),
    names: tuple[Any, ...] = (),
    varnames: tuple[Any, ...] = (),
    filename: str = "",
    name: str = "",
    qualname: str | None = None,
    firstlineno: int = 1,
    lnotab: bytes = b"",
    linetable: bytes = b"",
    exceptiontable: bytes = b"",
    freevars: tuple[Any, ...] = (),
    cellvars: tuple[Any, ...] = (),
    localsplusnames: tuple[Any, ...] | None = None,
) -> PycCode:
    resolved_localsplusnames = localsplusnames
    if resolved_localsplusnames is None:
        resolved_localsplusnames = (*varnames, *cellvars, *freevars)

    return PycCode(
        co_argcount=argcount,
        co_posonlyargcount=posonlyargcount,
        co_kwonlyargcount=kwonlyargcount,
        co_nlocals=nlocals,
        co_stacksize=stacksize,
        co_flags=flags,
        co_code=code,
        co_consts=consts,
        co_names=names,
        co_varnames=varnames,
        co_filename=filename,
        co_name=name,
        co_qualname=name if qualname is None else qualname,
        co_firstlineno=firstlineno,
        co_lnotab=lnotab,
        co_linetable=linetable,
        co_exceptiontable=exceptiontable,
        co_freevars=freevars,
        co_cellvars=cellvars,
        co_localsplusnames=resolved_localsplusnames,
    )

def make_legacy_code(
    argcount: int,
    nlocals: int,
    stacksize: int,
    flags: int,
    fields: dict[str, Any],
) -> PycCode:
    return make_code_object(
        argcount=argcount,
        nlocals=nlocals,
        stacksize=stacksize,
        flags=flags,
        code=as_bytes(fields["code"]),
        consts=tuple(fields["consts"]),
        names=as_text_tuple(fields["names"]),
        varnames=as_text_tuple(fields["varnames"]),
        filename=as_text(fields["filename"]),
        name=as_text(fields["name"]),
        firstlineno=int(fields["firstlineno"]),
        lnotab=as_bytes(fields["lnotab"]),
        freevars=as_text_tuple(fields["freevars"]),
        cellvars=as_text_tuple(fields["cellvars"]),
    )

def replace_code(code: PycCode, **changes: Any) -> PycCode:
    values = {field: getattr(code, field) for field in code.__dataclass_fields__}
    values.update(changes)
    return PycCode(**values)

def as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("latin-1", errors="surrogateescape")

    return bytes(value)

def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="surrogateescape")
    return str(value)

def as_text_tuple(values: Any) -> tuple[str, ...]:
    return tuple(as_text(value) for value in values)

def split_localsplus(
    names: Any, kinds: Any
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    local_names: list[Any] = []
    cell_names: list[Any] = []
    free_names: list[Any] = []

    for name, kind in zip(tuple(names), bytes(kinds), strict=True):
        if kind & 0x20:
            local_names.append(name)
        if kind & 0x40:
            cell_names.append(name)
        if kind & 0x80:
            free_names.append(name)

    return tuple(local_names), tuple(cell_names), tuple(free_names)
