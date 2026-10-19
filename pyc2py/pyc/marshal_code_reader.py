from collections.abc import Callable
from typing import Any, Protocol

from pyc2py.bytecode.pypy import table_in_cpython_form
from pyc2py.pyc.code import (
    PycCode,
    as_bytes,
    as_text,
    as_text_tuple,
    as_tuple,
    make_code_object,
    make_legacy_code,
    replace_code,
    split_localsplus,
)
from pyc2py.pyc.primitives import read_int32
from pyc2py.pyc.reference import ReferenceTable


class MarshalCodeReaderMixin(Protocol):
    data: bytes
    version: tuple[int, ...]
    refs: ReferenceTable
    opcodes: str
    code_extras: tuple[str, ...]
    read_object: Callable[[int, int], tuple[Any, int]]

    def read_code(self, offset: int, depth: int, has_ref: bool) -> tuple[PycCode, int]:
        reserved = self.refs.reserve(has_ref)

        # a format with extras keeps CPython 3.10's fields
        if self.version >= (3, 11) and not self.code_extras:
            code, offset = self.read_code_311(offset, depth)
        elif self.version >= (3, 8):
            code, offset = self.read_code_38(offset, depth)
        elif self.version >= (3, 0):
            code, offset = self.read_code_30(offset, depth)
        else:
            code, offset = self.read_code_21(offset, depth)

        if self.opcodes:
            table = code.co_exceptiontable
            if table:
                table = table_in_cpython_form(code.co_code, table, self.opcodes)

            code = replace_code(code, opcodes=self.opcodes, co_exceptiontable=table)

        return self.refs.set_reserved(reserved, code), offset

    def read_code_21(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        nlocals, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)
        fields, offset = self.read_common_code_fields(offset, depth)
        return make_legacy_code(argcount, nlocals, stacksize, flags, fields), offset

    def read_code_30(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        kwonlyargcount, offset = read_int32(self.data, offset)
        nlocals, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)
        fields, offset = self.read_common_code_fields(offset, depth)
        code = make_legacy_code(argcount, nlocals, stacksize, flags, fields)
        return replace_code(code, co_kwonlyargcount=kwonlyargcount), offset

    def read_code_38(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        posonlyargcount, offset = read_int32(self.data, offset)
        kwonlyargcount, offset = read_int32(self.data, offset)
        nlocals, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)
        fields, offset = self.read_common_code_fields(offset, depth)
        code = make_legacy_code(argcount, nlocals, stacksize, flags, fields)

        if self.version >= (3, 10):
            # 3.10 stores its new line table in co_lnotab
            code = replace_code(code, co_lnotab=b"", co_linetable=code.co_lnotab)
        if "qualname" in fields:
            code = replace_code(code, co_qualname=as_text(fields["qualname"]))
        if "exceptiontable" in fields:
            code = replace_code(
                code, co_exceptiontable=as_bytes(fields["exceptiontable"])
            )

        return (
            replace_code(
                code,
                co_posonlyargcount=posonlyargcount,
                co_kwonlyargcount=kwonlyargcount,
            ),
            offset,
        )

    def read_code_311(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        posonlyargcount, offset = read_int32(self.data, offset)
        kwonlyargcount, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)
        code, offset = self.read_object(offset, depth + 1)
        consts, offset = self.read_object(offset, depth + 1)
        names, offset = self.read_object(offset, depth + 1)
        localsplusnames, offset = self.read_object(offset, depth + 1)
        localspluskinds, offset = self.read_object(offset, depth + 1)
        filename, offset = self.read_object(offset, depth + 1)
        name, offset = self.read_object(offset, depth + 1)
        qualname, offset = self.read_object(offset, depth + 1)
        firstlineno, offset = read_int32(self.data, offset)
        linetable, offset = self.read_object(offset, depth + 1)
        exceptiontable, offset = self.read_object(offset, depth + 1)
        varnames, cellvars, freevars = split_localsplus(
            localsplusnames, localspluskinds
        )
        return (
            make_code_object(
                argcount=argcount,
                posonlyargcount=posonlyargcount,
                kwonlyargcount=kwonlyargcount,
                nlocals=len(varnames),
                stacksize=stacksize,
                flags=flags,
                code=as_bytes(code),
                consts=as_tuple(consts),
                names=as_text_tuple(names),
                varnames=as_text_tuple(varnames),
                filename=as_text(filename),
                name=as_text(name),
                qualname=as_text(qualname),
                firstlineno=firstlineno,
                linetable=as_bytes(linetable),
                exceptiontable=as_bytes(exceptiontable),
                freevars=as_text_tuple(freevars),
                cellvars=as_text_tuple(cellvars),
                localsplusnames=as_text_tuple(localsplusnames),
            ),
            offset,
        )

    def read_common_code_fields(
        self,
        offset: int,
        depth: int,
    ) -> tuple[dict[str, Any], int]:
        code, offset = self.read_object(offset, depth + 1)
        consts, offset = self.read_object(offset, depth + 1)
        names, offset = self.read_object(offset, depth + 1)
        varnames, offset = self.read_object(offset, depth + 1)
        freevars, offset = self.read_object(offset, depth + 1)
        cellvars, offset = self.read_object(offset, depth + 1)
        filename, offset = self.read_object(offset, depth + 1)
        name, offset = self.read_object(offset, depth + 1)
        extras: dict[str, Any] = {}

        if "qualname" in self.code_extras:
            extras["qualname"], offset = self.read_object(offset, depth + 1)

        firstlineno, offset = read_int32(self.data, offset)
        lnotab, offset = self.read_object(offset, depth + 1)

        if "exceptiontable" in self.code_extras:
            extras["exceptiontable"], offset = self.read_object(offset, depth + 1)

        return {
            **extras,
            "code": code,
            "consts": consts,
            "names": names,
            "varnames": varnames,
            "freevars": freevars,
            "cellvars": cellvars,
            "filename": filename,
            "name": name,
            "firstlineno": firstlineno,
            "lnotab": lnotab,
        }, offset
