from typing import Any

from pyc2py.pyc.code import (
    PycCode,
    as_bytes,
    as_text,
    as_text_tuple,
    make_code_object,
    make_legacy_code,
    replace_code,
    split_localsplus,
)
from pyc2py.pyc.primitives import read_int16, read_int32

class MarshalCodeReaderMixin:
    def read_code(self, offset: int, depth: int, has_ref: bool) -> tuple[PycCode, int]:
        reserved = self.refs.reserve(has_ref)

        if self.version < (1, 5):
            code, offset = self.read_code_ancient(offset, depth)
        elif self.version >= (3, 11):
            code, offset = self.read_code_311(offset, depth)
        elif self.version >= (3, 8):
            code, offset = self.read_code_38(offset, depth)
        elif self.version >= (3, 0):
            code, offset = self.read_code_30(offset, depth)
        elif self.version >= (2, 1):
            code, offset = self.read_code_21(offset, depth)
        else:
            code, offset = self.read_code_legacy(offset, depth)

        return self.refs.set_reserved(reserved, code), offset

    def read_code_ancient_compact(
        self,
        offset: int,
        depth: int,
        has_ref: bool,
    ) -> tuple[PycCode, int]:
        reserved = self.refs.reserve(has_ref)

        code, offset = self.read_object(offset, depth + 1)
        consts, offset = self.read_object(offset, depth + 1)
        names, offset = self.read_object(offset, depth + 1)
        filename, offset = self.read_object(offset, depth + 1)
        name, offset = self.read_object(offset, depth + 1)

        value = make_code_object(
            code=as_bytes(code),
            consts=tuple(consts),
            names=as_text_tuple(names),
            filename=as_text(filename),
            name=as_text(name),
        )
        return self.refs.set_reserved(reserved, value), offset

    def read_code_ancient(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int16(self.data, offset)
        nlocals, offset = read_int16(self.data, offset)
        stacksize, offset = read_int16(self.data, offset)

        code, offset = self.read_object(offset, depth + 1)
        consts, offset = self.read_object(offset, depth + 1)
        names, offset = self.read_object(offset, depth + 1)
        varnames, offset = self.read_object(offset, depth + 1)
        filename, offset = self.read_object(offset, depth + 1)
        name, offset = self.read_object(offset, depth + 1)

        return (
            make_code_object(
                argcount=argcount,
                nlocals=nlocals,
                stacksize=stacksize,
                code=as_bytes(code),
                consts=tuple(consts),
                names=as_text_tuple(names),
                varnames=as_text_tuple(varnames),
                filename=as_text(filename),
                name=as_text(name),
            ),
            offset,
        )

    def read_code_legacy(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = self.read_code_int(offset)
        nlocals, offset = self.read_code_int(offset)
        stacksize, offset = self.read_code_int(offset)
        flags, offset = self.read_code_int(offset)
        fields, offset = self.read_common_code_fields(offset, depth, has_closure=False)
        return make_legacy_code(argcount, nlocals, stacksize, flags, fields), offset

    def read_code_21(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = self.read_code_int(offset)
        nlocals, offset = self.read_code_int(offset)
        stacksize, offset = self.read_code_int(offset)
        flags, offset = self.read_code_int(offset)
        fields, offset = self.read_common_code_fields(offset, depth, has_closure=True)
        return make_legacy_code(argcount, nlocals, stacksize, flags, fields), offset

    def read_code_30(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        kwonlyargcount, offset = read_int32(self.data, offset)
        nlocals, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)

        fields, offset = self.read_common_code_fields(offset, depth, has_closure=True)
        code = make_legacy_code(argcount, nlocals, stacksize, flags, fields)

        return replace_code(code, co_kwonlyargcount=kwonlyargcount), offset

    def read_code_38(self, offset: int, depth: int) -> tuple[PycCode, int]:
        argcount, offset = read_int32(self.data, offset)
        posonlyargcount, offset = read_int32(self.data, offset)
        kwonlyargcount, offset = read_int32(self.data, offset)
        nlocals, offset = read_int32(self.data, offset)
        stacksize, offset = read_int32(self.data, offset)
        flags, offset = read_int32(self.data, offset)

        fields, offset = self.read_common_code_fields(offset, depth, has_closure=True)
        code = make_legacy_code(argcount, nlocals, stacksize, flags, fields)

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
                code=bytes(code),
                consts=tuple(consts),
                names=as_text_tuple(names),
                varnames=as_text_tuple(varnames),
                filename=as_text(filename),
                name=as_text(name),
                qualname=as_text(qualname),
                firstlineno=firstlineno,
                linetable=bytes(linetable),
                exceptiontable=bytes(exceptiontable),
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
        has_closure: bool,
    ) -> tuple[dict[str, Any], int]:
        code, offset = self.read_object(offset, depth + 1)
        consts, offset = self.read_object(offset, depth + 1)
        names, offset = self.read_object(offset, depth + 1)
        varnames, offset = self.read_object(offset, depth + 1)

        freevars: tuple[Any, ...] = ()
        cellvars: tuple[Any, ...] = ()
        if has_closure:
            freevars, offset = self.read_object(offset, depth + 1)
            cellvars, offset = self.read_object(offset, depth + 1)

        filename, offset = self.read_object(offset, depth + 1)
        name, offset = self.read_object(offset, depth + 1)
        firstlineno, offset = self.read_code_int(offset)
        lnotab, offset = self.read_object(offset, depth + 1)

        return {
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

    def read_code_int(self, offset: int) -> tuple[int, int]:
        if self.version < (2, 3):
            return read_int16(self.data, offset)
        return read_int32(self.data, offset)
