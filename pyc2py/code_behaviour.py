import dis
import inspect
from types import CodeType
from typing import Any, cast

from pyc2py.bytecode.decoder import code_opcode_table
from pyc2py.bytecode.metadata import parse_exception_table
from pyc2py.bytecode.operand import resolve_arg_value
from pyc2py.bytecode.scanner import scan_code
from pyc2py.types import is_tuple

SYNTHETIC = frozenset({"__annotate__"})


def iter_units(code: Any | None, structural: bool) -> list[tuple[str, Any]]:
    units: list[tuple[str, Any]] = []

    if code is None:
        return units

    def walk(unit: Any, name: str) -> None:
        units.append((name, unit))
        seen: dict[str, int] = {}

        for child in getattr(unit, "co_consts", ()):
            if not hasattr(child, "co_code"):
                continue

            child_name = str(getattr(child, "co_name", "?"))
            # nameless code (comprehensions, lambdas) is checked with its parent
            if structural and (child_name.startswith("<") or child_name in SYNTHETIC):
                continue

            seen[child_name] = seen.get(child_name, 0) + 1
            suffix = f"#{seen[child_name]}" if seen[child_name] > 1 else ""
            walk(
                child,
                (
                    f"{child_name}{suffix}"
                    if name == "<module>"
                    else f"{name}.{child_name}{suffix}"
                ),
            )

    walk(code, "<module>")
    return units


def code_shape(code: Any) -> tuple[object, ...]:
    argcount = getattr(code, "co_argcount", 0)
    # 3.0-3.3 pass a class body its namespace as __locals__
    if argcount and tuple(getattr(code, "co_varnames", ()))[:1] == ("__locals__",):
        argcount -= 1

    return (
        argcount,
        getattr(code, "co_posonlyargcount", 0),
        getattr(code, "co_kwonlyargcount", 0),
        int(getattr(code, "co_flags", 0)) & SIGNATURE_FLAGS,
    )


# one instruction as the key sees it: offset, name, jump target, value, arguments
Step = tuple[int, str, int | None, object, list[tuple[str, object]]]


def code_behaviour(code: CodeType) -> tuple[object, ...]:
    # what "compiles to the same bytecode" compares, for this Python's own code
    steps: list[Step] = [
        (
            item.offset,
            item.opname,
            int(item.argval) if item.opcode in JUMP_OPCODES else None,
            item.argval,
            native_arguments(item, code),
        )
        for item in dis.get_instructions(code)
    ]
    return behaviour_key(code, steps, is_class_body(code), docstring_constant(code))


def portable_behaviour(code: Any, version: tuple[int, ...]) -> tuple[object, ...]:
    # the same key for code of any version, read by pyc2py's own decoder
    table = code_opcode_table(code, version)
    steps: list[Step] = []

    for item in scan_code(bytes(code.co_code), table, table.layout):
        argval = resolve_arg_value(
            code, table, item.opcode, item.opname, item.arg, item.offset, table.layout
        )
        jumps = item.opcode in table.hasjrel or item.opcode in table.hasjabs
        steps.append(
            (
                item.offset,
                item.opname,
                int(argval) if jumps else None,
                argval,
                portable_arguments(item.opname, item.arg, argval, item.opcode, table),
            )
        )

    class_body = version >= (3, 13) and is_class_body(code)
    return behaviour_key(code, steps, class_body, portable_docstring(code, version))


def behaviour_key(
    code: Any, steps: list[Step], class_body: bool, docstring: object
) -> tuple[object, ...]:
    kept: list[tuple[str, object]] = []
    jumps: dict[int, int] = {}
    positions: dict[int, int] = {}

    for offset, opname, target, argval, arguments in steps:
        positions[offset] = len(kept)

        if opname in LAYOUT_ONLY_INSTRUCTIONS:
            continue

        # 3.13+ stores a class's first line, which the recovered source cannot know
        if class_body and opname == "STORE_NAME" and argval == "__firstlineno__":
            if kept:
                kept[-1] = (kept[-1][0], None)

            class_body = False

        if target is not None:
            jumps[len(kept)] = target

        kept.extend(arguments)

    for index, target_offset in jumps.items():
        kept[index] = (kept[index][0], positions.get(target_offset, len(kept)))

    return (
        tuple(kept),
        protected_instructions(code, positions, len(kept), frozenset(jumps)),
        docstring,
        tuple(code.co_varnames),
        code_signature(code),
    )


def portable_arguments(
    opname: str, arg: int | None, argval: object, opcode: int, table: Any
) -> list[tuple[str, object]]:
    if opname == "KW_NAMES" or opcode in table.hasconst:
        return [(opname, constant_key(argval))]
    if opcode in table.hasname:
        return [(opname, str(argval))]

    return unfused(opname, arg)


def portable_docstring(code: Any, version: tuple[int, ...]) -> object:
    if not code.co_consts:
        return None

    if version >= (3, 14):
        marked = code.co_flags & inspect.CO_HAS_DOCSTRING
    else:
        marked = code.co_flags & inspect.CO_OPTIMIZED

    return constant_key(code.co_consts[0]) if marked else None


SIGNATURE_FLAGS = (
    inspect.CO_VARARGS
    | inspect.CO_VARKEYWORDS
    | inspect.CO_GENERATOR
    | inspect.CO_COROUTINE
    | inspect.CO_ITERABLE_COROUTINE
    | inspect.CO_ASYNC_GENERATOR
)


def code_signature(code: Any) -> tuple[object, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags & SIGNATURE_FLAGS,
        tuple(code.co_freevars),
        tuple(code.co_cellvars),
        code.co_name == "<lambda>",
    )


LINE_FUSED_INSTRUCTIONS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
}
LAYOUT_ONLY_INSTRUCTIONS = frozenset({"EXTENDED_ARG", "NOP"})
JUMP_OPCODES = frozenset(dis.hasjrel) | frozenset(dis.hasjabs)
NAME_OPCODES = frozenset(dis.hasname)
CONSTANT_OPCODES = frozenset(dis.hasconst)


def protected_instructions(
    code: Any,
    positions: dict[int, int],
    count: int,
    jumps: frozenset[int],
) -> tuple[object, ...]:
    covered: list[object] = [None] * count
    table = getattr(code, "co_exceptiontable", b"")
    for entry in parse_exception_table(table):
        handler = (positions.get(entry.target_offset, count), entry.depth, entry.lasti)
        start = positions.get(entry.start_offset, count)
        for position in range(start, positions.get(entry.end_offset, count)):
            covered[position] = handler

    for position in jumps:
        covered[position] = None

    return tuple(covered)


def native_arguments(
    instruction: dis.Instruction,
    code: CodeType,
) -> list[tuple[str, object]]:
    if instruction.opname == "KW_NAMES" and instruction.arg is not None:
        return [(instruction.opname, constant_key(code.co_consts[instruction.arg]))]
    # dis's text keeps the flags a name operand carries, such as NULL +
    if instruction.opcode in NAME_OPCODES:
        return [(instruction.opname, instruction.argrepr)]
    if instruction.opcode in CONSTANT_OPCODES:
        return [(instruction.opname, constant_key(instruction.argval))]

    return unfused(instruction.opname, instruction.arg)


def constant_key(value: object) -> object:
    if hasattr(value, "co_code"):
        return CodeType
    if isinstance(value, float | complex):
        return (type(value), repr(value))
    if is_tuple(value):
        return tuple(constant_key(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(constant_key(item) for item in cast("frozenset[Any]", value))

    return (type(value), value)


def docstring_constant(code: CodeType) -> object:
    if not code.co_consts or not code.co_flags & inspect.CO_HAS_DOCSTRING:
        return None

    return constant_key(code.co_consts[0])


def is_class_body(code: Any) -> bool:
    return code.co_names[:3] == ("__name__", "__module__", "__qualname__")


def unfused(opname: str, arg: int | None) -> list[tuple[str, object]]:
    halves = LINE_FUSED_INSTRUCTIONS.get(opname)
    if halves is None or arg is None:
        return [(plain_load(opname), arg)]

    # both local indexes share the argument, four bits each
    return [(plain_load(halves[0]), arg >> 4), (plain_load(halves[1]), arg & 15)]


def plain_load(opname: str) -> str:
    return "LOAD_FAST" if opname == "LOAD_FAST_BORROW" else opname
