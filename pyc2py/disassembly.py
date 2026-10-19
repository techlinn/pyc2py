from typing import Any

from pyc2py.bytecode.decoder import decode_legacy_instructions
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import parse_exception_table
from pyc2py.pyc.code import code_objects
from pyc2py.types import PycModule, VersionTuple


def disassemble(module: PycModule, origin: str) -> str:
    version = module.header.version
    lines = [f"# pyc2py disassembly of {origin}"]

    for code in code_objects(module.code):
        lines += ["", *disassemble_code(code, version)]

    return "\n".join(lines) + "\n"


def disassemble_code(code: Any, version: VersionTuple | None) -> list[str]:
    first = getattr(code, "co_firstlineno", None)
    lines = [f"{code_name(code)} (line {first}):"]

    if version is None:
        return [*lines, "    no version, so the bytecode cannot be read"]

    try:
        instructions = decode_legacy_instructions(code, version)
    except (IndexError, KeyError, ValueError) as error:
        return [*lines, f"    cannot be read: {error}"]

    lines += [
        instruction_line(instruction)
        for instruction in instructions
        if instruction.opname != "CACHE"
    ]
    table = bytes(getattr(code, "co_exceptiontable", b"") or b"")
    if table and version >= (3, 11):
        lines.append("  exception table:")

        try:
            entries = parse_exception_table(table)
        except (IndexError, ValueError) as error:
            return [*lines, f"    cannot be read: {error}"]

        lines += [
            f"    {entry.start_offset} to {entry.end_offset} -> {entry.target_offset}"
            f" [{entry.depth}]{' lasti' if entry.lasti else ''}"
            for entry in entries
        ]

    return lines


def instruction_line(instruction: Instruction) -> str:
    line = "" if instruction.starts_line is None else str(instruction.starts_line)
    target = ">>" if instruction.is_jump_target else "  "
    text = f"{line:>5} {target} {instruction.offset:>6} {instruction.opname}"

    if instruction.arg is None:
        return text

    shown = argument_text(instruction)
    text = f"{text:<44} {instruction.arg}"
    # a constant equal to its own index still says which constant it is
    repeated = shown == str(instruction.arg) and "CONST" not in instruction.opname
    return f"{text} ({shown})" if shown and not repeated else text


def argument_text(instruction: Instruction) -> str:
    value = instruction.argval
    if hasattr(value, "co_code"):
        line = getattr(value, "co_firstlineno", "?")
        return f"code object {code_name(value)}, line {line}"

    if instruction.argrepr:
        return instruction.argrepr

    return repr(value)


def code_name(code: Any) -> str:
    return str(getattr(code, "co_qualname", None) or getattr(code, "co_name", "?"))
