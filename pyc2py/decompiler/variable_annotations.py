import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyc2py.astree import is_docstring_statement
from pyc2py.bytecode.instruction import Instruction
from pyc2py.decompiler.recover import append_annotation_statement

if TYPE_CHECKING:
    from pyc2py.decompiler.context import DecompilerContext

ANNOTATE_NAMES = frozenset({"__annotate__", "__annotate_func__"})
CONDITIONAL_SET = "__conditional_annotations__"


@dataclass(frozen=True, slots=True)
class VariableAnnotation:
    name: str
    annotation: ast.expr
    number: int | None


def restore_variable_annotations(
    statements: list[ast.stmt],
    instructions: list[Instruction],
    version: tuple[int, ...] | None,
    new_child: Callable[[object], "DecompilerContext"],
) -> list[ast.stmt]:
    statements = [item for item in statements if not assigns_conditional_set(item)]
    code = find_annotate_code(instructions)
    if code is None:
        return statements
    annotations = read_variable_annotations(code, version, new_child)
    if annotations is None:
        return statements
    body = [item for item in statements if not is_annotate_function(item)]
    numbered = {item.number: item for item in annotations if item.number is not None}
    place_unreached(body, numbered)
    replace_annotation_marks(body, numbered)
    place_unnumbered(body, [item for item in annotations if item.number is None])
    return body


def find_annotate_code(instructions: list[Instruction]) -> Any:
    code: Any = None
    for instruction in instructions:
        if instruction.opname == "LOAD_CONST" and hasattr(
            instruction.argval, "co_code"
        ):
            code = instruction.argval
        elif instruction.opname == "STORE_NAME":
            if instruction.argval in ANNOTATE_NAMES and code is not None:
                return code if code.co_name == "__annotate__" else None
            code = None
    return None


def read_variable_annotations(
    code: Any,
    version: tuple[int, ...] | None,
    new_child: Callable[[object], "DecompilerContext"],
) -> list[VariableAnnotation] | None:
    from pyc2py.bytecode.decoder import decode_instructions
    from pyc2py.decompiler.runtime import parse_body_or_empty

    stores = read_annotation_stores(decode_instructions(code, version))
    result = new_child(code).decompile()
    returned = [
        statement.value
        for statement in parse_body_or_empty(result).body
        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict)
    ]
    if len(returned) != 1 or len(returned[0].keys) != len(stores):
        return None
    annotations: list[VariableAnnotation] = []
    for key, value, (name, number) in zip(
        returned[0].keys, returned[0].values, stores, strict=True
    ):
        if not isinstance(key, ast.Constant) or key.value != name:
            return None
        annotations.append(VariableAnnotation(name, value, number))
    return annotations


def read_annotation_stores(
    instructions: list[Instruction],
) -> list[tuple[str, int | None]]:
    stores: list[tuple[str, int | None]] = []
    number: int | None = None
    guard_end = -1
    for index, instruction in enumerate(instructions):
        if instruction.offset >= guard_end:
            number = None
        guard = conditional_guard(instructions, index)
        if guard is not None:
            number, guard_end = guard
        if (
            instruction.opname == "STORE_SUBSCR"
            and index >= 2
            and instructions[index - 2].opname == "COPY"
            and instructions[index - 1].opname == "LOAD_CONST"
            and isinstance(instructions[index - 1].argval, str)
        ):
            stores.append((instructions[index - 1].argval, number))
    return stores


def conditional_guard(
    instructions: list[Instruction],
    index: int,
) -> tuple[int, int] | None:
    if index + 3 >= len(instructions):
        return None
    number, names, contains, jump = instructions[index : index + 4]
    if number.opname != "LOAD_CONST" or not isinstance(number.argval, int):
        return None
    if names.argval != CONDITIONAL_SET or contains.opname != "CONTAINS_OP":
        return None
    if jump.opname != "POP_JUMP_IF_FALSE":
        return None
    return number.argval, int(jump.argval)


def is_annotate_function(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.FunctionDef) and statement.name in ANNOTATE_NAMES


def assigns_conditional_set(statement: ast.stmt) -> bool:
    return assigns_name(statement, CONDITIONAL_SET)


def annotation_mark_number(statement: ast.stmt) -> int | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "add":
        return None
    if not is_conditional_set(call.func.value) or len(call.args) != 1:
        return None
    number = call.args[0]
    if isinstance(number, ast.Constant) and isinstance(number.value, int):
        return number.value
    return None


def is_conditional_set(node: Any) -> bool:
    return isinstance(node, ast.Name) and node.id == CONDITIONAL_SET


def replace_annotation_marks(
    body: list[ast.stmt],
    numbered: dict[int, VariableAnnotation],
) -> None:
    kept: list[ast.stmt] = []
    for statement in body:
        if isinstance(statement, ast.Expr) and is_conditional_set(statement.value):
            continue
        number = annotation_mark_number(statement)
        found = None if number is None else numbered.get(number)
        if found is None:
            replace_marks_in_blocks(statement, numbered)
            kept.append(statement)
        elif not append_annotation_statement(kept, found.name, found.annotation):
            kept.append(statement)
    body[:] = kept


def place_unreached(
    body: list[ast.stmt], numbered: dict[int, VariableAnnotation]
) -> None:
    marks = [
        {
            number
            for node in ast.walk(statement)
            if isinstance(node, ast.stmt)
            and (number := annotation_mark_number(node)) is not None
        }
        for statement in body
    ]
    seen = set().union(*marks)
    made: list[ast.If] = []
    for number in sorted(set(numbered) - seen):
        found = numbered[number]
        position = next(
            (
                index
                for index, inside in enumerate(marks)
                if any(mark > number for mark in inside)
            ),
            len(marks),
        )
        if 0 < position < len(body) and (
            annotation_mark_number(body[position]) is not None
        ):
            position -= isinstance(body[position - 1], ast.Assign)
        unreached: list[ast.stmt] = []
        if not append_annotation_statement(unreached, found.name, found.annotation):
            continue
        if position > 0 and any(body[position - 1] is block for block in made):
            made[-1].body.extend(unreached)
            continue
        made.append(ast.If(test=ast.Constant(value=False), body=unreached, orelse=[]))
        body.insert(position, made[-1])
        marks.insert(position, set())


def replace_marks_in_blocks(
    statement: ast.stmt,
    numbered: dict[int, VariableAnnotation],
) -> None:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for field in ("body", "orelse", "finalbody"):
        block = getattr(statement, field, None)
        if block:
            replace_annotation_marks(block, numbered)
            if not block:
                block.append(ast.Pass())
    for handler in getattr(statement, "handlers", []):
        replace_annotation_marks(handler.body, numbered)
    for case in getattr(statement, "cases", []):
        replace_annotation_marks(case.body, numbered)


def place_unnumbered(
    body: list[ast.stmt], annotations: list[VariableAnnotation]
) -> None:
    position = 0
    while position < len(body) and is_body_prologue(body[position]):
        position += 1
    for annotation in annotations:
        assigned = next(
            (
                index
                for index in range(position, len(body))
                if assigns_name(body[index], annotation.name)
            ),
            None,
        )
        statement = ast.AnnAssign(
            target=ast.Name(id=annotation.name, ctx=ast.Store()),
            annotation=annotation.annotation,
            value=None,
            simple=1,
        )
        if assigned is None:
            body.insert(position, statement)
            position += 1
            continue
        statement.value = getattr(body[assigned], "value", None)
        body[assigned] = statement
        position = assigned + 1


def is_body_prologue(statement: ast.stmt) -> bool:
    if is_docstring_statement(statement):
        return True
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return isinstance(target, ast.Name) and target.id.startswith("__")


def assigns_name(statement: ast.stmt, name: str) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return isinstance(target, ast.Name) and target.id == name
