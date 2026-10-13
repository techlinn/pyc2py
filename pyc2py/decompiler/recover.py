import ast
import keyword
from dataclasses import dataclass
from typing import Any

from pyc2py.astree import (
    is_docstring_statement,
    is_none_constant,
    make_arg,
    safe_identifier,
)

PRIVATE_NAME_FIELDS: dict[type[ast.AST], str] = {
    ast.Name: "id",
    ast.Attribute: "attr",
    ast.arg: "arg",
    ast.keyword: "arg",
    ast.FunctionDef: "name",
    ast.AsyncFunctionDef: "name",
}


def demangle_class_private_names(body: list[ast.stmt], class_name: str) -> None:
    prefix = f"_{class_name.lstrip('_')}__"
    work: list[ast.AST] = list(body)
    while work:
        node = work.pop()
        field = PRIVATE_NAME_FIELDS.get(type(node))
        name = getattr(node, field, None) if field is not None else None
        if field is not None and isinstance(name, str):
            setattr(node, field, demangle_class_private_name(name, prefix))
        if isinstance(node, ast.ClassDef):
            work.extend([*node.bases, *node.keywords, *node.decorator_list])
            continue
        work.extend(ast.iter_child_nodes(node))


def demangle_class_private_name(name: str, prefix: str) -> str:
    if not name.startswith(prefix):
        return name
    suffix = name[len(prefix) :]
    if not suffix or suffix.endswith("__"):
        return name
    return f"__{suffix}"


def append_annotation_statement(
    statements: list[ast.stmt],
    name: str,
    annotation: ast.expr,
) -> bool:
    if not is_simple_annotation_name(name):
        return False

    if statements and merge_previous_assignment(statements, name, annotation):
        return True

    statements.append(
        ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=annotation,
            value=None,
            simple=1,
        )
    )
    return True


def merge_previous_assignment(
    statements: list[ast.stmt],
    name: str,
    annotation: ast.expr,
) -> bool:
    previous = statements[-1]
    if not isinstance(previous, ast.Assign):
        return False
    if len(previous.targets) != 1:
        return False

    target = previous.targets[0]
    if not isinstance(target, ast.Name) or target.id != name:
        return False

    statements[-1] = ast.AnnAssign(
        target=ast.Name(id=name, ctx=ast.Store()),
        annotation=annotation,
        value=previous.value,
        simple=1,
    )
    return True


def is_simple_annotation_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


CLASS_BOOKKEEPING_NAMES = {
    "__classdict__",
    "__classdictcell__",
    "__classcell__",
    "__firstlineno__",
    "__module__",
    "__qualname__",
    "__static_attributes__",
    "__type_params__",
}


def clean_class_body(body: list[ast.stmt], class_name: str) -> list[ast.stmt]:
    cleaned = without_class_bookkeeping(body)
    demangle_class_private_names(cleaned, class_name)
    return cleaned or [ast.Pass()]


def without_class_bookkeeping(body: list[ast.stmt]) -> list[ast.stmt]:
    cleaned: list[ast.stmt] = []
    for statement in body:
        if is_class_bookkeeping(statement):
            continue
        if isinstance(statement, ast.Return):
            continue
        clean_class_blocks(statement)
        cleaned.append(statement)
    return cleaned


def clean_class_blocks(statement: ast.stmt) -> None:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for field in ("body", "orelse", "finalbody"):
        block = getattr(statement, field, None)
        if block:
            cleaned = without_class_bookkeeping(block)
            if not cleaned and field == "body":
                cleaned = [ast.Pass()]
            setattr(statement, field, cleaned)
    for handler in getattr(statement, "handlers", []):
        handler.body = without_class_bookkeeping(handler.body) or [ast.Pass()]


def is_class_bookkeeping(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return False
    return target.id in CLASS_BOOKKEEPING_NAMES


def make_class_node(
    name: str,
    bases: tuple[ast.expr, ...],
    body: list[ast.stmt],
) -> ast.ClassDef:
    return ast.ClassDef(
        name=safe_identifier(name),
        bases=list(bases),
        keywords=[],
        body=clean_class_body(body, name),
        decorator_list=[],
    )


def make_generator(
    target: ast.expr,
    iterator: ast.expr,
    conditions: list[ast.expr] | None = None,
    is_async: bool = False,
) -> ast.comprehension:
    return ast.comprehension(
        target=target,
        iter=iterator,
        ifs=[] if conditions is None else conditions,
        is_async=int(is_async),
    )


def make_list_comp(elt: ast.expr, generators: list[ast.comprehension]) -> ast.ListComp:
    return ast.ListComp(elt=elt, generators=generators)


def make_set_comp(elt: ast.expr, generators: list[ast.comprehension]) -> ast.SetComp:
    return ast.SetComp(elt=elt, generators=generators)


def make_generator_exp(
    elt: ast.expr, generators: list[ast.comprehension]
) -> ast.GeneratorExp:
    return ast.GeneratorExp(elt=elt, generators=generators)


def make_dict_comp(
    key: ast.expr,
    value: ast.expr,
    generators: list[ast.comprehension],
) -> ast.DictComp:
    return ast.DictComp(key=key, value=value, generators=generators)


def with_code_docstring(code: Any, body: list[ast.stmt]) -> list[ast.stmt]:
    constants = tuple(getattr(code, "co_consts", ()) or ())
    if not constants or not isinstance(constants[0], str):
        return body
    if body and is_docstring_statement(body[0]):
        return body
    return [ast.Expr(value=ast.Constant(value=constants[0])), *body]


def make_exec_call(
    source: ast.expr, globals_value: ast.expr, locals_value: ast.expr
) -> ast.Expr:
    args = [source]
    if not is_none_constant(globals_value):
        args.append(globals_value)
    if should_include_locals(globals_value, locals_value):
        args.append(locals_value)
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="exec", ctx=ast.Load()),
            args=args,
            keywords=[],
        )
    )


def should_include_locals(globals_value: ast.expr, locals_value: ast.expr) -> bool:
    if is_none_constant(locals_value):
        return False
    return ast.dump(globals_value) != ast.dump(locals_value)


def make_joined_string_parts(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.JoinedStr):
        return list(value.values)
    if isinstance(value, ast.FormattedValue):
        return [value]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return [value]
    return [ast.FormattedValue(value=value, conversion=-1, format_spec=None)]


def make_format_spec(value: ast.expr) -> ast.JoinedStr:
    if isinstance(value, ast.JoinedStr):
        return value
    return ast.JoinedStr(values=make_joined_string_parts(value))


CO_VARARGS = 0x04
CO_VARKEYWORDS = 0x08
CO_COROUTINE = 0x80
CO_ASYNC_GENERATOR = 0x200
CO_FUTURE_ANNOTATIONS = 0x1000000


def stores_annotation_text(code: object) -> bool:
    return bool(int(getattr(code, "co_flags", 0) or 0) & CO_FUTURE_ANNOTATIONS)


def annotation_from_text(value: ast.expr) -> ast.expr:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return value
    try:
        return ast.parse(value.value, mode="eval").body
    except SyntaxError:
        return value


def make_arguments(code: Any) -> ast.arguments:
    names = list(getattr(code, "co_varnames", ()) or ())
    arg_count = int(getattr(code, "co_argcount", 0) or 0)
    positional_only = int(getattr(code, "co_posonlyargcount", 0) or 0)
    keyword_only = int(getattr(code, "co_kwonlyargcount", 0) or 0)
    flags = int(getattr(code, "co_flags", 0) or 0)

    positional = [make_arg(name) for name in names[:arg_count]]
    posonlyargs = positional[:positional_only]
    args = positional[positional_only:]
    next_index = arg_count

    kwonlyargs = [
        make_arg(name) for name in names[next_index : next_index + keyword_only]
    ]
    next_index += keyword_only

    vararg = None
    if flags & CO_VARARGS and next_index < len(names):
        vararg = make_arg(names[next_index])
        next_index += 1

    kwarg = None
    if flags & CO_VARKEYWORDS and next_index < len(names):
        kwarg = make_arg(names[next_index])

    return ast.arguments(
        posonlyargs=posonlyargs,
        args=args,
        vararg=vararg,
        kwonlyargs=kwonlyargs,
        kw_defaults=[None] * len(kwonlyargs),
        kwarg=kwarg,
        defaults=[],
    )


def is_coroutine_code(code: Any) -> bool:
    flags = int(getattr(code, "co_flags", 0) or 0)
    return bool(flags & (CO_COROUTINE | CO_ASYNC_GENERATOR))


def make_function_node(
    name: str,
    code: Any,
    body: list[ast.stmt],
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    arguments = make_arguments(code)
    safe_name = safe_identifier(name)
    if is_coroutine_code(code):
        return ast.AsyncFunctionDef(
            name=safe_name,
            args=arguments,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
    return ast.FunctionDef(
        name=safe_name,
        args=arguments,
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )


@dataclass(frozen=True, slots=True)
class InlinedComprehension:
    container: ast.expr
    generators: tuple[ast.comprehension, ...]


def make_inlined_comprehension(
    container: ast.expr,
    generators: list[ast.comprehension],
) -> InlinedComprehension:
    return InlinedComprehension(container=container, generators=tuple(generators))
