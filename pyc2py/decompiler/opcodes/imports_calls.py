import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyc2py.decompiler.opcodes.stack_names import is_null_sentinel

@dataclass(frozen=True, slots=True)
class ImportValue:
    name: str
    level: Any
    fromlist: Any

@dataclass(frozen=True, slots=True)
class ImportedAttributeValue:
    module: ImportValue
    name: str

@dataclass(frozen=True, slots=True)
class CallArgumentPlan:
    raw_count: int
    legacy_keyword_count: int = 0
    has_star_args: bool = False
    has_star_kwargs: bool = False
    keyword_names_on_stack: bool = False

ExpressionFactory = Callable[[Any], ast.expr]

def make_import_statement(store_name: str, value: ImportValue) -> ast.stmt:
    module_name = value.name
    asname = None if store_name == module_name.split(".")[0] else store_name

    return ast.Import(names=[ast.alias(name=module_name, asname=asname)])

def make_import_from_statement(
    store_name: str, value: ImportedAttributeValue
) -> ast.stmt:
    module_name = value.module.name
    imported_name = value.name

    if is_dotted_import_alias(store_name, value):
        return ast.Import(names=[ast.alias(name=module_name, asname=store_name)])

    asname = None if store_name == imported_name else store_name
    return ast.ImportFrom(
        module=module_name,
        names=[ast.alias(name=imported_name, asname=asname)],
        level=get_import_level(value.module.level),
    )

def is_dotted_import_alias(store_name: str, value: ImportedAttributeValue) -> bool:
    if get_import_level(value.module.level) != 0:
        return False
    if not is_none_fromlist(value.module.fromlist):
        return False
    return value.module.name.endswith(f".{value.name}") and store_name != value.name

def is_none_fromlist(value: Any) -> bool:
    return value is None or (
        isinstance(value, ast.Constant) and value.value is None
    )

def make_import_star_statement(value: ImportValue) -> ast.ImportFrom:
    return ast.ImportFrom(
        module=value.name,
        names=[ast.alias(name="*", asname=None)],
        level=get_import_level(value.level),
    )

def get_import_level(value: Any) -> int:
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return max(0, value.value)
    return 0

LEGACY_CALL_OPS = frozenset(
    {
        "CALL_FUNCTION",
        "CALL_FUNCTION_VAR",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_VAR_KW",
        "CALL_METHOD",
    }
)

def uses_legacy_call_argument(opname: str, version: tuple[int, ...] | None) -> bool:
    if opname not in LEGACY_CALL_OPS:
        return False
    if version is None or len(version) < 2:
        return False
    return version < (3, 6)

def uses_stack_keyword_names(opname: str, version: tuple[int, ...] | None) -> bool:
    if opname == "CALL_KW":
        return True
    if opname != "CALL_FUNCTION_KW":
        return False
    if version is None or len(version) < 2:
        return False
    return version >= (3, 6)

def make_keywords(
    arguments: list[ast.expr], names: tuple[str, ...]
) -> list[ast.keyword]:
    if not names:
        return []
    if len(names) > len(arguments):
        return []

    values = arguments[-len(names) :]
    return [
        ast.keyword(arg=name, value=value)
        for name, value in zip(names, values, strict=True)
    ]

def keyword_name_from_expr(value: ast.expr) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.Str):
        return value.s
    return None

def split_legacy_call_counts(arg: int, opname: str) -> tuple[int, int, bool, bool]:
    if arg < 0:
        raise ValueError("call argument word must not be negative")

    positional_count = arg & 0xFF
    keyword_count = (arg >> 8) & 0xFF
    has_star_args = "VAR" in opname
    has_star_kwargs = "KW" in opname

    return positional_count, keyword_count, has_star_args, has_star_kwargs

def raw_legacy_call_argument_count(arg: int, opname: str) -> int:
    positional_count, keyword_count, has_star_args, has_star_kwargs = (
        split_legacy_call_counts(
            arg,
            opname,
        )
    )
    return (
        positional_count + keyword_count * 2 + int(has_star_args) + int(has_star_kwargs)
    )

def make_call_argument_plan(
    opname: str,
    arg: int,
    version: tuple[int, ...] | None,
) -> CallArgumentPlan:
    if opname == "CALL_FUNCTION_EX":
        return CallArgumentPlan(
            raw_count=2 if arg & 1 else 1,
            has_star_args=True,
            has_star_kwargs=bool(arg & 1),
        )

    if uses_stack_keyword_names(opname, version):
        return CallArgumentPlan(raw_count=arg, keyword_names_on_stack=True)

    if uses_legacy_call_argument(opname, version):
        _, keyword_count, has_star_args, has_star_kwargs = split_legacy_call_counts(
            arg, opname
        )
        return CallArgumentPlan(
            raw_count=raw_legacy_call_argument_count(arg, opname),
            legacy_keyword_count=keyword_count,
            has_star_args=has_star_args,
            has_star_kwargs=has_star_kwargs,
        )

    return CallArgumentPlan(raw_count=arg)

def split_call_arguments(
    raw_arguments: list[Any],
    keyword_names: tuple[str, ...],
    legacy_keyword_count: int,
    has_star_args: bool,
    has_star_kwargs: bool,
    expression_from_value: ExpressionFactory,
) -> tuple[list[ast.expr], list[ast.keyword]]:
    if legacy_keyword_count or has_star_args or has_star_kwargs:
        return split_legacy_call_arguments(
            raw_arguments,
            legacy_keyword_count,
            has_star_args,
            has_star_kwargs,
            expression_from_value,
        )

    arguments = [expression_from_value(value) for value in raw_arguments]
    keywords = make_keywords(arguments, keyword_names)
    positional_count = len(arguments) - len(keywords)

    return arguments[:positional_count], keywords

def split_legacy_call_arguments(
    raw_arguments: list[Any],
    keyword_count: int,
    has_star_args: bool,
    has_star_kwargs: bool,
    expression_from_value: ExpressionFactory,
) -> tuple[list[ast.expr], list[ast.keyword]]:
    working_arguments = list(raw_arguments)
    star_kwargs = (
        working_arguments.pop() if has_star_kwargs and working_arguments else None
    )
    star_args = working_arguments.pop() if has_star_args and working_arguments else None

    keyword_item_count = keyword_count * 2
    if keyword_item_count > len(working_arguments):
        return [expression_from_value(value) for value in working_arguments], []

    positional_values = working_arguments[: len(working_arguments) - keyword_item_count]
    keyword_items = working_arguments[len(positional_values) :]
    keywords = make_legacy_keywords(keyword_items, expression_from_value)
    positional = [expression_from_value(value) for value in positional_values]

    if star_args is not None:
        positional.extend(make_call_star_args(expression_from_value(star_args)))
    if star_kwargs is not None:
        keywords.extend(make_starred_keywords(expression_from_value(star_kwargs)))

    return positional, keywords

def make_call_star_args(expression: ast.expr) -> list[ast.expr]:
    if isinstance(expression, ast.Tuple):
        return list(expression.elts)

    if isinstance(expression, ast.List):
        return list(expression.elts)

    return [ast.Starred(value=expression, ctx=ast.Load())]

def make_legacy_keywords(
    keyword_items: list[Any],
    expression_from_value: ExpressionFactory,
) -> list[ast.keyword]:
    keywords: list[ast.keyword] = []

    for index in range(0, len(keyword_items), 2):
        name = keyword_name_from_expr(expression_from_value(keyword_items[index]))
        value = expression_from_value(keyword_items[index + 1])
        keywords.append(ast.keyword(arg=name, value=value))

    return keywords

def make_starred_keywords(expression: ast.expr) -> list[ast.keyword]:
    if is_null_sentinel(expression):
        return []

    if isinstance(expression, ast.Dict) and not expression.keys:
        return []

    if isinstance(expression, ast.Dict) and expression.keys:
        unpacked_keywords = make_unpacked_dict_keywords(expression)
        if unpacked_keywords is not None:
            return unpacked_keywords
        keywords: list[ast.keyword] = []
        for key, item_value in zip(expression.keys, expression.values, strict=True):
            if key is not None:
                return [ast.keyword(arg=None, value=expression)]
            keywords.append(ast.keyword(arg=None, value=item_value))
        return keywords

    return [ast.keyword(arg=None, value=expression)]

def make_unpacked_dict_keywords(expression: ast.Dict) -> list[ast.keyword] | None:
    keywords: list[ast.keyword] = []
    seen: set[str] = set()

    for key, value in zip(expression.keys, expression.values, strict=True):
        if key is None:
            keywords.append(ast.keyword(arg=None, value=value))
            continue
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        if not key.value.isidentifier() or key.value in seen:
            return None
        seen.add(key.value)
        keywords.append(ast.keyword(arg=key.value, value=value))

    return keywords

def keyword_names_from_value(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value

    if isinstance(value, ast.Tuple):
        names: list[str] = []
        for item in value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                return None
            names.append(item.value)
        return tuple(names)

    return None

def make_super_attribute(
    super_function: ast.expr,
    class_argument: ast.expr,
    self_argument: ast.expr,
    attr: str,
    uses_two_arguments: bool,
) -> ast.Attribute:
    arguments = [class_argument, self_argument] if uses_two_arguments else []

    return ast.Attribute(
        value=ast.Call(func=super_function, args=arguments, keywords=[]),
        attr=attr,
        ctx=ast.Load(),
    )
