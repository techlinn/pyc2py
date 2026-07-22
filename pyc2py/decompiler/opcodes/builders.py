import ast
from typing import cast


def make_sequence(opname: str, values: list[ast.expr]) -> ast.expr:
    if opname == "BUILD_TUPLE":
        return ast.Tuple(elts=values, ctx=ast.Load())

    if opname == "BUILD_LIST":
        return ast.List(elts=values, ctx=ast.Load())

    if opname == "BUILD_SET":
        return ast.Set(elts=values)
    raise ValueError(f"unsupported sequence builder: {opname}")


def make_unpacked_sequence(opname: str, values: list[ast.expr]) -> ast.expr:
    items: list[ast.expr] = [
        ast.Starred(value=value, ctx=ast.Load()) for value in values
    ]
    if opname in {"BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL"}:
        return ast.Tuple(elts=items, ctx=ast.Load())

    if opname == "BUILD_LIST_UNPACK":
        return ast.List(elts=items, ctx=ast.Load())

    if opname == "BUILD_SET_UNPACK":
        return ast.Set(elts=items)
    raise ValueError(f"unsupported unpack sequence builder: {opname}")


def make_unpack_map(values: list[ast.expr]) -> ast.Dict:
    return ast.Dict(keys=[None for _value in values], values=values)


def make_map_from_stack_items(values: list[ast.expr]) -> ast.Dict:
    keys = cast(list[ast.expr | None], values[0::2])
    return ast.Dict(keys=keys, values=values[1::2])


def make_const_key_map(
    keys: ast.expr, values: list[ast.expr], count: int
) -> ast.Dict | None:
    if not isinstance(keys, ast.Tuple) or len(keys.elts) != count:
        return None
    return ast.Dict(keys=list(keys.elts), values=values)


def iterable_to_literal_items(iterable: ast.expr) -> list[ast.expr]:
    if isinstance(iterable, (ast.List, ast.Set, ast.Tuple)):
        return list(iterable.elts)

    if isinstance(iterable, ast.Constant) and isinstance(iterable.value, tuple):
        return [ast.Constant(value=item) for item in iterable.value]
    return [ast.Starred(value=iterable, ctx=ast.Load())]


def append_to_container(
    container: ast.expr,
    value: ast.expr,
    method_name: str,
) -> ast.stmt | None:
    if isinstance(container, ast.List) and method_name == "append":
        container.elts.append(value)
        return None

    if isinstance(container, ast.Set) and method_name == "add":
        container.elts.append(value)
        return None

    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(value=container, attr=method_name, ctx=ast.Load()),
            args=[value],
            keywords=[],
        )
    )


def map_add_to_container(
    container: ast.expr,
    key: ast.expr,
    value: ast.expr,
) -> ast.stmt | None:
    if isinstance(container, ast.Dict):
        container.keys.append(key)
        container.values.append(value)
        return None

    target = ast.Subscript(value=container, slice=key, ctx=ast.Store())
    return ast.Assign(targets=[target], value=value)


def extend_container_literal(
    opname: str, container: ast.expr, iterable: ast.expr
) -> ast.stmt | None:
    items = iterable_to_literal_items(iterable)

    if opname == "LIST_EXTEND" and isinstance(container, ast.List):
        container.elts.extend(items)
        return None

    if opname == "SET_UPDATE" and isinstance(container, ast.Set):
        container.elts.extend(items)
        return None

    method_name = "extend" if opname == "LIST_EXTEND" else "update"
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(value=container, attr=method_name, ctx=ast.Load()),
            args=[iterable],
            keywords=[],
        )
    )


def update_dict_literal_or_statement(
    container: ast.expr,
    mapping: ast.expr,
) -> ast.stmt | None:
    if isinstance(container, ast.Dict):
        if isinstance(mapping, ast.Dict) and all(
            key is not None for key in mapping.keys
        ):
            container.keys.extend(mapping.keys)
            container.values.extend(mapping.values)
            return None
        container.keys.append(None)
        container.values.append(mapping)
        return None

    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(value=container, attr="update", ctx=ast.Load()),
            args=[mapping],
            keywords=[],
        )
    )
