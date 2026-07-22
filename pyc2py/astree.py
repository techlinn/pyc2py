import ast
import copy
import keyword
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pyc2py.pyc.objects import MarshalSetValue


def safe_identifier(name: str, default_name: str = "value") -> str:
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    return default_name


def make_name(name: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=safe_identifier(name), ctx=ctx or ast.Load())


def make_arg(name: object) -> ast.arg:
    return ast.arg(arg=safe_identifier(str(name), default_name="arg"), annotation=None)


def make_constant(value: Any) -> ast.expr:
    if isinstance(value, int) and not isinstance(value, bool):
        return ast.Constant(value=int(value))
    if isinstance(value, tuple):
        return ast.Tuple(elts=[make_constant(item) for item in value], ctx=ast.Load())
    if isinstance(value, list):
        return ast.List(elts=[make_constant(item) for item in value], ctx=ast.Load())
    if isinstance(value, MarshalSetValue):
        return ast.Set(elts=[make_constant(item) for item in value.items])
    if isinstance(value, frozenset):
        return ast.Set(
            elts=[make_constant(item) for item in sorted(value, key=constant_sort_key)]
        )
    return ast.Constant(value=value)


def constant_sort_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def is_none_constant(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def is_docstring_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def walk_bounded(root: ast.AST, max_nodes: int = 100_000) -> Iterator[ast.AST]:
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    work = [root]
    for _index in range(max_nodes):
        if not work:
            return
        node = work.pop()
        yield node
        work.extend(reversed(list(ast.iter_child_nodes(node))))
    raise ValueError("AST walk exceeded max_nodes")


@dataclass(frozen=True, slots=True)
class ASTShape:
    module_statements: int
    function_count: int
    class_count: int
    pass_only_bodies: int

    @property
    def checks(self) -> tuple[str, ...]:
        return (
            f"source module statements: {self.module_statements}",
            f"source function definitions: {self.function_count}",
            f"source class definitions: {self.class_count}",
            f"source pass-only bodies: {self.pass_only_bodies}",
        )


def analyze_ast_shape(root: ast.AST, max_nodes: int = 100_000) -> ASTShape:
    module_statements = len(root.body) if isinstance(root, ast.Module) else 0
    function_count = 0
    class_count = 0
    pass_only_bodies = 0

    for node in walk_bounded(root, max_nodes=max_nodes):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_count += 1
            if is_pass_only_body(node.body):
                pass_only_bodies += 1
        elif isinstance(node, ast.ClassDef):
            class_count += 1
            if is_pass_only_body(node.body):
                pass_only_bodies += 1

    return ASTShape(
        module_statements=module_statements,
        function_count=function_count,
        class_count=class_count,
        pass_only_bodies=pass_only_bodies,
    )


def is_pass_only_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    return isinstance(body[0], ast.Pass)


def clean_decompiled_body(body: list[ast.stmt], is_module: bool) -> list[ast.stmt]:
    result = list(body)
    if is_module:
        result = normalize_docstring_assignments(result)
        result = merge_adjacent_chained_assignments(result)
        result = normalize_augmented_assignments(result)
        result = merge_adjacent_import_from_statements(result)
        result = normalize_try_except_finally(result)
        result = remove_synthetic_exception_cleanups(result)
        result = remove_invalid_literal_call_statements(result)
        result = merge_simultaneous_store_assignments(result)
        result = remove_child_unreachable_after_terminal(result)
        result = normalize_boolean_conditions(result)
        result = normalize_pass_else_branches(result)
        result = remove_noop_optional_clauses(result)
    if is_module:
        result = remove_module_return_statements(result)
        return keep_module_code_after_raise(result)
    result = remove_unreachable_after_terminal(result)
    result = normalize_assert_statements(result)
    if result and isinstance(result[-1], ast.Return):
        value = result[-1].value
        if isinstance(value, ast.Constant) and value.value is None:
            return result[:-1]
    return result


def remove_module_return_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in statements:
        if isinstance(statement, ast.Return):
            continue
        result.append(remove_module_return_statement_children(statement))
    return result


def remove_module_return_statement_children(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = remove_module_return_statements(statement.body)
        statement.orelse = remove_module_return_statements(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = remove_module_return_statements(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = remove_module_return_statements(statement.body)
        statement.orelse = remove_module_return_statements(statement.orelse)
        statement.finalbody = remove_module_return_statements(statement.finalbody)
        for handler in statement.handlers:
            handler.body = remove_module_return_statements(handler.body)
        return statement
    return statement


def remove_invalid_literal_call_statements(
    statements: list[ast.stmt],
) -> list[ast.stmt]:
    cleaned: list[ast.stmt] = []
    for statement in statements:
        statement = remove_child_invalid_literal_call_statements(statement)
        if is_invalid_literal_call_statement(statement):
            continue
        if is_current_exception_statement(statement):
            continue
        cleaned.append(statement)
    return cleaned


def remove_child_invalid_literal_call_statements(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = remove_invalid_literal_call_statements(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = remove_invalid_literal_call_statements(statement.body)
        statement.orelse = remove_invalid_literal_call_statements(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = remove_invalid_literal_call_statements(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = remove_invalid_literal_call_statements(statement.body)
        statement.orelse = remove_invalid_literal_call_statements(statement.orelse)
        statement.finalbody = remove_invalid_literal_call_statements(
            statement.finalbody
        )
        for handler in statement.handlers:
            handler.body = remove_invalid_literal_call_statements(handler.body)
        return statement
    return statement


def is_invalid_literal_call_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.Call) and is_invalid_call_target(value.func)


def is_invalid_call_target(value: ast.expr) -> bool:
    return isinstance(
        value,
        (
            ast.Constant,
            ast.Dict,
            ast.List,
            ast.Set,
            ast.Tuple,
        ),
    )


def merge_adjacent_chained_assignments(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in statements:
        statement = merge_child_chained_assignments(statement)
        if result and can_merge_assignments(result[-1], statement):
            previous = result[-1]
            if isinstance(previous, ast.Assign) and isinstance(statement, ast.Assign):
                previous.targets.extend(statement.targets)
            continue
        result.append(statement)
    return result


def merge_child_chained_assignments(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if isinstance(statement, ast.ClassDef):
            statement.body = normalize_docstring_assignments(statement.body)
        statement.body = merge_adjacent_chained_assignments(statement.body)
        statement.body = normalize_augmented_assignments(statement.body)
        statement.body = merge_adjacent_import_from_statements(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = merge_adjacent_chained_assignments(statement.body)
        statement.orelse = merge_adjacent_chained_assignments(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = merge_adjacent_chained_assignments(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = merge_adjacent_chained_assignments(statement.body)
        statement.orelse = merge_adjacent_chained_assignments(statement.orelse)
        statement.finalbody = merge_adjacent_chained_assignments(statement.finalbody)
        for handler in statement.handlers:
            handler.body = merge_adjacent_chained_assignments(handler.body)
        return statement
    return statement


def can_merge_assignments(left: ast.stmt, right: ast.stmt) -> bool:
    if not isinstance(left, ast.Assign) or not isinstance(right, ast.Assign):
        return False
    if not left.targets or not right.targets:
        return False
    if assignment_targets_overlap(left.targets, right.targets):
        return False
    return left.value is right.value


def normalize_augmented_assignments(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [normalize_augmented_assignment(statement) for statement in statements]


def normalize_augmented_assignment(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, ast.Assign):
        augmented = augmented_assignment_from_assign(statement)
        return statement if augmented is None else augmented
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = normalize_augmented_assignments(statement.body)
        statement.orelse = normalize_augmented_assignments(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = normalize_augmented_assignments(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = normalize_augmented_assignments(statement.body)
        statement.orelse = normalize_augmented_assignments(statement.orelse)
        statement.finalbody = normalize_augmented_assignments(statement.finalbody)
        for handler in statement.handlers:
            handler.body = normalize_augmented_assignments(handler.body)
        return statement
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = normalize_augmented_assignments(statement.body)
        return statement
    return statement


def augmented_assignment_from_assign(statement: ast.Assign) -> ast.AugAssign | None:
    if len(statement.targets) != 1:
        return None
    if not isinstance(statement.value, ast.BinOp):
        return None
    if not getattr(statement.value, "_pyc2py_inplace", False):
        return None
    target = statement.targets[0]
    if not isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)):
        return None
    if not same_assignment_target(target, statement.value.left):
        return None
    return ast.AugAssign(
        target=target, op=statement.value.op, value=statement.value.right
    )


def same_assignment_target(target: ast.expr, value: ast.expr) -> bool:
    target_load = copy.deepcopy(target)
    set_expression_context(target_load, ast.Load())
    return ast.dump(target_load, include_attributes=False) == ast.dump(
        value,
        include_attributes=False,
    )


def set_expression_context(value: ast.expr, context: ast.expr_context) -> None:
    if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
        value.ctx = context
    if isinstance(value, (ast.Tuple, ast.List)):
        value.ctx = context
        for item in value.elts:
            set_expression_context(item, context)


def merge_adjacent_import_from_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in statements:
        if result and can_merge_import_from(result[-1], statement):
            previous = result[-1]
            if isinstance(previous, ast.ImportFrom) and isinstance(
                statement, ast.ImportFrom
            ):
                previous.names.extend(statement.names)
            continue
        result.append(statement)
    return result


def can_merge_import_from(left: ast.stmt, right: ast.stmt) -> bool:
    if not isinstance(left, ast.ImportFrom) or not isinstance(right, ast.ImportFrom):
        return False
    if left.module != right.module or left.level != right.level:
        return False
    return bool(left.names and right.names)


def normalize_try_except_finally(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [
        normalize_try_except_finally_statement(statement) for statement in statements
    ]


def normalize_try_except_finally_statement(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = normalize_try_except_finally(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = normalize_try_except_finally(statement.body)
        statement.orelse = normalize_try_except_finally(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = normalize_try_except_finally(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        return normalize_try_node(statement)
    return statement


def normalize_try_node(statement: ast.Try) -> ast.Try:
    statement.body = normalize_try_except_finally(statement.body)
    statement.orelse = normalize_try_except_finally(statement.orelse)
    statement.finalbody = normalize_try_except_finally(statement.finalbody)
    for handler in statement.handlers:
        handler.body = normalize_try_except_finally(handler.body)

    nested = combined_try_except_finally(statement)
    return statement if nested is None else nested


def combined_try_except_finally(statement: ast.Try) -> ast.Try | None:
    if statement.handlers or statement.orelse or len(statement.body) != 1:
        return None
    if not statement.finalbody:
        return None

    inner = statement.body[0]
    if not isinstance(inner, ast.Try):
        return None
    if inner.finalbody:
        return None
    if not inner.handlers:
        return None

    return ast.Try(
        body=inner.body or [ast.Pass()],
        handlers=inner.handlers,
        orelse=inner.orelse,
        finalbody=statement.finalbody,
    )


def remove_synthetic_exception_cleanups(statements: list[ast.stmt]) -> list[ast.stmt]:
    cleaned = [
        remove_synthetic_exception_cleanup(statement) for statement in statements
    ]
    return remove_synthetic_exception_helper_sequences(cleaned)


def remove_synthetic_exception_cleanup(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = remove_synthetic_exception_cleanups(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = remove_synthetic_exception_cleanups(statement.body)
        statement.orelse = remove_synthetic_exception_cleanups(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = remove_synthetic_exception_cleanups(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = remove_synthetic_exception_cleanups(statement.body)
        statement.orelse = remove_synthetic_exception_cleanups(statement.orelse)
        statement.finalbody = remove_synthetic_exception_cleanups(statement.finalbody)
        for handler in statement.handlers:
            handler.body = trim_exception_cleanup_pair(handler.body)
        return statement
    return statement


def trim_exception_cleanup_pair(statements: list[ast.stmt]) -> list[ast.stmt]:
    body = remove_synthetic_exception_cleanups(statements)
    if len(body) < 2:
        return body

    assigned_name = none_assignment_name(body[-2])
    if assigned_name is not None and is_delete_name_statement(body[-1], assigned_name):
        return body[:-2] or [ast.Pass()]

    if len(body) < 3 or not is_terminal_statement(body[-1]):
        return body
    assigned_name = none_assignment_name(body[-3])
    if assigned_name is None:
        return body
    if not is_delete_name_statement(body[-2], assigned_name):
        return body
    return [*body[:-3], body[-1]]


def remove_synthetic_exception_helper_sequences(
    statements: list[ast.stmt],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    cursor = 0
    while cursor < len(statements):
        statement = statements[cursor]
        synthetic_cleanup = synthetic_reraise_cleanup_statement(statement)
        if synthetic_cleanup is not None:
            result.extend(synthetic_cleanup)
            cursor += 1
            continue
        if (
            isinstance(statement, ast.Raise)
            and statement.exc is None
            and statement.cause is None
            and result
            and is_cleanup_statement(result[-1])
        ):
            cursor += 1
            continue
        if is_synthetic_with_cleanup_start(statements, cursor):
            cursor += 3
            continue
        if is_synthetic_with_reraise_guard(statements[cursor]):
            cursor += 1
            continue
        if is_synthetic_exception_match_reraise_guard(statements[cursor]):
            cursor += 1
            continue
        if (
            cursor + 1 < len(statements)
            and is_synthetic_exception_match_continue_guard(statements[cursor])
            and is_single_bare_raise_body([statements[cursor + 1]])
        ):
            cursor += 2
            while cursor < len(statements) and is_single_bare_raise_body(
                [statements[cursor]]
            ):
                cursor += 1
            continue
        result.append(statements[cursor])
        cursor += 1
    return result


def synthetic_reraise_cleanup_statement(statement: ast.stmt) -> list[ast.stmt] | None:
    if isinstance(statement, ast.Try):
        return synthetic_reraise_cleanup_try_body(statement)
    if isinstance(statement, ast.If) and not statement.orelse:
        body = synthetic_reraise_cleanup_body(statement.body)
        if body is None:
            return None
        statement.body = body or [ast.Pass()]
        return [statement]
    return None


def synthetic_reraise_cleanup_body(body: list[ast.stmt]) -> list[ast.stmt] | None:
    if len(body) != 1:
        return None
    cleanup = synthetic_reraise_cleanup_statement(body[0])
    if cleanup is None:
        return None
    return cleanup


def synthetic_reraise_cleanup_try_body(statement: ast.Try) -> list[ast.stmt] | None:
    if statement.orelse or statement.finalbody or len(statement.handlers) != 1:
        return None
    if not statement.body or not is_single_bare_raise_body(statement.body[-1:]):
        return None
    handler = statement.handlers[0]
    if handler.name is not None or not is_single_bare_raise_body(handler.body):
        return None
    if not is_exception_name(handler.type):
        return None
    cleanup = statement.body[:-1]
    if not cleanup or any(is_terminal_statement(item) for item in cleanup):
        return None
    return cleanup


def is_exception_name(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Name) and value.id in {"BaseException", "Exception"}


def is_cleanup_statement(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.If):
        return bool(statement.body) and all(
            is_cleanup_statement(item) for item in statement.body
        )
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Call)
    if isinstance(statement, ast.Try):
        return bool(statement.body) and all(
            is_cleanup_statement(item) for item in statement.body
        )
    return False


def is_synthetic_with_cleanup_start(
    statements: list[ast.stmt],
    index: int,
) -> bool:
    if index + 2 >= len(statements):
        return False
    return (
        is_none_call_statement(statements[index])
        and is_synthetic_with_reraise_guard(statements[index + 1])
        and is_current_exception_statement(statements[index + 2])
    )


def is_none_call_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.Call) and is_none_constant(value.func)


def is_current_exception_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.Name) and value.id == "__pyc2py_current_exception__"


def is_synthetic_with_reraise_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False
    if statement.orelse or not is_single_bare_raise_body(statement.body):
        return False
    test = statement.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return False
    return is_helper_call(test.operand, "__pyc2py_with_except_start__")


def is_synthetic_exception_match_reraise_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False
    if (
        is_pass_only_body(statement.body)
        and is_single_bare_raise_body(statement.orelse)
        and is_helper_call(statement.test, "__pyc2py_check_exc_match__")
    ):
        return True
    if statement.orelse or not is_single_bare_raise_body(statement.body):
        return False
    test = statement.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return False
    return is_helper_call(test.operand, "__pyc2py_check_exc_match__")


def is_synthetic_exception_match_continue_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False
    if statement.orelse or not statement.body:
        return False
    if not isinstance(statement.body[-1], ast.Continue):
        return False
    if any(is_terminal_statement(item) for item in statement.body[:-1]):
        return False
    return is_helper_call(statement.test, "__pyc2py_check_exc_match__")


def is_single_bare_raise_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    return body[0].exc is None and body[0].cause is None


def is_helper_call(value: ast.expr, name: str) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and (value.func.id == name)
    )


def is_terminal_statement(statement: ast.stmt) -> bool:
    return isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return))


def is_bare_raise_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Raise)
        and statement.exc is None
        and statement.cause is None
    )


def none_assignment_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return None
    if (
        not isinstance(statement.value, ast.Constant)
        or statement.value.value is not None
    ):
        return None
    return target.id


def is_delete_name_statement(statement: ast.stmt, name: str) -> bool:
    if not isinstance(statement, ast.Delete) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return isinstance(target, ast.Name) and target.id == name


def merge_simultaneous_store_assignments(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    cursor = 0
    while cursor < len(statements):
        group = store_group_id(statements[cursor])
        if group is None:
            result.append(
                merge_child_simultaneous_store_assignments(statements[cursor])
            )
            cursor += 1
            continue

        end = cursor + 1
        while end < len(statements) and store_group_id(statements[end]) == group:
            end += 1
        result.append(make_simultaneous_assignment(statements[cursor:end]))
        cursor = end
    return result


def merge_child_simultaneous_store_assignments(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = merge_simultaneous_store_assignments(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = merge_simultaneous_store_assignments(statement.body)
        statement.orelse = merge_simultaneous_store_assignments(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = merge_simultaneous_store_assignments(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = merge_simultaneous_store_assignments(statement.body)
        statement.orelse = merge_simultaneous_store_assignments(statement.orelse)
        statement.finalbody = merge_simultaneous_store_assignments(statement.finalbody)
        for handler in statement.handlers:
            handler.body = merge_simultaneous_store_assignments(handler.body)
        return statement
    return statement


def store_group_id(statement: ast.stmt) -> int | None:
    group = getattr(statement, "_pyc2py_store_group", None)
    return group if isinstance(group, int) else None


def make_simultaneous_assignment(statements: list[ast.stmt]) -> ast.stmt:
    assignments = [
        statement for statement in statements if isinstance(statement, ast.Assign)
    ]
    if len(assignments) != len(statements) or len(assignments) < 2:
        return merge_child_simultaneous_store_assignments(statements[0])
    if any(len(statement.targets) != 1 for statement in assignments):
        return merge_child_simultaneous_store_assignments(statements[0])

    return ast.Assign(
        targets=[
            ast.Tuple(
                elts=[statement.targets[0] for statement in assignments],
                ctx=ast.Store(),
            )
        ],
        value=ast.Tuple(
            elts=[statement.value for statement in assignments],
            ctx=ast.Load(),
        ),
    )


def normalize_docstring_assignments(statements: list[ast.stmt]) -> list[ast.stmt]:
    if not statements:
        return statements
    docstring = docstring_from_assignment(statements[0])
    if docstring is None:
        return statements
    return [docstring, *statements[1:]]


def docstring_from_assignment(statement: ast.stmt) -> ast.Expr | None:
    if not is_assignment_to(statement, "__doc__"):
        return None
    if not isinstance(statement, ast.Assign):
        return None
    value = statement.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return ast.Expr(value=ast.Constant(value=value.value))


def assignment_targets_overlap(
    left_targets: list[ast.expr],
    right_targets: list[ast.expr],
) -> bool:
    seen = {ast.dump(target, include_attributes=False) for target in left_targets}
    for target in right_targets:
        key = ast.dump(target, include_attributes=False)
        if key in seen:
            return True
        seen.add(key)
    return False


def remove_unreachable_after_terminal(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for index, statement in enumerate(statements):
        if is_bare_raise_statement(statement) and index + 1 < len(statements):
            continue
        result.append(statement)
        if is_terminal_statement(statement):
            return result
    return result


def remove_child_unreachable_after_terminal(
    statements: list[ast.stmt],
) -> list[ast.stmt]:
    return [
        remove_child_unreachable_after_terminal_statement(statement)
        for statement in statements
    ]


def remove_child_unreachable_after_terminal_statement(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = clean_non_handler_body(statement.body)
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        statement.body = clean_non_handler_body(statement.body)
        statement.orelse = clean_non_handler_body(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = clean_non_handler_body(statement.body)
        return statement
    if isinstance(statement, ast.Try):
        statement.body = clean_non_handler_body(statement.body)
        statement.orelse = clean_non_handler_body(statement.orelse)
        statement.finalbody = clean_non_handler_body(statement.finalbody)
        for handler in statement.handlers:
            handler.body = remove_unreachable_after_terminal(
                remove_child_unreachable_after_terminal(handler.body)
            )
        return statement
    return statement


def clean_non_handler_body(statements: list[ast.stmt]) -> list[ast.stmt]:
    return remove_bare_raise_tail(
        remove_unreachable_after_terminal(
            remove_child_unreachable_after_terminal(statements)
        )
    )


def remove_bare_raise_tail(statements: list[ast.stmt]) -> list[ast.stmt]:
    while statements and is_bare_raise_statement(statements[-1]):
        statements = statements[:-1]
    return statements or [ast.Pass()]


def normalize_boolean_conditions(statements: list[ast.stmt]) -> list[ast.stmt]:
    return BooleanStatementNormalizer().normalize_body(statements)


def normalize_boolean_condition(value: ast.expr) -> ast.expr:
    return normalize_boolean_expression(value)


def normalize_boolean_expression(value: ast.expr) -> ast.expr:
    normalizer = BooleanExpressionNormalizer()
    return normalizer.visit(value)


class BooleanExpressionNormalizer(ast.NodeTransformer):
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.op, ast.Not):
            return node
        inverted = invert_boolean_condition(node.operand)
        if inverted is None:
            return node
        return inverted

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        node.values = flatten_bool_values(type(node.op), node.values)
        simplified = simplify_boolean_absorption(node)
        if isinstance(simplified, ast.BoolOp) and isinstance(simplified.op, ast.Or):
            return simplify_or_dnf(simplified)
        return simplified

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        if is_nested_none_identity_compare(node):
            return node.left
        return node


def is_nested_none_identity_compare(node: ast.Compare) -> bool:
    if not isinstance(node.left, ast.Compare):
        return False
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Is):
        return False
    comparator = node.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value is None


class BooleanStatementNormalizer(ast.NodeTransformer):
    def normalize_body(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        return [self.visit_statement(statement) for statement in statements]

    def visit_statement(self, statement: ast.stmt) -> ast.stmt:
        return self.visit(statement)

    def visit_body_owner(self, node: Any) -> Any:
        node.body = self.normalize_body(node.body)
        return node

    def visit_body_else_owner(self, node: Any) -> Any:
        node.body = self.normalize_body(node.body)
        node.orelse = self.normalize_body(node.orelse)
        return node

    def visit_value_owner(self, node: Any) -> Any:
        node.value = normalize_boolean_expression(node.value)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self.visit_body_owner(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        return self.visit_body_owner(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self.visit_body_owner(node)

    def visit_If(self, node: ast.If) -> ast.If:
        node.test = normalize_boolean_condition(node.test)
        return self.visit_body_else_owner(node)

    def visit_While(self, node: ast.While) -> ast.While:
        node.test = normalize_boolean_condition(node.test)
        return self.visit_body_else_owner(node)

    def visit_For(self, node: ast.For) -> ast.For:
        return self.visit_body_else_owner(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        return self.visit_body_else_owner(node)

    def visit_With(self, node: ast.With) -> ast.With:
        return self.visit_body_owner(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AsyncWith:
        return self.visit_body_owner(node)

    def visit_Try(self, node: ast.Try) -> ast.Try:
        node.body = self.normalize_body(node.body)
        node.orelse = self.normalize_body(node.orelse)
        node.finalbody = self.normalize_body(node.finalbody)
        for handler in node.handlers:
            handler.body = self.normalize_body(handler.body)
        return node

    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        node.test = normalize_boolean_condition(node.test)
        if node.msg is not None:
            node.msg = normalize_boolean_expression(node.msg)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        return self.visit_value_owner(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value is not None:
            node.value = normalize_boolean_expression(node.value)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        return self.visit_value_owner(node)

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is not None:
            node.value = normalize_boolean_expression(node.value)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        return self.visit_value_owner(node)


def normalize_pass_else_branches(statements: list[ast.stmt]) -> list[ast.stmt]:
    normalized: list[ast.stmt] = []
    for statement in statements:
        normalized_statement = normalize_pass_else_branch_statement(statement)
        if normalized_statement is None:
            continue
        normalized.append(normalized_statement)
    return normalized


def normalize_child_statement_bodies(
    statement: ast.stmt,
    normalize: Any,
) -> ast.stmt:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        statement.body = normalize(statement.body)
    elif isinstance(statement, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        statement.body = normalize(statement.body)
        statement.orelse = normalize(statement.orelse)
    elif isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = normalize(statement.body)
    elif isinstance(statement, ast.Try):
        statement.body = normalize(statement.body)
        statement.orelse = normalize(statement.orelse)
        statement.finalbody = normalize(statement.finalbody)
        for handler in statement.handlers:
            handler.body = normalize(handler.body)
    return statement


def normalize_pass_else_branch_statement(statement: ast.stmt) -> ast.stmt | None:
    statement = normalize_child_statement_bodies(
        statement, normalize_pass_else_branches
    )
    if not isinstance(statement, ast.If):
        return statement
    if is_pass_only_body(statement.body) and statement.orelse:
        inverted = invert_boolean_operand(statement.test)
        if inverted is not None:
            return ast.If(
                test=normalize_boolean_condition(inverted),
                body=statement.orelse,
                orelse=[],
            )
    if is_pass_only_body(statement.body) and not statement.orelse:
        return None
    return statement


def remove_noop_optional_clauses(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [
        statement
        for statement in (
            remove_noop_optional_clause_statement(statement) for statement in statements
        )
        if statement is not None
    ]


def remove_noop_optional_clause_statement(statement: ast.stmt) -> ast.stmt | None:
    statement = normalize_child_statement_bodies(
        statement, remove_noop_optional_clauses
    )
    if isinstance(statement, ast.If):
        if is_pass_only_body(statement.orelse):
            statement.orelse = []
        return statement
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        if is_pass_only_body(statement.orelse):
            statement.orelse = []
        return statement
    if isinstance(statement, ast.Try):
        if is_pass_only_body(statement.orelse):
            statement.orelse = []
        if is_pass_only_body(statement.finalbody):
            statement.finalbody = []
        if not statement.handlers and not statement.finalbody:
            return None
    return statement


def flatten_bool_values(
    op_type: type[ast.boolop],
    values: list[ast.expr],
) -> list[ast.expr]:
    flattened: list[ast.expr] = []
    for item in values:
        if isinstance(item, ast.BoolOp) and isinstance(item.op, op_type):
            flattened.extend(item.values)
        else:
            flattened.append(item)
    return flattened


def simplify_boolean_absorption(value: ast.BoolOp) -> ast.expr:
    changed = True
    while changed:
        changed = False
        if isinstance(value.op, ast.Or):
            changed = simplify_or_absorption(value)
        elif isinstance(value.op, ast.And):
            changed = simplify_and_absorption(value)
    if len(value.values) == 1:
        return value.values[0]
    return value


def simplify_or_absorption(value: ast.BoolOp) -> bool:
    keys = {expression_key(item) for item in value.values}
    if not keys:
        return False

    changed = False
    simplified: list[ast.expr] = []
    for item in value.values:
        if isinstance(item, ast.BoolOp) and isinstance(item.op, ast.And):
            replacement = remove_complementary_term(item.values, keys, ast.And())
            if replacement is not None:
                simplified.append(replacement)
                changed = True
                continue
        simplified.append(item)
    if changed:
        value.values = flatten_bool_values(type(value.op), simplified)
    return changed


def simplify_and_absorption(value: ast.BoolOp) -> bool:
    keys = {expression_key(item) for item in value.values}
    if not keys:
        return False

    changed = False
    simplified: list[ast.expr] = []
    for item in value.values:
        if isinstance(item, ast.BoolOp) and isinstance(item.op, ast.Or):
            replacement = remove_complementary_term(item.values, keys, ast.Or())
            if replacement is not None:
                simplified.append(replacement)
                changed = True
                continue
        simplified.append(item)
    if changed:
        value.values = flatten_bool_values(type(value.op), simplified)
    return changed


def remove_complementary_term(
    values: list[ast.expr],
    keys: set[str],
    op: ast.boolop,
) -> ast.expr | None:
    remaining: list[ast.expr] = []
    removed = False
    for item in values:
        complement = complementary_expression_key(item)
        if complement in keys:
            removed = True
            continue
        remaining.append(item)

    if not removed:
        return None
    if not remaining:
        return ast.Constant(value=True)
    if len(remaining) == 1:
        return remaining[0]
    return ast.BoolOp(op=op, values=remaining)


def expression_key(value: ast.expr) -> str:
    return ast.dump(value, include_attributes=False)


def complementary_expression_key(value: ast.expr) -> str | None:
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return expression_key(value.operand)
    inverted = invert_compare_expression(value)
    if inverted is None:
        return None
    return expression_key(inverted)


MAX_DNF_TERMS = 64
MAX_DNF_LITERALS = 12


def simplify_or_dnf(value: ast.BoolOp) -> ast.expr:
    terms = dnf_terms(value)
    if terms is None:
        return value
    terms = simplify_dnf_terms(terms)
    if not terms:
        return ast.Constant(value=False)
    if any(not term for term in terms):
        return ast.Constant(value=True)
    rebuilt = [make_and_term(term) for term in terms]
    if len(rebuilt) == 1:
        return rebuilt[0]
    return ast.BoolOp(op=ast.Or(), values=rebuilt)


def dnf_terms(value: ast.expr) -> list[list[ast.expr]] | None:
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        return dnf_or_terms(value.values)

    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
        return dnf_and_terms(value.values)

    return [[value]]


def dnf_or_terms(values: list[ast.expr]) -> list[list[ast.expr]] | None:
    terms: list[list[ast.expr]] = []
    for item in values:
        item_terms = dnf_terms(item)
        if item_terms is None:
            return None
        terms.extend(item_terms)
        if len(terms) > MAX_DNF_TERMS:
            return None
    return terms


def dnf_and_terms(values: list[ast.expr]) -> list[list[ast.expr]] | None:
    terms: list[list[ast.expr]] = [[]]
    for item in values:
        item_terms = dnf_terms(item)
        if item_terms is None:
            return None
        product = dnf_product_terms(terms, item_terms)
        if product is None:
            return None
        terms = product
    return terms


def dnf_product_terms(
    left_terms: list[list[ast.expr]],
    right_terms: list[list[ast.expr]],
) -> list[list[ast.expr]] | None:
    combined: list[list[ast.expr]] = []
    for left in left_terms:
        for right in right_terms:
            merged = [*left, *right]
            if len(merged) > MAX_DNF_LITERALS:
                return None
            combined.append(merged)
            if len(combined) > MAX_DNF_TERMS:
                return None
    return combined


def simplify_dnf_terms(terms: list[list[ast.expr]]) -> list[list[ast.expr]]:
    current: list[list[ast.expr]] = []
    for term in terms:
        normalized = normalize_dnf_term(term)
        if normalized is None:
            continue
        current.append(normalized)

    changed = True
    while changed:
        changed = False
        reduced = remove_superset_terms(current)
        if len(reduced) != len(current):
            current = reduced
            changed = True
        merged = merge_complementary_dnf_terms(current)
        if len(merged) != len(current):
            current = merged
            changed = True
    return current


def normalize_dnf_term(term: list[ast.expr]) -> list[ast.expr] | None:
    seen: dict[str, ast.expr] = {}
    for item in term:
        key = expression_key(item)
        complement = complementary_expression_key(item)
        if complement in seen:
            return None
        seen[key] = item
    return list(seen.values())


def remove_superset_terms(terms: list[list[ast.expr]]) -> list[list[ast.expr]]:
    key_sets = [frozenset(expression_key(item) for item in term) for term in terms]
    keep: list[list[ast.expr]] = []
    for index, term in enumerate(terms):
        keys = key_sets[index]
        redundant = False
        for other_index, other_keys in enumerate(key_sets):
            if index == other_index:
                continue
            if other_keys < keys:
                redundant = True
                break
        if not redundant:
            keep.append(term)
    return keep


def merge_complementary_dnf_terms(terms: list[list[ast.expr]]) -> list[list[ast.expr]]:
    for left_index, left in enumerate(terms):
        for right_index in range(left_index + 1, len(terms)):
            right = terms[right_index]
            merged = merge_complementary_dnf_pair(left, right)
            if merged is None:
                continue
            return [
                term
                for index, term in enumerate(terms)
                if index not in {left_index, right_index}
            ] + [merged]
    return terms


def merge_complementary_dnf_pair(
    left: list[ast.expr],
    right: list[ast.expr],
) -> list[ast.expr] | None:
    left_keys = {expression_key(item): item for item in left}
    right_keys = {expression_key(item): item for item in right}
    common = set(left_keys) & set(right_keys)
    if len(left_keys) != len(right_keys):
        return None
    if len(left_keys) - len(common) != 1:
        return None
    left_only_key = next(iter(set(left_keys) - common))
    right_only_key = next(iter(set(right_keys) - common))
    if (
        complementary_expression_key(left_keys[left_only_key]) != right_only_key
        and complementary_expression_key(right_keys[right_only_key]) != left_only_key
    ):
        return None
    return [left_keys[key] for key in left_keys if key in common]


def make_and_term(term: list[ast.expr]) -> ast.expr:
    if not term:
        return ast.Constant(value=True)
    if len(term) == 1:
        return term[0]
    return ast.BoolOp(op=ast.And(), values=term)


def invert_boolean_condition(value: ast.expr) -> ast.expr | None:
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return normalize_boolean_condition(value.operand)
    inverted_compare = invert_compare_expression(value)
    if inverted_compare is not None:
        return inverted_compare
    if isinstance(value, ast.BoolOp):
        inverted_values = [invert_boolean_operand(item) for item in value.values]
        if any(item is None for item in inverted_values):
            return None
        op: ast.boolop = ast.And() if isinstance(value.op, ast.Or) else ast.Or()
        return ast.BoolOp(op=op, values=[item for item in inverted_values if item])
    return None


def invert_boolean_operand(value: ast.expr) -> ast.expr | None:
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return normalize_boolean_condition(value.operand)
    inverted_compare = invert_compare_expression(value)
    if inverted_compare is not None:
        return inverted_compare
    return ast.UnaryOp(op=ast.Not(), operand=value)


def invert_compare_expression(value: ast.expr) -> ast.Compare | None:
    if not isinstance(value, ast.Compare):
        return None
    if len(value.ops) != 1:
        return None
    inverted_op = invert_compare_op(value.ops[0])
    if inverted_op is None:
        return None
    return ast.Compare(
        left=value.left,
        ops=[inverted_op],
        comparators=value.comparators,
    )


INVERTED_COMPARE_OPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


def invert_compare_op(op: ast.cmpop) -> ast.cmpop | None:
    inverted_op = INVERTED_COMPARE_OPS.get(type(op))
    if inverted_op is None:
        return None
    return inverted_op()


def normalize_assert_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [normalize_assert_statement(statement) for statement in statements]


def normalize_assert_statement(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, ast.If):
        assertion = assert_from_if(statement)
        if assertion is not None:
            return assertion
        statement.body = normalize_assert_statements(statement.body)
        statement.orelse = normalize_assert_statements(statement.orelse)
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        statement.body = normalize_assert_statements(statement.body)
        statement.orelse = normalize_assert_statements(statement.orelse)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = normalize_assert_statements(statement.body)
    if isinstance(statement, ast.Try):
        statement.body = normalize_assert_statements(statement.body)
        statement.orelse = normalize_assert_statements(statement.orelse)
        statement.finalbody = normalize_assert_statements(statement.finalbody)
        for handler in statement.handlers:
            handler.body = normalize_assert_statements(handler.body)
    return statement


def assert_from_if(statement: ast.If) -> ast.Assert | None:
    if statement.orelse or len(statement.body) != 1:
        return None
    test = positive_assert_test(statement.test)
    if test is None:
        return None
    message = assertion_message(statement.body[0])
    if message is None and not raises_plain_assertion_error(statement.body[0]):
        return None
    return ast.Assert(test=test, msg=message)


def positive_assert_test(test: ast.expr) -> ast.expr | None:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return test.operand
    return None


def assertion_message(statement: ast.stmt) -> ast.expr | None:
    if not isinstance(statement, ast.Raise):
        return None
    call = statement.exc
    if not isinstance(call, ast.Call):
        return None
    if not is_assertion_error_expr(call.func) or len(call.args) != 1 or call.keywords:
        return None
    return call.args[0]


def raises_plain_assertion_error(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Raise):
        return False
    return is_assertion_error_expr(statement.exc)


def is_assertion_error_expr(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Name) and value.id == "AssertionError"


def keep_module_code_after_raise(statements: list[ast.stmt]) -> list[ast.stmt]:
    return statements


def add_global_declarations(
    statements: list[ast.stmt],
    names: list[str] | set[str],
) -> list[ast.stmt]:
    ordered_names = unique_global_names(names)
    if not ordered_names:
        return statements

    declaration = ast.Global(names=ordered_names)
    if not statements:
        return [declaration]
    if is_docstring_statement(statements[0]):
        return [statements[0], declaration, *statements[1:]]
    return [declaration, *statements]


def add_nonlocal_declarations(
    statements: list[ast.stmt],
    names: list[str] | set[str],
) -> list[ast.stmt]:
    ordered_names = unique_global_names(names)
    if not ordered_names:
        return statements

    declaration = ast.Nonlocal(names=ordered_names)
    if not statements:
        return [declaration]
    if is_docstring_statement(statements[0]):
        return [statements[0], declaration, *statements[1:]]
    return [declaration, *statements]


def add_module_global_declarations(
    statements: list[ast.stmt],
    names: list[str] | set[str],
) -> list[ast.stmt]:
    ordered_names = unique_global_names(names)
    if not ordered_names:
        return statements

    declaration = ast.Global(names=ordered_names)
    if not statements:
        return [declaration]

    insert_index = first_global_store_statement_index(statements, set(ordered_names))
    if insert_index is None:
        if is_docstring_statement(statements[0]):
            return [statements[0], declaration, *statements[1:]]
        return [declaration, *statements]
    return [*statements[:insert_index], declaration, *statements[insert_index:]]


def first_global_store_statement_index(
    statements: list[ast.stmt],
    names: set[str],
) -> int | None:
    for index, statement in enumerate(statements):
        if is_docstring_statement(statement):
            continue
        if statement_stores_any_name(statement, names):
            return index
    return None


def statement_stores_any_name(statement: ast.stmt, names: set[str]) -> bool:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in names
        ):
            return True
    return False


def unique_global_names(names: list[str] | set[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        if safe_identifier(name) != name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def is_name_store(node: ast.AST, name: str | None = None) -> bool:
    if not isinstance(node, ast.Name):
        return False
    if not isinstance(node.ctx, ast.Store):
        return False
    return name is None or node.id == name


def is_assignment_to(statement: ast.stmt, name: str) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    return is_name_store(statement.targets[0], name)
