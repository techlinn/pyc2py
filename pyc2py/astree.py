import ast
import copy
import importlib
import keyword
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from pyc2py.pyc.objects import MarshalSetValue


def safe_identifier(name: str, default_name: str = "value") -> str:
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    return default_name


def make_name(name: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=safe_identifier(name), ctx=ctx or ast.Load())


LINE_BREAK_ATTRIBUTE = "_pyc2py_line_break"
LINE_BREAK_PREFIX = "_pyc2py_line_break_"
LINE_BREAK_ALWAYS = "always"
LINE_BREAK_IN_COMPREHENSION = "comprehension"
COMPREHENSION_NODES = (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)


def name_line_breaks(tree: ast.AST) -> ast.AST:
    work: list[tuple[ast.AST, bool]] = [(tree, False)]
    while work:
        node, in_comprehension = work.pop()
        if isinstance(node, ast.JoinedStr):
            continue
        if isinstance(node, ast.Name) and is_line_break(node, in_comprehension):
            setattr(node, LINE_BREAK_ATTRIBUTE, None)
            node.id = LINE_BREAK_PREFIX + node.id
        if isinstance(node, ast.arguments):
            name_default_line_breaks(node)
        if isinstance(node, COMPREHENSION_NODES):
            move_element_line_break(node)
            work.extend(comprehension_parts(node, in_comprehension))
        else:
            work.extend(
                (child, in_comprehension) for child in ast.iter_child_nodes(node)
            )
    return tree


def name_default_line_breaks(node: ast.arguments) -> None:
    if not node.defaults:
        return
    parameters = [*node.posonlyargs, *node.args][-len(node.defaults) :]
    for parameter, default in zip(parameters, node.defaults, strict=False):
        if getattr(default, LINE_BREAK_ATTRIBUTE, None) == LINE_BREAK_ALWAYS:
            setattr(default, LINE_BREAK_ATTRIBUTE, None)
            parameter.arg = LINE_BREAK_PREFIX + parameter.arg


def move_element_line_break(node: Any) -> None:
    element = node.key if isinstance(node, ast.DictComp) else node.elt
    first = next(
        (item for item in ast.walk(element) if isinstance(item, ast.Name)), None
    )
    if getattr(first, LINE_BREAK_ATTRIBUTE, None) != LINE_BREAK_IN_COMPREHENSION:
        return
    setattr(first, LINE_BREAK_ATTRIBUTE, None)
    target = next(
        (
            item
            for item in ast.walk(node.generators[-1].target)
            if isinstance(item, ast.Name)
        ),
        None,
    )
    if target is not None:
        setattr(target, LINE_BREAK_ATTRIBUTE, LINE_BREAK_IN_COMPREHENSION)


def comprehension_parts(node: Any, outside: bool) -> list[tuple[ast.AST, bool]]:
    first = node.generators[0]
    parts: list[tuple[ast.AST, bool]] = [(first.iter, outside)]
    parts.extend(
        (child, True) for child in ast.iter_child_nodes(node) if child is not first
    )
    parts.extend(
        (child, True)
        for child in ast.iter_child_nodes(first)
        if child is not first.iter
    )
    return parts


def is_line_break(node: ast.Name, in_comprehension: bool) -> bool:
    kind = getattr(node, LINE_BREAK_ATTRIBUTE, None)
    return kind == LINE_BREAK_ALWAYS or (
        kind == LINE_BREAK_IN_COMPREHENSION and in_comprehension
    )


def make_assertion_error_name() -> ast.Name:
    name = make_name("AssertionError")
    setattr(name, "_pyc2py_assertion_error", True)
    return name


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
    result = flag_loop_exits_outside_loops(list(body))
    result = restore_guarded_loop_bodies(result)
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
        result = remove_noop_optional_clauses(result)
    if is_module:
        result = remove_module_return_statements(result)
        result = normalize_assert_statements(result)
        return keep_module_code_after_raise(result)
    result = remove_unreachable_after_terminal(result)
    result = normalize_assert_statements(result)
    if result and isinstance(result[-1], ast.Return):
        value = result[-1].value
        written = getattr(result[-1], "_pyc2py_written_return", False)
        if isinstance(value, ast.Constant) and value.value is None and not written:
            result = result[:-1]
    return remove_implicit_tail_returns(result)


def restore_guarded_loop_bodies(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [restore_guarded_loop_body(statement) for statement in statements]


def restore_guarded_loop_body(statement: ast.stmt) -> ast.stmt:
    statement = normalize_child_statement_bodies(statement, restore_guarded_loop_bodies)
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        statement.body = absorb_identity_guards(statement.body)
    return statement


def absorb_identity_guards(body: list[ast.stmt]) -> list[ast.stmt]:
    for index, statement in enumerate(body[:-1]):
        test = identity_guard_test(statement)
        if test is not None:
            rest = absorb_identity_guards(body[index + 1 :])
            return [*body[:index], ast.If(test=test, body=rest, orelse=[])]
    return body


def identity_guard_test(statement: ast.stmt) -> ast.expr | None:
    if not isinstance(statement, ast.If) or statement.orelse:
        return None
    if len(statement.body) != 1 or not isinstance(statement.body[0], ast.Continue):
        return None
    test = statement.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    return test.operand if is_identity_or_membership_test(test.operand) else None


def remove_implicit_tail_returns(statements: list[ast.stmt]) -> list[ast.stmt]:
    if not statements:
        return statements
    statements = else_behind_implicit_return(statements)
    last = statements[-1]
    if getattr(last, "_pyc2py_implicit_return", False):
        return remove_implicit_tail_returns(statements[:-1])
    if isinstance(last, ast.If):
        last.body = remove_implicit_tail_returns(last.body) or [ast.Pass()]
        last.orelse = remove_implicit_tail_returns(last.orelse)
    if isinstance(last, ast.Try) and not last.finalbody and not last.orelse:
        last.body = remove_implicit_tail_returns(last.body) or [ast.Pass()]
    return statements


def else_behind_implicit_return(statements: list[ast.stmt]) -> list[ast.stmt]:
    for index, statement in enumerate(statements[:-1]):
        if not isinstance(statement, ast.If) or statement.orelse:
            continue
        if getattr(statement.body[-1], "_pyc2py_implicit_return", False):
            statement.orelse = statements[index + 1 :]
            return statements[: index + 1]
    return statements


def flag_loop_exits_outside_loops(statements: list[ast.stmt]) -> list[ast.stmt]:
    flagger = LoopExitFlagger()
    return [flagger.visit(statement) for statement in statements]


class LoopExitFlagger(ast.NodeTransformer):
    def visit_scope(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = visit_scope
    visit_AsyncFunctionDef = visit_scope
    visit_ClassDef = visit_scope
    visit_Lambda = visit_scope

    def visit_loop(self, node: Any) -> Any:
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    visit_For = visit_loop
    visit_AsyncFor = visit_loop
    visit_While = visit_loop

    def visit_Break(self, node: ast.Break) -> ast.stmt:
        return make_helper_call_statement("__pyc2py_break_outside_loop__")

    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        return make_helper_call_statement("__pyc2py_continue_outside_loop__")


def make_helper_call_statement(name: str) -> ast.Expr:
    return ast.Expr(value=ast.Call(func=make_name(name), args=[], keywords=[]))


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
        statement.body = remove_module_return_statements(statement.body) or [ast.Pass()]
        statement.orelse = remove_module_return_statements(statement.orelse)
        return statement
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        statement.body = remove_module_return_statements(statement.body) or [ast.Pass()]
        return statement
    if isinstance(statement, ast.Try):
        statement.body = remove_module_return_statements(statement.body) or [ast.Pass()]
        statement.orelse = remove_module_return_statements(statement.orelse)
        statement.finalbody = remove_module_return_statements(statement.finalbody)
        for handler in statement.handlers:
            handler.body = remove_module_return_statements(handler.body) or [ast.Pass()]
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
    if not isinstance(statement, ast.Expr) or not is_none_call_statement(statement):
        return False
    call = statement.value
    if not isinstance(call, ast.Call) or call.keywords:
        return False
    return len(call.args) in {2, 3} and all(is_none_constant(arg) for arg in call.args)


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
        merge_child_import_from_statements(statement)
        if result and can_merge_import_from(result[-1], statement):
            previous = result[-1]
            if isinstance(previous, ast.ImportFrom) and isinstance(
                statement, ast.ImportFrom
            ):
                previous.names.extend(statement.names)
            continue
        result.append(statement)
    return result


def merge_child_import_from_statements(statement: ast.stmt) -> None:
    for field in ("body", "orelse", "finalbody"):
        children = getattr(statement, field, None)
        if isinstance(children, list):
            setattr(statement, field, merge_adjacent_import_from_statements(children))
    for handler in getattr(statement, "handlers", []):
        handler.body = merge_adjacent_import_from_statements(handler.body)


def can_merge_import_from(left: ast.stmt, right: ast.stmt) -> bool:
    if not isinstance(left, ast.ImportFrom) or not isinstance(right, ast.ImportFrom):
        return False
    source = getattr(left, "_pyc2py_import_source", None)
    if source is None or source is not getattr(right, "_pyc2py_import_source", None):
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


def normalize_try_node(statement: ast.Try) -> ast.stmt:
    statement.body = normalize_try_except_finally(statement.body)
    statement.orelse = normalize_try_except_finally(statement.orelse)
    statement.finalbody = normalize_try_except_finally(statement.finalbody)
    for handler in statement.handlers:
        handler.body = normalize_try_except_finally(handler.body)

    nested = combined_try_except_finally(statement)
    return statement if nested is None else nested


def combined_try_except_finally(
    statement: ast.Try,
) -> ast.stmt | None:
    if statement.handlers or statement.orelse or len(statement.body) != 1:
        return None
    if not statement.finalbody:
        return None

    candidate = statement.body[0]
    # ast.TryStar only exists on python 3.11+ and has the fields of ast.Try
    if not isinstance(candidate, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        return None
    inner = cast(ast.Try, candidate)
    if inner.finalbody:
        return None
    if not inner.handlers:
        return None

    return type(inner)(
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
            handler.body = trim_exception_cleanup_pair(handler.body, handler.name)
        return statement
    return statement


def trim_exception_cleanup_pair(
    statements: list[ast.stmt],
    handler_name: str | None,
) -> list[ast.stmt]:
    body = remove_synthetic_exception_cleanups(statements)
    if handler_name is None or len(body) < 2:
        return body

    if is_name_cleanup_pair(body[-2], body[-1], handler_name):
        return body[:-2] or [ast.Pass()]

    if len(body) < 3 or not is_terminal_statement(body[-1]):
        return body
    if not is_name_cleanup_pair(body[-3], body[-2], handler_name):
        return body
    return [*body[:-3], body[-1]]


def is_name_cleanup_pair(assign: ast.stmt, delete: ast.stmt, name: str) -> bool:
    return none_assignment_name(assign) == name and is_delete_name_statement(
        delete, name
    )


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
            is_synthetic_raise_statement(statement)
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


def is_synthetic_raise_statement(statement: ast.stmt) -> bool:
    return is_bare_raise_statement(statement) and getattr(
        statement, "_pyc2py_synthetic_raise", False
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
        result.extend(make_simultaneous_assignment(statements[cursor:end]))
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


def make_simultaneous_assignment(statements: list[ast.stmt]) -> list[ast.stmt]:
    assignments = [
        statement for statement in statements if isinstance(statement, ast.Assign)
    ]
    if (
        len(assignments) != len(statements)
        or len(assignments) < 2
        or any(len(statement.targets) != 1 for statement in assignments)
    ):
        return [
            merge_child_simultaneous_store_assignments(statement)
            for statement in statements
        ]
    if getattr(assignments[0], "_pyc2py_store_reversed", False):
        assignments.reverse()

    return [
        ast.Assign(
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
    ]


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
        if is_synthetic_raise_statement(statement) and index + 1 < len(statements):
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
    if not statements:
        return statements
    while statements and is_synthetic_raise_statement(statements[-1]):
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
        if statement.handlers and is_pass_only_body(statement.finalbody):
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
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return f"not {expression_key(value.operand)}"
    if isinstance(value, ast.Compare) and len(value.ops) == 1:
        operands = (value.left, *value.comparators)
        identity = " ".join(str(id(operand)) for operand in operands)
        return f"{type(value.ops[0]).__name__} {identity}"
    return str(id(value))


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
    literal_count = sum(len(term) for term in terms)
    terms = simplify_dnf_terms(terms)
    if sum(len(term) for term in terms) >= literal_count:
        return value
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
        if any(is_identity_or_membership_test(item) for item in value.values):
            return None
        inverted_values = [invert_boolean_operand(item) for item in value.values]
        if any(item is None for item in inverted_values):
            return None
        op: ast.boolop = ast.And() if isinstance(value.op, ast.Or) else ast.Or()
        return ast.BoolOp(op=op, values=[item for item in inverted_values if item])
    return None


def is_identity_or_membership_test(value: ast.expr) -> bool:
    if not isinstance(value, ast.Compare) or len(value.ops) != 1:
        return False
    if is_none_constant(value.comparators[0]):
        return False
    return isinstance(value.ops[0], (ast.Is, ast.IsNot, ast.In, ast.NotIn))


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
    if len(value.ops) != 1 or not is_none_constant(value.comparators[0]):
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
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


def invert_compare_op(op: ast.cmpop) -> ast.cmpop | None:
    inverted_op = INVERTED_COMPARE_OPS.get(type(op))
    if inverted_op is None:
        return None
    return inverted_op()


def normalize_assert_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    return [normalize_assert_statement(statement) for statement in statements]


def normalize_assert_statement(statement: ast.stmt) -> ast.stmt:
    if isinstance(statement, ast.Raise):
        constant_false = assert_from_raise(statement)
        return statement if constant_false is None else constant_false
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
    normalize_asserts_in_cases(statement)
    return statement


def normalize_asserts_in_cases(statement: ast.stmt) -> None:
    if isinstance(statement, ast.Match):
        for case in statement.cases:
            case.body = normalize_assert_statements(case.body)


def assert_from_if(statement: ast.If) -> ast.Assert | None:
    if statement.orelse or len(statement.body) != 1:
        return None
    message = assertion_message(statement.body[0])
    if message is None and not raises_plain_assertion_error(statement.body[0]):
        return None
    return ast.Assert(test=positive_assert_test(statement.test), msg=message)


def assert_from_raise(statement: ast.Raise) -> ast.Assert | None:
    message = assertion_message(statement)
    if message is None and not raises_plain_assertion_error(statement):
        return None
    return ast.Assert(test=ast.Constant(value=False), msg=message)


ASSERT_FLIPPED_OPERATORS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


def positive_assert_test(test: ast.expr) -> ast.expr:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return test.operand
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        flipped = ASSERT_FLIPPED_OPERATORS.get(type(test.ops[0]))
        if flipped is not None:
            return ast.Compare(
                left=test.left, ops=[flipped()], comparators=test.comparators
            )
    if isinstance(test, ast.BoolOp):
        inverted_bool = invert_boolean_condition(test)
        if inverted_bool is not None:
            return inverted_bool
    inverted = invert_compare_expression(test)
    if inverted is not None:
        return inverted
    if isinstance(test, ast.BoolOp):
        flipped_op = ast.Or() if isinstance(test.op, ast.And) else ast.And()
        values = [positive_assert_test(value) for value in test.values]
        return ast.BoolOp(op=flipped_op, values=values)
    return ast.UnaryOp(op=ast.Not(), operand=test)


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
    return isinstance(value, ast.Name) and getattr(
        value, "_pyc2py_assertion_error", False
    )


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


LINE_MARK = "_pyc2py_line"


def find_standard_unparser() -> Any:
    # the class behind ast.unparse is private and has moved before: it is
    # ast._Unparser up to python 3.13 and _ast_unparse.Unparser from 3.14
    found = getattr(ast, "_Unparser", None)
    if found is not None:
        return found
    try:
        module = importlib.import_module("_ast_unparse")
    except ImportError:
        return None
    return getattr(module, "Unparser", None)


STANDARD_UNPARSER: Any = find_standard_unparser()


def merge_copied_return_branches(statements: list[ast.stmt]) -> list[ast.stmt]:
    for index, statement in enumerate(statements):
        rest = statements[index + 1 :]
        if not isinstance(statement, ast.If) or statement.orelse or not rest:
            continue
        if len(statement.body) < 2 or not is_copied_return(
            statement.body[-1], rest[-1]
        ):
            continue
        joined = ast.If(
            test=statement.test,
            body=statement.body[:-1],
            orelse=merge_copied_return_branches([*rest])[:-1],
        )
        return [*statements[:index], joined, rest[-1]]
    return statements


def is_copied_return(first: ast.stmt, second: ast.stmt) -> bool:
    if not isinstance(first, ast.Return) or not isinstance(second, ast.Return):
        return False
    if first.value is None or second.value is None:
        return False
    line = getattr(first.value, LINE_MARK, None)
    if line is None or line != getattr(second.value, LINE_MARK, None):
        return False
    return ast.dump(first.value) == ast.dump(second.value)


def mark_line(node: ast.AST, line: int | None) -> None:
    if line is not None and not hasattr(node, LINE_MARK):
        setattr(node, LINE_MARK, line)


DEFAULT_LINES_MARK = "_pyc2py_default_lines"
DEF_LINE_MARK = "_pyc2py_def_line"


def mark_default_lines(value: Any, nop_lines: list[int], line: int | None) -> None:
    if line is not None and not hasattr(value, DEFAULT_LINES_MARK):
        try:
            setattr(value, DEFAULT_LINES_MARK, (tuple(nop_lines), line))
        except AttributeError:
            return


def spread_default_lines(value: object, defaults: list[ast.expr]) -> None:
    lines, def_line = getattr(value, DEFAULT_LINES_MARK, ((), None))
    if not lines or not defaults:
        return
    for index, default in enumerate(defaults):
        setattr(default, LINE_MARK, lines[min(index, len(lines) - 1)])
    setattr(defaults[0], DEF_LINE_MARK, def_line)


def unparse_keeping_lines(module: ast.AST) -> str:
    if STANDARD_UNPARSER is None:
        return ast.unparse(module)
    return str(LineKeepingUnparser().visit(module))


LineUnparserBase: Any = object if STANDARD_UNPARSER is None else STANDARD_UNPARSER


class LineKeepingUnparser(LineUnparserBase):
    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.bracket_depth = 0
        self.printed_line: int | None = None
        self.defaults_by_parameter: dict[int, ast.expr] = {}

    @contextmanager
    def delimit(self, start: str, end: str) -> Iterator[None]:
        self.bracket_depth += 1
        with super().delimit(start, end):
            yield
        self.bracket_depth -= 1

    def traverse(self, node: Any) -> None:
        if isinstance(node, ast.stmt):
            self.bracket_depth = 0
            self.printed_line = None
        self.start_marked_line(node)
        super().traverse(node)

    def start_marked_line(self, node: Any) -> None:
        line = getattr(node, LINE_MARK, None)
        if line is None:
            return
        if self.bracket_depth > 0 and self.printed_line not in {None, line}:
            if self._source and self._source[-1].endswith(" "):
                self._source[-1] = self._source[-1].rstrip(" ")
            self.write("\n" + "    " * (self._indent + 2))
        self.printed_line = line

    def visit_arguments(self, node: ast.arguments) -> None:
        positional = [*node.posonlyargs, *node.args]
        owners = positional[len(positional) - len(node.defaults) :]
        self.defaults_by_parameter = {
            id(owner): default
            for owner, default in zip(owners, node.defaults, strict=True)
        }
        def_line = getattr(node.defaults[0], DEF_LINE_MARK, None) if owners else None
        if def_line is not None:
            self.printed_line = def_line
        self.bracket_depth += 1
        super().visit_arguments(node)
        self.bracket_depth -= 1

    def visit_arg(self, node: ast.arg) -> None:
        self.start_marked_line(self.defaults_by_parameter.get(id(node)))
        super().visit_arg(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        value: ast.AST | None = node.value
        while value is not None and not hasattr(value, LINE_MARK):
            value = next(
                (
                    getattr(value, field)
                    for field in ("value", "func", "left")
                    if isinstance(getattr(value, field, None), ast.AST)
                ),
                None,
            )
        self.start_marked_line(value)
        super().visit_keyword(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.start_marked_line(node.target)
        super().visit_comprehension(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        depth, self.bracket_depth = self.bracket_depth, 0
        super().visit_JoinedStr(node)
        self.bracket_depth = depth
