import ast
from typing import Any

from pyc2py.astree import (
    LINE_MARK,
    SCOPES,
    STATEMENT_LINE,
    TERMINAL_STATEMENTS,
    percent_format,
)
from pyc2py.types import VersionTuple

VersionRanges = tuple[tuple[VersionTuple, VersionTuple | None], ...]

# set where the bytecode shows an if at a loop's end had an else, not a continue
ELSE_AT_LOOP_END = "_pyc2py_else_at_loop_end"

# where both spellings compile alike (tests/versions/sources/spellings.py)
FOR_END_ELSE: VersionRanges = (((3, 5), None),)
WHILE_END_ELSE: VersionRanges = (((3, 5), (3, 10)), ((3, 14), None))
HANDLER_END_ELSE: VersionRanges = (((3, 13), None),)
HANDLER_END_CONTINUE: VersionRanges = (((3, 10), (3, 13)),)
JOINED_TESTS: VersionRanges = (((3, 7), None),)
# on 3.5 and 3.6 a `not` test compiles differently in the two spellings
JOINED_PLAIN_TESTS: VersionRanges = (((3, 5), (3, 7)),)
PERCENT_FORMAT: VersionRanges = (((3, 11), None),)

PERCENT_ONLY = (ord("s"), ord("a"))


def in_ranges(version: VersionTuple, ranges: VersionRanges) -> bool:
    return any(
        first <= version and (last is None or version < last) for first, last in ranges
    )


def use_common_spellings(module: ast.Module, version: VersionTuple | None) -> None:
    # where two spellings compile the same, write the one real code uses
    if version is not None:
        CommonSpeller(version[:2]).visit(module)


class CommonSpeller(ast.NodeTransformer):
    def __init__(self, version: VersionTuple) -> None:
        self.version = version
        # the line of the statement being visited
        self.line: int | None = None

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, ast.stmt):
            return super().visit(node)

        outer, self.line = self.line, getattr(node, STATEMENT_LINE, self.line)

        try:
            return super().visit(node)
        finally:
            self.line = outer

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body = self.loop_end(node.body, FOR_END_ELSE)
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        self.generic_visit(node)
        node.body = self.loop_end(node.body, FOR_END_ELSE)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body = self.loop_end(node.body, WHILE_END_ELSE)
        return node

    def loop_end(self, body: list[ast.stmt], ranges: VersionRanges) -> list[ast.stmt]:
        # what follows the if here runs last in the loop, as its else would
        for index, statement in enumerate(body[:-1]):
            if (
                isinstance(statement, ast.If)
                and self.ends_with_else_continue(statement, ranges)
                and all(map(is_plain, body[index + 1 :]))
            ):
                statement.body = statement.body[:-1]
                statement.orelse = self.loop_end(body[index + 1 :], ranges)
                return body[: index + 1]

        last = body[-1] if body else None
        if isinstance(last, ast.Try) and not last.finalbody and not last.orelse:
            for handler in last.handlers:
                handler.body = self.handler_end(handler.body)

        return body

    def handler_end(self, body: list[ast.stmt]) -> list[ast.stmt]:
        trailing = len(body) > 1 and isinstance(body[-1], ast.Continue)
        if trailing and in_ranges(self.version, HANDLER_END_CONTINUE):
            body = body[:-1]

        return self.loop_end(body, HANDLER_END_ELSE)

    def ends_with_else_continue(self, statement: ast.If, ranges: VersionRanges) -> bool:
        if statement.orelse or len(statement.body) < 2:
            return False

        *rest, last = statement.body
        if not isinstance(last, ast.Continue) or not all(map(is_plain, rest)):
            return False
        if self.version < (3, 5):
            # before 3.5 a continue leaves the if's own jump behind it
            return bool(getattr(statement, ELSE_AT_LOOP_END, False))

        return in_ranges(self.version, ranges)

    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        inner = node.body[0] if len(node.body) == 1 else None
        if isinstance(inner, ast.If) and self.joins(node, inner):
            node.test = ast.BoolOp(
                op=ast.And(), values=[*and_terms(node.test), *and_terms(inner.test)]
            )
            node.body = inner.body

        return node

    def joins(self, outer: ast.If, inner: ast.If) -> bool:
        if outer.orelse or inner.orelse or not all(map(is_plain, inner.body)):
            return False
        if any(isinstance(part, ast.NamedExpr) for part in ast.walk(inner.test)):
            return False
        if in_ranges(self.version, JOINED_TESTS):
            return True

        negated = is_not_test(outer.test) or is_not_test(inner.test)
        return not negated and in_ranges(self.version, JOINED_PLAIN_TESTS)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.FormattedValue:
        # a format spec is part of its field, never a string of its own
        node.value = self.visit(node.value)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        self.generic_visit(node)

        if not in_ranges(self.version, PERCENT_FORMAT):
            return node

        # measured: '%s' % x where the f-string needs !s (or !a), f'{x!r}' otherwise
        fields = [part for part in node.values if isinstance(part, ast.FormattedValue)]
        if not any(part.conversion in PERCENT_ONLY for part in fields) and not (
            self.spans_lines(fields)
        ):
            return node

        return percent_format(node) or node

    def spans_lines(self, fields: list[ast.FormattedValue]) -> bool:
        # a value on a later line than its statement was written after a %
        return self.line is not None and any(
            getattr(item, LINE_MARK, 0) > self.line
            for part in fields
            for item in ast.walk(part.value)
        )


def is_plain(statement: ast.stmt) -> bool:
    # the measured shape: straight-line statements, no block of their own
    return not any(
        hasattr(statement, field) for field in ("body", "cases", "handlers")
    ) and not leaves_early([statement])


def leaves_early(statements: list[ast.stmt]) -> bool:
    # a block with its own way out compiles differently in the two spellings
    pending: list[ast.AST] = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, TERMINAL_STATEMENTS):
            return True
        if not isinstance(node, SCOPES):
            pending.extend(ast.iter_child_nodes(node))

    return False


def is_not_test(test: ast.expr) -> bool:
    return isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)


def and_terms(test: ast.expr) -> list[ast.expr]:
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return list(test.values)

    return [test]
