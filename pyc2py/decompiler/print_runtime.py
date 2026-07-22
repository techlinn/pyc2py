import ast
from typing import TypeGuard

from pyc2py.astree import make_name
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.runtime import coerce_expr


class PrintRuntimeMixin(DecompilerContext):
    def flush_legacy_line_boundary(self) -> None:
        if self.pending_print_items or self.pending_print_target is not None:
            self.flush_print_items(newline=False)
        self.flush_yield_expressions()

    def print_item(self) -> None:
        self.pending_print_items.append(coerce_expr(self.pop_or_none()))

    def print_newline(self) -> None:
        self.flush_print_items(newline=True, force=True)

    def print_item_to(self) -> None:
        target = coerce_expr(self.pop_or_none())
        value = coerce_expr(self.pop_or_none())
        self.set_print_target(target)
        self.pending_print_items.append(value)

    def print_newline_to(self) -> None:
        target = coerce_expr(self.pop_or_none())
        self.set_print_target(target)
        self.flush_print_items(newline=True, force=True)

    def set_print_target(self, target: ast.expr) -> None:
        if self.pending_print_target is None:
            self.pending_print_target = target
            return
        if ast.dump(self.pending_print_target) == ast.dump(target):
            return
        self.flush_print_items(newline=True, force=False)
        self.pending_print_target = target

    def flush_print_items(self, newline: bool, force: bool = False) -> None:
        if (
            not force
            and not self.pending_print_items
            and self.pending_print_target is None
        ):
            return
        self.statements.append(
            make_print_statement(
                self.pending_print_items,
                self.pending_print_target,
                newline,
            )
        )
        self.pending_print_items = []
        self.pending_print_target = None


def make_print_statement(
    values: list[ast.expr],
    target: ast.expr | None,
    newline: bool,
) -> ast.Expr:
    keywords: list[ast.keyword] = []
    if target is not None:
        keywords.append(ast.keyword(arg="file", value=target))
    if not newline:
        keywords.append(ast.keyword(arg="end", value=ast.Constant(value=" ")))
    return ast.Expr(
        value=ast.Call(
            func=make_name("print", ast.Load()),
            args=list(values),
            keywords=keywords,
        )
    )


def render_legacy_print_source(source: str) -> str:
    lines = [render_legacy_source_line(line) for line in source.splitlines()]
    text = "\n".join(lines).rstrip()
    if not text:
        return ""
    return text + "\n"


def render_legacy_source_line(line: str) -> str:
    rendered = render_legacy_print_line(line)
    if rendered != line:
        return rendered
    rendered = render_legacy_exec_line(line)
    if rendered != line:
        return rendered
    return render_legacy_call_line(line)


def render_legacy_print_line(line: str) -> str:
    parsed = parse_single_source_statement(line)
    if parsed is None:
        return line
    indent, statement = parsed
    if not isinstance(statement, ast.Expr):
        return line
    call = statement.value
    if not isinstance(call, ast.Call):
        return line
    if not isinstance(call.func, ast.Name) or call.func.id != "print":
        return line
    legacy = make_legacy_print_text(call)
    if legacy is None:
        return line
    return f"{indent}{legacy}"


def render_legacy_call_line(line: str) -> str:
    parsed = parse_single_source_statement(line)
    if parsed is None:
        return line
    indent, statement = parsed
    if isinstance(statement, ast.Expr) and contains_legacy_call_shape(statement.value):
        return f"{indent}{render_legacy_expression(statement.value)}"
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and contains_legacy_call_shape(statement.value)
    ):
        target = ast.unparse(statement.targets[0])
        value = render_legacy_expression(statement.value)
        return f"{indent}{target} = {value}"
    if (
        isinstance(statement, ast.Return)
        and statement.value is not None
        and contains_legacy_call_shape(statement.value)
    ):
        return f"{indent}return {render_legacy_expression(statement.value)}"
    return line


def render_legacy_exec_line(line: str) -> str:
    parsed = parse_single_source_statement(line)
    if parsed is None:
        return line
    indent, statement = parsed
    if not isinstance(statement, ast.Expr):
        return line
    call = statement.value
    if not is_exec_call(call):
        return line
    legacy = make_legacy_exec_text(call)
    if legacy is None:
        return line
    return f"{indent}{legacy}"


def parse_single_source_statement(line: str) -> tuple[str, ast.stmt] | None:
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line[len(indent) :]
    try:
        parsed = ast.parse(stripped)
    except SyntaxError:
        return None
    if len(parsed.body) != 1:
        return None
    return indent, parsed.body[0]


def make_legacy_exec_text(call: ast.Call) -> str | None:
    if call.keywords or not 1 <= len(call.args) <= 3:
        return None

    source = render_legacy_expression(call.args[0])
    if len(call.args) == 1:
        return f"exec {source}"

    globals_value = render_legacy_expression(call.args[1])
    if len(call.args) == 2:
        return f"exec {source} in {globals_value}"

    locals_value = render_legacy_expression(call.args[2])
    return f"exec {source} in {globals_value}, {locals_value}"


def make_legacy_print_text(call: ast.Call) -> str | None:
    target = find_print_keyword(call, "file")
    end_value = find_print_keyword(call, "end")
    if has_unsupported_print_keyword(call):
        return None

    trailing_comma = is_space_end_value(end_value)
    values = [render_legacy_expression(argument) for argument in call.args]
    prefix = "print"
    if target is not None:
        prefix = f"print >> {render_legacy_expression(target)}"
    if values:
        separator = ", " if target is not None else " "
        prefix = f"{prefix}{separator}{', '.join(values)}"
    if trailing_comma:
        prefix = f"{prefix},"
    return prefix


def is_exec_call(value: ast.expr) -> TypeGuard[ast.Call]:
    if not isinstance(value, ast.Call):
        return False
    return isinstance(value.func, ast.Name) and value.func.id == "exec"


def render_legacy_expression(value: ast.expr) -> str:
    if is_backtick_call(value):
        return f"`{render_legacy_expression(value.args[0])}`"
    if is_legacy_long_call(value):
        return format_legacy_long(value.args[0])
    if is_negative_legacy_long(value):
        return format_legacy_long(value)
    if isinstance(value, ast.Call):
        return render_legacy_call(value)
    if isinstance(value, ast.Starred):
        return f"*{render_legacy_expression(value.value)}"
    return ast.unparse(value)


def render_legacy_call(call: ast.Call) -> str:
    arguments: list[str] = []
    starred: list[str] = []
    keywords: list[str] = []
    star_kwargs: list[str] = []

    for argument in call.args:
        if isinstance(argument, ast.Starred):
            starred.append(f"*{render_legacy_expression(argument.value)}")
        else:
            arguments.append(render_legacy_expression(argument))
    for keyword in call.keywords:
        if keyword.arg is None:
            star_kwargs.append(f"**{render_legacy_expression(keyword.value)}")
        else:
            keywords.append(f"{keyword.arg}={render_legacy_expression(keyword.value)}")

    all_arguments = [*arguments, *keywords, *starred, *star_kwargs]
    return f"{render_legacy_expression(call.func)}({', '.join(all_arguments)})"


def contains_legacy_call_shape(value: ast.AST) -> bool:
    for node in ast.walk(value):
        if isinstance(node, ast.Call) and (
            is_backtick_call(node) or is_legacy_long_call(node)
        ):
            return True
        if not isinstance(node, ast.Call):
            continue
        if any(isinstance(argument, ast.Starred) for argument in node.args) and any(
            keyword.arg is not None for keyword in node.keywords
        ):
            return True
    return False


def is_legacy_long_call(value: ast.expr) -> TypeGuard[ast.Call]:
    if not isinstance(value, ast.Call):
        return False
    if len(value.args) != 1 or value.keywords:
        return False
    return isinstance(value.func, ast.Name) and value.func.id == "__pyc2py_long__"


def is_negative_legacy_long(value: ast.expr) -> TypeGuard[ast.UnaryOp]:
    if not isinstance(value, ast.UnaryOp) or not isinstance(value.op, ast.USub):
        return False
    return is_legacy_long_call(value.operand)


def format_legacy_long(value: ast.expr) -> str:
    if is_negative_legacy_long(value):
        call = value.operand
        if isinstance(call, ast.Call) and isinstance(call.args[0], ast.Constant):
            argument = call.args[0].value
            if isinstance(argument, int):
                return format_legacy_integer(-argument)
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.USub)
        and isinstance(value.operand, ast.Constant)
        and isinstance(value.operand.value, int)
    ):
        return format_legacy_integer(-value.operand.value)
    if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
        return f"{ast.unparse(value)}L"
    return format_legacy_integer(value.value)


def format_legacy_integer(integer: int) -> str:
    magnitude = abs(integer)
    if magnitude >= 0x80000000:
        prefix = "-" if integer < 0 else ""
        return f"{prefix}{hex(magnitude)}L"
    return f"{integer}L"


def is_backtick_call(value: ast.expr) -> TypeGuard[ast.Call]:
    if not isinstance(value, ast.Call):
        return False
    if len(value.args) != 1 or value.keywords:
        return False
    return isinstance(value.func, ast.Name) and value.func.id == "__pyc2py_backtick__"


def find_print_keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def has_unsupported_print_keyword(call: ast.Call) -> bool:
    return any(keyword.arg not in {"file", "end"} for keyword in call.keywords)


def is_space_end_value(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value == " "
