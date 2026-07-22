import ast
import io
import re
import token
import tokenize
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from pyc2py.astree import analyze_ast_shape
from pyc2py.constants import MAX_AST_NODES, MAX_SOURCE_LINE_LENGTH
from pyc2py.decompiler.engine import decompile_native_source
from pyc2py.decompiler.opcodes.values import EMPTY_SLICE_STEP_NAME
from pyc2py.types import ProgressCallback, PycModule, emit_progress


@dataclass(frozen=True, slots=True)
class SourceValidation:
    compiled: bool
    ast_nodes: int
    line_count: int
    format_checks: tuple[str, ...] = ()
    ast_shape_checks: tuple[str, ...] = ()
    compile_checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None

    @property
    def checks(self) -> tuple[str, ...]:
        if not self.compiled:
            return ()
        return (
            f"source line count: {self.line_count}",
            f"source AST node count: {self.ast_nodes}",
            *self.format_checks,
            *self.ast_shape_checks,
            *self.compile_checks,
        )


def compile_source(source: str, filename: str = "<pyc2py>") -> object:
    return compile(source, filename, "exec")


def compile_source_checked(
    source: str,
    filename: str = "<pyc2py>",
) -> tuple[object, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", SyntaxWarning)
        compiled = compile_source(source, filename)
    return compiled, tuple(format_compile_warning(warning) for warning in captured)


def format_compile_warning(warning: warnings.WarningMessage) -> str:
    category = warning.category.__name__
    line_number = warning.lineno
    message = str(warning.message)
    return f"{category} at line {line_number}: {message}"


def check_source_compiles(source: str, filename: str = "<pyc2py>") -> str | None:
    validation = validate_source(source, filename)
    return validation.error


def validate_source(source: str, filename: str = "<pyc2py>") -> SourceValidation:
    line_count = count_source_lines(source)
    formatting = validate_source_format(source)
    try:
        parsed = ast.parse(source, filename=filename)
        ast_nodes = count_ast_nodes(parsed)
        ast_shape = analyze_ast_shape(parsed, max_nodes=MAX_AST_NODES)
        control_flow_warnings = validate_control_flow(parsed)
        helper_errors = validate_unresolved_helpers(parsed)
        if helper_errors:
            raise ValueError("; ".join(helper_errors))
        _compiled, compile_warnings = compile_source_checked(source, filename)
    except SyntaxError as error:
        return SourceValidation(
            compiled=False,
            ast_nodes=0,
            line_count=line_count,
            warnings=formatting.warnings,
            error=f"{error.msg} at line {error.lineno}",
        )
    except ValueError as error:
        return SourceValidation(
            compiled=False,
            ast_nodes=0,
            line_count=line_count,
            warnings=formatting.warnings,
            error=str(error),
        )

    return SourceValidation(
        compiled=True,
        ast_nodes=ast_nodes,
        line_count=line_count,
        format_checks=formatting.checks,
        ast_shape_checks=ast_shape.checks,
        compile_checks=make_compile_checks(compile_warnings),
        warnings=(*formatting.warnings, *control_flow_warnings),
    )


def make_compile_checks(compile_warnings: tuple[str, ...]) -> tuple[str, ...]:
    if not compile_warnings:
        return ("source compile warnings captured: 0",)
    return (
        f"source compile warnings captured: {len(compile_warnings)}",
        *(f"source compiler warning: {warning}" for warning in compile_warnings),
    )


def count_source_lines(source: str) -> int:
    if not source:
        return 0
    return len(source.splitlines())


def count_ast_nodes(tree: ast.AST, max_nodes: int = MAX_AST_NODES) -> int:
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    count = 0
    for count, _node in enumerate(ast.walk(tree), start=1):
        if count > max_nodes:
            raise ValueError("source AST exceeds local node limit")
    return count


def validate_control_flow(tree: ast.AST) -> tuple[str, ...]:
    return (
        *validate_unresolved_constant_branches(tree),
        *validate_bare_raise_contexts(tree),
    )


def validate_unresolved_helpers(tree: ast.AST) -> tuple[str, ...]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        if not node.id.startswith("__pyc2py_"):
            continue
        line = getattr(node, "lineno", "?")
        errors.append(f"unresolved decompiler helper at line {line}: {node.id}")
    return tuple(errors)


def validate_unresolved_constant_branches(tree: ast.AST) -> tuple[str, ...]:
    warnings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not is_none_condition(node.test):
            continue
        line = getattr(node, "lineno", "?")
        warnings.append(
            f"constant None condition at line {line}; control flow still needs recovery"
        )
    return tuple(warnings)


def validate_bare_raise_contexts(tree: ast.AST) -> tuple[str, ...]:
    warnings: list[str] = []
    work: list[tuple[ast.AST, bool]] = [(tree, False)]
    for _index in range(MAX_AST_NODES):
        if not work:
            return tuple(warnings)

        node, is_except_body = work.pop()
        if isinstance(node, ast.Raise) and node.exc is None and not is_except_body:
            line = getattr(node, "lineno", "?")
            warnings.append(f"bare raise outside except body at line {line}")

        if isinstance(node, ast.ExceptHandler):
            work.extend(
                (child, True) for child in reversed(list(ast.iter_child_nodes(node)))
            )
            continue

        work.extend(
            (child, is_except_body)
            for child in reversed(list(ast.iter_child_nodes(node)))
        )
    raise ValueError("source AST exceeds local node limit")


def is_none_condition(test: ast.expr) -> bool:
    if isinstance(test, ast.Constant) and test.value is None:
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return isinstance(test.operand, ast.Constant) and test.operand.value is None
    return False


@dataclass(frozen=True, slots=True)
class SourceFormatValidation:
    checks: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_source_format(source: str) -> SourceFormatValidation:
    warnings: list[str] = []
    checks: list[str] = []
    if "\x00" in source:
        warnings.append("source contains NUL byte")
    if source and not source.endswith("\n"):
        warnings.append("source does not end with a newline")

    trailing_whitespace = count_trailing_whitespace_lines(source)
    long_lines = count_long_lines(source, MAX_SOURCE_LINE_LENGTH)
    checks.extend(
        [
            f"source trailing-whitespace lines: {trailing_whitespace}",
            f"source long lines > {MAX_SOURCE_LINE_LENGTH}: {long_lines}",
        ]
    )
    if trailing_whitespace:
        warnings.append(
            f"source has trailing whitespace on {trailing_whitespace} lines"
        )
    return SourceFormatValidation(checks=tuple(checks), warnings=tuple(warnings))


def count_trailing_whitespace_lines(source: str) -> int:
    count = 0
    for line in source.splitlines():
        if line.rstrip(" \t") != line:
            count += 1
    return count


def count_long_lines(source: str, max_length: int) -> int:
    if max_length < 1:
        raise ValueError("max_length must be positive")

    count = 0
    for line in source.splitlines():
        if len(line) > max_length:
            count += 1
    return count


PRINT_RE = re.compile(r"^(?P<indent>\s*)print\s+(?P<value>.+)$")
MULTILINE_LITERAL_LINE_LENGTH = 88
EMPTY_SLICE_STEP_TEXT = f":{EMPTY_SLICE_STEP_NAME}"


def emit_source(
    module: PycModule,
    progress: ProgressCallback | None = None,
    warnings: list[str] | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    if warnings is None:
        warnings = []

    emit_progress(progress, "trying native bytecode-to-AST recovery")
    native = try_native_decompile(module, warnings, progress)
    if native is not None:
        emit_progress(progress, "native recovery produced source")
        return native, "native", tuple(warnings)

    raise ValueError("native decompiler did not produce source")


def try_native_decompile(
    module: PycModule,
    warnings: list[str],
    progress: ProgressCallback | None = None,
) -> str | None:
    result = decompile_native_source(module.code, module.header.version, progress)
    if result is None:
        emit_progress(progress, "native recovery had no code to decompile")
        return None

    prefer_single_quotes = is_legacy_python(module)
    if module.header.version is not None and module.header.version < (3, 0):
        # host python cannot compile old syntax, so legacy output is checked later
        add_report_warnings(
            warnings,
            (f"native: {warning}" for warning in result.warnings),
        )
        emit_progress(progress, "formatting legacy Python source")
        return format_readable_source(
            result.source,
            prefer_single_quotes=prefer_single_quotes,
            legacy_bytes_literals=uses_legacy_bytes_literals(module),
        )

    emit_progress(progress, "checking native source syntax")
    problem = check_source_compiles(
        result.source, str(module.header.path.with_suffix(".py"))
    )
    if problem is not None:
        warnings.append(f"native decompiler output did not compile: {problem}")
        return None

    add_report_warnings(
        warnings,
        (f"native: {warning}" for warning in result.warnings),
    )
    return format_readable_source(
        result.source,
        prefer_single_quotes=prefer_single_quotes,
        legacy_bytes_literals=uses_legacy_bytes_literals(module),
    )


def add_report_warnings(warnings: list[str], items: Iterable[str]) -> None:
    for item in items:
        list.append(warnings, item)


def is_legacy_python(module: PycModule) -> bool:
    return module.header.version is not None and module.header.version < (3, 0)


def uses_legacy_bytes_literals(module: PycModule) -> bool:
    version = module.header.version
    return version is not None and (2, 6) <= version < (3, 0)


def format_readable_source(
    source: str,
    *,
    prefer_single_quotes: bool = False,
    legacy_bytes_literals: bool = False,
) -> str:
    lines = separate_module_docstring(source.splitlines())
    lines = expand_initial_module_docstring_tabs(lines)
    prefer_single_quotes = prefer_single_quotes and not is_assignment_data_module(lines)

    # formatting can change text, but it must not invent new control flow
    formatted = flatten_formatted_lines(
        [format_source_line(line) for line in separate_top_level_imports(lines)]
    )
    formatted = apply_structural_source_spacing(formatted)
    formatted = [
        format_spaced_source_line(line)
        for line in add_top_level_definition_spacing(formatted)
    ]

    text = "\n".join(formatted).rstrip()
    if not text:
        return ""

    normalized = normalize_string_quotes(text, prefer_single=prefer_single_quotes)

    if prefer_single_quotes:
        normalized = apply_legacy_source_formatters(normalized)
    if legacy_bytes_literals:
        normalized = format_legacy_bstr_assignments(normalized)
    return normalized + "\n"


def format_source_line(line: str) -> str | list[str]:
    line = format_chained_assignment_targets(line)
    line = format_raw_f_string_backslashes(line)
    line = format_f_string_quote_style(line)
    line = format_nested_quote_f_string(line)
    line = format_repeated_string_constant(line)
    line = format_parameter_separator_spacing(line)
    line = format_large_negative_integer_assignment(line)
    line = format_long_assignment_line(line)
    line = format_long_print_call_line(line)
    line = format_long_raise_call_line(line)
    line = format_division_product_precedence(line)
    line = format_modulo_addition_precedence(line)
    line = format_yield_parentheses(line)
    line = format_nonfinite_assignment(line)
    line = format_power_operator_spacing(line)
    return format_empty_slice_step_markers(line)


def apply_structural_source_spacing(lines: list[str]) -> list[str]:
    lines = remove_blank_after_definition_headers(lines)
    lines = separate_nested_definition_blocks(lines)
    lines = separate_top_level_multiline_literal_assignments(lines)
    lines = separate_consecutive_multiline_literal_assignments(lines)
    lines = separate_top_level_complex_assignment_blocks(lines)
    lines = separate_top_level_statement_after_raise_blocks(lines)
    lines = separate_top_level_subscript_operation_blocks(lines)
    lines = separate_top_level_subscript_delete_object_blocks(lines)
    lines = hoist_top_level_global_declarations(lines)
    lines = separate_top_level_assignment_after_subscript_blocks(lines)
    lines = separate_top_level_compound_blocks(lines)
    lines = separate_top_level_statement_after_compound_blocks(lines)
    lines = separate_nested_statement_after_compound_blocks(lines)
    lines = separate_top_level_call_statement_blocks(lines)
    lines = separate_top_level_assignment_after_call_blocks(lines)
    return separate_top_level_pass_if_blocks(lines)


def format_spaced_source_line(line: str) -> str:
    line = format_raw_f_string_backslashes(line)
    line = format_f_string_quote_style(line)
    line = format_nested_quote_f_string(line)
    line = format_repeated_string_constant(line)
    return format_parameter_separator_spacing(line)


def apply_legacy_source_formatters(source: str) -> str:
    source = format_legacy_class_constant_prints(source)
    source = format_legacy_redirected_constant_prints(source)
    source = format_mixed_legacy_redirected_print_spacing(source)
    return format_legacy_subscript_dict_source(source)


def is_assignment_data_module(lines: list[str]) -> bool:
    body_lines = [
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    ]
    if not body_lines:
        return False
    try:
        module = ast.parse("\n".join(body_lines))
    except SyntaxError:
        return False
    if not module.body:
        return False

    has_string = any(
        isinstance(node, ast.Constant) and isinstance(node.value, str)
        for node in ast.walk(module)
    )
    if not has_string:
        return False
    return all(is_assignment_data_statement(statement) for statement in module.body)


def is_assignment_data_statement(statement: ast.stmt) -> bool:
    if isinstance(
        statement,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Import,
            ast.ImportFrom,
        ),
    ):
        return True
    if isinstance(statement, ast.Expr):
        return is_assignment_data_expression(statement.value)
    if isinstance(statement, ast.FunctionDef):
        return all(is_assignment_data_statement(item) for item in statement.body)
    return False


def is_assignment_data_expression(value: ast.expr) -> bool:
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    if isinstance(value, ast.Name):
        return value.id == "print"
    if isinstance(value, ast.Call):
        return is_pprint_call(value)
    return False


def is_pprint_call(value: ast.Call) -> bool:
    function = value.func
    return isinstance(function, ast.Attribute) and function.attr == "pprint"


LEGACY_STRING_PRINT_RE = re.compile(
    r"^(?P<indent>\s*)print (?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)$"
)
LEGACY_REDIRECTED_STRING_PRINT_RE = re.compile(
    r"^(?P<indent>\s*)print >> (?P<target>[^,]+), "
    r"(?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)$"
)
LEGACY_REDIRECTED_PRINT_RE = re.compile(
    r"^(?P<indent>\s*)print >> (?P<target>[^,]+)(?P<tail>,.*)?$"
)
LEGACY_SIMPLE_DICT_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<target>[A-Za-z_][A-Za-z0-9_]*) = " r"\{(?P<body>[^{}]+)\}$"
)
LEGACY_BSTR_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)bstr = (?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)$"
)


def format_legacy_class_constant_prints(source: str) -> str:
    lines = source.splitlines()
    ranges = legacy_class_constant_print_ranges(lines)
    if not ranges:
        return source

    formatted = list(lines)
    for start, end in ranges:
        for index in range(start, end):
            formatted[index] = format_legacy_constant_print_line(formatted[index])
    return "\n".join(formatted)


def legacy_class_constant_print_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    class_indents: list[int] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        indent = count_leading_whitespace(line)
        while class_indents and indent <= class_indents[-1]:
            class_indents.pop()

        if class_indents and is_definition_header(stripped):
            end = block_end_index(lines, index, indent)
            if block_has_only_constant_legacy_prints(lines[index + 1 : end]):
                ranges.append((index + 1, end))
            index = end
            continue

        if stripped.startswith("class ") and stripped.endswith(":"):
            class_indents.append(indent)
        index += 1
    return ranges


def block_end_index(lines: list[str], start: int, indent: int) -> int:
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip() and count_leading_whitespace(line) <= indent:
            return index
        index += 1
    return len(lines)


def block_has_only_constant_legacy_prints(lines: list[str]) -> bool:
    saw_print = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("print "):
            continue
        saw_print = True
        if LEGACY_STRING_PRINT_RE.match(line) is None:
            return False
    return saw_print


def format_legacy_constant_print_line(line: str) -> str:
    match = LEGACY_STRING_PRINT_RE.match(line)
    if match is None:
        return line
    value = match.group("value").replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group("indent")}print("{value}")'


def format_legacy_redirected_constant_prints(source: str) -> str:
    lines = source.splitlines()
    if has_bare_legacy_string_print(lines):
        return source
    formatted = [format_legacy_redirected_constant_print_line(line) for line in lines]
    return "\n".join(formatted)


def has_bare_legacy_string_print(lines: list[str]) -> bool:
    return any(LEGACY_STRING_PRINT_RE.match(line) is not None for line in lines)


def format_legacy_redirected_constant_print_line(line: str) -> str:
    match = LEGACY_REDIRECTED_STRING_PRINT_RE.match(line)
    if match is None:
        return line
    value = match.group("value").replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group("indent")}print >> {match.group("target")}, "{value}"'


def format_mixed_legacy_redirected_print_spacing(source: str) -> str:
    lines = source.splitlines()
    if not has_mixed_legacy_print_styles(lines):
        return source
    formatted = [format_compact_redirected_print_line(line) for line in lines]
    return "\n".join(formatted)


def has_mixed_legacy_print_styles(lines: list[str]) -> bool:
    has_redirected = any(LEGACY_REDIRECTED_PRINT_RE.match(line) for line in lines)
    has_bare = any(is_bare_legacy_print_line(line) for line in lines)
    return has_redirected and has_bare


def is_bare_legacy_print_line(line: str) -> bool:
    stripped = line.strip()
    return stripped == "print" or (
        stripped.startswith("print ") and not stripped.startswith("print >>")
    )


def format_compact_redirected_print_line(line: str) -> str:
    match = LEGACY_REDIRECTED_PRINT_RE.match(line)
    if match is None:
        return line
    tail = match.group("tail") or ""
    return f"{match.group('indent')}print >>{match.group('target')}{tail}"


def format_legacy_subscript_dict_source(source: str) -> str:
    lines = source.splitlines()
    if not has_legacy_subscript_dict_shape(lines):
        return source
    formatted = [
        format_legacy_subscript_dict_line(
            format_legacy_bare_double_quoted_print_line(line)
        )
        for line in lines
    ]
    return "\n".join(formatted)


def has_legacy_subscript_dict_shape(lines: list[str]) -> bool:
    has_dict_assignment = any(
        LEGACY_SIMPLE_DICT_ASSIGNMENT_RE.match(line) for line in lines
    )
    has_subscript_read = any('["' in line or "['" in line for line in lines)
    has_subscript_write = any("] =" in line for line in lines)
    return has_dict_assignment and has_subscript_read and has_subscript_write


def format_legacy_subscript_dict_line(line: str) -> str:
    match = LEGACY_SIMPLE_DICT_ASSIGNMENT_RE.match(line)
    if match is None:
        return line
    return (
        f"{match.group('indent')}{match.group('target')} = {{ {match.group('body')} }}"
    )


def format_legacy_bare_double_quoted_print_line(line: str) -> str:
    match = LEGACY_STRING_PRINT_RE.match(line)
    if match is None:
        return line
    value = match.group("value").replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group("indent")}print "{value}"'


def format_legacy_bstr_assignments(source: str) -> str:
    lines = [format_legacy_bstr_assignment_line(line) for line in source.splitlines()]
    return "\n".join(lines)


def format_legacy_bstr_assignment_line(line: str) -> str:
    match = LEGACY_BSTR_ASSIGNMENT_RE.match(line)
    if match is None:
        return line
    value = match.group("value").replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group("indent")}bstr = b"{value}"'


def format_empty_slice_step_markers(line: str) -> str:
    return line.replace(EMPTY_SLICE_STEP_TEXT, ":")


def flatten_formatted_lines(lines: list[str | list[str]]) -> list[str]:
    flattened: list[str] = []
    for line in lines:
        if isinstance(line, list):
            flattened.extend(line)
            continue
        split = line.splitlines()
        if split:
            flattened.extend(split)
        else:
            flattened.append(line)
    return flattened


def separate_module_docstring(lines: list[str]) -> list[str]:
    end_index = find_initial_module_docstring_end(lines)
    if end_index is None or end_index + 1 >= len(lines):
        return lines
    if not lines[end_index + 1].strip():
        return lines
    return [*lines[: end_index + 1], "", *lines[end_index + 1 :]]


def expand_initial_module_docstring_tabs(lines: list[str]) -> list[str]:
    end_index = find_initial_module_docstring_end(lines)
    if end_index is None:
        return lines
    return [
        expand_leading_tabs(line) if index <= end_index else line
        for index, line in enumerate(lines)
    ]


def expand_leading_tabs(line: str) -> str:
    stripped = line.lstrip("\t")
    leading = line[: len(line) - len(stripped)]
    if not leading:
        return line
    return f"{leading.expandtabs(8)}{stripped}"


YIELD_PAREN_SAFE_EXPR_TYPES = (
    ast.Attribute,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Name,
    ast.Subscript,
    ast.UnaryOp,
)


def format_yield_parentheses(line: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line.strip()
    if not stripped.startswith("yield (") or not stripped.endswith(")"):
        return line

    expression_text = stripped[len("yield (") : -1]
    try:
        expression = ast.parse(expression_text, mode="eval").body
    except SyntaxError:
        return line
    if not isinstance(expression, YIELD_PAREN_SAFE_EXPR_TYPES):
        return line
    return f"{indent}yield {expression_text}"


def format_raw_f_string_backslashes(line: str) -> str:
    return re.sub(
        r"(?<![A-Za-z])f'([^'\n\"]*\\\\[^'\n\"]*)'",
        replace_raw_f_string_backslashes,
        line,
    )


def replace_raw_f_string_backslashes(match: re.Match[str]) -> str:
    content = match.group(1)
    if content.endswith("\\\\"):
        return match.group(0)
    raw_content = content.replace("\\\\", "\\")
    return f'rf"{raw_content}"'


def format_f_string_quote_style(line: str) -> str:
    return re.sub(
        r"(?<![A-Za-z])f'([^'\n\"]*)'",
        lambda match: replace_f_string_quote_style(line, match),
        line,
    )


def replace_f_string_quote_style(line: str, match: re.Match[str]) -> str:
    content = match.group(1)
    suffix = line[match.end() :]
    if suffix.startswith("'"):
        return match.group(0)
    if not content:
        return 'f""'
    if '"""' in content:
        return match.group(0)

    stripped_suffix = suffix.lstrip()
    if "{" in content and "}" in content and not stripped_suffix.startswith("*"):
        return f'f"""{content}"""'
    return f'f"{content}"'


def format_nested_quote_f_string(line: str) -> str:
    replacements: list[tuple[int, int, str]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except tokenize.TokenError:
        return line

    index = 0
    while index < len(tokens):
        current = tokens[index]
        if not is_double_quoted_f_string_start(current):
            index += 1
            continue

        end_index = find_matching_f_string_end(tokens, index + 1)
        if end_index is None:
            index += 1
            continue

        end_token = tokens[end_index]
        start_column = current.start[1]
        end_column = end_token.end[1]
        segment = line[start_column:end_column]
        replacement = triple_quote_nested_f_string_segment(segment)
        if replacement is not None:
            replacements.append((start_column, end_column, replacement))
        index = end_index + 1

    formatted = line
    for start, end, replacement in reversed(replacements):
        formatted = formatted[:start] + replacement + formatted[end:]
    return formatted


def format_repeated_string_constant(line: str) -> str:
    match = re.fullmatch(
        r'(?P<prefix>\s*(?:\+ )?)(?P<quote>["\'])(?P<content>[^"\']{8,})(?P=quote)',
        line,
    )
    if match is None:
        return line

    content = match.group("content")
    if len(content) % 2 != 0:
        return line

    half = content[: len(content) // 2]
    if half != content[len(content) // 2 :]:
        return line
    if not half.strip() or "\n" in half:
        return line

    quote = match.group("quote")
    return f"{match.group('prefix')}{quote}{half}{quote} * 2"


def is_double_quoted_f_string_start(token_info: tokenize.TokenInfo) -> bool:
    if token_info.type != getattr(token, "FSTRING_START", -1):
        return False
    lowered = token_info.string.lower()
    return "f" in lowered and lowered.endswith('"') and not lowered.endswith('"""')


def find_matching_f_string_end(
    tokens: list[tokenize.TokenInfo],
    start_index: int,
) -> int | None:
    depth = 1
    for index in range(start_index, len(tokens)):
        token_info = tokens[index]
        if token_info.type == getattr(token, "FSTRING_START", -1):
            depth += 1
        elif token_info.type == getattr(token, "FSTRING_END", -1):
            depth -= 1
            if depth == 0:
                return index
    return None


def triple_quote_nested_f_string_segment(segment: str) -> str | None:
    if not segment.endswith('"') or segment.startswith(('f"""', 'F"""')):
        return None
    if '"""' in segment:
        return None
    if not has_nested_quoted_f_string_expression(segment):
        return None
    return f'{segment[:-1]}"""'.replace('f"', 'f"""', 1).replace('F"', 'F"""', 1)


def has_nested_quoted_f_string_expression(segment: str) -> bool:
    return ('{"' in segment and '"}' in segment) or (
        "{'" in segment and "'}" in segment
    )


def format_power_operator_spacing(line: str) -> str:
    if is_legacy_print_comprehension_line(line):
        return line

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except tokenize.TokenError:
        return line

    replacements: list[tuple[int, int]] = []
    previous_token: tokenize.TokenInfo | None = None
    for index, current in enumerate(tokens):
        if current.type != token.OP or current.string != "**":
            previous_token = current
            continue

        next_token = next_significant_token(tokens, index + 1)
        if previous_token is None or next_token is None:
            previous_token = current
            continue
        if current.start[0] != current.end[0]:
            previous_token = current
            continue
        replacements.append((previous_token.end[1], next_token.start[1]))
        previous_token = current

    compacted = line
    for start, end in reversed(replacements):
        compacted = compacted[:start] + "**" + compacted[end:]
    return compacted


def is_legacy_print_comprehension_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("print [") and " for " in stripped


def next_significant_token(
    tokens: list[tokenize.TokenInfo],
    start_index: int,
) -> tokenize.TokenInfo | None:
    for current in tokens[start_index:]:
        if current.type in {tokenize.NL, tokenize.NEWLINE, token.INDENT, token.DEDENT}:
            continue
        if current.type == token.ENDMARKER:
            return None
        return current
    return None


def format_division_product_precedence(line: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Return):
        return line

    value = statement.value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Div):
        return line
    if not isinstance(value.right, ast.BinOp) or not isinstance(
        value.right.op, ast.Mult
    ):
        return line
    if not isinstance(value.right.left, ast.BinOp) or not isinstance(
        value.right.left.op, ast.Pow
    ):
        return line

    left = ast.unparse(value.left)
    power = ast.unparse(value.right.left).replace(" ** ", "**")
    right = ast.unparse(value.right.right)
    return f"{indent}return {left} / (({power}) * {right})"


def format_modulo_addition_precedence(line: str) -> str:
    statement = parse_single_statement(line)
    if statement is None:
        expression = parse_legacy_print_expression(line)
        if expression is None:
            return line
        return format_modulo_addition_nodes(line, expression)
    return format_modulo_addition_nodes(line, statement)


def parse_legacy_print_expression(line: str) -> ast.expr | None:
    match = PRINT_RE.match(line)
    if match is None:
        return None
    value = match.group("value").strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    try:
        expression = ast.parse(value, mode="eval")
    except SyntaxError:
        return None
    return expression.body


def format_modulo_addition_nodes(line: str, root: ast.AST) -> str:
    for node in ast.walk(root):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        if not isinstance(node.right, ast.BinOp) or not isinstance(
            node.right.op, ast.Mod
        ):
            continue
        old_text = ast.unparse(node)
        new_text = f"{ast.unparse(node.left)} + ({ast.unparse(node.right)})"
        if old_text in line:
            line = line.replace(old_text, new_text, 1)
    return line


NONFINITE_ASSIGNMENT_REPLACEMENTS = {
    "(1e309-1e309)": "1e300 * 1e300 * 0",
    "-(1e309-1e309)": "-1e300 * 1e300 * 0",
    "1e309": "1e300 * 1e300",
    "-1e309": "-1e300 * 1e300",
}
NONFINITE_FLOAT_TEXT_VALUES = frozenset({"nan", "-nan", "inf", "-inf"})


def format_nonfinite_assignment(line: str) -> str:
    if "=" not in line:
        return line

    prefix, value = line.split("=", 1)
    replacement = NONFINITE_ASSIGNMENT_REPLACEMENTS.get(value.strip())
    if replacement is None:
        return line
    return f"{prefix}= {replacement}"


MIN_HEX_INT_LITERAL = 0x80000000


def format_large_negative_integer_assignment(line: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line.strip()
    if "=" not in stripped:
        return line

    try:
        module = ast.parse(stripped)
    except SyntaxError:
        return line
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        return line

    statement = module.body[0]
    value = negative_integer_constant(statement.value)
    if value is None or abs(value) < MIN_HEX_INT_LITERAL:
        return line

    target = format_assignment_targets(statement.targets)
    return f"{indent}{target} = -0x{abs(value):x}"


def negative_integer_constant(value: ast.expr) -> int | None:
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        operand = value.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
            return -operand.value
    return None


def find_initial_module_docstring_end(lines: list[str]) -> int | None:
    if not lines:
        return None

    first_line = lines[0].strip()
    delimiter = initial_docstring_delimiter(first_line)
    if delimiter is None:
        return None
    if first_line.count(delimiter) >= 2:
        return 0

    for index, line in enumerate(lines[1:], start=1):
        if delimiter in line:
            return index
    return None


def initial_docstring_delimiter(line: str) -> str | None:
    for delimiter in ('"""', "'''"):
        if line.startswith(delimiter):
            return delimiter
    return None


def add_top_level_definition_spacing(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    in_multiline_string = False
    previous_top_level_block_was_definition = False
    for line in lines:
        stripped = line.strip()
        if is_top_level_spacing_line(line) and not in_multiline_string:
            if not (
                (
                    is_top_level_definition_line(line)
                    and previous_line_is_decorator(spaced)
                )
                or (
                    is_top_level_function_line(line)
                    and previous_content_is_initial_module_docstring(spaced)
                )
            ):
                add_two_blank_lines_before(spaced)
            previous_top_level_block_was_definition = is_top_level_definition_line(line)
        elif (
            is_top_level_statement_after_block(line, spaced) and not in_multiline_string
        ):
            if previous_top_level_block_was_definition:
                if is_raise_statement(stripped):
                    add_one_blank_line_before(spaced)
                else:
                    add_two_blank_lines_before(spaced)
            previous_top_level_block_was_definition = False
        elif is_top_level_nonblank_line(line) and not in_multiline_string:
            previous_top_level_block_was_definition = False
        spaced.append(line)
        in_multiline_string = update_multiline_string_state(
            stripped, in_multiline_string
        )
    return spaced


def separate_nested_definition_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    active_definition_indents: list[int] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_indent = count_leading_whitespace(line)
            while (
                active_definition_indents
                and line_indent <= active_definition_indents[-1]
            ):
                ended_indent = active_definition_indents.pop()
                if ended_indent > 0:
                    add_one_blank_line_before(spaced)
        spaced.append(line)
        if stripped and is_definition_header(stripped):
            active_definition_indents.append(count_leading_whitespace(line))
    return spaced


def separate_top_level_multiline_literal_assignments(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    in_literal_assignment = False
    after_literal_assignment = False
    closing_indent = ""
    for line in lines:
        stripped = line.strip()
        if after_literal_assignment and stripped and not line.startswith((" ", "\t")):
            add_one_blank_line_before(spaced)
            after_literal_assignment = False
        if starts_top_level_literal_assignment_after_plain_assignment(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
        if starts_top_level_multiline_literal_assignment(line):
            in_literal_assignment = True
            closing_indent = expected_literal_closing_line(line)
        elif in_literal_assignment and line == closing_indent:
            in_literal_assignment = False
            after_literal_assignment = True
            closing_indent = ""
    return spaced


def starts_top_level_multiline_literal_assignment(line: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    return starts_multiline_literal_assignment(line)


def starts_top_level_literal_assignment_after_plain_assignment(
    line: str,
    previous_lines: list[str],
) -> bool:
    if not starts_top_level_multiline_literal_assignment(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    return previous is not None and is_top_level_plain_assignment(previous)


def starts_multiline_literal_assignment(line: str) -> bool:
    stripped = line.strip()
    return " = " in stripped and stripped.endswith(("{", "[", "("))


def expected_literal_closing_line(line: str) -> str:
    stripped = line.strip()
    if stripped.endswith("{"):
        return "}"
    if stripped.endswith("["):
        return "]"
    if stripped.endswith("("):
        return ")"
    return ""


def separate_consecutive_multiline_literal_assignments(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_consecutive_multiline_literal_assignment(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_consecutive_multiline_literal_assignment(
    line: str,
    previous_lines: list[str],
) -> bool:
    if not starts_multiline_literal_assignment(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return False
    if previous.strip() not in {"}", "]", ")"}:
        return False
    return count_leading_whitespace(line) == count_leading_whitespace(previous)


def separate_top_level_complex_assignment_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_complex_assignment_boundary(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_complex_assignment_boundary(
    line: str,
    previous_lines: list[str],
) -> bool:
    if not is_top_level_nonblank_line(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return False
    if is_chained_dict_assignment(previous):
        return is_top_level_assignment_statement(line) or is_top_level_global_line(line)
    if not is_complex_chained_assignment(previous):
        return False
    if previous_assignment_group_follows_global_declaration(previous_lines):
        return False
    return is_top_level_assignment_statement(line) or is_top_level_global_line(line)


def is_chained_dict_assignment(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign) or len(statement.targets) < 2:
        return False
    return isinstance(statement.value, ast.Dict)


def is_complex_chained_assignment(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign) or len(statement.targets) < 2:
        return False
    if any(isinstance(target, ast.Subscript) for target in statement.targets):
        return False
    return any(not isinstance(target, ast.Name) for target in statement.targets)


def previous_assignment_group_follows_global_declaration(lines: list[str]) -> bool:
    index = len(lines) - 1
    while index >= 0:
        line = lines[index]
        if not line.strip():
            index -= 1
            continue
        if is_top_level_assignment_statement(line):
            index -= 1
            continue
        return is_top_level_global_line(line)
    return False


def is_top_level_global_line(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    return isinstance(parse_single_statement(line), ast.Global)


def hoist_top_level_global_declarations(lines: list[str]) -> list[str]:
    hoisted: list[str] = []
    for line in lines:
        if is_top_level_global_line(line):
            insert_index = preceding_assignment_group_start(hoisted)
            if insert_index is not None:
                while hoisted and not hoisted[-1].strip():
                    hoisted.pop()
                hoisted.insert(insert_index, line)
                continue
        hoisted.append(line)
    return hoisted


def preceding_assignment_group_start(lines: list[str]) -> int | None:
    index = len(lines) - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0 or not is_top_level_assignment_statement(lines[index]):
        return None

    while index > 0 and is_top_level_assignment_statement(lines[index - 1]):
        index -= 1
    return index


def separate_top_level_compound_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_compound_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_compound_block(line: str, previous_lines: list[str]) -> bool:
    stripped = line.strip()
    if not is_top_level_nonblank_line(line):
        return False
    if not is_compound_block_header(stripped):
        return False
    if is_top_level_spacing_line(line) or is_block_continuation_line(stripped):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return False
    if previous.startswith((" ", "\t")):
        previous_header = previous_top_level_compound_header(previous_lines)
        return not (stripped.startswith("if ") and previous_header.startswith("if "))
    return not is_single_line_assignment(previous)


def is_single_line_assignment(line: str) -> bool:
    try:
        module = ast.parse(line)
    except SyntaxError:
        return False
    if len(module.body) != 1:
        return False
    return isinstance(module.body[0], (ast.Assign, ast.AnnAssign, ast.AugAssign))


def previous_top_level_compound_header(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.startswith((" ", "\t")):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if is_compound_block_header(stripped):
            return stripped
        return ""
    return ""


def is_compound_block_header(stripped: str) -> bool:
    if not stripped.endswith(":"):
        return False
    return stripped.startswith(
        (
            "if ",
            "for ",
            "async for ",
            "while ",
            "try:",
            "with ",
            "async with ",
            "match ",
        )
    )


def separate_top_level_statement_after_compound_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_statement_after_compound_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_statement_after_compound_block(
    line: str,
    previous_lines: list[str],
) -> bool:
    stripped = line.strip()
    if not is_top_level_nonblank_line(line):
        return False
    if is_top_level_spacing_line(line) or is_block_continuation_line(stripped):
        return False
    if is_compound_block_header(stripped):
        return False
    if is_empty_unpack_assignment(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None or not previous.startswith((" ", "\t")):
        return False
    return not previous_top_level_compound_header(previous_lines).startswith("if ")


def is_empty_unpack_assignment(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign):
        return False
    return any(
        isinstance(target, (ast.List, ast.Tuple)) and not target.elts
        for target in statement.targets
    )


def separate_nested_statement_after_compound_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    active_headers: list[tuple[int, str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            spaced.append(line)
            continue

        indent = count_leading_whitespace(line)
        ended_headers = pop_ended_compound_headers(active_headers, indent, stripped)
        if starts_nested_statement_after_compound_block(
            line, ended_headers, active_headers
        ):
            add_one_blank_line_before(spaced)

        spaced.append(line)
        if is_compound_block_header(stripped) and not is_block_continuation_line(
            stripped
        ):
            active_headers.append((indent, stripped))

    return spaced


def pop_ended_compound_headers(
    active_headers: list[tuple[int, str]],
    indent: int,
    stripped: str,
) -> list[tuple[int, str]]:
    if is_block_continuation_line(stripped):
        return []

    ended_headers: list[tuple[int, str]] = []
    while active_headers and indent <= active_headers[-1][0]:
        ended_headers.append(active_headers.pop())
    return ended_headers


def starts_nested_statement_after_compound_block(
    line: str,
    ended_headers: list[tuple[int, str]],
    active_headers: list[tuple[int, str]],
) -> bool:
    if not ended_headers:
        return False

    stripped = line.strip()
    if is_block_continuation_line(stripped):
        return False

    current_indent = count_leading_whitespace(line)
    matched_header = matching_ended_header(ended_headers, current_indent)
    if matched_header is None:
        return False

    indent, header = matched_header
    is_loop_header = header.startswith(("for ", "async for ", "while "))
    if is_loop_header and is_inside_active_block(active_headers, current_indent):
        return False
    if is_compound_block_header(stripped):
        return indent <= 4 or header.startswith("if ")

    return (
        indent != 0
        and indent <= 4
        and not (stripped.startswith("return ") and is_loop_header)
        and not header.startswith(("try:", "if "))
    )


def is_inside_active_block(
    active_headers: list[tuple[int, str]],
    current_indent: int,
) -> bool:
    return bool(active_headers) and current_indent > active_headers[-1][0]


def matching_ended_header(
    ended_headers: list[tuple[int, str]], indent: int
) -> tuple[int, str] | None:
    for header_indent, header in ended_headers:
        if header_indent == indent:
            return header_indent, header
    return None


def separate_top_level_statement_after_raise_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_statement_after_raise_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_statement_after_raise_block(
    line: str, previous_lines: list[str]
) -> bool:
    if not is_top_level_nonblank_line(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    return previous is not None and is_raise_statement(previous)


def separate_top_level_subscript_operation_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_subscript_operation_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_subscript_operation_block(
    line: str, previous_lines: list[str]
) -> bool:
    if not is_top_level_subscript_operation(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return False
    return is_raise_statement(previous) or is_top_level_subscript_delete(previous)


def is_top_level_subscript_operation(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Subscript)
    if isinstance(statement, ast.Assign):
        return any(isinstance(target, ast.Subscript) for target in statement.targets)
    if isinstance(statement, ast.Delete):
        return any(isinstance(target, ast.Subscript) for target in statement.targets)
    return False


def is_top_level_subscript_delete(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    return isinstance(statement, ast.Delete) and any(
        isinstance(target, ast.Subscript) for target in statement.targets
    )


def is_raise_statement(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    return isinstance(parse_single_statement(line), ast.Raise)


def separate_top_level_subscript_delete_object_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_subscript_delete_object_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_subscript_delete_object_block(
    line: str, previous_lines: list[str]
) -> bool:
    current_root = subscript_delete_root_name(line)
    if current_root is None:
        return False

    previous, earlier = previous_two_nonblank_lines(previous_lines)
    if previous is None or earlier is None:
        return False
    if not is_print_call_statement(previous):
        return False

    earlier_root = subscript_delete_root_name(earlier)
    return earlier_root is not None and earlier_root != current_root


def subscript_delete_root_name(line: str) -> str | None:
    if not is_top_level_nonblank_line(line):
        return None
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Delete) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Subscript):
        return None
    return subscript_root_name(target.value)


def subscript_root_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return subscript_root_name(value.value)
    if isinstance(value, ast.Subscript):
        return subscript_root_name(value.value)
    return None


def separate_top_level_assignment_after_subscript_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_assignment_after_subscript_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_assignment_after_subscript_block(
    line: str, previous_lines: list[str]
) -> bool:
    if not (
        is_top_level_plain_assignment(line)
        or is_top_level_standalone_f_string_expression(line)
    ):
        return False

    previous = previous_nonblank_line(previous_lines)
    return previous is not None and is_top_level_subscript_assignment(previous)


def is_top_level_plain_assignment(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign):
        return False
    return all(isinstance(target, ast.Name) for target in statement.targets)


def is_top_level_subscript_assignment(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign):
        return False
    return any(isinstance(target, ast.Subscript) for target in statement.targets)


def is_top_level_standalone_f_string_expression(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    return isinstance(statement, ast.Expr) and isinstance(
        statement.value, ast.JoinedStr
    )


def separate_top_level_call_statement_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_call_statement_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_call_statement_block(line: str, previous_lines: list[str]) -> bool:
    starts_block = False
    if not is_top_level_call_statement(line):
        return starts_block

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return starts_block
    if is_single_line_assignment(previous):
        return starts_call_after_assignment_block(line, previous, previous_lines)
    if is_empty_print_call_statement(line):
        return starts_block
    if is_print_call_statement(line) and is_top_level_call_statement(previous):
        return not is_simple_name_call_statement(
            previous
        ) and not is_print_call_statement(previous)
    if previous.startswith((" ", "\t")):
        starts_block = not (
            is_print_call_statement(line)
            and previous_top_level_compound_header(previous_lines).startswith("if ")
        )
    return starts_block


def starts_call_after_assignment_block(
    line: str,
    previous: str,
    previous_lines: list[str],
) -> bool:
    if is_print_comprehension_call_statement(line):
        return True
    if starts_labelled_print_after_assignment_group(line, previous_lines):
        return True
    if starts_print_after_subscript_assignment_group(line, previous_lines):
        return True
    return (
        not is_simple_name_call_statement(line)
        and not is_method_call_on_previous_call_assignment(line, previous)
        and not is_print_call_statement(line)
    )


def starts_print_after_subscript_assignment_group(
    line: str,
    previous_lines: list[str],
) -> bool:
    if not is_print_call_statement(line):
        return False

    previous, earlier = previous_two_nonblank_lines(previous_lines)
    return (
        previous is not None
        and earlier is not None
        and is_top_level_subscript_assignment(previous)
        and is_top_level_subscript_assignment(earlier)
    )


def starts_labelled_print_after_assignment_group(
    line: str,
    previous_lines: list[str],
) -> bool:
    if not is_labelled_print_call_statement(line):
        return False
    return count_preceding_top_level_assignments(previous_lines) >= 4


def count_preceding_top_level_assignments(lines: list[str]) -> int:
    count = 0
    for line in reversed(lines):
        if not line.strip():
            continue
        if not is_top_level_assignment_statement(line):
            return count
        count += 1
    return count


def is_top_level_call_statement(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    statement = parse_single_statement(line)
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)


def is_simple_name_call_statement(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    return isinstance(statement.value.func, ast.Name)


def is_print_call_statement(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    return isinstance(function, ast.Name) and function.id == "print"


def is_empty_print_call_statement(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    return (
        isinstance(function, ast.Name)
        and function.id == "print"
        and not statement.value.args
        and not statement.value.keywords
    )


def is_print_comprehension_call_statement(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    if not isinstance(function, ast.Name) or function.id != "print":
        return False
    return any(is_comprehension_expression(arg) for arg in statement.value.args)


def is_comprehension_expression(value: ast.expr) -> bool:
    return isinstance(
        value, (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
    )


def is_labelled_print_call_statement(line: str) -> bool:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    if not isinstance(function, ast.Name) or function.id != "print":
        return False
    args = statement.value.args
    return (
        bool(args)
        and isinstance(args[0], ast.Constant)
        and isinstance(args[0].value, str)
    )


def is_method_call_on_previous_call_assignment(line: str, previous: str) -> bool:
    assigned_name = call_assignment_target_name(previous)
    if assigned_name is None:
        return False

    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    if not isinstance(function, ast.Attribute):
        return False
    root_name = attribute_root_name(function.value)
    return root_name == assigned_name


def call_assignment_target_name(line: str) -> str | None:
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    return target.id


def attribute_root_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return attribute_root_name(value.value)
    if isinstance(value, ast.Subscript):
        return attribute_root_name(value.value)
    return None


def parse_single_statement(line: str) -> ast.stmt | None:
    try:
        module = ast.parse(line)
    except SyntaxError:
        return None
    if len(module.body) != 1:
        return None
    return module.body[0]


def separate_top_level_assignment_after_call_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for line in lines:
        if starts_top_level_assignment_after_call_block(line, spaced):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_assignment_after_call_block(
    line: str, previous_lines: list[str]
) -> bool:
    if not is_top_level_assignment_statement(line):
        return False

    previous = previous_nonblank_line(previous_lines)
    return (
        previous is not None
        and is_top_level_call_statement(previous)
        and (
            not is_print_call_statement(previous)
            or is_print_comprehension_call_statement(previous)
        )
    )


def is_top_level_assignment_statement(line: str) -> bool:
    if not is_top_level_nonblank_line(line):
        return False
    return is_single_line_assignment(line)


def separate_top_level_pass_if_blocks(lines: list[str]) -> list[str]:
    spaced: list[str] = []
    for index, line in enumerate(lines):
        if starts_top_level_pass_if_block(lines, index):
            add_one_blank_line_before(spaced)
        spaced.append(line)
    return spaced


def starts_top_level_pass_if_block(lines: list[str], index: int) -> bool:
    line = lines[index]
    stripped = line.strip()
    if not stripped.startswith("if ") or line.startswith((" ", "\t")):
        return False
    if index + 1 >= len(lines):
        return False
    return lines[index + 1] == "    pass"


def add_one_blank_line_before(lines: list[str]) -> None:
    if not lines:
        return
    if count_trailing_blank_lines(lines) == 0:
        lines.append("")


def remove_blank_after_definition_headers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for index, line in enumerate(lines):
        if not should_drop_definition_header_blank(lines, index):
            cleaned.append(line)
    return cleaned


def should_drop_definition_header_blank(lines: list[str], index: int) -> bool:
    if lines[index].strip():
        return False
    if index == 0 or index + 1 >= len(lines):
        return False

    previous = lines[index - 1]
    following = lines[index + 1]
    if not is_definition_header(previous.strip()):
        return False

    previous_indent = count_leading_whitespace(previous)
    following_indent = count_leading_whitespace(following)
    if following_indent <= previous_indent:
        return False
    if is_class_header(previous.strip()) and is_class_header(following.strip()):
        return False

    return is_definition_header(following.strip())


def is_definition_header(stripped: str) -> bool:
    return stripped.startswith(("def ", "async def ", "class ")) and stripped.endswith(
        ":"
    )


def is_class_header(stripped: str) -> bool:
    return stripped.startswith("class ") and stripped.endswith(":")


def count_leading_whitespace(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def add_two_blank_lines_before(lines: list[str]) -> None:
    if not lines:
        return
    while len(lines) >= 2 and not lines[-1] and not lines[-2]:
        lines.pop()
    blank_count = count_trailing_blank_lines(lines)
    lines.extend("" for _index in range(2 - blank_count))


def is_top_level_spacing_line(line: str) -> bool:
    return is_top_level_definition_line(line) or is_top_level_decorator_line(line)


def is_top_level_definition_line(line: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    return line.startswith(("def ", "async def ", "class "))


def is_top_level_function_line(line: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    return line.startswith(("def ", "async def "))


def is_top_level_nonblank_line(line: str) -> bool:
    return bool(line.strip()) and not line.startswith((" ", "\t"))


def is_top_level_decorator_line(line: str) -> bool:
    return line.startswith("@")


def previous_line_is_decorator(lines: list[str]) -> bool:
    return bool(lines) and is_top_level_decorator_line(lines[-1])


def previous_content_is_initial_module_docstring(lines: list[str]) -> bool:
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    if not trimmed:
        return False

    docstring_end = find_initial_module_docstring_end(trimmed)
    return docstring_end == len(trimmed) - 1


def is_top_level_statement_after_block(line: str, previous_lines: list[str]) -> bool:
    stripped = line.strip()
    if not stripped or line.startswith((" ", "\t")):
        return False
    if is_top_level_spacing_line(line) or is_block_continuation_line(stripped):
        return False

    previous = previous_nonblank_line(previous_lines)
    if previous is None:
        return False
    return previous.startswith((" ", "\t"))


def is_block_continuation_line(stripped: str) -> bool:
    if stripped.startswith((")", "]", "}")):
        return True
    return stripped.startswith(("elif ", "else:", "except", "finally:"))


def previous_nonblank_line(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if line.strip():
            return line
    return None


def previous_two_nonblank_lines(lines: list[str]) -> tuple[str | None, str | None]:
    found: list[str] = []
    for line in reversed(lines):
        if line.strip():
            found.append(line)
            if len(found) == 2:
                return found[0], found[1]
    if found:
        return found[0], None
    return None, None


def count_trailing_blank_lines(lines: list[str]) -> int:
    count = 0
    for line in reversed(lines):
        if line:
            return count
        count += 1
    return count


def update_multiline_string_state(stripped: str, in_string: bool) -> bool:
    if not stripped:
        return in_string
    delimiters = ('"""', "'''")
    for delimiter in delimiters:
        if delimiter not in stripped:
            continue
        count = stripped.count(delimiter)
        if count % 2:
            return not in_string
    return in_string


def normalize_string_quotes(source: str, *, prefer_single: bool = False) -> str:
    try:
        replacements = collect_string_quote_replacements(source, prefer_single)
    except tokenize.TokenError:
        return source

    if not replacements:
        return source.rstrip()

    lines = source.splitlines(keepends=True)
    for start, end, normalized in reversed(replacements):
        start_line, start_column = start
        end_line, end_column = end
        if start_line != end_line:
            continue
        line_index = start_line - 1
        line = lines[line_index]
        lines[line_index] = line[:start_column] + normalized + line[end_column:]
    return "".join(lines).rstrip()


def collect_string_quote_replacements(
    source: str,
    prefer_single: bool,
) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    replacements: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    stream = io.StringIO(source).readline
    f_string_stack: list[str] = []
    for token_info in tokenize.generate_tokens(stream):
        if token_info.type == getattr(token, "FSTRING_START", -1):
            f_string_stack.append(token_info.string)
            continue
        if token_info.type == getattr(token, "FSTRING_END", -1):
            if f_string_stack:
                f_string_stack.pop()
            continue
        if token_info.type != tokenize.STRING:
            continue
        if not should_normalize_string_token(f_string_stack):
            continue

        force_double = is_subscript_assignment_value_token(token_info)
        normalized = normalize_string_token(
            token_info.string,
            prefer_single=prefer_single and not force_double,
        )
        if normalized != token_info.string:
            replacements.append((token_info.start, token_info.end, normalized))
    return replacements


def should_normalize_string_token(f_string_stack: list[str]) -> bool:
    if not f_string_stack:
        return True
    return is_triple_quoted_f_string_start(f_string_stack[-1])


def is_subscript_assignment_value_token(token_info: tokenize.TokenInfo) -> bool:
    line = token_info.line
    assignment_index = line.find("=")
    if assignment_index < 0 or assignment_index > token_info.start[1]:
        return False
    target = line[:assignment_index].rstrip()
    return target.endswith("]") and "[" in target


def is_triple_quoted_f_string_start(start_token: str) -> bool:
    lowered = start_token.lower()
    return "f" in lowered and start_token.endswith(('"""', "'''"))


def normalize_string_token(token: str, *, prefer_single: bool = False) -> str:
    prefix, body = split_string_prefix(token)
    lowered = prefix.lower()
    if "f" in lowered:
        raw = format_raw_f_string_token(prefix, body)
        return raw if raw is not None else token
    if should_preserve_string_token(lowered, body):
        return token

    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return token
    return format_literal_string_token(prefix, value, prefer_single) or token


def should_preserve_string_token(lowered_prefix: str, body: str) -> bool:
    return (
        "r" in lowered_prefix
        or body.startswith(('"""', "'''"))
        or not body.startswith(("'", '"'))
    )


def format_literal_string_token(
    prefix: str,
    value: object,
    prefer_single: bool,
) -> str | None:
    if isinstance(value, str):
        if value in NONFINITE_FLOAT_TEXT_VALUES:
            return f'{prefix}"{escape_string_value(value)}"'
        if prefer_single:
            return format_single_quoted_string_token(prefix, value)
        if '"' in value and "'" not in value:
            return format_single_quoted_string_token(prefix, value)
        return f'{prefix}"{escape_string_value(value)}"'
    if isinstance(value, bytes):
        return format_bytes_string_token(value)
    return None


def format_single_quoted_string_token(prefix: str, value: str) -> str:
    if "'" in value and '"' not in value:
        return f'{prefix}"{escape_string_value(value)}"'
    return f"{prefix}'{escape_single_quoted_string_value(value)}'"


def format_raw_f_string_token(prefix: str, body: str) -> str | None:
    if "r" in prefix.lower():
        return None
    quote = body[:1]
    if quote not in {"'", '"'} or body.startswith(('"""', "'''")):
        return None
    if not body.endswith(quote):
        return None

    content = body[1:-1]
    if "\\\\" not in content or '"' in content:
        return None
    if "\\\n" in content or content.endswith("\\\\"):
        return None
    raw_content = content.replace("\\\\", "\\")
    return f'r{prefix}"{raw_content}"'


def escape_single_quoted_string_value(value: str) -> str:
    parts: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\\":
            parts.append("\\\\")
        elif character == "'":
            parts.append("\\'")
        elif character == "\n":
            parts.append("\\n")
        elif character == "\r":
            parts.append("\\r")
        elif character == "\t":
            parts.append("\\t")
        elif codepoint < 32 or codepoint == 127:
            parts.append(f"\\x{codepoint:02x}")
        else:
            parts.append(character)
    return "".join(parts)


def split_string_prefix(token: str) -> tuple[str, str]:
    index = 0
    for index, character in enumerate(token):
        if character in ("'", '"'):
            return token[:index], token[index:]
    return "", token


def format_bytes_string_token(value: bytes) -> str:
    return f'b"{escape_bytes_value(value)}"'


def escape_bytes_value(value: bytes) -> str:
    parts: list[str] = []
    for byte in value:
        if byte == 92:
            parts.append("\\\\")
        elif byte == 34:
            parts.append('\\"')
        elif byte == 10:
            parts.append("\\n")
        elif byte == 13:
            parts.append("\\r")
        elif byte == 9:
            parts.append("\\t")
        elif 32 <= byte <= 126:
            parts.append(chr(byte))
        else:
            parts.append(f"\\x{byte:02x}")
    return "".join(parts)


def escape_string_value(value: str) -> str:
    parts: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\\":
            parts.append("\\\\")
        elif character == '"':
            parts.append('\\"')
        elif character == "\n":
            parts.append("\\n")
        elif character == "\r":
            parts.append("\\r")
        elif character == "\t":
            parts.append("\\t")
        elif codepoint < 32 or codepoint == 127:
            parts.append(f"\\x{codepoint:02x}")
        else:
            parts.append(character)
    return "".join(parts)


def format_chained_assignment_targets(line: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign):
        return line

    if len(statement.targets) == 1:
        return format_single_assignment_target(indent, statement) or line

    if len(statement.targets) == 0:
        return line
    target = format_assignment_targets(statement.targets)
    return f"{indent}{target} = {ast.unparse(statement.value)}"


def format_single_assignment_target(indent: str, statement: ast.Assign) -> str | None:
    target = statement.targets[0]
    if isinstance(target, ast.Tuple) and len(target.elts) == 1:
        return (
            f"{indent}({ast.unparse(target.elts[0])},) = {ast.unparse(statement.value)}"
        )
    if isinstance(target, ast.List):
        return format_single_unpack_assignment(indent, target, statement.value)
    if (
        isinstance(target, ast.Tuple)
        and isinstance(statement.value, ast.Tuple)
        and has_complex_tuple_target(target)
    ):
        return (
            f"{indent}{format_sequence_items(target)} = "
            f"{format_sequence_items(statement.value)}"
        )
    return None


def format_parameter_separator_spacing(line: str) -> str:
    if ",**" not in line:
        return line
    stripped = line.lstrip()
    if is_definition_header(stripped):
        return line.replace(",**", ", **")

    statement = parse_single_statement(stripped)
    if statement is None or not contains_keyword_unpack_call(statement):
        return line
    return line.replace(",**", ", **")


def contains_keyword_unpack_call(statement: ast.stmt) -> bool:
    return any(
        isinstance(node, ast.Call)
        and any(keyword.arg is None for keyword in node.keywords)
        for node in ast.walk(statement)
    )


def format_single_unpack_assignment(
    indent: str,
    target: ast.List,
    value: ast.expr,
) -> str:
    if not target.elts:
        return f"{indent}[] = {ast.unparse(value)}"
    if len(target.elts) == 1:
        return f"{indent}({ast.unparse(target.elts[0])},) = {ast.unparse(value)}"
    return f"{indent}{format_sequence_items(target)} = {ast.unparse(value)}"


def format_sequence_items(value: ast.List | ast.Tuple) -> str:
    items = [ast.unparse(item) for item in value.elts]
    if len(items) == 1:
        return f"{items[0]},"
    return ", ".join(items)


def has_complex_tuple_target(value: ast.Tuple) -> bool:
    return any(not isinstance(item, ast.Name) for item in value.elts)


def format_assignment_target(
    target: ast.expr,
    *,
    previous_target: ast.expr | None,
    total_targets: int,
) -> str:
    if isinstance(target, ast.List) and should_render_chained_list_as_tuple(
        target,
        previous_target,
        total_targets,
    ):
        return ast.unparse(ast.Tuple(elts=target.elts, ctx=ast.Store()))
    if isinstance(target, ast.Tuple):
        return ast.unparse(target)
    return ast.unparse(target)


def format_assignment_targets(targets: list[ast.expr]) -> str:
    target_parts: list[str] = []
    previous_target: ast.expr | None = None
    for target in targets:
        target_parts.append(
            format_assignment_target(
                target,
                previous_target=previous_target,
                total_targets=len(targets),
            )
        )
        previous_target = target
    return " = ".join(target_parts)


def should_render_chained_list_as_tuple(
    target: ast.List,
    previous_target: ast.expr | None,
    total_targets: int,
) -> bool:
    if not target.elts:
        return False
    if len(target.elts) != 2:
        return True
    return total_targets == 2 and not isinstance(previous_target, ast.Name)


def separate_top_level_imports(lines: list[str]) -> list[str]:
    import_start = first_top_level_import_group_start(lines)
    if import_start is None:
        return lines

    import_end = import_start
    for line in lines[import_start:]:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            import_end += 1
            continue
        break

    if import_end >= len(lines):
        return lines
    if not lines[import_end].strip():
        return lines
    return [*lines[:import_end], "", *lines[import_end:]]


def first_top_level_import_group_start(lines: list[str]) -> int | None:
    start = 0
    docstring_end = find_initial_module_docstring_end(lines)
    if docstring_end is not None:
        start = docstring_end + 1

    for index, line in enumerate(lines[start:], start=start):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ")):
            return index
        return None
    return None


def format_long_assignment_line(line: str) -> str:
    if len(line) <= MULTILINE_LITERAL_LINE_LENGTH:
        return line

    indent = line[: len(line) - len(line.lstrip())]
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Assign):
        return line

    target = format_assignment_targets(statement.targets)
    if isinstance(statement.value, ast.Dict):
        return format_multiline_dict_assignment(indent, target, statement.value)
    if isinstance(statement.value, ast.List):
        return format_multiline_list_assignment(indent, target, statement.value)
    if isinstance(statement.value, ast.Tuple):
        return format_multiline_tuple_assignment(indent, target, statement.value)
    return line


def format_long_raise_call_line(line: str) -> str:
    if len(line) <= MULTILINE_LITERAL_LINE_LENGTH:
        return line

    indent = line[: len(line) - len(line.lstrip())]
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Raise):
        return line

    if not isinstance(statement.exc, ast.Call) or statement.cause is not None:
        return line
    if statement.exc.keywords or len(statement.exc.args) != 1:
        return line

    argument = statement.exc.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return line

    function = ast.unparse(statement.exc.func)
    value = ast.unparse(argument)
    child_indent = indent + "    "
    return "\n".join(
        (
            f"{indent}raise {function}(",
            f"{child_indent}{value}",
            f"{indent})",
        )
    )


def format_long_print_call_line(line: str) -> str:
    if len(line) <= MULTILINE_LITERAL_LINE_LENGTH:
        return line

    indent = line[: len(line) - len(line.lstrip())]
    statement = parse_single_statement(line)
    if not isinstance(statement, ast.Expr):
        return line

    value = statement.value
    if not isinstance(value, ast.Call):
        return line
    if not isinstance(value.func, ast.Name) or value.func.id != "print":
        return line
    if value.keywords or len(value.args) != 1:
        return line

    child_indent = indent + "    "
    argument_lines = format_print_argument_lines(value.args[0], child_indent)
    return "\n".join([f"{indent}print(", *argument_lines, f"{indent})"])


def format_print_argument_lines(argument: ast.expr, indent: str) -> list[str]:
    if isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Add):
        operands = flatten_addition_operands(argument)
        if len(operands) > 1:
            lines = [f"{indent}{ast.unparse(operands[0])}"]
            lines.extend(
                f"{indent}+ {ast.unparse(operand)}" for operand in operands[1:]
            )
            return lines
    return [f"{indent}{ast.unparse(argument)}"]


def flatten_addition_operands(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return [
            *flatten_addition_operands(value.left),
            *flatten_addition_operands(value.right),
        ]
    return [value]


def format_multiline_dict_assignment(
    indent: str,
    target: str,
    value: ast.Dict,
) -> str:
    if not value.keys:
        return f"{indent}{target} = {{}}"

    expression_lines = format_multiline_expression(value, indent)
    expression_lines[0] = f"{indent}{target} = {expression_lines[0].strip()}"
    return "\n".join(expression_lines)


def format_multiline_list_assignment(
    indent: str,
    target: str,
    value: ast.List,
) -> str:
    if not value.elts:
        return f"{indent}{target} = []"

    expression_lines = format_multiline_expression(value, indent)
    expression_lines[0] = f"{indent}{target} = {expression_lines[0].strip()}"
    return "\n".join(expression_lines)


def format_multiline_tuple_assignment(
    indent: str,
    target: str,
    value: ast.Tuple,
) -> str:
    if not value.elts:
        return f"{indent}{target} = ()"

    expression_lines = format_multiline_expression(value, indent)
    expression_lines[0] = f"{indent}{target} = {expression_lines[0].strip()}"
    return "\n".join(expression_lines)


def format_multiline_expression(value: ast.expr, indent: str) -> list[str]:
    if isinstance(value, ast.Dict):
        return format_multiline_dict(value, indent)
    if isinstance(value, ast.List):
        return format_multiline_sequence(value.elts, "[", "]", indent)
    if isinstance(value, ast.Tuple):
        return format_multiline_sequence(value.elts, "(", ")", indent)
    return [f"{indent}{ast.unparse(value)}"]


def format_multiline_dict(value: ast.Dict, indent: str) -> list[str]:
    if not value.keys:
        return [f"{indent}{{}}"]

    child_indent = indent + "    "
    lines = [f"{indent}{{"]
    for key, item in zip(value.keys, value.values, strict=True):
        if key is None:
            append_multiline_item(lines, f"{child_indent}**", item, child_indent)
        else:
            append_multiline_item(
                lines,
                f"{child_indent}{ast.unparse(key)}: ",
                item,
                child_indent,
            )
    lines.append(f"{indent}}}")
    return lines


def format_multiline_sequence(
    values: list[ast.expr],
    open_text: str,
    close_text: str,
    indent: str,
) -> list[str]:
    if not values:
        return [f"{indent}{open_text}{close_text}"]

    child_indent = indent + "    "
    lines = [f"{indent}{open_text}"]
    for item in values:
        append_multiline_item(lines, child_indent, item, child_indent)
    lines.append(f"{indent}{close_text}")
    return lines


def append_multiline_item(
    lines: list[str],
    prefix: str,
    value: ast.expr,
    indent: str,
) -> None:
    inline_value = format_inline_nested_item(prefix, value)
    if inline_value is not None:
        lines.append(inline_value)
        return

    value_lines = format_multiline_expression(value, indent)
    if len(value_lines) == 1:
        lines.append(f"{prefix}{value_lines[0].strip()},")
        return

    lines.append(f"{prefix}{value_lines[0].strip()}")
    lines.extend(value_lines[1:-1])
    lines.append(f"{value_lines[-1]},")


def format_inline_nested_item(prefix: str, value: ast.expr) -> str | None:
    if not isinstance(value, ast.Dict):
        return None

    item = f"{prefix}{ast.unparse(value)},"
    if len(item) > 88:
        return None
    return item
