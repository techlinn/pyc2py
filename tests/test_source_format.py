import ast
import py_compile
import tempfile
import unittest
from pathlib import Path

from pyc2py.astree import (
    LINE_MARK,
    STATEMENT_LINE,
    unparse_keeping_lines,
)
from pyc2py.pipeline import decompile_file_to_path
from pyc2py.source import format_readable_source, place_on_original_lines
from pyc2py.types import OUTPUT_OPTIONS, OutputOptions

from support import statement_lines


def decompile_source(source: str) -> str:
    with tempfile.TemporaryDirectory() as folder:
        source_path = Path(folder) / "case.py"
        pyc_path = Path(folder) / "case.pyc"
        target = Path(folder) / "case.pyc.py"
        source_path.write_text(source, encoding="utf-8")
        py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
        decompile_file_to_path(pyc_path, target)
        return target.read_text(encoding="utf-8")


class SourceFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        token = OUTPUT_OPTIONS.set(OutputOptions(keep_original_lines=False))
        self.addCleanup(OUTPUT_OPTIONS.reset, token)

    def test_stacked_decorators_stay_together(self) -> None:
        source = "x = 1\n@first\n@second(3)\ndef f():\n    return 1\n"
        self.assertEqual(
            format_readable_source(source),
            "x = 1\n\n\n@first\n@second(3)\ndef f():\n    return 1\n",
        )

    def test_legacy_print_keeps_its_escapes(self) -> None:
        source = "class A:\n    def f(self):\n        print 'a\\n'\n"
        self.assertEqual(
            format_readable_source(source, prefer_single_quotes=True),
            'class A:\n    def f(self):\n        print("a\\n")\n',
        )

    def test_legacy_bytes_literal_keeps_its_escapes(self) -> None:
        self.assertEqual(
            format_readable_source("bstr = 'a\\x00'\n", legacy_bytes_literals=True),
            'bstr = b"a\\x00"\n',
        )

    def test_a_marked_keyword_already_on_its_own_line_stays(self) -> None:
        source = (
            "def f(a, b, c):\n"
            "    x = g(a,\n"
            "            key=_pyc2py_line_break_b,\n"
            "            other=_pyc2py_line_break_c)\n"
        )
        self.assertEqual(
            format_readable_source(source),
            "def f(a, b, c):\n    x = g(a,\n            key=b,\n            other=c)\n",
        )

    def test_marked_names_start_a_new_line(self) -> None:
        source = (
            "def f(a, b, c):\n"
            "    x = g(a, key=_pyc2py_line_break_b)\n"
            "    return x + _pyc2py_line_break_c\n"
        )
        self.assertEqual(
            format_readable_source(source),
            "def f(a, b, c):\n"
            "    x = g(a,\n"
            "            key=b)\n"
            "    return x + \\\n"
            "            c\n",
        )


class PlacementTests(unittest.TestCase):
    def test_statements_go_back_to_their_placed_lines(self) -> None:
        placed = "x = 1\n\n\ndef f():\n\n    return 1\n"
        formatted = "x = 1\ndef f():\n    return 1\n\n\n"
        self.assertEqual(place_on_original_lines(formatted, placed), placed)

    def test_statements_that_do_not_pair_stay_as_formatted(self) -> None:
        placed = "x = 1\n\n\ny = 2\n"
        formatted = "y = 2\nx = 1\n"
        self.assertIsNone(place_on_original_lines(formatted, placed))

    def test_a_long_line_is_not_split_beyond_its_blank_lines(self) -> None:
        names = ", ".join(f"'name{index}'" for index in range(20))
        placed = f"names = ({names})\nnext_line = 1\n"
        with_room = f"names = ({names})\n" + "\n" * 30 + "next_line = 1\n"
        self.assertEqual(format_readable_source(placed).count("\n"), 2)
        self.assertGreater(format_readable_source(with_room).count("\n"), 3)

    def test_lazy_imports_pair_on_any_host(self) -> None:
        placed = "import os\n\n\nlazy import re\nx = 1\n"
        formatted = "import os\nlazy import re\nx = 1\n"
        self.assertEqual(place_on_original_lines(formatted, placed), placed)

    def test_handler_types_stay_on_the_except_line(self) -> None:
        source = (
            "def f(log):\n"
            "    try:\n"
            "        log.append(1)\n"
            "    except (TypeError, ValueError):\n"
            "        log.append(2)\n"
            "    return log\n"
        )
        self.assertEqual(
            statement_lines(decompile_source(source)), statement_lines(source)
        )

    def test_one_line_bodies_and_semicolons_keep_their_line(self) -> None:
        source = (
            "class E(Exception): pass\n"
            "def g(self):\n"
            "    x = 1; y = 2\n"
            "    for i in x: print(i)\n"
            "    try:\n"
            "        x = 1\n"
            "    except E: pass\n"
            "    while x: x -= 1\n"
            "    return x\n"
            "def h(): pass\n"
        )
        self.assertEqual(
            statement_lines(decompile_source(source)), statement_lines(source)
        )

    def test_nested_headers_without_code_keep_their_lines(self) -> None:
        source = (
            "def f(log):\n"
            "    try:\n"
            "        try:\n"
            "            raise ValueError(1)\n"
            "        except ValueError as e:\n"
            "            raise KeyError(2) from e\n"
            "    except KeyError:\n"
            "        log.append(2)\n"
            "    finally:\n"
            "        log.append(3)\n"
            "    return log\n"
        )
        self.assertEqual(
            statement_lines(decompile_source(source)), statement_lines(source)
        )

    def test_a_one_line_block_that_wraps_goes_below_its_colon(self) -> None:
        module = ast.parse("if a:\n    pass\nelse:\n    f(x, y)\n    z = 1\n")
        branch = module.body[0]
        assert isinstance(branch, ast.If)
        for node, line in (
            (branch, 1),
            (branch.body[0], 2),
            *((s, 3) for s in branch.orelse),
        ):
            setattr(node, STATEMENT_LINE, line)
        call = branch.orelse[0]
        assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
        setattr(call.value.args[0], LINE_MARK, 3)
        setattr(call.value.args[1], LINE_MARK, 4)
        recovered = unparse_keeping_lines(module)
        self.assertEqual(ast.dump(ast.parse(recovered)), ast.dump(module))


if __name__ == "__main__":
    unittest.main()
