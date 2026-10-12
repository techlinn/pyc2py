import ast
import dis
import sys
import unittest

from pyc2py.decompiler.engine import NativeDecompiler, NativeResult
from pyc2py.decompiler.runtime import (
    mark_scope_warnings,
    parse_body_or_empty,
    take_scope_warnings,
)


class ScopeWarningTests(unittest.TestCase):
    def test_marked_warnings_are_taken_once_with_the_scope_name(self) -> None:
        body = ast.parse("def f():\n    return lambda: 1\n").body
        function = body[0]
        assert isinstance(function, ast.FunctionDef)
        returned = function.body[0]
        assert isinstance(returned, ast.Return) and returned.value is not None
        mark_scope_warnings(returned.value, "<lambda>", ["inner"])
        mark_scope_warnings(function, "f", ["outer"])

        self.assertEqual(take_scope_warnings(body), ["f: outer", "<lambda>: inner"])
        self.assertEqual(take_scope_warnings(body), [])

    def test_body_that_does_not_parse_is_reported(self) -> None:
        warnings: list[str] = []
        result = NativeResult(source="f() = 1\n", warnings=())

        module = parse_body_or_empty(result, warnings)

        self.assertEqual(module.body, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("not valid Python", warnings[0])

    def test_nested_body_warning_reaches_the_module(self) -> None:
        module = compile("def f(a):\n    return a\n", "case", "exec")
        function = next(c for c in module.co_consts if hasattr(c, "co_code"))
        underflow = bytes([dis.opmap["POP_TOP"], 0, dis.opmap["RETURN_VALUE"], 0])
        broken = function.replace(co_code=function.co_code[:2] + underflow)
        module = module.replace(
            co_consts=tuple(
                broken if constant is function else constant
                for constant in module.co_consts
            )
        )

        result = NativeDecompiler(code=module, version=sys.version_info[:2]).decompile()

        assert result is not None
        self.assertTrue(result.warnings)
        self.assertTrue(all(w.startswith("f: ") for w in result.warnings))
