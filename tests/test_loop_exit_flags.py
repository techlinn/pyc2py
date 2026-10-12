import ast
import unittest

from pyc2py.astree import clean_decompiled_body
from pyc2py.source import validate_source

SOURCE = """\
if a:
    continue
for x in y:
    break
else:
    break
while a:
    if b:
        continue
"""

FLAGGED = """\
if a:
    __pyc2py_continue_outside_loop__()
for x in y:
    break
else:
    __pyc2py_break_outside_loop__()
while a:
    if b:
        continue"""


class LoopExitFlagTests(unittest.TestCase):
    def test_exit_without_a_loop_becomes_a_helper_call(self) -> None:
        body = clean_decompiled_body(ast.parse(SOURCE).body, is_module=False)
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        self.assertEqual(ast.unparse(module), FLAGGED)

    def test_flagged_source_compiles_and_fails_validation(self) -> None:
        compile(FLAGGED, "case", "exec")
        error = validate_source(FLAGGED).error or ""
        self.assertIn("__pyc2py_continue_outside_loop__", error)


if __name__ == "__main__":
    unittest.main()
