import ast
import py_compile
import tempfile
import unittest
from pathlib import Path

from pyc2py.pipeline import decompile_file_to_path


def decompile_source(source: str) -> str:
    with tempfile.TemporaryDirectory() as folder:
        source_path = Path(folder) / "case.py"
        pyc_path = Path(folder) / "case.pyc"
        target = Path(folder) / "case.pyc.py"
        source_path.write_text(source, encoding="utf-8")
        py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
        decompile_file_to_path(pyc_path, target)
        return target.read_text(encoding="utf-8")


class CompareConditionTests(unittest.TestCase):
    # python 3.13+ flags a comparison whose result is only used as a truth
    # value, and dis then prints the operator as bool(==)
    def test_comparison_used_as_condition_is_recovered(self) -> None:
        for symbol in ("<", "<=", "==", "!=", ">", ">="):
            with self.subTest(symbol=symbol):
                source = f"if a {symbol} b:\n    x = 1\n"
                recovered = decompile_source(source)
                self.assertEqual(
                    ast.dump(ast.parse(recovered)), ast.dump(ast.parse(source))
                )


if __name__ == "__main__":
    unittest.main()
