import ast
import importlib.util
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

from pyc2py.bytecode.operand import import_is_lazy
from pyc2py.decompiler.opcodes.values import make_constant
from pyc2py.pipeline import decompile_file_to_path
from pyc2py.pyc.magic import find_known_version


def decompile_source(source: str) -> str:
    with tempfile.TemporaryDirectory() as folder:
        source_path = Path(folder) / "case.py"
        pyc_path = Path(folder) / "case.pyc"
        target = Path(folder) / "case.pyc.py"
        source_path.write_text(source, encoding="utf-8")
        py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
        decompile_file_to_path(pyc_path, target)
        return target.read_text(encoding="utf-8")


def assert_round_trip(test_case: unittest.TestCase, source: str) -> None:
    recovered = decompile_source(source)
    test_case.assertEqual(ast.dump(ast.parse(recovered)), ast.dump(ast.parse(source)))


class MagicNumberTests(unittest.TestCase):
    def test_known_magic_numbers_map_to_their_versions(self) -> None:
        self.assertEqual(find_known_version(3627), (3, 14))
        self.assertEqual(find_known_version(3666), (3, 15))

    def test_running_interpreter_magic_is_known(self) -> None:
        if sys.version_info[:2] < (3, 12):
            self.skipTest("magic table covers 3.12 and newer")
        magic = int.from_bytes(importlib.util.MAGIC_NUMBER[:2], "little")
        self.assertEqual(find_known_version(magic), sys.version_info[:2])


class ImportLazyFlagTests(unittest.TestCase):
    def test_flag_bit_marks_lazy_imports_on_315(self) -> None:
        self.assertTrue(import_is_lazy(5, (3, 15)))
        self.assertFalse(import_is_lazy(4, (3, 15)))

    def test_older_versions_never_report_lazy(self) -> None:
        self.assertFalse(import_is_lazy(5, (3, 14)))
        self.assertFalse(import_is_lazy(1, (3, 12)))


@unittest.skipUnless(sys.version_info[:2] == (3, 15), "lazy imports are 3.15 bytecode")
class LazyImportTests(unittest.TestCase):
    def test_lazy_import_keeps_the_lazy_keyword(self) -> None:
        assert_round_trip(self, "lazy import json\n")

    def test_lazy_from_import_keeps_the_lazy_keyword(self) -> None:
        assert_round_trip(self, "lazy from pathlib import Path\n")

    def test_plain_imports_stay_eager(self) -> None:
        assert_round_trip(self, "import json\nfrom pathlib import Path\n")

    def test_lazy_import_of_dotted_module(self) -> None:
        assert_round_trip(self, "lazy import xml.etree.ElementTree\n")


@unittest.skipUnless(
    sys.version_info[:2] == (3, 15), "unpacking in comprehensions is 3.15 bytecode"
)
class ComprehensionUnpackingTests(unittest.TestCase):
    def test_starred_list_comprehension_is_recovered(self) -> None:
        assert_round_trip(self, "def f(xs):\n    return [*x for x in xs]\n")

    def test_starred_set_comprehension_is_recovered(self) -> None:
        assert_round_trip(self, "def f(xs):\n    return {*x for x in xs}\n")

    def test_starred_dict_comprehension_is_recovered(self) -> None:
        assert_round_trip(self, "def f(ds):\n    return {**d for d in ds}\n")

    def test_plain_comprehensions_keep_working(self) -> None:
        assert_round_trip(self, "def f(xs):\n    return [x for x in xs]\n")

    def test_starred_comprehension_with_filter(self) -> None:
        assert_round_trip(self, "def f(xs):\n    return [*x for x in xs if x]\n")


@unittest.skipUnless(sys.version_info[:2] == (3, 15), "frozendict is a 3.15 builtin")
class FrozenDictConstantTests(unittest.TestCase):
    def test_frozendict_constant_renders_as_a_call(self) -> None:
        rendered = ast.unparse(make_constant(frozendict({"a": 1, "b": 2})))
        self.assertEqual(rendered, "frozendict({'a': 1, 'b': 2})")


if __name__ == "__main__":
    unittest.main()
