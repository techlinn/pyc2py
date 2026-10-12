import py_compile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyc2py.pipeline import decompile_file_to_path
from pyc2py.reporting import validate_generated_source


class ValidationReportTests(unittest.TestCase):
    def test_source_that_does_not_compile_is_written_and_fails(self) -> None:
        broken = SimpleNamespace(source="def f(:\n    pass\n", warnings=())
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / "case.py"
            pyc_path = Path(folder) / "case.pyc"
            target = Path(folder) / "case.pyc.py"
            source_path.write_text("x = 1\n", encoding="utf-8")
            py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
            with mock.patch(
                "pyc2py.source.decompile_native_source", return_value=broken
            ):
                result = decompile_file_to_path(pyc_path, target)
            self.assertFalse(result.report.passed)
            self.assertTrue(result.report.errors)
            self.assertEqual(target.read_text(encoding="utf-8"), broken.source)

    def test_legacy_source_is_reported_as_unchecked(self) -> None:
        report = validate_generated_source(
            Path("case.pyc"), Path("case.pyc.py"), "print 'a'\n", [], (2, 7)
        )
        self.assertTrue(any("neither compiled" in item for item in report.warnings))

    def test_trailing_whitespace_inside_a_docstring_is_not_a_fault(self) -> None:
        source = '"""doc  \nmore"""\nx = 1\n'
        report = validate_generated_source(
            Path("case.pyc"), Path("case.pyc.py"), source, [], (2, 7)
        )
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
