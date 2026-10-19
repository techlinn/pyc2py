import py_compile
import tempfile
import unittest
from pathlib import Path

from pyc2py.pipeline import closing_record
from pyc2py.pyc.loader import load_pyc
from pyc2py.reporting import validate_generated_source


class ValidationReportTests(unittest.TestCase):
    def test_source_that_does_not_compile_fails_and_its_record_says_so(self) -> None:
        broken = "def f(:\n    pass\n"
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / "case.py"
            pyc_path = Path(folder) / "case.pyc"
            source_path.write_text("x = 1\n", encoding="utf-8")
            py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
            module = load_pyc(pyc_path)
            report = validate_generated_source(
                pyc_path, source_path, broken, [], module.header.version, module.code
            )
        self.assertFalse(report.passed)
        self.assertTrue(report.errors)
        written = broken + closing_record(module, report)
        self.assertIn("# pyc2py: verification FAILED", written)

    def test_legacy_source_is_reported_as_unchecked(self) -> None:
        report = validate_generated_source(
            Path("case.pyc"), Path("case.pyc.py"), "print 'a'\n", [], (2, 7)
        )
        self.assertEqual(report.status, "unverified")
        self.assertTrue(any("cannot be compiled" in item for item in report.warnings))

    def test_trailing_whitespace_inside_a_docstring_is_not_a_fault(self) -> None:
        source = '"""doc  \nmore"""\nx = 1\n'
        report = validate_generated_source(
            Path("case.pyc"), Path("case.pyc.py"), source, [], (2, 7)
        )
        self.assertTrue(report.source_valid)
        self.assertEqual(report.status, "unverified")


if __name__ == "__main__":
    unittest.main()
