import unittest

from pyc2py.source import format_readable_source


class SourceFormatTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
