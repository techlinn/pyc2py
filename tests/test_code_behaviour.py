import sys
import unittest
from types import CodeType

from pyc2py.reporting import code_behaviour


def function_code(source: str) -> CodeType:
    module = compile(source, "case", "exec")
    return next(c for c in module.co_consts if isinstance(c, CodeType))


class CodeBehaviourTests(unittest.TestCase):
    def assert_same(self, first: str, second: str) -> None:
        self.assertEqual(
            code_behaviour(function_code(first)), code_behaviour(function_code(second))
        )

    def assert_different(self, first: str, second: str) -> None:
        self.assertNotEqual(
            code_behaviour(function_code(first)), code_behaviour(function_code(second))
        )

    def test_arguments_on_separate_lines_are_the_same_call(self) -> None:
        self.assert_same(
            "def f(a, b):\n    return g(a,\n             b)\n",
            "def f(a, b):\n    return g(a, b)\n",
        )

    def test_store_and_load_on_separate_lines_are_the_same(self) -> None:
        self.assert_same(
            "def f(xs):\n    for x in xs: x.go()\n",
            "def f(xs):\n    for x in xs:\n        x.go()\n",
        )

    def test_a_statement_without_effect_is_ignored(self) -> None:
        self.assert_same(
            "def f(a):\n    while a:\n        pass\n        continue\n",
            "def f(a):\n    while a:\n        continue\n",
        )

    def test_a_jump_far_enough_to_need_an_extended_argument(self) -> None:
        body = "".join(f"        g({index})\n" for index in range(60))
        self.assert_same(
            f"def f(a, b):\n    if a:\n{body}    return h(a,\n             b)\n",
            f"def f(a, b):\n    if a:\n{body}    return h(a, b)\n",
        )

    def test_real_differences_stay_different(self) -> None:
        self.assert_different(
            "def f(a, b):\n    return g(a, b)\n",
            "def f(a, b):\n    return g(b, a)\n",
        )
        self.assert_different(
            "def f(a, b):\n    if a:\n        g()\n    h()\n",
            "def f(a, b):\n    if a:\n        g()\n        h()\n",
        )
        self.assert_different(
            "def f(a, b):\n    a = b\n    return a\n",
            "def f(a, b):\n    b = a\n    return a\n",
        )

    # the else block is compiled before the handlers, so the names it uses get
    # lower numbers than when the same code follows the try statement
    # on python 3.11 the handlers of the two forms are laid out differently,
    # so there the comparison still reports a difference: a false warning
    @unittest.skipIf(sys.version_info < (3, 12), "handler layout differs on 3.11")
    def test_the_order_of_first_use_is_ignored(self) -> None:
        self.assert_same(
            "def f(p):\n    try:\n        s = a(p)\n    except E:\n        return 0\n"
            "    else:\n        if b(s):\n            return 1\n    return c(p)\n",
            "def f(p):\n    try:\n        s = a(p)\n    except E:\n        return 0\n"
            "    if b(s):\n        return 1\n    return c(p)\n",
        )

    def test_constants_are_compared_by_value_and_type(self) -> None:
        self.assert_different(
            "def f():\n    return 0.0\n", "def f():\n    return -0.0\n"
        )
        self.assert_different("def f():\n    return 1\n", "def f():\n    return True\n")
        self.assert_different(
            "def f():\n    return g(a=1, b=2)\n", "def f():\n    return g(a=1, c=2)\n"
        )
        self.assert_same(
            "def f(x):\n    return x in {'a', 'b', 'c'}\n",
            "def f(x):\n    return x in {'c', 'b', 'a'}\n",
        )

    def test_a_docstring_is_part_of_the_behaviour(self) -> None:
        self.assert_different(
            'def f():\n    "one"\n    return 1\n', 'def f():\n    "two"\n    return 1\n'
        )
        self.assert_different(
            'def f():\n    "one"\n    return 1\n', "def f():\n    return 1\n"
        )

    def test_names_are_compared_by_name(self) -> None:
        self.assert_different(
            "def f(a):\n    return a.b()\n", "def f(a):\n    return a.c()\n"
        )
        self.assert_different("def f():\n    return g\n", "def f():\n    return h\n")

    def test_a_longer_try_body_is_a_difference(self) -> None:
        self.assert_different(
            "def f(p):\n    try:\n        a = g(p)\n    except E:\n        raise K\n"
            "    b = h(a)\n    return b\n",
            "def f(p):\n    try:\n        a = g(p)\n        b = h(a)\n        return b\n"
            "    except E:\n        raise K\n",
        )

    # a continue on its own line is a jump inside the with block; the jump the
    # compiler makes for the if statement is not covered by it
    # only holds where the two forms get the same handler table, which is
    # python 3.12 and 3.13; elsewhere the comparison errs on the strict side
    @unittest.skipUnless(
        sys.version_info[:2] in {(3, 12), (3, 13)}, "handler table differs"
    )
    def test_whether_a_jump_is_covered_is_ignored(self) -> None:
        self.assert_same(
            "def f(d):\n    with g(d):\n        for w in d:\n            o = w()\n"
            "            if o is not None:\n                h(o)\n",
            "def f(d):\n    with g(d):\n        for w in d:\n            o = w()\n"
            "            if o is None:\n                continue\n            h(o)\n",
        )

    def test_how_arguments_are_taken_is_part_of_the_behaviour(self) -> None:
        self.assert_different("def f(a):\n    return a\n", "def f(*a):\n    return a\n")
        self.assert_different(
            "def f(a):\n    return a\n", "def f(**a):\n    return a\n"
        )
        self.assert_different(
            "def f(a, b):\n    return a\n", "def f(a, *, b):\n    return a\n"
        )
        self.assert_different(
            "def f(a):\n    return a\n", "def f(a, /):\n    return a\n"
        )
        self.assert_different(
            "def f(a):\n    return a\n", "def f():\n    a = g\n    return a\n"
        )

    def test_a_lambda_is_not_a_def(self) -> None:
        self.assert_different("f = lambda: 1\n", "def f():\n    return 1\n")

    def test_only_the_first_line_store_of_a_class_is_ignored(self) -> None:
        self.assert_different(
            "class C:\n    __firstlineno__ = 5\n",
            "class C:\n    __firstlineno__ = 7\n",
        )

    def test_the_line_of_a_class_is_ignored(self) -> None:
        self.assert_same("class C:\n    x = 1\n", "\n\nclass C:\n    x = 1\n")


if __name__ == "__main__":
    unittest.main()
