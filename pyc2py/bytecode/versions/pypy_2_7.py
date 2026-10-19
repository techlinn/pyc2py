from pyc2py.bytecode.versions import python_2_7

NAME = "PyPy 2.7"
VERSION = (2, 7)
MAGICS = (62218,)
CPYTHON_MAGICS: tuple[int, ...] = ()
READS_AS = (2, 7)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset()
LAYOUT = (2, 7)
CODE_EXTRAS: tuple[str, ...] = ()
POSITIONS = False

OPMAP: dict[int, str] = {
    **python_2_7.OPMAP,
    201: "LOOKUP_METHOD",
    202: "CALL_METHOD",
    203: "BUILD_LIST_FROM_ARG",
    204: "JUMP_IF_NOT_DEBUG",
}

RELATIVE_JUMPS = frozenset(
    {
        "FOR_ITER",
        "JUMP_FORWARD",
        "JUMP_IF_NOT_DEBUG",
        "SETUP_EXCEPT",
        "SETUP_FINALLY",
        "SETUP_LOOP",
        "SETUP_WITH",
    }
)
ABSOLUTE_JUMPS = frozenset(
    {
        "CONTINUE_LOOP",
        "JUMP_ABSOLUTE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
    }
)
