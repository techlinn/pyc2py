from pyc2py.bytecode.versions import python_3_8

NAME = "PyPy 3.8"
VERSION = (3, 8)
MAGICS = (256,)
CPYTHON_MAGICS: tuple[int, ...] = (3413,)
READS_AS = (3, 8)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset()
LAYOUT = (3, 8)
CODE_EXTRAS: tuple[str, ...] = ()
POSITIONS = False

OPMAP: dict[int, str] = {
    **{
        number: name
        for number, name in python_3_8.OPMAP.items()
        if number not in {6, 158, 160}
    },
    121: "SETUP_EXCEPT",
    201: "LOOKUP_METHOD",
    203: "BUILD_LIST_FROM_ARG",
    204: "CALL_METHOD_KW",
}

RELATIVE_JUMPS = frozenset(
    {
        "CALL_FINALLY",
        "FOR_ITER",
        "JUMP_FORWARD",
        "SETUP_ASYNC_WITH",
        "SETUP_EXCEPT",
        "SETUP_FINALLY",
        "SETUP_WITH",
    }
)
ABSOLUTE_JUMPS = frozenset(
    {
        "JUMP_ABSOLUTE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
    }
)
