from pyc2py.bytecode.versions import python_3_6

NAME = "PyPy 3.7"
VERSION = (3, 7)
MAGICS = (224, 240)
CPYTHON_MAGICS: tuple[int, ...] = ()
READS_AS = (3, 7)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset()
LAYOUT = (3, 7)
CODE_EXTRAS: tuple[str, ...] = ()
POSITIONS = False

OPMAP: dict[int, str] = {
    **{
        number: name for number, name in python_3_6.OPMAP.items() if number not in {158}
    },
    161: "CALL_METHOD",
    201: "LOOKUP_METHOD",
    203: "BUILD_LIST_FROM_ARG",
    204: "CALL_METHOD_KW",
}

RELATIVE_JUMPS = frozenset(
    {
        "FOR_ITER",
        "JUMP_FORWARD",
        "SETUP_ASYNC_WITH",
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
