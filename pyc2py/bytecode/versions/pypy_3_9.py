from pyc2py.bytecode.versions import python_3_9

NAME = "PyPy 3.9"
VERSION = (3, 9)
MAGICS = (336,)
CPYTHON_MAGICS: tuple[int, ...] = (3425,)
READS_AS = (3, 9)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset()
LAYOUT = (3, 9)
CODE_EXTRAS: tuple[str, ...] = ()
POSITIONS = False

OPMAP: dict[int, str] = {
    **{
        number: name
        for number, name in python_3_9.OPMAP.items()
        if number not in {162, 163}
    },
    120: "SETUP_EXCEPT",
    164: "LIST_EXTEND",
    165: "SET_UPDATE",
    166: "DICT_MERGE",
    167: "DICT_UPDATE",
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
        "SETUP_WITH",
    }
)
ABSOLUTE_JUMPS = frozenset(
    {
        "JUMP_ABSOLUTE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_NOT_EXC_MATCH",
        "JUMP_IF_TRUE_OR_POP",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
    }
)
