from pyc2py.bytecode.versions import python_3_10

NAME = "PyPy 3.11 (7.3.20)"
VERSION = (3, 11)
MAGICS = (416,)
CPYTHON_MAGICS: tuple[int, ...] = ()
READS_AS = (3, 9)
FACTS: frozenset[str] = frozenset({"two_item_exceptions"})
ARGUMENT_OPS: frozenset[str] = frozenset()
LAYOUT = (3, 10)
CODE_EXTRAS: tuple[str, ...] = ("qualname",)
POSITIONS = True

OPMAP: dict[int, str] = {
    **{
        number: name
        for number, name in python_3_10.OPMAP.items()
        if number not in {129, 162, 163}
    },
    37: "CHECK_EG_MATCH",
    88: "PREP_RERAISE_STAR",
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
