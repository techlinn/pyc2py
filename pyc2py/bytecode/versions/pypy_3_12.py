from pyc2py.bytecode.versions import python_3_10

NAME = "PyPy 3.12"
VERSION = (3, 12)
MAGICS = (448,)
CPYTHON_MAGICS: tuple[int, ...] = ()
READS_AS = (3, 9)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset({"COPY"})
LAYOUT = (3, 10)
CODE_EXTRAS: tuple[str, ...] = ("qualname", "exceptiontable")
POSITIONS = True

OPMAP: dict[int, str] = {
    **{
        number: name
        for number, name in python_3_10.OPMAP.items()
        if number not in {2, 3, 4, 5, 73, 122, 129, 135, 141, 143}
    },
    6: "ROT_TWO",
    7: "ROT_THREE",
    8: "DUP_TOP",
    13: "DUP_TOP_TWO",
    35: "PUSH_EXC_INFO",
    36: "CHECK_EXC_MATCH",
    37: "CHECK_EG_MATCH",
    38: "COPY",
    53: "BEFORE_WITH",
    87: "LOAD_LOCALS",
    88: "PREP_RERAISE_STAR",
    99: "SWAP",
    113: "JUMP_IF_NOT_EXC_MATCH",
    121: "RETURN_CONST",
    136: "LOAD_CLOSURE",
    137: "LOAD_DEREF",
    138: "STORE_DEREF",
    139: "DELETE_DEREF",
    153: "JUMP_ABSOLUTE",
    154: "POP_JUMP_FORWARD_IF_NONE",
    158: "POP_JUMP_FORWARD_IF_NOT_NONE",
    159: "GET_AWAITABLE",
    170: "CALL_FUNCTION_KW",
    175: "LOAD_FROM_DICT_OR_GLOBALS",
    176: "LOAD_FROM_DICT_OR_DEREF",
    203: "BUILD_LIST_FROM_ARG",
    204: "CALL_METHOD_KW",
}

RELATIVE_JUMPS = frozenset(
    {
        "FOR_ITER",
        "JUMP_FORWARD",
        "POP_JUMP_FORWARD_IF_NONE",
        "POP_JUMP_FORWARD_IF_NOT_NONE",
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
