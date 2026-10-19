from pyc2py.bytecode.versions import python_3_10

NAME = "PyPy 3.11"
VERSION = (3, 11)
MAGICS = (432,)
CPYTHON_MAGICS: tuple[int, ...] = ()
READS_AS = (3, 9)
FACTS: frozenset[str] = frozenset()
ARGUMENT_OPS: frozenset[str] = frozenset({"COPY", "GET_AWAITABLE"})
LAYOUT = (3, 10)
CODE_EXTRAS: tuple[str, ...] = ("qualname", "exceptiontable")
POSITIONS = True

OPMAP: dict[int, str] = {
    **{
        number: name
        for number, name in python_3_10.OPMAP.items()
        if number not in {87, 122, 143, 154, 162, 163}
    },
    35: "PUSH_EXC_INFO",
    36: "CHECK_EXC_MATCH",
    37: "CHECK_EG_MATCH",
    38: "COPY",
    53: "BEFORE_WITH",
    88: "PREP_RERAISE_STAR",
    99: "SWAP",
    128: "POP_JUMP_FORWARD_IF_NOT_NONE",
    129: "POP_JUMP_FORWARD_IF_NONE",
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
