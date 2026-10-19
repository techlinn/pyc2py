import ast

NO_VALUE_OPS = {
    "CACHE",
    "COPY_FREE_VARS",
    "EXTENDED_ARG",
    "MAKE_CELL",
    "NOP",
    "NOT_TAKEN",
    "PRECALL",
    "RESUME",
    "RETURN_GENERATOR",
    "SETUP_ANNOTATIONS",
    "SETUP_LOOP",
    "POP_BLOCK",
    "END_FINALLY",
    "TRACE_RECORD",
}


LOAD_OPS = {
    "LOAD_NAME",
    "LOAD_GLOBAL",
    "LOAD_FAST",
    "LOAD_FAST_AND_CLEAR",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "LOAD_DEREF",
    "LOAD_CLASSDEREF",
    "LOAD_CLOSURE",
}

ROTATION_COUNTS = {
    "ROT_TWO": 2,
    "ROT_THREE": 3,
    "ROT_FOUR": 4,
}


def rotation_count(opname: str) -> int:
    return ROTATION_COUNTS[opname]


def is_annotations_name(value: ast.expr) -> bool:
    return isinstance(value, ast.Name) and value.id == "__annotations__"


def is_null_sentinel(value: object) -> bool:
    return isinstance(value, ast.Name) and value.id == "NULL"
