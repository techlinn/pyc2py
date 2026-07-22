import ast
import keyword

NO_VALUE_OPS = {
    "CACHE",
    "COPY_FREE_VARS",
    "ENTER_EXECUTOR",
    "EXTENDED_ARG",
    "EXIT_INIT_CHECK",
    "MAKE_CELL",
    "NOP",
    "NOT_TAKEN",
    "PRECALL",
    "RESUME",
    "RETURN_GENERATOR",
    "SET_LINENO",
    "SETUP_ANNOTATIONS",
    "SETUP_LOOP",
    "POP_BLOCK",
    "END_FINALLY",
    "TRACE_RECORD",
}

STORE_OPS = {"STORE_NAME", "STORE_GLOBAL", "STORE_FAST", "STORE_DEREF"}

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


def safe_identifier(name: str) -> str:
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    return "value"


def make_name(name: str, ctx: ast.expr_context) -> ast.Name:
    return ast.Name(id=safe_identifier(name), ctx=ctx)


def is_annotations_name(value: ast.expr) -> bool:
    return isinstance(value, ast.Name) and value.id == "__annotations__"


def is_null_sentinel(value: ast.expr) -> bool:
    return isinstance(value, ast.Name) and value.id == "NULL"
