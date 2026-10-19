import ast
import copy
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from itertools import combinations, pairwise
from types import CodeType
from typing import Any, cast

from pyc2py.astree import STATEMENT_LINE, unparse_keeping_lines
from pyc2py.bytecode.formats import pypy_format
from pyc2py.code_behaviour import code_behaviour, iter_units, portable_behaviour
from pyc2py.compilers import Compiler, CompilerError, compiler_for
from pyc2py.decompiler.print_runtime import render_legacy_print_source
from pyc2py.types import OUTPUT_OPTIONS

# set on statements pyc2py adds that the source may not have had
ADDED_INITIALIZER = "_pyc2py_added_initializer"

MAX_TRIES_PER_UNIT = 60
MAX_TRIES_PER_MODULE = 600

Rewrite = Callable[[], None]
Respelling = Callable[[ast.AST], list[Rewrite]]
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
LOOPS = (ast.For, ast.AsyncFor, ast.While)


def match_original_bytecode(
    module: ast.Module, original: Any, version: tuple[int, ...] | None
) -> None:
    # only the file's own Python, this one or --verify-with, compiles it exactly
    if version is None:
        return

    if isinstance(original, CodeType) and tuple(version[:2]) == sys.version_info[:2]:
        Matcher(module, original, compiled_here, code_behaviour).run()
        return

    opcodes = getattr(original, "opcodes", "")
    language = pypy_format(opcodes).version if opcodes else version
    compiler = compiler_for(language, opcodes)
    if compiler is not None:
        Matcher(
            module,
            original,
            partial(compiled_elsewhere, compiler),
            partial(portable_behaviour, version=compiler.version),
        ).run()


Compile = Callable[[ast.Module], Any]
Behaviour = Callable[[Any], tuple[object, ...]]


class Matcher:
    def __init__(
        self, module: ast.Module, original: Any, compile: Compile, behaviour: Behaviour
    ) -> None:
        self.module = module
        self.originals = dict(iter_units(original, structural=False))
        self.compiled = compile
        self.behaviour = behaviour
        self.tries = 0

    def run(self) -> None:
        recovered = self.compiled(self.module)
        if recovered is None:
            return

        counterparts = dict(iter_units(recovered, structural=False))

        for name, unit in self.originals.items():
            other = counterparts.get(name)
            if name == "<module>" or "#" in name or other is None:
                continue

            wanted = self.behaviour_of(unit)
            if wanted is not None and wanted != self.behaviour_of(other):
                self.respell(name, wanted)

    def respell(self, name: str, wanted: tuple[object, ...]) -> None:
        located = locate(self.module, name)
        if located is None:
            return

        top_index, node = located
        for tried, candidate in enumerate(candidates(node)):
            if tried >= MAX_TRIES_PER_UNIT or self.tries >= MAX_TRIES_PER_MODULE:
                return

            self.tries += 1
            if self.matches(name, top_index, node, candidate, wanted):
                replace_node(self.module, node, candidate)
                return

    def matches(
        self,
        name: str,
        top_index: int,
        node: ast.AST,
        candidate: ast.AST,
        wanted: tuple[object, ...],
    ) -> bool:
        # the candidate stands in the tree only while its statement compiles
        replace_node(self.module, node, candidate)

        try:
            top = self.module.body[top_index]
            trial = ast.Module(
                body=[*future_imports(self.module), top], type_ignores=[]
            )
            compiled = self.compiled(trial)
        finally:
            replace_node(self.module, candidate, node)

        if compiled is None:
            return False

        unit = dict(iter_units(compiled, structural=False)).get(name)
        return unit is not None and self.behaviour_of(unit) == wanted

    def behaviour_of(self, code: Any) -> tuple[object, ...] | None:
        # damaged bytecode cannot be compared; the verifier reports it
        try:
            return self.behaviour(code)
        except Exception:
            return None


def compiled_here(module: ast.Module) -> CodeType | None:
    try:
        return compile(on_marked_lines(module), "<pyc2py>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return None


def compiled_elsewhere(compiler: Compiler, module: ast.Module) -> Any:
    # the text a reader of the output would compile, with each statement on its line
    options = OUTPUT_OPTIONS.get()
    token = OUTPUT_OPTIONS.set(replace(options, keep_original_lines=True))

    try:
        text = unparse_keeping_lines(on_marked_lines(module)) + "\n"
    except Exception:
        return None
    finally:
        OUTPUT_OPTIONS.reset(token)

    if compiler.version < (3, 0) and not prints_as_function(module):
        text = render_legacy_print_source(text)

    try:
        return compiler.compile(text)
    except CompilerError:
        return None


def prints_as_function(module: ast.Module) -> bool:
    return any(
        alias.name == "print_function"
        for statement in future_imports(module)
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    )


def on_marked_lines(module: ast.Module) -> ast.Module:
    # from 3.12 the compiler's choices depend on which statements share a line
    give_lines(module, 1)
    return module


def give_lines(node: ast.AST, line: int) -> None:
    if isinstance(node, ast.stmt):
        line = getattr(node, STATEMENT_LINE, None) or line

    if "lineno" in node._attributes:
        for field in ("lineno", "end_lineno"):
            setattr(node, field, line)

        for field in ("col_offset", "end_col_offset"):
            setattr(node, field, 0)

    for child in ast.iter_child_nodes(node):
        give_lines(child, line)


def locate(module: ast.Module, name: str) -> tuple[int, ast.AST] | None:
    path = name.split(".")

    for index, statement in enumerate(module.body):
        found = follow(statement, path)
        if found is not None:
            return index, found

    return None


def follow(node: ast.AST, path: list[str]) -> ast.AST | None:
    if not isinstance(node, DEFINITIONS) or node.name != path[0]:
        return None
    if len(path) == 1:
        return node

    for child in ast.walk(node):
        if child is not node and isinstance(child, DEFINITIONS):
            found = follow(child, path[1:])
            if found is not None:
                return found

    return None


def replace_node(tree: ast.AST, old: ast.AST, new: ast.AST) -> None:
    if tree is old:
        tree.__dict__.update(new.__dict__)
        return

    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, field, new)
                return

            if not isinstance(value, list):
                continue

            items = cast("list[object]", value)
            if any(item is old for item in items):
                items[[id(item) for item in items].index(id(old))] = new
                return


def future_imports(module: ast.Module) -> list[ast.stmt]:
    return [
        statement
        for statement in module.body
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
    ]


def candidates(node: ast.AST) -> Iterator[ast.AST]:
    # each respelling per place, then everywhere, then two kinds together
    applicable: list[Respelling] = []

    for respelling in RESPELLINGS:
        # finding the places changes nothing; only a copy is rewritten
        count = len(respelling(node))
        for index in range(count):
            yield respelled(node, [(respelling, [index])])

        if count > 1:
            yield respelled(node, [(respelling, list(range(count)))])
        if count:
            applicable.append(respelling)

    for first, second in combinations(applicable, 2):
        yield respelled(node, [(first, None), (second, None)])


def respelled(
    node: ast.AST, steps: list[tuple[Respelling, list[int] | None]]
) -> ast.AST:
    copied = copy.deepcopy(node)

    for respelling, indexes in steps:
        rewrites = respelling(copied)
        chosen = range(len(rewrites)) if indexes is None else indexes
        for index in sorted(chosen, reverse=True):
            rewrites[index]()

    return copied


def own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    # the nodes of a definition, without those of definitions inside it
    pending = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        yield child

        if not isinstance(child, (*DEFINITIONS, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(child))


def statement_lists(node: ast.AST) -> Iterator[list[ast.stmt]]:
    for child in [node, *own_nodes(node)]:
        for field in ("body", "orelse", "finalbody"):
            block = getattr(child, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def become(node: ast.AST, replacement: ast.AST) -> None:
    node.__class__ = replacement.__class__
    node.__dict__.clear()
    node.__dict__.update(replacement.__dict__)


def added_initializers(node: ast.AST) -> list[Rewrite]:
    body: list[ast.stmt] = getattr(node, "body", [])
    return [
        partial(body.remove, statement)
        for statement in body
        if getattr(statement, ADDED_INITIALIZER, False)
    ]


def copied_loop_returns(node: ast.AST) -> list[Rewrite]:
    # a return copied into a loop from just after it was a break
    rewrites: list[Rewrite] = []

    for block in statement_lists(node):
        for loop, after in pairwise(block):
            if not isinstance(loop, LOOPS) or loop.orelse:
                continue
            if not isinstance(after, ast.Return):
                continue

            rewrites.extend(
                partial(become, child, ast.Break())
                for child in loop_own_nodes(loop)
                if isinstance(child, ast.Return) and ast.dump(child) == ast.dump(after)
            )

    return rewrites


def loop_own_nodes(loop: ast.For | ast.AsyncFor | ast.While) -> Iterator[ast.AST]:
    # a break inside a nested loop would leave that loop instead
    pending: list[ast.AST] = list(loop.body)
    while pending:
        child = pending.pop()
        yield child

        if not isinstance(child, (*DEFINITIONS, ast.Lambda, *LOOPS)):
            pending.extend(ast.iter_child_nodes(child))


def negated_bool_ops(node: ast.AST) -> list[Rewrite]:
    return [
        partial(apply_de_morgan, child)
        for child in own_nodes(node)
        if isinstance(child, ast.BoolOp)
        and all(
            isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
            for value in child.values
        )
    ]


def apply_de_morgan(node: ast.BoolOp) -> None:
    # not a and not b is not (a or b); not a or not b is not (a and b)
    other = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    operands = [
        value.operand for value in node.values if isinstance(value, ast.UnaryOp)
    ]
    become(
        node, ast.UnaryOp(op=ast.Not(), operand=ast.BoolOp(op=other, values=operands))
    )


def loop_end_elses(node: ast.AST) -> list[Rewrite]:
    return [
        partial(continue_spelling, child)
        for child in [node, *own_nodes(node)]
        if isinstance(child, LOOPS)
        and child.body
        and isinstance(child.body[-1], ast.If)
        and child.body[-1].orelse
    ]


def continue_spelling(loop: ast.For | ast.AsyncFor | ast.While) -> None:
    last = loop.body[-1]
    assert isinstance(last, ast.If)
    rest, last.orelse = last.orelse, []
    last.body = [*last.body, ast.Continue()]
    loop.body.extend(rest)


def returns_after_with(node: ast.AST) -> list[Rewrite]:
    # a return written inside a with leaves it the same way
    return [
        partial(move_into_with, block, index)
        for block in statement_lists(node)
        for index, (statement, after) in enumerate(pairwise(block))
        if isinstance(statement, (ast.With, ast.AsyncWith))
        and isinstance(after, ast.Return)
    ]


def move_into_with(block: list[ast.stmt], index: int) -> None:
    statement = block[index]
    assert isinstance(statement, (ast.With, ast.AsyncWith))
    body = [item for item in statement.body if not isinstance(item, ast.Pass)]
    statement.body = [*body, block.pop(index + 1)]


def loop_elses(node: ast.AST) -> list[Rewrite]:
    # after a loop with no break, the rest runs exactly when its else would
    return [
        partial(move_into_orelse, block, index)
        for block in statement_lists(node)
        for index, statement in enumerate(block[:-1])
        if isinstance(statement, LOOPS)
        and not statement.orelse
        and not any(isinstance(child, ast.Break) for child in loop_own_nodes(statement))
    ]


def try_elses(node: ast.AST) -> list[Rewrite]:
    # after a try whose handlers all leave, the rest runs exactly when its else would
    return [
        partial(move_into_orelse, block, index)
        for block in statement_lists(node)
        for index, statement in enumerate(block[:-1])
        if isinstance(statement, ast.Try)
        and not statement.orelse
        and not statement.finalbody
        and all(
            handler.body and isinstance(handler.body[-1], LEAVES)
            for handler in statement.handlers
        )
    ]


def move_into_orelse(block: list[ast.stmt], index: int) -> None:
    statement = block[index]
    assert isinstance(statement, (*LOOPS, ast.Try, ast.If))
    statement.orelse = block[index + 1 :]
    del block[index + 1 :]


def exit_elses(node: ast.AST) -> list[Rewrite]:
    # after an if that always leaves, the rest runs exactly when its else would
    return [
        partial(move_into_orelse, block, index)
        for block in statement_lists(node)
        for index, statement in enumerate(block[:-1])
        if isinstance(statement, ast.If)
        and not statement.orelse
        and isinstance(statement.body[-1], LEAVES)
    ]


def continue_elses(node: ast.AST) -> list[Rewrite]:
    # in a loop body, if ...: continue then the rest is if ...: else: the rest
    return [
        partial(drop_continue_into_orelse, block, index)
        for loop in [node, *own_nodes(node)]
        if isinstance(loop, LOOPS)
        for block in [loop.body]
        for index, statement in enumerate(block[:-1])
        if isinstance(statement, ast.If)
        and not statement.orelse
        and len(statement.body) > 1
        and isinstance(statement.body[-1], ast.Continue)
    ]


def drop_continue_into_orelse(block: list[ast.stmt], index: int) -> None:
    statement = block[index]
    assert isinstance(statement, ast.If)
    statement.body = statement.body[:-1]
    move_into_orelse(block, index)


LEAVES = (ast.Break, ast.Continue, ast.Return, ast.Raise)

RESPELLINGS: tuple[Respelling, ...] = (
    added_initializers,
    copied_loop_returns,
    negated_bool_ops,
    loop_end_elses,
    returns_after_with,
    loop_elses,
    try_elses,
    exit_elses,
    continue_elses,
)
