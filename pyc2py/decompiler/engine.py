import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pyc2py.astree import (
    add_global_declarations,
    add_module_global_declarations,
    add_nonlocal_declarations,
    clean_decompiled_body,
)
from pyc2py.bytecode.decoder import decode_instructions
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import make_offset_index
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.decompiler.control import ControlRecoveryMixin
from pyc2py.decompiler.exception_structures import (
    try_translate_except,
    try_translate_exception_table_except,
    try_translate_exception_table_except_finally,
    try_translate_exception_table_except_star,
    try_translate_exception_table_finally,
    try_translate_finally,
)
from pyc2py.decompiler.opcode_runtime import OpcodeRuntimeMixin
from pyc2py.decompiler.opcodes.flow import is_jump_op, is_terminal_op
from pyc2py.decompiler.print_runtime import (
    PrintRuntimeMixin,
    render_legacy_print_source,
)
from pyc2py.decompiler.stack_runtime import StackRuntimeMixin
from pyc2py.decompiler.structures import (
    is_forward_conditional_jump,
    skip_returning_with_exception_handler,
)
from pyc2py.stack import FastStack
from pyc2py.types import LiveWarningList, ProgressCallback


@dataclass(slots=True)
class NativeResult:
    source: str
    warnings: tuple[str, ...]


@dataclass(slots=True)
class NativeDecompiler(
    ControlRecoveryMixin,
    PrintRuntimeMixin,
    StackRuntimeMixin,
    OpcodeRuntimeMixin,
):
    code: Any
    version: tuple[int, ...] | None
    is_module: bool = True
    progress: ProgressCallback | None = None
    stack: FastStack[Any] = field(default_factory=FastStack)
    statements: list[ast.stmt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kw_names: tuple[str, ...] = ()
    unpack_groups: list[Any] = field(default_factory=list)
    pending_print_items: list[ast.expr] = field(default_factory=list)
    pending_print_target: ast.expr | None = None
    global_names: list[str] = field(default_factory=list)
    pending_simultaneous_store_count: int = 0
    simultaneous_store_group: int = 0
    saw_unicode_literals_future: bool = False
    saw_print_function_future: bool = False
    loop_continue_offsets: frozenset[int] = field(default_factory=frozenset)
    loop_break_offsets: frozenset[int] = field(default_factory=frozenset)
    loop_none_return_is_break: bool = False

    def __post_init__(self) -> None:
        if self.progress is not None and type(self.warnings) is list:
            self.warnings = LiveWarningList(self.progress, prefix="warning: native: ")

    def uses_unicode_literals(self) -> bool:
        flags = int(getattr(self.code, "co_flags", 0) or 0)
        return self.saw_unicode_literals_future or bool(
            flags & CO_FUTURE_UNICODE_LITERALS
        )

    def decompile(self) -> NativeResult | None:
        self.emit_progress(f"decoding bytecode for {self.code_name()}")
        instructions = decode_instructions(self.code, self.version)
        if not instructions:
            self.emit_progress("no instructions found")
            return None

        self.emit_progress(f"decoded {len(instructions)} instruction(s)")
        self.emit_progress("translating bytecode instructions")
        self.statements = self.translate_range(instructions, 0, len(instructions))
        self.emit_progress("flushing pending print statements")
        self.flush_print_items(newline=True)
        if not self.is_module:
            self.emit_progress("recovering function global declarations")
            self.add_global_names(find_loaded_global_store_sources(instructions))

        self.emit_progress("cleaning recovered AST")
        body = clean_decompiled_body(self.statements, self.is_module)
        if self.is_module:
            self.emit_progress("recovering module global declarations")
            body = add_module_global_declarations(
                body, find_module_explicit_global_names(instructions)
            )
        else:
            body = add_missing_local_initializers(body, instructions, self.code)
            body = add_global_declarations(body, self.global_names)
            body = add_nonlocal_declarations(
                body, find_explicit_nonlocal_names(self.code, instructions)
            )
        if not body and not self.is_module:
            body = [ast.Pass()]
        self.emit_progress("fixing AST locations")
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        self.emit_progress("unparsing AST to source")
        source = ast.unparse(module) + "\n"
        if (
            self.should_render_legacy_print_source()
            and not self.saw_print_function_future
        ):
            self.emit_progress("rendering legacy print statements")
            source = render_legacy_print_source(source)
        return NativeResult(source=source, warnings=tuple(self.warnings))

    def emit_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def code_name(self) -> str:
        return str(
            getattr(self.code, "co_qualname", None)
            or getattr(self.code, "co_name", "code")
        )

    def should_render_legacy_print_source(self) -> bool:
        if self.version is None or self.version >= (3, 0):
            return False
        return self.is_module

    def should_skip_loop_break_cleanup_pop(
        self,
        instructions: list[Instruction],
        cursor: int,
    ) -> bool:
        if instructions[cursor].opname != "POP_TOP":
            return False

        next_index = next_non_metadata_index(instructions, cursor + 1)
        if next_index is None:
            return False

        next_instruction = instructions[next_index]
        if next_instruction.opname not in {"JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD"}:
            return False
        return next_instruction.argval in self.loop_break_offsets

    def skip_orphan_call_suffix(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> int | None:
        if cursor >= end_index or instructions[cursor].opname != "CACHE":
            return None
        if not instructions[cursor].is_jump_target:
            return None

        call_index = next_non_metadata_index(instructions, cursor)
        if call_index is None or call_index >= end_index:
            return None
        call = instructions[call_index]

        precall_index = previous_non_metadata_index(instructions, call_index - 1)
        if not self.is_orphan_call_suffix(call, precall_index, instructions):
            return None

        cursor = call_index + 1
        while cursor < end_index and instructions[cursor].opname == "CACHE":
            cursor += 1
        if cursor < end_index and instructions[cursor].opname == "POP_TOP":
            return cursor + 1
        return cursor

    def is_orphan_call_suffix(
        self,
        call: Instruction,
        precall_index: int | None,
        instructions: list[Instruction],
    ) -> bool:
        return (
            call.opname in {"CALL", "CALL_FUNCTION"}
            and precall_index is not None
            and instructions[precall_index].opname == "PRECALL"
            and len(self.stack) < int(call.arg or 0) + 1
        )

    def add_global_name(self, name: str) -> None:
        if name not in self.global_names:
            self.global_names.append(name)

    def add_global_names(self, names: list[str]) -> None:
        for name in names:
            self.add_global_name(name)

    def translate_range(
        self,
        instructions: list[Instruction],
        start_index: int,
        end_index: int,
    ) -> list[ast.stmt]:
        saved_statements = self.statements
        self.statements = []
        offset_to_index = make_offset_index(instructions)

        cursor = start_index
        while cursor < end_index:
            skipped_cursor = self.skip_unstructured_instruction(
                instructions,
                cursor,
                start_index,
                end_index,
            )
            if skipped_cursor is not None:
                cursor = skipped_cursor
                continue

            if self.is_module and is_after_implicit_module_return(
                instructions,
                cursor,
                start_index,
            ):
                break

            advanced = self.try_translate_structured_range(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
            if advanced is None:
                self.run_instruction(instructions[cursor])
                cursor += 1
            else:
                cursor = advanced

        self.flush_yield_expressions()
        self.flush_print_items(newline=False)
        result = self.statements
        self.statements = saved_statements
        return result

    def skip_unstructured_instruction(
        self,
        instructions: list[Instruction],
        cursor: int,
        start_index: int,
        end_index: int,
    ) -> int | None:
        orphan_call_suffix = self.skip_orphan_call_suffix(
            instructions,
            cursor,
            end_index,
        )
        if orphan_call_suffix is not None:
            return orphan_call_suffix

        skipped_with_cleanup = should_skip_returning_with_cleanup_after_terminal(
            instructions,
            cursor,
            start_index,
            end_index,
        )
        if skipped_with_cleanup is not None:
            return skipped_with_cleanup
        if should_skip_range_cursor(self, instructions, cursor, start_index):
            return cursor + 1
        return None

    def try_translate_structured_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        retry_loop = self.try_translate_backward_retry_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if retry_loop is not None:
            return retry_loop

        advanced = self.try_translate_exception_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        advanced = self.try_translate_statement_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        return self.try_translate_loop_or_branch(
            instruction,
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_exception_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        opname = normalized_opcode_name(instruction.opname)
        if opname == "NOP" and cursor + 1 < end_index:
            advanced = self.try_translate_exception_range(
                instructions,
                offset_to_index,
                cursor + 1,
                end_index,
            )
            if advanced is not None:
                return advanced

        advanced = self.try_translate_exception_table_range(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced

        if opname == "SETUP_FINALLY":
            return try_translate_finally(
                self,
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname == "SETUP_EXCEPT":
            return try_translate_except(
                self,
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        return None

    def try_translate_exception_table_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        for handler in (
            try_translate_exception_table_finally,
            try_translate_exception_table_except_finally,
            try_translate_exception_table_except,
            try_translate_exception_table_except_star,
        ):
            advanced = handler(self, instructions, offset_to_index, cursor, end_index)
            if advanced is not None:
                return advanced
        return None

    def try_translate_statement_range(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        opname = normalized_opcode_name(instruction.opname)
        short_circuit = self.try_translate_short_circuit_expression(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if short_circuit is not None:
            return short_circuit
        cleanup_return = self.try_translate_loop_cleanup_return(
            instructions,
            cursor,
            end_index,
        )
        if cleanup_return is not None:
            return cleanup_return
        if opname in {"BEFORE_ASYNC_WITH", "BEFORE_WITH"}:
            return self.try_translate_with(instructions, cursor, end_index)
        if opname in {"GET_AWAITABLE", "GET_YIELD_FROM_ITER"}:
            return self.try_translate_send_value(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname in {
            "COPY",
            "LOAD_DEREF",
            "LOAD_FAST",
            "LOAD_FAST_BORROW",
            "LOAD_FAST_CHECK",
            "LOAD_GLOBAL",
            "LOAD_NAME",
            "MATCH_MAPPING",
            "MATCH_SEQUENCE",
        }:
            return self.try_translate_match_or_with_statement(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        if opname == "SWAP":
            self.prepare_simultaneous_store_group(instructions, cursor, end_index)
        return None

    def try_translate_match_or_with_statement(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        with_region = self.try_translate_with(instructions, cursor, end_index)
        if with_region is not None:
            return with_region
        return self.try_translate_simple_match(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_short_circuit_expression(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        instruction = instructions[cursor]
        if instruction.opname not in {"JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"}:
            return None
        if not self.stack:
            return None
        if not isinstance(instruction.argval, int):
            return None

        target_index = offset_to_index.get(instruction.argval)
        if (
            target_index is None
            or target_index <= cursor + 1
            or target_index > end_index
        ):
            return None

        child = self.make_child()
        child.pop_or_none()
        child.translate_range(instructions, cursor + 1, target_index)
        if not child.stack:
            return None

        left = self.expression_from_stack_value(self.pop_or_none())
        right = child.expression_from_stack_value(child.stack[-1])
        self.warnings.extend(child.warnings)
        self.stack.append(
            ast.BoolOp(
                op=ast.Or()
                if instruction.opname == "JUMP_IF_TRUE_OR_POP"
                else ast.And(),
                values=[left, right],
            )
        )
        return target_index

    def prepare_simultaneous_store_group(
        self,
        instructions: list[Instruction],
        cursor: int,
        end_index: int,
    ) -> None:
        count = int(instructions[cursor].arg or 0)
        if count < 2:
            return
        if not has_following_subscript_stores(
            instructions, cursor + 1, end_index, count
        ):
            return
        self.simultaneous_store_group += 1
        self.pending_simultaneous_store_count = count

    def try_translate_loop_or_branch(
        self,
        instruction: Instruction,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        opname = normalized_opcode_name(instruction.opname)
        infinite_loop = self.try_translate_modern_infinite_while_loop(
            instructions,
            cursor,
            end_index,
        )
        if infinite_loop is not None:
            return infinite_loop

        loop_handler = self.loop_branch_handler(opname)
        if loop_handler is not None:
            return loop_handler(instructions, offset_to_index, cursor, end_index)
        if is_forward_conditional_jump(instruction):
            return self.try_translate_conditional(
                instructions,
                offset_to_index,
                cursor,
                end_index,
            )
        return None

    def loop_branch_handler(
        self, opname: str
    ) -> Callable[[list[Instruction], dict[int, int], int, int], int | None] | None:
        return {
            "GET_ITER": self.try_translate_iter_loop,
            "GET_AITER": self.try_translate_async_iter_loop,
            "SETUP_LOOP": self.try_translate_while_loop,
            "FOR_LOOP": self.try_translate_legacy_for_loop,
        }.get(opname)

    def try_translate_iter_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        advanced = self.try_translate_inlined_comprehension(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced
        return self.try_translate_for_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )

    def try_translate_async_iter_loop(
        self,
        instructions: list[Instruction],
        offset_to_index: dict[int, int],
        cursor: int,
        end_index: int,
    ) -> int | None:
        advanced = self.try_translate_async_inlined_comprehension(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )
        if advanced is not None:
            return advanced
        return self.try_translate_async_for_loop(
            instructions,
            offset_to_index,
            cursor,
            end_index,
        )


def decompile_native_source(
    code: Any,
    version: tuple[int, ...] | None,
    progress: ProgressCallback | None = None,
) -> NativeResult | None:
    return NativeDecompiler(
        code=code,
        version=version,
        is_module=True,
        progress=progress,
    ).decompile()


def add_missing_local_initializers(
    body: list[ast.stmt],
    instructions: list[Instruction],
    code: Any,
) -> list[ast.stmt]:
    missing = missing_loaded_stored_locals(body, instructions, code)
    if not missing:
        return body

    initializers = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Constant(value=None),
        )
        for name in sorted(missing)
    ]
    return [*initializers, *body]


def missing_loaded_stored_locals(
    body: list[ast.stmt],
    instructions: list[Instruction],
    code: Any,
) -> set[str]:
    local_names = {
        str(name)
        for name in getattr(code, "co_varnames", ())
        if isinstance(name, str) and name.isidentifier()
    }
    if not local_names:
        return set()

    bytecode_stores = {
        str(instruction.argval)
        for instruction in instructions
        if instruction.opname in {"STORE_FAST", "STORE_NAME"}
        and isinstance(instruction.argval, str)
    }
    if not bytecode_stores:
        return set()

    loaded: set[str] = set()
    assigned: set[str] = set()
    for node in walk_current_scope_nodes(body):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            assigned.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            assigned.add(node.rest)
    return (loaded & bytecode_stores & local_names) - assigned


def walk_current_scope_nodes(body: list[ast.stmt]) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    work: list[ast.AST] = list(reversed(body))
    while work:
        node = work.pop()
        nodes.append(node)
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.FunctionDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            continue
        work.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


CO_FUTURE_UNICODE_LITERALS = 0x20000

STORE_TARGET_PREFIX_OPS = {
    "BINARY_SUBSCR",
    "CACHE",
    "LOAD_ATTR",
    "LOAD_CONST",
    "LOAD_FAST",
    "LOAD_GLOBAL",
    "LOAD_NAME",
}


def has_following_subscript_stores(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    count: int,
) -> bool:
    stores = 0
    for index in range(start_index, end_index):
        opname = instructions[index].opname
        if opname == "STORE_SUBSCR":
            stores += 1
            if stores == count:
                return True
            continue
        if opname in STORE_TARGET_PREFIX_OPS:
            continue
        return False
    return False


GLOBAL_CHAIN_PREFIX_OPS = {"CACHE", "EXTENDED_ARG", "NOP", "SET_LINENO"}
STORE_CHAIN_OPS = {
    "STORE_ATTR",
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_GLOBAL",
    "STORE_NAME",
    "STORE_SUBSCR",
}


def should_skip_dead_jump_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index:
        return False

    instruction = instructions[cursor]
    if instruction.is_jump_target:
        return False
    if not is_terminal_op(instructions[cursor - 1].opname):
        return False
    return is_jump_op(instruction.opname)


def should_skip_generator_prologue_pop(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor != start_index + 1:
        return False
    if instructions[cursor].opname != "POP_TOP":
        return False
    return instructions[start_index].opname == "RETURN_GENERATOR"


def should_skip_iterator_cleanup_before_return(
    instructions: list[Instruction],
    cursor: int,
) -> bool:
    instruction = instructions[cursor]
    if instruction.opname == "SWAP":
        return should_skip_swap_iterator_cleanup(instructions, cursor)

    if instruction.opname != "POP_TOP":
        return False

    swap_index = previous_non_metadata_index(instructions, cursor - 1)
    if swap_index is None:
        return False

    swap = instructions[swap_index]
    if swap.opname != "SWAP" or int(swap.arg or 0) != 2:
        return False

    return_index = next_non_metadata_index(instructions, cursor + 1)
    return return_index is not None and instructions[return_index].opname in {
        "RETURN_VALUE",
        "RETURN_CONST",
    }


def should_skip_swap_iterator_cleanup(
    instructions: list[Instruction],
    cursor: int,
) -> bool:
    instruction = instructions[cursor]
    if int(instruction.arg or 0) != 2:
        return False

    pop_index = next_non_metadata_index(instructions, cursor + 1)
    if pop_index is None or instructions[pop_index].opname != "POP_TOP":
        return False

    return_index = next_non_metadata_index(instructions, pop_index + 1)
    return return_index is not None and instructions[return_index].opname in {
        "RETURN_VALUE",
        "RETURN_CONST",
    }


def should_skip_range_cursor(
    decompiler: NativeDecompiler,
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    return (
        (
            not decompiler.is_module
            and should_skip_unreachable_after_terminal(
                instructions,
                cursor,
                start_index,
            )
        )
        or should_skip_dead_jump_after_terminal(instructions, cursor, start_index)
        or should_skip_generator_prologue_pop(instructions, cursor, start_index)
        or should_skip_iterator_cleanup_before_return(instructions, cursor)
        or decompiler.should_skip_loop_break_cleanup_pop(instructions, cursor)
    )


def should_skip_unreachable_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index or instructions[cursor].is_jump_target:
        return False

    for index in range(cursor - 1, start_index - 1, -1):
        instruction = instructions[index]
        if instruction.is_jump_target:
            return False
        if is_terminal_op(instruction.opname):
            return True
    return False


def is_after_implicit_module_return(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
) -> bool:
    if cursor <= start_index:
        return False
    previous_index = previous_non_metadata_index(instructions, cursor - 1)
    if previous_index is None:
        return False
    previous = instructions[previous_index]
    return previous.opname == "RETURN_CONST" and previous.argval is None


def should_skip_returning_with_cleanup_after_terminal(
    instructions: list[Instruction],
    cursor: int,
    start_index: int,
    end_index: int,
) -> int | None:
    if cursor <= start_index:
        return None
    previous_index = previous_non_metadata_index(instructions, cursor - 1)
    if previous_index is None:
        return None
    if not is_terminal_op(instructions[previous_index].opname):
        return None

    skipped = skip_returning_with_exception_handler(instructions, cursor, end_index)
    if skipped == cursor:
        return None
    return skipped


def previous_non_metadata_index(
    instructions: list[Instruction],
    cursor: int,
) -> int | None:
    while cursor >= 0:
        if instructions[cursor].opname not in {"CACHE", "SET_LINENO", "NOP"}:
            return cursor
        cursor -= 1
    return None


def find_loaded_global_store_sources(instructions: list[Instruction]) -> list[str]:
    names: list[str] = []
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_GLOBAL":
            continue
        next_index = next_non_metadata_index(instructions, index + 1)
        if next_index is None:
            continue
        if starts_global_store_chain(instructions, next_index):
            name = str(instruction.argval)
            if name not in names:
                names.append(name)
    return names


def find_explicit_nonlocal_names(
    code: Any,
    instructions: list[Instruction],
) -> list[str]:
    freevars = set(getattr(code, "co_freevars", ()) or ())
    if not freevars:
        return []

    names: list[str] = []
    for instruction in instructions:
        if instruction.opname not in {"DELETE_DEREF", "STORE_DEREF"}:
            continue
        name = str(instruction.argval)
        if name in freevars and name not in names:
            names.append(name)
    return names


def starts_global_store_chain(
    instructions: list[Instruction],
    start_index: int,
) -> bool:
    for index in range(start_index, len(instructions)):
        opname = instructions[index].opname
        if opname in GLOBAL_CHAIN_PREFIX_OPS:
            continue
        if opname == "DUP_TOP":
            continue
        return opname == "STORE_GLOBAL"
    return False


def next_non_metadata_index(
    instructions: list[Instruction],
    start_index: int,
) -> int | None:
    for index in range(start_index, len(instructions)):
        if instructions[index].opname not in GLOBAL_CHAIN_PREFIX_OPS:
            return index
    return None


def find_module_explicit_global_names(instructions: list[Instruction]) -> list[str]:
    names: list[str] = []
    chain: list[Instruction] = []
    for instruction in instructions:
        if instruction.opname in GLOBAL_CHAIN_PREFIX_OPS:
            continue
        if instruction.opname == "DUP_TOP":
            continue
        if instruction.opname in STORE_CHAIN_OPS:
            chain.append(instruction)
            continue
        add_mixed_store_global_names(names, chain)
        chain = []
    add_mixed_store_global_names(names, chain)
    return names


def add_mixed_store_global_names(
    names: list[str],
    chain: list[Instruction],
) -> None:
    if not chain:
        return
    has_name_store = any(instruction.opname == "STORE_NAME" for instruction in chain)
    if not has_name_store:
        return
    for instruction in chain:
        if instruction.opname != "STORE_GLOBAL":
            continue
        name = str(instruction.argval)
        if name not in names:
            names.append(name)
