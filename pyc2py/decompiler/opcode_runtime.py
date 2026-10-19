from __future__ import annotations

import ast
from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from pyc2py.astree import (
    is_none_constant,
    make_assertion_error_name,
    make_name,
    mark_default_lines,
    mark_line,
    safe_identifier,
)
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.decompiler.context import UnpackGroup, UnpackSlot
from pyc2py.decompiler.opcode_access_runtime import OpcodeAccessRuntimeMixin
from pyc2py.types import BuiltinName

if TYPE_CHECKING:
    from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcode_call_runtime import (
    OpcodeCallRuntimeMixin,
    unbound_type_param,
)
from pyc2py.decompiler.opcodes.builders import (
    append_to_container,
    extend_container_literal,
    make_const_key_map,
    make_map_from_stack_items,
    make_sequence,
    make_unpack_map,
    make_unpacked_sequence,
    map_add_to_container,
    update_dict_literal_or_statement,
)
from pyc2py.decompiler.opcodes.dispatch import dispatch_instruction
from pyc2py.decompiler.opcodes.imports_calls import (
    ImportedAttributeValue,
    ImportValue,
    make_import_from_statement,
    make_import_statement,
)
from pyc2py.decompiler.opcodes.values import (
    BINARY_OPS,
    COMPARE_OPS,
    LEGACY_BINARY_OPS,
    conversion_for_format_flags,
    make_constant,
    make_converted_formatted_value,
    make_formatted_with_spec,
    make_joined_formatted_value,
)
from pyc2py.decompiler.recover import (
    make_format_spec,
    make_joined_string_parts,
)
from pyc2py.decompiler.runtime import (
    BuildClassValue,
    ClassValue,
    FunctionValue,
    HiddenLocalRestore,
    TypeAliasValue,
    coerce_expr,
    make_class_def,
    make_function_def,
    make_lambda_expr,
    map_add_key_is_on_top,
    parse_body_or_empty,
    unpack_counts,
    unwrap_lazy_expr,
)
from pyc2py.decompiler.structures import unwrap_truth_test

STORE_NAME_HANDLED = object()
LINE_START_LOADS = {
    "LOAD_CONST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_GLOBAL",
    "LOAD_NAME",
    "LOAD_SMALL_INT",
}


class OpcodeRuntimeMixin(OpcodeCallRuntimeMixin, OpcodeAccessRuntimeMixin, ABC):
    def run_instruction(self, instruction: Instruction) -> None:
        if instruction.starts_line is not None:
            self.flush_legacy_line_boundary()

        dispatch_instruction(self, instruction)

        if instruction.opname in LINE_START_LOADS and self.stack:
            value = self.stack[-1]
            if isinstance(value, (ast.Constant, ast.Name, ast.Tuple)):
                mark_line(value, instruction.starts_line)

        self.note_nop_line(instruction)

    def note_nop_line(self, instruction: Instruction) -> None:
        if instruction.opname in {"CACHE", "EXTENDED_ARG"}:
            return

        if instruction.opname == "NOP":
            if instruction.starts_line is not None:
                self.nop_lines.append(instruction.starts_line)

            return

        folded = instruction.opname == "LOAD_CONST" and isinstance(
            instruction.argval, tuple
        )
        if folded and self.nop_lines and self.stack:
            mark_default_lines(self.stack[-1], self.nop_lines, instruction.starts_line)

        self.nop_lines = []

    def build_class_marker(self) -> BuildClassValue:
        return BuildClassValue()

    def store_name(self, name: str, opname: str) -> None:
        if opname == "STORE_GLOBAL":
            self.add_global_name(safe_identifier(name))

        value = self.release_copied_store(self.pop_or_none())
        if self.keep_copied_store(name, value):
            return

        result = self.store_name_result(name, opname, value)
        if result is STORE_NAME_HANDLED:
            if self.pending_simultaneous_store_count > 0:
                self.pending_simultaneous_store_count -= 1

            return

        if result is None:
            target = make_name(name, ast.Store())
            result = ast.Assign(targets=[target], value=coerce_expr(value))

        if not isinstance(result, ast.stmt):
            raise TypeError("name store did not produce a statement")

        self.append_store_statement(result)

    def store_name_result(
        self,
        name: str,
        opname: str,
        value: Any,
    ) -> ast.stmt | object | None:
        if isinstance(value, HiddenLocalRestore):
            return self.store_hidden_local_restore(name, opname, value)

        if isinstance(value, UnpackSlot):
            self.store_unpack_slot(value, name)
            return STORE_NAME_HANDLED

        return make_named_value_statement(self.new_child, name, value, self.version)

    def store_hidden_local_restore(
        self,
        name: str,
        opname: str,
        value: HiddenLocalRestore,
    ) -> object:
        if opname == "STORE_FAST":
            return STORE_NAME_HANDLED
        if safe_identifier(name) != value.name:
            self.warnings.append(
                f"hidden local restore target changed: {value.name} -> {name}"
            )

        return STORE_NAME_HANDLED

    def load_fast_load_fast(self, names: Any) -> None:
        first_name, second_name = self.unpack_local_pair(names, "LOAD_FAST_LOAD_FAST")
        self.stack.append(make_name(first_name, ast.Load()))
        self.stack.append(make_name(second_name, ast.Load()))

    def load_const_load_fast(self, values: Any) -> None:
        const_value, local_name = self.unpack_value_pair(values, "LOAD_CONST_LOAD_FAST")
        self.stack.append(make_constant(const_value))
        self.stack.append(make_name(str(local_name), ast.Load()))

    def load_fast_load_const(self, values: Any) -> None:
        local_name, const_value = self.unpack_value_pair(values, "LOAD_FAST_LOAD_CONST")
        self.stack.append(make_name(str(local_name), ast.Load()))
        self.stack.append(make_constant(const_value))

    def load_common_constant(self, value: Any) -> None:
        if value == "AssertionError":
            self.stack.append(make_assertion_error_name())
            return

        if isinstance(value, BuiltinName):
            self.stack.append(make_name(str(value), ast.Load()))
            return

        self.warnings.append(f"unsupported LOAD_COMMON_CONSTANT value: {value}")
        self.stack.append(ast.Constant(value=None))

    def store_fast_load_fast(self, names: Any) -> None:
        store_name, load_name = self.unpack_local_pair(names, "STORE_FAST_LOAD_FAST")
        self.store_name(store_name, "STORE_FAST")
        self.stack.append(make_name(load_name, ast.Load()))

    def store_fast_store_fast(self, names: Any) -> None:
        first_name, second_name = self.unpack_local_pair(names, "STORE_FAST_STORE_FAST")
        self.store_name(first_name, "STORE_FAST")
        self.store_name(second_name, "STORE_FAST")

    def unpack_local_pair(self, names: Any, opname: str) -> tuple[str, str]:
        if isinstance(names, tuple):
            items = cast("tuple[object, ...]", names)
            if len(items) == 2:
                return str(items[0]), str(items[1])

        self.warnings.append(f"{opname} did not resolve to two local names")
        return "value", "value"

    def unpack_value_pair(self, values: Any, opname: str) -> tuple[Any, Any]:
        if isinstance(values, tuple):
            items = cast("tuple[Any, ...]", values)
            if len(items) == 2:
                return items

        self.warnings.append(f"{opname} did not resolve to two values")
        return ast.Constant(value=None), "value"

    def delete_name(self, name: str, opname: str) -> None:
        if opname == "DELETE_GLOBAL":
            self.add_global_name(safe_identifier(name))

        target = make_name(name, ast.Del())
        self.statements.append(ast.Delete(targets=[target]))

    def store_unpack_slot(self, slot: UnpackSlot, name: str) -> None:
        self.store_unpack_target(slot, make_name(name, ast.Store()))

    def store_unpack_target(self, slot: object, target: ast.expr) -> None:
        if not isinstance(slot, UnpackSlot):
            return
        if slot.group.starred_index == slot.index:
            target = ast.Starred(value=target, ctx=ast.Store())

        slot.group.targets[slot.index] = target
        if any(target is None for target in slot.group.targets):
            return

        targets = [target for target in slot.group.targets if target is not None]
        name_dropped_unpack_targets(targets)

        if slot.group.parent is not None:
            self.store_unpack_target(
                slot.group.parent,
                ast.Tuple(elts=targets, ctx=ast.Store()),
            )
            return

        self.statements.append(
            ast.Assign(
                targets=[ast.Tuple(elts=targets, ctx=ast.Store())],
                value=slot.group.value,
            )
        )

    def pop_expression_statement(self) -> None:
        value = self.pop_or_none()
        if isinstance(value, UnpackSlot):
            self.store_unpack_target(value, make_dropped_unpack_target())
            return

        if self.pending_print_items or self.pending_print_target is not None:
            self.flush_print_items(newline=False)
            return

        if value is None or isinstance(value, (ImportedAttributeValue, ImportValue)):
            return

        expression = coerce_expr(value)
        if is_none_constant(expression):
            return

        self.statements.append(ast.Expr(value=expression))

    def return_value(self, instruction: Instruction) -> None:
        if self.loop_none_return_is_break and self.is_none_return(instruction):
            self.stack.clear()
            self.statements.append(ast.Break())
            return

        if self.is_module:
            self.stack.clear()
            return

        if normalized_opcode_name(instruction.opname) == "RETURN_CONST":
            value = make_constant(instruction.argval)
        else:
            value = self.expression_from_stack_value(self.pop_or_none())

        statement = ast.Return(value=value)

        if instruction.offset in self.implicit_return_offsets:
            setattr(statement, "_pyc2py_implicit_return", True)
        elif (
            instruction.opname == "RETURN_CONST" and instruction.is_jump_target
        ) or instruction.offset in self.written_return_offsets:
            setattr(statement, "_pyc2py_written_return", True)

        self.statements.append(statement)

    def is_none_return(self, instruction: Instruction) -> bool:
        if normalized_opcode_name(instruction.opname) == "RETURN_CONST":
            return instruction.argval is None
        if not self.stack:
            return False

        return is_none_constant(coerce_expr(self.stack[-1]))

    def prep_reraise_star(self) -> None:
        raised_and_reraised = self.expression_from_stack_value(self.pop_or_none())
        original_group = self.expression_from_stack_value(self.pop_or_none())
        self.prep_reraise_star_from_values(original_group, raised_and_reraised)

    def push_exc_info(self) -> None:
        exception = self.expression_from_stack_value(self.pop_or_none())
        self.warnings.append("PUSH_EXC_INFO represented with current-exception helper")
        self.stack.append(make_name("__pyc2py_current_exception__", ast.Load()))
        self.stack.append(exception)

    def check_exc_match(self) -> None:
        match_type = self.expression_from_stack_value(self.pop_or_none())

        if not self.stack:
            self.warnings.append("CHECK_EXC_MATCH on shallow stack")
            self.stack.append(ast.Constant(value=False))
            return

        exception = self.expression_from_stack_value(self.stack[-1])
        self.warnings.append("CHECK_EXC_MATCH represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_check_exc_match__", ast.Load()),
                args=[exception, match_type],
                keywords=[],
            )
        )

    def check_eg_match(self) -> None:
        match_type = self.expression_from_stack_value(self.pop_or_none())

        if not self.stack:
            self.warnings.append("CHECK_EG_MATCH on shallow stack")
            self.stack.append(ast.Constant(value=None))
            return

        exception_group = self.expression_from_stack_value(self.pop_or_none())
        self.warnings.append("CHECK_EG_MATCH represented as helper call")
        result = ast.Call(
            func=make_name("__pyc2py_check_eg_match__", ast.Load()),
            args=[exception_group, match_type],
            keywords=[],
        )
        self.stack.append(
            ast.Subscript(
                value=result,
                slice=ast.Constant(value=0),
                ctx=ast.Load(),
            )
        )
        self.stack.append(
            ast.Subscript(
                value=result,
                slice=ast.Constant(value=1),
                ctx=ast.Load(),
            )
        )

    def prep_reraise_star_from_values(
        self,
        original_group: ast.expr,
        raised_and_reraised: ast.expr,
    ) -> None:
        self.warnings.append("PREP_RERAISE_STAR represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_prep_reraise_star__", ast.Load()),
                args=[original_group, raised_and_reraised],
                keywords=[],
            )
        )

    def unpack_sequence(self, instruction: Instruction) -> None:
        raw_value = self.pop_or_none()
        value = coerce_expr(raw_value)
        count, starred_index = unpack_counts(instruction)
        if count == 0:
            self.statements.append(
                ast.Assign(
                    targets=[ast.List(elts=[], ctx=ast.Store())],
                    value=value,
                )
            )
            return

        group = UnpackGroup(
            value=value,
            targets=[None] * count,
            starred_index=starred_index,
            parent=raw_value if isinstance(raw_value, UnpackSlot) else None,
        )
        self.unpack_groups.append(group)

        for index in reversed(range(count)):
            self.stack.append(UnpackSlot(group=group, index=index))

    def build_sequence(self, opname: str, count: int) -> None:
        values = [
            self.expression_from_stack_value(value)
            for value in self.pop_many_raw(count)
        ]
        self.stack.append(make_sequence(opname, values))

    def build_sequence_unpack(self, opname: str, count: int) -> None:
        values = self.pop_many(count)
        self.stack.append(make_unpacked_sequence(opname, values))

    def build_string(self, count: int) -> None:
        values = self.pop_many(count)
        parts: list[ast.expr] = []

        for value in values:
            parts.extend(make_joined_string_parts(value))

        self.stack.append(ast.JoinedStr(values=parts))

    def build_interpolation(self, arg: int) -> None:
        format_value: ast.expr = ast.Constant(value=None)

        if arg & 1:
            format_value = coerce_expr(self.pop_or_none())

        string_value = coerce_expr(self.pop_or_none())
        value = coerce_expr(self.pop_or_none())
        interpolation = make_interpolation(value, string_value, arg, format_value)
        if interpolation is not None:
            self.stack.append(interpolation)
            return

        self.warnings.append("BUILD_INTERPOLATION represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_build_interpolation__", ast.Load()),
                args=[
                    value,
                    string_value,
                    ast.Constant(value=arg >> 2),
                    format_value,
                ],
                keywords=[],
            )
        )

    def build_template(self) -> None:
        interpolations = coerce_expr(self.pop_or_none())
        strings = coerce_expr(self.pop_or_none())
        template = make_template(strings, interpolations)
        if template is not None:
            self.stack.append(template)
            return

        self.warnings.append("BUILD_TEMPLATE represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_build_template__", ast.Load()),
                args=[strings, interpolations],
                keywords=[],
            )
        )

    def build_map(self, count: int) -> None:
        if count == 0:
            self.stack.append(ast.Dict(keys=[], values=[]))
            return

        values = self.pop_many(count * 2)
        self.stack.append(make_map_from_stack_items(values))

    def build_map_unpack(self, count: int) -> None:
        values = self.pop_many(count)
        self.stack.append(make_unpack_map(values))

    def build_const_key_map(self, count: int) -> None:
        keys = self.pop_or_none()
        values = self.pop_many(count)
        mapping = make_const_key_map(coerce_expr(keys), values, count)
        if mapping is None:
            self.warnings.append("BUILD_CONST_KEY_MAP keys were not a matching tuple")
            self.stack.append(ast.Dict(keys=[], values=[]))
            return

        self.stack.append(mapping)

    def store_map(self) -> None:
        key = coerce_expr(self.pop_or_none())
        value = coerce_expr(self.pop_or_none())
        mapping = self.pop_or_none()
        if not isinstance(mapping, ast.Dict):
            self.warnings.append("STORE_MAP target was not a dict")
            self.stack.append(coerce_expr(mapping))
            return

        mapping.keys.append(key)
        mapping.values.append(value)
        self.stack.append(mapping)

    def list_append(self, opcode_depth: int) -> None:
        value = self.expression_from_stack_value(self.pop_or_none())
        container = self.peek_container(opcode_depth, "LIST_APPEND")
        if container is None:
            return

        statement = append_to_container(coerce_expr(container), value, "append")
        if statement is not None:
            self.statements.append(statement)

    def set_add(self, opcode_depth: int) -> None:
        value = self.expression_from_stack_value(self.pop_or_none())
        container = self.peek_container(opcode_depth, "SET_ADD")
        if container is None:
            return

        statement = append_to_container(coerce_expr(container), value, "add")
        if statement is not None:
            self.statements.append(statement)

    def map_add(self, opcode_depth: int) -> None:
        if map_add_key_is_on_top(self.version):
            key = self.expression_from_stack_value(self.pop_or_none())
            value = self.expression_from_stack_value(self.pop_or_none())
        else:
            value = self.expression_from_stack_value(self.pop_or_none())
            key = self.expression_from_stack_value(self.pop_or_none())

        container = self.peek_container(opcode_depth, "MAP_ADD")
        if container is None:
            return

        statement = map_add_to_container(coerce_expr(container), key, value)
        if statement is not None:
            self.statements.append(statement)

    def extend_container(self, opname: str, opcode_depth: int) -> None:
        iterable = self.expression_from_stack_value(self.pop_or_none())
        container = self.peek_container(opcode_depth, opname)
        if container is None:
            return

        statement = extend_container_literal(opname, coerce_expr(container), iterable)
        if statement is not None:
            self.statements.append(statement)

    def update_dict(self, opname: str, opcode_depth: int) -> None:
        mapping = self.expression_from_stack_value(self.pop_or_none())
        container = self.peek_container(opcode_depth, opname)
        if container is None:
            return

        statement = update_dict_literal_or_statement(coerce_expr(container), mapping)
        if statement is not None:
            self.statements.append(statement)

    def list_to_tuple(self) -> None:
        value = self.expression_from_stack_value(self.pop_or_none())
        if isinstance(value, ast.List):
            self.stack.append(ast.Tuple(elts=value.elts, ctx=ast.Load()))
            return

        self.stack.append(
            ast.Call(
                func=make_name("tuple", ast.Load()),
                args=[value],
                keywords=[],
            )
        )

    def peek_container(self, opcode_depth: int, opname: str) -> Any | None:
        if opcode_depth < 1:
            self.warnings.append(f"{opname} with invalid depth: {opcode_depth}")
            return None

        depth = opcode_depth - 1
        if len(self.stack) <= depth:
            self.warnings.append(f"{opname}_{opcode_depth} on shallow stack")
            return None

        return self.stack[-opcode_depth]

    def binary_op(self, symbol: str) -> None:
        is_inplace = symbol.endswith("=")
        op_symbol = symbol[:-1] if is_inplace else symbol
        op_type = BINARY_OPS.get(op_symbol)
        if op_type is None:
            self.warnings.append(f"unsupported binary operator: {symbol}")
            self.stack.clear()
            return

        right, left = self.pop_binary()
        expression = ast.BinOp(left=left, op=op_type(), right=right)

        if is_inplace:
            setattr(expression, "_pyc2py_inplace", True)

        self.stack.append(expression)

    def legacy_binary_op(self, opname: str) -> None:
        right, left = self.pop_binary()
        expression = ast.BinOp(left=left, op=LEGACY_BINARY_OPS[opname](), right=right)

        if opname.startswith("INPLACE_"):
            setattr(expression, "_pyc2py_inplace", True)

        self.stack.append(expression)

    def unary_op(self, op: ast.unaryop) -> None:
        value = coerce_expr(self.pop_or_none())

        if isinstance(op, ast.Not):
            value = unwrap_truth_test(value)

        self.stack.append(ast.UnaryOp(op=op, operand=value))

    def compare_op(self, symbol: str) -> None:
        op_type = COMPARE_OPS.get(symbol)
        if op_type is None:
            self.warnings.append(f"unsupported comparison operator: {symbol}")
            self.stack.clear()
            return

        right, left = self.pop_binary()
        if isinstance(left, ast.Compare) and getattr(left, "_pyc2py_chain", None):
            self.stack.append(
                ast.Compare(
                    left=left.left,
                    ops=[*left.ops, op_type()],
                    comparators=[*left.comparators, right],
                )
            )
            return

        compare = ast.Compare(left=left, ops=[op_type()], comparators=[right])
        self.stack.append(self.join_compare_chain(compare))

    def chained_compare_link(self, last_link_op: str | None) -> None:
        comparison = unwrap_truth_test(coerce_expr(self.pop_or_none()))
        self.pop_or_none()
        setattr(comparison, "_pyc2py_chain", last_link_op or "open")
        self.stack.append(comparison)

    def format_value(self, flags: int) -> None:
        format_spec = None

        if flags & 0x04:
            format_spec = make_format_spec(coerce_expr(self.pop_or_none()))

        value = coerce_expr(self.pop_or_none())
        self.stack.append(
            make_joined_formatted_value(
                value, conversion_for_format_flags(flags), format_spec
            )
        )

    def convert_value(self, arg: int) -> None:
        value = coerce_expr(self.pop_or_none())
        self.stack.append(make_converted_formatted_value(value, arg))

    def format_simple(self) -> None:
        value = coerce_expr(self.pop_or_none())
        if not isinstance(value, ast.FormattedValue):
            value = ast.FormattedValue(value=value, conversion=-1, format_spec=None)

        self.stack.append(ast.JoinedStr(values=[value]))

    def format_with_spec(self) -> None:
        format_spec = make_format_spec(coerce_expr(self.pop_or_none()))
        value = coerce_expr(self.pop_or_none())
        self.stack.append(make_formatted_with_spec(value, format_spec))

    def unary_convert(self) -> None:
        value = coerce_expr(self.pop_or_none())
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_backtick__", ast.Load()),
                args=[value],
                keywords=[],
            )
        )

    def to_bool(self) -> None:
        value = coerce_expr(self.pop_or_none())
        expression = ast.Call(
            func=make_name("bool", ast.Load()),
            args=[value],
            keywords=[],
        )
        setattr(expression, "_pyc2py_truth_test", True)
        self.stack.append(expression)


def make_named_value_statement(
    new_child: Callable[[object], DecompilerContext],
    name: str,
    value: Any,
    version: tuple[int, ...] | None,
) -> ast.stmt | None:
    if isinstance(value, FunctionValue):
        return make_function_store_statement(new_child, name, value, version)
    if isinstance(value, TypeAliasValue):
        return make_type_alias_statement(name, value)
    if isinstance(value, ClassValue):
        return make_class_store_statement(new_child, name, value, version)
    if isinstance(value, ImportValue):
        return make_import_statement(name, value)
    if isinstance(value, ImportedAttributeValue):
        return make_import_from_statement(name, value)

    return make_inplace_name_assignment(name, value)


def make_function_store_statement(
    new_child: Callable[[object], DecompilerContext],
    name: str,
    value: FunctionValue,
    version: tuple[int, ...] | None,
) -> ast.stmt:
    if getattr(value.code, "co_name", "") == "<lambda>":
        return ast.Assign(
            targets=[make_name(name, ast.Store())],
            value=make_lambda_expr(
                new_child,
                value.code,
                version,
                value.defaults,
                value.kw_defaults,
            ),
        )

    type_alias = make_type_alias_from_generic_function(new_child, name, value, version)
    if type_alias is not None:
        return type_alias

    return make_function_def(
        new_child,
        name,
        value.code,
        version,
        value.defaults,
        value.kw_defaults,
        value.annotations,
        value.decorators,
        tuple(type_alias_params(value.type_params) or ()),
        value.annotate,
    )


def make_class_store_statement(
    new_child: Callable[[object], DecompilerContext],
    name: str,
    value: ClassValue,
    version: tuple[int, ...] | None,
) -> ast.stmt:
    return make_class_def(
        new_child,
        name,
        value.code,
        version,
        value.bases,
        tuple(type_alias_params(value.type_params) or ()),
        value.keywords,
        value.decorators,
    )


def make_inplace_name_assignment(name: str, value: Any) -> ast.AugAssign | None:
    if not getattr(value, "_pyc2py_inplace", False):
        return None
    if not isinstance(value, ast.BinOp):
        return None
    if not isinstance(value.left, ast.Name):
        return None

    target_name = safe_identifier(name)
    if value.left.id != target_name:
        return None

    return ast.AugAssign(
        target=make_name(target_name, ast.Store()),
        op=type(value.op)(),
        value=value.right,
    )


def make_type_alias_statement(name: str, value: TypeAliasValue) -> ast.stmt:
    params = type_alias_params(value.type_params) or []
    return make_type_alias_statement_with_params(name, value, params)


def make_type_alias_statement_with_params(
    name: str,
    value: TypeAliasValue,
    params: list[ast.type_param],
) -> ast.stmt:
    alias_name = value.name

    if safe_identifier(name) != value.name:
        alias_name = safe_identifier(name)

    return ast.TypeAlias(
        name=ast.Name(id=alias_name, ctx=ast.Store()),
        type_params=params,
        value=value.value,
    )


def make_type_alias_from_generic_function(
    new_child: Callable[[object], DecompilerContext],
    name: str,
    value: FunctionValue,
    version: tuple[int, ...] | None,
) -> ast.stmt | None:
    code_name = str(getattr(value.code, "co_name", ""))
    if not code_name.startswith("<generic parameters of "):
        return None

    result = new_child(value.code).decompile()
    module = parse_body_or_empty(result)
    for statement in module.body:
        if not isinstance(statement, ast.Return):
            continue

        alias = type_alias_value_from_call(statement.value)
        if alias is None:
            continue

        params = type_alias_params(alias.type_params)
        if params is None:
            continue
        if contains_unsupported_type_alias_expr(alias.value):
            continue

        return make_type_alias_statement_with_params(name, alias, params)

    return None


def type_alias_value_from_call(value: ast.expr | None) -> TypeAliasValue | None:
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Name) or value.func.id != "TypeAliasType":
        return None
    if len(value.args) < 2:
        return None

    name = value.args[0]
    if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
        return None

    return TypeAliasValue(
        name=name.value,
        value=value.args[1],
        type_params=type_alias_call_params(value),
    )


def type_alias_call_params(value: ast.Call) -> tuple[ast.expr, ...]:
    for keyword in value.keywords:
        if keyword.arg != "type_params":
            continue
        if isinstance(keyword.value, ast.Tuple):
            return tuple(keyword.value.elts)

    return ()


def contains_unsupported_type_alias_expr(value: ast.AST) -> bool:
    for node in ast.walk(value):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "tuple":
            continue
        if node.args and isinstance(node.args[0], ast.List):
            return True

    return False


def type_alias_params(
    type_params: tuple[ast.expr, ...],
) -> list[ast.type_param] | None:
    params: list[ast.type_param] = []

    for param in type_params:
        converted = type_alias_param(param)
        if converted is None:
            return None

        params.append(converted)

    return params


def type_alias_param(param: ast.expr) -> ast.type_param | None:
    parsed = type_alias_param_call(param)
    if parsed is None:
        return None

    func_name, name, call = parsed
    if has_type_param_default(call):
        return None

    return make_type_alias_param_node(func_name, name, call)


def type_alias_param_call(param: ast.expr) -> tuple[str, str, ast.Call] | None:
    param = unbound_type_param(param)
    if not isinstance(param, ast.Call):
        return None
    if not isinstance(param.func, ast.Name):
        return None
    if not param.args:
        return None

    name_value = param.args[0]
    if not isinstance(name_value, ast.Constant) or not isinstance(
        name_value.value, str
    ):
        return None

    return param.func.id, name_value.value, param


def make_type_alias_param_node(
    func_name: str,
    name: str,
    param: ast.Call,
) -> ast.type_param | None:
    if func_name == "TypeVar":
        return make_type_var_param(name, param)
    if func_name == "ParamSpec":
        return ast.ParamSpec(name=name)
    if func_name == "TypeVarTuple":
        return ast.TypeVarTuple(name=name)

    return None


def make_type_var_param(name: str, param: ast.Call) -> ast.type_param | None:
    if has_unsupported_type_param_constraints(param):
        return None

    return ast.TypeVar(
        name=name,
        bound=type_param_bound(param),
    )


def has_type_param_default(param: ast.Call) -> bool:
    return any(keyword.arg == "default" for keyword in param.keywords)


def has_unsupported_type_param_constraints(param: ast.Call) -> bool:
    return len(param.args) > 1 and any(
        type_param_constraint_expr(arg) is None for arg in param.args[1:]
    )


def type_param_bound(param: ast.Call) -> ast.expr | None:
    for keyword in param.keywords:
        if keyword.arg != "bound":
            continue

        return unwrap_lazy_expr(keyword.value)

    if len(param.args) > 1:
        return ast.Tuple(
            elts=[
                constraint
                for arg in param.args[1:]
                for constraint in type_param_constraint_values(arg)
            ],
            ctx=ast.Load(),
        )

    return None


def type_param_constraint_values(value: ast.expr) -> list[ast.expr]:
    constraint = type_param_constraint_expr(value)
    if constraint is None:
        return []
    if isinstance(constraint, ast.Tuple):
        return list(constraint.elts)

    return [constraint]


def type_param_constraint_expr(value: ast.expr) -> ast.expr | None:
    if isinstance(value, ast.Starred):
        value = value.value
    if isinstance(value, ast.Lambda):
        return value.body

    return value


def make_dropped_unpack_target() -> ast.Name:
    target = ast.Name(id="_", ctx=ast.Store())
    setattr(target, "_pyc2py_dropped_store", True)
    return target


def name_dropped_unpack_targets(targets: list[ast.expr]) -> None:
    following: str | None = None

    for target in reversed(targets):
        if not isinstance(target, ast.Name):
            continue
        if getattr(target, "_pyc2py_dropped_store", False) and following is not None:
            target.id = following

        following = target.id


# BUILD_INTERPOLATION oparg: conversion << 2 | has_spec
INTERPOLATION_CONVERSIONS = {0: -1, 1: ord("s"), 2: ord("r"), 3: ord("a")}


def make_interpolation(
    value: ast.expr, text: ast.expr, arg: int, format_value: ast.expr
) -> ast.expr | None:
    conversion = INTERPOLATION_CONVERSIONS.get(arg >> 2)
    if conversion is None or not isinstance(text, ast.Constant):
        return None
    if not isinstance(text.value, str):
        return None

    format_spec: ast.expr | None = None

    if arg & 1:
        if isinstance(format_value, ast.Constant) and isinstance(
            format_value.value, str
        ):
            format_spec = ast.JoinedStr(values=[format_value])
        elif isinstance(format_value, ast.JoinedStr):
            format_spec = format_value
        else:
            return None

    interpolation: ast.expr = ast.Interpolation(
        value=value, str=text.value, conversion=conversion, format_spec=format_spec
    )
    return interpolation


def make_template(strings: ast.expr, interpolations: ast.expr) -> ast.expr | None:
    string_values = constant_strings(strings)
    if string_values is None or not isinstance(interpolations, ast.Tuple):
        return None

    parts = interpolations.elts
    if len(string_values) != len(parts) + 1:
        return None
    if not all(isinstance(part, ast.Interpolation) for part in parts):
        return None

    values: list[ast.expr] = []

    for index, text in enumerate(string_values):
        if text:
            values.append(ast.Constant(value=text))
        if index < len(parts):
            values.append(parts[index])

    template: ast.expr = ast.TemplateStr(values=values)
    return template


def constant_strings(value: ast.expr) -> list[str] | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, tuple):
        items: list[object] = list(value.value)
    elif isinstance(value, ast.Tuple):
        items = [
            item.value if isinstance(item, ast.Constant) else None
            for item in value.elts
        ]
    else:
        return None

    if not all(isinstance(item, str) for item in items):
        return None

    return [str(item) for item in items]
