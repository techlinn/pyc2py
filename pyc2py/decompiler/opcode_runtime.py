from __future__ import annotations

import ast
from typing import Any
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.astree import make_constant, make_name, safe_identifier
from pyc2py.bytecode.instruction import Instruction
from pyc2py.decompiler.opcodes.builders import (
    append_to_container,
    extend_container_literal,
    make_const_key_map,
    make_map_from_stack_items,
    make_unpack_map,
    make_unpacked_sequence,
    make_sequence,
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
    is_none_constant,
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
    UnpackGroup,
    UnpackSlot,
    coerce_expr,
    make_class_def,
    make_function_def,
    make_lambda_expr,
    parse_body_or_empty,
    unpack_counts,
    unwrap_lazy_expr,
)
from pyc2py.decompiler.opcode_access_runtime import OpcodeAccessRuntimeMixin
from pyc2py.decompiler.opcode_call_runtime import OpcodeCallRuntimeMixin

STORE_NAME_HANDLED = object()

class OpcodeRuntimeMixin(OpcodeCallRuntimeMixin, OpcodeAccessRuntimeMixin):
    def run_instruction(self, instruction: Instruction) -> None:
        if instruction.starts_line is not None or instruction.opname == "SET_LINENO":
            self.flush_legacy_line_boundary()
        dispatch_instruction(self, instruction)

    def build_class_marker(self) -> BuildClassValue:
        return BuildClassValue()

    def store_name(self, name: str, opname: str) -> None:
        if opname == "STORE_GLOBAL":
            self.add_global_name(safe_identifier(name))

        value = self.pop_or_none()
        result = self.store_name_result(name, opname, value)

        if result is STORE_NAME_HANDLED:
            return
        if result is None:
            target = make_name(name, ast.Store())
            result = ast.Assign(targets=[target], value=coerce_expr(value))
        self.statements.append(result)

    def store_name_result(
        self,
        name: str,
        opname: str,
        value: Any,
    ) -> ast.stmt | object | None:
        if isinstance(value, HiddenLocalRestore):
            return self.store_hidden_local_restore(name, opname, value)
        if isinstance(value, UnpackSlot):
            self.store_unpack_slot(value, name, opname)
            return STORE_NAME_HANDLED
        return make_named_value_statement(name, value, self.version)

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
        if value in {
            "AssertionError",
            "NotImplementedError",
            "tuple",
            "all",
            "any",
            "list",
            "set",
            "frozenset",
        }:
            self.stack.append(make_name(str(value), ast.Load()))
            return
        if value == ():
            self.stack.append(ast.Tuple(elts=[], ctx=ast.Load()))
            return
        if value in {None, "", True, False, -1}:
            self.stack.append(ast.Constant(value=value))
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
        if isinstance(names, tuple) and len(names) == 2:
            return str(names[0]), str(names[1])
        self.warnings.append(f"{opname} did not resolve to two local names")
        return "value", "value"

    def unpack_value_pair(self, values: Any, opname: str) -> tuple[Any, Any]:
        if isinstance(values, tuple) and len(values) == 2:
            return values
        self.warnings.append(f"{opname} did not resolve to two values")
        return ast.Constant(value=None), "value"

    def delete_name(self, name: str, opname: str) -> None:
        if opname == "DELETE_GLOBAL":
            self.add_global_name(safe_identifier(name))
        target = make_name(name, ast.Del())
        self.statements.append(ast.Delete(targets=[target]))

    def store_unpack_slot(self, slot: "UnpackSlot", name: str, opname: str) -> None:
        target = make_name(name, ast.Store())
        if opname == "STORE_DEREF":
            target._pyc2py_force_tuple_unpack = True
        self.store_unpack_target(slot, target)

    def store_unpack_target(self, slot: "UnpackSlot", target: ast.expr) -> None:
        if slot.group.starred_index == slot.index:
            target = ast.Starred(value=target, ctx=ast.Store())
        slot.group.targets[slot.index] = target

        if any(target is None for target in slot.group.targets):
            return

        targets = [target for target in slot.group.targets if target is not None]
        self.statements.append(
            ast.Assign(
                targets=[make_unpack_assignment_target(targets)],
                value=slot.group.value,
            )
        )

    def pop_expression_statement(self) -> None:
        value = self.pop_or_none()

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
        self.statements.append(ast.Return(value=value))

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
        value = coerce_expr(self.pop_or_none())
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
            value=value, targets=[None] * count, starred_index=starred_index
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
        format_value = ast.Constant(value=None)
        if arg & 1:
            format_value = coerce_expr(self.pop_or_none())
        string_value = coerce_expr(self.pop_or_none())
        value = coerce_expr(self.pop_or_none())

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
        self.warnings.append("BUILD_TEMPLATE represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_build_template__", ast.Load()),
                args=[strings, interpolations],
                keywords=[],
            )
        )

    def build_map(self, count: int) -> None:
        if count == 0 or len(self.stack) < count * 2:
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
        if self.version is not None and self.version < (3, 8):
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
            expression._pyc2py_inplace = True
        self.stack.append(expression)

    def legacy_binary_op(self, opname: str) -> None:
        right, left = self.pop_binary()
        expression = ast.BinOp(left=left, op=LEGACY_BINARY_OPS[opname](), right=right)
        if opname.startswith("INPLACE_"):
            expression._pyc2py_inplace = True
        self.stack.append(expression)

    def unary_op(self, op: ast.unaryop) -> None:
        value = coerce_expr(self.pop_or_none())
        self.stack.append(ast.UnaryOp(op=op, operand=value))

    def compare_op(self, symbol: str, force_bool: bool = False) -> None:
        op_type = COMPARE_OPS.get(symbol)
        if op_type is None:
            self.warnings.append(f"unsupported comparison operator: {symbol}")
            self.stack.clear()
            return

        right, left = self.pop_binary()
        expression: ast.expr = ast.Compare(
            left=left, ops=[op_type()], comparators=[right]
        )
        if force_bool:
            expression = ast.Call(
                func=make_name("bool", ast.Load()),
                args=[expression],
                keywords=[],
            )
        self.stack.append(expression)

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
        self.stack.append(ast.JoinedStr(values=make_joined_string_parts(value)))

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
        expression._pyc2py_truth_test = True
        self.stack.append(expression)

def make_unpack_assignment_target(targets: list[ast.expr]) -> ast.expr:
    if len(targets) != 2:
        return ast.Tuple(elts=targets, ctx=ast.Store())
    if has_attribute_target(targets) or has_forced_tuple_unpack_target(targets):
        return ast.Tuple(elts=targets, ctx=ast.Store())
    return ast.List(elts=targets, ctx=ast.Store())

def make_named_value_statement(
    name: str,
    value: Any,
    version: tuple[int, ...] | None,
) -> ast.stmt | None:
    if isinstance(value, FunctionValue):
        return make_function_store_statement(name, value, version)
    if isinstance(value, TypeAliasValue):
        return make_type_alias_statement(name, value)
    if isinstance(value, ClassValue):
        return make_class_store_statement(name, value, version)
    if isinstance(value, ImportValue):
        return make_import_statement(name, value)
    if isinstance(value, ImportedAttributeValue):
        return make_import_from_statement(name, value)
    return make_inplace_name_assignment(name, value)

def make_function_store_statement(
    name: str,
    value: FunctionValue,
    version: tuple[int, ...] | None,
) -> ast.stmt:
    if getattr(value.code, "co_name", "") == "<lambda>":
        return ast.Assign(
            targets=[make_name(name, ast.Store())],
            value=make_lambda_expr(
                value.code,
                version,
                value.defaults,
                value.kw_defaults,
            ),
        )

    type_alias = make_type_alias_from_generic_function(name, value, version)
    if type_alias is not None:
        return type_alias

    return make_function_def(
        name,
        value.code,
        version,
        value.defaults,
        value.kw_defaults,
        value.annotations,
        value.decorators,
        type_alias_params(value.type_params) or (),
        value.annotate,
    )

def make_class_store_statement(
    name: str,
    value: ClassValue,
    version: tuple[int, ...] | None,
) -> ast.stmt:
    return make_class_def(
        name,
        value.code,
        version,
        value.bases,
        type_alias_params(value.type_params) or (),
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

def has_attribute_target(targets: list[ast.expr]) -> bool:
    return any(contains_attribute_target(target) for target in targets)

def contains_attribute_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Attribute):
        return True
    if isinstance(target, ast.Starred):
        return contains_attribute_target(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return has_attribute_target(target.elts)
    return False

def has_forced_tuple_unpack_target(targets: list[ast.expr]) -> bool:
    return any(contains_forced_tuple_unpack_target(target) for target in targets)

def contains_forced_tuple_unpack_target(target: ast.expr) -> bool:
    if bool(getattr(target, "_pyc2py_force_tuple_unpack", False)):
        return True
    if isinstance(target, ast.Starred):
        return contains_forced_tuple_unpack_target(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return has_forced_tuple_unpack_target(target.elts)
    return False

def make_type_alias_statement(name: str, value: TypeAliasValue) -> ast.stmt:
    params = type_alias_params(value.type_params) or []
    return make_type_alias_statement_with_params(name, value, params)

def make_type_alias_statement_with_params(
    name: str,
    value: TypeAliasValue,
    params: list[ast.AST],
) -> ast.stmt:
    alias_name = value.name
    if safe_identifier(name) != value.name:
        alias_name = safe_identifier(name)
    if not SUPPORTS_TYPE_ALIAS:
        return ast.Assign(
            targets=[ast.Name(id=alias_name, ctx=ast.Store())],
            value=value.value,
        )
    return ast.TypeAlias(
        name=ast.Name(id=alias_name, ctx=ast.Store()),
        type_params=params,
        value=value.value,
    )

def make_type_alias_from_generic_function(
    name: str,
    value: FunctionValue,
    version: tuple[int, ...] | None,
) -> ast.stmt | None:
    code_name = str(getattr(value.code, "co_name", ""))
    if not code_name.startswith("<generic parameters of "):
        return None

    from pyc2py.decompiler.engine import NativeDecompiler

    result = NativeDecompiler(
        code=value.code,
        version=version,
        is_module=False,
    ).decompile()
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

def type_alias_params(type_params: tuple[ast.expr, ...]) -> list[ast.AST] | None:
    params: list[ast.AST] = []
    for param in type_params:
        converted = type_alias_param(param)
        if converted is None:
            return None
        params.append(converted)
    return params

def type_alias_param(param: ast.expr) -> ast.AST | None:
    parsed = type_alias_param_call(param)
    if parsed is None:
        return None
    func_name, name, call = parsed
    if has_type_param_default(call):
        return None
    return make_type_alias_param_node(func_name, name, call)

def type_alias_param_call(param: ast.expr) -> tuple[str, str, ast.Call] | None:
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

# type parameter and alias AST nodes only exist on host Python 3.12+, so older
# hosts decompiling newer bytecode fall back to a plain alias assignment
SUPPORTS_TYPE_PARAMS = hasattr(ast, "TypeVar")
SUPPORTS_TYPE_ALIAS = hasattr(ast, "TypeAlias")

def make_type_alias_param_node(
    func_name: str,
    name: str,
    param: ast.Call,
) -> ast.AST | None:
    if not SUPPORTS_TYPE_PARAMS:
        return None
    if func_name == "TypeVar":
        return make_type_var_param(name, param)
    if func_name == "ParamSpec":
        return ast.ParamSpec(name=name)
    if func_name == "TypeVarTuple":
        return ast.TypeVarTuple(name=name)
    return None

def make_type_var_param(name: str, param: ast.Call) -> ast.TypeVar | None:
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
