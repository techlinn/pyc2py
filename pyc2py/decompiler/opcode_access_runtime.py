import ast
from typing import Any

from pyc2py.astree import make_name, safe_identifier
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.flow import legacy_slice_mode, make_raise
from pyc2py.decompiler.opcodes.imports_calls import ImportValue, make_super_attribute
from pyc2py.decompiler.opcodes.stack_names import is_annotations_name, is_null_sentinel
from pyc2py.decompiler.opcodes.values import (
    is_none_constant,
    make_attr_delete,
    make_attr_load,
    make_attr_store,
    make_binary_slice,
    make_binary_subscript,
    make_empty_slice_step,
    make_legacy_slice_operand,
    make_slice,
    make_slice_operand,
    make_subscript,
    make_subscript_delete,
    make_subscript_store,
)
from pyc2py.decompiler.recover import append_annotation_statement
from pyc2py.decompiler.runtime import (
    FunctionValue,
    UnpackSlot,
    coerce_expr,
    is_coroutine_code,
    make_lambda_expr,
    none_to_empty,
)
from pyc2py.decompiler.structures import is_yield_expression
from pyc2py.stack import FastStack


class OpcodeAccessRuntimeMixin(DecompilerContext):
    def load_attr(self, name: str, arg: int = 0) -> None:
        raw_value = self.pop_or_none()
        if is_null_sentinel(coerce_expr(raw_value)) and self.stack:
            raw_value = self.pop_or_none()
        if isinstance(raw_value, ImportValue) and raw_value.name.endswith(f".{name}"):
            self.stack.append(raw_value)
            return

        value = coerce_expr(raw_value)
        attribute = make_attr_load(name, value)
        if self.version is not None and self.version >= (3, 12) and arg & 1:
            self.push_loaded_method(attribute)
            return
        self.stack.append(attribute)

    def load_method(self, name: str) -> None:
        raw_value = self.pop_or_none()
        if is_null_sentinel(coerce_expr(raw_value)) and self.stack:
            raw_value = self.pop_or_none()
        if isinstance(raw_value, ImportValue) and raw_value.name.endswith(f".{name}"):
            self.push_loaded_method(coerce_expr(raw_value))
            return

        self.push_loaded_method(make_attr_load(name, coerce_expr(raw_value)))

    def load_super_attr(self, name: str, arg: int) -> None:
        self_argument = coerce_expr(self.pop_or_none())
        class_argument = coerce_expr(self.pop_or_none())
        super_function = coerce_expr(self.pop_or_none())

        attribute = make_super_attribute(
            super_function=super_function,
            class_argument=class_argument,
            self_argument=self_argument,
            attr=name,
            uses_two_arguments=bool(arg & 0x02),
        )
        if arg & 1:
            self.push_loaded_method(attribute)
            return
        self.stack.append(attribute)

    def push_loaded_method(self, attribute: ast.expr) -> None:
        null_value = make_name("NULL", ast.Load())
        if self.version is not None and self.version < (3, 13):
            self.stack.append(null_value)
            self.stack.append(attribute)
            return

        self.stack.append(attribute)
        self.stack.append(null_value)

    def load_special(self, arg: int) -> None:
        value = coerce_expr(self.pop_or_none())
        name = special_method_name(arg)
        if name is None:
            self.warnings.append(f"unsupported LOAD_SPECIAL operand: {arg}")
            self.stack.append(
                ast.Call(
                    func=make_name("__pyc2py_load_special__", ast.Load()),
                    args=[value, ast.Constant(value=arg)],
                    keywords=[],
                )
            )
            self.stack.append(make_name("NULL", ast.Load()))
            return

        self.stack.append(make_attr_load(name, value))
        self.stack.append(make_name("NULL", ast.Load()))

    def load_locals(self) -> None:
        self.stack.append(
            ast.Call(func=make_name("locals", ast.Load()), args=[], keywords=[])
        )

    def load_from_dict_or_name(self, name: str) -> None:
        self.pop_or_none()
        self.stack.append(make_name(name, ast.Load()))

    def before_with(self) -> None:
        manager = self.expression_from_stack_value(self.pop_or_none())
        self.warnings.append("BEFORE_WITH represented as explicit context calls")

        self.stack.append(ast.Attribute(value=manager, attr="__exit__", ctx=ast.Load()))
        self.stack.append(
            ast.Call(
                func=ast.Attribute(value=manager, attr="__enter__", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )

    def get_aiter(self) -> None:
        value = self.expression_from_stack_value(self.pop_or_none())
        self.stack.append(
            ast.Call(
                func=ast.Attribute(value=value, attr="__aiter__", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )

    def get_anext(self) -> None:
        if not self.stack:
            self.warnings.append("GET_ANEXT on empty stack")
            self.stack.append(ast.Constant(value=None))
            return

        value = self.expression_from_stack_value(self.stack[-1])
        self.warnings.append("GET_ANEXT represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_get_awaitable__", ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Attribute(
                            value=value,
                            attr="__anext__",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )

    def before_async_with(self) -> None:
        manager = self.expression_from_stack_value(self.pop_or_none())
        self.warnings.append(
            "BEFORE_ASYNC_WITH represented as explicit async context calls"
        )

        self.stack.append(
            ast.Attribute(value=manager, attr="__aexit__", ctx=ast.Load())
        )
        self.stack.append(
            ast.Call(
                func=ast.Attribute(value=manager, attr="__aenter__", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )

    def end_async_for(self) -> None:
        self.pop_or_none()
        self.pop_or_none()

    def end_send(self) -> None:
        if len(self.stack) < 2:
            self.warnings.append("END_SEND on shallow stack")
            self.pop_or_none()
            return
        del self.stack[-2]

    def cleanup_throw(self) -> None:
        sent_value, exception = self.pop_many(2)
        self.warnings.append("CLEANUP_THROW represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_cleanup_throw__", ast.Load()),
                args=[sent_value, exception],
                keywords=[],
            )
        )

    def with_except_start(self) -> None:
        self.warnings.append("WITH_EXCEPT_START represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_with_except_start__", ast.Load()),
                args=[coerce_expr(item) for item in self.stack[-4:]],
                keywords=[],
            )
        )

    def get_len(self) -> None:
        if not self.stack:
            self.warnings.append("GET_LEN on empty stack")
            self.stack.append(ast.Constant(value=None))
            return
        value = coerce_expr(self.stack[-1])
        self.stack.append(
            ast.Call(func=make_name("len", ast.Load()), args=[value], keywords=[])
        )

    def match_mapping(self) -> None:
        self.stack.append(self.match_type_call("__pyc2py_match_mapping__"))

    def match_sequence(self) -> None:
        self.stack.append(self.match_type_call("__pyc2py_match_sequence__"))

    def match_type_call(self, helper_name: str) -> ast.expr:
        if not self.stack:
            self.warnings.append(f"{helper_name} on empty stack")
            return ast.Constant(value=False)
        value = coerce_expr(self.stack[-1])
        self.warnings.append(f"{helper_name} represented as helper call")
        return ast.Call(
            func=make_name(helper_name, ast.Load()), args=[value], keywords=[]
        )

    def match_keys(self) -> None:
        if len(self.stack) < 2:
            self.warnings.append("MATCH_KEYS on shallow stack")
            self.stack.append(ast.Constant(value=None))
            return

        keys = coerce_expr(self.stack[-1])
        subject = coerce_expr(self.stack[-2])
        self.warnings.append("MATCH_KEYS represented as helper call")

        result = ast.Call(
            func=make_name("__pyc2py_match_keys__", ast.Load()),
            args=[subject, keys],
            keywords=[],
        )
        self.stack.append(result)
        if self.version is not None and self.version < (3, 11):
            self.stack.append(
                ast.Compare(
                    left=result,
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                )
            )

    def match_class(self, count: int) -> None:
        names = coerce_expr(self.pop_or_none())
        class_value = coerce_expr(self.pop_or_none())
        subject = coerce_expr(self.pop_or_none())
        self.warnings.append("MATCH_CLASS represented as helper call")

        result = ast.Call(
            func=make_name("__pyc2py_match_class__", ast.Load()),
            args=[subject, class_value, names, ast.Constant(value=count)],
            keywords=[],
        )
        self.stack.append(result)
        if self.version is not None and self.version < (3, 11):
            self.stack.append(
                ast.Compare(
                    left=result,
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                )
            )

    def copy_dict_without_keys(self) -> None:
        keys = coerce_expr(self.pop_or_none())
        if not self.stack:
            self.warnings.append("COPY_DICT_WITHOUT_KEYS without subject")
            self.stack.append(ast.Dict(keys=[], values=[]))
            return
        subject = coerce_expr(self.stack[-1])
        self.warnings.append("COPY_DICT_WITHOUT_KEYS represented as helper call")
        self.stack.append(
            ast.Call(
                func=make_name("__pyc2py_copy_dict_without_keys__", ast.Load()),
                args=[subject, keys],
                keywords=[],
            )
        )

    def binary_subscript(self) -> None:
        index, value = self.pop_binary()
        self.stack.append(make_binary_subscript(value, index))

    def binary_slice(self) -> None:
        stop = none_to_empty(coerce_expr(self.pop_or_none()))
        start = none_to_empty(coerce_expr(self.pop_or_none()))
        value = coerce_expr(self.pop_or_none())

        self.stack.append(make_binary_slice(value, start, stop))

    def legacy_slice(self, opname: str) -> None:
        target_value, slice_value = self.pop_legacy_slice_target(opname)
        self.stack.append(make_subscript(target_value, slice_value, ast.Load()))

    def build_slice(self, count: int) -> None:
        if count == 2:
            stop = none_to_empty(coerce_expr(self.pop_or_none()))
            start = none_to_empty(coerce_expr(self.pop_or_none()))
            self.stack.append(
                make_slice_operand(count, start, stop) or make_slice(None, None)
            )
            return
        if count == 3:
            step = none_to_empty(coerce_expr(self.pop_or_none()))
            if step is None:
                step = make_empty_slice_step()
            stop = none_to_empty(coerce_expr(self.pop_or_none()))
            start = none_to_empty(coerce_expr(self.pop_or_none()))
            slice_value = make_slice_operand(count, start, stop, step)
            if slice_value is None:
                self.stack.append(ast.Constant(value=None))
                return
            self.stack.append(slice_value)
            return
        self.warnings.append(f"unsupported BUILD_SLICE count: {count}")
        self.stack.append(ast.Constant(value=None))

    def yield_value(self) -> None:
        value = coerce_expr(self.pop_or_none())
        self.stack.append(ast.Yield(value=value))

    def yield_from(self) -> None:
        ignored = self.pop_or_none()
        value = coerce_expr(self.pop_or_none())

        if is_none_constant(value):
            value = coerce_expr(ignored)
        if is_coroutine_code(self.code):
            self.stack.append(ast.Await(value=value))
            return
        self.stack.append(ast.YieldFrom(value=value))

    def flush_yield_expressions(self) -> None:
        remaining: list[Any] = []

        for value in self.stack:
            expression = coerce_expr(value)
            if is_yield_expression(expression):
                self.statements.append(ast.Expr(value=expression))
                continue
            remaining.append(value)

        self.stack = FastStack(remaining)

    def store_subscript(self) -> None:
        raw_index = self.pop_or_none()
        index = (
            raw_index if isinstance(raw_index, ast.Slice) else coerce_expr(raw_index)
        )
        target_value = coerce_expr(self.pop_or_none())
        raw_value = self.pop_or_none()

        if isinstance(raw_value, UnpackSlot):
            target = make_subscript(target_value, index, ast.Store())
            self.store_unpack_target(raw_value, target)
            return

        value = self.expression_from_stack_value(raw_value)
        if isinstance(target_value, ast.Dict):
            target_value.keys.append(coerce_expr(index))
            target_value.values.append(value)
            return
        if self.try_store_annotation(target_value, index, value):
            return
        self.append_store_statement(make_subscript_store(target_value, index, value))

    def append_store_statement(self, statement: ast.stmt) -> None:
        if self.pending_simultaneous_store_count > 0:
            setattr(statement, "_pyc2py_store_group", self.simultaneous_store_group)
            self.pending_simultaneous_store_count -= 1
        self.statements.append(statement)

    def expression_from_stack_value(self, value: Any) -> ast.expr:
        if isinstance(value, FunctionValue):
            return make_lambda_expr(
                value.code,
                self.version,
                value.defaults,
                value.kw_defaults,
            )
        return coerce_expr(value)

    def try_store_annotation(
        self,
        target_value: ast.expr,
        index: ast.expr | ast.slice,
        value: ast.expr,
    ) -> bool:
        if not is_annotations_name(target_value):
            return False
        if not isinstance(index, ast.Constant) or not isinstance(index.value, str):
            return False
        return append_annotation_statement(self.statements, index.value, value)

    def store_slice(self) -> None:
        stop = none_to_empty(coerce_expr(self.pop_or_none()))
        start = none_to_empty(coerce_expr(self.pop_or_none()))
        target_value = coerce_expr(self.pop_or_none())
        value = coerce_expr(self.pop_or_none())

        self.statements.append(
            make_subscript_store(target_value, make_slice(start, stop), value)
        )

    def store_legacy_slice(self, opname: str) -> None:
        target_value, slice_value = self.pop_legacy_slice_target(opname)
        value = coerce_expr(self.pop_or_none())

        self.statements.append(make_subscript_store(target_value, slice_value, value))

    def store_attr(self, name: str) -> None:
        target_value = coerce_expr(self.pop_or_none())
        raw_value = self.pop_or_none()

        if isinstance(raw_value, UnpackSlot):
            target = ast.Attribute(
                value=target_value,
                attr=safe_identifier(name),
                ctx=ast.Store(),
            )
            self.store_unpack_target(raw_value, target)
            return

        value = coerce_expr(raw_value)
        self.statements.append(
            make_attr_store(safe_identifier(name), target_value, value)
        )

    def delete_subscript(self) -> None:
        raw_index = self.pop_or_none()
        index = (
            raw_index if isinstance(raw_index, ast.Slice) else coerce_expr(raw_index)
        )
        target_value = coerce_expr(self.pop_or_none())

        self.statements.append(make_subscript_delete(target_value, index))

    def delete_attr(self, name: str) -> None:
        target_value = coerce_expr(self.pop_or_none())
        self.statements.append(make_attr_delete(safe_identifier(name), target_value))

    def delete_slice(self, opname: str) -> None:
        target_value, slice_value = self.pop_legacy_slice_target(opname)
        target = make_subscript(target_value, slice_value, ast.Del())

        self.statements.append(ast.Delete(targets=[target]))

    def pop_legacy_slice_target(self, opname: str) -> tuple[ast.expr, ast.Slice]:
        mode = legacy_slice_mode(opname)
        if mode == "0":
            target_value = coerce_expr(self.pop_or_none())
            return make_legacy_slice_operand(mode, target_value)
        if mode == "1":
            lower = none_to_empty(self.pop_or_none())
            target_value = coerce_expr(self.pop_or_none())
            return make_legacy_slice_operand(mode, target_value, lower=lower)
        if mode == "2":
            upper = none_to_empty(self.pop_or_none())
            target_value = coerce_expr(self.pop_or_none())
            return make_legacy_slice_operand(mode, target_value, upper=upper)
        upper = none_to_empty(self.pop_or_none())
        lower = none_to_empty(self.pop_or_none())
        target_value = coerce_expr(self.pop_or_none())
        return make_legacy_slice_operand(mode, target_value, lower=lower, upper=upper)

    def raise_varargs(self, count: int) -> None:
        values = [] if count <= 0 else self.pop_many(count)
        self.statements.append(make_raise(values))

    def reraise(self) -> None:
        self.statements.append(make_raise([]))


LOAD_SPECIAL_METHOD_NAMES = (
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
)


def special_method_name(arg: int) -> str | None:
    if arg < 0 or arg >= len(LOAD_SPECIAL_METHOD_NAMES):
        return None
    return LOAD_SPECIAL_METHOD_NAMES[arg]
