import ast
from typing import Any

from pyc2py.astree import is_assertion_error_expr as is_assertion_error_call_target
from pyc2py.bytecode.decoder import decode_instructions
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.opcode_table import normalized_opcode_name
from pyc2py.decompiler.context import DecompilerContext
from pyc2py.decompiler.opcodes.imports_calls import (
    ImportedAttributeValue,
    ImportValue,
    keyword_names_from_value,
    make_call_argument_plan,
    make_import_star_statement,
)
from pyc2py.decompiler.opcodes.imports_calls import (
    split_call_arguments as split_call_argument_values,
)
from pyc2py.decompiler.opcodes.stack_names import is_null_sentinel
from pyc2py.decompiler.opcodes.values import is_code_constant
from pyc2py.decompiler.recover import make_exec_call
from pyc2py.decompiler.runtime import (
    BuildClassValue,
    ClassValue,
    FunctionValue,
    TypeAliasValue,
    coerce_expr,
    function_annotations_from_code,
    make_class_value,
    make_comprehension_expr,
    make_lambda_expr,
    unwrap_lazy_expr,
)
from pyc2py.decompiler.runtime import (
    annotation_dict_from_ast_dict_literal as annotation_dict_from_ast_dict,
)


class OpcodeCallRuntimeMixin(DecompilerContext):
    def call_function(self, instruction: Instruction) -> None:
        if self.try_call_assertion_error(instruction):
            return

        (
            raw_arguments,
            legacy_keyword_count,
            has_star_args,
            has_star_kwargs,
        ) = self.pop_call_arguments(instruction)
        function, self_argument = self.pop_call_target(instruction)
        (
            handled,
            function,
            self_argument,
            raw_arguments,
        ) = self.prepare_call_target(
            function,
            self_argument,
            raw_arguments,
        )
        if handled:
            return

        positional, keywords = split_call_argument_values(
            raw_arguments,
            self.kw_names,
            legacy_keyword_count,
            has_star_args,
            has_star_kwargs,
            self.expression_from_stack_value,
        )
        self.kw_names = ()
        if self_argument is not None:
            positional.insert(0, self.expression_from_stack_value(self_argument))
        if isinstance(function, FunctionValue):
            function = make_lambda_expr(
                function.code,
                self.version,
                function.defaults,
                function.kw_defaults,
            )
        else:
            function = coerce_expr(function)
        if is_null_sentinel(function):
            function = coerce_expr(self.pop_or_none())
        elif self.stack and is_null_sentinel(coerce_expr(self.stack[-1])):
            self.pop_or_none()

        self.stack.append(ast.Call(func=function, args=positional, keywords=keywords))

    def prepare_call_target(
        self,
        function: Any,
        self_argument: Any | None,
        raw_arguments: list[Any],
    ) -> tuple[bool, Any, Any | None, list[Any]]:
        if self.try_call_split_function_decorator(
            function,
            self_argument,
            raw_arguments,
        ):
            return True, function, self_argument, raw_arguments

        if isinstance(function, FunctionValue) and self_argument is not None:
            if raw_arguments:
                return False, function, self_argument, raw_arguments

            iterable = self.expression_from_stack_value(self_argument)
            comprehension = make_comprehension_expr(
                function.code, iterable, self.version
            )
            if comprehension is not None:
                self.stack.append(comprehension)
                return True, function, self_argument, raw_arguments
            return False, function, None, [self_argument]

        return self.prepare_plain_call_target(function, self_argument, raw_arguments)

    def prepare_plain_call_target(
        self,
        function: Any,
        self_argument: Any | None,
        raw_arguments: list[Any],
    ) -> tuple[bool, Any, Any | None, list[Any]]:
        if isinstance(function, FunctionValue) and not raw_arguments:
            if self.try_call_stacked_function_decorator(function):
                return True, function, self_argument, raw_arguments
            generic_value = self.evaluate_generic_parameter_function(function)
            if generic_value is not None:
                self.stack.append(generic_value)
                return True, function, self_argument, raw_arguments
            self.stack.append(function)
            return True, function, self_argument, raw_arguments
        if isinstance(function, BuildClassValue):
            self.stack.append(make_class_value(raw_arguments, self.version))
            return True, function, self_argument, raw_arguments
        if self.try_call_function_decorator(function, raw_arguments):
            return True, function, self_argument, raw_arguments
        return False, function, self_argument, raw_arguments

    def try_call_split_function_decorator(
        self,
        function: Any,
        self_argument: Any | None,
        raw_arguments: list[Any],
    ) -> bool:
        if raw_arguments:
            return False
        if not isinstance(self_argument, FunctionValue):
            return False
        if getattr(self_argument.code, "co_name", "") == "<lambda>":
            return False

        decorator = self.expression_from_stack_value(function)
        if not is_valid_function_decorator(decorator):
            return False

        self.stack.append(
            FunctionValue(
                code=self_argument.code,
                defaults=self_argument.defaults,
                kw_defaults=self_argument.kw_defaults,
                annotations=self_argument.annotations,
                decorators=(decorator, *self_argument.decorators),
                type_params=self_argument.type_params,
                annotate=self_argument.annotate,
            )
        )
        return True

    def try_call_stacked_function_decorator(self, decorated: FunctionValue) -> bool:
        if not self.stack:
            return False

        decorator = self.stack[-1]
        if is_null_sentinel(coerce_expr(decorator)):
            return False
        decorator_expr = self.expression_from_stack_value(decorator)
        if not is_valid_function_decorator(decorator_expr):
            return False

        self.pop_or_none()
        self.stack.append(
            FunctionValue(
                code=decorated.code,
                defaults=decorated.defaults,
                kw_defaults=decorated.kw_defaults,
                annotations=decorated.annotations,
                decorators=(decorator_expr, *decorated.decorators),
                type_params=decorated.type_params,
                annotate=decorated.annotate,
            )
        )
        return True

    def pop_call_target(
        self,
        instruction: Instruction | None = None,
    ) -> tuple[Any, Any | None]:
        function = self.pop_or_none()
        if (
            instruction is not None
            and call_opcode_name(instruction.opname, self.version) == "CALL_FUNCTION_EX"
        ):
            return function, None
        if not uses_split_call_target(instruction, self.version):
            return function, None
        if not self.stack:
            return function, None

        if is_null_sentinel(coerce_expr(function)):
            callable_value = self.pop_or_none()
            return callable_value, None
        if is_null_sentinel(coerce_expr(self.stack[-1])):
            self.pop_or_none()
            return function, None

        callable_value = self.pop_or_none()
        return callable_value, function

    def evaluate_generic_parameter_function(
        self, function: FunctionValue
    ) -> Any | None:
        code_name = str(getattr(function.code, "co_name", ""))
        if not code_name.startswith("<generic parameters of "):
            return None

        from pyc2py.decompiler.engine import NativeDecompiler

        child = NativeDecompiler(
            code=function.code,
            version=self.version,
            is_module=False,
        )
        for instruction in decode_instructions(function.code, self.version):
            if normalized_opcode_name(instruction.opname) != "RETURN_VALUE":
                child.run_instruction(instruction)
                continue

            value = child.pop_or_none()
            self.warnings.extend(child.warnings)
            if isinstance(value, ClassValue):
                return class_value_with_generic_type_params(value, child.statements)
            if isinstance(value, (ClassValue, FunctionValue, TypeAliasValue)):
                return value
            return None

        self.warnings.extend(child.warnings)
        return None

    def try_call_function_decorator(
        self,
        function: Any,
        raw_arguments: list[Any],
    ) -> bool:
        if len(raw_arguments) != 1:
            return False
        decorated = raw_arguments[0]
        if not isinstance(decorated, FunctionValue):
            return False
        if getattr(decorated.code, "co_name", "") == "<lambda>":
            return False

        if is_null_sentinel(coerce_expr(function)):
            return False
        decorator = self.expression_from_stack_value(function)
        if not is_valid_function_decorator(decorator):
            return False

        self.stack.append(
            FunctionValue(
                code=decorated.code,
                defaults=decorated.defaults,
                kw_defaults=decorated.kw_defaults,
                annotations=decorated.annotations,
                decorators=(decorator, *decorated.decorators),
                type_params=decorated.type_params,
                annotate=decorated.annotate,
            )
        )
        return True

    def try_call_assertion_error(self, instruction: Instruction) -> bool:
        if (
            call_opcode_name(instruction.opname, self.version) != "CALL"
            or int(instruction.arg or 0) != 0
        ):
            return False
        if len(self.stack) < 2:
            return False

        function = coerce_expr(self.stack[-2])
        if not is_assertion_error_call_target(function):
            return False

        message = self.expression_from_stack_value(self.pop_or_none())
        self.pop_or_none()
        self.stack.append(ast.Call(func=function, args=[message], keywords=[]))
        return True

    def pop_call_arguments(
        self, instruction: Instruction
    ) -> tuple[list[Any], int, bool, bool]:
        arg = int(instruction.arg or 0)
        plan = make_call_argument_plan(
            call_opcode_name(instruction.opname, self.version), arg, self.version
        )
        if plan.keyword_names_on_stack:
            self.set_kw_names_from_stack()
        return (
            self.pop_many_raw(plan.raw_count),
            plan.legacy_keyword_count,
            plan.has_star_args,
            plan.has_star_kwargs,
        )

    def set_kw_names(self, value: Any) -> None:
        names = keyword_names_from_value(value)
        if names is not None:
            self.kw_names = names
            return
        self.warnings.append("KW_NAMES operand was not a tuple of strings")
        self.kw_names = ()

    def set_kw_names_from_stack(self) -> None:
        value = self.pop_or_none()
        names = keyword_names_from_value(value)
        if names is not None:
            self.kw_names = names
            return
        self.warnings.append("CALL_KW keyword tuple had a non-string item")
        self.kw_names = ()

    def call_intrinsic_1(self, instruction: Instruction) -> None:
        name = intrinsic_name(instruction)
        raw_value = self.pop_or_none()
        if self.call_intrinsic_1_raw(name, raw_value):
            return

        value = self.expression_from_stack_value(raw_value)
        if self.call_intrinsic_1_value(name, value):
            return

        self.warnings.append(f"CALL_INTRINSIC_1 represented as helper call: {name}")
        self.stack.append(make_intrinsic_call("__pyc2py_intrinsic_1", name, [value]))

    def call_intrinsic_1_raw(self, name: str, raw_value: Any) -> bool:
        if name == "INTRINSIC_PRINT":
            self.stack.append(
                ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[self.expression_from_stack_value(raw_value)],
                    keywords=[],
                )
            )
            return True

        if name == "INTRINSIC_IMPORT_STAR":
            self.stack.append(raw_value)
            self.import_star()
            self.stack.append(ast.Constant(value=None))
            return True
        return False

    def call_intrinsic_1_value(self, name: str, value: ast.expr) -> bool:
        if self.call_intrinsic_1_direct_value(name, value):
            return True

        if name in INTRINSIC_TYPING_CONSTRUCTORS:
            self.stack.append(make_typing_constructor_call(name, [value]))
            return True

        if name == "INTRINSIC_TYPEALIAS":
            return self.call_intrinsic_type_alias(value)
        return False

    def call_intrinsic_1_direct_value(self, name: str, value: ast.expr) -> bool:
        if name == "INTRINSIC_UNARY_POSITIVE":
            self.stack.append(ast.UnaryOp(op=ast.UAdd(), operand=value))
            return True
        if name == "INTRINSIC_ASYNC_GEN_WRAP":
            self.stack.append(value)
            return True
        if name == "INTRINSIC_LIST_TO_TUPLE":
            self.stack.append(make_list_to_tuple_value(value))
            return True
        if name == "INTRINSIC_STOPITERATION_ERROR":
            self.stack.append(ast.Attribute(value=value, attr="value", ctx=ast.Load()))
            return True
        if name == "INTRINSIC_SUBSCRIPT_GENERIC":
            self.stack.append(make_generic_subscript(value))
            return True
        return False

    def call_intrinsic_type_alias(self, value: ast.expr) -> bool:
        type_alias = make_type_alias_value(value)
        if type_alias is not None:
            self.stack.append(type_alias)
            return True
        type_alias_call = make_type_alias_call(value)
        if type_alias_call is not None:
            self.stack.append(type_alias_call)
            return True
        return False

    def call_intrinsic_2(self, instruction: Instruction) -> None:
        name = intrinsic_name(instruction)
        raw_right = self.pop_or_none()
        raw_left = self.pop_or_none()
        if self.call_intrinsic_2_value(name, raw_left, raw_right):
            return

        right = self.expression_from_stack_value(raw_right)
        left = self.expression_from_stack_value(raw_left)
        self.warnings.append(f"CALL_INTRINSIC_2 represented as helper call: {name}")
        self.stack.append(
            make_intrinsic_call("__pyc2py_intrinsic_2", name, [left, right])
        )

    def call_intrinsic_2_value(
        self,
        name: str,
        raw_left: Any,
        raw_right: Any,
    ) -> bool:
        if name == "INTRINSIC_PREP_RERAISE_STAR":
            right = self.expression_from_stack_value(raw_right)
            left = self.expression_from_stack_value(raw_left)
            self.prep_reraise_star_from_values(left, right)
            return True
        if name == "INTRINSIC_TYPEVAR_WITH_BOUND":
            right = self.expression_from_stack_value(raw_right)
            left = self.expression_from_stack_value(raw_left)
            self.stack.append(
                ast.Call(
                    func=ast.Name(id="TypeVar", ctx=ast.Load()),
                    args=[left],
                    keywords=[ast.keyword(arg="bound", value=right)],
                )
            )
            return True
        if name == "INTRINSIC_TYPEVAR_WITH_CONSTRAINTS":
            right = self.expression_from_stack_value(raw_right)
            left = self.expression_from_stack_value(raw_left)
            self.stack.append(make_typevar_constraints_call(left, right))
            return True
        if name == "INTRINSIC_SET_FUNCTION_TYPE_PARAMS":
            return self.call_set_function_type_params(name, raw_left, raw_right)
        if name == "INTRINSIC_SET_TYPEPARAM_DEFAULT":
            return self.call_set_typeparam_default(name, raw_left, raw_right)
        return False

    def call_set_function_type_params(
        self,
        name: str,
        raw_left: Any,
        raw_right: Any,
    ) -> bool:
        right = self.expression_from_stack_value(raw_right)
        type_params = type_params_from_value(right)
        if type_params is None or not isinstance(raw_left, FunctionValue):
            self.warnings.append(
                "INTRINSIC_SET_FUNCTION_TYPE_PARAMS was not recognized"
            )
            left = self.expression_from_stack_value(raw_left)
            self.stack.append(
                make_intrinsic_call("__pyc2py_intrinsic_2", name, [left, right])
            )
            return True

        self.stack.append(
            FunctionValue(
                code=raw_left.code,
                defaults=raw_left.defaults,
                kw_defaults=raw_left.kw_defaults,
                annotations=raw_left.annotations,
                decorators=raw_left.decorators,
                type_params=type_params,
                annotate=raw_left.annotate,
            )
        )
        return True

    def call_set_typeparam_default(
        self,
        name: str,
        raw_left: Any,
        raw_right: Any,
    ) -> bool:
        left = self.expression_from_stack_value(raw_left)
        right = self.expression_from_stack_value(raw_right)
        default_call = make_typeparam_default_call(left, right)
        if default_call is None:
            self.warnings.append("INTRINSIC_SET_TYPEPARAM_DEFAULT was not recognized")
            self.stack.append(
                make_intrinsic_call("__pyc2py_intrinsic_2", name, [left, right])
            )
            return True

        self.stack.append(default_call)
        return True

    def make_function(self, flags: int = 0) -> None:
        value = self.pop_or_none()
        if not is_code_constant(value) and self.stack:
            value = self.pop_or_none()
        if value is None or not is_code_constant(value):
            self.stack.append(ast.Constant(value=None))
            return

        defaults, kw_defaults, annotations = self.pop_function_defaults(flags)
        self.stack.append(
            FunctionValue(
                code=value.value,
                defaults=tuple(defaults),
                kw_defaults=kw_defaults,
                annotations=annotations,
            )
        )

    def pop_function_defaults(
        self,
        flags: int,
    ) -> tuple[list[ast.expr], dict[str, ast.expr] | None, dict[str, ast.expr] | None]:
        if self.version is not None and self.version >= (3, 13):
            return [], None, None
        if self.version is not None and self.version >= (3, 6):
            return self.pop_modern_function_defaults(flags)
        return self.pop_legacy_function_defaults(flags)

    def pop_legacy_function_defaults(
        self,
        flags: int,
    ) -> tuple[list[ast.expr], dict[str, ast.expr] | None, dict[str, ast.expr] | None]:
        count = flags & 0xFF
        kw_defaults = self.pop_legacy_keyword_defaults((flags >> 8) & 0xFF)
        defaults = [
            self.expression_from_stack_value(value)
            for value in self.pop_many_raw(count)
        ]
        return defaults, kw_defaults, None

    def pop_legacy_keyword_defaults(self, count: int) -> dict[str, ast.expr] | None:
        if count <= 0:
            return None

        values = self.pop_many_raw(count * 2)
        result: dict[str, ast.expr] = {}
        for index in range(0, len(values), 2):
            key = self.expression_from_stack_value(values[index])
            value = self.expression_from_stack_value(values[index + 1])
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                result[key.value] = value
            else:
                self.warnings.append("legacy keyword default key was not a string")
        return result or None

    def pop_modern_function_defaults(
        self,
        flags: int,
    ) -> tuple[list[ast.expr], dict[str, ast.expr] | None, dict[str, ast.expr] | None]:
        if flags & 0x08:
            self.pop_or_none()
        annotations = (
            self.function_annotations_from_value(self.pop_or_none())
            if flags & 0x04
            else None
        )
        kw_defaults = self.pop_modern_keyword_defaults() if flags & 0x02 else None
        defaults = self.pop_modern_positional_defaults() if flags & 0x01 else []
        return defaults, kw_defaults, annotations

    def pop_modern_positional_defaults(self) -> list[ast.expr]:
        return self.positional_defaults_from_value(self.pop_or_none())

    def positional_defaults_from_value(self, value: Any) -> list[ast.expr]:
        defaults = self.expression_from_stack_value(value)
        if not isinstance(defaults, ast.Tuple):
            return []
        return list(defaults.elts)

    def pop_modern_keyword_defaults(self) -> dict[str, ast.expr] | None:
        return self.keyword_defaults_from_value(self.pop_or_none())

    def keyword_defaults_from_value(self, value: Any) -> dict[str, ast.expr] | None:
        defaults = self.expression_from_stack_value(value)
        if not isinstance(defaults, ast.Dict):
            return None
        result: dict[str, ast.expr] = {}
        for key, value in zip(defaults.keys, defaults.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                result[key.value] = value
        return result

    def function_annotations_from_value(
        self,
        value: Any,
    ) -> dict[str, ast.expr] | None:
        if isinstance(value, FunctionValue):
            return self.function_annotations_from_code(value.code)

        annotations = self.expression_from_stack_value(value)
        if isinstance(annotations, ast.Dict):
            return annotation_dict_from_ast_dict(annotations)
        if isinstance(annotations, ast.Tuple):
            return annotation_dict_from_ast_tuple(annotations)
        return None

    def function_annotations_from_code(self, code: Any) -> dict[str, ast.expr] | None:
        return function_annotations_from_code(code, self.version)

    def set_function_attribute(self, flag: int) -> None:
        function = self.pop_or_none()
        value = self.pop_or_none()
        if not isinstance(function, FunctionValue):
            self.warnings.append(
                f"SET_FUNCTION_ATTRIBUTE without function for flag {flag}"
            )
            self.stack.append(coerce_expr(function))
            return
        if flag == 0x01:
            self.stack.append(
                FunctionValue(
                    code=function.code,
                    defaults=tuple(self.positional_defaults_from_value(value)),
                    kw_defaults=function.kw_defaults,
                    annotations=function.annotations,
                    decorators=function.decorators,
                    type_params=function.type_params,
                    annotate=function.annotate,
                )
            )
            return
        if flag == 0x02:
            self.stack.append(
                FunctionValue(
                    code=function.code,
                    defaults=function.defaults,
                    kw_defaults=self.keyword_defaults_from_value(value),
                    annotations=function.annotations,
                    decorators=function.decorators,
                    type_params=function.type_params,
                    annotate=function.annotate,
                )
            )
            return
        if flag == 0x04:
            annotations = self.function_annotations_from_value(value)
            if annotations is None:
                self.warnings.append(
                    "SET_FUNCTION_ATTRIBUTE annotations were not recognized"
                )
                self.stack.append(function)
                return
            self.stack.append(
                FunctionValue(
                    code=function.code,
                    defaults=function.defaults,
                    kw_defaults=function.kw_defaults,
                    annotations=annotations,
                    decorators=function.decorators,
                    type_params=function.type_params,
                    annotate=function.annotate,
                )
            )
            return
        if flag == 0x10:
            self.set_function_annotate_attribute(function, value)
            return
        if flag == 0x08:
            self.warnings.append(
                f"SET_FUNCTION_ATTRIBUTE flag {flag:#x} is not rendered in source"
            )
        self.stack.append(function)

    def set_function_annotate_attribute(
        self,
        function: FunctionValue,
        value: Any,
    ) -> None:
        annotations = self.function_annotations_from_value(value)
        annotate = value if isinstance(value, FunctionValue) else function.annotate
        if annotations is None:
            if annotate is None:
                self.warnings.append(
                    "SET_FUNCTION_ATTRIBUTE annotate function was not recognized"
                )
            self.stack.append(
                FunctionValue(
                    code=function.code,
                    defaults=function.defaults,
                    kw_defaults=function.kw_defaults,
                    annotations=function.annotations,
                    decorators=function.decorators,
                    type_params=function.type_params,
                    annotate=annotate,
                )
            )
            return

        self.stack.append(
            FunctionValue(
                code=function.code,
                defaults=function.defaults,
                kw_defaults=function.kw_defaults,
                annotations=annotations,
                decorators=function.decorators,
                type_params=function.type_params,
                annotate=annotate,
            )
        )

    def make_closure(self) -> None:
        value = self.pop_or_none()
        if not is_code_constant(value) and self.stack:
            value = self.pop_or_none()
        if not is_code_constant(value):
            self.stack.append(ast.Constant(value=None))
            return

        self.discard_closure_tuple()
        self.stack.append(FunctionValue(code=value.value))

    def discard_closure_tuple(self) -> None:
        if not self.stack:
            self.warnings.append("MAKE_CLOSURE without closure tuple")
            return
        closure = self.stack[-1]
        if isinstance(closure, ast.Tuple):
            self.stack.pop()
            return
        self.warnings.append("MAKE_CLOSURE closure operand was not a tuple")

    def import_name(self, name: str) -> None:
        level = None
        fromlist = None
        if self.version is not None and self.version >= (2, 0):
            fromlist = self.pop_or_none()
        if self.version is not None and self.version >= (2, 5):
            level = self.pop_or_none()

        self.stack.append(ImportValue(name=name, level=level, fromlist=fromlist))

    def import_from(self, name: str) -> None:
        if self.stack and isinstance(self.stack[-1], ImportedAttributeValue):
            self.stack.append(
                ImportedAttributeValue(module=self.stack[-1].module, name=name)
            )
            return
        if not self.stack or not isinstance(self.stack[-1], ImportValue):
            self.warnings.append("IMPORT_FROM without module import on stack")
            self.stack.append(ast.Constant(value=None))
            return

        module = self.stack[-1]
        if module.name == "__future__" and name == "unicode_literals":
            self.saw_unicode_literals_future = True
        if module.name == "__future__" and name == "print_function":
            self.saw_print_function_future = True
        self.stack.append(ImportedAttributeValue(module=module, name=name))

    def import_star(self) -> None:
        value = self.pop_or_none()
        if not isinstance(value, ImportValue):
            self.warnings.append("IMPORT_STAR without module import on stack")
            return
        self.statements.append(make_import_star_statement(value))

    def exec_stmt(self) -> None:
        locals_value = coerce_expr(self.pop_or_none())
        globals_value = coerce_expr(self.pop_or_none())
        source = coerce_expr(self.pop_or_none())
        self.statements.append(make_exec_call(source, globals_value, locals_value))

    def build_legacy_class(self) -> None:
        function = self.pop_or_none()
        bases = coerce_expr(self.pop_or_none())
        name = self.pop_or_none()
        if not isinstance(function, FunctionValue):
            self.stack.append(ast.Constant(value=None))
            return
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            self.stack.append(ast.Constant(value=None))
            return

        base_items = tuple(bases.elts) if isinstance(bases, ast.Tuple) else ()
        self.stack.append(ClassValue(code=function.code, bases=base_items))


def annotation_dict_from_ast_tuple(value: ast.Tuple) -> dict[str, ast.expr] | None:
    if len(value.elts) % 2 == 0:
        return alternating_annotation_items(value.elts)
    return named_tail_annotation_items(value.elts)


def alternating_annotation_items(
    items: list[ast.expr],
) -> dict[str, ast.expr] | None:
    result: dict[str, ast.expr] = {}
    for index in range(0, len(items), 2):
        name = annotation_name(items[index])
        if name is None:
            return None
        result[name] = items[index + 1]
    return result


def named_tail_annotation_items(
    items: list[ast.expr],
) -> dict[str, ast.expr] | None:
    names = annotation_names(items[-1])
    annotations = items[:-1]
    if names is None or len(names) != len(annotations):
        return None
    return dict(zip(names, annotations, strict=True))


def annotation_names(value: ast.expr) -> tuple[str, ...] | None:
    if not isinstance(value, ast.Tuple):
        return None

    names: list[str] = []
    for item in value.elts:
        name = annotation_name(item)
        if name is None:
            return None
        names.append(name)
    return tuple(names)


def annotation_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def is_valid_function_decorator(value: ast.expr) -> bool:
    return isinstance(value, (ast.Name, ast.Attribute, ast.Call))


def intrinsic_name(instruction: Instruction) -> str:
    if isinstance(instruction.argval, str):
        return instruction.argval
    if instruction.argrepr:
        return str(instruction.argrepr)
    return str(instruction.arg)


def call_opcode_name(
    opname: str,
    version: tuple[int, ...] | None = None,
) -> str:
    if (
        version is not None
        and version >= (3, 13)
        and opname.startswith("CALL_")
        and opname.endswith("_WITH_KEYWORDS")
    ):
        return "CALL_KW"
    return normalized_opcode_name(opname)


def uses_split_call_target(
    instruction: Instruction | None,
    version: tuple[int, ...] | None,
) -> bool:
    if instruction is None:
        return False

    opname = call_opcode_name(instruction.opname, version)
    if opname == "CALL_METHOD":
        return version is None or version >= (3, 7)

    if opname in {"CALL", "CALL_KW"}:
        return version is None or version >= (3, 11)
    return False


def make_intrinsic_call(
    helper_name: str,
    intrinsic: str,
    arguments: list[ast.expr],
) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=helper_name, ctx=ast.Load()),
        args=[ast.Constant(value=intrinsic), *arguments],
        keywords=[],
    )


def make_list_to_tuple_value(value: ast.expr) -> ast.expr:
    if isinstance(value, ast.List):
        return ast.Tuple(elts=value.elts, ctx=ast.Load())
    return ast.Call(
        func=ast.Name(id="tuple", ctx=ast.Load()),
        args=[value],
        keywords=[],
    )


def make_generic_subscript(value: ast.expr) -> ast.Subscript:
    return ast.Subscript(
        value=ast.Name(id="Generic", ctx=ast.Load()),
        slice=value,
        ctx=ast.Load(),
    )


INTRINSIC_TYPING_CONSTRUCTORS = {
    "INTRINSIC_TYPEVAR": "TypeVar",
    "INTRINSIC_PARAMSPEC": "ParamSpec",
    "INTRINSIC_TYPEVARTUPLE": "TypeVarTuple",
}


def make_typing_constructor_call(name: str, arguments: list[ast.expr]) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=INTRINSIC_TYPING_CONSTRUCTORS[name], ctx=ast.Load()),
        args=arguments,
        keywords=[],
    )


def make_typevar_constraints_call(name: ast.expr, constraints: ast.expr) -> ast.Call:
    arguments = [name]
    if isinstance(constraints, ast.Tuple):
        arguments.extend(constraints.elts)
    else:
        arguments.append(ast.Starred(value=constraints, ctx=ast.Load()))
    return ast.Call(
        func=ast.Name(id="TypeVar", ctx=ast.Load()),
        args=arguments,
        keywords=[],
    )


def make_typeparam_default_call(value: ast.expr, default: ast.expr) -> ast.Call | None:
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Name):
        return None
    if value.func.id not in {"TypeVar", "ParamSpec", "TypeVarTuple"}:
        return None
    return ast.Call(
        func=value.func,
        args=value.args,
        keywords=[
            *value.keywords,
            ast.keyword(arg="default", value=unwrap_lazy_expr(default)),
        ],
    )


def class_value_with_generic_type_params(
    value: ClassValue,
    statements: list[ast.stmt],
) -> ClassValue:
    type_params = generic_type_params_from_statements(statements)
    if not type_params:
        return value

    return ClassValue(
        code=value.code,
        bases=generic_class_bases_without_synthetic_base(value.bases),
        type_params=type_params,
    )


def generic_type_params_from_statements(
    statements: list[ast.stmt],
) -> tuple[ast.expr, ...]:
    for statement in statements:
        if not isinstance(statement, ast.Assign):
            continue
        if not isinstance(statement.value, ast.Tuple):
            continue
        type_params = tuple(statement.value.elts)
        if type_params and all(
            is_typing_constructor_call(param) for param in type_params
        ):
            return type_params
    return ()


def is_typing_constructor_call(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"TypeVar", "ParamSpec", "TypeVarTuple"}
    )


def generic_class_bases_without_synthetic_base(
    bases: tuple[ast.expr, ...],
) -> tuple[ast.expr, ...]:
    return tuple(
        base
        for base in bases
        if not (isinstance(base, ast.Name) and base.id == "value")
    )


def type_params_from_value(value: ast.expr) -> tuple[ast.expr, ...] | None:
    if isinstance(value, ast.Tuple):
        return tuple(value.elts)
    if isinstance(value, ast.List):
        return tuple(value.elts)
    return None


def make_type_alias_call(value: ast.expr) -> ast.Call | None:
    if not isinstance(value, ast.Tuple) or len(value.elts) != 3:
        return None

    name, type_params, alias_value = value.elts
    return ast.Call(
        func=ast.Name(id="TypeAliasType", ctx=ast.Load()),
        args=[name, alias_value],
        keywords=[ast.keyword(arg="type_params", value=type_params)],
    )


def make_type_alias_value(value: ast.expr) -> TypeAliasValue | None:
    if not isinstance(value, ast.Tuple) or len(value.elts) != 3:
        return None

    name, type_params, alias_value = value.elts
    if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
        return None

    alias_expr = type_alias_expr(alias_value)
    if alias_expr is None:
        return None

    return TypeAliasValue(
        name=name.value,
        value=alias_expr,
        type_params=type_alias_type_params(type_params),
    )


def type_alias_expr(value: ast.expr) -> ast.expr | None:
    if isinstance(value, ast.Lambda):
        return value.body
    if isinstance(value, ast.Constant) and value.value is None:
        return None
    return value


def type_alias_type_params(value: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(value, ast.Tuple):
        return tuple(value.elts)
    return ()
