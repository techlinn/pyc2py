import ast
from dataclasses import dataclass
from typing import Any

from pyc2py.astree import (
    expression_key,
    make_name,
    safe_identifier,
)
from pyc2py.bytecode.decoder import decode_instructions
from pyc2py.bytecode.instruction import Instruction
from pyc2py.bytecode.metadata import make_offset_index, skip_ignorable_instructions
from pyc2py.decompiler.opcodes.imports_calls import (
    ImportedAttributeValue,
    ImportValue,
)
from pyc2py.decompiler.opcodes.values import is_code_constant
from pyc2py.decompiler.recover import (
    make_arguments as recover_make_arguments,
)
from pyc2py.decompiler.recover import (
    make_class_node as recover_make_class_node,
)
from pyc2py.decompiler.recover import (
    make_dict_comp,
    make_generator,
    make_generator_exp,
    make_list_comp,
    make_set_comp,
)
from pyc2py.decompiler.recover import (
    make_function_node as recover_make_function_node,
)
from pyc2py.decompiler.recover import (
    with_code_docstring as recover_with_code_docstring,
)
from pyc2py.decompiler.structures import invert_condition


@dataclass(frozen=True, slots=True)
class FunctionValue:
    code: Any
    defaults: tuple[ast.expr, ...] = ()
    kw_defaults: dict[str, ast.expr] | None = None
    annotations: dict[str, ast.expr] | None = None
    decorators: tuple[ast.expr, ...] = ()
    type_params: tuple[ast.expr, ...] = ()
    annotate: "FunctionValue | None" = None


@dataclass(frozen=True, slots=True)
class BuildClassValue:
    # marker for a pending __build_class__ call
    pass


@dataclass(frozen=True, slots=True)
class ClassValue:
    code: Any
    bases: tuple[ast.expr, ...]
    type_params: tuple[ast.expr, ...] = ()


@dataclass(frozen=True, slots=True)
class ComprehensionLoopShape:
    instructions: list[Instruction]
    offset_to_index: dict[int, int]
    load_iterator_index: int
    for_iter_index: int
    loop_end_index: int
    store_index: int
    target: ast.expr


@dataclass(frozen=True, slots=True)
class HiddenLocalRestore:
    name: str


@dataclass(frozen=True, slots=True)
class TypeAliasValue:
    name: str
    value: ast.expr
    type_params: tuple[ast.expr, ...] = ()


@dataclass(frozen=True, slots=True)
class UnpackSlot:
    group: "UnpackGroup"
    index: int


@dataclass(slots=True)
class UnpackGroup:
    value: ast.expr
    targets: list[ast.expr | None]
    starred_index: int | None = None


def decompile_native_source(code: Any, version: tuple[int, ...] | None) -> Any | None:
    from pyc2py.decompiler.engine import NativeDecompiler

    return NativeDecompiler(code=code, version=version, is_module=True).decompile()


def unpack_counts(instruction: Instruction) -> tuple[int, int | None]:
    if instruction.opname == "UNPACK_SEQUENCE_TWO_TUPLE":
        return 2, None

    arg = int(instruction.arg or 0)
    if instruction.opname != "UNPACK_EX":
        return arg, None

    before_count = arg & 0xFF
    after_count = (arg >> 8) & 0xFF
    return before_count + after_count + 1, before_count


def make_function_def(
    name: str,
    code: Any,
    version: tuple[int, ...] | None,
    defaults: tuple[ast.expr, ...] = (),
    kw_defaults: dict[str, ast.expr] | None = None,
    annotations: dict[str, ast.expr] | None = None,
    decorators: tuple[ast.expr, ...] = (),
    type_params: tuple[ast.AST, ...] = (),
    annotate: FunctionValue | None = None,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    from pyc2py.decompiler.engine import NativeDecompiler

    body_result = NativeDecompiler(
        code=code, version=version, is_module=False
    ).decompile()
    body_module = parse_body_or_empty(body_result)
    body = recover_with_code_docstring(code, body_module.body or [ast.Pass()])
    function = recover_make_function_node(name, code, body)
    function.args.defaults = list(defaults)
    if kw_defaults:
        function.args.kw_defaults = [
            kw_defaults.get(argument.arg) for argument in function.args.kwonlyargs
        ]

    resolved_annotations = annotations
    if resolved_annotations is None and annotate is not None:
        resolved_annotations = function_annotations_from_code(annotate.code, version)
    if resolved_annotations:
        apply_function_annotations(function, resolved_annotations)

    function.decorator_list = list(decorators)
    setattr(function, "type_params", list(type_params))
    return function


def function_annotations_from_code(
    code: Any,
    version: tuple[int, ...] | None,
) -> dict[str, ast.expr] | None:
    from pyc2py.decompiler.engine import NativeDecompiler

    result = NativeDecompiler(
        code=code,
        version=version,
        is_module=False,
    ).decompile()
    module = parse_body_or_empty(result)
    for statement in module.body:
        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
            return annotation_dict_from_ast_dict_literal(statement.value)
    return None


def annotation_dict_from_ast_dict_literal(
    value: ast.Dict,
) -> dict[str, ast.expr] | None:
    result: dict[str, ast.expr] = {}
    for key, annotation in zip(value.keys, value.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        result[key.value] = annotation
    return result


def apply_function_annotations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    annotations: dict[str, ast.expr],
) -> None:
    for argument in function_annotation_arguments(function.args):
        annotation = annotations.get(argument.arg)
        if annotation is not None:
            argument.annotation = annotation
    if function.args.vararg is not None and function.args.vararg.annotation is not None:
        function.args.vararg.annotation = normalize_vararg_annotation(
            function.args.vararg.annotation
        )

    returns = annotations.get("return")
    if returns is not None:
        function.returns = returns


def normalize_vararg_annotation(annotation: ast.expr) -> ast.expr:
    if not isinstance(annotation, ast.Subscript):
        return annotation
    if not isinstance(annotation.slice, ast.Constant) or annotation.slice.value != 0:
        return annotation
    if not isinstance(annotation.value, ast.Name):
        return annotation
    return ast.Starred(value=annotation.value, ctx=ast.Load())


def function_annotation_arguments(arguments: ast.arguments) -> list[ast.arg]:
    result = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        result.append(arguments.vararg)
    if arguments.kwarg is not None:
        result.append(arguments.kwarg)
    return result


def make_class_def(
    name: str,
    code: Any,
    version: tuple[int, ...] | None,
    bases: tuple[ast.expr, ...],
    type_params: tuple[ast.AST, ...] = (),
) -> ast.ClassDef:
    from pyc2py.decompiler.engine import NativeDecompiler

    body_result = NativeDecompiler(
        code=code, version=version, is_module=False
    ).decompile()
    body_module = parse_body_or_empty(body_result)
    class_node = recover_make_class_node(name, bases, body_module.body)
    setattr(class_node, "type_params", list(type_params))
    return class_node


def make_class_value(arguments: list[Any], version: tuple[int, ...] | None) -> Any:
    if len(arguments) < 2:
        return ast.Constant(value=None)

    function = arguments[0]
    class_name = arguments[1]
    if not isinstance(function, FunctionValue):
        return ast.Constant(value=None)
    if not isinstance(class_name, ast.Constant) or not isinstance(
        class_name.value, str
    ):
        return ast.Constant(value=None)

    bases = make_class_bases(arguments[2:], version)
    return ClassValue(code=function.code, bases=bases)


def make_class_bases(
    raw_bases: list[Any],
    version: tuple[int, ...] | None,
) -> tuple[ast.expr, ...]:
    bases = tuple(raw_bases)
    if version is not None and (3, 0) <= version < (3, 5):
        return tuple(reversed(bases))
    return bases


def make_lambda_expr(
    code: Any,
    version: tuple[int, ...] | None,
    defaults: tuple[ast.expr, ...] = (),
    kw_defaults: dict[str, ast.expr] | None = None,
) -> ast.Lambda:
    from pyc2py.decompiler.engine import NativeDecompiler

    child = NativeDecompiler(code=code, version=version, is_module=False)
    instructions = decode_instructions(code, version)
    statements = child.translate_range(instructions, 0, len(instructions))
    for statement in statements:
        if isinstance(statement, ast.Return) and statement.value is not None:
            return make_lambda_node(code, statement.value, defaults, kw_defaults)
    return make_lambda_node(code, ast.Constant(value=None), defaults, kw_defaults)


def make_lambda_node(
    code: Any,
    body: ast.expr,
    defaults: tuple[ast.expr, ...],
    kw_defaults: dict[str, ast.expr] | None,
) -> ast.Lambda:
    arguments = recover_make_arguments(code)
    arguments.defaults = list(defaults)
    if kw_defaults:
        arguments.kw_defaults = [
            kw_defaults.get(argument.arg) for argument in arguments.kwonlyargs
        ]
    return ast.Lambda(args=arguments, body=body)


def make_generator_expr(
    code: Any,
    iterable: ast.expr,
    version: tuple[int, ...] | None,
) -> ast.GeneratorExp | None:
    if getattr(code, "co_name", "") != "<genexpr>":
        return None

    instructions = decode_instructions(code, version)
    offset_to_index = make_offset_index(instructions)
    load_iterator_index = find_generator_iterator_load(instructions)
    if load_iterator_index is None:
        return None

    async_generator = make_async_generator_expr(
        code,
        iterable,
        version,
        instructions,
        offset_to_index,
        load_iterator_index,
    )
    if async_generator is not None:
        return async_generator

    shape = read_comprehension_loop_shape(
        instructions,
        offset_to_index,
        load_iterator_index,
    )
    if shape is None:
        return None
    return make_sync_generator_expr(code, iterable, version, shape)


def make_sync_generator_expr(
    code: Any,
    iterable: ast.expr,
    version: tuple[int, ...] | None,
    shape: ComprehensionLoopShape,
) -> ast.GeneratorExp | None:
    instructions = shape.instructions
    offset_to_index = shape.offset_to_index
    yield_index = find_generator_yield(
        instructions,
        shape.store_index + 1,
        shape.loop_end_index,
    )
    if yield_index is None:
        return None

    generators = [make_generator(target=shape.target, iterator=iterable)]
    expression_scan_start = shape.store_index + 1
    nested = read_nested_generator_loop(
        code,
        version,
        instructions,
        offset_to_index,
        expression_scan_start,
        yield_index,
    )
    if nested is not None:
        nested_generator, expression_scan_start = nested
        generators.append(nested_generator)

    expression_start, conditions = read_generator_conditions(
        code,
        version,
        instructions,
        offset_to_index,
        expression_scan_start,
        yield_index,
        int(instructions[shape.for_iter_index].offset),
    )
    element = evaluate_generator_expression(
        code, version, instructions, expression_start, yield_index
    )
    if element is None:
        return None

    generators[-1].ifs.extend(conditions)
    return make_generator_exp(element, generators)


def make_comprehension_expr(
    code: Any,
    iterable: ast.expr,
    version: tuple[int, ...] | None,
) -> ast.expr | None:
    code_name = str(getattr(code, "co_name", ""))
    if code_name == "<genexpr>":
        return make_generator_expr(code, iterable, version)
    if code_name not in {"<listcomp>", "<setcomp>", "<dictcomp>"}:
        return None

    instructions = decode_instructions(code, version)
    offset_to_index = make_offset_index(instructions)
    load_iterator_index = find_generator_iterator_load(instructions)
    if load_iterator_index is None:
        return None

    shape = read_comprehension_loop_shape(
        instructions,
        offset_to_index,
        load_iterator_index,
    )
    if shape is None:
        return None
    return make_sync_comprehension_expr(code, iterable, version, code_name, shape)


def make_sync_comprehension_expr(
    code: Any,
    iterable: ast.expr,
    version: tuple[int, ...] | None,
    code_name: str,
    shape: ComprehensionLoopShape,
) -> ast.expr | None:
    instructions = shape.instructions
    offset_to_index = shape.offset_to_index
    update_index = find_comprehension_update(
        instructions,
        shape.store_index + 1,
        shape.loop_end_index,
        code_name,
    )
    if update_index is None:
        return None

    expression_start, conditions = read_generator_conditions(
        code,
        version,
        instructions,
        offset_to_index,
        shape.store_index + 1,
        update_index,
        int(instructions[shape.for_iter_index].offset),
    )
    stack_values = evaluate_expression_stack(
        code,
        version,
        instructions,
        expression_start,
        update_index,
    )
    if not stack_values:
        return None

    generator = make_generator(target=shape.target, iterator=iterable)
    generator.ifs.extend(conditions)
    if code_name == "<listcomp>":
        return make_list_comp(stack_values[-1], [generator])
    if code_name == "<setcomp>":
        return make_set_comp(stack_values[-1], [generator])
    if len(stack_values) < 2:
        return None
    return make_dict_comp(stack_values[-2], stack_values[-1], [generator])


def read_comprehension_loop_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    load_iterator_index: int,
) -> ComprehensionLoopShape | None:
    for_iter_index = skip_ignorable_instructions(
        instructions,
        load_iterator_index + 1,
        len(instructions),
    )
    if for_iter_index >= len(instructions):
        return None
    for_iter = instructions[for_iter_index]
    if for_iter.opname != "FOR_ITER":
        return None

    loop_end_index = offset_to_index.get(int(for_iter.argval))
    if loop_end_index is None:
        return None

    store_index = skip_ignorable_instructions(
        instructions,
        for_iter_index + 1,
        loop_end_index,
    )
    if store_index >= loop_end_index:
        return None

    target = comprehension_target_from_store(instructions, store_index)
    if target is None:
        return None
    return ComprehensionLoopShape(
        instructions=instructions,
        offset_to_index=offset_to_index,
        load_iterator_index=load_iterator_index,
        for_iter_index=for_iter_index,
        loop_end_index=loop_end_index,
        store_index=store_index,
        target=target,
    )


def make_async_generator_expr(
    code: Any,
    iterable: ast.expr,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    load_iterator_index: int,
) -> ast.GeneratorExp | None:
    get_anext_index = skip_ignorable_instructions(
        instructions, load_iterator_index + 1, len(instructions)
    )
    if get_anext_index >= len(instructions):
        return None
    if instructions[get_anext_index].opname != "GET_ANEXT":
        return None

    store_index = find_async_generator_store(
        instructions,
        offset_to_index,
        get_anext_index,
        len(instructions),
    )
    if store_index is None:
        return None

    yield_wrap_index = find_async_generator_yield_wrap(
        instructions,
        store_index + 1,
        len(instructions),
    )
    if yield_wrap_index is None:
        return None

    target = ast.Name(
        id=safe_identifier(str(instructions[store_index].argval)),
        ctx=ast.Store(),
    )
    expression_start, conditions = read_generator_conditions(
        code,
        version,
        instructions,
        offset_to_index,
        store_index + 1,
        yield_wrap_index,
        int(instructions[get_anext_index].offset),
    )
    element = evaluate_generator_expression(
        code, version, instructions, expression_start, yield_wrap_index
    )
    if element is None:
        return None

    generator = make_generator(
        target=target,
        iterator=unwrap_async_iterator_call(iterable),
        is_async=True,
    )
    generator.ifs.extend(conditions)
    return make_generator_exp(element, [generator])


def find_async_generator_store(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_anext_index: int,
    end_index: int,
) -> int | None:
    if get_anext_index + 3 >= end_index:
        return None
    if instructions[get_anext_index + 1].opname != "LOAD_CONST":
        return None

    end_send_index = async_generator_end_send_index(
        instructions,
        offset_to_index,
        get_anext_index + 2,
        end_index,
    )
    if end_send_index is None:
        return None

    store_index = skip_ignorable_instructions(
        instructions,
        end_send_index + 1,
        end_index,
    )
    if store_index >= end_index or instructions[store_index].opname != "STORE_FAST":
        return None
    return store_index


def async_generator_end_send_index(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    send_index: int,
    end_index: int,
) -> int | None:
    send = instructions[send_index]
    if send.opname != "SEND":
        return None

    end_send_index = offset_to_index.get(int(send.argval))
    if end_send_index is None or end_send_index + 1 >= end_index:
        return None
    if instructions[end_send_index].opname != "END_SEND":
        return None
    return end_send_index


def find_async_generator_yield_wrap(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        instruction = instructions[index]
        if instruction.opname != "CALL_INTRINSIC_1":
            continue
        if intrinsic_1_name(instruction) == "INTRINSIC_ASYNC_GEN_WRAP":
            return index
    return None


def intrinsic_1_name(instruction: Instruction) -> str:
    if isinstance(instruction.argval, str):
        return instruction.argval
    if instruction.argrepr:
        return str(instruction.argrepr)
    if instruction.arg == 4:
        return "INTRINSIC_ASYNC_GEN_WRAP"
    return str(instruction.arg)


def unwrap_async_iterator_call(iterable: ast.expr) -> ast.expr:
    if not isinstance(iterable, ast.Call):
        return iterable
    if iterable.args or iterable.keywords:
        return iterable
    if not isinstance(iterable.func, ast.Attribute):
        return iterable
    if iterable.func.attr != "__aiter__":
        return iterable
    return iterable.func.value


def find_generator_iterator_load(instructions: list[Instruction]) -> int | None:
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_FAST":
            continue
        if instruction.argval == ".0":
            return index
    return None


def comprehension_target_from_store(
    instructions: list[Instruction],
    store_index: int,
) -> ast.expr | None:
    instruction = instructions[store_index]
    if instruction.opname in {"STORE_DEREF", "STORE_FAST"}:
        return ast.Name(
            id=safe_identifier(str(instruction.argval)),
            ctx=ast.Store(),
        )
    if instruction.opname not in {"UNPACK_SEQUENCE", "UNPACK_TUPLE"}:
        return None

    targets: list[ast.expr] = []
    cursor = store_index + 1
    count = int(instruction.arg or 0)
    for _index in range(count):
        cursor = skip_ignorable_instructions(instructions, cursor, len(instructions))
        if cursor >= len(instructions) or instructions[cursor].opname not in {
            "STORE_DEREF",
            "STORE_FAST",
        }:
            return None
        targets.append(
            ast.Name(
                id=safe_identifier(str(instructions[cursor].argval)),
                ctx=ast.Store(),
            )
        )
        cursor += 1
    return ast.Tuple(elts=targets, ctx=ast.Store())


def find_comprehension_update(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
    code_name: str,
) -> int | None:
    expected = {
        "<dictcomp>": "MAP_ADD",
        "<listcomp>": "LIST_APPEND",
        "<setcomp>": "SET_ADD",
    }.get(code_name)
    if expected is None:
        return None
    for index in range(start_index, end_index):
        if instructions[index].opname == expected:
            return index
    return None


def find_generator_yield(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname == "YIELD_VALUE":
            return index
    return None


def read_nested_generator_loop(
    code: Any,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    yield_index: int,
) -> tuple[ast.comprehension, int] | None:
    get_iter_index = find_nested_generator_get_iter(
        instructions,
        start_index,
        yield_index,
    )
    if get_iter_index is None:
        return None

    iterator = evaluate_generator_expression(
        code,
        version,
        instructions,
        start_index,
        get_iter_index,
    )
    if iterator is None:
        return None

    nested = read_nested_generator_loop_shape(
        instructions,
        offset_to_index,
        get_iter_index,
        yield_index,
    )
    if nested is None:
        return None
    target, store_index = nested
    return make_generator(target=target, iterator=iterator), store_index + 1


def read_nested_generator_loop_shape(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    get_iter_index: int,
    yield_index: int,
) -> tuple[ast.expr, int] | None:
    for_iter_index = skip_ignorable_instructions(
        instructions,
        get_iter_index + 1,
        yield_index,
    )
    if for_iter_index >= yield_index:
        return None
    for_iter = instructions[for_iter_index]
    if for_iter.opname != "FOR_ITER":
        return None

    loop_end_index = offset_to_index.get(int(for_iter.argval))
    if loop_end_index is None or loop_end_index <= for_iter_index:
        return None

    store_index = skip_ignorable_instructions(
        instructions,
        for_iter_index + 1,
        min(loop_end_index, yield_index),
    )
    if store_index >= yield_index:
        return None
    target = comprehension_target_from_store(instructions, store_index)
    if target is None:
        return None
    return target, store_index


def find_nested_generator_get_iter(
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index):
        if instructions[index].opname == "GET_ITER":
            return index
        if instructions[index].opname in {"FOR_ITER", "YIELD_VALUE"}:
            return None
    return None


def read_generator_conditions(
    code: Any,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    start_index: int,
    yield_index: int,
    loop_offset: int,
) -> tuple[int, list[ast.expr]]:
    cursor = start_index
    conditions: list[ast.expr] = []
    for _index in range(16):
        jump_index = find_generator_condition_jump(
            instructions, cursor, yield_index, loop_offset
        )
        if jump_index is None:
            return cursor, conditions

        condition = evaluate_generator_expression(
            code, version, instructions, cursor, jump_index
        )
        if condition is None:
            return start_index, []
        if "IF_FALSE" in instructions[jump_index].opname:
            condition = invert_condition(condition)
        conditions.append(condition)

        target_index = offset_to_index.get(int(instructions[jump_index].argval))
        if (
            target_index is None
            or target_index <= jump_index
            or target_index > yield_index
        ):
            return start_index, []
        cursor = target_index
    return start_index, []


def find_generator_condition_jump(
    instructions: list[Instruction],
    start_index: int,
    yield_index: int,
    loop_offset: int,
) -> int | None:
    for index in range(start_index, yield_index):
        instruction = instructions[index]
        if "JUMP" not in instruction.opname:
            continue
        if "IF_TRUE" not in instruction.opname and "IF_FALSE" not in instruction.opname:
            continue
        next_index = skip_ignorable_instructions(instructions, index + 1, yield_index)
        if next_index >= yield_index:
            return None
        next_instruction = instructions[next_index]
        if next_instruction.opname != "JUMP_BACKWARD":
            continue
        if int(next_instruction.argval) != loop_offset:
            continue
        return index
    return None


def evaluate_generator_expression(
    code: Any,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> ast.expr | None:
    conditional = evaluate_conditional_expression(
        code,
        version,
        instructions,
        start_index,
        end_index,
    )
    if conditional is not None:
        return conditional

    values = evaluate_expression_stack(
        code,
        version,
        instructions,
        start_index,
        end_index,
    )
    if not values:
        return None
    return values[-1]


def evaluate_conditional_expression(
    code: Any,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> ast.IfExp | None:
    offset_to_index = make_offset_index(instructions)
    for jump_index in range(start_index, end_index):
        jump = instructions[jump_index]
        if not jump.opname.startswith("POP_JUMP"):
            continue

        bridge_index = skip_ignorable_instructions(
            instructions,
            jump_index + 1,
            end_index,
        )
        if bridge_index >= end_index:
            continue
        bridge = instructions[bridge_index]
        if bridge.opname not in {"JUMP_FORWARD", "JUMP"}:
            continue

        body_start = offset_to_index.get(int(bridge.argval))
        orelse_start = offset_to_index.get(int(jump.argval))
        if body_start is None or orelse_start is None:
            continue
        if body_start <= bridge_index or orelse_start <= jump_index:
            continue

        body_end = find_conditional_expression_body_end(
            instructions,
            offset_to_index,
            body_start,
            orelse_start,
            end_index,
        )
        if body_end is None:
            continue

        test = evaluate_generator_expression(
            code,
            version,
            instructions,
            start_index,
            jump_index + 1,
        )
        body = evaluate_generator_expression(
            code,
            version,
            instructions,
            body_start,
            body_end,
        )
        orelse = evaluate_generator_expression(
            code,
            version,
            instructions,
            orelse_start,
            end_index,
        )
        if test is None or body is None or orelse is None:
            return None

        return ast.IfExp(test=test, body=body, orelse=orelse)
    return None


def find_conditional_expression_body_end(
    instructions: list[Instruction],
    offset_to_index: dict[int, int],
    body_start: int,
    orelse_start: int,
    end_index: int,
) -> int | None:
    if body_start >= orelse_start:
        return None
    for index in range(body_start, orelse_start):
        instruction = instructions[index]
        if instruction.opname not in {"JUMP_FORWARD", "JUMP"}:
            continue
        target_index = offset_to_index.get(int(instruction.argval))
        if target_index == end_index:
            return index
    return None


def evaluate_expression_stack(
    code: Any,
    version: tuple[int, ...] | None,
    instructions: list[Instruction],
    start_index: int,
    end_index: int,
) -> list[ast.expr]:
    from pyc2py.decompiler.engine import NativeDecompiler

    child = NativeDecompiler(code=code, version=version, is_module=False)
    starting_depth = len(child.stack)
    pending_conditions: list[ast.expr] = []
    for instruction in instructions[start_index:end_index]:
        if run_expression_jump(child, instruction, pending_conditions):
            continue
        child.run_instruction(instruction)
    merge_expression_jump_conditions(child, pending_conditions)

    if len(child.stack) <= starting_depth:
        return []
    return [coerce_expr(value) for value in child.stack[starting_depth:]]


def run_expression_jump(
    decompiler: Any,
    instruction: Instruction,
    pending_conditions: list[ast.expr],
) -> bool:
    if "JUMP" not in instruction.opname:
        return False
    if instruction.opname.startswith("POP_JUMP"):
        pending_conditions.append(coerce_expr(decompiler.pop_or_none()))
    return True


def merge_expression_jump_conditions(
    decompiler: Any,
    pending_conditions: list[ast.expr],
) -> None:
    if not pending_conditions:
        return

    current = coerce_expr(decompiler.stack.pop()) if decompiler.stack else None
    for condition in reversed(pending_conditions):
        if current is None:
            current = condition
            continue
        merged = merge_chained_compare(condition, current)
        if merged is None:
            if decompiler.stack:
                decompiler.stack.append(current)
            return
        current = merged
    if current is not None:
        decompiler.stack.append(current)


def merge_chained_compare(left: ast.expr, right: ast.expr) -> ast.Compare | None:
    if not isinstance(left, ast.Compare) or not isinstance(right, ast.Compare):
        return None
    if not left.comparators or len(right.ops) != 1 or len(right.comparators) != 1:
        return None
    if expression_key(left.comparators[-1]) != expression_key(right.left):
        return None
    return ast.Compare(
        left=left.left,
        ops=[*left.ops, *right.ops],
        comparators=[*left.comparators, *right.comparators],
    )


def coerce_expr(value: Any) -> ast.expr:
    coerced = coerce_runtime_value(value)
    if coerced is not None:
        return coerced
    if isinstance(value, ast.expr):
        if is_code_constant(value):
            return ast.Constant(value=None)
        return value
    return ast.Constant(value=None)


def coerce_runtime_value(value: Any) -> ast.expr | None:
    if isinstance(value, BuildClassValue):
        return make_name("__build_class__", ast.Load())
    if isinstance(value, FunctionValue | ClassValue):
        return ast.Constant(value=None)
    if isinstance(value, TypeAliasValue):
        return make_type_alias_call_expr(value)
    if isinstance(value, ImportValue | ImportedAttributeValue):
        return coerce_import_expr(value)
    if isinstance(value, UnpackSlot):
        return ast.Subscript(
            value=value.group.value,
            slice=ast.Constant(value=value.index),
            ctx=ast.Load(),
        )
    return None


def coerce_import_expr(value: ImportValue | ImportedAttributeValue) -> ast.expr:
    if isinstance(value, ImportValue):
        return make_name(value.name.split(".")[0], ast.Load())
    return make_name(value.name, ast.Load())


def make_type_alias_call_expr(value: TypeAliasValue) -> ast.Call:
    type_params: ast.expr
    if value.type_params:
        type_params = ast.Tuple(elts=list(value.type_params), ctx=ast.Load())
    else:
        type_params = ast.Constant(value=None)
    return ast.Call(
        func=ast.Name(id="TypeAliasType", ctx=ast.Load()),
        args=[ast.Constant(value=value.name), value.value],
        keywords=[ast.keyword(arg="type_params", value=type_params)],
    )


def parse_body_or_empty(result: Any | None) -> ast.Module:
    if result is None:
        return ast.Module(body=[], type_ignores=[])
    try:
        return ast.parse(result.source)
    except SyntaxError:
        return ast.Module(body=[], type_ignores=[])


def is_coroutine_code(code: Any) -> bool:
    flags = int(getattr(code, "co_flags", 0) or 0)
    return bool(flags & 0x280)


def make_slice(start: Any, stop: Any) -> ast.Slice:
    lower = none_to_empty(start)
    upper = none_to_empty(stop)
    return ast.Slice(lower=lower, upper=upper, step=None)


def none_to_empty(value: Any) -> ast.expr | None:
    expr = coerce_expr(value)
    if isinstance(expr, ast.Constant) and expr.value is None:
        return None
    return expr


def unwrap_lazy_expr(value: ast.expr) -> ast.expr:
    if isinstance(value, ast.Lambda):
        return value.body
    return value
