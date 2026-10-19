import ast
import base64
import binascii
import bz2
import codecs
import lzma
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyc2py.constants import MAX_LAYER_BYTES
from pyc2py.decompiler.engine import decompile_native_source
from pyc2py.pyc.code import code_objects

RUNNERS = ("exec", "eval")
FAILURES = (
    ValueError,
    TypeError,
    EOFError,
    LookupError,
    RecursionError,
    OSError,
    zlib.error,
    lzma.LZMAError,
)


class NotStatic(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Marshalled:
    data: bytes


@dataclass(frozen=True, slots=True)
class Layer:
    # the expression the wrapper runs, as written
    how: str
    # what it unpacks to: source text, or a marshalled code object
    value: str | Marshalled


@dataclass(frozen=True, slots=True)
class Module:
    name: str


@dataclass(frozen=True, slots=True)
class Function:
    name: str


@dataclass(frozen=True, slots=True)
class Method:
    owner: str | bytes
    name: str


@dataclass(frozen=True, slots=True)
class Compiled:
    source: str | bytes


def module_tree(code: Any, version: tuple[int, ...] | None) -> ast.Module | None:
    # only a module naming exec or eval can wrap code, so only it is read again
    if not any(
        name in RUNNERS
        for item in code_objects(code)
        for name in getattr(item, "co_names", ())
    ):
        return None

    result = decompile_native_source(code, version)
    if result is None or result.body is None:
        return None

    return ast.Module(body=list(result.body), type_ignores=[])


def wrapped_layers(module: ast.Module) -> list[Layer]:
    # the code a module hands to exec or eval, found without running anything
    bound = bound_names(module)
    names = static_names(module, bound)
    layers: list[Layer] = []

    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in RUNNERS
            and node.func.id not in bound
            and node.args
        ):
            continue

        try:
            value = evaluate(node.args[0], names)
        except (NotStatic, *FAILURES):
            continue

        layer = as_layer(value, ast.unparse(node.args[0]))
        if layer is not None:
            layers.append(layer)

    return layers


def as_layer(value: Any, how: str) -> Layer | None:
    if isinstance(value, Compiled):
        value = value.source

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    return Layer(how, value) if isinstance(value, (str, Marshalled)) else None


def bound_names(module: ast.Module) -> Counter[str]:
    bound: Counter[str] = Counter()

    for node in ast.walk(module):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound[node.id] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound[node.name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )

    return bound


def static_names(module: ast.Module, bound: Counter[str]) -> dict[str, Any]:
    # imports, names bound once to a known value, and builtins the module keeps
    names: dict[str, Any] = {
        name: value for name, value in BUILTINS.items() if name not in bound
    }

    for statement in module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                name = alias.asname or alias.name.split(".")[0]
                if bound[name] == 1:
                    names[name] = Module(alias.name)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            for alias in statement.names:
                name = alias.asname or alias.name
                if bound[name] == 1:
                    names[name] = Function(f"{statement.module}.{alias.name}")
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and bound[statement.targets[0].id] == 1
        ):
            try:
                names[statement.targets[0].id] = evaluate(statement.value, names)
            except (NotStatic, *FAILURES):
                continue

    return names


def evaluate(node: ast.expr, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes, int)):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]

    if isinstance(node, ast.Attribute):
        owner = evaluate(node.value, names)
        if isinstance(owner, Module):
            return Function(f"{owner.name}.{node.attr}")
        if isinstance(owner, (str, bytes)):
            return Method(owner, node.attr)
    elif isinstance(node, ast.Subscript):
        return reversed_value(evaluate(node.value, names), node.slice)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = evaluate(node.left, names), evaluate(node.right, names)
        if type(left) is type(right) and isinstance(left, (str, bytes)):
            return bounded(left + right)
    elif isinstance(node, ast.Call) and not node.keywords:
        function = evaluate(node.func, names)
        return call(function, [evaluate(argument, names) for argument in node.args])

    raise NotStatic(type(node).__name__)


def reversed_value(value: Any, index: ast.expr) -> Any:
    if (
        isinstance(index, ast.Slice)
        and index.lower is None
        and index.upper is None
        and isinstance(index.step, ast.UnaryOp)
        and isinstance(index.step.op, ast.USub)
        and isinstance(index.step.operand, ast.Constant)
        and index.step.operand.value == 1
        and isinstance(value, (str, bytes))
    ):
        return value[::-1]

    raise NotStatic("subscript")


def call(function: Any, arguments: list[Any]) -> Any:
    if isinstance(function, Method) and function.name in METHODS:
        return bounded(METHODS[function.name](function.owner, *arguments))
    if isinstance(function, Function) and function.name in DECODERS:
        return bounded(DECODERS[function.name](*arguments))

    raise NotStatic("call")


def bounded(value: Any) -> Any:
    if isinstance(value, (str, bytes)) and len(value) > MAX_LAYER_BYTES:
        raise ValueError("the layer is larger than pyc2py unpacks")

    return value


def as_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("latin-1")
    if isinstance(value, bytes):
        return value

    raise TypeError(type(value).__name__)


def decompressed(stream: Any, data: Any) -> bytes:
    # the limit stops a small blob from unpacking into a huge one
    return bytes(bounded(stream.decompress(as_bytes(data), MAX_LAYER_BYTES + 1)))


def inflated(data: Any, wbits: int = zlib.MAX_WBITS) -> bytes:
    return decompressed(zlib.decompressobj(wbits), data)


def decoded(value: Any, encoding: str = "utf-8") -> Any:
    codec = CODECS.get(str(encoding).lower().replace("-", "_"))
    if codec is not None:
        return codec(value)

    return as_bytes(value).decode(str(encoding))


def on_bytes(decode: Callable[[bytes], bytes]) -> Callable[[object], bytes]:
    return lambda data: decode(as_bytes(data))


def rot13(value: object) -> str:
    return codecs.decode(str(value), "rot13")


def from_hex(text: object) -> bytes:
    return bytes.fromhex(str(text))


def bz2_decompressed(data: object) -> bytes:
    return decompressed(bz2.BZ2Decompressor(), data)


def lzma_decompressed(data: object) -> bytes:
    return decompressed(lzma.LZMADecompressor(), data)


def gzip_decompressed(data: object) -> bytes:
    return inflated(data, 16 + zlib.MAX_WBITS)


def marshalled(data: object) -> Marshalled:
    return Marshalled(as_bytes(data))


def compiled(source: object, *_rest: object) -> Compiled:
    if not isinstance(source, str | bytes):
        raise NotStatic("compile")

    return Compiled(source)


def imported(name: object, *_rest: object) -> Module:
    return Module(str(name))


def encoded(owner: object, *args: str) -> bytes:
    return str(owner).encode(*args)


CODECS: dict[str, Callable[[object], object]] = {
    "rot13": rot13,
    "rot_13": rot13,
    "base64": on_bytes(base64.b64decode),
    "hex": on_bytes(binascii.unhexlify),
    "zlib": inflated,
    "zip": inflated,
    "bz2": bz2_decompressed,
}
DECODERS: dict[str, Callable[..., object]] = {
    "base64.b64decode": on_bytes(base64.b64decode),
    "base64.standard_b64decode": on_bytes(base64.b64decode),
    "base64.urlsafe_b64decode": on_bytes(base64.urlsafe_b64decode),
    "base64.b32decode": on_bytes(base64.b32decode),
    "base64.b16decode": on_bytes(base64.b16decode),
    "base64.b85decode": on_bytes(base64.b85decode),
    "base64.a85decode": on_bytes(base64.a85decode),
    "base64.decodestring": on_bytes(base64.b64decode),
    "base64.decodebytes": on_bytes(base64.b64decode),
    "binascii.unhexlify": on_bytes(binascii.unhexlify),
    "binascii.a2b_hex": on_bytes(binascii.unhexlify),
    "binascii.a2b_base64": on_bytes(binascii.a2b_base64),
    "bytes.fromhex": from_hex,
    "zlib.decompress": inflated,
    "bz2.decompress": bz2_decompressed,
    "lzma.decompress": lzma_decompressed,
    "gzip.decompress": gzip_decompressed,
    "codecs.decode": decoded,
    "marshal.loads": marshalled,
    "compile": compiled,
    "__import__": imported,
}
METHODS: dict[str, Callable[..., object]] = {"decode": decoded, "encode": encoded}
BUILTINS: dict[str, Any] = {
    "compile": Function("compile"),
    "__import__": Function("__import__"),
    "bytes": Module("bytes"),
}
