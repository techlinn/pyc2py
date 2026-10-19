import atexit
import subprocess
from pathlib import Path

from pyc2py.bytecode.pypy import pypy_shared_magics
from pyc2py.pyc.code import PycCode
from pyc2py.pyc.magic import find_format
from pyc2py.pyc.marshal_reader import load_marshal_code
from pyc2py.types import OUTPUT_OPTIONS, VersionTuple

# runs in the interpreter it compiles for, from Python 2.7 on, over length lines
HELPER = r"""
import marshal, sys
stdin = getattr(sys.stdin, "buffer", sys.stdin)
stdout = getattr(sys.stdout, "buffer", sys.stdout)
def send(kind, data):
    stdout.write(("%d\n" % (len(data) + 1)).encode("ascii") + kind + data)
    stdout.flush()
try:
    from importlib.util import MAGIC_NUMBER as magic
except ImportError:
    from imp import get_magic
    magic = get_magic()
first, second = bytearray(magic[:2])
name = sys.implementation.name if hasattr(sys, "implementation") else "cpython"
send(b"V", ("%d %s" % (first | second << 8, name)).encode("ascii"))
while True:
    head = stdin.readline()
    if not head:
        break
    source = stdin.read(int(head))
    try:
        send(b"C", marshal.dumps(compile(source, "<pyc2py>", "exec", 0, True)))
    except Exception as error:
        send(b"E", ("%s: %s" % (type(error).__name__, error)).encode("utf-8"))
"""


class CompilerError(Exception):
    pass


class Compiler:
    def __init__(self, python: Path) -> None:
        self.python = python

        try:
            self.process = subprocess.Popen(
                [str(python), "-E", "-s", "-c", HELPER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise CompilerError(f"{python} could not be started: {error}") from None

        try:
            self.version, self.opcodes = self.written_format()
        except CompilerError:
            self.close()
            raise

    def written_format(self) -> tuple[VersionTuple, str]:
        kind, data = self.receive()
        magic, _, implementation = data.decode("ascii", "replace").partition(" ")
        found = find_format(int(magic)) if kind == b"V" and magic.isdigit() else None
        if found is not None and implementation == "pypy" and not found[1]:
            opcodes = pypy_shared_magics().get(int(magic))
            found = None if opcodes is None else (found[0], opcodes)

        if found is None:
            raise CompilerError(f"{self.python} is not a Python pyc2py reads")

        return found

    def compile(self, source: str) -> PycCode:
        data = source.encode("utf-8")

        try:
            stdin = self.process.stdin
            assert stdin is not None
            stdin.write(b"%d\n" % len(data) + data)
            stdin.flush()
        except OSError as error:
            raise CompilerError(f"{self.python} stopped: {error}") from None

        kind, answer = self.receive()
        if kind != b"C":
            raise CompilerError(answer.decode("utf-8", "replace"))

        try:
            return load_marshal_code(answer, self.version, None, self.opcodes)
        except (EOFError, ValueError) as error:
            reason = f"{self.python} wrote code pyc2py cannot read: {error}"
            raise CompilerError(reason) from error

    def receive(self) -> tuple[bytes, bytes]:
        stdout = self.process.stdout
        assert stdout is not None

        try:
            head = stdout.readline()
            message = stdout.read(int(head)) if head.strip().isdigit() else b""
        except OSError:
            message = b""

        if not message:
            raise CompilerError(f"{self.python} stopped")

        return message[:1], message[1:]

    def close(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                pipe.close()

        self.process.wait()


RUNNING: dict[Path, Compiler | CompilerError] = {}


@atexit.register
def stop_compilers() -> None:
    while RUNNING:
        compiler = RUNNING.popitem()[1]
        if isinstance(compiler, Compiler):
            compiler.close()


def started(python: Path) -> Compiler | CompilerError:
    if python not in RUNNING:
        try:
            RUNNING[python] = Compiler(python)
        except CompilerError as error:
            RUNNING[python] = error

    return RUNNING[python]


def compiler_for(version: VersionTuple | None, opcodes: str) -> Compiler | None:
    if version is None:
        return None

    for python in OUTPUT_OPTIONS.get().verify_with:
        compiler = started(python)
        if isinstance(compiler, Compiler) and (compiler.version, compiler.opcodes) == (
            tuple(version[:2]),
            opcodes,
        ):
            return compiler

    return None
