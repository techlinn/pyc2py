# pyc2py

`pyc2py` is a Python bytecode decompiler.

It takes `.pyc` files and writes recovered `.py` files. You can point it at one
file, or at a folder of `.pyc` files. The project is built around native
bytecode to AST recovery, version aware opcode tables, old marshal/code-object
reading, and validation that tries to catch bad output before you trust it. The
validation is strict about internal placeholders: generated source with
unresolved `__pyc2py_*` helper calls should fail instead of looking clean.

This is beta software. Some files will decompile cleanly. Some files will
produce warnings. Some files will fail. That is expected at this stage, and the
warnings are meant to be useful enough to fix the decompiler instead of hiding
the broken part.

`pyc2py` is not claiming to beat `pycdc`, `uncompyle6`, or any other decompiler
yet. That claim needs a real public corpus and side by side results. For now the
goal is simpler: make the output compile, make the warnings honest, and keep
improving the recovery logic.

## install

Use Python 3.10 or newer.

The decompiler has no runtime dependencies.

For local development:

```bash
python -m pip install -e ".[dev]"
```

The current dev extras are only formatting/static check tools.

## cli

Decompile one file:
```bash
python main.py path/to/file.pyc
```

That writes beside the input file:
```text
path/to/file.pyc.py
```

Decompile one file into a folder:
```bash
python main.py path/to/file.pyc --out-dir recovered
```

Decompile a folder of `.pyc` files:
```bash
python main.py path/to/pyc_folder --out-dir recovered
```

Scan nested folders too:
```bash
python main.py path/to/pyc_folder --out-dir recovered --recursive
```

The positional input can also be written with `--input`:
```bash
python main.py --input path/to/pyc_folder --output-dir recovered
```

Folder mode requires `--out-dir`. `--recursive` only works with folder input.

## console output

The CLI clears the console between phases so a run is easier to follow:

```text
phase 0 - job setup
phase 1 - input checks
phase 2 - loading bytecode
phase 3 - recovering source
phase 4 - writing output
phase 5 - validating result
phase 6 - final report
```

At the end, the final report prints every warning and every error. It does not
shorten the list. If a file fails, paste the final diagnostics when reporting
the issue.

## python api

The small API in `pyc2py/api.py` wraps the same pipeline used by the CLI:

```python
from pyc2py.api import decompile, decompile_folder, decompile_to_folder

result = decompile("example.pyc")
folder_results = decompile_folder("pyc_files", "recovered", recursive=True)
```

Each result contains:

- `output.path`
- `output.source`
- `output.strategy`
- `output.warnings`
- `report.passed`
- `report.warnings`
- `report.errors`
- `report.checks`

## how it works

The main flow is:

```text
load .pyc
decode code object
decode bytecode instructions
recover an AST
unparse and format source
write .py file
validate generated output
print full diagnostics
```

The important modules are:

```text
main.py                         cli, phases, final diagnostics
pyc2py/api.py                   public python wrapper
pyc2py/pipeline.py              file-to-source pipeline
pyc2py/types.py                 shared result/report data

pyc2py/pyc/loader.py            native and legacy pyc loading
pyc2py/pyc/header.py            magic/header metadata
pyc2py/pyc/magic.py             magic number to version lookup
pyc2py/pyc/marshal_reader.py    legacy marshal object reader
pyc2py/pyc/marshal_code_reader.py old code-object shapes

pyc2py/bytecode/opcode_table.py versioned opcode tables
pyc2py/bytecode/decoder.py      native and legacy instruction decoding
pyc2py/bytecode/operand.py      operand resolution
pyc2py/bytecode/scanner.py      raw bytecode scanning
pyc2py/bytecode/stack_effect.py stack-effect tables
pyc2py/bytecode/versions/       per-version opcode maps

pyc2py/decompiler/engine.py     native bytecode-to-AST engine
pyc2py/decompiler/control.py    structured control-flow recovery
pyc2py/decompiler/control_loop.py loop recovery
pyc2py/decompiler/exception_structures.py exception recovery
pyc2py/decompiler/opcodes/      opcode handlers
pyc2py/decompiler/runtime.py    runtime values used during recovery

pyc2py/source.py                source formatting and source validation
pyc2py/reporting.py             bytecode/cfg/stack/source checks
pyc2py/cfg.py                   control-flow graph helpers
pyc2py/stack.py                 stack simulation and stack validation
pyc2py/astree.py                AST cleanup and AST shape checks
```

## supported bytecode

The project has opcode tables for Python 1.0 through Python 3.15.

That does not mean every version has the same reliability. Newer versions can
use the host Python `dis` and `marshal` support when the `.pyc` matches the
running interpreter. Older versions go through the custom marshal reader and
versioned opcode tables.

For Python 3 output, `pyc2py` validates by parsing and compiling the generated
source. For Python 1.x and 2.x output, the host Python cannot compile old syntax,
so validation is limited to formatting and bytecode/code-object checks.

## validation

A passing run means the generated source passed the checks that apply to that
file. It does not mean the output is byte-for-byte or behavior-for-behavior equal
to the original source.

Current checks include:

- generated source parse/compile checks for Python 3 source
- unresolved internal `__pyc2py_*` helper detection
- source formatting checks
- recovered AST shape checks
- bytecode decode checks
- operand and jump target checks
- line table checks
- CFG checks
- stack-effect checks
- exception table checks where available

Warnings matter. If the final report has warnings, inspect the output before
using it.

Recent reliability work tightened two small but important cases:

- match pattern capture names now count as real local assignments, so the
  decompiler should not inject fake `name = None` initializers for captures like
  `case [x]` or `case {"a": a, **rest}`
- unresolved internal helper calls now fail validation, because code that calls
  a missing `__pyc2py_*` helper can compile but still crash at runtime

## known limits

The weak spots are the normal hard parts of Python decompilation:

- exception cleanup
- unusual control flow
- nested comprehensions and generators
- async bytecode
- pattern matching
- superinstructions and changed opcodes in newer Python versions
- very old Python syntax that modern Python cannot compile
- generated source that compiles but is not semantically identical
- source that still needs an internal helper means the recovery logic is not done

There is no deobfuscation or automatic decryption layer in this release. This
release is about decompiling `.pyc` files, not unpacking protected programs.

## contributing

Keep changes focused on decompilation reliability.

Good fixes usually improve one of these areas:

- opcode table accuracy
- marshal/code-object parsing
- operand resolution
- structured control-flow recovery
- exception recovery
- AST cleanup
- validation warnings that point at real decompiler bugs

Avoid adding generated outputs, caches, or one-off sample files to commits.

Ignored/generated files include:

```text
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.pyc
*.pyc.py
tmp_*/
build/
dist/
```

Before pushing, run:

```bash
python -B -c "import ast, pathlib; paths=[pathlib.Path('main.py'), *pathlib.Path('pyc2py').rglob('*.py')]; [ast.parse(path.read_text(encoding='utf-8'), filename=str(path)) for path in paths]; import pyc2py.pipeline; import pyc2py.decompiler; import pyc2py.bytecode; import pyc2py.pyc; print('ok')"
```

If you run manual `.pyc` smoke tests, remove the generated files before
committing.

## tests

The test suite is self-generating: it compiles `.py` snippets to `.pyc` in
memory (and via any other `python3.x` interpreters found on `PATH`), runs the
full pipeline, and asserts the output compiles and contains no unresolved
`__pyc2py_*` helper. No `.pyc` binaries are committed.

```bash
python -m pip install -e ".[dev]"
ruff check .
pyflakes pyc2py main.py
pytest -q
```

Snippets live in `tests/snippets/`. Known-hard recovery cases are marked
`xfail`, so fixing one shows up as an unexpected pass instead of silently
regressing. CI runs the same checks on a Python 3.10–3.13 matrix.
