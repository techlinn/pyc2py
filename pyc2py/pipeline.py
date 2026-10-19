import ast
import os
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from pyc2py.astree import SOURCE_LINE_LIMIT
from pyc2py.bytecode.decoder import decoding_once
from pyc2py.bytecode.formats import pypy_format
from pyc2py.compilers import compiler_for
from pyc2py.constants import MAX_SOURCE_LINES
from pyc2py.containers import PYC_SUFFIXES, ContainerError, container_members
from pyc2py.disassembly import disassemble
from pyc2py.pyc.loader import (
    PycLoadError,
    load_pyc_data,
    read_input,
    reading_version,
)
from pyc2py.pyc.magic import find_format, version_in_development
from pyc2py.reporting import validate_generated_source
from pyc2py.source import emit_source
from pyc2py.types import (
    FAILED_UNIT_STATUSES,
    FILE_STATUSES,
    OUTPUT_OPTIONS,
    DecompiledSource,
    LiveWarningList,
    OutputOptions,
    ProgressCallback,
    PycHeader,
    PycModule,
    StatedCode,
    VerificationReport,
    VersionTuple,
    emit_progress,
)
from pyc2py.unpacking import Marshalled, module_tree, wrapped_layers

# covers the deepest expression CPython compiles (UPGRADE.md N7)
RECURSION_LIMIT = 200_000


@contextmanager
def deep_recursion() -> Generator[None]:
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, RECURSION_LIMIT))

    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


@dataclass(frozen=True, slots=True)
class UnpackedLayer:
    path: Path
    # None for source text, which is written as the wrapper holds it
    result: DecompileResult | None = None


@dataclass(frozen=True, slots=True)
class DecompileResult:
    output: DecompiledSource
    report: VerificationReport
    # code the file unpacks and runs, each written beside the output
    layers: tuple[UnpackedLayer, ...] = ()

    @property
    def status(self) -> str:
        statuses = [self.report.status]
        statuses += [layer.result.status for layer in self.layers if layer.result]
        return max(statuses, key=FILE_STATUSES.index)


def decompile_file(path: Path) -> DecompileResult:
    return decompile_file_to_path(path, path.with_suffix(path.suffix + ".py"))


def decompile_file_to_directory(
    path: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
) -> DecompileResult:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    return decompile_file_to_path(
        path, output_dir / f"{path.name}.py", progress, version
    )


def decompile_file_to_path(
    path: Path,
    target: Path,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
) -> DecompileResult:
    emit_progress(progress, f"checking input: {path}")
    return decompile_data_to_path(read_input(path), path, target, progress, version)


def decompile_data_to_path(
    data: bytes,
    path: Path,
    target: Path,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
    stated: StatedCode | None = None,
) -> DecompileResult:
    with deep_recursion(), decoding_once():
        return decompile_data(data, path, target, progress, version, stated)


def decompile_data(
    data: bytes,
    path: Path,
    target: Path,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
    stated: StatedCode | None = None,
) -> DecompileResult:
    # version is the user's --python-version; stated is what a container says
    if target.exists() and target.is_dir():
        raise IsADirectoryError(target)

    stated = stated or StatedCode()
    emit_progress(progress, "loading pyc header and code object")
    module = load_pyc_data(data, path, version, stated)
    emit_progress(progress, describe_module(module))

    if OUTPUT_OPTIONS.get().write_disassembly:
        write_disassembly(module, target, progress)

    # this list prints warnings live and keeps the same text for the final report
    warnings = LiveWarningList(progress)

    if version is not None:
        warnings.append(forced_version_note(module))
    if module.header.doubt:
        warnings.append(module.header.doubt)

    emit_progress(progress, "recovering Python source")
    size = module.header.source_size
    # a source of n bytes has at most n + 1 lines; 0 is what tools write when unknown
    limit = SOURCE_LINE_LIMIT.set(size + 1 if size else MAX_SOURCE_LINES)

    try:
        source, strategy, _source_warnings = emit_source(module, progress, warnings)
    finally:
        SOURCE_LINE_LIMIT.reset(limit)

    compiler = compiler_for(module.header.version, module.header.opcodes)
    if compiler is None and OUTPUT_OPTIONS.get().verify_with:
        warnings.append(
            f"no --verify-with interpreter writes {bytecode_name(module.header)}"
            " bytecode, so the file was checked by this Python only"
        )

    emit_progress(progress, "validating generated source and bytecode recovery")
    report = validate_generated_source(
        path,
        target,
        source,
        warnings,
        reading_version(module.header),
        module.code,
        progress,
        compiler,
    )
    emit_progress(progress, f"validation {report.status}")
    # appended, since prepending would shift __firstlineno__
    source = source + closing_record(module, report)
    emit_progress(progress, f"creating output folder: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    emit_progress(progress, f"writing source: {target}")
    target.write_text(source, encoding="utf-8")
    tree = module_tree(module.code, reading_version(module.header))
    layers = () if tree is None else write_layers(tree, module, path, target, progress)
    return DecompileResult(
        output=DecompiledSource(
            path=target,
            source=source,
            strategy=strategy,
            warnings=tuple(warnings),
        ),
        report=report,
        layers=layers,
    )


def write_disassembly(
    module: PycModule, target: Path, progress: ProgressCallback | None
) -> None:
    listing = target.with_suffix(".dis")
    emit_progress(progress, f"writing disassembly: {listing}")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text(disassemble(module, describe_origin(module)), encoding="utf-8")


# a wrapper that wraps itself deeper than this is left at that depth
MAX_LAYER_DEPTH = 8
LAYER_DEPTH: ContextVar[int] = ContextVar("LAYER_DEPTH", default=0)


def write_layers(
    tree: ast.Module,
    module: PycModule,
    path: Path,
    target: Path,
    progress: ProgressCallback | None,
) -> tuple[UnpackedLayer, ...]:
    depth = LAYER_DEPTH.get() + 1
    if depth > MAX_LAYER_DEPTH:
        return ()

    written: list[UnpackedLayer] = []
    token = LAYER_DEPTH.set(depth)

    try:
        for number, layer in enumerate(wrapped_layers(tree), start=1):
            layer_target = target.with_name(f"{target.stem}.layer{number}.py")
            emit_progress(progress, f"unpacking {layer.how}: {layer_target}")

            if isinstance(layer.value, Marshalled):
                header = module.header

                try:
                    result = decompile_data(
                        layer.value.data,
                        path,
                        layer_target,
                        progress,
                        stated=StatedCode(header.version, opcodes=header.opcodes),
                    )
                except PycLoadError as error:
                    details = f"unpacked layer {layer_target.name}: {error}"
                    result = failed_result(path, layer_target, details)

                written += [UnpackedLayer(layer_target, result), *result.layers]
                continue

            layer_target.write_text(layer.value, encoding="utf-8")
            written.append(UnpackedLayer(layer_target))

            try:
                inner = ast.parse(layer.value)
            except SyntaxError:
                continue

            written += write_layers(inner, module, path, layer_target, progress)
    finally:
        LAYER_DEPTH.reset(token)

    return tuple(written)


RECORD_START = "\n# pyc2py: recovered from "


def closing_record(module: PycModule, report: VerificationReport) -> str:
    lines = provenance_lines(module)

    if not report.passed:
        lines += verification_lines(report)

    return "\n" + "\n".join(lines) + "\n"


def without_closing_record(text: str) -> str:
    start = text.rfind(RECORD_START)
    return text if start < 0 else text[:start]


def provenance_lines(module: PycModule) -> list[str]:
    header = module.header
    lines = [f"{RECORD_START.lstrip()}{describe_origin(module)}"]

    if header.source_hash is not None:
        checked = "checked" if header.checks_source else "not checked"
        lines.append(
            f"#   source: hash {header.source_hash.hex()}, {checked} on import"
        )
    elif header.timestamp is not None:
        size = "" if header.source_size is None else f"{header.source_size} bytes, "
        modified = datetime.fromtimestamp(header.timestamp, UTC)
        lines.append(f"#   source: {size}modified {modified:%Y-%m-%d %H:%M:%S} UTC")

    return lines


def describe_origin(module: PycModule) -> str:
    header = module.header
    if header.magic_int is None:
        origin = f"a {bytecode_name(header)} code object with no pyc header"
    else:
        origin = f"{bytecode_name(header)} bytecode (magic {header.magic_int})"

    filename = str(getattr(module.code, "co_filename", "") or "")
    if not filename:
        return origin

    return f"{origin}, compiled from {comment_text(filename)}"


def comment_text(text: str) -> str:
    # text from the input must not end the comment it is written in
    return text if text.isprintable() else repr(text)


def verification_lines(report: VerificationReport) -> list[str]:
    if report.status == "failed":
        lines = [
            "# pyc2py: verification FAILED; parts of this file differ from the original"
        ]
    else:
        lines = [
            "# pyc2py: NOT VERIFIED; this file could not be compared with the original"
        ]

    lines += [f"#   error: {comment_text(error)}" for error in report.errors]
    lines += [f"#   {comment_text(line)}" for line in unit_lines(report)]
    return lines


def unit_lines(report: VerificationReport) -> list[str]:
    lines = [
        f"{unit.status}: {unit.name} ({unit.reason})"
        for unit in report.units
        if unit.status in FAILED_UNIT_STATUSES
    ]
    unverified = [unit for unit in report.units if unit.status == "unverified"]
    if unverified:
        lines.append(
            f"unverified: {len(unverified)} of {len(report.units)} code objects"
            f" ({unverified[0].reason})"
        )

    return lines


def bytecode_name(header: PycHeader) -> str:
    if header.opcodes:
        return pypy_format(header.opcodes).name

    return "Python " + ".".join(map(str, header.version or ()))


def forced_version_note(module: PycModule) -> str:
    header = module.header
    name = ".".join(map(str, header.version or ()))
    known = None if header.magic_int is None else find_format(header.magic_int)

    if header.magic_int is None:
        found = "the file has no pyc header"
    elif known == (header.version, ""):
        found = "which matches its header"
    elif known is not None:
        version, opcodes = known
        other = bytecode_name(replace(header, version=version, opcodes=opcodes))
        found = f"its header says {other} (magic {header.magic_int})"
    elif (development := version_in_development(header.magic_int)) is not None:
        other = ".".join(map(str, development))
        found = (
            f"its header has magic {header.magic_int}, from a pre-release or"
            f" patched build of Python {other}"
        )
    else:
        found = f"its header has magic {header.magic_int}, not a CPython magic"

    return f"read as Python {name}, as requested with --python-version; {found}"


def find_pyc_files(source_dir: Path, recursive: bool) -> list[Path]:
    found = source_dir.rglob("*") if recursive else source_dir.glob("*")
    return sorted(
        path for path in found if path.suffix.lower() in PYC_SUFFIXES and path.is_file()
    )


@dataclass(frozen=True, slots=True)
class BatchJob:
    source: Path
    target: Path
    data: bytes | None = None
    stated: StatedCode = field(default_factory=StatedCode)
    failure: str | None = None


def container_jobs(
    path: Path, output_dir: Path, version: VersionTuple | None = None
) -> list[BatchJob]:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    jobs = [
        BatchJob(
            path / member.path,
            output_dir / member.path.with_name(f"{member.path.name}.py"),
            member.data,
            member.stated,
            member.failure,
        )
        for member in container_members(path, version)
    ]
    if not jobs:
        raise FileNotFoundError(f"no .pyc or .pyo files found in {path}")

    return jobs


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    job: BatchJob
    result: DecompileResult | None
    messages: tuple[str, ...]
    details: str = ""
    error: Exception | None = None


def decompile_archive(
    path: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
) -> list[DecompileResult]:
    return finished_results(
        container_jobs(path, output_dir, version), progress, version
    )


def decompile_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    progress: ProgressCallback | None = None,
    version: VersionTuple | None = None,
) -> list[DecompileResult]:
    return finished_results(
        folder_jobs(source_dir, output_dir, recursive), progress, version
    )


def folder_jobs(source_dir: Path, output_dir: Path, recursive: bool) -> list[BatchJob]:
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    pyc_paths = find_pyc_files(source_dir, recursive)
    if not pyc_paths:
        raise FileNotFoundError(f"no .pyc or .pyo files found in {source_dir}")

    return [
        BatchJob(
            path,
            output_dir / path.relative_to(source_dir).with_name(f"{path.name}.py"),
        )
        for path in pyc_paths
    ]


def finished_results(
    jobs: list[BatchJob],
    progress: ProgressCallback | None,
    version: VersionTuple | None,
) -> list[DecompileResult]:
    results: list[DecompileResult] = []

    for outcome in decompile_batch(jobs, version, workers=1):
        if progress is not None:
            progress(f"decompiling {outcome.job.source} -> {outcome.job.target}")

            for message in outcome.messages:
                progress(message)

        if outcome.error is not None:
            raise outcome.error

        assert outcome.result is not None
        results.append(outcome.result)

    return results


def decompile_batch(
    jobs: list[BatchJob],
    version: VersionTuple | None = None,
    workers: int | None = None,
) -> Iterator[BatchOutcome]:
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers <= 1:
        for job in jobs:
            yield run_batch_job(job, version, keep_error=True)

        return

    # imported here, as is traceback below: slow to import, and one file needs neither
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=apply_output_options,
        initargs=(OUTPUT_OPTIONS.get(),),
    ) as pool:
        futures = [pool.submit(run_batch_job, job, version) for job in jobs]
        for job, future in zip(jobs, futures, strict=True):
            try:
                yield future.result()
            except BrokenProcessPool:
                details = "the worker decompiling it stopped (out of memory or a crash)"
                yield BatchOutcome(job, None, (), details)


def apply_output_options(options: OutputOptions) -> None:
    OUTPUT_OPTIONS.set(options)


@contextmanager
def output_options(
    compact: bool = False,
    disassemble: bool = False,
    verify_with: tuple[Path, ...] = (),
) -> Generator[None]:
    token = OUTPUT_OPTIONS.set(OutputOptions(not compact, disassemble, verify_with))

    try:
        yield
    finally:
        OUTPUT_OPTIONS.reset(token)


def run_batch_job(
    job: BatchJob,
    version: VersionTuple | None,
    keep_error: bool = False,
    progress: ProgressCallback | None = None,
) -> BatchOutcome:
    # progress shows messages live; otherwise they are kept for the outcome
    messages: list[str] = []

    if job.failure is not None:
        failure = ContainerError(f"cannot read {job.source}: {job.failure}")
        return BatchOutcome(
            job, None, (), str(failure), failure if keep_error else None
        )

    progress = progress or messages.append

    try:
        if job.data is None:
            result = decompile_file_to_path(job.source, job.target, progress, version)
        else:
            result = decompile_data_to_path(
                job.data, job.source, job.target, progress, version, job.stated
            )
    except Exception as error:
        return BatchOutcome(
            job,
            None,
            tuple(messages),
            failure_details(error),
            error if keep_error else None,
        )

    return BatchOutcome(job, result, tuple(messages))


def failed_result(path: Path, target: Path, details: str) -> DecompileResult:
    return DecompileResult(
        output=DecompiledSource(path=target, source="", strategy="failed", warnings=()),
        report=VerificationReport(
            source_path=target,
            pyc_path=path,
            compiled=False,
            source_valid=False,
            errors=[details],
        ),
    )


def failure_details(error: Exception) -> str:
    if isinstance(error, PycLoadError):
        return str(error)

    import traceback

    return "".join(traceback.format_exception(error)).rstrip()


def describe_module(module: PycModule) -> str:
    header = module.header
    version = (
        "unknown" if header.version is None else ".".join(map(str, header.version))
    )
    magic = "unknown" if header.magic_int is None else str(header.magic_int)
    compiler = (
        f", compiler={pypy_format(header.opcodes).name}" if header.opcodes else ""
    )
    return (
        f"loaded pyc: version={version}, magic={magic}{compiler},"
        f" bytes={header.raw_size}"
    )
