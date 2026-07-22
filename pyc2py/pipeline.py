from dataclasses import dataclass
from pathlib import Path

from pyc2py.pyc.loader import load_pyc
from pyc2py.reporting import validate_generated_source
from pyc2py.source import emit_source
from pyc2py.types import (
    DecompiledSource,
    LiveWarningList,
    ProgressCallback,
    PycModule,
    VerificationReport,
    emit_progress,
)


@dataclass(frozen=True, slots=True)
class DecompileResult:
    output: DecompiledSource
    report: VerificationReport


def decompile_file(path: Path) -> DecompileResult:
    return decompile_file_to_path(path, path.with_suffix(path.suffix + ".py"))


def decompile_file_to_directory(
    path: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> DecompileResult:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    return decompile_file_to_path(path, output_dir / f"{path.name}.py", progress)


def decompile_file_to_path(
    path: Path,
    target: Path,
    progress: ProgressCallback | None = None,
) -> DecompileResult:
    emit_progress(progress, f"checking input: {path}")
    if path.suffix.lower() != ".pyc":
        raise ValueError(f"expected a .pyc file: {path}")
    if target.exists() and target.is_dir():
        raise IsADirectoryError(target)

    emit_progress(progress, "loading pyc header and code object")
    module = load_pyc(path)
    emit_progress(progress, describe_module(module))

    # this list prints warnings live and keeps the same text for the final report
    warnings = LiveWarningList(progress)
    emit_progress(progress, "recovering Python source")
    source, strategy, _source_warnings = emit_source(module, progress, warnings)

    emit_progress(progress, f"creating output folder: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    emit_progress(progress, f"writing source: {target}")
    target.write_text(source, encoding="utf-8")

    emit_progress(progress, "validating generated source and bytecode recovery")
    report = validate_generated_source(
        path,
        target,
        source,
        warnings,
        module.header.version,
        module.code,
        progress,
    )
    emit_progress(
        progress, "validation passed" if report.passed else "validation failed"
    )

    return DecompileResult(
        output=DecompiledSource(
            path=target,
            source=source,
            strategy=strategy,
            warnings=tuple(warnings),
        ),
        report=report,
    )


def decompile_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    progress: ProgressCallback | None = None,
) -> list[DecompileResult]:
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    pyc_paths = sorted(
        source_dir.rglob("*.pyc") if recursive else source_dir.glob("*.pyc")
    )
    if not pyc_paths:
        raise FileNotFoundError(f"no .pyc files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DecompileResult] = []
    for path in pyc_paths:
        target = output_dir / path.relative_to(source_dir).with_name(f"{path.name}.py")
        if progress is not None:
            progress(f"decompiling {path} -> {target}")

        results.append(decompile_file_to_path(path, target, progress))

    return results


def describe_module(module: PycModule) -> str:
    header = module.header
    version = (
        "unknown" if header.version is None else ".".join(map(str, header.version))
    )
    magic = "unknown" if header.magic_int is None else str(header.magic_int)
    return f"loaded pyc: version={version}, magic={magic}, bytes={header.raw_size}"
