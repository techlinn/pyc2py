import argparse
import os
import traceback
from pathlib import Path

from pyc2py.pipeline import (
    DecompileResult,
    decompile_file_to_directory,
    decompile_file_to_path,
)
from pyc2py.types import DecompiledSource, VerificationReport


class ConsoleProgress:
    def __init__(self) -> None:
        self.current_phase: str | None = None
        self.step_number = 0
        self.clear()

    def __call__(self, message: str) -> None:
        phase = phase_for_message(message)
        if phase != self.current_phase:
            self.start_manual_phase(phase)

        self.step_number += 1
        print(f"[{self.step_number:02}] {message}", flush=True)

    def start_manual_phase(self, phase: str) -> None:
        self.current_phase = phase
        self.step_number = 0
        self.clear()
        print(f"== {phase} ==", flush=True)
        print(flush=True)

    def print_manual(self, message: str) -> None:
        self.step_number += 1
        print(f"[{self.step_number:02}] {message}", flush=True)

    def clear(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")


def phase_for_message(message: str) -> str:
    # keep these phrases synced with pipeline progress text
    lowered = message.lower()
    if lowered.startswith("checking input"):
        return "Phase 1 - Input checks"
    if "loading pyc" in lowered or lowered.startswith("loaded pyc"):
        return "Phase 2 - Loading bytecode"
    if any(
        phrase in lowered
        for phrase in (
            "recovering python source",
            "trying native",
            "decoding bytecode",
            "decoded ",
            "translating bytecode",
            "flushing pending",
            "cleaning recovered ast",
            "recovering module",
            "fixing ast",
            "unparsing ast",
            "checking native",
            "native recovery",
            "building structured fallback",
            "checking fallback",
            "formatting recovered",
        )
    ):
        return "Phase 3 - Recovering source"
    if "creating output" in lowered or "writing source" in lowered:
        return "Phase 4 - Writing output"
    if "validating" in lowered or lowered.startswith("parsing and compiling"):
        return "Phase 5 - Validating result"
    if lowered.startswith(("warning:", "error:")):
        return "Phase 5 - Validating result"
    if "validation passed" in lowered or "validation failed" in lowered:
        return "Phase 5 - Validating result"
    return "Progress"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pyc2py")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=".pyc file or folder to decompile",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=Path,
        help=".pyc file or folder to decompile",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        "--output-dir",
        type=Path,
        help="folder for decompiled output",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="decompile .pyc files in nested folders",
    )
    args = parser.parse_args(argv)
    if args.path is None and args.input_path is None:
        parser.error("provide a .pyc file or folder as PATH or --input")

    if args.path is not None and args.input_path is not None:
        parser.error("use PATH or --input, not both")

    args.path = args.input_path if args.path is None else args.path
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = ConsoleProgress()
    if args.path.is_dir():
        return decompile_folder_command(
            args.path, args.out_dir, args.recursive, progress
        )

    if args.recursive:
        raise SystemExit("--recursive can only be used with a folder input")

    result = decompile_file_command(args.path, args.out_dir, progress)
    print_final_report([result], progress)
    return 0 if result.report.passed else 1


def decompile_file_command(
    path: Path,
    output_dir: Path | None,
    progress: ConsoleProgress,
) -> DecompileResult:
    target = path.with_suffix(path.suffix + ".py")
    if output_dir is not None:
        target = output_dir / f"{path.name}.py"

    progress.start_manual_phase("Phase 0 - Job setup")
    progress.print_manual(f"input file: {path}")
    progress.print_manual(f"output file: {target}")
    progress.print_manual("mode: single file")
    progress.print_manual("starting decompiler pipeline")
    try:
        if output_dir is None:
            return decompile_file_to_path(path, target, progress)

        return decompile_file_to_directory(path, output_dir, progress)
    except Exception as error:
        return make_failed_result(path, target, error)


def decompile_folder_command(
    source_dir: Path,
    output_dir: Path | None,
    recursive: bool,
    progress: ConsoleProgress,
) -> int:
    if output_dir is None:
        raise SystemExit("--out-dir is required when decompiling a folder")

    mode = "recursive" if recursive else "flat"
    progress.start_manual_phase("Phase 0 - Job setup")
    progress.print_manual(f"input folder: {source_dir}")
    progress.print_manual(f"output folder: {output_dir}")
    progress.print_manual(f"folder scan mode: {mode}")
    progress.print_manual("looking for .pyc files")
    pyc_paths = sorted(
        source_dir.rglob("*.pyc") if recursive else source_dir.glob("*.pyc")
    )
    if not pyc_paths:
        raise FileNotFoundError(f"no .pyc files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress.print_manual(f"found {len(pyc_paths)} .pyc file(s)")

    results: list[DecompileResult] = []
    for path in pyc_paths:
        target = output_dir / path.relative_to(source_dir).with_name(f"{path.name}.py")
        progress.start_manual_phase("Phase 0 - Job setup")
        progress.print_manual(f"input file: {path}")
        progress.print_manual(f"output file: {target}")
        progress.print_manual("mode: folder item")
        progress.print_manual("starting decompiler pipeline")
        try:
            results.append(decompile_file_to_path(path, target, progress))
        except Exception as error:
            results.append(make_failed_result(path, target, error))
    print_final_report(results, progress)

    return 0 if all(result.report.passed for result in results) else 1


def make_failed_result(path: Path, target: Path, error: Exception) -> DecompileResult:
    # keep the full traceback, users paste this back when a file breaks
    details = "".join(traceback.format_exception(error)).rstrip()
    return DecompileResult(
        output=DecompiledSource(
            path=target,
            source="",
            strategy="failed",
            warnings=(),
        ),
        report=VerificationReport(
            source_path=target,
            pyc_path=path,
            compiled=False,
            source_valid=False,
            checks=[],
            warnings=[],
            errors=[details],
        ),
    )


def print_final_report(
    results: list[DecompileResult], progress: ConsoleProgress
) -> None:
    progress.start_manual_phase("Phase 6 - Final report")
    passed = sum(1 for result in results if result.report.passed)
    failed = len(results) - passed
    progress.print_manual(f"files processed: {len(results)}")
    progress.print_manual(f"passed: {passed}")
    progress.print_manual(f"failed: {failed}")
    progress.print_manual("generated files:")

    for result in results:
        status = "passed" if result.report.passed else "failed"
        warning_count = len(result.report.warnings)
        error_count = len(result.report.errors)
        print(
            f"    {status}: {result.output.path} "
            f"({result.output.strategy}, {warning_count} warning(s), "
            f"{error_count} error(s))",
            flush=True,
        )

    print_diagnostics(results)


def print_diagnostics(results: list[DecompileResult]) -> None:
    # do not truncate diagnostics, hidden warnings make decompiler bugs harder to fix
    warnings = [
        (result.report.pyc_path, warning)
        for result in results
        for warning in result.report.warnings
    ]
    errors = [
        (result.report.pyc_path, error)
        for result in results
        for error in result.report.errors
    ]
    if not warnings and not errors:
        print(flush=True)
        print("Diagnostics: no warnings or errors.", flush=True)
        return

    print(flush=True)
    print("Diagnostics:", flush=True)
    if warnings:
        print(flush=True)
        print("Warnings:", flush=True)
        for path, warning in warnings:
            print(f"- {path}: {warning}", flush=True)
    if errors:
        print(flush=True)
        print("Errors:", flush=True)
        for path, error in errors:
            print(f"- {path}: {error}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
