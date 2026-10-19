import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from pyc2py.bytecode.opcode_table import SUPPORTED_VERSIONS
from pyc2py.compilers import CompilerError, started
from pyc2py.containers import ContainerError, container_kind
from pyc2py.pipeline import (
    BatchJob,
    DecompileResult,
    container_jobs,
    decompile_batch,
    failed_result,
    folder_jobs,
    output_options,
    run_batch_job,
    unit_lines,
)
from pyc2py.types import FILE_STATUSES, VersionTuple

EXIT_CODES = {"passed": 0, "failed": 1, "unverified": 2}
EXIT_USAGE = 3
EPILOG = """\
Each output file ends with a record of where its bytecode came from and, unless
it was verified, why not.

exit status:
  0  every file passed: each function recompiles to the original bytecode
  1  a file failed: part of it could not be recovered or differs
  2  a file is unverified: it could not be compared with the original
  3  nothing was decompiled: the command line or the input folder is wrong
"""


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
        if not sys.stdout.isatty():
            return

        if os.name == "nt":
            os.system("cls")
        else:
            print("\033[H\033[2J\033[3J", end="", flush=True)


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
            "matching functions",
            "checking native",
            "native recovery",
            "building structured fallback",
            "checking fallback",
            "formatting recovered",
        )
    ):
        return "Phase 3 - Recovering source"
    if "validating" in lowered or lowered.startswith("parsing and compiling"):
        return "Phase 4 - Validating result"
    if lowered.startswith(("warning:", "error:", "compiling the recovered")):
        return "Phase 4 - Validating result"
    if "validation passed" in lowered or "validation failed" in lowered:
        return "Phase 4 - Validating result"
    if "creating output" in lowered or "writing source" in lowered:
        return "Phase 5 - Writing output"

    return "Progress"


def parse_python_version(text: str) -> VersionTuple:
    names = ", ".join(".".join(map(str, version)) for version in SUPPORTED_VERSIONS)

    try:
        major, minor = (int(part) for part in text.split("."))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a version like 3.12, one of {names}"
        ) from None

    if (major, minor) not in SUPPORTED_VERSIONS:
        raise argparse.ArgumentTypeError(
            f"Python {text} is not supported; choose one of {names}"
        )

    return major, minor


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


INPUT_HELP = (
    ".pyc/.pyo file, code object, zip archive, PyInstaller bundle,"
    " py2exe executable or folder to decompile"
)


def build_parser() -> Parser:
    parser = Parser(
        prog="pyc2py",
        description="Recover Python source from compiled bytecode and check it.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", type=Path, help=INPUT_HELP)
    parser.add_argument("-i", "--input", dest="input_path", type=Path, help=INPUT_HELP)
    parser.add_argument(
        "-o",
        "--out-dir",
        "--output-dir",
        type=Path,
        help="folder for decompiled output (needed for a folder or archive)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="decompile .pyc and .pyo files in nested folders",
    )
    parser.add_argument(
        "--python-version",
        type=parse_python_version,
        metavar="X.Y",
        help=(
            "read the input as this Python version, whatever its header says;"
            " needed for a code object without a pyc header"
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "do not move statements to their original line numbers (by default"
            " each statement starts on the line it had in the original file)"
        ),
    )
    parser.add_argument(
        "--disassemble",
        action="store_true",
        help=(
            "also write each file's bytecode as a listing (<output>.dis): every"
            " code object with offsets, lines, opcodes, arguments, jump targets"
            " and its exception table"
        ),
    )
    parser.add_argument(
        "--verify-with",
        action="append",
        type=Path,
        default=[],
        metavar="PYTHON",
        help=(
            "an interpreter of the files' own Python version (may be given more"
            " than once): pyc2py compiles each recovered file with it and"
            " compares every function with the original"
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.path is not None and args.input_path is not None:
        parser.error("use PATH or --input, not both")

    args.path = args.input_path if args.path is None else args.path
    if args.path is None:
        parser.error("provide a .pyc file or folder as PATH or --input")
    if args.recursive and not args.path.is_dir():
        parser.error("--recursive can only be used with a folder input")

    for python in args.verify_with:
        compiler = started(python)
        if isinstance(compiler, CompilerError):
            parser.error(f"--verify-with: {compiler}")

    args.jobs = planned_jobs(args, parser)
    return args


def planned_jobs(args: argparse.Namespace, parser: Parser) -> list[BatchJob]:
    path, output_dir = args.path, args.out_dir
    kind = None

    if path.is_dir():
        kind = "folder"
    elif path.is_file():
        kind = container_kind(path)

    if kind is None:
        target = path.with_suffix(path.suffix + ".py")

        if output_dir is not None:
            target = output_dir / f"{path.name}.py"

        return [BatchJob(path, target)]

    if output_dir is None:
        parser.error(f"--out-dir is required when decompiling a {kind}")

    try:
        if kind == "folder":
            return folder_jobs(path, output_dir, args.recursive)

        return container_jobs(path, output_dir, args.python_version)
    except (OSError, ContainerError) as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with output_options(
        compact=args.compact,
        disassemble=args.disassemble,
        verify_with=tuple(args.verify_with),
    ):
        return run_jobs(args.jobs, args.python_version, ConsoleProgress())


def run_jobs(
    jobs: list[BatchJob], version: VersionTuple | None, progress: ConsoleProgress
) -> int:
    if len(jobs) == 1:
        show_job(jobs[0], progress)
        outcomes = iter([run_batch_job(jobs[0], version, progress=progress)])
    else:
        outcomes = decompile_batch(jobs, version)

    results: list[DecompileResult] = []

    for outcome in outcomes:
        job = outcome.job

        if len(jobs) > 1:
            show_job(job, progress)

            for message in outcome.messages:
                progress(message)

        if outcome.result is None:
            results.append(failed_result(job.source, job.target, outcome.details))
        else:
            results.append(outcome.result)

    print_final_report(results, progress)
    return exit_code(results)


def show_job(job: BatchJob, progress: ConsoleProgress) -> None:
    progress.start_manual_phase("Phase 0 - Job setup")
    progress.print_manual(f"input file: {job.source}")
    progress.print_manual(f"output file: {job.target}")


def exit_code(results: list[DecompileResult]) -> int:
    worst = max((result.status for result in results), key=FILE_STATUSES.index)
    return EXIT_CODES[worst]


def print_final_report(
    results: list[DecompileResult], progress: ConsoleProgress
) -> None:
    progress.start_manual_phase("Phase 6 - Final report")
    progress.print_manual(f"files processed: {len(results)}")

    for status in FILE_STATUSES:
        count = sum(1 for result in results if result.status == status)
        progress.print_manual(f"{status}: {count}")

    progress.print_manual("generated files:")

    for result in results:
        print_result(result, "    ")

    print_diagnostics(results)


def print_result(result: DecompileResult, indent: str) -> None:
    report = result.report

    if result.output.strategy == "failed":
        written = f"{result.output.path} not written"
    else:
        written = (
            f"{result.output.path} ({result.output.strategy},"
            f" {len(report.warnings)} warning(s), {len(report.errors)} error(s))"
        )

    print(f"{indent}{report.status}: {written}", flush=True)

    for line in unit_lines(report):
        print(f"{indent}    {line}", flush=True)

    for layer in result.layers:
        if layer.result is None:
            print(f"{indent}    unpacked source: {layer.path}", flush=True)
        else:
            print(f"{indent}    unpacked code:", flush=True)
            print_result(layer.result, indent + "        ")


def print_diagnostics(results: list[DecompileResult]) -> None:
    # do not truncate diagnostics, hidden warnings make decompiler bugs harder to fix
    reports = [result.report for result in results]
    reports += [
        layer.result.report
        for result in results
        for layer in result.layers
        if layer.result
    ]
    warnings = [
        (report.pyc_path, warning) for report in reports for warning in report.warnings
    ]
    errors = [(report.pyc_path, error) for report in reports for error in report.errors]
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
