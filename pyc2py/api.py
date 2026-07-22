from pathlib import Path

from pyc2py.pipeline import (
    DecompileResult,
    decompile_directory,
    decompile_file,
    decompile_file_to_directory,
)


def decompile(path: str | Path) -> DecompileResult:
    return decompile_file(Path(path))


def decompile_to_folder(path: str | Path, output_dir: str | Path) -> DecompileResult:
    return decompile_file_to_directory(Path(path), Path(output_dir))


def decompile_folder(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = False,
) -> list[DecompileResult]:
    return decompile_directory(Path(source_dir), Path(output_dir), recursive=recursive)
