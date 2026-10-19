import sys

# a friendly message where pip's requires-python did not stop an older Python
if sys.version_info < (3, 14):  # noqa: UP036
    raise SystemExit(
        f"pyc2py needs Python 3.14 or newer to run (this is {sys.version.split()[0]})"
    )

from pyc2py.api import decompile, decompile_folder, decompile_to_folder

__all__ = ["decompile", "decompile_folder", "decompile_to_folder"]
