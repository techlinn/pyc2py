import importlib.util
import struct
import sys
from pyc2py.types import VersionTuple

MAGIC_VERSION_MAP: dict[int, VersionTuple] = {
    39170: (1, 0),
    39171: (1, 1),
    11913: (1, 3),
    5892: (1, 4),
    20121: (1, 5),
    50428: (1, 6),
    50823: (2, 0),
    60202: (2, 1),
    60717: (2, 2),
    62011: (2, 3),
    62061: (2, 4),
    62131: (2, 5),
    62161: (2, 6),
    62211: (2, 7),
    3131: (3, 0),
    3151: (3, 1),
    3180: (3, 2),
    3230: (3, 3),
    3310: (3, 4),
    3350: (3, 5),
    3351: (3, 5),
    3379: (3, 6),
    3394: (3, 7),
    3413: (3, 8),
    3425: (3, 9),
    3439: (3, 10),
    3495: (3, 11),
    3531: (3, 12),
    3571: (3, 13),
    3627: (3, 14),
    3666: (3, 15),
    3701: (3, 16),
}

def find_known_version(magic_int: int) -> VersionTuple | None:
    return MAGIC_VERSION_MAP.get(magic_int)

CURRENT_MAGIC_INT = struct.unpack("<H", importlib.util.MAGIC_NUMBER[:2])[0]

def read_magic_int(data: bytes) -> int:
    if len(data) < 2:
        raise ValueError("pyc data is shorter than the magic word")
    return struct.unpack("<H", data[:2])[0]

def find_version(magic_int: int) -> tuple[int, ...] | None:
    if magic_int == CURRENT_MAGIC_INT:
        return sys.version_info[:2]
    return find_known_version(magic_int)
