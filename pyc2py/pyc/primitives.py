import struct

PY_LONG_MARSHAL_SHIFT = 15
PY_LONG_MARSHAL_BASE = 1 << PY_LONG_MARSHAL_SHIFT
MAX_MARSHAL_SIZE = 20_000_000


def read_int32(data: bytes, offset: int) -> tuple[int, int]:
    chunk = read_exact(data, offset, 4)
    return struct.unpack("<i", chunk)[0], offset + 4


def read_int16(data: bytes, offset: int) -> tuple[int, int]:
    chunk = read_exact(data, offset, 2)
    return struct.unpack("<h", chunk)[0], offset + 2


def read_int64(data: bytes, offset: int) -> tuple[int, int]:
    chunk = read_exact(data, offset, 8)
    return struct.unpack("<q", chunk)[0], offset + 8


def read_uint8(data: bytes, offset: int) -> tuple[int, int]:
    chunk = read_exact(data, offset, 1)
    return chunk[0], offset + 1


def read_float_text(data: bytes, offset: int) -> tuple[float, int]:
    size, offset = read_uint8(data, offset)
    raw = read_exact(data, offset, size)

    return float(raw.decode("ascii")), offset + size


def read_binary_float(data: bytes, offset: int) -> tuple[float, int]:
    chunk = read_exact(data, offset, 8)
    return struct.unpack("<d", chunk)[0], offset + 8


def read_marshal_number(
    data: bytes, type_code: int, offset: int
) -> tuple[float | complex, int]:
    if type_code == ord("f"):
        return read_float_text(data, offset)
    if type_code == ord("g"):
        return read_binary_float(data, offset)
    if type_code == ord("x"):
        real, offset = read_float_text(data, offset)
        imag, offset = read_float_text(data, offset)
        return complex(real, imag), offset
    if type_code == ord("y"):
        real, offset = read_binary_float(data, offset)
        imag, offset = read_binary_float(data, offset)
        return complex(real, imag), offset
    raise ValueError(f"unsupported marshal numeric type: {type_code!r}")


def read_long(data: bytes, offset: int) -> tuple[int, int]:
    digit_count, offset = read_int32(data, offset)
    sign = -1 if digit_count < 0 else 1
    digit_count = abs(digit_count)

    if digit_count > MAX_MARSHAL_SIZE:
        raise ValueError("marshal long has too many digits")

    value = 0
    for digit_index in range(digit_count):
        digit = read_exact(data, offset, 2)
        offset += 2
        part = digit[0] | (digit[1] << 8)
        if part >= PY_LONG_MARSHAL_BASE:
            raise ValueError("marshal long digit is out of range")
        value += part << (digit_index * PY_LONG_MARSHAL_SHIFT)

    return sign * value, offset


def read_size(data: bytes, offset: int) -> tuple[int, int]:
    size, offset = read_int32(data, offset)
    if size < 0:
        raise ValueError("marshal size is negative")
    if size > MAX_MARSHAL_SIZE:
        raise ValueError("marshal size exceeds local limit")
    return size, offset


def read_exact(data: bytes, offset: int, size: int) -> bytes:
    if size < 0:
        raise ValueError("read size must not be negative")

    end = offset + size
    if offset < 0 or end > len(data):
        raise EOFError("marshal data ended unexpectedly")

    return data[offset:end]
