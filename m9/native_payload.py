"""Bounded JSON control and uint16 shared-slot metadata; no model imports."""
from __future__ import annotations

import base64
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import socket
import struct

import numpy as np

MAX_FRAME_BYTES = 65536
MAX_SLOT_BYTES = 512 * 1024 * 1024


class LayoutMiss(ValueError):
    """The native image cannot safely use the optional shared slot."""


def send_message(sock, message):
    if not isinstance(message, dict):
        raise ValueError("Control message must be an object")
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("Control frame exceeds the small JSON limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_message(sock, stop_event=None):
    """Retain partial frames across timeouts; EOF/stop never returns partial JSON."""
    def receive(size):
        result = bytearray()
        while len(result) < size:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Control reader stopped")
            try:
                part = sock.recv(size - len(result))
            except socket.timeout:
                continue
            if not part:
                raise EOFError("Control channel closed during a frame" if result else "Control channel closed")
            result.extend(part)
        return result
    length = struct.unpack("!I", receive(4))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ValueError("Invalid control frame length")
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    message = json.loads(receive(length).decode("utf-8"), parse_constant=invalid_constant)
    if not isinstance(message, dict):
        raise ValueError("Control message must be an object")
    return message


def path_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError("Native input is not a nonempty file")
    return dict(path=str(path), device=int(stat.st_dev), inode=int(stat.st_ino), size=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns), ctime_ns=int(stat.st_ctime_ns))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dense_strides(shape, axes):
    strides, step = [0] * len(shape), np.dtype(np.uint16).itemsize
    for axis in axes:
        strides[axis], step = step, step * shape[axis]
    return strides


def array_layout(rgb):
    """Discover dense K axis order with eight elements, including negative/gapped views."""
    if (type(rgb) is not np.ndarray or rgb.dtype != np.dtype(np.uint16) or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape) <= 1):
        raise LayoutMiss("Shared native slot requires non-singleton native uint16 HxWx3")
    span = rgb.itemsize
    for axis in sorted(range(3), key=lambda i: abs(rgb.strides[i])):
        stride = abs(rgb.strides[axis])
        if stride < span:
            raise LayoutMiss("Overlapping, zero or ambiguous native strides")
        span += (rgb.shape[axis] - 1) * stride
    # Same ndim preserves the original K axis order without allocating a full-image prototype.
    probe = np.empty_like(rgb, dtype=np.uint16, order="K", shape=(2, 2, 2))
    axes = sorted(range(3), key=lambda i: probe.strides[i])
    if sorted(probe.strides) != [2, 4, 8]:
        raise LayoutMiss("NumPy did not produce an unambiguous dense K probe")
    shape = [int(n) for n in rgb.shape]
    nbytes = math.prod(shape) * rgb.itemsize
    if nbytes > MAX_SLOT_BYTES:
        raise LayoutMiss("Decoded native image exceeds the shared-slot limit")
    return dict(shape=shape, dtype=rgb.dtype.str, strides=_dense_strides(shape, axes), nbytes=nbytes,
                source_strides=[int(n) for n in rgb.strides], probe_strides=[int(n) for n in probe.strides])


def slot_array(buffer, layout):
    """Validate an exact dense positive layout before creating an external-buffer view."""
    shape, strides = layout.get("shape"), layout.get("strides")
    if (not isinstance(shape, list) or not isinstance(strides, list) or len(shape) != 3 or len(strides) != 3
            or any(type(n) is not int or n <= 1 for n in shape) or shape[2] != 3
            or any(type(n) is not int or n <= 0 for n in strides) or layout.get("dtype") != np.dtype(np.uint16).str):
        raise ValueError("Invalid shared native shape/dtype/strides")
    axes = sorted(range(3), key=lambda i: strides[i])
    nbytes = math.prod(shape) * 2
    view = memoryview(buffer)
    try:
        available = view.nbytes
    finally:
        view.release()
    if (strides != _dense_strides(shape, axes) or type(layout.get("nbytes")) is not int
            or layout["nbytes"] != nbytes or not 0 < nbytes <= min(available, MAX_SLOT_BYTES)):
        raise ValueError("Shared native layout is not dense or exceeds its buffer")
    return np.ndarray(tuple(shape), dtype=np.uint16, buffer=buffer, strides=tuple(strides))


def _encode(value):
    # Typed small metadata preserves Python tuple/list and NumPy scalar/array identity.
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biuf" or value.nbytes > MAX_FRAME_BYTES:
            raise ValueError("Unsupported metadata array")
        return dict(type="ndarray", dtype=value.dtype.str, shape=list(value.shape), strides=list(value.strides),
                    data=base64.b64encode(value.tobytes(order="C")).decode("ascii"))
    if isinstance(value, np.generic):
        if value.dtype.hasobject or value.dtype.kind not in "biuf":
            raise ValueError("Unsupported metadata scalar")
        return dict(type="numpy_scalar", dtype=value.dtype.str, data=base64.b64encode(value.tobytes()).decode("ascii"))
    if value is None or type(value) in (bool, str, int, float):
        return dict(type=type(value).__name__, value=value)
    if type(value) in (list, tuple):
        return dict(type=type(value).__name__, items=[_encode(item) for item in value])
    if type(value) is dict:
        return dict(type="dict", items=[[_encode(key), _encode(item)] for key, item in value.items()])
    raise ValueError("Unsupported native metadata type: " + type(value).__name__)


def _decode(value):
    kind = value["type"]
    if kind in ("NoneType", "bool", "str", "int", "float"):
        result = value["value"]
        expected = {"NoneType": type(None), "bool": bool, "str": str, "int": int, "float": float}[kind]
        if type(result) is not expected or (kind == "float" and not math.isfinite(result)):
            raise ValueError("Invalid typed metadata scalar")
        return result
    if kind in ("tuple", "list"):
        result = [_decode(item) for item in value["items"]]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict":
        pairs = [(_decode(key), _decode(item)) for key, item in value["items"]]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate typed metadata keys")
        return result
    if kind not in ("ndarray", "numpy_scalar"):
        raise ValueError("Unknown typed native metadata")
    dtype = np.dtype(value["dtype"])
    data = base64.b64decode(value["data"], validate=True)
    if dtype.hasobject or dtype.kind not in "biuf" or len(data) > MAX_FRAME_BYTES:
        raise ValueError("Invalid metadata numeric dtype/size")
    if kind == "numpy_scalar":
        if len(data) != dtype.itemsize:
            raise ValueError("Invalid NumPy metadata scalar size")
        return np.frombuffer(data, dtype=dtype)[0]
    shape, strides = value["shape"], value["strides"]
    if (not isinstance(shape, list) or not isinstance(strides, list) or len(shape) != len(strides) or len(shape) > 3
            or any(type(n) is not int or n <= 0 for n in shape) or any(type(n) is not int for n in strides)
            or math.prod(shape) * dtype.itemsize != len(data)):
        raise ValueError("Invalid metadata array geometry")
    for order in ("C", "F"):
        result = np.empty(shape, dtype=dtype, order=order)
        if list(result.strides) == strides:
            break
    else:
        low = sum(min(0, (n - 1) * stride) for n, stride in zip(shape, strides))
        high = sum(max(0, (n - 1) * stride) for n, stride in zip(shape, strides)) + dtype.itemsize
        if not 0 < high - low <= MAX_FRAME_BYTES:
            raise ValueError("Metadata array stride span exceeds its limit")
        result = np.ndarray(shape, dtype=dtype, buffer=bytearray(high - low), offset=-low, strides=strides)
    np.copyto(result, np.frombuffer(data, dtype=dtype).reshape(shape), casting="no")
    return result


def encode_matrix(matrix):
    if type(matrix) is not np.ndarray or matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or matrix.shape != (3, 3):
        raise ValueError("Native camera matrix must be native float32 or float64 3x3")
    return _encode(matrix)


def decode_matrix(value):
    result = _decode(value)
    if type(result) is not np.ndarray or result.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("Invalid native camera matrix")
    return result


def encode_raw_info(info):
    return _encode({field.name: getattr(info, field.name) for field in fields(info)})


def decode_raw_info(value):
    from .frontend import RawInfo
    result = _decode(value)
    if type(result) is not dict or set(result) != {field.name for field in fields(RawInfo)}:
        raise ValueError("Invalid native RawInfo fields")
    return RawInfo(**result)
