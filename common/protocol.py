from __future__ import annotations

import json
from typing import Any

import numpy as np


class ProtocolError(ValueError):
    pass


def encode_message(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[bytes]:
    if not isinstance(metadata, dict):
        raise ProtocolError("metadata must be a mapping")
    frames = [json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")]
    seen: set[str] = set()
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or name in seen:
            raise ProtocolError(f"invalid or duplicate array name: {name!r}")
        seen.add(name)
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ProtocolError(f"object dtype is not supported for {name}")
        contiguous = np.ascontiguousarray(array)
        descriptor = {
            "name": name,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "nbytes": contiguous.nbytes,
        }
        frames.append(json.dumps(descriptor, separators=(",", ":")).encode("ascii"))
        frames.append(contiguous.tobytes(order="C"))
    return frames


def decode_message(frames: list[bytes]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not frames:
        raise ProtocolError("message has no metadata frame")
    try:
        metadata = json.loads(frames[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ProtocolError("metadata frame must decode to a mapping")
    if (len(frames) - 1) % 2 != 0:
        raise ProtocolError("message has a trailing frame")
    arrays: dict[str, np.ndarray] = {}
    for index in range(1, len(frames), 2):
        try:
            descriptor = json.loads(frames[index].decode("ascii"))
            name = descriptor["name"]
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(item) for item in descriptor["shape"])
            expected_nbytes = int(descriptor["nbytes"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid array descriptor") from exc
        if not isinstance(name, str) or not name or name in arrays:
            raise ProtocolError(f"invalid or duplicate array name: {name!r}")
        if dtype.hasobject:
            raise ProtocolError(f"object dtype is not supported for {name}")
        payload = frames[index + 1]
        if len(payload) != expected_nbytes:
            raise ProtocolError(f"byte-size mismatch for {name}")
        try:
            array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        except ValueError as exc:
            raise ProtocolError(f"invalid shape for {name}") from exc
        arrays[name] = array
    return metadata, arrays
