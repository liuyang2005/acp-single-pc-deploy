from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return value


def create_run_directory(root: str | Path, prefix: str, timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(root).expanduser().resolve() / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream = self.path.open("x", encoding="utf-8")
        self._lock = Lock()

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(to_jsonable(row), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()
