"""Small file operations shared by experiments and analysis commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ._persistence import atomic_json


def digest(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def data_path(root: Path, relative: str) -> Path:
    """Resolve a portable file reference within an already resolved data root."""
    name = Path(relative)
    path = (root / name).resolve()
    if name.anchor or not path.is_relative_to(root):
        raise ValueError(f"Path is outside the data directory: {relative}")
    return path


def json_value(value: Any) -> Any:
    """Convert NumPy values and unavailable floating-point results to JSON."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write a complete JSON file, replacing the previous file atomically."""
    atomic_json(path, json_value(value))


def write_npz(path: str | Path, **values) -> None:
    """Commit a complete compressed NumPy file after flushing its contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
