"""Standard-library persistence shared with the independent solver worker."""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # A neighbouring temporary file keeps replacement on the same filesystem.
    # Readers see either the previous complete file or the new complete file.
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


class FileLock:
    """An OS lock released automatically on process exit, including a crash."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            # The OS releases this lock on process exit. The small lock file can
            # remain in place between successive invocations.
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError(f"Another process is using {self.path}") from error
        self.stream = stream
        return self

    def __exit__(self, *args):
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def lock_is_held(path):
    try:
        with FileLock(path):
            return False
    except RuntimeError:
        return True
