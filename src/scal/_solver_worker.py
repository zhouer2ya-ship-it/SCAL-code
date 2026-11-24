"""Independent command runner; invoked as a script without importing NumPy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _persistence import FileLock, atomic_json


def main(directory):
    directory = Path(directory).resolve()
    with FileLock(directory / "execution.lock"):
        # This lock lets a recovering caller recognize active supervision even
        # after the original parent process has disappeared.
        record = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        atomic_json(directory / "worker_started.json", {"pid": os.getpid()})
        started = time.perf_counter()
        result = {"returncode": None, "error": None, "error_type": None}
        try:
            with (
                (directory / "stdout.log").open("wb") as stdout,
                (directory / "stderr.log").open("wb") as stderr,
            ):
                process = subprocess.run(
                    record["command"],
                    cwd=directory,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=record["timeout_seconds"],
                    check=False,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            result["returncode"] = process.returncode
        except Exception as error:
            result.update(error=str(error), error_type=type(error).__name__)
        result["elapsed_seconds"] = time.perf_counter() - started
        # The parent reads this commit before parsing the response file.
        atomic_json(directory / "exit_status.json", result)


if __name__ == "__main__":
    main(sys.argv[1])
