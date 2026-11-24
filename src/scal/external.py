"""Adapt a blocking command-line simulator to FunctionExperiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Sequence

import numpy as np

from ._persistence import FileLock, lock_is_held
from .checkpoint import callable_identity, signature
from .io import digest, read_json, write_json


class SimulationError(RuntimeError):
    """A solver job failed; its directory contains the inputs and logs."""

    def __init__(self, message: str, directory: Path):
        self.directory = directory
        super().__init__(f"{message}. Job files: {directory}")


@dataclass(frozen=True)
class SimulationResult:
    directory: Path
    response: float
    elapsed_seconds: float


def write_parameter_json(path: Path, parameters: dict[str, float]) -> None:
    """Default input: a JSON object with a named parameters mapping."""
    write_json(path, {"parameters": parameters})


def read_response_json(path: Path) -> float:
    """Default output: a JSON object with one scalar response field."""
    return read_json(path)["response"]


class ExternalSimulator:
    """Evaluate each input row in a separate directory and solver process.

    command is a sequence of arguments, with {input}, {output} and {workdir}
    placeholders. The process must wait for the solver to finish before exiting.
    Input and output callbacks allow native text, CSV or other file formats.
    Successful responses are indexed by exact parameters and model identity.
    An independent worker records completion while the caller manages training
    and validates the returned physical response.
    """

    def __init__(
        self,
        command: Sequence[str],
        parameter_names: Sequence[str],
        work_root: str | Path,
        *,
        timeout_seconds: float = 300,
        input_name: str = "input.json",
        output_name: str = "output.json",
        input_writer: Callable[[Path, dict[str, float]], None] = write_parameter_json,
        output_reader: Callable[[Path], float] = read_response_json,
        model_id: str | None = None,
        model_files: Sequence[str | Path] = (),
        retry_unconfirmed: bool = False,
    ):
        if (
            isinstance(command, str)
            or not command
            or any(not isinstance(arg, str) for arg in command)
        ):
            raise ValueError("command must be a nonempty list of argument strings")
        if isinstance(parameter_names, str):
            raise ValueError("parameter_names must be a sequence of names")
        names = tuple(parameter_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("parameter_names must contain distinct nonempty names")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        for filename in (input_name, output_name):
            if not filename or Path(filename).name != filename or filename in (".", ".."):
                raise ValueError("Input and output names must be filenames without directories")
            if filename.casefold() in {
                "job.json",
                "stdout.log",
                "stderr.log",
                "exit_status.json",
                "worker_started.json",
                "execution.lock",
                "worker.log",
            }:
                raise ValueError(f"Reserved job filename: {filename}")
        if input_name.casefold() == output_name.casefold():
            raise ValueError("Input and output filenames must differ")
        self.command = tuple(command)
        self.parameter_names = names
        self.work_root = Path(work_root).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.input_name, self.output_name = input_name, output_name
        self.input_writer, self.output_reader = input_writer, output_reader
        self.checkpoint_options = {"timeout_seconds": self.timeout_seconds}
        self.retry_unconfirmed = retry_unconfirmed
        dependencies = {Path(path).expanduser().resolve() for path in model_files}
        for index, argument in enumerate(command):
            try:
                path = Path(argument)
                if (
                    path.is_absolute()
                    and path.is_file()
                    and (
                        index == 0
                        or path.suffix.lower()
                        in {".py", ".pyw", ".exe", ".sh", ".bat", ".cmd", ".ps1"}
                    )
                ):
                    dependencies.add(path.resolve())
            except (OSError, ValueError):
                pass
        self.checkpoint_identity = {
            "schema": 1,
            "command": list(command),
            "parameters": list(names),
            "input": input_name,
            "output": output_name,
            "input_writer": callable_identity(input_writer),
            "output_reader": callable_identity(output_reader),
            "model_id": model_id,
            "model_files": {str(path): digest(path) for path in sorted(dependencies)},
        }
        self._identity_hash = signature(self.checkpoint_identity)
        self.completed: list[SimulationResult] = []
        self.new_jobs = 0
        self.cache_hits = 0
        self._next_case = 0

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != len(self.parameter_names):
            raise ValueError(f"Expected inputs with shape (n, {len(self.parameter_names)})")
        if not np.isfinite(inputs).all():
            raise ValueError("Simulator inputs must be finite")
        values = []
        with FileLock(self.work_root / ".cache.lock"):
            cache = self._load_cache()
            for row in inputs:
                parameters = dict(zip(self.parameter_names, map(float, row)))
                # float.hex preserves every binary digit, keeping nearby design
                # points distinct in the cache.
                key = signature([(name, value.hex()) for name, value in parameters.items()])
                result = None
                if key in cache:
                    result = self._recover(cache[key])
                    if result is not None:
                        self.cache_hits += 1
                if result is None:
                    result = self._run_point(parameters, key)
                    cache[key] = result.directory
                self.completed.append(result)
                values.append(result.response)
        return np.asarray(values)

    def _load_cache(self):
        # Rebuilding from individual jobs recovers results written immediately
        # before an interruption of the adaptive training process.
        manifest = self.work_root / "simulator.json"
        if manifest.exists():
            if read_json(manifest) != self.checkpoint_identity:
                raise ValueError(
                    "Simulator configuration or model files changed; cannot reuse cached responses"
                )
        else:
            write_json(manifest, self.checkpoint_identity)
        cache = {}
        for path in sorted(self.work_root.glob("case_*/job.json")):
            record = read_json(path)
            if record.get("cache_schema") != 1:
                continue
            if record.get("simulator_sha256") != self._identity_hash:
                raise SimulationError("Simulator identity mismatch", path.parent)
            cache[record["point_sha256"]] = path.parent
        return cache

    def _read_result(self, directory, record):
        output = directory / self.output_name
        if not output.is_file():
            raise FileNotFoundError(f"Simulator did not produce {self.output_name}")
        value = self.output_reader(output)
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError("Simulator output must be one scalar number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Simulator output is not finite")
        if digest(directory / self.input_name) != record["input_sha256"]:
            raise ValueError("Solver input checksum mismatch")
        return value

    def _recover(self, directory):
        record = read_json(directory / "job.json")
        if record["status"] == "completed":
            try:
                if digest(directory / self.output_name) != record["output_sha256"]:
                    raise ValueError("Cached output checksum mismatch")
                value = self._read_result(directory, record)
                if value != record["response"]:
                    raise ValueError("Cached response mismatch")
            except Exception as error:
                raise SimulationError(str(error), directory) from error
            return SimulationResult(directory, value, record["elapsed_seconds"])
        if record["status"] in {"failed", "prepared", "abandoned"}:
            return None
        # An independent worker can finish even if its SCAL parent has exited.
        # The OS lock distinguishes a live worker from an interrupted machine.
        deadline = time.monotonic() + 3
        while not (directory / "exit_status.json").exists():
            if lock_is_held(directory / "execution.lock"):
                time.sleep(0.05)
                continue
            if not (directory / "worker_started.json").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
                continue
            if self.retry_unconfirmed:
                record.update(status="abandoned", error="User allowed retry of an unconfirmed job")
                write_json(directory / "job.json", record)
                return None
            raise SimulationError(
                "Previous solver completion is unconfirmed. Check that no solver is still running; "
                "only then use retry_unconfirmed=True to retry this unfinished point",
                directory,
            )
        outcome = read_json(directory / "exit_status.json")
        if outcome["returncode"] != 0 or outcome["error"]:
            self._mark_failed(directory, record, outcome)
            return None
        return self._finish_result(directory, record, outcome)

    def _mark_failed(self, directory, record, outcome):
        message = (
            outcome["error"]
            or f"Simulator exited with code {outcome['returncode']}; see stderr.log"
        )
        if outcome["error_type"] == "TimeoutExpired":
            message = f"Simulator exceeded {record['timeout_seconds']:g} seconds"
        record.update(outcome, status="failed", error=message)
        write_json(directory / "job.json", record)
        return message

    def _finish_result(self, directory, record, outcome):
        try:
            value = self._read_result(directory, record)
        except Exception as error:
            record.update(
                outcome, status="failed", error=str(error), error_type=type(error).__name__
            )
            write_json(directory / "job.json", record)
            raise SimulationError(str(error), directory) from error
        record.update(
            outcome,
            status="completed",
            response=value,
            output_sha256=digest(directory / self.output_name),
        )
        write_json(directory / "job.json", record)
        return SimulationResult(directory, value, outcome["elapsed_seconds"])

    def _new_directory(self) -> Path:
        # Creation is deferred until the first response request. FunctionExperiment
        # can therefore check and create its output directory first.
        self.work_root.mkdir(parents=True, exist_ok=True)
        while True:
            self._next_case += 1
            directory = self.work_root / f"case_{self._next_case:06d}"
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                continue

    def _run_point(self, parameters: dict[str, float], key: str) -> SimulationResult:
        directory = self._new_directory()
        input_path, output_path = directory / self.input_name, directory / self.output_name
        replacements = {
            "{input}": str(input_path),
            "{output}": str(output_path),
            "{workdir}": str(directory),
        }
        command = []
        for argument in self.command:
            for token, value in replacements.items():
                argument = argument.replace(token, value)
            command.append(argument)
        record = {
            "status": "prepared",
            "cache_schema": 1,
            "simulator_sha256": self._identity_hash,
            "point_sha256": key,
            "parameters": parameters,
            "command": command,
            "input": self.input_name,
            "output": self.output_name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "returncode": None,
        }
        write_json(directory / "job.json", record)
        started = time.perf_counter()
        try:
            self.input_writer(input_path, parameters)
            # Commit submission before launch so a crash in that interval leaves
            # an explicit unconfirmed state for recovery.
            record.update(status="submitted", input_sha256=digest(input_path))
            write_json(directory / "job.json", record)
            with (directory / "worker.log").open("wb") as log:
                # The detached worker supervises the solver after a parent exit;
                # resume reads its completion record or waits for the live job.
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("_solver_worker.py")),
                        str(directory),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                    if os.name == "nt"
                    else 0,
                    start_new_session=os.name != "nt",
                )
            self.new_jobs += 1
        except Exception as error:
            record.update(
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                error=str(error),
                error_type=type(error).__name__,
            )
            write_json(directory / "job.json", record)
            raise SimulationError(str(error), directory) from error
        # KeyboardInterrupt leaves the worker and its logs recoverable.
        process.wait()
        if not (directory / "exit_status.json").exists():
            raise SimulationError(
                "Solver worker stopped without a completion record; see worker.log", directory
            )
        outcome = read_json(directory / "exit_status.json")
        if outcome["returncode"] != 0 or outcome["error"]:
            message = self._mark_failed(directory, record, outcome)
            raise SimulationError(message, directory)
        return self._finish_result(directory, record, outcome)
