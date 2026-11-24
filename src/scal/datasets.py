"""Find the research directory and load checked benchmark inputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig, PaperConfig
from .io import data_path, read_json


def find_root(root: str | Path | None = None) -> Path:
    """Use an explicit directory, SCAL_HOME, or the nearest research directory."""
    explicit = root if root is not None else os.environ.get("SCAL_HOME")
    candidates = (
        [Path(explicit).expanduser().resolve()] if explicit else [Path.cwd(), *Path.cwd().parents]
    )
    for candidate in candidates:
        if (candidate / "configs/paper_benchmarks.json").is_file() and (
            candidate / "data/benchmark_inputs.json"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Benchmark workspace not found. Set --root PATH to a directory with "
        "configs/paper_benchmarks.json and data/benchmark_inputs.json. "
        "For a standalone demonstration, run examples/simple_function.py."
    )


@dataclass
class TrainingData:
    """Initial observations and a candidate pool with fixed, unique integer IDs."""

    inputs: np.ndarray
    responses: np.ndarray
    candidates: np.ndarray
    candidate_ids: np.ndarray
    source_path: str

    def validate(self, problem: BenchmarkConfig) -> None:
        if self.inputs.shape != (problem.initial, problem.dimension):
            raise ValueError("Initial input shape differs from the benchmark configuration")
        if self.responses.shape != (problem.initial,):
            raise ValueError("Initial responses must have one value per input row")
        if self.candidates.ndim != 2 or self.candidates.shape[1] != problem.dimension:
            raise ValueError("Candidate dimensions differ from the benchmark")
        if (
            self.candidate_ids.shape != (len(self.candidates),)
            or self.candidate_ids.dtype.kind not in "iu"
        ):
            raise ValueError("Candidate IDs must be an integer vector aligned with the pool")
        if len(np.unique(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Candidate IDs must be unique")
        if any(
            not np.isfinite(array).all() for array in (self.inputs, self.responses, self.candidates)
        ):
            raise ValueError("Training inputs, responses and candidates must be finite")


class DatasetRepository:
    """Resolve files from the bundled data and an optional larger data directory.

    An external directory can contain just the additional training fixtures.
    Files absent there are read from the bundled data directory.
    """

    def __init__(self, root: str | Path | None = None, data_root: str | Path | None = None):
        self.root = find_root(root)
        self.data_root = Path(data_root).expanduser().resolve() if data_root else self.root / "data"
        self.config = PaperConfig(self.root / "configs/paper_benchmarks.json")
        self.entries = {
            entry["path"]: entry
            for entry in read_json(self.root / "data/benchmark_inputs.json")["files"]
        }

    def input_path(self, relative: str, *, verify: bool = True) -> Path:
        if relative not in self.entries:
            raise ValueError(f"Input is not listed in benchmark_inputs.json: {relative}")
        choices = [self.data_root / relative, self.root / "data" / relative]
        path = next((item for item in choices if item.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Missing input: {relative}. Supply the listed files with --data-root PATH; "
                "see docs/DATA.md."
            )
        if verify and path.stat().st_size != self.entries[relative]["bytes"]:
            raise ValueError(f"Input size differs: {relative}")
        return path

    def training(self, problem: BenchmarkConfig, seed: int) -> TrainingData:
        relative = f"benchmarks/{problem.name}/seed_{seed:03d}/training.npz"
        with np.load(self.input_path(relative), allow_pickle=False) as arrays:
            data = TrainingData(
                inputs=arrays["initial_X"].copy(),
                responses=arrays["initial_y"].ravel().copy(),
                candidates=arrays["candidate_pool"].copy(),
                candidate_ids=arrays["candidate_ids"].copy(),
                source_path=relative,
            )
        data.validate(problem)
        return data

    def paper_test(self, problem: str) -> dict:
        manifest = read_json(self.data_root / "evaluation/mc100k/test_sets.json")
        if manifest["schema"] != "scal-mc100k-v1" or problem not in manifest["tests"]:
            raise ValueError(f"No SCAL MC100k test set for {problem}")
        return manifest["tests"][problem]

    def test(self, problem: str, *, test_set: str = "legacy") -> tuple[np.ndarray, np.ndarray]:
        """Load independent evaluation arrays; the training runner never calls this."""
        if test_set == "legacy":
            files = [
                self.input_path(f"benchmarks/{problem}/test_{label}.npy") for label in ("X", "y")
            ]
            inputs, responses = [np.load(path, mmap_mode="r", allow_pickle=False) for path in files]
            return inputs, responses.ravel()
        if test_set != "mc100k":
            raise ValueError(f"Unknown test set: {test_set}")
        test = self.paper_test(problem)
        arrays = []
        for label in ("X", "y"):
            record = test[label]
            path = data_path(self.data_root, record["path"])
            if "bytes" in record and path.stat().st_size != record["bytes"]:
                raise ValueError(f"Test set size differs: {path}")
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        inputs, responses = arrays
        truth = responses.ravel()
        if (
            test["n"] != 100_000
            or inputs.shape != (100_000, len(test["input_law"]))
            or truth.shape != (100_000,)
        ):
            raise ValueError(f"Incorrect MC100k shape for {problem}")
        return inputs, truth
