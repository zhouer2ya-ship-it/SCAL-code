"""Run HCBP and one-point sampling with a user-supplied scalar response."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from .acquisition import SamplingPolicy
from ._persistence import FileLock, atomic_bytes
from .checkpoint import FunctionCheckpoint, callable_identity, runtime_identity
from .config import SAMPLERS
from .experiment import ExperimentResult, Progress
from .io import digest, read_json, write_json, write_npz
from .solving.HCBP_solver import HCBPSolver
from .storage import RunStore


class InputSpace:
    """Independent uniform, normal or lognormal physical inputs.

    Each row is (0, lower, upper), (1, mean, std), or (2, mean_log, std_log).
    The same matrix is passed to the polynomial basis and the sampling rule.
    """

    def __init__(self, poly_matrix):
        matrix = np.asarray(poly_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 3 or len(matrix) == 0:
            raise ValueError("poly_matrix must have shape (dimension, 3)")
        if not np.isfinite(matrix).all() or not np.isin(matrix[:, 0], [0, 1, 2]).all():
            raise ValueError("Input distributions must be finite and use kind 0, 1 or 2")
        for kind, first, second in matrix:
            if (kind == 0 and first >= second) or (kind != 0 and second <= 0):
                raise ValueError(
                    "Uniform bounds must increase; normal/lognormal std must be positive"
                )
        self.poly_matrix = matrix.copy()
        self.poly_matrix.setflags(write=False)

    @property
    def dimension(self) -> int:
        return len(self.poly_matrix)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Draw physical inputs using the supplied random-number generator."""
        if type(count) is not int or count < 1:
            raise ValueError("Sample count must be a positive integer")
        inputs = np.empty((count, self.dimension))
        for column, (kind, first, second) in enumerate(self.poly_matrix):
            if kind == 0:
                values = rng.uniform(first, second, count)
            elif kind == 1:
                values = rng.normal(first, second, count)
            else:
                values = rng.lognormal(first, second, count)
            inputs[:, column] = values
        if not np.isfinite(inputs).all():
            raise ValueError("Input distributions produced non-finite samples")
        return inputs


@dataclass(frozen=True)
class FunctionConfig:
    """Settings for a new scalar problem; budget includes initial observations."""

    initial: int = 24
    budget: int = 60
    pool_size: int = 4096
    degree: int = 4
    q: float = 1.0
    seed: int = 42
    sampler: str = "AHSC"

    def __post_init__(self):
        for name in ("initial", "budget", "pool_size", "degree", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.initial < 3 or self.budget < self.initial:
            raise ValueError("Require 3 <= initial <= budget")
        if self.pool_size < max(1, self.budget - self.initial):
            raise ValueError("Candidate pool is too small for the requested budget")
        if self.degree < 1 or self.seed < 0 or not 0 < self.q <= 1:
            raise ValueError("Require degree >= 1, seed >= 0 and 0 < q <= 1")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"Unknown sampler: {self.sampler}")


class FunctionExperiment:
    """Fit a custom response using generated inputs and a fixed candidate pool.

    response(X) must return one finite scalar for each physical input row.
    Initial responses are requested together; subsequent calls contain one row.
    Test responses are evaluated separately by the caller after training.
    """

    def __init__(
        self,
        response: Callable[[np.ndarray], np.ndarray],
        poly_matrix,
        config: FunctionConfig | None = None,
        *,
        name: str = "custom_function",
        response_id: str | None = None,
        metadata: dict | None = None,
    ):
        if not callable(response):
            raise ValueError("response must be callable")
        self.response = response
        self.space = InputSpace(poly_matrix)
        self.config = config or FunctionConfig()
        self.name = name
        self.response_id = response_id
        self.metadata = metadata or {}
        self.policy = SamplingPolicy(self.config.sampler, self.space.poly_matrix)

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Call the response and validate one scalar per input row."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != self.space.dimension:
            raise ValueError(f"Expected inputs with shape (n, {self.space.dimension})")
        if not np.isfinite(inputs).all():
            raise ValueError("Response inputs must be finite")
        values = np.asarray(self.response(inputs.copy()), dtype=float)
        if values.shape not in ((len(inputs),), (len(inputs), 1)):
            raise ValueError("response(X) must return one scalar per row, shape (n,) or (n, 1)")
        values = values.ravel()
        if not np.isfinite(values).all():
            raise ValueError("response(X) returned non-finite values")
        return values

    def run(
        self,
        output: str | Path,
        *,
        progress: Callable[[Progress], None] | None = None,
        resume: bool = False,
    ) -> ExperimentResult:
        """Train or resume using committed responses, selections and model fits.

        Use the same output directory and resume=True after an interruption.
        ExternalSimulator additionally caches partially completed initial batches.
        """
        store = RunStore(output)
        if resume and not (store.directory / "settings.json").is_file():
            raise FileNotFoundError(f"No run settings to resume: {store.directory}")
        with FileLock(store.directory / ".run.lock"):
            return self._run_locked(store, progress, resume)

    def _run_locked(self, store, progress, resume):
        config = self.config
        checkpoint = FunctionCheckpoint(
            store,
            {
                "name": self.name,
                "run": asdict(config),
                "poly_matrix": self.space.poly_matrix,
                "algorithm_version": "SCAL-2026.10.18-ahsc-theta",
                "solver": "HCBPSolver V11",
                "checkpoint_schema": 1,
                "runtime": runtime_identity(),
                "response": getattr(
                    self.response, "checkpoint_identity", callable_identity(self.response)
                ),
                "response_id": self.response_id,
                "response_options": getattr(self.response, "checkpoint_options", {}),
                "metadata": self.metadata,
            },
            resume,
        )
        inputs, candidates, candidate_ids = checkpoint.inputs(self.space, config)
        # Resume with the saved design, candidate pool and candidate IDs.
        input_file = store.directory / "generated_inputs.npz"
        selected_ids, history = [], []
        previous = None
        started = time.perf_counter()
        try:
            responses = checkpoint.initial_responses(inputs, self.evaluate)
            self._save_observations(store, inputs, responses, selected_ids)
            with threadpool_limits(limits=1):
                for budget in range(config.initial, config.budget + 1):
                    # A committed fit supplies the next budget's warm start.
                    # An interrupted fit is repeated using the saved responses.
                    model, record = checkpoint.load_fit(budget, config, inputs, responses)
                    restored = model is not None
                    if model is None:
                        tick = time.perf_counter()
                        model = HCBPSolver(
                            self.space.poly_matrix,
                            max_degree=config.degree,
                            q=config.q,
                            last_model=previous,
                        ).fit(inputs, responses)
                        model.last_model = None
                        record = {
                            "budget": budget,
                            "fit_seconds": time.perf_counter() - tick,
                            "previous_selected": bool(model.warm_start_used_),
                            "next": None,
                        }
                        checkpoint.save_fit(model, record)
                    history.append(
                        {
                            "budget": budget,
                            "mean": float(model.mean_),
                            "std": float(model.std_),
                            "support_size": len(model.support_indices_),
                            "corrected_press": float(model.error_),
                            "fit_seconds": record["fit_seconds"],
                        }
                    )
                    next_id = None
                    if budget < config.budget:
                        if record["next"] is None:
                            available = ~np.isin(candidate_ids, selected_ids)
                            selection = self.policy.select(
                                model,
                                inputs,
                                responses,
                                candidates[available],
                                candidate_ids[available],
                            )
                            # Commit selection before starting an expensive solver.
                            record["next"] = dict(selection.record, point=selection.point[0])
                            checkpoint.save_record(record)
                        pending = record["next"]
                        next_id = int(pending["candidate_id"])
                        point = np.asarray(pending["point"], dtype=float).reshape(1, -1)
                        match = np.flatnonzero(candidate_ids == next_id)
                        if (
                            len(match) != 1
                            or next_id in selected_ids
                            or not np.array_equal(point[0], candidates[match[0]])
                        ):
                            raise ValueError(f"Invalid saved candidate at N={budget}")
                        if "response" not in pending:
                            pending["response"] = float(self.evaluate(point)[0])
                            # Commit before the enlarged training set reaches the
                            # next fit, so this response survives a fitting error.
                            checkpoint.save_record(record)
                        value = float(pending["response"])
                        if not np.isfinite(value):
                            raise ValueError(f"Invalid saved response at N={budget}")
                        selected_ids.append(next_id)
                        inputs = np.vstack([inputs, point])
                        responses = np.r_[responses, value]
                        self._save_observations(store, inputs, responses, selected_ids)
                    if progress and (not restored or not resume):
                        progress(
                            Progress(
                                budget,
                                config.budget,
                                len(model.support_indices_),
                                next_id,
                            )
                        )
                    previous = model
        except BaseException as error:
            write_json(
                store.directory / "failure.json",
                {"error": str(error), "type": type(error).__name__},
            )
            raise
        if resume and (store.directory / "run.json").exists():
            return ExperimentResult(store.directory, read_json(store.directory / "run.json"))
        summary = {
            "problem": self.name,
            "seed": config.seed,
            "solver": "HCBPSolver",
            "sampler": config.sampler,
            "initial": config.initial,
            "final": config.budget,
            "fits": len(history),
            "training_response_evaluations": len(responses),
            "candidate_responses_precomputed": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "numeric_threads": 1,
            "algorithm_version": "SCAL-2026.10.18-ahsc-theta",
            "input_sha256": digest(input_file),
            "selected_ids": selected_ids,
            "checkpoint_schema": 1,
            "resumed": resume,
        }
        import io

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        atomic_bytes(store.directory / "history.csv", stream.getvalue().encode("utf-8"))
        store.finish(model, inputs, responses, selected_ids, summary)
        return ExperimentResult(store.directory, summary)

    @staticmethod
    def _save_observations(store, inputs, responses, selected_ids):
        write_npz(
            store.directory / "observations.npz",
            X=inputs,
            y=responses,
            selected_ids=np.asarray(selected_ids, dtype=np.int64),
        )
