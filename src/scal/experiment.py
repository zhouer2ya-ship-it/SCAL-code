"""Sequential fitting, response evaluation and sampling for one benchmark run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from .acquisition import SamplingPolicy
from .benchmarks import response
from .config import RunConfig
from .datasets import DatasetRepository
from .io import write_json
from .solving.registry import make_solver
from .storage import RunStore


@dataclass(frozen=True)
class Progress:
    budget: int
    final_budget: int
    selected_terms: int
    next_candidate_id: int | None


@dataclass(frozen=True)
class ExperimentResult:
    directory: Path
    summary: dict


class Experiment:
    """Run a configured paper benchmark and save each fitted model.

    Example
    -------
    >>> experiment = Experiment.from_directory(".", RunConfig(stop=35))
    >>> result = experiment.run("outputs/example")

    The repository supplies initial observations and candidates. Test data are
    loaded separately by BenchmarkEvaluator after the training run finishes.
    """

    def __init__(self, repository: DatasetRepository, config: RunConfig):
        self.repository = repository
        self.config = config
        self.problem = repository.config.problem(config.problem)
        self.final_budget = config.validate(self.problem)
        self.policy = SamplingPolicy(config.sampler, self.problem.poly_matrix)

    @classmethod
    def from_directory(
        cls,
        root: str | Path | None = None,
        config: RunConfig | None = None,
        *,
        data_root: str | Path | None = None,
    ) -> Experiment:
        return cls(DatasetRepository(root, data_root), config or RunConfig())

    def _fit(self, inputs, responses, previous):
        return make_solver(
            self.config.solver,
            self.problem.poly_matrix,
            random_state=42,
            max_degree=self.problem.degree,
            q=self.problem.q,
            last_model=previous,
            **self.repository.config.solver_parameters[self.config.solver],
        ).fit(inputs, responses)

    def run(
        self, output: str | Path, *, progress: Callable[[Progress], None] | None = None
    ) -> ExperimentResult:
        """Fit, select one point, evaluate it, append its response, and repeat."""
        data = self.repository.training(self.problem, self.config.seed)
        # Check saved initial responses against the bundled analytic function.
        np.testing.assert_allclose(
            response(self.problem.name, data.inputs), data.responses, rtol=2e-13, atol=1e-13
        )
        store = RunStore(output)
        store.create(
            {"run": asdict(self.config), "algorithm_version": self.repository.config.version}
        )
        inputs, responses = data.inputs, data.responses
        selected_ids: list[int] = []
        previous = None
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=1):
                for budget in range(self.problem.initial, self.final_budget + 1):
                    tick = time.perf_counter()
                    model = self._fit(inputs, responses, previous)
                    # Break the fitted model's link to its warm start.
                    model.last_model = None
                    record = {
                        "budget": budget,
                        "fit_seconds": time.perf_counter() - tick,
                        "previous_selected": bool(model.warm_start_used_),
                        "next": None,
                    }
                    selection = None
                    if budget < self.problem.maximum:
                        # Immutable IDs preserve tie-breaking as pool rows are removed.
                        available = ~np.isin(data.candidate_ids, selected_ids)
                        selection = self.policy.select(
                            model,
                            inputs,
                            responses,
                            data.candidates[available],
                            data.candidate_ids[available],
                        )
                        value = float(response(self.problem.name, selection.point)[0])
                        # The saved next-query record is also useful when stopping early.
                        record["next"] = dict(
                            selection.record, response=value, point=selection.point[0]
                        )
                    store.save_fit(model, record)
                    if progress:
                        progress(
                            Progress(
                                budget,
                                self.final_budget,
                                len(model.support_indices_),
                                selection.candidate_id if selection else None,
                            )
                        )
                    if budget < self.final_budget:
                        # The next fit sees exactly one additional true response.
                        selected_ids.append(selection.candidate_id)
                        inputs = np.vstack([inputs, selection.point])
                        responses = np.r_[responses, value]
                    previous = model
        except Exception as error:
            write_json(
                store.directory / "failure.json",
                {"error": str(error), "type": type(error).__name__},
            )
            raise
        summary = {
            "problem": self.config.problem,
            "seed": self.config.seed,
            "solver": self.config.solver,
            "sampler": self.config.sampler,
            "fits": self.final_budget - self.problem.initial + 1,
            "initial": self.problem.initial,
            "final": self.final_budget,
            "elapsed_seconds": time.perf_counter() - started,
            "input_path": data.source_path,
            "numeric_threads": 1,
            "test_reads_during_training": 0,
            "algorithm_version": self.repository.config.version,
            "selected_ids": selected_ids,
        }
        store.finish(model, inputs, responses, selected_ids, summary)
        return ExperimentResult(store.directory, summary)
