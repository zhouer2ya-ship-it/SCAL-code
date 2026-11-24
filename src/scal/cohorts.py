"""Run a named paper comparison over explicitly selected random seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import RunConfig
from .datasets import DatasetRepository
from .experiment import Experiment, Progress
from .io import read_json, write_json


class CohortRunner:
    """Validate all requested inputs and output directories before fitting."""

    def __init__(self, repository: DatasetRepository):
        self.repository = repository
        self.tasks = {
            task["id"]: task
            for task in read_json(repository.root / "configs/paper_tasks.json")["tasks"]
        }

    def run(
        self,
        task_id: str,
        output: str | Path,
        *,
        seeds: list[int] | None = None,
        stop: int | None = None,
        progress: Callable[[Progress], None] | None = None,
    ) -> dict:
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task {task_id!r}; use 'scal list' to see available tasks")
        task = self.tasks[task_id]
        seeds = task["seeds"] if seeds is None else seeds
        if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(task["seeds"]):
            raise ValueError("Choose distinct seeds from the task's published seed list")
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        configurations = [
            RunConfig(task["problem"], seed, task["solver"], task["sampler"], stop)
            for seed in seeds
        ]
        for config in configurations:
            config.validate(self.repository.config.problem(config.problem))
            self.repository.input_path(
                f"benchmarks/{config.problem}/seed_{config.seed:03d}/training.npz"
            )
        runs = []
        for config in configurations:
            result = Experiment(self.repository, config).run(
                output / f"seed_{config.seed:03d}", progress=progress
            )
            runs.append(result.summary)
        report = {"task": task_id, "seeds": seeds, "runs": runs}
        write_json(output / "cohort.json", report)
        return report
