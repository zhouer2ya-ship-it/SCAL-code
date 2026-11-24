"""Typed experiment settings loaded from the paper's JSON configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json
from .solving.registry import SOLVERS

SAMPLERS = ("AHSC", "Theta", "SOptimal", "MAXGradient", "MAXDistance")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed input law, polynomial dictionary and sample counts for one problem."""

    name: str
    poly_matrix: np.ndarray
    degree: int
    q: float
    initial: int
    maximum: int
    seeds: tuple[int, ...]
    reference_mean: float
    reference_std: float

    @property
    def dimension(self) -> int:
        return len(self.poly_matrix)


@dataclass(frozen=True)
class RunConfig:
    """Choices for a run; the mathematical benchmark settings remain in PaperConfig.

    ``stop`` is a sample count, including the initial observations. ``None``
    uses the paper's final sample count. Every run uses one numerical thread.
    """

    problem: str = "ishigami"
    seed: int = 0
    solver: str = "HCBPSolver"
    sampler: str = "AHSC"
    stop: int | None = None

    def __post_init__(self) -> None:
        for name in ("problem", "solver", "sampler"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.stop is not None and type(self.stop) is not int:
            raise ValueError("stop must be an integer or null")

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        values = read_json(path)
        if not isinstance(values, dict):
            raise ValueError("Run configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown run options: {', '.join(sorted(unknown))}")
        return cls(**values)

    def validate(self, problem: BenchmarkConfig) -> int:
        """Check settings and return the final sample count."""
        if self.problem != problem.name:
            raise ValueError("Run and benchmark names differ")
        if type(self.seed) is not int or self.seed not in problem.seeds:
            raise ValueError(f"Seed {self.seed!r} is outside the {problem.name} seed list")
        if self.solver not in SOLVERS:
            raise ValueError(f"Unknown solver {self.solver!r}; choose from {', '.join(SOLVERS)}")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"Unknown sampler {self.sampler!r}; choose from {', '.join(SAMPLERS)}")
        final = problem.maximum if self.stop is None else self.stop
        if type(final) is not int or not problem.initial <= final <= problem.maximum:
            raise ValueError(f"stop must be an integer from {problem.initial} to {problem.maximum}")
        return final


class PaperConfig:
    """Read the paper configuration once and expose named benchmark settings."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = read_json(self.path)
        self.version: str = self.raw["version"]
        self.solver_parameters: dict[str, dict[str, Any]] = self.raw["solver_parameters"]

    @property
    def problem_names(self) -> tuple[str, ...]:
        return tuple(self.raw["problems"])

    def problem(self, name: str) -> BenchmarkConfig:
        if name not in self.raw["problems"]:
            raise ValueError(
                f"Unknown benchmark {name!r}; choose from {', '.join(self.problem_names)}"
            )
        values = self.raw["problems"][name]
        mean, std = self.raw["reference_moments"][name]
        return BenchmarkConfig(
            name=name,
            poly_matrix=np.asarray(values["poly"], dtype=float),
            degree=values["degree"],
            q=values["q"],
            initial=values["initial"],
            maximum=values["maximum"],
            seeds=tuple(values["seeds"]),
            reference_mean=mean,
            reference_std=std,
        )
