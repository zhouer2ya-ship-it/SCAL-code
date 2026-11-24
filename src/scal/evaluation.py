"""Evaluate completed runs on the paper's independent test samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .datasets import DatasetRepository
from .io import read_json
from .solving.utils import construct_matrix_optimized, normalization_all
from .storage import FitSnapshot, RunStore


class BenchmarkEvaluator:
    """Compute test errors after training, using a shared basis for all fits."""

    def __init__(self, repository: DatasetRepository):
        self.repository = repository

    def evaluate(
        self,
        directory: str | Path,
        output: str | Path | None = None,
        *,
        test_set: str = "legacy",
    ) -> pd.DataFrame:
        directory = Path(directory)
        run = read_json(directory / "run.json")
        snapshots = RunStore(directory).snapshots()
        expected_budgets = list(range(run["initial"], run["final"] + 1))
        if [item.budget for item in snapshots] != expected_budgets:
            raise ValueError("Saved fits do not cover every budget in run.json")
        return self._evaluate(run["problem"], directory / "basis.json", snapshots, output, test_set)

    def evaluate_archive(
        self,
        path: str | Path,
        problem: str,
        basis: str | Path,
        output: str | Path | None = None,
        *,
        test_set: str = "mc100k",
    ) -> pd.DataFrame:
        """Evaluate archived N####.npz files with their frozen basis and test set."""
        path = Path(path)
        if path.is_file():
            snapshots = [FitSnapshot.read(path, require_record=False)]
        else:
            snapshots = RunStore(path).snapshots(require_records=False)
        return self._evaluate(problem, Path(basis), snapshots, output, test_set)

    def _evaluate(self, problem, basis_path, snapshots, output, test_set) -> pd.DataFrame:
        inputs, truth = self.repository.test(problem, test_set=test_set)
        if test_set == "mc100k":
            test = self.repository.paper_test(problem)
            ref_mean, ref_std = test["reference_mean"], test["reference_std"]
            test_id = test["test_set_id"]
        else:
            config = self.repository.config.problem(problem)
            ref_mean, ref_std = config.reference_mean, config.reference_std
            test_id = "legacy"
        denominator = len(truth) * np.var(truth)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"RelMSE needs finite, positive test response variance for {problem}")
        basis = read_json(basis_path)
        poly = np.asarray(basis["poly_matrix"])
        coefficients = np.asarray([item.values["coef"] for item in snapshots])
        keys = list(basis["mapping"])
        if coefficients.shape != (len(snapshots), len(keys)) or len(poly) != inputs.shape[1]:
            raise ValueError("Saved coefficients or test inputs do not match the supplied basis")
        active = np.flatnonzero(np.any(coefficients != 0, axis=0))
        mapping = {keys[index]: basis["mapping"][keys[index]] for index in active}
        squared_error = np.zeros(len(snapshots))
        with threadpool_limits(limits=1):
            for start in range(0, len(truth), 1024):
                matrix = construct_matrix_optimized(
                    mapping, normalization_all(inputs[start : start + 1024], poly).T
                )
                prediction = matrix @ coefficients[:, active].T
                squared_error += np.sum(
                    (prediction - truth[start : start + len(matrix), None]) ** 2, axis=0
                )
        rows = []
        for index, snapshot in enumerate(snapshots):
            values = snapshot.values
            mean, std = float(values["mean"]), float(values["std"])
            rows.append(
                {
                    "budget": snapshot.budget,
                    "relmse": squared_error[index] / denominator,
                    "mean": mean,
                    "std": std,
                    "mean_standardized_error": abs(mean - ref_mean) / ref_std,
                    "std_relative_error": abs(std - ref_std) / ref_std,
                    "support_size": len(values["support"]),
                }
            )
        frame = pd.DataFrame(rows)
        frame.attrs["independent_test_samples"] = len(truth)
        frame.attrs["test_set_id"] = test_id
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False, float_format="%.17g")
        return frame
