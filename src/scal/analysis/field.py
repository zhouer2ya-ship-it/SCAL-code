"""Re-evaluate the frozen field models, queries and fixed-point integral."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import eval_hermitenorm
from threadpoolctl import threadpool_limits

from ..applications import FieldAnalysis, design
from ..data import Sources
from ..io import write_json
from ..sampling.ahsc import modal_ahsc_candidate
from .tables import TableChecker


def aligned_grid_indices(model_ids, evaluation_ids):
    order = np.argsort(model_ids)
    sorted_ids = model_ids[order]
    indices = sorted_ids.searchsorted(evaluation_ids)
    if (indices == len(sorted_ids)).any() or not np.array_equal(
        sorted_ids[indices], evaluation_ids
    ):
        raise ValueError("The field model lacks positions in the common grid")
    return order[indices]


class FieldReplay:
    """Use the paper's 11 saved PCA/PCE fits and common 30-case evaluation grid."""

    def __init__(self, data_root: str | Path):
        self.sources = Sources(data_root=data_root)

    def _fixed_point(self):
        with np.load(
            self.sources.path("Field_fixed_point_scalar_model.npz"), allow_pickle=False
        ) as model:
            degrees, coefficients, offset = (
                model["degrees"],
                model["coefficients"],
                float(model["offset"]),
            )
        with np.load(
            self.sources.path("field_fixed_point_draws.npz"), allow_pickle=False
        ) as original:
            inputs = original["standard_normal_inputs"]
        truth = np.load(
            self.sources.path("field_fixed_point_stress.npy"), mmap_mode="r", allow_pickle=False
        )
        if inputs.shape != (1_000_000, 3) or truth.shape != (1_000_000,):
            raise ValueError("Expected the frozen million-draw fixed-point quadrature")
        curve = self.sources.frame("Field_fixed_point_failure_curve.csv")
        counts = np.zeros(len(curve), dtype=np.int64)
        maximum = 0.0
        thresholds = curve.threshold_dyn_cm2.to_numpy()
        for start in range(0, len(inputs), 25_000):
            x = inputs[start : start + 25_000]
            matrix = np.ones((len(x), len(coefficients)))
            for axis in range(x.shape[1]):
                matrix *= eval_hermitenorm(degrees[:, axis], x[:, axis, None])
            values = matrix @ coefficients + offset
            recorded = truth[start : start + len(values)]
            np.testing.assert_allclose(values, recorded, rtol=1e-12, atol=1e-7)
            maximum = max(maximum, float(np.max(abs(values - recorded))))
            counts += (values[:, None] > thresholds).sum(axis=0)
        np.testing.assert_array_equal(counts, curve.model_exceedance_count)
        at_threshold = int(counts[np.flatnonzero(thresholds == 100_000)[0]])
        return {
            "test_draws": len(inputs),
            "thresholds": len(counts),
            "threshold_dyn_cm2": 100_000,
            "exceedance_count": at_threshold,
            "probability": at_threshold / len(inputs),
            "max_abs_stress_difference": maximum,
        }

    def run(self, output: str | Path) -> dict:
        checker = TableChecker(self.sources, output)
        final = FieldAnalysis(self.sources.path("field_final_model.npz"))
        protocol = self.sources.reanalysis_protocol()["field"]
        case_ids = tuple(protocol["figure_6_display_cases"])
        truth = final.data["truth"]
        moments, case_errors, queries = [], [], []
        pool = np.load(self.sources.path("field_candidate_pool.npy"), allow_pickle=False)
        ids = np.arange(len(pool))
        training = final.data["training_X"]
        # The archived pool contains immutable original IDs. A selected ID
        # disappears from the next vote when its coordinate becomes an observation.
        with threadpool_limits(limits=1):
            final_checks = final.calculate(case_ids=case_ids)
            for n in range(10, 21):
                if n == 20:
                    model = final.data
                else:
                    with np.load(
                        self.sources.path(f"field_N{n:04d}.npz"), allow_pickle=False
                    ) as arrays:
                        model = {key: arrays[key] for key in arrays.files}
                alignment = aligned_grid_indices(model["grid_ids"], final.data["grid_ids"])
                matrix = design(
                    final.data["test_inputs"], model["basis_degrees"], model["poly_matrix"]
                )
                modes = matrix @ model["coefficients"]
                predictions = (
                    modes @ model["pca_reconstruction"][alignment].T + model["pca_mean"][alignment]
                )
                error = np.mean((predictions - truth) ** 2, axis=1) / truth.var(axis=1)
                case_errors.extend(
                    {"budget": n, "case_id": i, "relmse": e} for i, e in enumerate(error, 1)
                )
                row = {"N": n}
                for key, statistic in [("mean", np.mean), ("sd", np.std)]:
                    reference = statistic(truth, axis=0)
                    row[f"{key}_field_relative_l2"] = np.linalg.norm(
                        statistic(predictions, axis=0) - reference
                    ) / np.linalg.norm(reference)
                moments.append(row)
                if n < 20:
                    mapping = {}
                    for degrees in model["basis_degrees"]:
                        factors = [(j, d) for j, d in enumerate(degrees) if d]
                        key = "*".join(f"X{j}_{d}" for j, d in factors) or "x00"
                        expression = "*".join(
                            f"hermite_polynomial(X[{j}], {d})" for j, d in factors
                        )
                        mapping[key] = expression or "1"
                    fitted = [
                        SimpleNamespace(
                            is_fitted_=True,
                            poly_matrix=model["poly_matrix"],
                            mapping_=mapping,
                            coef_=coef,
                        )
                        for coef in model["coefficients"].T
                    ]
                    winner = modal_ahsc_candidate(training[:n], pool, fitted, candidate_ids=ids)
                    np.testing.assert_array_equal(pool[winner], training[n])
                    queries.append({"budget": n + 1, "candidate_id": int(ids[winner])})
            fixed_point = self._fixed_point()
        checker.compare("field_case_metrics.csv", pd.DataFrame(case_errors), ["budget", "case_id"])
        checker.compare("Field_moment_trajectory.csv", pd.DataFrame(moments), ["N"])
        checker.compare("field_query_order.csv", pd.DataFrame(queries), ["budget"])
        with np.load(
            self.sources.path("Field_final_statistics.npz"), allow_pickle=False
        ) as expected:
            for key in ("population_mean", "population_sd"):
                np.testing.assert_allclose(final_checks[key], expected[key], rtol=3e-12, atol=1e-8)
        report = {
            "passed": True,
            "scope": "field",
            "budgets": len(moments),
            "display_cases": case_ids,
            "checks": checker.checks,
            "figure_6l": fixed_point,
        }
        write_json(checker.output / "verification.json", report)
        return report
