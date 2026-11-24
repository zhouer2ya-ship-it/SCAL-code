"""Recompute benchmark summaries, mechanism tables and application statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..applications import FieldAnalysis, SRAMAnalysis
from ..data import Sources
from ..io import write_json
from .tables import TableChecker, summarize

COMPOSITION_FIELDS = [
    "selected_resolved_oracle_count",
    "selected_subthreshold_oracle_count",
    "selected_structural_anchor_count",
    "selected_other_nonoracle_count",
    "terminal_support_size",
]


class PaperAnalysis:
    """Calculate benchmark summaries, component comparisons and application statistics."""

    def __init__(self, root: str | Path | None = None, data_root: str | Path | None = None):
        self.sources = Sources(root, data_root)

    def run(self, output: str | Path) -> dict:
        checker = TableChecker(self.sources, output)
        self.benchmark_curves(checker)
        self.query_percentiles(checker)
        self.support_composition(checker)
        self.support_stability(checker)
        self.shared_theta(checker)
        self.score_components(checker)
        with threadpool_limits(limits=1):
            self.sram(checker)
            self.field(checker)
        report = {
            "passed": True,
            "checks": checker.checks,
            "coverage": "Reference examples and comparison tolerances are in docs/REPRODUCIBILITY.md.",
        }
        write_json(Path(output) / "verification.json", report)
        return report

    def benchmark_curves(self, checker: TableChecker) -> None:
        parts = []
        suites = [
            "ishigami_solvers",
            "rackwitz_solvers",
            "100d_solvers",
            "ishigami_sampling",
            "rosen_sampling",
            "100d_sampling",
        ]
        for index, suite in enumerate(suites, 1):
            frame = self.sources.frame(f"validation_V{index}_curves.csv")
            frame["suite"] = suite
            # The saved table uses ThetaOriginal as the name of the exact Theta rule.
            frame["method"] = frame.method.replace(
                {"Theta": "ThetaOriginal", "LarsCVSolver (tied-drop fix)": "LarsCVSolver"}
            )
            frame["e_mu"], frame["e_sigma"] = (
                frame.mean_standardized_error,
                frame.std_relative_error,
            )
            parts.append(frame)
        parts.append(self.sources.frame("soptimal_sampling_metrics.csv"))
        raw = pd.concat(parts, ignore_index=True)
        if raw.duplicated(["suite", "method", "seed", "budget"]).any():
            raise ValueError("Duplicate seed/budget records")
        keys = ["suite", "problem", "method", "budget"]
        checker.compare(
            "components_curve_statistics.csv",
            summarize(raw, keys, ["e_mu", "e_sigma", "relmse", "support_size"], counts=True),
            keys,
        )

    def query_percentiles(self, checker: TableChecker) -> None:
        frame = self.sources.frame("Figure4_query_diagnostics.csv")
        keys = ["query_policy", "budget"]
        checker.compare(
            "Figure4_query_percentile_summary.csv",
            summarize(frame, keys, ["gradient_percentile", "theta_percentile"]),
            keys,
        )

    def support_composition(self, checker: TableChecker) -> None:
        frame = self.sources.frame("Figure5_support_composition_by_seed.csv")
        np.testing.assert_array_equal(
            frame[COMPOSITION_FIELDS[:-1]].sum(axis=1), frame.terminal_support_size
        )
        actual = frame.groupby("method_id", as_index=False)[COMPOSITION_FIELDS].mean()
        actual["n_seeds"] = 100
        checker.compare("Figure4_support_composition_means.csv", actual, ["method_id"])

    def support_stability(self, checker: TableChecker) -> None:
        frame = self.sources.frame("ED3_support_stability_by_seed.csv")
        keys = ["suite", "method", "phase"]
        metrics = [
            "mean_adjacent_jaccard",
            "member_replacement_fraction",
            "any_support_change_fraction",
        ]
        checker.compare("ED3_support_stability_summary.csv", summarize(frame, keys, metrics), keys)

    def shared_theta(self, checker: TableChecker) -> None:
        exchange = self.sources.frame("source_exchange_primary_curves.csv")
        shared = exchange[exchange.task.isin(["ThetaPath_H", "ThetaPath_R"])]
        keys = ["model_state", "budget"]
        checker.compare(
            "ED5_shared_Theta_error_curves.csv",
            summarize(shared, keys, ["std_relative_error", "relmse", "support_size"]),
            keys,
        )
        frame = self.sources.frame("SupportIdentity_SeedSummary.csv")
        actual = frame.groupby("state", as_index=False)[COMPOSITION_FIELDS].mean()
        actual["n_seeds"] = 420
        checker.compare("ED5_terminal_composition_means.csv", actual, ["state"])

    def score_components(self, checker: TableChecker) -> None:
        scores = self.sources.frame("S5_candidate_scores.csv")
        np.testing.assert_allclose(
            scores.normalized_gradient + scores.normalized_Theta,
            scores.AHSC_score,
            atol=5e-16,
            rtol=0,
        )
        checker.checks.append(
            {
                "source": "S5_candidate_scores.csv",
                "rows": len(scores),
                "columns": ["unit_coefficient_sum"],
            }
        )

    def sram(self, checker: TableChecker) -> None:
        result = SRAMAnalysis(self.sources.file("applications/sram.npz")).calculate()
        moments = pd.DataFrame(
            {key: result[key] for key in ["mean", "sd", "e_mu", "e_sigma", "terms"]}
            | {"N": result["budgets"]}
        )
        checker.compare("SRAM_moment_trajectory.csv", moments, ["N"])
        checker.compare(
            "SRAM_distribution_trajectory.csv", pd.DataFrame(result["distribution"]), ["N"]
        )
        sobol = pd.DataFrame(
            {
                "input_index_zero_based": np.arange(159),
                "first_order": result["first_order"],
                "total_order": result["total_order"],
            }
        )
        checker.compare("SRAM_sobol_indices_labeled.csv", sobol, ["input_index_zero_based"])

    def field(self, checker: TableChecker) -> None:
        result = FieldAnalysis(self.sources.file("applications/field.npz")).calculate()
        reference = self.sources.frame("Field_case_trajectory.csv")
        reference = reference[reference.N == 20].sort_values("case")
        np.testing.assert_allclose(result["case_relmse"], reference.RelMSE, rtol=3e-12, atol=1e-14)
        moments = self.sources.frame("Field_moment_trajectory.csv").set_index("N").loc[20]
        for key in ("mean_field_relative_l2", "sd_field_relative_l2"):
            np.testing.assert_allclose(result[key], moments[key], rtol=3e-12, atol=1e-14)
        with np.load(self.sources.path("Field_final_statistics.npz"), allow_pickle=False) as arrays:
            for key in ("population_mean", "population_sd"):
                np.testing.assert_allclose(result[key], arrays[key], rtol=3e-12, atol=1e-8)
        checker.checks.append(
            {
                "source": "field final surrogate and responses",
                "cases": result["test_cases"],
                "positions": result["positions"],
                "typical_case": result["typical_case"],
                "worst_case": result["worst_case"],
            }
        )
