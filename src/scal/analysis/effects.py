"""Paired effects, resampling intervals and original-family Holm corrections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import Sources
from ..io import write_json
from ..statistics import comparisons


class PairedEffectAnalysis:
    """Recompute either the Theta-exchange or mechanism comparison family."""

    SCOPES = ("theta_exchange", "mechanism")
    KEYS = ["statistical_family", "contrast_id", "metric", "floor"]

    def __init__(self, root: str | Path | None = None, data_root: str | Path | None = None):
        self.sources = Sources(root, data_root)

    def run(self, output: str | Path, scope: str = "theta_exchange") -> dict:
        if scope not in self.SCOPES:
            raise ValueError(f"Choose one of {self.SCOPES}")
        if scope == "theta_exchange":
            curves = self.sources.frame("source_exchange_primary_curves.csv")
            curves["task"] = curves.outcome
            name, values_name = (
                "source_exchange_Theta_effects.csv",
                "source_exchange_Theta_seed_effects.csv",
            )
        else:
            curves = self.sources.frame("mechanism_curves.csv")
            name, values_name = "mechanism_effects.csv", "mechanism_seed_effects.csv"
        expected = self.sources.frame(name)
        definitions = [
            {
                "family": row.statistical_family,
                "contrast": row.contrast_id,
                "weights": json.loads(row.weights),
                "n": int(row.planned_n),
                "label": row.interpretation,
                "external": bool(row.external),
            }
            for _, row in expected.drop_duplicates(["statistical_family", "contrast_id"]).iterrows()
        ]
        actual, seed_values = comparisons(curves, definitions)
        self._compare(actual, expected, seed_values, values_name)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        actual.to_csv(output / name, index=False, float_format="%.17g")
        seed_values.to_csv(output / values_name, index=False, float_format="%.17g")
        report = {
            "passed": True,
            "scope": scope,
            "effects": len(actual),
            "seed_effects": len(seed_values),
            "bootstrap_draws": 5000,
            "sign_flip_draws": 100000,
            "holm": "entire original family, including non-displayed contrasts",
        }
        write_json(output / "verification.json", report)
        return report

    def _compare(self, actual, expected, seed_values, values_name: str) -> None:
        actual = actual.set_index(self.KEYS).sort_index()
        expected = expected.set_index(self.KEYS).sort_index()
        np.testing.assert_array_equal(actual.index.to_numpy(), expected.index.to_numpy())
        columns = [
            "median",
            "q1",
            "q3",
            "ci_low",
            "ci_high",
            "one_sided_95_upper",
            "p_raw",
            "p_adjusted",
            "n_valid",
            "negative_count",
            "positive_count",
            "exact_zero_count",
        ]
        for column in columns:
            np.testing.assert_allclose(
                actual[column],
                expected[column],
                rtol=3e-11,
                atol=2e-13,
                equal_nan=True,
                err_msg=column,
            )
        old = self.sources.frame(values_name).set_index(self.KEYS + ["seed"]).sort_index()
        new = seed_values.set_index(self.KEYS + ["seed"]).sort_index()
        np.testing.assert_array_equal(new.index.to_numpy(), old.index.to_numpy())
        np.testing.assert_allclose(new.difference, old.difference, rtol=3e-11, atol=2e-13)
