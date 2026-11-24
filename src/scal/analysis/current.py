"""Recompute paired benchmark effects, recovery endpoints and timing summaries."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ..data import Sources
from ..io import write_json
from .tables import TableChecker, summarize

METRICS = ("mean_standardized_error", "std_relative_error", "relmse")
FLOORS = (1e-20, 1e-16)
MODEL_CONTROLS = ("NoHeredityModule", "NoBackwardSearch", "FrozenImportance", "NoHistory")
CROSSED_CONTRASTS = {
    "R_H": {"C_on_H": 1, "H_on_H": -1},
    "D_H": {"H_on_C": 1, "H_on_H": -1},
    "R_C": {"C_on_C": 1, "H_on_C": -1},
    "D_C": {"C_on_C": 1, "C_on_H": -1},
    "I": {"C_on_C": 1, "H_on_C": -1, "C_on_H": -1, "H_on_H": 1},
    "T": {"C_on_C": 1, "H_on_H": -1},
}


def log_areas(frame, groups, initial, final, n_seeds):
    """Keep every planned seed and every prescribed budget in the integration grid."""
    grid = pd.MultiIndex.from_product([range(n_seeds), range(initial, final + 1)])
    result = {}
    for key, rows in frame.groupby(groups, sort=True):
        values = rows.set_index(["seed", "budget"])[list(METRICS)]
        if len(values.index.difference(grid)):
            raise ValueError(f"Unplanned seed or budget for {key}")
        paths = values.reindex(grid).to_numpy().reshape(n_seeds, final - initial + 1, len(METRICS))
        valid = (np.isfinite(paths) & (paths >= 0)).all(axis=1)
        for floor in FLOORS:
            logvalues = np.log10(np.maximum(paths, floor))
            integrals = np.trapezoid(logvalues, axis=1) / (final - initial)
            integrals[~valid] = np.nan
            result[key, floor] = dict(zip(METRICS, integrals.T, strict=True))
    return result


def paired_median(values, bootstrap):
    valid = values[np.isfinite(values)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = np.nanmedian(values[bootstrap], axis=1)
    draws = draws[np.isfinite(draws)]
    q25, median, q75 = np.quantile(valid, [0.25, 0.5, 0.75]) if len(valid) else [np.nan] * 3
    low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else [np.nan] * 2
    return {
        "n_planned": len(values),
        "n_valid": len(valid),
        "median": median,
        "q25": q25,
        "q75": q75,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_valid": len(draws),
        "bootstrap_total": len(bootstrap),
    }


class CurrentAnalysis:
    """Verify current figure and table results using the published-data layout."""

    def __init__(self, data_root: str | Path):
        self.sources = Sources(data_root=data_root)
        self.protocol = self.sources.reanalysis_protocol()
        self.problems = self.protocol["problems"]
        self.bootstraps = {
            problem: np.load(self.sources.path(f"bootstrap_{problem}.npy"), allow_pickle=False)
            for problem in self.problems
        }
        for problem, indices in self.bootstraps.items():
            n = self.problems[problem]["n_seeds"]
            if (
                indices.shape != (5000, n)
                or indices.dtype.kind not in "iu"
                or np.any((indices < 0) | (indices >= n))
            ):
                raise ValueError(f"Invalid whole-seed bootstrap indices for {problem}")

    def _path_effects(self, name, baseline, controls):
        data = self.sources.frame(name)
        effect_rows = []
        for problem, rows in data.groupby("problem"):
            plan = self.problems[problem]
            areas = log_areas(rows, "arm", plan["initial"], plan["maximum"], plan["n_seeds"])
            for floor in FLOORS:
                ref = areas[baseline, floor]
                for arm in controls:
                    if (arm, floor) not in areas:
                        continue
                    for metric in METRICS:
                        values = areas[arm, floor][metric] - ref[metric]
                        key = {"problem": problem, "control": arm, "metric": metric, "floor": floor}
                        effect_rows.append(
                            {**key, **paired_median(values, self.bootstraps[problem])}
                        )
        return pd.DataFrame(effect_rows)

    def _matched(self, checker):
        data = self.sources.frame("matched_pairs.csv")
        count_keys = ["problem", "seed", "parent_budget"]
        differences = [metric + "_difference" for metric in METRICS]
        same_query = data.query_id == data.H_query_id
        np.testing.assert_array_equal(data.query_changed, ~same_query)
        np.testing.assert_array_equal(data.child_support_changed, data.child_support_jaccard < 1)
        np.testing.assert_array_equal(data.loc[same_query, differences], 0)
        for metric in METRICS:
            np.testing.assert_allclose(
                data[metric + "_difference"],
                data["child_" + metric] - data["H_child_" + metric],
                rtol=2e-12,
                atol=2e-14,
            )
        if data.duplicated(count_keys + ["arm"]).any():
            raise ValueError("Duplicate matched intervention")
        rows = []
        for (problem, arm), group in data.groupby(["problem", "arm"]):
            plan = self.problems[problem]
            grid = pd.MultiIndex.from_product([range(plan["n_seeds"]), plan["parent_budgets"]])
            values = group.set_index(["seed", "parent_budget"])[differences].sort_index()
            if not values.index.equals(grid):
                raise ValueError(f"Matched parent schedule differs for {problem}/{arm}")
            by_seed = values.groupby(level=0).median()
            by_seed[values.groupby(level=0).count() != len(plan["parent_budgets"])] = np.nan
            for metric in METRICS:
                values = by_seed[metric + "_difference"].to_numpy()
                rows.append(
                    {
                        "problem": problem,
                        "control": arm,
                        "metric": metric,
                        **paired_median(values, self.bootstraps[problem]),
                        "parent_states": len(group),
                        "same_query_states": int((group.query_id == group.H_query_id).sum()),
                        "zero_seed_effects": int((values == 0).sum()),
                    }
                )
        summary = pd.DataFrame(rows)
        expected = self.sources.frame("matched_effects.csv")
        # This calculation checks the three paper error contrasts. Other columns of
        # matched_pairs.csv retain the original per-parent diagnostic quantities.
        expected = expected[expected.metric.isin(METRICS)].set_index(
            ["problem", "control", "metric"]
        )
        self._compare_subset(checker, "matched_error_effects.csv", summary, expected)
        counts = {
            "parents": len(data[count_keys].drop_duplicates()),
            "interventions": len(data),
            "identical_queries": int(same_query.sum()),
            "changed_queries": int((~same_query).sum()),
            "changed_child_supports": int(data.child_support_changed.sum()),
        }
        if counts != self.protocol["matched_counts"]:
            raise ValueError(f"Matched intervention counts differ: {counts}")
        return counts

    @staticmethod
    def _compare_subset(checker, name, actual, expected):
        indexed = actual.set_index(expected.index.names)
        if indexed.index.has_duplicates or set(indexed.index) != set(expected.index):
            raise ValueError(f"Recomputed rows differ for {name}")
        columns = [column for column in indexed if column in expected]
        np.testing.assert_allclose(
            indexed.reindex(expected.index)[columns], expected[columns], rtol=2e-12, atol=2e-14
        )
        actual.to_csv(checker.output / name, index=False, float_format="%.17g")
        checker.checks.append({"source": name, "rows": len(actual), "columns": columns})

    def _crossed(self, checker):
        data = self.sources.frame("cross_cell_metrics.csv")
        plan = self.problems["ishigami"]
        areas = log_areas(
            data, ["control", "cell"], plan["initial"], plan["maximum"], plan["n_seeds"]
        )
        effect_rows, seed_rows = [], []
        for control in sorted(data.control.unique()):
            for floor in FLOORS:
                for metric in METRICS:
                    for contrast, weights in CROSSED_CONTRASTS.items():
                        values = sum(
                            weight * areas[(control, cell), floor][metric]
                            for cell, weight in weights.items()
                        )
                        key = dict(
                            problem="ishigami",
                            control=control,
                            metric=metric,
                            floor=floor,
                            contrast=contrast,
                        )
                        effect_rows.append(
                            {
                                **key,
                                **paired_median(values, self.bootstraps["ishigami"]),
                                "mean": float(np.nanmean(values))
                                if np.isfinite(values).any()
                                else np.nan,
                            }
                        )
                        seed_rows.extend(
                            {**key, "seed": seed, "effect": value}
                            for seed, value in enumerate(values)
                        )
        keys = ["problem", "control", "metric", "floor", "contrast"]
        effects = pd.DataFrame(effect_rows)
        checker.compare("cross_effects.csv", effects, keys)
        checker.compare("table2_crossed_effects.csv", effects, keys)
        checker.compare("cross_seed_effects.csv", pd.DataFrame(seed_rows), [*keys, "seed"])
        self._crossed_attainment(data, plan, checker)

    def _crossed_attainment(self, data, plan, checker):
        budgets = np.arange(plan["initial"], plan["maximum"] + 1)
        n = plan["n_seeds"]
        grid = pd.MultiIndex.from_product([range(n), budgets])
        records = []
        for (control, cell), rows in data.groupby(["control", "cell"]):
            values = rows.set_index(["seed", "budget"])[list(METRICS[:2])].reindex(grid)
            worst_error = values.to_numpy().max(axis=1).reshape(n, len(budgets))
            complete = np.isfinite(worst_error).all(axis=1)
            for threshold in (0.01, 0.001):
                meets = worst_error <= threshold
                sustained = np.logical_and.accumulate(meets[:, ::-1], axis=1)[:, ::-1]
                eligible = sustained.any(axis=1) & complete
                completion_budgets = budgets[sustained[eligible].argmax(axis=1)]
                boot = eligible[self.bootstraps["ishigami"]].mean(axis=1)
                q = (
                    np.quantile(completion_budgets, [0.25, 0.5, 0.75])
                    if len(completion_budgets)
                    else [np.nan] * 3
                )
                lo, hi = np.quantile(boot, [0.025, 0.975])
                records.append(
                    {
                        "control": control,
                        "cell": cell,
                        "threshold": threshold,
                        "n_planned": n,
                        "n_complete": int(complete.sum()),
                        "attained": int(eligible.sum()),
                        "attainment_rate": eligible.mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                        "budget_n_conditional": len(completion_budgets),
                        "budget_q25": q[0],
                        "budget_median": q[1],
                        "budget_q75": q[2],
                    }
                )
        checker.compare(
            "cross_attainment_summary.csv", pd.DataFrame(records), ["control", "cell", "threshold"]
        )

    def _fixed_design(self, checker):
        data = self.sources.frame("fixed_design_fits.csv")
        keys = ["problem", "setting"]
        if len(data) != 2040 or data.unit_id.duplicated().any():
            raise ValueError("Expected 2,040 unique fixed-design units")
        data = data[data.budget == data.groupby(keys).budget.transform("max")]
        if data.duplicated([*keys, "seed"]).any():
            raise ValueError("Repeated seed among fixed-design endpoint fits")
        statistics = data.groupby(keys, as_index=False).agg(
            endpoint_budget=("budget", "max"),
            n_designs=("seed", "size"),
            basis_count=("basis_count", "first"),
            exact_support_count=("exact_support", "sum"),
            median_precision=("support_precision", "median"),
            median_recall=("support_recall", "median"),
        )
        checker.compare("fixed_design_recovery_summary.csv", statistics, keys)

    def _timing(self, checker):
        data = self.sources.frame("timing_seed_totals.csv")
        keys = ["problem", "seed"]
        data["algorithm_seconds"] = data.fit_seconds + data.selection_seconds
        baseline = data.loc[data.arm == "H"].set_index(keys)["algorithm_seconds"]
        data = data.join(baseline.rename("baseline"), on=keys, validate="many_to_one")
        data["algorithm_seconds_ratio_to_H"] = data.algorithm_seconds / data.baseline
        metrics = [
            "fit_seconds",
            "selection_seconds",
            "algorithm_seconds",
            "algorithm_seconds_ratio_to_H",
        ]
        groups = ["problem", "arm"]
        summary = summarize(data, groups, metrics)
        summary = summary.rename(
            columns={
                "n": "timing_n",
                **{
                    f"{field}_{old}": f"{field}_{new}"
                    for field in metrics
                    for old, new in [("q1", "q25"), ("q3", "q75")]
                },
            }
        )
        checker.compare("timing_summary.csv", summary, groups)
        return len(data)

    def run(self, output: str | Path) -> dict:
        checker = TableChecker(self.sources, output)
        keys = ["problem", "control", "metric", "floor"]
        models = self._path_effects("closed_loop_metrics.csv", "H", MODEL_CONTROLS)
        checker.compare("table2_component_effects.csv", models, keys)
        model_display = models[
            models.problem.isin(["weak_parent", "rackwitz"]) & (models.floor == FLOORS[0])
        ]
        checker.compare("figure4_model_effects.csv", model_display, keys)
        sampling = self._path_effects("sampling_metrics.csv", "AHSC", ("Theta", "GradientOnly"))
        sampling_display = sampling[sampling.floor == FLOORS[0]]
        checker.compare("figure4_sampling_effects.csv", sampling_display, keys)
        counts = self._matched(checker)
        self._crossed(checker)
        timing_rows = self._timing(checker)
        self._fixed_design(checker)
        report = {
            "passed": True,
            "scope": "comparisons",
            "checks": checker.checks,
            "matched": counts,
            "timing_runs": timing_rows,
        }
        write_json(checker.output / "verification.json", report)
        return report
