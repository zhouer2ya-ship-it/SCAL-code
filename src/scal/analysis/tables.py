"""Group seed-level records and compare tabulated results."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import Sources


def summarize(
    frame: pd.DataFrame, keys: list[str], metrics: list[str], *, counts: bool = False
) -> pd.DataFrame:
    """Count rows and calculate the 25th, 50th and 75th percentiles per group."""
    groups = frame.groupby(keys, sort=True, dropna=False)
    parts = [groups.size().rename("n")]
    for metric in metrics:
        if counts:
            parts.append(groups[metric].count().rename(metric + "_n"))
        for quantile, label in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
            parts.append(groups[metric].quantile(quantile).rename(metric + "_" + label))
    return pd.concat(parts, axis=1).reset_index()


class TableChecker:
    """Compare published rows, write their numeric columns and record the check."""

    def __init__(self, sources: Sources, output: str | Path):
        self.sources = sources
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checks: list[dict] = []

    def compare(self, name: str, actual: pd.DataFrame, keys: list[str]) -> None:
        expected = self.sources.frame(name).set_index(keys)
        actual = actual.set_index(keys)
        if actual.index.has_duplicates or expected.index.has_duplicates:
            raise ValueError(f"Duplicate rows in {name}")
        if len(expected.index.difference(actual.index)):
            raise ValueError(f"Calculated rows are missing from {name}")
        extra_rows = len(actual.index.difference(expected.index))
        actual = actual.reindex(expected.index)
        checked = []
        for column in expected:
            if column in actual and pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(
                    actual[column],
                    expected[column],
                    rtol=2e-12,
                    atol=2e-14,
                    equal_nan=True,
                    err_msg=f"{name}:{column}",
                )
                checked.append(column)
        if not checked:
            raise ValueError(f"No numeric columns checked: {name}")
        actual.reset_index().to_csv(self.output / name, index=False, float_format="%.17g")
        self.checks.append(
            {
                "source": name,
                "rows": len(expected),
                "columns": checked,
                "rows_outside_saved_table": extra_rows,
            }
        )
