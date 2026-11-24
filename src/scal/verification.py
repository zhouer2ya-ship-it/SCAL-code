"""Compare saved runs and verify a Git checkout."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from .io import read_json, write_json
from .storage import RunStore


class ReferenceComparison:
    """Check every fit requested by a run against an independent saved result."""

    def __init__(self, expected: str | Path):
        self.expected = Path(expected)

    def compare(self, actual: str | Path, output: str | Path | None = None) -> dict:
        actual = Path(actual)
        run = read_json(actual / "run.json")
        snapshots = RunStore(actual).snapshots()
        budgets = list(range(run["initial"], run["final"] + 1))
        if [snapshot.budget for snapshot in snapshots] != budgets or run["fits"] != len(budgets):
            raise ValueError("Saved fits do not cover every budget in run.json")
        # IC takes the logarithm of the fitted error, amplifying small BLAS
        # differences. Model coefficients, moments and query scores retain the
        # tighter tolerance; supports and selected candidate IDs remain exact.
        tolerances = {
            "default": {"rtol": 2e-12, "atol": 1e-13},
            "IC": {"rtol": 1e-10, "atol": 1e-13},
        }
        max_absolute_difference_by_array = {}
        checks = []
        for snapshot in snapshots:
            path = self.expected / f"N{snapshot.budget:04d}.npz"
            exact, difference = True, 0.0
            with np.load(path, allow_pickle=False) as expected:
                if set(expected.files) != set(snapshot.values):
                    raise ValueError(f"Saved arrays differ: {path.name}")
                np.testing.assert_array_equal(expected["support"], snapshot.values["support"])
                for key in expected.files:
                    if key == "support":
                        continue
                    reference, observed = expected[key], snapshot.values[key]
                    np.testing.assert_allclose(
                        observed,
                        reference,
                        **tolerances.get(key, tolerances["default"]),
                        equal_nan=True,
                        err_msg=f"{path.name}:{key}",
                    )
                    exact = exact and np.array_equal(reference, observed, equal_nan=True)
                    finite = np.isfinite(reference) & np.isfinite(observed)
                    array_difference = 0.0
                    if np.any(finite):
                        array_difference = float(np.max(abs(reference[finite] - observed[finite])))
                        difference = max(difference, array_difference)
                    max_absolute_difference_by_array[key] = max(
                        max_absolute_difference_by_array.get(key, 0.0), array_difference
                    )
            reference_record = read_json(path.with_suffix(".json"))
            expected_query, actual_query = reference_record["next"], snapshot.record["next"]
            expected_id = None if expected_query is None else expected_query["candidate_id"]
            actual_id = None if actual_query is None else actual_query["candidate_id"]
            if actual_id != expected_id:
                raise AssertionError(
                    f"N={snapshot.budget}: candidate {actual_id} != saved {expected_id}"
                )
            if expected_query:
                for key in (
                    "normalized_gradient",
                    "normalized_theta",
                    "score",
                    "raw_gradient_norm",
                    "raw_original_theta",
                ):
                    if expected_query.get(key) is not None:
                        np.testing.assert_allclose(
                            actual_query[key],
                            expected_query[key],
                            **tolerances["default"],
                            err_msg=key,
                        )
            checks.append(
                {
                    "budget": snapshot.budget,
                    "candidate_id": actual_id,
                    "exact_numeric_equality": bool(exact),
                    "max_absolute_difference": difference,
                }
            )
        report = {
            "passed": True,
            "fits": len(checks),
            "all_numeric_arrays_exact": all(item["exact_numeric_equality"] for item in checks),
            "candidate_ids_exact": True,
            "supports_exact": True,
            "numeric_tolerances": tolerances,
            "max_absolute_difference_by_array": max_absolute_difference_by_array,
            "checks": checks,
        }
        if output is not None:
            write_json(output, report)
        return report


def verify_release(root: str | Path, ref: str = "HEAD") -> dict:
    """Compare the checkout's tracked files with a Git commit or tag."""
    root = Path(root).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, encoding="utf-8"
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "A Git checkout is required")
        return result.stdout

    commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").strip()
    tracked = git("ls-tree", "-rz", "--name-only", commit)
    worktree = git("diff", "--name-only", "-z", commit, "--")
    staged = git("diff", "--cached", "--name-only", "-z", commit, "--")
    changed = sorted(set(worktree.split("\0") + staged.split("\0")) - {""})
    return {
        "passed": not changed,
        "git_ref": ref,
        "git_commit": commit,
        "files": tracked.count("\0"),
        "changed": changed,
    }
