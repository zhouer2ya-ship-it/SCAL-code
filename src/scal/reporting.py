"""Create compact figures and tables from saved SCAL results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .io import read_json
from .storage import RunStore


def _validation(directory: Path, validation):
    if validation is None and (directory / "validation.npz").is_file():
        with np.load(directory / "validation.npz", allow_pickle=False) as arrays:
            validation = arrays["y"].copy(), arrays["prediction"].copy()
    if validation is None:
        return None
    truth, prediction = (np.asarray(values, dtype=float) for values in validation)
    if truth.ndim != 1 or truth.shape != prediction.shape or len(truth) < 2:
        raise ValueError(
            "Validation requires two matching one-dimensional arrays with at least two values"
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Validation responses and predictions must be finite")
    return truth, prediction


def _line(axis, budgets, values, ylabel, *, log=False, count=False):
    from matplotlib.ticker import MaxNLocator

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    axis.plot(budgets, np.where(finite, values, np.nan), color="#0072B2", linewidth=1.6)
    if len(budgets) == 1:
        axis.plot(budgets, np.where(finite, values, np.nan), "o", color="#0072B2", markersize=4)
    if log and finite.any() and np.all(values[finite] > 0):
        axis.set_yscale("log")
    if count:
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.set(xlabel="Training samples", ylabel=ylabel)
    if not finite.all():
        axis.text(
            0.03, 0.95, "Non-finite values omitted", transform=axis.transAxes, va="top", fontsize=8
        )


def _make_figure(history, summary, validation):
    # Use a file-rendering canvas without opening windows or changing a user's backend.
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(style):
        figure = Figure(figsize=(9.4, 6.4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        budgets = [row["budget"] for row in history]
        _line(axes[0, 0], budgets, [row["raw_PRESS"] for row in history], "Raw PRESS", log=True)
        axes[0, 0].set_title("a  Cross-validation error", loc="left")
        _line(
            axes[0, 1],
            budgets,
            [row["active_terms"] for row in history],
            "Active terms",
            count=True,
        )
        axes[0, 1].set_title("b  Model size", loc="left")
        if validation is None:
            _line(axes[1, 0], budgets, [row["mean"] for row in history], "Mean")
            _line(axes[1, 1], budgets, [row["std"] for row in history], "Standard deviation")
            axes[1, 0].set_title("c  Surrogate mean", loc="left")
            axes[1, 1].set_title("d  Surrogate standard deviation", loc="left")
        else:
            truth, prediction = validation
            residual = prediction - truth
            # Keep large reports compact while retaining extrema and using all points for metrics.
            shown = np.arange(len(truth))
            if len(truth) > 5000:
                shown = np.unique(
                    np.r_[
                        np.linspace(0, len(truth) - 1, 4996, dtype=int),
                        np.argmin(truth),
                        np.argmax(truth),
                        np.argmin(residual),
                        np.argmax(residual),
                    ]
                )
            low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
            padding = max((high - low) * 0.05, abs(high) * 1e-6, 1e-12)
            limits = low - padding, high + padding
            axis = axes[1, 0]
            axis.plot(limits, limits, "--", color="#666666", linewidth=1.1, label="1:1")
            axis.scatter(
                truth[shown],
                prediction[shown],
                s=10,
                color="#0072B2",
                alpha=0.35,
                linewidths=0,
            )
            axis.set(
                xlim=limits, ylim=limits, xlabel="Reference response", ylabel="Predicted response"
            )
            axis.set_aspect("equal", adjustable="box", anchor="W")
            axis.set_title("c  Independent validation", loc="left")
            text = f"n = {len(truth):,}"
            if len(shown) != len(truth):
                text += f"\n{len(shown):,} shown"
            if summary["validation_relmse"] is not None:
                text += f"\nRelMSE = {summary['validation_relmse']:.2e}"
            else:
                text += f"\nRMSE = {summary['validation_rmse']:.3g}"
            axis.text(0.04, 0.96, text, transform=axis.transAxes, va="top", fontsize=9)
            axis.legend(loc="lower right", frameon=False, fontsize=9)
            axis = axes[1, 1]
            axis.axhline(0, color="#666666", linestyle="--", linewidth=1.1)
            axis.scatter(
                truth[shown],
                residual[shown],
                s=10,
                color="#D55E00",
                alpha=0.35,
                linewidths=0,
            )
            axis.set(xlabel="Reference response", ylabel="Prediction - reference")
            axis.set_title("d  Validation residuals", loc="left")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        for axis in axes.flat:
            # Reserve the same title band for panels with scientific-notation offsets.
            axis.set_title(axis.get_title(loc="left"), loc="left", y=1.08)
            axis.grid(True, color="#D8DDE3", linewidth=0.6, alpha=0.6)
            axis.set_axisbelow(True)
        problem = str(summary["problem"]).replace("_", " ")
        figure.suptitle(f"SCAL | {problem} | N = {summary['training_samples']}", fontsize=13)
        figure.align_ylabels()
    return figure


def save_report(
    directory: str | Path, output: str | Path | None = None, *, validation=None
) -> dict:
    """Write summary.png/.svg, summary.csv and training_history.csv.

    Read the fitted trajectory from ``directory``. Independent validation is an
    optional pair ``(reference, prediction)`` or the saved ``validation.npz``.
    The function performs no training and makes no response-function or simulator calls.
    A report without validation shows the surrogate mean and standard deviation.
    """
    directory = Path(directory).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output is not None else directory
    run = read_json(directory / "run.json")
    snapshots = RunStore(directory).snapshots()
    expected = list(range(run["initial"], run["final"] + 1))
    if [snapshot.budget for snapshot in snapshots] != expected:
        raise ValueError("Saved fits do not cover every budget in run.json")
    validation = _validation(directory, validation)
    history = [
        {
            "budget": snapshot.budget,
            "active_terms": len(snapshot.values["support"]),
            "mean": float(snapshot.values["mean"]),
            "std": float(snapshot.values["std"]),
            "error": float(snapshot.values["error"]),
            "raw_PRESS": float(snapshot.values["raw_PRESS"]),
        }
        for snapshot in snapshots
    ]
    last = history[-1]
    summary = {
        "problem": run["problem"],
        "solver": run["solver"],
        "sampler": run["sampler"],
        "training_samples": run["final"],
        "fitted_models": len(history),
        "active_terms": last["active_terms"],
        "model_mean": last["mean"],
        "model_std": last["std"],
        "raw_PRESS": last["raw_PRESS"],
        "validation_samples": 0,
    }
    if validation is not None:
        truth, prediction = validation
        residual = prediction - truth
        total = float(np.sum((truth - truth.mean()) ** 2))
        summary.update(
            validation_samples=len(truth),
            validation_rmse=float(np.sqrt(np.mean(residual**2))),
            validation_relmse=float(np.sum(residual**2) / total) if total > 0 else None,
            validation_max_absolute_error=float(np.max(np.abs(residual))),
        )
    # Validate the saved trajectory before replacing any derived report files.
    figure = _make_figure(history, summary, validation)
    figure.canvas.draw()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(last), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    import matplotlib as mpl

    with mpl.rc_context({"svg.fonttype": "none"}):
        figure.savefig(output / "summary.png", dpi=300)
        figure.savefig(output / "summary.svg", dpi=300)
    figure.clear()
    return {
        "summary_csv": str(output / "summary.csv"),
        "training_history_csv": str(output / "training_history.csv"),
        "figure_png": str(output / "summary.png"),
        "figure_svg": str(output / "summary.svg"),
        "validation_samples": summary["validation_samples"],
    }
