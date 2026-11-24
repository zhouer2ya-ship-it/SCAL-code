"""Plot Figure 4 from the panel tables listed in the SCAL data catalog."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.colors import LinearSegmentedColormap

from scal.analysis.current import METRICS, MODEL_CONTROLS
from scal.data import Sources

ARMS = ["H", *MODEL_CONTROLS]
BUDGETS = [24, 32, 48, 64, 96, 128]
W, H = 180, 226
COLOURS = {
    "H": "#176B9C",
    "NoHeredityModule": "#2B8581",
    "NoBackwardSearch": "#80648C",
    "FrozenImportance": "#BA974B",
    "NoHistory": "#C6754C",
    "Theta": "#BA974B",
    "GradientOnly": "#C6754C",
}
LINES = {
    "H": "-",
    "NoHeredityModule": "--",
    "NoBackwardSearch": "-.",
    "FrozenImportance": ":",
    "NoHistory": (0, (4, 1, 1, 1)),
}
LABELS = {
    "H": "Full HCBP",
    "NoHeredityModule": "No structural constraints",
    "NoBackwardSearch": "No backward search",
    "FrozenImportance": "Fixed importance",
    "NoHistory": "No history",
}
SHORT = {
    "NoHeredityModule": "No structural constraints",
    "NoBackwardSearch": "No backward",
    "FrozenImportance": "Fixed importance",
    "NoHistory": "No history",
}
INK = "#233641"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6.4,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def ax(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def text(fig, x, y, s, **kw):
    return fig.text(x / W, y / H, s, **kw)


def title(fig, x, y, letter, label):
    text(fig, x, y, letter, fontsize=8.5, fontweight="bold")
    text(fig, x + 4.5, y, label, fontsize=7, fontweight="bold")


def curve(a, data, metric, problem, log):
    data = data[data.problem.eq(problem)]
    for arm in ARMS[1:] + ["H"]:
        g = data[data.arm.eq(arm)].sort_values("budget")
        x = g.budget.to_numpy()
        v = g[[metric + "_q25", metric + "_median", metric + "_q75"]].to_numpy()
        if log:
            v = np.maximum(v, 1e-20)
        a.fill_between(x, v[:, 0], v[:, 2], color=COLOURS[arm], alpha=0.09, lw=0)
        a.plot(x, v[:, 1], color=COLOURS[arm], ls=LINES[arm], lw=1.0 if arm == "H" else 0.85)
    a.set_xlim((24, 128) if problem == "weak_parent" else (80, 200))
    a.set_xticks([24, 48, 80, 104, 128] if problem == "weak_parent" else [80, 110, 140, 170, 200])
    a.set_xlabel("Response evaluations, N", labelpad=2)
    if log:
        a.set_yscale("log")
        a.set_ylim(5e-21, 3)
        a.set_yticks([1, 1e-5, 1e-10, 1e-15, 1e-20])
        a.yaxis.set_minor_locator(NullLocator())
        a.set_ylabel("RelMSE", labelpad=2)
    else:
        a.set_ylim(-0.025, 1.04)
        a.set_yticks([0, 0.5, 1])
        a.set_ylabel("Reference-basis recall", labelpad=2)


def effects(fig, x, data, groups, symlog=False):
    # Ten aligned y slots provide group titles without crowding labels.
    slots = []
    headers = []
    next_y = 8.3
    for group, settings in groups:
        headers.append((next_y + 0.8, group))
        for problem, control in settings:
            slots.append((next_y, problem, control))
            next_y -= 1
        next_y -= 1.25
    minimum = min(y for y, _, _ in slots) - 0.6
    maximum = max(y for y, _ in headers) + 0.3
    for j, metric in enumerate(METRICS):
        a = ax(fig, x + 27 + j * 19.1, 11, 16.4, 31)
        for y, problem, control in slots:
            r = data[
                data.problem.eq(problem) & data.control.eq(control) & data.metric.eq(metric)
            ].iloc[0]
            col = COLOURS[control]
            a.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.1, solid_capstyle="round")
            a.scatter(r["median"], y, s=12, c=col, edgecolors="white", linewidths=0.25, zorder=3)
        a.axvline(0, color="#929CA1", ls=":", lw=0.6, zorder=0)
        a.set_ylim(minimum, maximum)
        a.set_yticks([v[0] for v in slots])
        labels = [SHORT.get(c, "Theta" if c == "Theta" else "Direct gradient") for _, _, c in slots]
        a.set_yticklabels(labels if j == 0 else [], fontsize=5.9)
        a.tick_params(axis="y", length=0, pad=2)
        a.spines["left"].set_visible(False)
        if symlog:
            a.set_xscale("symlog", linthresh=0.3, linscale=0.8)
            a.set_xlim(-0.09, 12)
            a.set_xticks([0, 1, 10])
            a.set_xticklabels(["0", "1", "10"])
        else:
            a.set_xlim(-0.3, 2.1)
            a.set_xticks([0, 1, 2])
        a.xaxis.set_minor_locator(NullLocator())
        a.tick_params(axis="x", labelsize=5.8)
        a.set_title(["Mean", "SD", "RelMSE"][j], fontsize=6.4, pad=3)
        if j == 0:
            for y, label in headers:
                a.text(
                    -1.63,
                    y,
                    label,
                    transform=a.get_yaxis_transform(),
                    fontsize=6,
                    fontweight="bold",
                    va="center",
                    clip_on=False,
                )


def plot(data_root, output):
    sources = Sources(data_root=data_root)

    def frame(name):
        return sources.frame(name + ".csv")

    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    curves = frame("closed_loop_curves")
    support = frame("finite_support_curves")
    title(fig, 4, 219, "a", "Weak-parent convergence")
    title(fig, 94, 219, "b", "Rackwitz convergence")
    curve(ax(fig, 16, 179, 68, 33), curves, "relmse", "weak_parent", True)
    curve(ax(fig, 106, 179, 68, 33), curves, "relmse", "rackwitz", True)
    fig.legend(
        handles=[Line2D([], [], c=COLOURS[a], ls=LINES[a], label=LABELS[a]) for a in ARMS],
        loc="center",
        bbox_to_anchor=(0.5, 168 / H),
        ncol=5,
        columnspacing=1.25,
        handlelength=2.0,
        handletextpad=0.5,
    )
    title(fig, 4, 158, "c", "Weak-parent support coverage")
    title(fig, 94, 158, "d", "Rackwitz support coverage")
    curve(ax(fig, 16, 125, 68, 26), support, "support_recall", "weak_parent", False)
    curve(ax(fig, 106, 125, 68, 26), support, "support_recall", "rackwitz", False)
    title(fig, 4, 113, "e", "Retention of weak terms and strong interactions")
    title(fig, 94, 113, "f", "Exact support on fixed observations")
    freq = frame("weak_parent_term_frequency")
    mat = np.zeros((10, 30))
    for ai, arm in enumerate(ARMS):
        for ni, n in enumerate(BUDGETS):
            g = freq[freq.arm.eq(arm) & freq.budget.eq(n)].sort_values("display_order")
            assert len(g) == 10
            mat[:, ai * 6 + ni] = g.retention_frequency.to_numpy()
    e = ax(fig, 20, 66, 65, 36)
    cmap = LinearSegmentedColormap.from_list("retained", ["#F4F6F7", "#97BBD0", "#176B9C"])
    im = e.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    e.set_yticks(
        range(10),
        [r"$x_{" + str(i) + "}$" for i in range(1, 9)] + [r"$x_1x_2$", r"$x_3x_4$"],
        fontsize=6.1,
    )
    e.set_xticks(range(30), [str(n) for _ in ARMS for n in BUDGETS], rotation=90, fontsize=5.6)
    e.tick_params(axis="both", length=0, pad=2)
    e.set_xticks(np.arange(-0.5, 30, 1), minor=True)
    e.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    e.grid(which="minor", color="white", linewidth=0.35)
    e.tick_params(which="minor", length=0)
    for boundary in (5.5, 11.5, 17.5, 23.5):
        e.axvline(boundary, color="white", lw=1.8)
    for boundary in (3.5, 7.5):
        e.axhline(boundary, color="white", lw=1.7)
    for ai, label in enumerate(
        ["Full", "No\nstructure", "No\nbackward", "Fixed\nimportance", "No\nhistory"]
    ):
        e.text(ai * 6 + 2.5, -0.9, label, ha="center", va="bottom", fontsize=5.9, linespacing=0.95)
    for y, label in [(1.5, "Parents"), (5.5, "Other linear"), (8.5, "Interactions")]:
        e.text(-5.9, y, label, ha="center", va="center", rotation=90, fontsize=5.7, clip_on=False)
    for y, coeff in [(1.5, "0.02"), (5.5, "0.02"), (8, "2"), (9, "1.5")]:
        e.text(30.3, y, coeff, fontsize=5.8, ha="left", va="center", clip_on=False)
    e.text(30.6, -0.9, r"$\beta$", fontsize=6.4, ha="center", va="bottom", clip_on=False)
    cb = fig.colorbar(
        im, cax=ax(fig, 20, 53.5, 36, 1.8), orientation="horizontal", ticks=[0, 0.5, 1]
    )
    cb.ax.tick_params(labelsize=5.7, length=1.5, pad=1)
    cb.outline.set_visible(False)
    text(fig, 59, 54, "Retained fraction", fontsize=5.8, va="center")
    fixed = frame("fixed_design_recovery_summary")
    settings = [
        "baseline",
        "H-C01:raw_union",
        "H-C04:pruning_off",
        "H-C05:unrestricted_deletion",
        "H-C06:fixed_initial_importance",
        "H-C10:forward_parent_admissible",
    ]
    labels = [
        "Full HCBP",
        "Raw union",
        "No pruning",
        "Unrestricted deletion",
        "Fixed importance",
        "Forward admissible",
    ]
    fa = ax(fig, 139, 69, 34, 32)
    for j, (problem, marker, label) in enumerate(
        [("weak_parent", "o", "Weak parent, N=128"), ("rackwitz", "s", "Rackwitz, N=240")]
    ):
        g = fixed[fixed.problem.eq(problem)].set_index("setting").loc[settings]
        fa.scatter(
            g.exact_support_count.to_numpy(),
            np.arange(6) + (j - 0.5) * 0.25,
            s=15,
            marker=marker,
            facecolors="#176B9C" if j == 0 else "white",
            edgecolors="#233641",
            linewidths=0.6,
            label=label,
            zorder=3,
        )
    fa.set_yticks(range(6), labels, fontsize=6.0)
    fa.invert_yaxis()
    fa.set_ylim(5.55, -0.55)
    fa.set_xlim(-1, 21)
    fa.set_xticks([0, 10, 20], ["0/20", "10/20", "20/20"])
    fa.tick_params(axis="y", length=0, pad=3)
    fa.set_xlabel("Exact recovery", labelpad=2)
    fa.spines["left"].set_visible(False)
    fa.grid(axis="x", color="#DCE1E4", lw=0.5, zorder=0)
    fa.legend(
        loc="upper left",
        bbox_to_anchor=(-1.25, 1.28),
        fontsize=5.8,
        handletextpad=0.25,
        labelspacing=0.3,
    )
    text(fig, 134, 57.5, "Six settings; no preceding model", ha="center", fontsize=5.8)
    title(fig, 4, 46, "g", "Model-component contributions")
    title(fig, 94, 46, "h", "AHSC-component contributions")
    model = frame("figure4_model_effects")
    sampling = frame("figure4_sampling_effects")
    groups = [("Weak parent · n=20", [("weak_parent", a) for a in ARMS[1:]])]
    groups.append(("Rackwitz · n=20", [("rackwitz", a) for a in ARMS[1:]]))
    effects(fig, 4, model, groups, True)
    effects(
        fig,
        94,
        sampling,
        [
            ("Ishigami · n=100", [("ishigami", "Theta"), ("ishigami", "GradientOnly")]),
            ("Borehole · n=20", [("borehole", "Theta"), ("borehole", "GradientOnly")]),
            ("Rackwitz · n=20", [("rackwitz", "Theta")]),
        ],
    )
    text(
        fig,
        90,
        2.8,
        "Paired log-error-area difference: control − full (median and 95% bootstrap CI)",
        ha="center",
        fontsize=6.2,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    hidden = set()
    for a in fig.axes:
        for axis, lim in [(a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())]:
            lo, hi = sorted(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    hidden.update([id(tick.label1), id(tick.label2)])
    for obj in fig.findobj(matplotlib.text.Text):
        if id(obj) in hidden or not obj.get_visible() or not obj.get_text():
            continue
        b = obj.get_window_extent(renderer)
        if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.width + 2 or b.y1 > fig.bbox.height + 2:
            outside.append(obj.get_text())

    if outside:
        raise ValueError(f"Figure text outside the page: {outside}")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png", "tiff"):
        settings = {"dpi": 600} if suffix in ("png", "tiff") else {}
        if suffix == "tiff":
            settings["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(output / f"Main_Figure_04.{suffix}", **settings)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data_root, args.output)


if __name__ == "__main__":
    main()
