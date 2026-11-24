"""Paired effect estimates and inference for benchmark comparisons."""

import hashlib
import json
import numpy as np
import pandas as pd

STAGE = "theta-primary-mechanism-20261021-v1"
METRICS = ["std_relative_error", "relmse"]
FLOORS = [1e-20, 1e-16]


def seed_channel(family, contrast, metric, floor, purpose):
    key = "\x1f".join(map(str, [STAGE, family, contrast, metric, floor, purpose]))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def paired(values, family, contrast, metric, floor, planned, inference=False):
    x = np.asarray(values, float)
    assert len(x) == planned and np.isfinite(x).all()
    seed = seed_channel(family, contrast, metric, floor, "bootstrap")
    gen = np.random.default_rng(seed)
    boot = np.median(x[gen.integers(0, len(x), (5000, len(x)))], axis=1)
    lo, hi, upper = np.quantile(boot, [0.025, 0.975, 0.95])
    out = dict(
        n_total=planned,
        n_valid=len(x),
        planned_n=planned,
        median=np.median(x),
        q1=np.quantile(x, 0.25),
        q3=np.quantile(x, 0.75),
        ci_low=lo,
        ci_high=hi,
        one_sided_95_upper=upper,
        negative_count=int((x < 0).sum()),
        positive_count=int((x > 0).sum()),
        exact_zero_count=int((x == 0).sum()),
        bootstrap_draws=5000,
        bootstrap_random_seed=str(seed),
        p_raw=np.nan,
        p_adjusted=np.nan,
        sign_flip_draws=100000 if inference else 0,
        analysis_status="dated_20261021_amendment",
    )
    if inference:
        seed = seed_channel(family, contrast, metric, floor, "sign_flip")
        gen = np.random.default_rng(seed)
        observed = abs(np.median(x))
        extreme = 0
        for j in range(10):
            signs = gen.integers(0, 2, (10000, len(x)), dtype=np.int8) * 2 - 1
            extreme += int((abs(np.median(signs * x, axis=1)) >= observed).sum())
        out.update(
            p_raw=(1 + extreme) / 100001,
            sign_flip_extreme_count=extreme,
            sign_flip_random_seed=str(seed),
        )
    return out


def holm(p):
    p = np.asarray(p, float)
    ix = np.argsort(p, kind="stable")
    out = np.empty_like(p)
    out[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * (len(p) - np.arange(len(p)))))
    return out


def area(f, metric, floor, nseeds):
    result = {}
    for seed, g in f.groupby("seed"):
        g = g.sort_values("budget")
        n = g.budget.to_numpy()
        assert np.array_equal(n, np.arange(n[0], n[-1] + 1))
        result[int(seed)] = float(
            np.trapezoid(np.log10(np.maximum(g[metric].to_numpy(), floor)), n) / (n[-1] - n[0])
        )
    s = pd.Series(result).sort_index()
    assert s.index.tolist() == list(range(nseeds))
    return s


def comparisons(curves, definitions):
    rows = []
    seedrows = []
    cache = {}
    for d in definitions:
        family, cid, weights, n = d["family"], d["contrast"], d["weights"], d["n"]
        for metric in METRICS:
            for floor in FLOORS:
                value = pd.Series(0.0, index=np.arange(n))
                for task, w in weights.items():
                    key = task, metric, floor, n
                    if key not in cache:
                        cache[key] = area(curves[curves.task.eq(task)], metric, floor, n)
                    value += w * cache[key]
                infer = metric == METRICS[0] and floor == FLOORS[0] and not d.get("external", False)
                st = paired(value, family, cid, metric, floor, n, infer)
                rows.append(
                    dict(
                        contrast_id=cid,
                        statistical_family=family,
                        metric=metric,
                        floor=floor,
                        primary=infer,
                        confirmatory=False,
                        quantity="nauc_full",
                        weights=json.dumps(weights, sort_keys=True),
                        interpretation=d["label"],
                        external=d.get("external", False),
                        **st,
                    )
                )
                seedrows.extend(
                    dict(
                        contrast_id=cid,
                        statistical_family=family,
                        metric=metric,
                        floor=floor,
                        seed=int(s),
                        difference=v,
                    )
                    for s, v in value.items()
                )
    effects = pd.DataFrame(rows)
    for family, g in effects[effects.primary].groupby("statistical_family"):
        effects.loc[g.index, "p_adjusted"] = holm(g.p_raw)
    # Reuse the primary summary when both floors give identical paired differences.
    for (family, cid, metric), g in effects.groupby(
        ["statistical_family", "contrast_id", "metric"]
    ):
        main = g[np.isclose(g.floor, 1e-20, rtol=0, atol=0)]
        sensitivity = g[np.isclose(g.floor, 1e-16, rtol=0, atol=0)]
        v = pd.DataFrame(seedrows)
        v = v[v.statistical_family.eq(family) & v.contrast_id.eq(cid) & v.metric.eq(metric)]
        x = v[np.isclose(v.floor, 1e-20, rtol=0, atol=0)].sort_values("seed").difference.to_numpy()
        y = v[np.isclose(v.floor, 1e-16, rtol=0, atol=0)].sort_values("seed").difference.to_numpy()
        if np.array_equal(x, y):
            cols = ["median", "ci_low", "ci_high", "one_sided_95_upper"]
            effects.loc[sensitivity.index, cols] = main[cols].to_numpy()
    return effects, pd.DataFrame(seedrows)
