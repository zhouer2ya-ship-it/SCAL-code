"""Scalar Theta criterion from Novák et al. (2021), CMAME 386:114105, Eqs. (24)–(26).

Distances and densities are evaluated in the polynomial germ xi. The score uses
coefficients from the fitted polynomial chaos expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_hermitenorm, eval_legendre, gammaln


def _matrix(value, name):
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or min(out.shape) == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return out


def _vector(value, n, name, *, zero_density=False, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = np.full(n, default)
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.full(n, float(out))
    if out.shape != (n,):
        raise ValueError(f"{name} must be scalar or have shape ({n},)")
    valid = np.isfinite(out) | (np.isneginf(out) if zero_density else False)
    if not np.all(valid):
        raise ValueError(f"{name} contains invalid values")
    return out


def standardize_inputs(physical, poly_matrix):
    """Map independent uniform/normal/lognormal inputs to Legendre/Hermite germs.

    Rows use the existing project convention: (0, lower, upper),
    (1, mean, std), or (2, mean_log, std_log). A lognormal input is mapped
    through its underlying Gaussian; its germ density is standard normal.
    """
    x = _matrix(physical, "physical")
    poly = _matrix(poly_matrix, "poly_matrix")
    if poly.shape != (x.shape[1], 3):
        raise ValueError("poly_matrix must have shape (input_dimension, 3)")
    xi = np.empty_like(x)
    families = []
    for j, (kind, a, b) in enumerate(poly):
        if kind == 0:
            if b <= a:
                raise ValueError("Uniform upper bound must exceed lower bound")
            xi[:, j] = 2 * ((x[:, j] - a) / (b - a)) - 1
            families.append("legendre")
        elif kind in (1, 2):
            if b <= 0:
                raise ValueError("Normal/lognormal scale must be positive")
            if kind == 2 and np.any(x[:, j] <= 0):
                raise ValueError("Lognormal samples must be positive")
            xi[:, j] = ((np.log(x[:, j]) if kind == 2 else x[:, j]) - a) / b
            families.append("hermite")
        else:
            raise ValueError(f"Unsupported input distribution code: {kind}")
    if not np.all(np.isfinite(xi)):
        raise FloatingPointError("Nonfinite germ coordinates")
    return xi, tuple(families)


def germ_logpdf(germ, families):
    """Joint log p_xi in standard polynomial coordinates, including zero support."""
    xi = _matrix(germ, "germ")
    if len(families) != xi.shape[1]:
        raise ValueError("One polynomial family is required for each input")
    result = np.zeros(len(xi))
    for j, family in enumerate(families):
        if family == "legendre":
            result -= math.log(2)
            result[np.abs(xi[:, j]) > 1] = -np.inf
        elif family == "hermite":
            with np.errstate(over="ignore"):
                result += -0.5 * xi[:, j] ** 2 - 0.5 * math.log(2 * math.pi)
        else:
            raise ValueError(f"Unsupported polynomial family: {family}")
    return result


def orthonormal_design(germ, multi_indices, families):
    """Tensor Legendre/Hermite basis with probability-measure normalization."""
    xi = _matrix(germ, "germ")
    indices = np.asarray(multi_indices)
    if (
        indices.ndim != 2
        or indices.shape[1] != xi.shape[1]
        or len(indices) == 0
        or len(families) != xi.shape[1]
        or not np.all(np.isfinite(indices))
        or np.any(indices < 0)
        or np.any(indices != np.floor(indices))
    ):
        raise ValueError("Invalid multi-indices or families")
    indices = indices.astype(int)
    result = np.ones((len(xi), len(indices)))
    for j, family in enumerate(families):
        for degree in np.unique(indices[:, j]):
            if family == "legendre":
                values = eval_legendre(int(degree), xi[:, j]) * np.sqrt(2 * degree + 1)
            elif family == "hermite":
                values = eval_hermitenorm(int(degree), xi[:, j]) * np.exp(
                    -0.5 * gammaln(degree + 1)
                )
            else:
                raise ValueError(f"Unsupported polynomial family: {family}")
            result[:, indices[:, j] == degree] *= values[:, None]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Nonfinite polynomial basis")
    return result


def _local_logroot(design, coefficients, constant_index, logpdf, weights):
    # Sum the nonconstant terms directly: no loss from subtracting a large mean.
    nonconstant = np.arange(len(coefficients)) != constant_index
    centered = design[:, nonconstant] @ coefficients[nonconstant]
    if not np.all(np.isfinite(centered)):
        raise FloatingPointError("Nonfinite centered PCE response")
    with np.errstate(divide="ignore"):
        return np.log(np.abs(centered)) + np.log(weights) + 0.5 * logpdf


def _exp(value):
    # Raw output may under/overflow for extreme dimensions; selection uses logs.
    with np.errstate(under="ignore", over="ignore"):
        return np.exp(value)


@dataclass(frozen=True)
class ThetaScores:
    theta: np.ndarray
    logtheta: np.ndarray
    candidate_variance_density: np.ndarray
    existing_variance_density: np.ndarray
    geometric_mean_density: np.ndarray
    distance_power: np.ndarray
    distance: np.ndarray
    nearest: np.ndarray
    dimension: int


def _nearest_neighbours(train, candidate, block_size):
    """Exact germ-space KD-tree query, with first-training-row distance ties.

    Query the second neighbour only to detect ambiguous distances. In those
    rare cases, collect the small tied neighbourhood and compare its Euclidean
    distances in training-row order. This also resolves roundoff near a tie;
    no tolerance is applied to the final distance comparison itself.
    """
    tree = cKDTree(train)
    nearest = np.empty(len(candidate), dtype=np.int64)
    distances = np.empty(len(candidate))
    roundoff = 8 * np.finfo(float).eps * train.shape[1]
    for start in range(0, len(candidate), block_size):
        stop = min(start + block_size, len(candidate))
        points = candidate[start:stop]
        lengths, indices = tree.query(points, k=2, eps=0, p=2, workers=1)
        distances[start:stop] = lengths[:, 0]
        nearest[start:stop] = indices[:, 0]
        if len(train) == 1:
            continue
        ambiguous = np.flatnonzero(lengths[:, 1] <= lengths[:, 0] * (1 + roundoff))
        for row in ambiguous:
            radius = np.nextafter(lengths[row, 1] * (1 + roundoff), np.inf)
            neighbours = np.sort(tree.query_ball_point(points[row], radius, p=2, eps=0))
            delta = train[neighbours] - points[row]
            # Ordered accumulation matches the former Euclidean cdist reduction.
            squared = np.add.accumulate(delta * delta, axis=1)[:, -1]
            exact = np.sqrt(squared)
            winner = int(np.argmin(exact))
            nearest[start + row] = neighbours[winner]
            distances[start + row] = exact[winner]
    return nearest, distances


def theta_scores(
    existing_germ,
    candidate_germ,
    existing_design,
    candidate_design,
    coefficients,
    *,
    existing_logpdf,
    candidate_logpdf,
    constant_index=0,
    existing_weights=None,
    candidate_weights=None,
    distance_block_size=2048,
):
    """Evaluate Eqs. (24)-(26) without fitting or mutating the fitted PCE.

    Designs and coefficients must use the same basis convention. The original
    orthonormal convention and a consistently rescaled basis give identical
    centered predictions. The constant column is identified explicitly.
    Weights are row multipliers w(xi), not their squares. Input log densities
    are mandatory so that Gaussian/other nonconstant densities cannot be lost.
    All logarithms are natural. Zero Theta is represented by logtheta=-inf.
    Nearest neighbours use exact cKDTree queries (eps=0, workers=1); ties use
    the first training row. distance_block_size controls query batching only.
    """
    train = _matrix(existing_germ, "existing_germ")
    candidate = _matrix(candidate_germ, "candidate_germ")
    a = _matrix(existing_design, "existing_design")
    c = _matrix(candidate_design, "candidate_design")
    coef = np.asarray(coefficients, dtype=float)
    if coef.ndim != 1 or coef.size == 0 or not np.all(np.isfinite(coef)):
        raise ValueError("coefficients must be a finite nonempty vector")
    if (
        train.shape[1] != candidate.shape[1]
        or a.shape != (len(train), len(coef))
        or c.shape != (len(candidate), len(coef))
    ):
        raise ValueError("Coordinate, design, and coefficient shapes disagree")
    if not isinstance(constant_index, (int, np.integer)) or not 0 <= constant_index < len(coef):
        raise ValueError("constant_index must identify the constant basis column")
    if not (np.all(a[:, constant_index] == 1) and np.all(c[:, constant_index] == 1)):
        raise ValueError("The constant basis column must equal one")
    if not isinstance(distance_block_size, (int, np.integer)) or distance_block_size < 1:
        raise ValueError("distance_block_size must be a positive integer")
    lp_a = _vector(existing_logpdf, len(train), "existing_logpdf", zero_density=True)
    lp_c = _vector(candidate_logpdf, len(candidate), "candidate_logpdf", zero_density=True)
    wa = _vector(existing_weights, len(train), "existing_weights", default=1)
    wc = _vector(candidate_weights, len(candidate), "candidate_weights", default=1)
    if np.any(wa < 0) or np.any(wc < 0):
        raise ValueError("Sampling weights must be nonnegative")
    nearest, distances = _nearest_neighbours(train, candidate, distance_block_size)
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("Nonfinite nearest-neighbour distances")
    logroot_a = _local_logroot(a, coef, constant_index, lp_a, wa)
    logroot_c = _local_logroot(c, coef, constant_index, lp_c, wc)
    with np.errstate(divide="ignore"):
        logvolume = train.shape[1] * np.log(distances)
    logmean = logroot_c + logroot_a[nearest]
    logtheta = logmean + logvolume
    return ThetaScores(
        theta=_exp(logtheta),
        logtheta=logtheta,
        candidate_variance_density=_exp(2 * logroot_c),
        existing_variance_density=_exp(2 * logroot_a),
        geometric_mean_density=_exp(logmean),
        distance_power=_exp(logvolume),
        distance=distances,
        nearest=nearest,
        dimension=train.shape[1],
    )


def select_candidate(scores, *, available=None, candidate_ids=None):
    """Select the highest Theta score among unused points with positive distance.

    For exact ties, including an all-zero score vector, use the smallest
    candidate ID. Candidate IDs default to row indices in the pool.
    """
    n = len(scores.logtheta)
    mask = np.ones(n, dtype=bool) if available is None else np.asarray(available)
    if mask.shape != (n,) or mask.dtype != bool:
        raise ValueError("available must be a Boolean vector")
    ids = np.arange(n) if candidate_ids is None else np.asarray(candidate_ids)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != n:
        raise ValueError("candidate_ids must be unique integer IDs")
    valid = np.flatnonzero(mask & (scores.distance > 0))
    if not len(valid):
        raise RuntimeError("NO_UNUSED_DISTINCT_CANDIDATE")
    maximum = np.max(scores.logtheta[valid])
    ties = valid[scores.logtheta[valid] == maximum]
    return int(ties[np.argmin(ids[ties])])


def theta_from_physical(
    existing, candidates, existing_design, candidate_design, coefficients, poly_matrix, **kwargs
):
    """Evaluate Theta for physical inputs with independent marginals."""
    a, families = standardize_inputs(existing, poly_matrix)
    c, _ = standardize_inputs(candidates, poly_matrix)
    return theta_scores(
        a,
        c,
        existing_design,
        candidate_design,
        coefficients,
        existing_logpdf=germ_logpdf(a, families),
        candidate_logpdf=germ_logpdf(c, families),
        **kwargs,
    )


class ThetaSampler:
    """Scalar-PCE Theta sampler from Novák et al. (2021), Eqs. (24)–(26).

    Inputs use the fitted model's physical coordinates and ``poly_matrix``.
    Select one point, evaluate it, append the observation and refit the PCE,
    then call ``set_model`` again. There is no AHSC separation radius, score
    normalization, or covariance calculation. Optional weights multiply basis
    rows as w(xi), so they are squared inside the local variance density.
    """

    def __init__(self, poly_matrix=None, verbose=True):
        self.poly_matrix = (
            None if poly_matrix is None else np.array(poly_matrix, dtype=float, copy=True)
        )
        self.verbose = bool(verbose)
        self.model = self.X_train = self.y_train = self.candidate_pool = None
        self.candidate_ids = self.sample_weights = self.candidate_weights = None
        self.last_scores = None
        self.last_selection_report = []
        self.last_selection_reason = None
        self._pending_point = self._pending_weight = None

    def initialize(
        self,
        X_train,
        y_train,
        candidate_pool,
        candidate_ids=None,
        *,
        sample_weights=None,
        candidate_weights=None,
    ):
        train = _matrix(X_train, "X_train")
        pool = np.asarray(candidate_pool, dtype=float)
        if pool.ndim != 2 or pool.shape[1] != train.shape[1] or not np.all(np.isfinite(pool)):
            raise ValueError("candidate_pool must be a finite matrix aligned with X_train")
        ids = (
            np.arange(len(pool), dtype=np.int64)
            if candidate_ids is None
            else np.asarray(candidate_ids)
        )
        if (
            ids.shape != (len(pool),)
            or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Candidate IDs must be unique integers aligned with the pool")
        target = None if y_train is None else np.asarray(y_train, dtype=float)
        if target is not None:
            if target.shape == (len(train), 1):
                target = target[:, 0]
            if target.shape != (len(train),) or not np.all(np.isfinite(target)):
                raise ValueError("ThetaSampler requires one scalar response per training point")
        wa = _vector(sample_weights, len(train), "sample_weights", default=1)
        wc = _vector(candidate_weights, len(pool), "candidate_weights", default=1)
        if np.any(wa < 0) or np.any(wc < 0):
            raise ValueError("Sampling weights must be nonnegative")
        self.X_train, self.candidate_pool, self.candidate_ids = (
            train.copy(),
            pool.copy(),
            ids.copy(),
        )
        self.y_train = None if target is None else target.copy()
        self.sample_weights, self.candidate_weights = wa.copy(), wc.copy()
        self.model = self.last_scores = self._pending_point = self._pending_weight = None
        self.last_selection_report, self.last_selection_reason = [], None

    def set_model(self, model):
        if self.X_train is None:
            raise RuntimeError("Initialize the sampler before setting its model")
        if not getattr(model, "is_fitted_", False):
            raise RuntimeError("ThetaSampler requires a fitted scalar PCE")
        mapping = getattr(model, "mapping_", None)
        coef = np.asarray(getattr(model, "coef_", None), dtype=float)
        if (
            not isinstance(mapping, dict)
            or not mapping
            or coef.shape != (len(mapping),)
            or not np.all(np.isfinite(coef))
        ):
            raise ValueError("The fitted PCE needs aligned mapping_ and scalar coef_")
        model_poly = np.asarray(model.poly_matrix, dtype=float)
        if self.poly_matrix is not None and not np.array_equal(self.poly_matrix, model_poly):
            raise ValueError("Sampler and model must use the same poly_matrix")
        standardize_inputs(self.X_train, model_poly)  # validate the germ definition
        fitted_x = getattr(model, "X", None)
        if self._pending_point is not None:
            expected = np.vstack([self.X_train, self._pending_point])
            if fitted_x is None or not np.array_equal(fitted_x, expected):
                raise RuntimeError(
                    "Append the selected observation and refit before setting the model; "
                    "reinitialize explicitly if the training design is reordered"
                )
            self.X_train = expected
            self.y_train = (
                None if getattr(model, "Y", None) is None else np.asarray(model.Y).ravel().copy()
            )
            self.sample_weights = np.r_[self.sample_weights, self._pending_weight]
        elif fitted_x is not None and not np.array_equal(fitted_x, self.X_train):
            raise ValueError("Initialize with the observations used to fit the supplied PCE")
        self.poly_matrix, self.model = model_poly.copy(), model
        self._pending_point = self._pending_weight = None
        self.last_scores = None

    def score_candidates(self):
        if self.model is None:
            raise RuntimeError("Set a fitted PCE before scoring candidates")
        if self._pending_point is not None:
            raise RuntimeError(
                "Evaluate the selected point, refit, and call set_model before scoring again"
            )
        if not len(self.candidate_pool):
            raise RuntimeError("NO_UNUSED_DISTINCT_CANDIDATE")
        from ..solving.utils import construct_matrix_optimized, normalization_all

        keys = list(self.model.mapping_)
        constants = [
            i for i, key in enumerate(keys) if str(self.model.mapping_[key]).strip() in {"1", "1.0"}
        ]
        if len(constants) != 1:
            raise ValueError("The fitted PCE must identify exactly one constant basis term")
        coef = np.asarray(self.model.coef_, dtype=float)
        active = np.union1d(constants, np.flatnonzero(coef))
        mapping = {keys[i]: self.model.mapping_[keys[i]] for i in active}
        a = construct_matrix_optimized(mapping, normalization_all(self.X_train, self.poly_matrix).T)
        c = construct_matrix_optimized(
            mapping, normalization_all(self.candidate_pool, self.poly_matrix).T
        )
        self.last_scores = theta_from_physical(
            self.X_train,
            self.candidate_pool,
            a,
            c,
            coef[active],
            self.poly_matrix,
            constant_index=int(np.flatnonzero(active == constants[0])[0]),
            existing_weights=self.sample_weights,
            candidate_weights=self.candidate_weights,
        )
        return self.last_scores

    def _compute_scores(self):
        """Return raw Eq. (24) scores; internal selection uses log scores."""
        return self.score_candidates().theta

    def select_samples(self, k=1):
        if not isinstance(k, (int, np.integer)) or k not in (0, 1):
            raise ValueError(
                "Original sequential Theta supports k=1; refit after each new observation"
            )
        if self.candidate_pool is None:
            raise RuntimeError("Initialize the sampler before selecting")
        if k == 0 or not len(self.candidate_pool):
            self.last_selection_report = []
            self.last_selection_reason = "NO_FEASIBLE_CANDIDATE" if k else None
            return np.empty((0, self.X_train.shape[1]))
        scores = self.score_candidates()
        if not np.any(scores.distance > 0):
            self.last_selection_report = []
            self.last_selection_reason = "NO_FEASIBLE_CANDIDATE"
            return np.empty((0, self.X_train.shape[1]))
        winner = select_candidate(scores, candidate_ids=self.candidate_ids)
        valid = scores.distance > 0
        self.last_selection_report = [
            dict(
                candidate_id=int(self.candidate_ids[winner]),
                selected_local_index=winner,
                criterion="Theta_original_Eq24_26",
                theta=float(scores.theta[winner]),
                logtheta=float(scores.logtheta[winner])
                if np.isfinite(scores.logtheta[winner])
                else None,
                dimension=scores.dimension,
                minimum_distance=float(scores.distance[winner]),
                nearest_existing_index=int(scores.nearest[winner]),
                all_zero_theta=bool(np.all(np.isneginf(scores.logtheta[valid]))),
                exact_max_ties=int(
                    np.count_nonzero(scores.logtheta[valid] == scores.logtheta[winner])
                ),
                feasible_count=int(valid.sum()),
            )
        ]
        selected = self.candidate_pool[[winner]].copy()
        self._pending_point, self._pending_weight = (
            selected.copy(),
            float(self.candidate_weights[winner]),
        )
        self.candidate_pool = np.delete(self.candidate_pool, winner, axis=0)
        self.candidate_ids = np.delete(self.candidate_ids, winner)
        self.candidate_weights = np.delete(self.candidate_weights, winner)
        self.last_selection_reason = None
        return selected
