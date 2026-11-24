"""Greedy S-value sampling on a fitted scalar PCE's selected basis.

Fajraoui, Marelli and Sudret (2017), Eq. (31), DOI 10.1137/16M1103488.
This module implements acquisition only; it neither fits a surrogate nor
selects a polynomial degree or a regression support. It is independent of
hybrid LAR and of the full FMS_Sopt workflow used in archived comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from .base import CandidateFeasibility
from .theta import standardize_inputs
from ..coordinates import TargetGeometry
from ..solving.utils import construct_matrix_optimized, normalization_all

SOPT_VERSION = "S-optimal-2026.10.28-selected-support-single-row"
RANK_RTOL = 1e-12


def s_optimal_scores(existing_design, candidate_design):
    """Return exact one-row log S-values and SVD diagnostics.

    For B = [A; c], log S(B) equals
    (log det(B.T B) - sum_j log ||B[:, j]||**2) / (2 K).
    The determinant lemma avoids one factorization per candidate. There is
    no residual scale, covariance estimate, ridge, or pseudoinverse fallback.
    Designs must use the same columns and column scaling. Consistent column
    rescaling leaves the mathematical criterion invariant.
    """
    A, C = np.asarray(existing_design, dtype=float), np.asarray(candidate_design, dtype=float)
    if (
        A.ndim != 2
        or min(A.shape) == 0
        or C.ndim != 2
        or C.shape[1] != A.shape[1]
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(C))
    ):
        raise ValueError(
            "S-optimal designs must be finite aligned matrices with a nonempty training design"
        )
    n, k = A.shape
    _, singular, vt = svd(A, full_matrices=False, check_finite=True)
    threshold = RANK_RTOL * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    if n < k or rank < k:
        raise np.linalg.LinAlgError(
            f"S-optimal requires a full-column-rank selected support: N={n}, K={k}, rank={rank}"
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        squared_norms = np.sum(A * A, axis=0)
        leverage = np.sum(((C @ vt.T) / singular) ** 2, axis=1)
        logdet = 2 * np.log(singular).sum()
        values = (
            logdet + np.log1p(leverage) - np.log(squared_norms[None, :] + C * C).sum(axis=1)
        ) / (2 * k)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite S-optimal log scores")
    return values, dict(
        N=n,
        K=k,
        rank=rank,
        rank_rtol=RANK_RTOL,
        rank_threshold=float(threshold),
        branch="exact_greedy_log_S",
        regularized=False,
    )


@dataclass(frozen=True)
class SOptimalScores:
    log_s: np.ndarray
    s_value: np.ndarray
    valid: np.ndarray
    distance: np.ndarray
    duplicate_tolerance: float
    diagnostics: dict


def select_candidate(scores, candidate_ids):
    """Maximize log S over unused distinct candidates; exact ties use min ID."""
    ids = np.asarray(candidate_ids)
    if (
        ids.shape != scores.log_s.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Candidate IDs must be unique integers aligned with the scores")
    active = np.flatnonzero(scores.valid)
    if not len(active):
        raise RuntimeError("NO_FEASIBLE_CANDIDATE")
    return int(active[np.lexsort((ids[active], -scores.log_s[active]))[0]])


class SOptimalSampler:
    """Select one physical input using the current fitted support's S-value.

    Initialize with the fitted observations and an immutable-ID candidate
    pool, then set a fitted scalar PCE. Explicit support_indices_ are kept,
    including zero-coefficient structural columns. Without explicit support,
    use the intercept and nonzero coefficient columns. The intercept is always
    included. A full-column-rank selected design is required.

    After selection, evaluate the response, append it and refit before setting
    a model and selecting again. No cross-pool candidate-matrix cache is kept.
    Geometry serves only to reject duplicates at floating-point tolerance;
    no separation radius or geometric score modifies S-value maximization.
    """

    def __init__(self, poly_matrix=None, verbose=True):
        self.poly_matrix = (
            None if poly_matrix is None else np.array(poly_matrix, dtype=float, copy=True)
        )
        self.verbose = bool(verbose)
        self.model = self.X_train = self.y_train = self.candidate_pool = self.candidate_ids = None
        self.support_indices_ = self.mapping_ = self.column_scale_ = None
        self.last_scores = self._pending_point = None
        self.last_selection_report = []
        self.last_selection_reason = None

    def initialize(self, X_train, y_train, candidate_pool, candidate_ids=None):
        train, pool = np.asarray(X_train, dtype=float), np.asarray(candidate_pool, dtype=float)
        if (
            train.ndim != 2
            or min(train.shape) == 0
            or pool.ndim != 2
            or pool.shape[1] != train.shape[1]
            or not np.all(np.isfinite(train))
            or not np.all(np.isfinite(pool))
        ):
            raise ValueError("Training inputs and candidate pool must be finite aligned matrices")
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
                raise ValueError("SOptimalSampler requires one scalar response per training point")
        self.X_train, self.candidate_pool, self.candidate_ids = (
            train.copy(),
            pool.copy(),
            ids.copy(),
        )
        self.y_train = None if target is None else target.copy()
        self.model = self.last_scores = self._pending_point = None
        self.support_indices_ = self.mapping_ = self.column_scale_ = None
        self.last_selection_report, self.last_selection_reason = [], None

    def set_model(self, model):
        if self.X_train is None:
            raise RuntimeError("Initialize the sampler before setting its model")
        if not getattr(model, "is_fitted_", False):
            raise RuntimeError("SOptimalSampler requires a fitted scalar PCE")
        mapping = getattr(model, "mapping_", None)
        coef = np.asarray(getattr(model, "coef_", None), dtype=float)
        if (
            not isinstance(mapping, dict)
            or not mapping
            or coef.shape != (len(mapping),)
            or not np.all(np.isfinite(coef))
        ):
            raise ValueError("The fitted PCE needs aligned mapping_ and scalar coef_")
        poly = np.asarray(model.poly_matrix, dtype=float)
        if self.poly_matrix is not None and not np.array_equal(self.poly_matrix, poly):
            raise ValueError("Sampler and model must use the same poly_matrix")
        expected = (
            self.X_train
            if self._pending_point is None
            else np.vstack([self.X_train, self._pending_point])
        )
        standardize_inputs(expected, poly)
        fitted_x = getattr(model, "X", None)
        if self._pending_point is not None:
            if fitted_x is None or not np.array_equal(fitted_x, expected):
                raise RuntimeError(
                    "Append the selected observation and refit before setting the model"
                )
        elif fitted_x is not None and not np.array_equal(fitted_x, expected):
            raise ValueError("Initialize with the observations used to fit the supplied PCE")
        constants = [
            i
            for i, expression in enumerate(mapping.values())
            if str(expression).strip() in {"1", "1.0"}
        ]
        if len(constants) != 1:
            raise ValueError("The fitted PCE must identify exactly one constant basis term")
        support = getattr(model, "support_indices_", None)
        if support is None:
            support = np.flatnonzero(coef)
        support = np.asarray(support)
        if (
            support.ndim != 1
            or not np.issubdtype(support.dtype, np.integer)
            or len(np.unique(support)) != len(support)
            or np.any(support < 0)
            or np.any(support >= len(mapping))
        ):
            raise ValueError("support_indices_ must contain distinct valid integer column indices")
        support = np.union1d(constants, support)
        if not np.all(np.isin(np.flatnonzero(coef), support)):
            raise ValueError("The selected support must contain every nonzero fitted coefficient")
        gamma = getattr(model, "gamma_", None)
        scale = np.ones(len(support))
        if gamma is not None:
            gamma = np.asarray(gamma, dtype=float)
            if gamma.shape != coef.shape or not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
                raise ValueError(
                    "gamma_ must contain finite positive basis norms aligned with mapping_"
                )
            scale = np.sqrt(gamma[support])
        keys = list(mapping)
        self.mapping_ = {keys[i]: mapping[keys[i]] for i in support}
        self.support_indices_, self.column_scale_ = support, scale
        self.poly_matrix, self.model = poly.copy(), model
        self.X_train = expected.copy()
        if self._pending_point is not None:
            self.y_train = None
        self._pending_point = self.last_scores = None

    def score_candidates(self):
        if self.model is None:
            raise RuntimeError("Set a fitted PCE before scoring candidates")
        if self._pending_point is not None:
            raise RuntimeError(
                "Evaluate the selected point, refit, and call set_model before scoring again"
            )
        if not len(self.candidate_pool):
            raise RuntimeError("NO_FEASIBLE_CANDIDATE")
        standardize_inputs(self.candidate_pool, self.poly_matrix)
        A = construct_matrix_optimized(
            self.mapping_, normalization_all(self.X_train, self.poly_matrix).T
        )
        C = construct_matrix_optimized(
            self.mapping_, normalization_all(self.candidate_pool, self.poly_matrix).T
        )
        log_s, diagnostics = s_optimal_scores(A / self.column_scale_, C / self.column_scale_)
        geometry = TargetGeometry.from_poly(self.poly_matrix)
        state = CandidateFeasibility(
            geometry.transform(self.X_train), geometry.transform(self.candidate_pool), ratio=0.0
        )
        with np.errstate(under="ignore"):
            s_value = np.exp(log_s)
        self.last_scores = SOptimalScores(
            log_s,
            s_value,
            state.mask.copy(),
            state.distances.copy(),
            float(state.duplicate_tolerance),
            diagnostics,
        )
        return self.last_scores

    def _compute_scores(self):
        """Natural-log S-values; selection avoids exponentiation underflow."""
        return self.score_candidates().log_s

    def select_samples(self, k=1):
        if not isinstance(k, (int, np.integer)) or k not in (0, 1):
            raise ValueError("Sequential S-optimal supports k=1; refit after each new observation")
        if self.candidate_pool is None:
            raise RuntimeError("Initialize the sampler before selecting")
        if k == 0 or not len(self.candidate_pool):
            self.last_selection_report = []
            self.last_selection_reason = "NO_FEASIBLE_CANDIDATE" if k else None
            return np.empty((0, self.X_train.shape[1]))
        raw = self.score_candidates()
        if not np.any(raw.valid):
            self.last_selection_report = []
            self.last_selection_reason = "NO_FEASIBLE_CANDIDATE"
            return np.empty((0, self.X_train.shape[1]))
        winner = select_candidate(raw, self.candidate_ids)
        self.last_selection_report = [
            dict(
                candidate_id=int(self.candidate_ids[winner]),
                selected_local_index=winner,
                criterion="S-optimal",
                acquisition_version=SOPT_VERSION,
                score=float(raw.log_s[winner]),
                log_s=float(raw.log_s[winner]),
                s_value=float(raw.s_value[winner]),
                support_indices=self.support_indices_.tolist(),
                support_size=len(self.support_indices_),
                feasible_count=int(raw.valid.sum()),
                exact_max_ties=int(np.count_nonzero(raw.log_s[raw.valid] == raw.log_s[winner])),
                minimum_distance=float(raw.distance[winner]),
                distance_threshold=0.0,
                duplicate_tolerance=raw.duplicate_tolerance,
                **raw.diagnostics,
            )
        ]
        selected = self.candidate_pool[[winner]].copy()
        self._pending_point = selected.copy()
        self.candidate_pool = np.delete(self.candidate_pool, winner, axis=0)
        self.candidate_ids = np.delete(self.candidate_ids, winner)
        self.last_selection_reason = None
        return selected
