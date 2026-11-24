"""Heredity-constrained sparse solver for polynomial-chaos expansions."""

from __future__ import annotations

import itertools
from functools import cmp_to_key

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import enet_path

from .base import BasePCESolver

MIN_RESIDUAL_DF = 2
CONTRIBUTION_TOL = 1e-12
SCORING_POLICY = "closed_support_cleanup_1e-12_refit_df2"


def _fit_corrected_details(A, y, gamma, total_columns):
    """OLS/PRESS with the correction expressed in theoretical basis norms.

    ``gamma[j] = E[phi_j**2]``; the inverse-Gram trace for orthonormalized
    columns is ``gamma @ diag((A.T @ A)**-1)``. Only N - K >= 2, full column rank,
    valid PRESS denominators and representable errors make a fit eligible.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float)
    out = dict(
        coef=None,
        rank=0,
        raw=np.inf,
        corrected=np.inf,
        factor=np.inf,
        score=np.inf,
        eligible=False,
        inv_diag=None,
        minimum_denominator=None,
    )
    if A.ndim != 2:
        return out
    n, k = A.shape
    if (
        n != len(y)
        or min(n, k) == 0
        or gamma.shape != (k,)
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(gamma))
        or np.any(gamma <= 0)
    ):
        return out
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            return _corrected_svd_details(A, y, gamma, total_columns, out)
    except (np.linalg.LinAlgError, FloatingPointError):
        return out


def _corrected_svd_details(A, y, gamma, total_columns, out):
    """Numerical kernel; the caller handles floating-point/SVD failures."""
    n, k = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > np.finfo(float).eps * max(A.shape) * s[0]))
    out["rank"] = rank
    if rank < k:
        return out
    B = Vt.T / s
    coef = B @ (U.T @ y)
    if not np.all(np.isfinite(coef)):
        return out
    out.update(coef=coef, inv_diag=np.sum(B * B, axis=1))
    if n - k < MIN_RESIDUAL_DF:
        return out
    denom = 1 - np.sum(U * U, axis=1)
    out["minimum_denominator"] = float(denom.min())
    if np.any(denom <= np.finfo(float).eps * max(A.shape)):
        return out
    residual = (y - A @ coef) / denom
    scale = float(np.max(np.abs(residual)))
    rms = scale * np.sqrt(np.mean((residual / scale) ** 2)) if scale else 0.0
    raw = float(rms * rms)
    if not np.isfinite(raw) or (scale and raw == 0):
        return out
    factor = n / (n - k) * (1 + float(gamma @ out["inv_diag"]))
    corrected = raw * factor
    eligible = np.isfinite(corrected)
    score = calc_pce_ebic(corrected, k, total_columns, n) if eligible else np.inf
    out.update(
        raw=raw, factor=factor, corrected=corrected, score=float(score), eligible=bool(eligible)
    )
    return out


def _fit_ols_for_pruning(design, target):
    """Raw-design SVD minimum-norm coefficients at every pruning size.

    Full-column-rank coefficients equal OLS. Underdetermined/rank-deficient
    coefficients only rank legal leaves; they never make PRESS/IC admissible.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    rank = 0
    if (
        design.ndim != 2
        or design.shape[0] != target.size
        or min(design.shape) == 0
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(target))
    ):
        return None, rank
    try:
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
        keep = singular > tolerance
        rank = int(np.sum(keep))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            if rank == design.shape[1]:
                coef = (vt.T / singular) @ (u.T @ target)
            else:
                coef = (vt[keep].T / singular[keep]) @ (u[:, keep].T @ target)
        return (coef if np.all(np.isfinite(coef)) else None), rank
    except np.linalg.LinAlgError:
        return None, rank


def _mapped_inherited_support(previous, mapping, gamma):
    """Validate legacy support metadata before mapping a complete support."""
    old_mapping = getattr(previous, "mapping_", None)
    old_gamma = getattr(previous, "gamma_", None)
    if not isinstance(old_mapping, dict) or not old_mapping or old_gamma is None:
        return None, "missing_support_metadata"
    old_names = list(old_mapping)
    try:
        old_gamma = np.asarray(old_gamma, dtype=float)
        if (
            old_gamma.shape != (len(old_names),)
            or not np.all(np.isfinite(old_gamma))
            or np.any(old_gamma <= 0)
        ):
            return None, "invalid_support_metadata"
        support = getattr(previous, "support_indices_", None)
        if support is None:
            coef = getattr(previous, "coef_", None)
            if coef is None:
                return None, "missing_support_metadata"
            coef = np.asarray(coef, dtype=float)
            if coef.shape != (len(old_names),) or not np.all(np.isfinite(coef)):
                return None, "invalid_support_metadata"
            support = np.union1d([0], np.flatnonzero(coef))
        support = np.asarray(support)
        if (
            support.ndim != 1
            or not support.size
            or support.dtype.kind not in "iu"
            or np.any(support < 0)
            or np.any(support >= len(old_names))
            or len(np.unique(support)) != len(support)
        ):
            return None, "invalid_support_metadata"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_support_metadata"
    indices = {name: idx for idx, name in enumerate(mapping)}
    names = [old_names[idx] for idx in support]
    if any(name not in indices for name in names):
        return None, "support_missing_in_current_dictionary"
    if any(
        old_mapping[name] != mapping[name] or old_gamma[idx] != gamma[indices[name]]
        for idx, name in zip(support, names)
    ):
        return None, "support_definition_changed"
    inherited = sorted(indices[name] for name in names)
    ancestors, _ = build_heredity_dicts(list(mapping))
    if not _is_heredity_closed(inherited, ancestors):
        return None, "support_not_closed"
    return inherited, "refit_candidate_not_selected"


class HCBPSolver(BasePCESolver):
    """Fit sparse PCE models with heredity-aware backward pruning.

    ``support_indices_`` records cleaned fitted columns, including required
    ancestors even when their coefficients are numerically zero. Cleaning a
    scoring copy never changes the original SVD leaf-removal path.
    ``ic_`` and ``error_`` describe that support on the current training data.
    IC and cross-round comparisons use corrected LOOCV. A strict increase
    above 10% triggers a current-data refit of the previous support. Raw PRESS
    is retained separately in ``raw_error_`` and ``press_error_``.
    """

    def __init__(self, poly_matrix=None, max_degree=3, l1_ratio=0.95, last_model=None, q=1):
        """Initialize the HCBP solver.

        Parameters
        ----------
        poly_matrix : np.ndarray | None, default=None
            Distribution matrix with shape ``(n_features, 3)``. Column 0 is
            the distribution code (``0`` uniform, ``1`` normal, ``2``
            lognormal). Columns 1 and 2 store ``[lower_bound, upper_bound]``
            for uniform variables or ``[mean, std]`` for normal/lognormal
            variables.
        max_degree : int, default=3
            Maximum basis degree allowed in each dimension.
        l1_ratio : float, default=0.95
            Elastic-net mixing ratio used in the path search stage.
        last_model : HCBPSolver | None, default=None
            Previous model supplied for warm starts or inherited fits.
        q : float, default=1
            Hyperbolic truncation exponent controlling basis sparsity.
        """

        super().__init__(poly_matrix, max_degree, l1_ratio, last_model, q)
        self.poly_matrix = poly_matrix
        self.max_degree = max_degree
        self.l1_ratio = l1_ratio
        self.last_model = last_model
        self.q = q
        self.ic_ = None
        self.support_indices_ = None
        self.error_name_ = "corrected_LOOCV_MSE"
        self.corrected_error_ = None
        self.raw_error_ = None
        self.correction_factor_ = None
        self.scoring_policy_ = SCORING_POLICY

    def train_model(self, X_data, y):
        """Train the sparse PCE coefficients from an expanded design matrix.

        Parameters
        ----------
        X_data : np.ndarray
            Expanded polynomial-chaos design matrix with shape
            ``(n_samples, n_basis)``.
        y : np.ndarray
            Target values aligned with ``X_data``. Expected shape is
            ``(n_samples,)`` or ``(n_samples, 1)``.

        Returns
        -------
        tuple[np.ndarray, dict[str, float]]
            Final coefficient vector with shape ``(n_basis,)`` and a mapping
            of active basis names to their retained coefficients.
        """

        n_dim = self.poly_matrix.shape[0]
        expr_mapping = getattr(self, "mapping_", None)
        feature_names = list(expr_mapping.keys()) if expr_mapping else []

        self.search_diagnostics_ = {}
        self.error_name_ = "corrected_LOOCV_MSE"
        self.scoring_policy_ = SCORING_POLICY
        evaluation_cache = {}
        coef, ic_value, loo_cv, best_set, ols_coef = improved_enet_ols_selection(
            X_data,
            y,
            n_dim,
            expr_mapping=expr_mapping,
            l1_ratio=self.l1_ratio,
            diagnostics=self.search_diagnostics_,
            basis_norms=self.gamma_,
            evaluation_cache=evaluation_cache,
        )
        final_coef = coef
        final_set = [] if best_set is None else list(best_set)
        final_values = ols_coef
        new_valid = (
            final_values is not None
            and np.isfinite(loo_cv)
            and loo_cv >= 0
            and (np.isfinite(ic_value) or np.isneginf(ic_value))
        )
        rollback = dict(
            policy="corrected_loocv_10pct_guard",
            increase_threshold=0.10,
            error_definition="corrected_LOOCV_MSE",
            previous_loo=None,
            new_loo=float(loo_cv),
            new_ic=float(ic_value),
            refitted_loo=None,
            refitted_ic=None,
            triggered=False,
            trigger_reason=None,
            rolled_back=False,
        )

        if self.last_model is not None and self.warm_start_compatible_:
            # Preserve the fitted support, even when OLS retained a zero term.
            # Legacy models lack this metadata; keep their intercept/nonzeros.
            inherited_set, self.warm_start_reason_ = _mapped_inherited_support(
                self.last_model, expr_mapping, self.gamma_
            )
            if inherited_set is not None:
                try:
                    previous_loo = float(self.last_model.error_)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    previous_loo = np.nan
                previous_valid = (
                    np.isfinite(previous_loo)
                    and previous_loo >= 0
                    and getattr(self.last_model, "error_name_", None) == "corrected_LOOCV_MSE"
                )
                if previous_valid:
                    rollback["previous_loo"] = previous_loo
                # A multiplicative threshold also covers a previous error of zero.
                # In that case, any positive new error triggers a refit.
                if not new_valid:
                    rollback.update(triggered=True, trigger_reason="new_fit_invalid")
                elif not previous_valid:
                    self.warm_start_reason_ = "previous_loo_unavailable"
                elif float(loo_cv) > 1.10 * previous_loo:
                    rollback.update(triggered=True, trigger_reason="loo_increase")
                else:
                    self.warm_start_reason_ = "loo_increase_not_above_10pct"

                if rollback["triggered"]:
                    n_samples, n_features = X_data.shape
                    ancestors, _ = build_heredity_dicts(list(expr_mapping))
                    inherited_path = inherited_set.copy()
                    inherited_set, inherited_details, cleaning = _clean_scoring_support(
                        X_data,
                        np.asarray(y).reshape(-1),
                        inherited_set,
                        self.gamma_,
                        ancestors,
                        n_dim=n_dim,
                        evaluation_cache=evaluation_cache,
                    )
                    inherited_set = list(inherited_set)
                    inherited_ic = inherited_details["score"]
                    inherited_error = inherited_details["corrected"]
                    inherited_values = (
                        inherited_details["coef"] if inherited_details["eligible"] else None
                    )
                    rollback.update(
                        inherited_path_support=inherited_path,
                        refitted_support=inherited_set.copy(),
                        refitted_k=len(inherited_set),
                        cleaning=cleaning,
                    )
                    rollback.update(
                        refitted_loo=float(inherited_error), refitted_ic=float(inherited_ic)
                    )
                    inherited_valid = (
                        inherited_values is not None
                        and np.isfinite(inherited_error)
                        and inherited_error >= 0
                        and (np.isfinite(inherited_ic) or np.isneginf(inherited_ic))
                    )
                    if not inherited_valid:
                        self.warm_start_reason_ = "refitted_support_invalid"
                    elif not new_valid or inherited_error <= loo_cv:
                        # Compare current-data LOOCV only. Equal errors retain
                        # the previous structure, regardless of IC or its size.
                        final_coef = np.zeros(n_features)
                        final_coef[inherited_set] = inherited_values
                        final_set = inherited_set
                        final_values = inherited_values
                        ic_value, loo_cv = inherited_ic, inherited_error
                        self.warm_start_used_ = True
                        self.warm_start_reason_ = "refitted_support_selected"
                        rollback["rolled_back"] = True
                    else:
                        self.warm_start_reason_ = "refitted_support_loo_worse"

        rollback["reason"] = self.warm_start_reason_
        self.search_diagnostics_["cross_round_selection"] = rollback

        if (
            final_values is None
            or not np.isfinite(loo_cv)
            or loo_cv < 0
            or not (np.isfinite(ic_value) or np.isneginf(ic_value))
        ):
            raise ValueError(
                "No candidate support has a valid full-rank raw PRESS / corrected LOOCV fit; "
                "require N-K >= 2 and check numerical rank / PRESS representability."
            )
        # Saved with the existing model object; all three describe this fit.
        self.support_indices_ = np.asarray(final_set, dtype=int)
        self.ic_ = float(ic_value)
        self.error_ = self.corrected_error_ = float(loo_cv)
        final_details = evaluation_cache.get(tuple(final_set))
        if not final_details:
            final_details = _fit_corrected_details(
                X_data[:, final_set], y, self.gamma_[final_set], X_data.shape[1]
            )
        self.raw_error_ = float(final_details["raw"])
        self.correction_factor_ = float(final_details["factor"])
        self.search_diagnostics_.update(
            winner_source=(
                "warm_start"
                if self.warm_start_used_
                else self.search_diagnostics_.get("winner_source")
            ),
            warm_start_won=bool(self.warm_start_used_),
            warm_start_reason=self.warm_start_reason_,
            final_support=list(map(int, final_set)),
            final_k=len(final_set),
            final_IC=float(ic_value),
            final_PRESS=self.raw_error_,
            final_corrected_LOO=self.corrected_error_,
            final_correction_factor=self.correction_factor_,
            error_definition=self.error_name_,
        )

        final_values = np.asarray(final_values, dtype=float).reshape(-1)

        effective_basis = {}
        if feature_names:
            max_idx = len(feature_names)
            for col_idx, value in zip(final_set, final_values):
                basis_name = feature_names[col_idx] if col_idx < max_idx else f"Basis_{col_idx}"
                effective_basis[basis_name] = value
        else:
            for col_idx, value in zip(final_set, final_values):
                effective_basis[f"Basis_{col_idx}"] = value

        return final_coef, effective_basis


def build_heredity_dicts(keys):
    """Build parent-child relationships implied by polynomial basis names.

    Parameters
    ----------
    keys : list[str]
        Ordered list of basis identifiers.

    Returns
    -------
    tuple[dict[int, set[int]], dict[int, set[int]]]
        Ancestor and child relationships expressed in basis-column indices.
    """

    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    ancestors = {idx: set() for idx in range(len(keys))}
    children = {idx: set() for idx in range(len(keys))}
    constant_key = "x00" if "x00" in key_to_idx else "1" if "1" in key_to_idx else None

    for idx, key in enumerate(keys):
        if key in {"x00", "1"}:
            continue

        parts = key.split("*")
        variable_degree = {}
        valid = True
        for part in parts:
            if "_" in part and part.startswith("X"):
                try:
                    variable_name, degree_name = part.split("_")
                    variable_degree[int(variable_name[1:])] = int(degree_name)
                except ValueError:
                    valid = False
            else:
                valid = False
        if not valid:
            continue

        variable_indices = sorted(variable_degree.keys())
        chains = []
        for variable_idx in variable_indices:
            degree = variable_degree[variable_idx]
            chain = list(range(degree, 0, -2))
            chain.append(0)
            chains.append(chain)

        for combo in itertools.product(*chains):
            ancestor_parts = []
            for variable_idx, degree in zip(variable_indices, combo):
                if degree > 0:
                    ancestor_parts.append(f"X{variable_idx}_{degree}")
            ancestor_key = constant_key if not ancestor_parts else "*".join(ancestor_parts)
            if ancestor_key != key and ancestor_key and ancestor_key in key_to_idx:
                ancestor_idx = key_to_idx[ancestor_key]
                ancestors[idx].add(ancestor_idx)
                children[ancestor_idx].add(idx)

    return ancestors, children


def _is_heredity_closed(support, ancestors):
    """Check fitted support membership, including ancestors with zero coefficients."""
    support_set = set(support)
    return all(ancestors.get(idx, set()).issubset(support_set) for idx in support_set)


def _is_better_model(ic_value, num_bases, incumbent_ic, incumbent_bases):
    """Compare IC, then fitted support size, retaining the incumbent on ties.

    Infinite scores are branched explicitly. Finite scores within 1e-12
    absolute/relative tolerance are treated as ties; NaN/+inf are invalid.
    """

    if np.isnan(ic_value) or np.isposinf(ic_value):
        return False
    if np.isnan(incumbent_ic) or np.isposinf(incumbent_ic):
        return True
    if np.isneginf(ic_value):
        return not np.isneginf(incumbent_ic) or num_bases < incumbent_bases
    if np.isneginf(incumbent_ic):
        return False
    if np.isclose(ic_value, incumbent_ic, rtol=1e-12, atol=1e-12):
        return num_bases < incumbent_bases
    return ic_value < incumbent_ic


def calc_pce_ebic(loocv, num_bases, n_all_bases, n_samples, gamma=None):
    """Compute the extended Bayesian information criterion.

    Parameters
    ----------
    loocv : float
        Error used for scoring; SCAL passes corrected LOOCV MSE, without a floor.
    num_bases : int
        Fitted support size, including the intercept and retained zero terms.
    n_all_bases : int
        Total number of basis terms available before pruning.
    n_samples : int
        Number of training samples.
    gamma : float or None, default=None
        Retained for call compatibility. Scoring always fixes gamma at 0.5.

    Returns
    -------
    float
        EBIC score; lower values indicate a better model.
    """

    if (
        n_samples <= 0
        or num_bases < 0
        or num_bases > n_all_bases
        or not np.isfinite(loocv)
        or loocv < 0
    ):
        return np.inf
    if loocv == 0:
        return -np.inf
    log_comb = (
        gammaln(n_all_bases + 1) - gammaln(num_bases + 1) - gammaln(n_all_bases - num_bases + 1)
    )
    return n_samples * np.log(loocv) + num_bases * np.log(n_samples) + log_comb


def calc_pce_aicc(loo_cv, num_bases, n_samples):
    """Compute AICc for a fitted support. HCBP uses EBIC for model selection.

    Parameters
    ----------
    loo_cv : float
        Leave-one-out error estimate.
    num_bases : int
        Number of active basis terms in the candidate model.
    n_samples : int
        Number of training samples.

    Returns
    -------
    float
        AICc score. Lower values indicate a better model.
    """

    aic = 2 * num_bases + n_samples * np.log(loo_cv + 1e-12)
    if n_samples > num_bases + 1:
        return aic + 2 * num_bases * (num_bases + 1) / (n_samples - num_bases - 1)
    return np.inf


def evaluate_model(
    X_sub, y_flat, num_bases, n_samples, n_features, n_dim, *, basis_norms=None, details=None
):
    """Evaluate OLS with corrected LOOCV (unit basis norms when omitted).

    Parameters
    ----------
    X_sub : np.ndarray
        Design submatrix containing the candidate active basis columns.
    y_flat : np.ndarray
        One-dimensional target vector.
    num_bases : int
        Number of active basis terms represented by ``X_sub``.
    n_samples : int
        Number of training samples.
    n_features : int
        Total number of basis columns available before selection.
    n_dim : int
        Number of input variables, retained for call compatibility. Every
        dimension uses N log(E_corr) + k log(N) + log C(P, k).
    basis_norms : np.ndarray | None
        Theoretical squared basis norms, aligned with X_sub. Omission assumes
        an orthonormal dictionary; HCBPSolver always supplies its actual norms.
    details : dict | None
        Optional destination for raw PRESS, correction factor and fit details.

    Returns
    -------
    tuple[float, float, np.ndarray | None, int]
        Information-criterion value, corrected LOOCV MSE, OLS coefficients, and
        effective numerical rank of ``X_sub``.
    """

    if X_sub.shape != (n_samples, num_bases):
        return np.inf, np.inf, None, 0
    norms = np.ones(num_bases) if basis_norms is None else np.asarray(basis_norms, dtype=float)
    fitted = _fit_corrected_details(X_sub, y_flat, norms, n_features)
    if details is not None:
        details.update(fitted)
    return (
        fitted["score"],
        fitted["corrected"],
        fitted["coef"] if fitted["eligible"] else None,
        fitted["rank"],
    )


def _prepare_search_response(y):
    """Dimensionless, population-standardized response used only by Elastic Net.

    Subtract a reference first, then scale before squaring. No absolute cutoff
    in response units is used to classify a response as constant.
    """
    target = np.asarray(y, dtype=float).reshape(-1)
    if not target.size or not np.all(np.isfinite(target)):
        raise ValueError("Support-search response must be nonempty and finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        delta = target - target[0]
        offset = delta.mean()
        centered = delta - offset
    if not np.all(np.isfinite(centered)) or not np.isfinite(offset):
        raise ValueError("Response centering is not representable in float64")
    amplitude = float(np.max(np.abs(centered)))
    diagnostics = dict(
        response_preprocessing="center_scale_for_support_search",
        response_reference=float(target[0]),
        response_center=float(target[0] + offset),
        response_center_offset=float(offset),
        response_scale=0.0,
        constant_response=amplitude == 0,
        alpha_max=None,
        alpha_min=None,
        n_alphas=0,
    )
    if amplitude == 0:
        diagnostics["numerical_status"] = "constant_at_input_precision"
        return np.zeros_like(target), diagnostics, True
    unit = centered / amplitude
    rms = float(np.sqrt(np.mean(unit * unit)))
    search = unit / rms
    diagnostics["response_scale"] = amplitude * rms
    diagnostics["numerical_status"] = (
        "subnormal_response_scale" if amplitude * rms == 0 else "finite"
    )
    if not np.all(np.isfinite(search)) or not np.isfinite(diagnostics["response_center"]):
        raise ValueError("Standardized support-search response is not finite")
    return search, diagnostics, False


def _support_fit_details(X, y, support, norms, n_dim, cache):
    """Cache actual fits by their columns, never by a parent search path."""
    key = tuple(sorted(support))
    if key not in cache:
        details = {}
        ic, error, coef, rank = evaluate_model(
            X[:, key],
            y,
            len(key),
            len(X),
            X.shape[1],
            n_dim,
            basis_norms=norms[list(key)],
            details=details,
        )
        # Preserve coefficients from a full-rank fit even when PRESS/df rejects
        # its score: they can identify a smaller, admissible scoring support.
        details.setdefault("coef", coef)
        details.setdefault("rank", rank)
        details.setdefault("score", ic)
        details.setdefault("corrected", error)
        details.setdefault("eligible", coef is not None and (np.isfinite(ic) or np.isneginf(ic)))
        cache[key] = details
    return cache[key]


def _clean_scoring_support(
    X,
    y,
    support,
    norms,
    ancestors,
    *,
    n_dim=0,
    evaluation_cache=None,
    column_std=None,
    response_scale=None,
):
    """Refit a private support after removing negligible, unneeded columns.

    Each full-column-rank OLS fit defines relative contributions
    abs(c_j) * std(phi_j) / std(y). Keep the constant, significant terms, and
    every ancestor needed by those terms. Refit after deletions, repeating
    until the support stops changing. Required zero ancestors still count in
    K and use their unmodified OLS coefficients in PRESS. The input support
    and cached OLS coefficients are never edited.
    """
    cache = {} if evaluation_cache is None else evaluation_cache
    path = tuple(sorted(support))
    current = path
    if column_std is None:
        column_std = np.std(X, axis=0)
    if response_scale is None:
        _, response_info, _ = _prepare_search_response(y)
        response_scale = response_info["response_scale"]
    steps = []
    protected = []
    while True:
        fitted = _support_fit_details(X, y, current, norms, n_dim, cache)
        coef = fitted["coef"]
        if coef is None or fitted["rank"] != len(current):
            break
        if response_scale == 0:
            # A constant response has no nonconstant contribution to retain.
            small = np.ones(len(current), dtype=bool)
        else:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                relative = np.abs(coef) / response_scale * column_std[list(current)]
            small = np.isfinite(relative) & (relative < CONTRIBUTION_TOL)
        active = set(current)
        required = {idx for idx, negligible in zip(current, small) if idx == 0 or not negligible}
        # Pruning retains ancestors already in the proposal. The search stage
        # adds missing ancestors before comparing heredity-closed supports.
        while True:
            parents = set().union(*(ancestors.get(idx, set()) for idx in required))
            enlarged = required | (parents & active)
            if enlarged == required:
                break
            required = enlarged
        cleaned = tuple(idx for idx in current if idx in required)
        protected = [
            idx
            for idx, negligible in zip(current, small)
            if idx != 0 and negligible and idx in required
        ]
        if cleaned == current:
            break
        steps.append(
            dict(
                from_k=len(current),
                to_k=len(cleaned),
                removed=[idx for idx in current if idx not in required],
            )
        )
        current = cleaned
    cleaning = dict(
        tau=CONTRIBUTION_TOL,
        path_k=len(path),
        scoring_k=len(current),
        removed=[idx for idx in path if idx not in current],
        protected_zero_ancestors=protected,
        refit_steps=steps,
    )
    return current, fitted, cleaning


def improved_enet_ols_selection(
    X,
    Y,
    n_dim,
    expr_mapping=None,
    l1_ratio=0.95,
    *,
    diagnostics=None,
    basis_norms=None,
    evaluation_cache=None,
):
    """Select a sparse basis with heredity-aware backward pruning.

    Parameters
    ----------
    X : np.ndarray
        Expanded polynomial-chaos design matrix.
    Y : np.ndarray
        Target vector aligned with ``X``.
    n_dim : int
        Number of input dimensions.
    expr_mapping : dict[str, str] | None, default=None
        Optional mapping used to infer heredity relations between basis terms.
    l1_ratio : float, default=0.95
        Elastic-net mixing ratio used to generate the regularization path.

    Returns
    -------
    tuple[np.ndarray, float, float, list[int], np.ndarray | None]
        Final coefficient vector, best information-criterion value, corrected LOOCV
        error, selected basis indices, and OLS coefficients on the active set.
    """

    n_samples, n_features = X.shape
    norms = np.ones(n_features) if basis_norms is None else np.asarray(basis_norms, dtype=float)
    if norms.shape != (n_features,) or not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("basis_norms must contain one finite positive theoretical norm per column")
    detail_cache = {} if evaluation_cache is None else evaluation_cache

    def evaluate_support(support):
        fitted = _support_fit_details(X, y_flat, support, norms, n_dim, detail_cache)
        return (
            fitted["score"],
            fitted["corrected"],
            fitted["coef"] if fitted["eligible"] else None,
            fitted["rank"],
        )

    if l1_ratio == 0:
        l1_ratio = 1e-3

    y_flat = Y.flatten() if np.asarray(Y).ndim > 1 else np.asarray(Y, dtype=float)
    ancestors_dict, children_dict = ({}, {})
    if expr_mapping is not None:
        ancestors_dict, children_dict = build_heredity_dicts(list(expr_mapping.keys()))

    y_search, response_diagnostics, constant = _prepare_search_response(y_flat)
    if diagnostics is not None:
        diagnostics.update(response_diagnostics)
        diagnostics.update(
            N=int(n_samples),
            P=int(n_features),
            candidate_policy=SCORING_POLICY,
            contribution_tolerance=CONTRIBUTION_TOL,
            minimum_residual_df=MIN_RESIDUAL_DF,
            error_definition="corrected_LOOCV_MSE",
            raw_counts={"k_lt_N": 0, "k_eq_N": 0, "k_gt_N": 0},
            groups=[],
            winner_source="intercept",
            warm_start_won=False,
        )
    # Pruning uses the actual standard deviation, including zero. The separate
    # EN normalization below guards zero divisors without changing this score.
    x_std_full = np.std(X, axis=0)

    X_rest = X[:, 1:]
    # At least two residual degrees of freedom are required even for the
    # intercept. N=3 admits only that support; N<3 returns an invalid fit.
    if X_rest.shape[1] == 0 or n_samples <= 3:
        if diagnostics is not None:
            diagnostics["intercept_reason"] = (
                "no_nonconstant_columns" if X_rest.shape[1] == 0 else "two_residual_df_limit"
            )
        ic_value, loo_cv, ols_coef, _ = evaluate_support([0])
        final_coef = np.zeros(n_features)
        if ols_coef is not None:
            final_coef[0] = ols_coef[0]
        return final_coef, ic_value, loo_cv, [0], ols_coef
    x_mean = np.mean(X_rest, axis=0)
    x_std = np.std(X_rest, axis=0)
    x_std[x_std == 0] = 1.0
    x_norm = (X_rest - x_mean) / x_std

    # The path and its alpha range must use the same standardized columns.
    xt_y = x_norm.T @ y_search
    max_abs_xt_y = float(np.max(np.abs(xt_y))) if xt_y.size else 0.0
    if max_abs_xt_y <= 1e-12:
        ic_value, loo_cv, ols_coef, _ = evaluate_support([0])
        final_coef = np.zeros(n_features)
        if ols_coef is not None:
            final_coef[0] = ols_coef[0]
        return final_coef, ic_value, loo_cv, [0], ols_coef
    alpha_max = max_abs_xt_y / (n_samples * l1_ratio)
    alpha_min = alpha_max * 1e-6
    if alpha_max <= alpha_min:
        alpha_max = 1.0

    n_alpha = 100
    if n_features > 1000:
        n_alpha = 75
        if n_features > 2000:
            n_alpha = 50
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alpha)
    if diagnostics is not None:
        diagnostics.update(
            alpha_max=alpha_max, alpha_min=alpha_min, n_alphas=n_alpha, l1_ratio=float(l1_ratio)
        )
    alphas, coefs_path, _ = enet_path(x_norm, y_search, l1_ratio=l1_ratio, alphas=alphas)

    final_best_ic, final_best_err = np.inf, np.inf
    final_best_set, final_best_ols_coef = [], None
    winner_source = None
    support_cache, pruning_cache, scoring_copies = {}, {}, {}
    scoring_records = []

    def score_support(support, source):
        # A cleaned/refitted copy supplies the score. Return original-path OLS
        # coefficients to the pruning loop so cleaning cannot reorder leaves.
        nonlocal final_best_ic, final_best_err, final_best_set, final_best_ols_coef, winner_source
        support = tuple(sorted(support))
        if support not in support_cache:
            original = _support_fit_details(X, y_flat, support, norms, n_dim, detail_cache)
            scoring_set, fitted, cleaning = _clean_scoring_support(
                X,
                y_flat,
                support,
                norms,
                ancestors_dict,
                n_dim=n_dim,
                evaluation_cache=detail_cache,
                column_std=x_std_full,
                response_scale=response_diagnostics["response_scale"],
            )
            scoring_copies[support] = (scoring_set, fitted, cleaning)
            support_cache[support] = (
                fitted["score"],
                fitted["corrected"],
                original["coef"],
                original["rank"],
            )
        result = support_cache[support]
        scoring_set, fitted, cleaning = scoring_copies[support]
        ic_value, loo_cv = fitted["score"], fitted["corrected"]
        scoring_records.append(
            dict(
                source=source,
                path_support=list(support),
                path_k=len(support),
                scoring_support=list(scoring_set),
                scoring_k=len(scoring_set),
                IC=float(ic_value),
                corrected_LOO=float(loo_cv),
                eligible=bool(fitted["eligible"]),
                cleaning=cleaning,
            )
        )
        if (
            _is_heredity_closed(scoring_set, ancestors_dict)
            and fitted["eligible"]
            and _is_better_model(ic_value, len(scoring_set), final_best_ic, len(final_best_set))
        ):
            final_best_ic, final_best_err = ic_value, loo_cv
            final_best_set, final_best_ols_coef = list(scoring_set), fitted["coef"]
            winner_source = source
        return result

    def pruning_fit(support):
        support = tuple(support)
        if support not in pruning_cache:
            pruning_cache[support] = _fit_ols_for_pruning(X[:, support], y_flat)
        return pruning_cache[support]

    score_support([0], "intercept")

    def compare_records(left, right):
        if _is_better_model(
            left["ic_value"], left["scoring_k"], right["ic_value"], right["scoring_k"]
        ):
            return -1
        if _is_better_model(
            right["ic_value"], right["scoring_k"], left["ic_value"], left["scoring_k"]
        ):
            return 1
        return 0

    regular_records, square_records, path_records = [], [], []
    raw_counts = {"k_lt_N": 0, "k_eq_N": 0, "k_gt_N": 0}
    for path_idx in range(coefs_path.shape[1]):
        path_coef = coefs_path[:, path_idx]
        active_idx = np.flatnonzero(np.abs(path_coef) > 0)
        current_set = sorted({0}.union(int(idx + 1) for idx in active_idx))
        k = len(current_set)
        raw_counts["k_lt_N" if k < n_samples else "k_eq_N" if k == n_samples else "k_gt_N"] += 1
        record = dict(
            path_index=path_idx,
            alpha=float(alphas[path_idx]),
            math_bases_count=k,
            support=current_set,
            group=None,
            ic_value=None,
            loo_error=None,
            rank=None,
        )
        # Apply the original-support count filter BEFORE either group's top_k.
        # In particular an underdetermined raw proposal is never scored/ranked.
        if k > n_samples:
            record["exclusion"] = "raw_k_gt_N"
        elif not np.isfinite(alphas[path_idx]) or not np.all(np.isfinite(path_coef)):
            record["exclusion"] = "nonfinite_path"
        elif k >= n_samples - 1:
            coef, rank = pruning_fit(current_set)
            record["rank"] = rank
            if coef is not None and rank == k:
                record["group"] = "square_exploration"
                record["exploration_reason"] = "fewer_than_two_residual_df"
                square_records.append(record)
            else:
                record["exclusion"] = "square_rank_or_coefficient_invalid"
        else:
            ic_value, loo_cv, coef, rank = score_support(current_set, "raw_strict")
            scoring_set, _, cleaning = scoring_copies[tuple(current_set)]
            record.update(
                scoring_support=list(scoring_set), scoring_k=len(scoring_set), cleaning=cleaning
            )
            record["rank"] = rank
            if coef is not None and (np.isfinite(ic_value) or np.isneginf(ic_value)):
                record.update(group="regular", ic_value=float(ic_value), loo_error=float(loo_cv))
                regular_records.append(record)
            else:
                record["exclusion"] = "strict_PRESS_invalid"
        path_records.append(record)

    top_k = 1 if n_features > 2000 else 3 if n_features > 1000 else 5
    regular_records.sort(key=cmp_to_key(compare_records))
    square_records.sort(key=lambda record: (-record["alpha"], record["path_index"]))
    group_diagnostics = []

    for group_name, records in [
        ("regular", regular_records),
        ("square_exploration", square_records),
    ]:
        if not records:
            continue
        selected_records = records[:top_k]  # Count path records, including repeated supports.
        union_set = {0}
        for record in selected_records:
            union_set.update(record["support"])
        expanded_set = set(union_set)
        while True:
            parents = set().union(*(ancestors_dict.get(idx, set()) for idx in expanded_set))
            previous_size = len(expanded_set)
            expanded_set.update(parents)
            if len(expanded_set) == previous_size:
                break
        current_set = sorted(expanded_set)
        group = dict(
            name=group_name,
            selected_paths=selected_records,
            merged_k=len(union_set),
            expanded_k=len(current_set),
            expanded_support=current_set.copy(),
            prepruning=[],
            prepruning_importance="svd_min_norm_contribution",
            pruning_policy="svd_min_norm_contribution",
            first_strict_valid_k=None,
            formal_valid_candidates=0,
            formal_pruning=[],
            status="searching",
        )
        group_diagnostics.append(group)

        def drop_leaf(coef, rank, trace):
            current_fast = set(current_set)
            leaves = [
                idx
                for idx in current_set
                if idx != 0 and current_fast.isdisjoint(children_dict.get(idx, set()))
            ]
            event = dict(
                k=len(current_set),
                rank=None if rank is None else int(rank),
                removable_leaves=leaves,
            )
            if not leaves:
                event["reason"] = "no_removable_leaf"
                trace.append(event)
                return False
            if coef is None:
                raise ValueError(
                    "SVD pruning coefficients are unavailable; no EN or column-order fallback"
                )
            # Recompute on each current support. Exact ties follow ascending columns.
            contributions = np.abs(coef) * x_std_full[current_set]
            leaf = min(leaves, key=lambda idx: contributions[current_set.index(idx)])
            event.update(
                reason="svd_min_norm_contribution",
                importance=float(contributions[current_set.index(leaf)]),
            )
            event["dropped_leaf"] = int(leaf)
            trace.append(event)
            current_set.remove(leaf)
            return True

        # The df/PRESS gate is applied to the refitted scoring copy. The path
        # itself retains its original columns and unthresholded SVD fit.
        while True:
            if len(current_set) > n_samples:
                coef, rank = pruning_fit(current_set)
                valid = False
            elif len(current_set) < n_samples:
                result = score_support(current_set, group_name + ":first_valid")
                ic, _, coef, rank = result
                valid = coef is not None and (np.isfinite(ic) or np.isneginf(ic))
            else:
                coef, rank = pruning_fit(current_set)
                valid = False
            if valid:
                group["first_strict_valid_k"] = len(current_set)
                group["first_strict_valid_support"] = current_set.copy()
                group["formal_valid_candidates"] = 1
                break
            # The minimum-norm SVD still supplies a leaf ranking when PRESS is invalid.
            if coef is None:
                coef, rank = pruning_fit(current_set)
            if not drop_leaf(coef, rank, group["prepruning"]):
                group["status"] = "invalid_no_removable_leaf"
                break
        if group["first_strict_valid_k"] is None:
            continue

        while len(current_set) > 1:
            if coef is None:
                coef, rank = pruning_fit(current_set)
            if not drop_leaf(coef, rank, group["formal_pruning"]):
                break
            ic, _, coef, rank = score_support(current_set, group_name + ":backward")
            if coef is not None and (np.isfinite(ic) or np.isneginf(ic)):
                group["formal_valid_candidates"] += 1
        group["status"] = "complete"

    final_coef = np.zeros(n_features)
    if final_best_ols_coef is not None:
        final_coef[final_best_set] = final_best_ols_coef
    if diagnostics is not None:
        diagnostics.update(
            raw_counts=raw_counts,
            path_records=path_records,
            scoring_records=scoring_records,
            top_k=top_k,
            regular_eligible=len(regular_records),
            square_eligible=len(square_records),
            groups=group_diagnostics,
            winner_source=winner_source,
            final_support=final_best_set.copy(),
            final_k=len(final_best_set),
            final_IC=float(final_best_ic),
            final_corrected_LOO=float(final_best_err),
            final_PRESS=detail_cache.get(tuple(final_best_set), {}).get("raw", np.inf),
            final_correction_factor=detail_cache.get(tuple(final_best_set), {}).get(
                "factor", np.inf
            ),
        )
    return final_coef, final_best_ic, final_best_err, final_best_set, final_best_ols_coef


def ols_wave(X, Y, coef, *, basis_norms=None):
    """Refit active coefficients with ordinary least squares.

    Parameters
    ----------
    X : np.ndarray
        Design matrix containing all basis columns.
    Y : np.ndarray
        Target vector aligned with ``X``.
    coef : np.ndarray
        Sparse coefficient vector whose support defines the active set.

    Returns
    -------
    tuple[np.ndarray, float]
        Updated coefficient vector and corrected LOOCV MSE. Omitted theoretical
        norms imply an orthonormal dictionary; generic raw PRESS is available
        separately from ``base.ols_wave``.
    """

    support = np.union1d([0], np.flatnonzero(coef)).astype(int)
    norms = np.ones(X.shape[1]) if basis_norms is None else np.asarray(basis_norms, dtype=float)
    _, error, values, _ = evaluate_model(
        X[:, support], Y, len(support), len(X), X.shape[1], 0, basis_norms=norms[support]
    )
    updated = np.zeros(X.shape[1])
    if values is not None:
        updated[support] = values
    return updated, error


__all__ = ["HCBPSolver"]
