"""Polynomial evaluation, coordinate normalization and regression diagnostics."""

from __future__ import annotations

import math
import re

import numpy as np


def get_array(values: dict[str, np.ndarray]) -> np.ndarray:
    """Stack dictionary values into a 2D array.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Mapping of basis names to vectors or arrays.

    Returns
    -------
    np.ndarray
        Array built from the mapping values in insertion order.
    """

    return np.asarray(list(values.values()))


def legendre_polynomial(x: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate a Legendre polynomial.

    Parameters
    ----------
    x : np.ndarray
        Evaluation coordinates in ``[-1, 1]``.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Value of the Legendre polynomial of the requested degree.
    """

    if degree == 0:
        return np.ones_like(x, dtype=float)
    if degree == 1:
        return np.asarray(x, dtype=float)

    previous = np.ones_like(x, dtype=float)
    current = np.asarray(x, dtype=float)
    for order in range(2, degree + 1):
        nxt = ((2 * order - 1) * x * current - (order - 1) * previous) / order
        previous, current = current, nxt
    return current


def hermite_polynomial(x: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate a probabilists' Hermite polynomial.

    Parameters
    ----------
    x : np.ndarray
        Standard-normal coordinates.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Value of the Hermite polynomial of the requested degree.
    """

    if degree == 0:
        return np.ones_like(x, dtype=float)
    if degree == 1:
        return np.asarray(x, dtype=float)

    previous = np.ones_like(x, dtype=float)
    current = np.asarray(x, dtype=float)
    for order in range(2, degree + 1):
        nxt = x * current - (order - 1) * previous
        previous, current = current, nxt
    return current


def normalization_u(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Normalize a uniform variable to ``[-1, 1]``.

    Parameters
    ----------
    x : np.ndarray
        Raw values sampled from a uniform interval.
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.

    Returns
    -------
    np.ndarray
        Values mapped to the Legendre polynomial domain.
    """

    return 2.0 * (x - lower) / (upper - lower) - 1.0


def normalization_n(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize a Gaussian variable to standard-normal coordinates.

    Parameters
    ----------
    x : np.ndarray
        Raw Gaussian samples.
    mean : float
        Mean of the Gaussian variable.
    std : float
        Standard deviation of the Gaussian variable.

    Returns
    -------
    np.ndarray
        Values mapped to the Hermite polynomial domain.
    """

    scale = std if std > 0 else 1.0
    return (x - mean) / scale


def normalization_lognormal(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize a lognormal variable to its underlying Gaussian coordinates.

    Parameters
    ----------
    x : np.ndarray
        Positive samples drawn as ``exp(mean + std * Z)``.
    mean : float
        Mean of the underlying Gaussian variable.
    std : float
        Standard deviation of the underlying Gaussian variable.

    Returns
    -------
    np.ndarray
        Values mapped to standard-normal coordinates.
    """

    scale = std if std > 0 else 1.0
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr <= 0):
        raise ValueError("Lognormal variables must be positive before normalization.")
    return (np.log(x_arr) - mean) / scale


def normalization_all(X: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Normalize every input dimension based on the distribution matrix.

    Parameters
    ----------
    X : np.ndarray
        Raw input samples in the physical variable space. Supported shapes are
        ``(n_samples, n_features)`` and ``(n_features,)`` for a single sample.
    poly : np.ndarray
        Distribution matrix with shape ``(n_features, 3)``. Column 0 is the
        distribution code (``0`` uniform, ``1`` normal, ``2`` lognormal).
        Columns 1 and 2 store ``[lower_bound, upper_bound]`` for uniform
        variables or ``[mean, std]`` for normal/lognormal variables.

    Returns
    -------
    np.ndarray
        Normalized samples with shape ``(n_samples, n_features)`` suitable for
        basis evaluation.
    """

    x_arr = np.asarray(X, dtype=float)
    poly_arr = np.asarray(poly, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if poly_arr.ndim == 1:
        poly_arr = poly_arr.reshape(1, -1)

    normalized = np.zeros_like(x_arr, dtype=float)
    for idx in range(x_arr.shape[1]):
        dist_type, a, b = poly_arr[idx]
        if int(dist_type) == 0:
            normalized[:, idx] = normalization_u(x_arr[:, idx], a, b)
        elif int(dist_type) == 1:
            normalized[:, idx] = normalization_n(x_arr[:, idx], a, b)
        elif int(dist_type) == 2:
            normalized[:, idx] = normalization_lognormal(x_arr[:, idx], a, b)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return normalized


def _fit_ols_press_diagnostics(design, target):
    """Return SVD OLS coefficients, raw PRESS MSE, and numerical rank.

    The columns are the *fitted support*, including retained zero coefficients.
    Rank-deficient fits and numerically undefined leave-one-out fits are invalid
    (infinite error); neither leverage denominators nor the MSE are clipped.
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
        return None, np.inf, rank, np.full(target.size, np.nan)
    try:
        u_matrix, singular_values, vt_matrix = np.linalg.svd(design, full_matrices=False)
        tolerance = np.finfo(float).eps * max(design.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        if rank < design.shape[1] or design.shape[0] <= design.shape[1]:
            return None, np.inf, rank, np.full(target.size, np.nan)

        leverage = np.sum(u_matrix**2, axis=1)
        denominator = 1.0 - leverage
        leverage_tolerance = np.finfo(float).eps * max(design.shape)
        if np.any(denominator <= leverage_tolerance):
            return None, np.inf, rank, np.full(target.size, np.nan)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            ols_coef = (vt_matrix.T / singular_values) @ (u_matrix.T @ target)
            residuals = target - design @ ols_coef
            loo_errors = residuals / denominator
            if not np.all(np.isfinite(ols_coef)) or not np.all(np.isfinite(loo_errors)):
                return None, np.inf, rank, np.full(target.size, np.nan)
            scale = float(np.max(np.abs(loo_errors)))
            if scale == 0.0:
                loo_cv = 0.0
            else:
                # Scale before squaring to avoid avoidable overflow/underflow.
                rms = scale * np.sqrt(np.mean((loo_errors / scale) ** 2))
                loo_cv = float(rms * rms)
                # A nonzero residual must never be scored as an exact fit.
                if loo_cv == 0.0 or not np.isfinite(loo_cv):
                    return None, np.inf, rank, np.full(target.size, np.nan)
        return ols_coef, loo_cv, rank, loo_errors
    except np.linalg.LinAlgError:
        return None, np.inf, rank, np.full(target.size, np.nan)


def _fit_ols_press(design, target):
    """Compatibility API: coefficients, raw PRESS MSE, rank."""
    return _fit_ols_press_diagnostics(design, target)[:3]


def calculate_loocv(X: np.ndarray, Y: np.ndarray, coef: np.ndarray) -> float:
    """Compute leave-one-out cross-validation mean squared error.

    Parameters
    ----------
    X : np.ndarray
        Full polynomial-chaos design matrix with shape
        ``(n_samples, n_basis)``.
    Y : np.ndarray
        Target vector aligned with ``X`` and shape ``(n_samples,)`` or
        ``(n_samples, 1)``.
    coef : np.ndarray
        Sparse coefficient vector with shape ``(n_basis,)`` defining the
        active basis terms.

    Returns
    -------
    float
        Raw leave-one-out mean squared error, or infinity for an invalid fit.
        This legacy entry point infers support from coefficients and the
        intercept. Solvers with explicit fitted support reuse their scored MSE.
    """

    constant_idx = 0
    nonzero_indices = np.where(np.abs(coef) > 0)[0]
    selected = np.unique(np.append(constant_idx, nonzero_indices))

    _, loo_cv, _ = _fit_ols_press(X[:, selected], Y)
    return loo_cv


def calculate_pce_mean_std(coef: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the mean and standard deviation implied by a PCE model.

    Parameters
    ----------
    coef : np.ndarray
        PCE coefficient vector with shape ``(n_basis,)``.
    gamma : np.ndarray
        Norm factors associated with the basis functions, with shape
        ``(n_basis,)``.

    Returns
    -------
    tuple[np.ndarray, float]
        Mean term and standard deviation implied by the non-constant terms.
        The mean is the constant coefficient and the standard deviation is
        computed from the remaining coefficients and ``gamma``.
    """

    coef_arr = np.asarray(coef, dtype=float).reshape(-1)
    gamma_arr = np.asarray(gamma, dtype=float).reshape(-1)
    mean = coef_arr[0] if coef_arr.size > 1 else coef_arr
    variance = float(np.sum((coef_arr[1:] ** 2) * gamma_arr[1:])) if coef_arr.size > 1 else 0.0
    return mean, math.sqrt(max(variance, 0.0))


def construct_matrix_optimized(mapping: dict[str, str], X: np.ndarray) -> np.ndarray:
    """Rebuild a PCE design matrix from a basis-expression mapping.

    Parameters
    ----------
    mapping : dict[str, str]
        Mapping from basis identifiers to symbolic expressions. Its length is
        the number of basis terms ``n_basis``.
    X : np.ndarray
        Normalized inputs arranged as ``(n_features, n_samples)``. This is the
        transposed form of the public raw-input convention
        ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Design matrix with shape ``(n_samples, n_basis)``.
    """

    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] > 1 and x_arr.shape[1] == 1:
            x_arr = x_arr.T

    n_features, n_samples = x_arr.shape
    design = np.ones((len(mapping), n_samples), dtype=float)

    poly_pattern = re.compile(r"(legendre|hermite)_polynomial\(X\[(\d+)\],\s*(\d+)\)")
    linear_pattern = re.compile(r"^X\[(\d+)\]$")

    for row_idx, key in enumerate(mapping):
        expression = str(mapping[key]).strip()
        if expression in {"1", "1.0"}:
            continue

        for term in expression.split("*"):
            token = term.strip()
            while token.startswith("(") and token.endswith(")"):
                token = token[1:-1].strip()
            poly_match = poly_pattern.fullmatch(token)
            if poly_match:
                family, feature_idx, degree = poly_match.groups()
                feature_idx_int = int(feature_idx)
                if feature_idx_int >= n_features:
                    raise IndexError(
                        f"Mapping requests X[{feature_idx_int}] but only {n_features} features are available."
                    )
                values = x_arr[feature_idx_int]
                degree_int = int(degree)
                if family == "legendre":
                    design[row_idx] *= legendre_polynomial(values, degree_int)
                else:
                    design[row_idx] *= hermite_polynomial(values, degree_int)
                continue

            linear_match = linear_pattern.fullmatch(token)
            if linear_match:
                design[row_idx] *= x_arr[int(linear_match.group(1))]
                continue

            context = {
                "X": x_arr,
                "legendre_polynomial": legendre_polynomial,
                "hermite_polynomial": hermite_polynomial,
            }
            design[row_idx] *= eval(token, {}, context)

    return design.T


__all__ = [
    "calculate_loocv",
    "calculate_pce_mean_std",
    "construct_matrix_optimized",
    "get_array",
    "hermite_polynomial",
    "legendre_polynomial",
    "normalization_all",
    "normalization_lognormal",
]
