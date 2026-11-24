"""Basis construction utilities for sparse polynomial-chaos expansions."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from .utils import (
    construct_matrix_optimized,
    get_array,
    hermite_polynomial,
    legendre_polynomial,
    normalization_all,
)


def resolve_q(n_variables, q):
    """Resolve the existing dimension-dependent truncation rule in one place."""
    value = (
        (
            1.0
            if n_variables < 5
            else 0.9
            if n_variables < 10
            else 0.8
            if n_variables < 20
            else 0.7
            if n_variables < 50
            else 0.5
        )
        if q is None
        else float(q)
    )
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("q must be in (0, 1]")
    return value


def basis_definition(poly_params, max_degree, q, *, variable_ids=None, transform_identity=None):
    """Exact, serializable basis identity; input rows always retain their order.

    Without variable_ids, callers remain responsible for physical column order.
    A model's saved definition describes the fitted basis, even if its public
    constructor parameters are subsequently edited.
    """
    poly = np.asarray(poly_params, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 3 or not np.all(np.isfinite(poly)):
        raise ValueError("poly_matrix must be a finite (n_features, 3) matrix")
    if int(max_degree) != max_degree or max_degree < 0:
        raise ValueError("max_degree must be a nonnegative integer")
    return dict(
        n_variables=len(poly),
        poly_matrix=poly.tolist(),
        max_degree=int(max_degree),
        q=resolve_q(len(poly), q),
        max_interaction=min(3 if len(poly) < 20 else 2, len(poly)),
        convention="AUQ_legendre_hermite_physical_norms_v1",
        variable_ids=None if variable_ids is None else list(variable_ids),
        transform_identity=transform_identity,
    )


def model_basis_definition(model):
    """Read fitted metadata, or reconstruct a fully specified legacy definition."""
    if model is None:
        return None
    saved = getattr(model, "basis_definition_", None)
    if saved is not None:
        required = set(basis_definition([[0.0, -1.0, 1.0]], 1, 1.0))
        return saved if isinstance(saved, dict) and required <= saved.keys() else None
    if not all(hasattr(model, name) for name in ("poly_matrix", "max_degree", "q")):
        return None
    try:
        return basis_definition(
            model.poly_matrix,
            model.max_degree,
            model.q,
            variable_ids=getattr(model, "variable_ids_", None),
            transform_identity=getattr(model, "input_transform_identity_", None),
        )
    except (ValueError, TypeError, OverflowError):
        return None


def same_basis_coordinates(left, right):
    if left is None or right is None:
        return False
    fields = ("n_variables", "poly_matrix", "convention", "variable_ids", "transform_identity")
    return all(left.get(key) == right.get(key) for key in fields)


@lru_cache(maxsize=128)
def count_hyperbolic_terms(d, p, q):
    """Count the actual dictionary with the same interaction and boundary rule."""
    q = resolve_q(d, q)
    if int(p) != p or p < 0:
        raise ValueError("max_degree must be a nonnegative integer")
    limit = min(3 if d < 20 else 2, d)
    powers = np.power(np.arange(p + 1), q)
    return 1 + sum(
        math.comb(d, k)
        * sum(
            float(sum(powers[a] for a in degrees)) <= float(p**q) + 1e-9
            for degrees in product(range(1, p + 1), repeat=k)
        )
        for k in range(1, limit + 1)
    )


@lru_cache(maxsize=64)
def _basis_catalog(poly_rows, degree, q):
    poly = np.asarray(poly_rows, dtype=float)
    # Only expressions and norms are retained, but the dummy physical input
    # still passes through normalization. A lognormal coordinate must be positive.
    probe = np.zeros((len(poly), 1))
    lognormal = poly[:, 0] == 2
    probe[lognormal, 0] = np.exp(poly[lognormal, 1])
    _, mapping, gamma = get_hyperbolic_basis_direct(probe, poly, degree, None, q)
    return mapping, dict(zip(mapping, gamma))


def reusable_basis(model, definition):
    """Verify saved mapping expressions and norms, allowing aligned reordering."""
    if model is None or model_basis_definition(model) != definition:
        return False
    mapping = getattr(model, "mapping_", None)
    gamma = getattr(model, "gamma_", None)
    if (
        not isinstance(mapping, dict)
        or gamma is None
        or not mapping
        or next(iter(mapping)) != "x00"
    ):
        return False
    expected, norms = _basis_catalog(
        tuple(tuple(row) for row in definition["poly_matrix"]),
        definition["max_degree"],
        definition["q"],
    )
    return (
        mapping == expected
        and np.asarray(gamma).shape == (len(mapping),)
        and np.array_equal(gamma, [norms[name] for name in mapping])
    )


def get_hyperbolic_basis_direct(
    X: np.ndarray,
    poly_params: np.ndarray,
    max_degree: int,
    last_model,
    q,
    *,
    definition=None,
) -> tuple[np.ndarray, dict[str, str], np.ndarray]:
    """Build a hyperbolically truncated PCE basis.

    Parameters
    ----------
    X : np.ndarray
        Input samples arranged as ``(n_features, n_samples)``. This is the
        transposed convention of the public API, where raw samples are usually
        provided as ``(n_samples, n_features)``.
    poly_params : np.ndarray
        Distribution matrix with shape ``(n_features, 3)``. Column 0 is the
        distribution code (``0`` uniform, ``1`` normal, ``2`` lognormal).
        Columns 1 and 2 store ``[lower_bound, upper_bound]`` for uniform
        variables or ``[mean, std]`` for normal/lognormal variables.
    max_degree : int
        Maximum polynomial degree considered in each dimension.
    last_model : Any
        Previously fitted model that can be reused when the basis definition is
        unchanged.
    q : float | None
        Hyperbolic truncation exponent controlling interaction sparsity.

    Returns
    -------
    tuple[np.ndarray, dict[str, str], np.ndarray]
        Expanded design matrix with shape ``(n_samples, n_basis)``, symbolic
        basis mapping of length ``n_basis``, and basis norm factors with shape
        ``(n_basis,)``.
    """

    current_definition = definition or basis_definition(poly_params, max_degree, q)
    if reusable_basis(last_model, current_definition):
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        normalized = normalization_all(x_arr.T, poly_params)
        design = construct_matrix_optimized(last_model.mapping_, normalized.T)
        return design, dict(last_model.mapping_), np.asarray(last_model.gamma_).copy()

    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)

    n_variables, n_samples = x_arr.shape
    q = resolve_q(n_variables, q)

    normalized = normalization_all(x_arr.T, poly_params).T
    basis_cache: list[list[np.ndarray]] = []
    norm_cache: list[list[float]] = []

    for idx in range(n_variables):
        values = normalized[idx, :]
        poly_type = int(poly_params[idx, 0])

        variable_basis = [np.ones(n_samples, dtype=float)]
        variable_norms = [1.0]
        for degree in range(1, max_degree + 1):
            if poly_type == 0:
                variable_basis.append(legendre_polynomial(values, degree))
                variable_norms.append(1.0 / (2 * degree + 1))
            elif poly_type in (1, 2):
                variable_basis.append(hermite_polynomial(values, degree))
                variable_norms.append(float(math.factorial(degree)))
            else:
                raise ValueError(f"Unsupported distribution type: {poly_type}")

        basis_cache.append(variable_basis)
        norm_cache.append(variable_norms)

    final_basis: dict[str, np.ndarray] = {"x00": np.ones(n_samples, dtype=float)}
    expr_mapping: dict[str, str] = {"x00": "1"}
    gamma_dict: dict[str, float] = {"x00": 1.0}

    threshold = float(max_degree**q)
    power_lookup = np.power(np.arange(max_degree + 1), q)
    max_interaction = min(3 if n_variables < 20 else 2, n_variables)

    for interaction_size in range(1, max_interaction + 1):
        for variable_indices in combinations(range(n_variables), interaction_size):
            for degrees in product(range(1, max_degree + 1), repeat=interaction_size):
                if float(sum(power_lookup[degree] for degree in degrees)) > threshold + 1e-9:
                    continue

                key_parts = [
                    f"X{var_idx}_{degree}" for var_idx, degree in zip(variable_indices, degrees)
                ]
                basis_key = "*".join(key_parts)

                component_values = np.asarray(
                    [
                        basis_cache[var_idx][degree]
                        for var_idx, degree in zip(variable_indices, degrees)
                    ]
                )
                final_basis[basis_key] = np.prod(component_values, axis=0)

                gamma = 1.0
                expr_parts = []
                for var_idx, degree in zip(variable_indices, degrees):
                    gamma *= norm_cache[var_idx][degree]
                    dist_type = int(poly_params[var_idx, 0])
                    if dist_type == 0:
                        polynomial_name = "legendre"
                    elif dist_type in (1, 2):
                        polynomial_name = "hermite"
                    else:
                        raise ValueError(f"Unsupported distribution type: {dist_type}")
                    expr_parts.append(f"{polynomial_name}_polynomial(X[{var_idx}], {degree})")
                gamma_dict[basis_key] = gamma
                expr_mapping[basis_key] = "*".join(expr_parts)

    gamma_array = np.asarray(list(gamma_dict.values()), dtype=float)
    design_matrix = get_array(final_basis).T
    return design_matrix, expr_mapping, gamma_array


__all__ = ["get_hyperbolic_basis_direct"]
