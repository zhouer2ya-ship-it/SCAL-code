"""Only the gradient and candidate-feasibility dependencies used by SCAL."""

from __future__ import annotations
import re
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_hermitenorm, eval_legendre, eval_jacobi
from ..coordinates import TargetGeometry


def stable_norm(values, axis=-1, keepdims=False):
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values), axis=axis, keepdims=True, initial=0.0)
    unit = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
    result = scale * np.sqrt(np.sum(unit * unit, axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


class CandidateFeasibility:
    """One tree and one fixed radius per batch; increment distances after each pick."""

    def __init__(self, training_z, candidate_z, ratio=0.1):
        self.z = np.asarray(candidate_z, dtype=float)
        training = np.asarray(training_z, dtype=float).reshape(-1, self.z.shape[1])
        if not np.all(np.isfinite(self.z)) or not np.all(np.isfinite(training)):
            raise ValueError("Non-finite sampling coordinates")
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError("min_distance_ratio must be finite and nonnegative")
        # Roundoff tolerance expressed only in dimensionless coordinates.
        magnitude = max(
            1.0, np.max(np.abs(training), initial=0.0), np.max(np.abs(self.z), initial=0.0)
        )
        self.duplicate_tolerance = 32 * np.finfo(float).eps * magnitude * np.sqrt(self.z.shape[1])
        unique = np.unique(training, axis=0)
        if len(unique) > 1:
            # Remove near-duplicates deterministically before estimating spacing.
            tree = cKDTree(unique)
            keep = np.ones(len(unique), dtype=bool)
            for i in range(len(unique)):
                if keep[i]:
                    neighbors = tree.query_ball_point(unique[i], self.duplicate_tolerance)
                    keep[[j for j in neighbors if j > i]] = False
            unique = unique[keep]
        self.tree = cKDTree(unique) if len(unique) else None
        nearest = (
            self.tree.query(unique, k=2, workers=1)[0][:, 1] if len(unique) > 1 else np.zeros(1)
        )
        self.stats = dict(
            mean=float(nearest.mean()), min=float(nearest.min()), max=float(nearest.max())
        )
        self.radius = ratio * self.stats["mean"]
        self.distances = (
            self.tree.query(self.z, k=1, workers=1)[0]
            if self.tree
            else np.full(len(self.z), np.inf)
        )
        self.selected = np.zeros(len(self.z), dtype=bool)

    @property
    def mask(self):
        return (
            (~self.selected)
            & (self.distances >= self.radius)
            & (self.distances > self.duplicate_tolerance)
        )

    def add(self, index):
        if not self.mask[index]:
            raise ValueError("Selected candidate violates the shared feasibility mask")
        self.selected[index] = True
        np.minimum(self.distances, stable_norm(self.z - self.z[index]), out=self.distances)


def compute_gradient_vectors(
    X, model, batch_size=10000, sparse_threshold=0.0, input_transform=None, geometry=None
):
    """Return df/dz, applying the full physical-to-model Jacobian."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    support = getattr(model, "support_indices_", None)
    support = np.arange(len(model.coef_)) if support is None else np.asarray(support, dtype=int)
    indices = support[np.abs(np.asarray(model.coef_)[support]) > sparse_threshold]
    if not len(indices):
        return np.zeros_like(X)
    if geometry is None:
        geometry = TargetGeometry.from_poly(
            input_transform.target_poly if input_transform is not None else model.poly_matrix
        )
    names = list(model.mapping_)
    relevant = [names[i] for i in indices]
    gradients = []
    for start in range(0, len(X), batch_size):
        physical = X[start : start + batch_size]
        inputs = physical if input_transform is None else input_transform.transform(physical)
        raw = process_batch(
            inputs, model.poly_matrix, relevant, np.asarray(model.coef_)[indices], sparse_threshold
        )
        if input_transform is None:
            raw *= geometry.scale
        else:
            jacobian = input_transform.jacobian(physical) * geometry.scale
            raw = raw @ jacobian if jacobian.ndim == 2 else np.einsum("ni,nij->nj", raw, jacobian)
        gradients.append(raw)
    return np.vstack(gradients) if gradients else np.zeros_like(X)


def compute_gradient_importance(X, model, batch_size=10000, sparse_threshold=0.0, **kwargs):
    return calculate_gradient_norm(
        compute_gradient_vectors(X, model, batch_size, sparse_threshold, **kwargs)
    )


def process_batch(X_batch, poly, relevant_basis, relevant_coef, sparse_threshold):
    """Compute surrogate gradients for a subset of candidate points.

    Parameters
    ----------
    X_batch : np.ndarray
        Candidate points to evaluate in the current batch, with shape
        ``(n_batch_samples, n_features)``.
    poly : np.ndarray
        Distribution matrix with shape ``(n_features, 3)`` used to normalize
        the inputs. Column 0 is the distribution code (``0`` uniform,
        ``1`` normal, ``2`` lognormal). Columns 1 and 2 store
        ``[lower_bound, upper_bound]`` for uniform variables or ``[mean, std]``
        for normal/lognormal variables.
    relevant_basis : list[str]
        Names of basis terms with non-negligible coefficients.
    relevant_coef : np.ndarray
        Coefficients associated with ``relevant_basis``, with shape
        ``(n_active_basis,)``.
    sparse_threshold : float
        Minimum absolute coefficient magnitude treated as active.

    Returns
    -------
    np.ndarray
        Gradient matrix with shape ``(n_batch_samples, n_features)``.
    """

    if X_batch.size == 0:
        return np.zeros((0, poly.shape[0]), dtype=float)
    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(1, -1)

    n_samples, n_vars = X_batch.shape
    normalized = normalization_all(X_batch, poly)
    gradients = np.zeros((n_samples, n_vars), dtype=float)
    chain_factors = compute_chain_factors(poly, X_batch)

    basis_structures = {}
    for basis_name in relevant_basis:
        if basis_name == "x00":
            continue
        parsed = parse_basis_structure(basis_name)
        if parsed:
            basis_structures[basis_name] = parsed

    max_degrees = np.zeros(n_vars, dtype=int)
    for pairs in basis_structures.values():
        for idx, degree in pairs:
            max_degrees[idx] = max(max_degrees[idx], degree)

    basis_values, basis_derivatives = precompute_basis_values(normalized, poly, max_degrees)

    for basis_name, coefficient in zip(relevant_basis, relevant_coef):
        if basis_name == "x00" or abs(coefficient) < sparse_threshold:
            continue
        pairs = basis_structures.get(basis_name, [])
        if not pairs:
            continue

        for var_idx, degree in pairs:
            derivative = basis_derivatives[var_idx].get(degree, np.zeros(n_samples))
            remainder = np.ones(n_samples, dtype=float)
            skipped_current_factor = False
            for other_idx, other_degree in pairs:
                if other_idx == var_idx and other_degree == degree and not skipped_current_factor:
                    skipped_current_factor = True
                    continue
                remainder *= basis_values[other_idx].get(
                    other_degree, np.ones(n_samples, dtype=float)
                )
            gradients[:, var_idx] += (
                coefficient * chain_factors[:, var_idx] * derivative * remainder
            )

    return gradients


def compute_chain_factors(poly, X=None):
    """Compute input-normalization derivatives for each variable.

    Parameters
    ----------
    poly : np.ndarray
        Distribution matrix with shape ``(n_features, 3)``. Column 0 is the
        distribution code (``0`` uniform, ``1`` normal, ``2`` lognormal).
        Columns 1 and 2 store ``[lower_bound, upper_bound]`` for uniform
        variables or ``[mean, std]`` for normal/lognormal variables.
    X : np.ndarray | None, default=None
        Physical samples used by lognormal variables to compute the
        ``d log(x) / dx`` chain-rule factor.

    Returns
    -------
    np.ndarray
        Multiplicative chain-rule factors that convert derivatives from
        normalized coordinates back to physical coordinates.
    """

    if X is None:
        X = np.ones((1, poly.shape[0]), dtype=float)
    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)

    factors = np.zeros((x_arr.shape[0], poly.shape[0]), dtype=float)
    for idx in range(poly.shape[0]):
        dist_type, a, b = poly[idx, 0], poly[idx, 1], poly[idx, 2]
        scale = b if b > 0 else 1.0
        if int(dist_type) == 0:
            factors[:, idx] = 2.0 / (b - a)
        elif int(dist_type) == 1:
            factors[:, idx] = 1.0 / scale
        elif int(dist_type) == 2:
            if np.any(x_arr[:, idx] <= 0):
                raise ValueError("Lognormal variables must be positive for gradient computation.")
            factors[:, idx] = 1.0 / (scale * x_arr[:, idx])
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return factors


def calculate_gradient_norm(gradients):
    """Collapse per-dimension gradients to Euclidean norms.

    Parameters
    ----------
    gradients : np.ndarray
        Gradient matrix with one row per sample.

    Returns
    -------
    np.ndarray
        Column vector of gradient magnitudes.
    """

    return stable_norm(gradients, axis=1).reshape(-1, 1)


def parse_basis_structure(name):
    """Parse a basis name into ``(variable, degree)`` pairs.

    Parameters
    ----------
    name : str
        Basis identifier such as ``"X0_2*X3_1"``.

    Returns
    -------
    list[tuple[int, int]]
        Variable indices and polynomial degrees encoded in ``name``.
    """

    pairs = []
    for part in name.split("*") if "*" in name else [name]:
        match = re.match(r"X(\d+)_(\d+)", part)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    return pairs


def precompute_basis_values(X, poly, max_degrees):
    """Precompute basis values and derivatives required for gradient assembly.

    Parameters
    ----------
    X : np.ndarray
        Normalized candidate inputs with shape
        ``(n_batch_samples, n_features)``.
    poly : np.ndarray
        Distribution matrix with shape ``(n_features, 3)``. Column 0 selects
        the orthogonal-polynomial family: ``0`` for Legendre (uniform inputs)
        and ``1``/``2`` for Hermite (normal/lognormal inputs).
    max_degrees : np.ndarray
        Maximum polynomial degree needed per variable, with shape
        ``(n_features,)``.

    Returns
    -------
    tuple[list[dict[int, np.ndarray]], list[dict[int, np.ndarray]]]
        Cached polynomial values and derivatives indexed by variable and degree.
    """

    n_samples, n_vars = X.shape
    values = [{} for _ in range(n_vars)]
    derivatives = [{} for _ in range(n_vars)]
    for idx in range(n_vars):
        values[idx][0] = np.ones(n_samples, dtype=float)
        derivatives[idx][0] = np.zeros(n_samples, dtype=float)
        for degree in range(1, max_degrees[idx] + 1):
            if int(poly[idx, 0]) == 0:
                values[idx][degree] = eval_legendre(degree, X[:, idx])
                derivatives[idx][degree] = legendre_derivative(X[:, idx], degree)
            elif int(poly[idx, 0]) in (1, 2):
                # Prediction uses probabilists' Hermite: He_n' = n He_(n-1).
                values[idx][degree] = eval_hermitenorm(degree, X[:, idx])
                derivatives[idx][degree] = degree * eval_hermitenorm(degree - 1, X[:, idx])
            else:
                raise ValueError(f"Unsupported distribution type: {poly[idx, 0]}")
    return values, derivatives


def legendre_derivative(x, degree):
    """Evaluate the derivative of a Legendre polynomial.

    Parameters
    ----------
    x : np.ndarray
        Evaluation coordinates in ``[-1, 1]``.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Derivative values evaluated at ``x``.
    """

    if degree == 0:
        return np.zeros_like(x, dtype=float)
    return (degree + 1.0) / 2.0 * eval_jacobi(degree - 1, 1.0, 1.0, x)


def normalization_all(X, poly):
    """Normalize raw inputs according to the polynomial distribution matrix.

    Parameters
    ----------
    X : np.ndarray
        Raw input samples in the physical variable space. Supported shapes are
        ``(n_samples, n_features)`` and ``(n_features,)`` for a single sample.
    poly : np.ndarray
        Distribution matrix with shape ``(n_features, 3)``. Column 0 is the
        distribution code: ``0`` denotes a uniform variable, ``1`` denotes a
        Gaussian variable, and ``2`` denotes a lognormal variable. Columns 1
        and 2 store ``[lower_bound, upper_bound]`` or ``[mean, std]``.

    Returns
    -------
    np.ndarray
        Normalized samples compatible with the orthogonal polynomial basis,
        with shape ``(n_samples, n_features)``.
    """

    x_arr = np.asarray(X, dtype=float)
    if x_arr.size == 0:
        return x_arr
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)

    normalized = np.zeros_like(x_arr, dtype=float)
    for idx in range(x_arr.shape[1]):
        dist_type, a, b = poly[idx, 0], poly[idx, 1], poly[idx, 2]
        if int(dist_type) == 0:
            normalized[:, idx] = 2.0 * (x_arr[:, idx] - a) / (b - a) - 1.0
        elif int(dist_type) == 1:
            normalized[:, idx] = (x_arr[:, idx] - a) / (b if b > 0 else 1.0)
        elif int(dist_type) == 2:
            if np.any(x_arr[:, idx] <= 0):
                raise ValueError("Lognormal variables must be positive before normalization.")
            normalized[:, idx] = (np.log(x_arr[:, idx]) - a) / (b if b > 0 else 1.0)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return normalized
