"""Fixed target-distribution coordinates, independent of the adaptive design.

For lognormal rows (2, a, b), log(X) ~ Normal(a, b**2).
Deterministic variables must be removed from the random input dimensions.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, qmc
from sklearn.decomposition import PCA


def target_location_scale(poly):
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 3 or not np.all(np.isfinite(poly)):
        raise ValueError("poly_matrix must be a finite (dimension, 3) matrix")
    center, scale = np.empty(len(poly)), np.empty(len(poly))
    for j, (kind, a, b) in enumerate(poly):
        if kind == 0:
            center[j], scale[j] = a / 2 + b / 2, (b - a) / np.sqrt(12.0)
        elif kind == 1:
            center[j], scale[j] = a, b
        elif kind == 2:
            if b <= 0:
                raise ValueError("Lognormal log-standard-deviation must be positive")
            with np.errstate(over="ignore", invalid="ignore"):
                center[j] = np.exp(a + b * b / 2)
                scale[j] = center[j] * np.sqrt(np.expm1(b * b))
        else:
            raise ValueError(f"Unsupported distribution code: {kind}")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("Target scales must be finite and positive; separate deterministic inputs")
    return center, scale


def from_unit(U, poly):
    target_location_scale(poly)
    U = np.asarray(U, dtype=float)
    if np.any(~np.isfinite(U)) or np.any((U < 0) | (U > 1)):
        raise ValueError("Unit coordinates must be finite and in [0, 1]")
    X = np.empty_like(U)
    for j, (kind, a, b) in enumerate(poly):
        if kind == 0:
            X[:, j] = a + (b - a) * U[:, j]
        else:
            # Endpoint clipping is explicit and limited to representable probabilities.
            u = np.clip(U[:, j], np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            X[:, j] = a + b * norm.ppf(u)
            if kind == 2:
                X[:, j] = np.exp(X[:, j])
    return X


def to_unit(X, poly):
    target_location_scale(poly)
    X = np.asarray(X, dtype=float)
    U = np.empty_like(X)
    for j, (kind, a, b) in enumerate(poly):
        if kind == 0:
            U[:, j] = (X[:, j] - a) / (b - a)
        elif kind == 1:
            U[:, j] = norm.cdf((X[:, j] - a) / b)
        else:
            if np.any(X[:, j] <= 0):
                raise ValueError("Lognormal inputs must be positive")
            U[:, j] = norm.cdf((np.log(X[:, j]) - a) / b)
    return U


@dataclass
class TargetGeometry:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_poly(cls, poly):
        return cls(*target_location_scale(poly))

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.center) / self.scale

    def jacobian(self, X=None):
        return np.diag(1 / self.scale)


class FixedInputTransform:
    """PCA in full standardized target coordinates, fitted once on input-only Sobol data."""

    def __init__(self, poly, n_components=None, seed=42, reference_power=12):
        self.target_poly = np.asarray(poly, dtype=float).copy()
        self.geometry = TargetGeometry.from_poly(poly)
        d = len(poly)
        k = d if n_components is None else int(n_components)
        if not 1 <= k <= d:
            raise ValueError("input_pca_components must be in [1, n_features]")
        self.seed, self.reference_power = (
            int(seed),
            max(int(reference_power), int(np.ceil(np.log2(d + 1)))),
        )
        reference = from_unit(
            qmc.Sobol(d, scramble=True, seed=self.seed).random_base2(self.reference_power), poly
        )
        pca = PCA(n_components=k, svd_solver="full", whiten=False).fit(
            self.geometry.transform(reference)
        )
        self.components_ = pca.components_.copy()
        self.mean_ = pca.mean_.copy()
        self.n_components_ = k
        self.explained_variance_ratio_ = pca.explained_variance_ratio_.copy()
        self.explained_variance_ = pca.explained_variance_.copy()
        self.analytic_orthogonal = bool(np.all(self.target_poly[:, 0] == 1))
        # Independent physical normals give exactly Cov(z)=I, not empirical PCA variance.
        if self.analytic_orthogonal:
            mu = -self.components_ @ self.mean_
            sd = np.ones(k)
        else:
            mu = np.zeros(k)
            sd = np.sqrt(pca.explained_variance_)
        self.feature_poly = np.column_stack([np.ones(k), mu, sd])

    def transform(self, X):
        return (self.geometry.transform(X) - self.mean_) @ self.components_.T

    def jacobian(self, X=None):
        """du/dx; rows are latent coordinates, columns physical coordinates."""
        return self.components_ / self.geometry.scale[None, :]

    def jacobian_z(self, X=None):
        return self.components_


class WeightedOutputPCA:
    """Full SVD with numerical-rank screening and an explicit physical reconstruction."""

    def __init__(self, Y, weights, n_components=None):
        Y = np.asarray(Y, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.mean_ = Y[0] + np.mean(Y - Y[0], axis=0)
        centered = (Y - self.mean_) * np.sqrt(self.weights)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        self.rank_tolerance_ = (
            float(singular[0] * max(Y.shape) * np.finfo(float).eps) if singular.size else 0.0
        )
        self.numerical_rank_ = int(np.sum(singular > self.rank_tolerance_))
        if n_components is not None and int(n_components) < 1:
            raise ValueError("pca_components must be positive")
        k = (
            self.numerical_rank_
            if n_components is None
            else min(int(n_components), self.numerical_rank_)
        )
        self.components_ = vt[:k]
        self.n_components_ = k
        self.reconstruction_ = self.components_.T / np.sqrt(self.weights)[:, None]
        self.explained_variance_ratio_ = (
            singular[:k] ** 2 / np.sum(singular**2) if k else np.empty(0)
        )

    def transform(self, Y):
        return ((np.asarray(Y) - self.mean_) * np.sqrt(self.weights)) @ self.components_.T

    def inverse_transform(self, T):
        return np.asarray(T) @ self.reconstruction_.T + self.mean_
