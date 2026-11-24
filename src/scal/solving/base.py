"""Base abstractions shared by polynomial-chaos surrogate solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .get_basis import (
    get_hyperbolic_basis_direct,
    basis_definition,
    model_basis_definition,
    same_basis_coordinates,
    reusable_basis,
)
from .utils import (
    _fit_ols_press,
    _fit_ols_press_diagnostics,
    calculate_loocv,
    calculate_pce_mean_std,
    construct_matrix_optimized,
    normalization_all,
)


class BasePCESolver(ABC):
    """Shared interface for SCAL polynomial-chaos surrogate solvers.

    Attributes
    ----------
    poly_matrix : np.ndarray | None
        Distribution matrix with shape ``(n_features, 3)``. Each row describes
        one input variable. Column 0 is the distribution code: ``0`` means a
        uniform variable, ``1`` means a normal variable, and ``2`` means a
        lognormal variable. For uniform rows, columns 1 and 2 are
        ``[lower_bound, upper_bound]``. For normal and lognormal rows, columns
        1 and 2 are ``[mean, std]`` in Gaussian coordinates.
    max_degree : int
        Maximum polynomial degree used to build the basis.
    l1_ratio : float
        Elastic-net mixing ratio used by sparse solvers.
    last_model : BasePCESolver | None
        Previously fitted model used for warm starts or inherited basis reuse.
    q : float | None
        Hyperbolic-truncation exponent controlling interaction sparsity.
    X, Y : np.ndarray | None
        Raw training inputs and outputs used to fit the model. ``X`` has shape
        ``(n_samples, n_features)`` and ``Y`` has shape ``(n_samples, 1)``.
    X_pce : np.ndarray | None
        Expanded polynomial-chaos design matrix with shape
        ``(n_samples, n_basis)``.
    coef_ : np.ndarray | None
        Final coefficient vector with shape ``(n_basis,)`` in basis order.
    mapping_ : dict[str, str] | None
        Mapping from basis names to symbolic expressions. The mapping length is
        ``n_basis`` and matches the column order of ``X_pce`` and ``coef_``.
    gamma_ : np.ndarray | None
        Basis norm factors with shape ``(n_basis,)``.
    mean_, std_ : np.ndarray | float | None
        Mean and standard deviation implied by the fitted PCE.
    error_ : float | None
        Leave-one-out cross-validation error estimate.
    num_basis : int | None
        Number of active basis functions retained by the solver.
    n_features_in_ : int | None
        Number of input variables seen during fitting.
    is_fitted_ : bool
        Whether the solver has completed a successful fit.
    """

    def __init__(self, poly_matrix=None, max_degree=3, l1_ratio=0.95, last_model=None, q=None):
        """Initialize common solver configuration.

        Parameters
        ----------
        poly_matrix : np.ndarray | None, default=None
            Distribution matrix with shape ``(n_features, 3)``. Column 0 is
            the distribution code (``0`` uniform, ``1`` normal, ``2``
            lognormal). Columns 1 and 2 store ``[lower_bound, upper_bound]``
            for uniform variables or ``[mean, std]`` for normal/lognormal
            variables.
        max_degree : int, default=3
            Maximum polynomial degree allowed in the basis.
        l1_ratio : float, default=0.95
            Elastic-net mixing parameter used by sparse descendants.
        last_model : BasePCESolver | None, default=None
            Previously fitted model that may be reused as a warm start.
        q : float | None, default=None
            Hyperbolic truncation exponent. ``None`` selects the basis default.
        """

        self.poly_matrix = poly_matrix
        self.max_degree = max_degree
        self.l1_ratio = l1_ratio
        self.last_model = last_model
        self.q = q

        self.X = None
        self.Y = None
        self.X_pce = None
        self.coef_ = None
        self.mapping_ = None
        self.gamma_ = None
        self.mean_ = None
        self.std_ = None
        self.error_ = None
        self.num_basis = None
        self.n_features_in_ = None
        self.is_fitted_ = False
        self.support_indices_ = None
        self.error_name_ = "raw_PRESS_MSE"
        self.moment_method_ = "target_orthogonal_pce"

    def fit(self, X, y):
        """Fit the solver on raw samples and responses.

        Parameters
        ----------
        X : array-like
            Input sample matrix with shape ``(n_samples, n_features)`` in the
            original physical variable space.
        y : array-like
            Response values aligned with ``X``. Accepted shapes are
            ``(n_samples,)`` and ``(n_samples, 1)``.

        Returns
        -------
        BasePCESolver
            The fitted solver instance.
        """

        self.validate_data(X, y)
        self.basis_definition_ = basis_definition(
            self.poly_matrix,
            self.max_degree,
            self.q,
            variable_ids=getattr(self, "variable_ids_", None),
            transform_identity=getattr(self, "input_transform_identity_", None),
        )
        old_definition = model_basis_definition(self.last_model)
        self.basis_reused_ = reusable_basis(self.last_model, self.basis_definition_)
        self.warm_start_compatible_ = same_basis_coordinates(self.basis_definition_, old_definition)
        self.warm_start_used_ = False
        self.warm_start_reason_ = (
            "no_previous_model"
            if self.last_model is None
            else "missing_basis_metadata"
            if old_definition is None
            else "compatible_support_not_used"
            if self.warm_start_compatible_
            else "basis_coordinates_changed"
        )
        design, mapping, gamma = get_hyperbolic_basis_direct(
            np.asarray(X, dtype=float).T,
            self.poly_matrix,
            self.max_degree,
            self.last_model,
            self.q,
            definition=self.basis_definition_,
        )
        target = np.asarray(y, dtype=float).reshape(-1, 1)

        self.X = np.asarray(X, dtype=float)
        self.Y = target
        self.X_pce = design
        self.mapping_ = mapping
        self.gamma_ = gamma

        # Each fit records the solver's selection error and the raw fixed-support PRESS.
        # Reset fitted values before a new training round.
        self.error_ = None
        self.support_indices_ = None
        coef, basis_dict = self.train_model(design, target)
        coef = np.asarray(coef, dtype=float).reshape(-1)
        if coef.size != design.shape[1] or not np.all(np.isfinite(coef)):
            raise ValueError("Solver returned invalid coefficients")
        if getattr(self, "support_indices_", None) is None:
            self.support_indices_ = np.union1d([0], np.flatnonzero(coef))
        _, self.press_error_, self.press_rank_, self.press_residuals_ = _fit_ols_press_diagnostics(
            design[:, self.support_indices_], target
        )
        self.press_definition_ = "fixed_support_OLS_PRESS"
        mean, std = calculate_pce_mean_std(coef, gamma)
        error = self.error_ if self.error_ is not None else calculate_loocv(design, target, coef)

        self.coef_ = coef
        self.mean_ = mean
        self.std_ = std
        self.error_ = error
        self.num_basis = len(self.support_indices_)
        self.n_features_in_ = self.X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict responses for new raw input samples.

        Parameters
        ----------
        X : array-like
            Input samples in the original physical variable space. Accepted
            shapes are ``(n_features,)`` for one sample and
            ``(n_samples, n_features)`` for a batch.

        Returns
        -------
        np.ndarray
            Predicted responses with shape ``(n_samples,)``.
        """

        if not self.is_fitted_:
            raise RuntimeError("The solver must be fitted before calling predict().")

        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        normalized = normalization_all(x_arr, self.poly_matrix)
        design = construct_matrix_optimized(self.mapping_, normalized.T)
        return design @ self.coef_

    @abstractmethod
    def train_model(self, X, y):
        """Train on an already expanded PCE design matrix.

        Parameters
        ----------
        X : np.ndarray
            Expanded polynomial-chaos design matrix with shape
            ``(n_samples, n_basis)``.
        y : np.ndarray
            Target values aligned with ``X``. Expected shape is
            ``(n_samples,)`` or ``(n_samples, 1)``.

        Returns
        -------
        tuple[np.ndarray, dict[str, float]]
            Fitted coefficient vector with shape ``(n_basis,)`` and a
            dictionary describing the active basis terms.
        """

    def validate_data(self, X, y):
        """Validate sample and response shapes before fitting.

        Parameters
        ----------
        X : array-like
            Candidate input sample matrix with shape
            ``(n_samples, n_features)``.
        y : array-like
            Candidate response vector with shape ``(n_samples,)`` or
            ``(n_samples, 1)``.

        Raises
        ------
        ValueError
            Raised when the number of samples is inconsistent or too small.
        """

        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {x_arr.shape[0]} rows but y has {y_arr.shape[0]} values.")
        if x_arr.shape[0] < 2:
            raise ValueError("At least two samples are required to fit a PCE solver.")


def ols_wave(X, Y, coef):
    """Refit active coefficients with SVD-based ordinary least squares.

    Parameters
    ----------
    X : np.ndarray
        Design matrix containing all candidate basis columns, with shape
        ``(n_samples, n_basis)``.
    Y : np.ndarray
        Target values aligned with ``X`` and shape ``(n_samples,)`` or
        ``(n_samples, 1)``.
    coef : np.ndarray
        Sparse coefficient vector with shape ``(n_basis,)`` whose nonzero
        entries define the active set.

    Returns
    -------
    tuple[np.ndarray, float]
        Updated coefficient vector and the corresponding leave-one-out error.
    """

    selected = np.flatnonzero(np.abs(coef) > 0)
    if len(selected) == 0:
        return np.zeros_like(coef), np.inf

    ols_coef, loo_cv, _ = _fit_ols_press(X[:, selected], Y)
    if ols_coef is None:
        return coef, np.inf
    updated_coef = np.zeros_like(coef, dtype=float)
    updated_coef[selected] = ols_coef
    return updated_coef, loo_cv


__all__ = ["BasePCESolver", "ols_wave"]
