import numpy as np
from sklearn.linear_model import LassoLarsCV
from .legacy_base import LegacyNativeSolver as BasePCESolver


class LarsCVSolver(BasePCESolver):
    """
    LARS with cross-validation, implemented with sklearn.linear_model.LassoLarsCV.
    Select the sparsity of a polynomial-chaos expansion along the regularization path.
    """

    def __init__(
        self,
        poly_matrix=None,
        max_degree=3,
        cv=10,
        n_jobs=-1,
        max_n_alphas=200,
        eps=1e-6,
        max_iter=500,
        fit_intercept=False,
        normalize=False,
        last_model=None,
        q=1,
        **kwargs,
    ):
        """
        Initialize the LARS-CV solver.

        Parameters
        ----------
        poly_matrix : array-like
            Input-distribution specification used to build the polynomial basis.
        max_degree : int
            Maximum polynomial degree.
        cv : int, default=10
            Number of cross-validation folds.
        n_jobs : int, default=-1
            Number of parallel jobs. Set to -1 to use all processors.
        max_n_alphas : int, default=200
            Constructor setting. Each fit uses a fixed limit of 100 alpha values.
        eps : float, default=1e-6
            Cholesky diagonal regularization for numerical stability.
        max_iter : int, default=500
            Maximum number of path iterations.
        fit_intercept : bool, default=False
            Fit a separate intercept. The PCE basis already contains a constant term.
        normalize : bool, default=False
            Stored for compatibility. The estimator uses the supplied design matrix.
        """
        super().__init__(poly_matrix, max_degree, last_model=last_model, q=q)
        self.poly_matrix = poly_matrix
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_n_alphas = max_n_alphas
        self.eps = eps
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.poly_matrix = poly_matrix
        self.max_degree = max_degree
        self.last_model = last_model
        self.q = q

    def train_model(self, X_data, y):
        """Fit LassoLarsCV and select the regularization strength by cross-validation."""
        print("Fitting with LarsCVSolver.")
        n_samples, n_features = X_data.shape
        self.max_n_alphas = 100
        self.lars_cv_model = LassoLarsCV(
            cv=self.cv,
            max_iter=self.max_iter,
            max_n_alphas=self.max_n_alphas,
            eps=self.eps,
            fit_intercept=self.fit_intercept,
            n_jobs=self.n_jobs,
            verbose=False,
        )

        self.lars_cv_model.fit(X_data, y.ravel())

        self.coefficients_ = self.lars_cv_model.coef_

        self.best_alpha_ = self.lars_cv_model.alpha_

        self.active_set_ = np.where(np.abs(self.coefficients_) > 1e-10)[0]

        self.alphas_path_ = self.lars_cv_model.alphas_
        self.mse_path_ = self.lars_cv_model.mse_path_
        non_zero_mask = np.abs(self.coefficients_) > 0
        coefficients1 = self.coefficients_[non_zero_mask]

        return self.coefficients_, coefficients1

    def get_sparsity_ratio(self):
        """Return the fraction of active polynomial terms."""
        if hasattr(self, "active_set_"):
            n_total = len(self.coefficients_)
            if n_total == 0:
                return 0.0
            n_active = len(self.active_set_)
            return n_active / n_total
        return 1.0

    def get_active_features(self):
        """Return active feature indices."""
        if hasattr(self, "active_set_"):
            return self.active_set_.copy()
        return np.array([], dtype=int)

    def get_best_alpha(self):
        """Return the alpha value selected by cross-validation."""
        if hasattr(self, "best_alpha_"):
            return self.best_alpha_
        return None

    def get_model_statistics(self, X_data, y):
        """Return model statistics, including cross-validation results when available."""
        if not hasattr(self, "coefficients_"):
            return {}

        n_samples, n_features = X_data.shape
        n_active = len(self.active_set_)

        y_pred = self.lars_cv_model.predict(X_data)

        rss = np.sum((y - y_pred) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - rss / tss if tss > 0 else 0
        mse = rss / n_samples

        stats = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_active_features": n_active,
            "sparsity_ratio": n_active / n_features if n_features > 0 else 0,
            "r_squared": r_squared,
            "rss": rss,
            "mse": mse,
            "best_alpha": self.best_alpha_,
        }

        if hasattr(self, "mse_path_"):
            stats["min_cv_mse"] = np.min(np.mean(self.mse_path_, axis=1))

        return stats
