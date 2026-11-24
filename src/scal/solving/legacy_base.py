"""Common polynomial-chaos interface for the comparison solvers."""

import numpy as np
from sklearn.linear_model import LinearRegression
from .base import BasePCESolver


class LegacyNativeSolver(BasePCESolver):
    def fit(self, X, y):
        super().fit(X, y)
        # This diagnostic pairs each OLS residual with every
        # denominator (1 - h_jj) before averaging the N by N squared ratios.
        support = np.union1d([0], np.flatnonzero(self.coef_))
        design = self.X_pce[:, support]
        target = np.asarray(y, dtype=float).reshape(-1, 1)
        ols = LinearRegression(fit_intercept=False).fit(design, target)
        residual = target - ols.predict(design)
        H = design @ np.linalg.pinv(design.T @ design) @ design.T
        denominator = np.clip(1 - np.diag(H), 1e-12, None)
        self.error_ = float(np.mean((residual / denominator) ** 2))
        self.error_name_ = "legacy_support_OLS_broadcast_MSE"
        self.native_diagnostics_ = dict(
            legacy_public_error=self.error_, definition=self.error_name_
        )
        if hasattr(self, "mse_path_"):
            self.native_diagnostics_["CV_MSE"] = float(np.min(np.mean(self.mse_path_, axis=1)))
        return self
