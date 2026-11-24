import numpy as np
from scipy.linalg import solve_triangular, cholesky
from .legacy_base import LegacyNativeSolver as BasePCESolver


class SBLSolver(BasePCESolver):
    """Sparse Bayesian learning for polynomial-chaos expansions, following Zhou et al. (2019)."""

    def __init__(
        self,
        poly_matrix=None,
        max_degree=3,
        max_iter=1000,
        tol=1e-8,
        alpha_threshold=1e6,
        stability_factor=1e-12,
        sparsity_threshold=1e-5,
        last_model=None,
        q=1,
        **kwargs,
    ):
        """Set the convergence tolerance, pruning thresholds and variance floor."""
        super().__init__(poly_matrix, max_degree, last_model=last_model, q=q)
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_threshold = alpha_threshold
        self.stability_factor = stability_factor
        self.sparsity_threshold = sparsity_threshold
        self.poly_matrix = poly_matrix
        self.max_degree = max_degree
        self.last_model = last_model
        self.q = q

    def _ensure_1d(self, array):
        """Return a one-dimensional array."""
        array = np.asarray(array)
        if array.ndim > 1:
            return array.flatten()
        return array

    def unbiasedXy1(self, X6, Y6):
        """
        Center and standardize the design matrix and response.

        Returns the transformed arrays together with their means and standard deviations.
        """
        X_means = X6.mean(axis=0, keepdims=True)  # Keep the row dimension for broadcasting.
        Y_means = Y6.mean(axis=0, keepdims=True)

        # Use unit scale for constant columns and constant responses.
        X_stds = X6.std(axis=0, ddof=1, keepdims=True)
        X_stds[X_stds == 0] = 1
        Y_stds = Y6.std(axis=0, ddof=1, keepdims=True)
        Y_stds[Y_stds == 0] = 1

        X_unbiased = (X6 - X_means) / X_stds
        Y_unbiased = (Y6 - Y_means) / Y_stds
        X_unbiased[:, 0] = 1
        return (
            np.squeeze(X_unbiased),
            np.squeeze(Y_unbiased),
            np.squeeze(X_means),
            np.squeeze(X_stds),
            np.squeeze(Y_means),
            np.squeeze(Y_stds),
        )

    def rebiasCoefs(self, unbiased_coefs, X_means, X_stds, y_means, y_std):
        """
        Map standardized coefficients back to the original scale.

        Parameters
        ----------
        unbiased_coefs : ndarray
            Coefficients fitted to standardized data.
        X_means, X_stds : ndarray
            Feature means and standard deviations.
        y_means, y_std : float
            Response mean and standard deviation.

        Returns
        -------
        ndarray
            Coefficients on the original scale.
        """
        n = len(unbiased_coefs)  # Include the intercept and all slope terms.
        coefs5 = np.zeros(n, dtype=float)

        for j in range(1, n):  # Index zero is the intercept.
            coefs5[j] = unbiased_coefs[j] * y_std / X_stds[j]

        coefs5[0] = (
            unbiased_coefs[0] * y_std
            + y_means
            - np.sum(unbiased_coefs[1:] * X_means[1:] * (y_std / X_stds[1:]))
        )
        return coefs5

    def train_model(self, X_data, y):
        """Fit a polynomial-chaos model by sparse Bayesian learning."""
        print("Fitting with SBLSolver.")
        Sigma_active = None
        mu_active = None
        n_samples, n_features = X_data.shape

        y = self._ensure_1d(y)

        X, y, X_means, X_stds, y_means, y_std = self.unbiasedXy1(X_data, y)
        gamma = np.ones(n_features) * 0.01
        sigma2 = np.var(y) * 0.01

        active_mask = np.ones(n_features, dtype=bool)
        active_indices = np.arange(n_features)

        for iteration in range(self.max_iter):
            X_active = X[:, active_mask]
            gamma_active = gamma[active_mask]

            # E-step. Fit the posterior on the active design.
            try:
                mu_active, Sigma_active = self._compute_posterior(X_active, y, gamma_active, sigma2)
            except np.linalg.LinAlgError:
                # Use least squares if the posterior solve is numerically unstable.
                mu_active = np.linalg.lstsq(X_active, y, rcond=None)[0]
                Sigma_active = np.eye(len(mu_active)) * self.stability_factor

            mu_active = self._ensure_1d(mu_active)

            mu_full = np.zeros(n_features)
            mu_full[active_mask] = mu_active

            # M-step. Update the coefficient and noise variances.
            gamma_old = gamma.copy()
            sigma2_old = sigma2

            for i, idx in enumerate(active_indices):
                if i < len(mu_active):
                    gamma[idx] = mu_active[i] ** 2 + Sigma_active[i, i]
                else:
                    gamma[idx] = 1e-6

            y_pred = X_active @ mu_active
            residual = y - y_pred
            rss = np.sum(residual**2)

            trace_term = 0
            for i in range(len(mu_active)):
                if gamma_active[i] > self.stability_factor:
                    trace_term += 1 - Sigma_active[i, i] / gamma_active[i]

            sigma2 = max((rss + sigma2_old * trace_term) / n_samples, self.stability_factor)

            # Prune every iteration through index 19, then every fifth iteration.
            if iteration % 5 == 0 or iteration < 20:
                gamma_prune_mask = gamma < self.alpha_threshold

                coef_prune_mask = np.abs(mu_full) > self.sparsity_threshold

                new_active_mask = gamma_prune_mask & coef_prune_mask

                # Retain at least one feature.
                if not np.any(new_active_mask):
                    max_coef_idx = np.argmax(np.abs(mu_full))
                    new_active_mask[max_coef_idx] = True

                if np.array_equal(active_mask, new_active_mask):
                    # Test convergence once the active set stabilizes.
                    if (
                        iteration > 10
                        and max(np.abs(gamma - gamma_old)) < self.tol
                        and abs(sigma2 - sigma2_old) < self.tol
                    ):
                        print(f"SBL converged at iteration {iteration}.")
                        print(f"Active features: {np.sum(active_mask)}/{n_features}")
                        break

                active_mask = new_active_mask
                active_indices = np.where(active_mask)[0]

            gamma = np.clip(gamma, self.stability_factor, 1e12)

        coefficients_full = np.zeros(n_features)
        coefficients_full[active_mask] = mu_active

        coefficients = self.rebiasCoefs(coefficients_full, X_means, X_stds, y_means, y_std)

        final_sparse_mask = np.abs(coefficients) > self.sparsity_threshold
        coefficients[~final_sparse_mask] = 0.0

        print(
            f"Coefficients above 1e-10 in magnitude: {np.sum(np.abs(coefficients) > 1e-10)}/{n_features}"
        )
        non_zero_mask = np.abs(coefficients) > 0
        coefficients1 = coefficients[non_zero_mask]

        return coefficients, coefficients1

    def _compute_posterior(self, X, y, gamma, sigma2):
        """Compute the coefficient posterior distribution."""
        Gamma_inv = None
        XTX = None
        n_samples, n_features = X.shape

        y = self._ensure_1d(y)

        try:
            # Assemble the posterior precision matrix.
            Gamma_inv = np.diag(1.0 / (gamma + self.stability_factor))
            XTX = X.T @ X

            H = Gamma_inv + (1.0 / sigma2) * XTX
            H += self.stability_factor * np.eye(n_features)

            L = cholesky(H, lower=True, check_finite=False)
            Linv = solve_triangular(L, np.eye(n_features), lower=True, check_finite=False)
            Sigma = Linv.T @ Linv

            mu = (1.0 / sigma2) * Sigma @ X.T @ y
            mu = self._ensure_1d(mu)

            return mu, Sigma

        except (np.linalg.LinAlgError, ValueError):
            # Fall back to a pseudoinverse.
            try:
                H = Gamma_inv + (1.0 / sigma2) * XTX
                H_inv = np.linalg.pinv(H + self.stability_factor * np.eye(n_features))
                Sigma = H_inv
                mu = (1.0 / sigma2) * Sigma @ X.T @ y
                mu = self._ensure_1d(mu)
                return mu, Sigma
            except:
                # Fall back to least squares.
                mu = np.linalg.lstsq(X, y, rcond=None)[0]
                mu = self._ensure_1d(mu)
                Sigma = np.eye(n_features) * self.stability_factor
                return mu, Sigma

    def get_sparsity_ratio(self):
        """Return the fraction of active polynomial terms."""
        if hasattr(self, "coefficients_"):
            n_total = len(self.coefficients_)
            n_active = np.sum(np.abs(self.coefficients_) > 1e-10)
            return n_active / n_total if n_total > 0 else 1.0
        return 1.0

    def get_active_features(self):
        """Return active feature indices."""
        if hasattr(self, "coefficients_"):
            return np.where(np.abs(self.coefficients_) > 1e-10)[0]
        return np.array([], dtype=int)
