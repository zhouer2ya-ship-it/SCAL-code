import numpy as np
from .legacy_base import LegacyNativeSolver as BasePCESolver


class LARSSolver(BasePCESolver):
    """Hybrid least-angle regression following the forward-selection and OLS workflow in uq_lar.m."""

    def __init__(
        self,
        poly_matrix=None,
        max_degree=3,
        max_iter=1000,
        tol=1e-7,
        sparsity_threshold=1e-5,
        last_model=None,
        q=1,
        **kwargs,
    ):
        super().__init__(poly_matrix, max_degree, last_model=last_model, q=q)

        # Defaults follow the options in uq_lar.m.
        self.normalize_columns = 1
        self.early_stop = 1
        self.hybrid_lars = 1
        self.modified_loo = 1
        self.hybrid_loo = 1
        self.no_selection = 0
        self.generalized_ls = 0  # No response covariance CY is supplied.

        self.max_iter = max_iter
        self.poly_matrix = poly_matrix
        self.max_degree = max_degree
        self.last_model = last_model
        self.q = q

    def _ensure_1d(self, array):
        array = np.asarray(array)
        if array.ndim > 1:
            return array.flatten()
        return array

    def _compute_loo(self, Psi_active, M, Y, coeffs, modified_loo, modi_diag):
        """Compute corrected leave-one-out error using the matrix-trace penalty in uq_PCE_loo_error.m."""
        N, P_active = Psi_active.shape

        # With hybrid_loo enabled, refit the current support by ordinary least squares.
        if coeffs is None:
            coeffs = M @ Psi_active.T @ Y

        residual = Y - Psi_active @ coeffs

        # Add the intercept contribution to the diagonal of the hat matrix.
        h = np.sum((Psi_active @ M) * Psi_active, axis=1) + modi_diag

        h[h >= 1] = 0.9999

        loo_err = np.mean((residual / (1 - h)) ** 2)

        if modified_loo:
            # Apply the trace correction from Blatman, Eq. 5.13.
            try:
                T_factor = (N / (N - P_active)) * (1 + np.trace(M))
                loo_err = loo_err * T_factor
            except ZeroDivisionError:
                pass

        # Scale the error by the sample variance when the response varies.
        var_Y = np.var(Y, ddof=1) if N > 1 else np.var(Y)
        if var_Y > 0:
            loo_err = loo_err / var_Y

        return loo_err

    def train_model(self, X_data, y):
        print("Fitting with LARSSolver.")

        Psi = np.asarray(X_data).copy()
        Y = self._ensure_1d(y).copy()
        N, P = Psi.shape

        # Fit a single regressor directly by least squares.
        if P == 1:
            M = np.linalg.pinv(Psi.T @ Psi)
            coeffs = M @ Psi.T @ Y
            return coeffs, coeffs[np.abs(coeffs) > 0]

        # Center the design and response when the basis includes a constant.
        constidx = ~np.any(np.diff(Psi, axis=0), axis=0)
        constindices = np.where(constidx)[0]
        bool_const_and_center = np.any(constidx)

        if bool_const_and_center:
            constval = Psi[0, constidx]
            mu_Psi = np.mean(Psi, axis=0)
            Psi = Psi - mu_Psi  # Centering makes the constant column zero.
            mu_Y = np.mean(Y)
            Y = Y - mu_Y
            modi_diag = (1.0 / N) * np.ones(N)
            run_lars_iterations = np.var(Y) > 0
        else:
            modi_diag = np.zeros(N)
            run_lars_iterations = True

        if self.normalize_columns:
            normPsi = np.sqrt(np.sum(Psi**2, axis=0) / (N - 1))
            nznorm_indices = normPsi != 0
            Psi[:, nznorm_indices] = Psi[:, nznorm_indices] / normPsi[nznorm_indices]

        nvars = min(N - 2, P)
        mu = np.zeros(N)
        a_coeff = []
        i_coeff = list(range(P))
        M = np.array([])
        maxk = 8 * nvars

        coeff_array = np.zeros((nvars + 1, P))
        a_scores = -np.inf * np.ones(nvars + 1)
        loo_scores = np.inf * np.ones(nvars + 1)
        refscore = np.inf

        # Score the intercept-only fit before adding terms.
        if bool_const_and_center:
            ones_col = np.ones((N, 1))
            M_ones = np.linalg.pinv(ones_col.T @ ones_col)
            loo_scores[0] = self._compute_loo(
                ones_col, M_ones, Y, None, self.modified_loo, np.zeros(N)
            )
            a_scores[0] = 1 - loo_scores[0]

        maxiter = min(maxk, nvars)

        for k in range(maxiter):
            if not run_lars_iterations:
                break

            cj = Psi.T @ (Y - mu)

            C = np.max(np.abs(cj[i_coeff]))
            idx_rel = np.argmax(np.abs(cj[i_coeff]))
            idx = i_coeff[idx_rel]

            # Update the inverse Gram matrix for the enlarged support.
            if k == 0:
                M = np.linalg.pinv(Psi[:, [idx]].T @ Psi[:, [idx]])
            else:
                x = Psi[:, a_coeff].T @ Psi[:, idx]
                r = Psi[:, idx].T @ Psi[:, idx]

                # Use the Schur complement for the block inverse.
                S_val = r - x.T @ M @ x

                if np.abs(S_val) < 1e-15 or np.any(~np.isfinite(M)):
                    try:
                        M = np.linalg.pinv(Psi[:, a_coeff + [idx]].T @ Psi[:, a_coeff + [idx]])
                    except np.linalg.LinAlgError:
                        i_coeff.remove(idx)
                        M = np.linalg.pinv(Psi[:, a_coeff].T @ Psi[:, a_coeff])
                        continue
                else:
                    S_inv = 1.0 / S_val
                    M_top = np.hstack(
                        (M + S_inv * M @ np.outer(x, x) @ M, -S_inv * (M @ x).reshape(-1, 1))
                    )
                    M_bot = np.hstack((-S_inv * (x.T @ M).reshape(1, -1), [[S_inv]]))
                    M = np.vstack((M_top, M_bot))

            a_coeff.append(idx)

            s = np.sign(cj[a_coeff])
            s[s == 0] = 1

            c_val = 1.0 / np.sqrt(s.T @ M @ s)
            w = c_val * (M @ s)
            u = Psi[:, a_coeff] @ w
            aj = Psi.T @ u

            # Move along the equiangular direction to the next knot (Efron et al., 2004).
            if k < nvars - 1:
                tmp1 = (C - cj[i_coeff]) / (c_val - aj[i_coeff])
                tmp2 = (C + cj[i_coeff]) / (c_val + aj[i_coeff])
                tmp = np.concatenate([tmp1, tmp2])

                tmp_positive = tmp[tmp > 0]
                gamma = np.min(tmp_positive) if len(tmp_positive) > 0 else 0
            else:
                gamma = C / c_val

            i_coeff.remove(idx)
            mu = mu + gamma * u

            coeff_array[k + 1, a_coeff] = coeff_array[k, a_coeff] + gamma * w

            # Score each step by its leave-one-out Q-squared.
            if not self.hybrid_loo:
                loo = self._compute_loo(
                    Psi[:, a_coeff], M, Y, coeff_array[k + 1, a_coeff], self.modified_loo, modi_diag
                )
            else:
                loo = self._compute_loo(Psi[:, a_coeff], M, Y, None, self.modified_loo, modi_diag)

            loo_scores[k + 1] = loo
            a_scores[k + 1] = 1 - loo

            if loo < refscore:
                refscore = loo

            mm = int(np.round(nvars * 0.1))
            mm = max(mm, 100)
            mm = min(mm, nvars)

            if k + 1 > mm:
                if (loo_scores[(k + 1) - mm] <= refscore) and self.early_stop:
                    break

        if not self.no_selection:
            k_best = np.argmax(a_scores)
        else:
            k_best = len(a_scores) - 1

        # Recover the selected support on the original scale.
        nz_idx = np.abs(coeff_array[k_best, :]) > 0

        # Restore the column scales before the final fit.
        if self.normalize_columns:
            Psi[:, nznorm_indices] = Psi[:, nznorm_indices] * normPsi[nznorm_indices]

        if bool_const_and_center:
            Psi = Psi + mu_Psi
            Y = Y + mu_Y
            nz_idx[constindices[0]] = True

        coefficients = np.zeros(P)

        if self.hybrid_lars:
            # Refit the selected support by ordinary least squares.
            Psi_sub = Psi[:, nz_idx]
            try:
                coeffs_sub = np.linalg.pinv(Psi_sub.T @ Psi_sub) @ Psi_sub.T @ Y
            except np.linalg.LinAlgError:
                coeffs_sub = np.zeros(np.sum(nz_idx))
            coefficients[nz_idx] = coeffs_sub
        else:
            coefficients[nz_idx] = coeff_array[k_best, nz_idx]
            if self.normalize_columns:
                coefficients[nznorm_indices] = (
                    coefficients[nznorm_indices] / normPsi[nznorm_indices]
                )
            if bool_const_and_center:
                coefficients[constindices[0]] = np.mean(Y - Psi @ coefficients) / constval[0]

        coefficients1 = coefficients[np.abs(coefficients) > 0]

        return coefficients, coefficients1
