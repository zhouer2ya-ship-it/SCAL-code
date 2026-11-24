import numpy as np
from .legacy_base import LegacyNativeSolver as BasePCESolver


class BCSSolver(BasePCESolver):
    """Independent implementation of Fast Laplace Bayesian compressive sensing."""

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
        """Configure the basis, sparse-search iterations and convergence tolerance."""
        super().__init__(poly_matrix, max_degree, last_model=last_model, q=q)
        self.max_iter = max_iter
        self.tol = tol
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

    def train_model(self, X_data, y):
        print("Fitting with BCSSolver.")

        Psi = np.asarray(X_data)
        Y = self._ensure_1d(y)

        N, P = Psi.shape

        # Start the noise variance at 1% of the response variance.
        sigma2 = np.var(Y) * 0.01
        if sigma2 == 0:
            sigma2 = 1e-6

        beta = 1.0 / sigma2
        eta_stop = self.tol
        max_iterations = self.max_iter

        PsiTPsi = Psi.T @ Psi
        PsiTY = Psi.T @ Y

        # Scale each squared inner product with Y by the squared norm of its design column.
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_criteria = (PsiTY**2) / np.diag(PsiTPsi)
        ind_start = np.nanargmax(correlation_criteria)

        # A local index of -1 marks a term outside the active set.
        ind_global_to_local = np.full(P, -1, dtype=int)
        ind_global_to_local[ind_start] = 0
        num_active = 1
        active_indices = [ind_start]
        deleted_indices = []

        bcs_path = [ind_start]

        gamma = np.zeros(P)
        gamma[ind_start] = (PsiTY[ind_start] ** 2 - sigma2 * PsiTPsi[ind_start, ind_start]) / (
            PsiTPsi[ind_start, ind_start] ** 2
        )

        Sigma = np.array([[1.0 / (beta * PsiTPsi[ind_start, ind_start] + 1.0 / gamma[ind_start])]])
        mu = np.array([Sigma[0, 0] * PsiTY[ind_start] * beta])

        tmp1 = beta * PsiTPsi[ind_start, :]
        S = beta * np.diag(PsiTPsi) - Sigma[0, 0] * (tmp1**2)
        Q = beta * PsiTY - mu[0] * tmp1

        # Adjust the initial sparsity and quality factors for the active term.
        tmp2 = np.ones(P)
        q0tilde = PsiTY[ind_start]
        s0tilde = PsiTPsi[ind_start, ind_start]
        tmp2[ind_start] = s0tilde / (q0tilde**2) / beta

        s = S / tmp2
        q = Q / tmp2

        with np.errstate(divide="ignore", invalid="ignore"):
            lambda_val = 2.0 * (num_active - 1) / np.sum(gamma)

        it_counter = 0
        Delta_L_max_arr = []

        while True:
            it_counter += 1

            # Compute the proposed prior variance for each term.
            gamma_potential = np.zeros(P)
            if lambda_val == 0:  # RVM
                cond = (q**2 - s) > lambda_val
                gamma_potential[cond] = (q[cond] ** 2 - s[cond]) / (s[cond] ** 2)
            else:
                a = lambda_val * (s**2)
                b = s**2 + 2 * lambda_val * s
                c = lambda_val + s - q**2

                cond = c < 0
                discriminant = b[cond] ** 2 - 4 * a[cond] * c[cond]
                # Clip the discriminant to its nonnegative domain.
                discriminant = np.maximum(discriminant, 0)

                gamma_potential[cond] = (-b[cond] + np.sqrt(discriminant)) / (2 * a[cond])

            l_gamma = (
                -np.log(np.abs(1 + gamma * s))
                + (q**2 * gamma) / (1 + gamma * s)
                - lambda_val * gamma
            )
            l_gamma_potential = (
                -np.log(np.abs(1 + gamma_potential * s))
                + (q**2 * gamma_potential) / (1 + gamma_potential * s)
                - lambda_val * gamma_potential
            )

            Delta_L_potential = l_gamma_potential - l_gamma

            # Exclude terms removed in earlier iterations.
            if deleted_indices:
                Delta_L_potential[deleted_indices] = -np.inf

            ind_L_max = np.argmax(Delta_L_potential)
            Delta_L_max_val = Delta_L_potential[ind_L_max]

            # Keep at least one active term.
            if (
                num_active == 1
                and ind_L_max == active_indices[0]
                and gamma_potential[ind_L_max] == 0
            ):
                Delta_L_potential[ind_L_max] = -np.inf
                ind_L_max = np.argmax(Delta_L_potential)
                Delta_L_max_val = Delta_L_potential[ind_L_max]

            Delta_L_max_arr.append(Delta_L_max_val)

            # Stop at a nonpositive gain or two gains below the relative tolerance.
            if Delta_L_max_val <= 0 or (
                it_counter > 1
                and np.all(np.abs(Delta_L_max_arr[-2:]) < np.sum(Delta_L_max_arr) * eta_stop)
            ):
                print(f"BCS converged at iteration {it_counter}.")
                break

            what_changed = int(gamma[ind_L_max] == 0) - int(gamma_potential[ind_L_max] == 0)

            if what_changed == 1:  # adding a regressor
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                Sigma_ii = 1.0 / (1.0 / gamma[ind_L_max] + S[ind_L_max])
                x_i = Sigma @ PsiTPsi[active_indices, ind_L_max]
                tmp_1 = -(beta * Sigma_ii) * x_i

                Sigma_top = np.hstack(
                    (Sigma + (beta**2 * Sigma_ii) * np.outer(x_i, x_i), tmp_1[:, np.newaxis])
                )
                Sigma_bottom = np.hstack((tmp_1[np.newaxis, :], np.array([[Sigma_ii]])))
                Sigma = np.vstack((Sigma_top, Sigma_bottom))

                mu_i = Sigma_ii * Q[ind_L_max]
                mu = np.append(mu - (beta * mu_i) * x_i, mu_i)

                tmp2_update = beta * (
                    PsiTPsi[:, ind_L_max] - beta * (PsiTPsi[:, active_indices] @ x_i)
                )
                S = S - Sigma_ii * (tmp2_update**2)
                Q = Q - mu_i * tmp2_update

                ind_global_to_local[ind_L_max] = num_active
                num_active += 1
                active_indices.append(ind_L_max)
                bcs_path.append(ind_L_max)

            elif what_changed == 0:  # recomputation
                if ind_global_to_local[ind_L_max] == -1:
                    raise ValueError(
                        f"cannot recompute index {ind_L_max} -- not yet part of the model!"
                    )

                gamma_i_new = gamma_potential[ind_L_max]
                gamma_i_old = gamma[ind_L_max]
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                local_ind = ind_global_to_local[ind_L_max]

                kappa_i = 1.0 / (
                    Sigma[local_ind, local_ind] + 1.0 / (1.0 / gamma_i_new - 1.0 / gamma_i_old)
                )
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - kappa_i * np.outer(Sigma_i_col, Sigma_i_col)
                mu_i = mu[local_ind]
                mu = mu - (kappa_i * mu_i) * Sigma_i_col

                tmp1_update = beta * (Sigma_i_col @ PsiTPsi[active_indices, :])
                S = S + kappa_i * (tmp1_update**2)
                Q = Q + (kappa_i * mu_i) * tmp1_update

                bcs_path.append(ind_L_max + 0.1)

            elif what_changed == -1:  # Remove regressor
                gamma[ind_L_max] = 0.0
                local_ind = ind_global_to_local[ind_L_max]

                Sigma_ii_inv = 1.0 / Sigma[local_ind, local_ind]
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - Sigma_ii_inv * np.outer(Sigma_i_col, Sigma_i_col)
                Sigma = np.delete(Sigma, local_ind, axis=0)
                Sigma = np.delete(Sigma, local_ind, axis=1)

                mu_i = mu[local_ind]
                mu = mu - (mu_i * Sigma_ii_inv) * Sigma_i_col
                mu = np.delete(mu, local_ind)

                tmp1_update = beta * (Sigma_i_col @ PsiTPsi[active_indices, :])
                S = S + Sigma_ii_inv * (tmp1_update**2)
                Q = Q + (mu_i * Sigma_ii_inv) * tmp1_update

                num_active -= 1

                mask = ind_global_to_local > local_ind
                ind_global_to_local[mask] -= 1
                ind_global_to_local[ind_L_max] = -1

                active_indices.pop(local_ind)
                deleted_indices.append(ind_L_max)
                bcs_path.append(-ind_L_max)

            tmp3 = 1 - gamma * S
            s = S / tmp3
            q = Q / tmp3

            lambda_val = 2.0 * (num_active - 1) / np.sum(gamma)

            if it_counter > max_iterations:
                break

        coeffs = np.zeros(P)
        if len(active_indices) > 0:
            coeffs[active_indices] = mu

        final_sparse_mask = np.abs(coeffs) > self.sparsity_threshold
        coeffs[~final_sparse_mask] = 0.0

        print(f"Nonzero coefficients: {np.sum(np.abs(coeffs) > 0)}/{P}")

        coefficients1 = coeffs[np.abs(coeffs) > 0]

        return coeffs, coefficients1
