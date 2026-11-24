"""AHSC: separately normalised polynomial-germ gradient and reproduced Theta."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .theta import ThetaSampler, standardize_inputs, germ_logpdf, theta_scores, theta_from_physical

AHSC_VERSION = "AHSC-2026.10.18-gradient-theta-sum"
THETA_VERSION = "Theta-2026.10.15-original-eq24-26-exact-kdtree"


def minmax_component(values, valid):
    """Normalise exactly the available set; only exact constants are constants."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("Scores and candidate mask must be aligned vectors")
    result = np.zeros_like(values)
    active = values[valid]
    if not np.all(np.isfinite(active)) or np.any(active < 0):
        raise FloatingPointError("AHSC components must be finite and nonnegative")
    if not len(active):
        return result, dict(min=None, max=None, constant=True)
    low, high = float(active.min()), float(active.max())
    if high == low:
        result[valid] = float(high != 0)
    else:
        result[valid] = (active - low) / (high - low)
    return result, dict(min=low, max=high, constant=bool(high == low))


def polynomial_germ_gradient(X, model, poly_matrix=None, input_transform=None):
    """Differentiate the full fitted PCE with respect to the input germ xi.

    The coordinate-gradient routine returns dh/dz, with z standardised by
    physical target SD. The final Jacobian converts z to Legendre/Hermite xi;
    for lognormal inputs this Jacobian depends on the evaluation point.
    """
    from .base import compute_gradient_vectors
    from ..coordinates import TargetGeometry
    from types import SimpleNamespace

    x = np.atleast_2d(np.asarray(X, dtype=float))
    poly = np.asarray(model.poly_matrix if poly_matrix is None else poly_matrix, dtype=float)
    standardize_inputs(x, poly)
    geometry = TargetGeometry.from_poly(poly)
    coef = np.asarray(model.coef_, dtype=float)
    # Protected structural terms may have zero coefficients; every nonzero
    # coefficient still belongs to the current response, even outside support_.
    proxy = SimpleNamespace(
        poly_matrix=model.poly_matrix,
        mapping_=model.mapping_,
        coef_=coef,
        support_indices_=np.flatnonzero(coef),
    )
    gradient = compute_gradient_vectors(
        x, proxy, input_transform=input_transform, geometry=geometry
    )
    factors = np.ones_like(x)
    for j, (kind, a, b) in enumerate(poly):
        if kind == 0:
            factors[:, j] = np.sqrt(3.0)
        elif kind == 2:
            factors[:, j] = b * x[:, j] / geometry.scale[j]
    gradient *= factors
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Nonfinite polynomial-germ gradient")
    return gradient


def _model_design(model, X, input_transform=None):
    from ..solving.utils import construct_matrix_optimized, normalization_all

    keys = list(model.mapping_)
    constants = [
        i for i, key in enumerate(keys) if str(model.mapping_[key]).strip() in {"1", "1.0"}
    ]
    if len(constants) != 1:
        raise ValueError("The PCE must identify exactly one constant basis term")
    coef = np.asarray(model.coef_, dtype=float)
    active = np.union1d(constants, np.flatnonzero(coef))
    mapping = {keys[i]: model.mapping_[keys[i]] for i in active}
    physical = X if input_transform is None else input_transform.transform(X)
    design = construct_matrix_optimized(mapping, normalization_all(physical, model.poly_matrix).T)
    return design, coef[active], int(np.flatnonzero(active == constants[0])[0])


def model_set_components(
    train,
    pool,
    models,
    poly_matrix,
    *,
    input_transform=None,
    reconstruction=None,
    output_weights=None,
):
    """Scalar criterion, or explicitly defined weighted-field extension.

    For a field, the centred amplitude is ||W^(1/2)(h-E[h])||_2 and the
    gradient is ||W^(1/2) J_xi h||_F. PCA reconstruction precedes both norms.
    Nearest-neighbour geometry and density stay in the original input germ.
    """
    from .base import stable_norm

    if not models:
        raise ValueError("At least one fitted PCE is required")
    k = len(models)
    V = np.eye(k) if reconstruction is None else np.asarray(reconstruction, dtype=float)
    if V.ndim != 2 or V.shape[1] != k or not np.all(np.isfinite(V)):
        raise ValueError("Output reconstruction must be finite with one column per model")
    weights = np.ones(len(V)) if output_weights is None else np.asarray(output_weights, dtype=float)
    if weights.shape != (len(V),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("One finite positive weight is required per physical output")
    target_centering = input_transform is not None and not input_transform.analytic_orthogonal
    if target_centering:
        # PCA can change the joint measure for non-Gaussian inputs.
        # Use each model's target mean to center its predictions.
        for model in models:
            method = getattr(model, "moment_method_", "")
            if method not in {
                "target_distribution_sobol",
                "target_distribution_analytic",
            } or not np.isfinite(getattr(model, "mean_", np.nan)):
                raise ValueError(
                    "Non-Gaussian input PCA requires a verified target-distribution response mean; "
                    "use the engine moment calculation or provide mean_ and moment_method_"
                )
    gram = V.T @ (weights[:, None] * V)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
    designs_a, designs_c, coefficients, constants = [], [], [], []
    for model in models:
        a, coef, constant = _model_design(model, train, input_transform)
        c, _, _ = _model_design(model, pool, input_transform)
        designs_a.append(a)
        designs_c.append(c)
        coefficients.append(coef)
        constants.append(constant)
    if k == 1 and V.shape == (1, 1) and V[0, 0] == 1 and weights[0] == 1 and not target_centering:
        raw = theta_from_physical(
            train,
            pool,
            designs_a[0],
            designs_c[0],
            coefficients[0],
            poly_matrix,
            constant_index=constants[0],
        )
    else:
        centered = []
        for designs in (designs_a, designs_c):
            columns = []
            for model, a, coef, constant in zip(models, designs, coefficients, constants):
                active = np.arange(len(coef)) != constant
                centered_model = a[:, active] @ coef[active]
                if target_centering:
                    centered_model = centered_model + (coef[constant] - model.mean_)
                columns.append(centered_model)
            weighted_modes = np.column_stack(columns) @ root.T
            centered.append(stable_norm(weighted_modes, axis=1))
        xi_a, families = standardize_inputs(train, poly_matrix)
        xi_c, _ = standardize_inputs(pool, poly_matrix)
        # The low-level reproduced Theta routine accepts any consistent
        # centred response design; the second column is the field amplitude.
        raw = theta_scores(
            xi_a,
            xi_c,
            np.column_stack([np.ones(len(train)), centered[0]]),
            np.column_stack([np.ones(len(pool)), centered[1]]),
            [0.0, 1.0],
            existing_logpdf=germ_logpdf(xi_a, families),
            candidate_logpdf=germ_logpdf(xi_c, families),
        )
    gradient = np.stack(
        [polynomial_germ_gradient(pool, m, poly_matrix, input_transform) for m in models], axis=1
    )
    # Compute in modal coordinates: this avoids allocating candidates x mesh
    # points x inputs for a large PCA-reconstructed spatial field.
    weighted = np.einsum("ab,nbd->nad", root, gradient)
    return raw, stable_norm(weighted.reshape(len(pool), -1))


class AHSCSampler(ThetaSampler):
    """S=N(||grad_xi h||_2)+N(Theta), with one observation/refit per query."""

    def __init__(
        self,
        poly_matrix=None,
        verbose=True,
        input_transform=None,
        alpha=None,
        min_distance_ratio=None,
    ):
        if alpha is not None or min_distance_ratio not in (None, 0):
            raise ValueError(
                "AHSC has fixed unit coefficients and no separation radius; "
                "use AHSCOriginalSampler for historical alpha/radius settings"
            )
        super().__init__(poly_matrix=poly_matrix, verbose=verbose)
        self.input_transform = input_transform
        self.models = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.models = None

    def set_model(self, model):
        if self.input_transform is not None:
            self.set_models([model])
        else:
            super().set_model(model)
            self.models = None

    def set_models(self, models, *, reconstruction=None, output_weights=None):
        if (
            self.X_train is None
            or not models
            or not all(getattr(m, "is_fitted_", False) for m in models)
        ):
            raise RuntimeError("Initialise with observations and supply fitted PCEs")
        if self._pending_point is not None:
            raise RuntimeError(
                "Reinitialise with evaluated observations after refitting the model set"
            )
        poly = models[0].poly_matrix if self.poly_matrix is None else self.poly_matrix
        standardize_inputs(self.X_train, poly)
        expected = (
            self.X_train
            if self.input_transform is None
            else self.input_transform.transform(self.X_train)
        )
        for model in models:
            if not np.all(np.isfinite(model.coef_)):
                raise ValueError("PCE coefficients must be finite")
            fitted_x = getattr(model, "X", None)
            if fitted_x is not None and not np.array_equal(fitted_x, expected):
                raise ValueError("Models must be fitted on the supplied observation design")
            expected_poly = (
                poly if self.input_transform is None else self.input_transform.feature_poly
            )
            if not np.array_equal(model.poly_matrix, expected_poly):
                raise ValueError("Model and input-transform polynomial laws disagree")
        self.poly_matrix = np.asarray(poly, dtype=float).copy()
        self.models, self.model = list(models), models[0]
        self.reconstruction, self.output_weights = reconstruction, output_weights
        self.last_scores = None

    def score_candidates(self):
        if self.models is None:
            raw = super().score_candidates()
            from .base import stable_norm

            gradient = stable_norm(polynomial_germ_gradient(self.candidate_pool, self.model))
        else:
            if self._pending_point is not None:
                raise RuntimeError("Evaluate, refit and reinitialise before another query")
            if not len(self.candidate_pool):
                raise RuntimeError("NO_UNUSED_DISTINCT_CANDIDATE")
            raw, gradient = model_set_components(
                self.X_train,
                self.candidate_pool,
                self.models,
                self.poly_matrix,
                input_transform=self.input_transform,
                reconstruction=self.reconstruction,
                output_weights=self.output_weights,
            )
        valid = raw.distance > 0
        self.raw_theta_scores, self.gradient_norm = raw, gradient
        self.normalized_gradient, gradient_bounds = minmax_component(gradient, valid)
        theta_values = raw.theta
        domain = "raw_theta"
        # Positive common rescaling preserves min-max exactly in real
        # arithmetic; use logs only when raw scores overflow or underflow.
        finite_logs = valid & np.isfinite(raw.logtheta)
        if np.any(valid & ~np.isfinite(theta_values)) or np.any(finite_logs & (theta_values == 0)):
            maximum = np.max(raw.logtheta[finite_logs])
            with np.errstate(under="ignore"):
                theta_values = np.exp(raw.logtheta - maximum)
            domain = "common_log_rescaling"
        self.normalized_theta, theta_bounds = minmax_component(theta_values, valid)
        total = self.normalized_gradient + self.normalized_theta
        with np.errstate(divide="ignore"):
            logtotal = np.log(total)
        self.normalization_bounds = dict(
            gradient=gradient_bounds, theta=theta_bounds, theta_domain=domain
        )
        self.last_scores = replace(raw, theta=total, logtheta=logtotal)
        return self.last_scores

    def select_samples(self, k=1):
        selected = super().select_samples(k)
        for record in self.last_selection_report:
            i = record["selected_local_index"]
            # Parent theta/logtheta describe the combined score here; retain
            # the actual reproduced Theta separately to avoid ambiguous logs.
            record.pop("all_zero_theta", None)
            record.update(
                criterion="AHSC",
                acquisition_version=AHSC_VERSION,
                theta_version=THETA_VERSION,
                score=float(self.last_scores.theta[i]),
                raw_original_theta=float(self.raw_theta_scores.theta[i])
                if np.isfinite(self.raw_theta_scores.theta[i])
                else None,
                log_original_theta=float(self.raw_theta_scores.logtheta[i])
                if np.isfinite(self.raw_theta_scores.logtheta[i])
                else None,
                raw_gradient_norm=float(self.gradient_norm[i]),
                normalized_gradient=float(self.normalized_gradient[i]),
                normalized_theta=float(self.normalized_theta[i]),
                gradient_weight=1,
                theta_weight=1,
                normalization_bounds=self.normalization_bounds,
                normalization="separate_minmax_over_unused_distinct_candidates",
                constant_rule="all-zero stays 0; nonzero constant becomes 1",
                gradient_coordinates="original Legendre/Hermite input germ xi",
                theta_geometry="original Euclidean input germ",
                distance_threshold=0.0,
                centering_source="target_distribution_moments"
                if self.input_transform is not None and not self.input_transform.analytic_orthogonal
                else "polynomial_constant_coefficient",
                all_zero_original_theta=bool(
                    np.all(
                        np.isneginf(
                            self.raw_theta_scores.logtheta[self.raw_theta_scores.distance > 0]
                        )
                    )
                ),
                output_extension="weighted_field"
                if self.models is not None
                and (len(self.models) > 1 or self.reconstruction is not None)
                else "scalar",
            )
        return selected


AdaptiveHybridSampler = AHSCSampler
GradientThetaSumSampler = AHSCSampler
LatestSampler = AHSCSampler


def modal_vote(scores, feasible, candidate_ids):
    """Select a pool row from scalar mode scores using the field experiment rule."""
    scores = np.asarray(scores, dtype=float)
    feasible = np.asarray(feasible)
    ids = np.asarray(candidate_ids)
    if (
        scores.ndim != 2
        or not len(scores)
        or scores.shape[1:] != ids.shape
        or feasible.shape != ids.shape
        or feasible.dtype != bool
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Expected mode-by-candidate scores, a Boolean mask and unique integer IDs")
    available = np.flatnonzero(feasible)
    if not len(available):
        raise RuntimeError("NO_UNUSED_DISTINCT_CANDIDATE")
    valid_scores = scores[:, available]
    if not np.isfinite(valid_scores).all() or np.any(valid_scores < 0):
        raise FloatingPointError("Invalid scalar AHSC scores in modal vote")
    active = np.any(valid_scores != 0, axis=1)
    votes = np.zeros(len(available), dtype=int)
    for row in valid_scores[active]:
        tied = np.flatnonzero(row == row.max())
        votes[tied[np.argmin(ids[available[tied]])]] += 1
    finalists = np.flatnonzero(votes == votes.max())
    mean_score = valid_scores.mean(axis=0)
    ranked = np.lexsort((ids[available[finalists]], -mean_score[finalists]))
    return int(available[finalists[ranked[0]]])


def modal_ahsc_candidate(training_inputs, candidates, models, *, candidate_ids=None):
    """Score each refitted scalar mode and return one candidate row in the supplied pool."""
    models = list(models)
    if not models:
        raise ValueError("Supply at least one fitted mode")
    sampler = AHSCSampler(models[0].poly_matrix, verbose=False)
    sampler.initialize(training_inputs, None, candidates, candidate_ids=candidate_ids)
    scores = []
    for model in models:
        sampler.set_model(model)
        scores.append(sampler.score_candidates())
    return modal_vote(
        np.stack([item.theta for item in scores]),
        scores[0].distance > 0,
        sampler.candidate_ids,
    )
