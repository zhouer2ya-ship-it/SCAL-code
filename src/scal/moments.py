"""Target-distribution moments with optional full output covariance.

The default diagonal mode never forms an M x M output array. Output PCA still
uses the full latent covariance, including cross-mode terms.
"""

import numpy as np
from scipy.stats import qmc
from .coordinates import from_unit


def coefficient_moments(models):
    """Joint PCE moments including cross-mode covariance in one dictionary."""
    if not models:
        return np.empty(0), np.empty((0, 0))
    mapping = list(models[0].mapping_)
    if any(
        list(m.mapping_) != mapping
        or not np.array_equal(m.poly_matrix, models[0].poly_matrix)
        or not np.array_equal(m.gamma_, models[0].gamma_)
        for m in models
    ):
        raise ValueError("Analytic cross moments require the same target-orthogonal dictionary")
    coefs = np.column_stack([m.coef_ for m in models])
    factors = np.sqrt(models[0].gamma_[1:])[:, None] * coefs[1:]
    return coefs[0], factors.T @ factors


def covariance_preflight(n_outputs, max_covariance_bytes=None):
    """Check the requested float64 covariance-array budget before allocation.

    Above 64 MiB, callers must explicitly provide the array budget. Numerical
    integration also holds repeat covariances; this limit describes one array,
    not total process memory.
    """
    elements = int(n_outputs) ** 2
    required = 8 * elements
    if max_covariance_bytes is not None:
        if int(max_covariance_bytes) != max_covariance_bytes or max_covariance_bytes < 0:
            raise ValueError("max_covariance_bytes must be a nonnegative integer")
        if required > max_covariance_bytes:
            raise MemoryError(
                f"Full covariance requires {elements} float64 elements ({required} bytes), "
                f"exceeding max_covariance_bytes={max_covariance_bytes}"
            )
    elif required > 64 * 1024**2:
        raise MemoryError(
            f"Full covariance requires {required} bytes; explicitly set max_covariance_bytes"
        )
    return dict(
        covariance_elements=elements,
        covariance_array_bytes=required,
        max_covariance_bytes=max_covariance_bytes,
    )


def _psd_root(cov):
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        raise ValueError("Non-finite covariance")
    if not cov.size:
        return cov.copy()
    eig, vectors = np.linalg.eigh((cov + cov.T) * 0.5)
    tolerance = 128 * np.finfo(float).eps * len(cov) * np.max(np.abs(eig), initial=0.0)
    if np.min(eig) < -tolerance:
        raise ValueError("Covariance is not positive semidefinite beyond roundoff tolerance")
    return vectors * np.sqrt(np.maximum(eig, 0.0))[None, :]


def _project_moments(mean, cov, V, center, full, block_size):
    root = _psd_root(cov)
    output_mean = np.asarray(center) + V @ mean
    variance = np.empty(len(V))
    for start in range(0, len(V), block_size):
        factor = V[start : start + block_size] @ root
        variance[start : start + block_size] = np.sum(factor * factor, axis=1)
    if not np.all(np.isfinite(variance)):
        raise ValueError("Reconstructed output variance is not representable")
    if full:
        full_factor = V @ root
        covariance = full_factor @ full_factor.T
    else:
        covariance = None
    return output_mean, variance, covariance


def _stream_moments(
    predict, unit, poly, chunk_size, full, max_covariance_bytes=None, *, enforce_output_budget=True
):
    count = 0
    mean = accum = None
    for start in range(0, len(unit), chunk_size):
        Y = np.asarray(predict(from_unit(unit[start : start + chunk_size], poly)), dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if (
            Y.ndim != 2
            or len(Y) != min(chunk_size, len(unit) - start)
            or not np.all(np.isfinite(Y))
        ):
            raise ValueError(
                "Non-finite or misaligned predictions during target-distribution integration"
            )
        if full and count == 0 and enforce_output_budget:
            covariance_preflight(Y.shape[1], max_covariance_bytes)
        block_mean = Y.mean(axis=0)
        centered = Y - block_mean
        block_accum = centered.T @ centered if full else np.sum(centered * centered, axis=0)
        n = len(Y)
        if count == 0:
            mean, accum = block_mean, block_accum
        else:
            delta = block_mean - mean
            accum += block_accum + (np.outer(delta, delta) if full else delta * delta) * (
                count * n / (count + n)
            )
            mean += delta * n / (count + n)
        count += n
    if not np.all(np.isfinite(accum)):
        raise ValueError("Integrated covariance/variance is not representable")
    return mean, accum / count


def integrate_moments(
    predict,
    poly,
    *,
    seed=42,
    repeats=8,
    min_power=12,
    max_power=16,
    rtol=1e-3,
    atol=0.0,
    chunk_size=2048,
    covariance=False,
    max_covariance_bytes=None,
    output_block_size=64,
    reconstruction=None,
    output_center=None,
    prediction_groups=None,
):
    """Independent scrambled Sobol repeats with doubling and streamed moments.

    With reconstruction, integrate the small latent joint covariance and
    reconstruct each repeat before estimating output standard errors.
    prediction_groups is an internal sequence of callables sharing every Sobol
    point; diagonal integration evaluates one output group at a time.
    """
    if (
        repeats < 2
        or min_power < 1
        or max_power < min_power
        or rtol < 0
        or atol < 0
        or chunk_size < 1
        or output_block_size < 1
    ):
        raise ValueError("Invalid integration configuration")
    if not isinstance(covariance, (bool, np.bool_)):
        raise ValueError("covariance must be a boolean")
    full = bool(covariance)
    V = None if reconstruction is None else np.asarray(reconstruction, dtype=float)
    if full and V is not None:
        covariance_preflight(len(V), max_covariance_bytes)
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(repeats)]
    previous = None
    converged = False
    for power in range(min_power, max_power + 1):
        means, covs, model_means, model_vars = [], [], [], []
        for repeat_seed in seeds:
            unit = qmc.Sobol(len(poly), scramble=True, seed=repeat_seed).random_base2(power)
            if V is not None:
                # This r x r covariance is required even in diagonal output mode.
                # The caller's budget applies to the M x M output covariance only.
                mu, cov = _stream_moments(
                    predict, unit, poly, chunk_size, True, enforce_output_budget=False
                )
                model_means.append(mu)
                model_vars.append(np.diag(cov))
                mu, var, cov = _project_moments(mu, cov, V, output_center, full, output_block_size)
                statistic = cov if full else var
            elif prediction_groups is not None and not full:
                parts = [
                    _stream_moments(group, unit, poly, chunk_size, False)
                    for group in prediction_groups
                ]
                mu = np.concatenate([part[0] for part in parts])
                statistic = np.concatenate([part[1] for part in parts])
                model_means.append(mu)
                model_vars.append(statistic)
            else:
                mu, statistic = _stream_moments(
                    predict, unit, poly, chunk_size, full, max_covariance_bytes
                )
                model_means.append(mu)
                model_vars.append(np.diag(statistic) if full else statistic)
            means.append(mu)
            covs.append(statistic)
        means, covs = np.array(means), np.array(covs)
        mean, cov = means.mean(axis=0), covs.mean(axis=0)
        variances = np.diagonal(covs, axis1=1, axis2=2) if full else covs
        if np.any(variances < 0):
            raise ValueError("Negative integrated output variance")
        std = np.sqrt(variances.mean(axis=0))
        repeat_stds = np.sqrt(variances)
        mean_se = means.std(axis=0, ddof=1) / np.sqrt(repeats)
        std_se = repeat_stds.std(axis=0, ddof=1) / np.sqrt(repeats)
        cov_se = covs.std(axis=0, ddof=1) / np.sqrt(repeats)
        scale_mean = np.maximum(abs(mean), std)
        cov_scale = np.outer(std, std) if full else std * std
        within = np.all(mean_se <= atol + rtol * scale_mean) and np.all(std_se <= atol + rtol * std)
        within &= np.all(cov_se <= atol * atol + rtol * cov_scale)
        if previous is not None:
            within &= np.all(abs(mean - previous[0]) <= atol + rtol * scale_mean)
            within &= np.all(abs(std - previous[1]) <= atol + rtol * std)
            within &= np.all(abs(cov - previous[2]) <= atol * atol + rtol * cov_scale)
            if within:
                converged = True
                break
        previous = mean, std, cov
    metadata = dict(
        moment_method="scrambled_sobol",
        target_distribution=np.asarray(poly).tolist(),
        covariance_mode="full" if full else "diagonal",
        points_per_repeat=2**power,
        repeats=repeats,
        total_points=repeats * 2**power,
        seed=seed,
        repeat_seeds=seeds,
        mean_standard_error=mean_se.tolist(),
        std_standard_error=std_se.tolist(),
        variance_standard_error=(np.diag(cov_se) if full else cov_se).tolist(),
        covariance_standard_error=cov_se.tolist() if full and cov_se.size <= 4096 else None,
        covariance_standard_error_max=float(np.max(cov_se, initial=0.0)) if full else None,
        integration_converged=bool(converged),
        integration_rtol=rtol,
        integration_atol=atol,
        integration_space="latent_joint_reconstructed_per_repeat"
        if V is not None
        else "original_outputs",
        error_source="randomized_integration_only",
    )
    if full:
        metadata.update(covariance_preflight(len(mean), max_covariance_bytes))
        metadata["repeat_covariance_arrays"] = repeats
    return dict(
        mean=mean,
        std=std,
        covariance=cov if full else None,
        metadata=metadata,
        model_moments=dict(
            mean=np.mean(model_means, axis=0), std=np.sqrt(np.mean(model_vars, axis=0))
        ),
    )


def surrogate_moments(
    bundle, poly, *, covariance=False, max_covariance_bytes=None, output_block_size=64, **options
):
    if not isinstance(covariance, (bool, np.bool_)):
        raise ValueError("covariance must be a boolean")
    if output_block_size < 1:
        raise ValueError("output_block_size must be positive")
    full = bool(covariance)
    transform = bundle.input_pca_model
    pca = bundle.pca_model
    n_outputs = len(pca.mean_) if pca is not None else len(bundle.models)
    allocation = covariance_preflight(n_outputs, max_covariance_bytes) if full else {}
    analytic = transform is None or getattr(transform, "analytic_orthogonal", False)
    if analytic or (pca is not None and not bundle.models):
        if pca is not None:
            latent_mean, latent_cov = coefficient_moments(bundle.models)
            mean, variance, cov = _project_moments(
                latent_mean, latent_cov, pca.reconstruction_, pca.mean_, full, output_block_size
            )
            model_stats = dict(mean=latent_mean, std=np.sqrt(np.maximum(np.diag(latent_cov), 0.0)))
        elif full:
            mean, cov = coefficient_moments(bundle.models)
            variance = np.diag(cov)
            model_stats = dict(mean=mean, std=np.sqrt(variance))
        else:
            mean = np.asarray([m.coef_[0] for m in bundle.models])
            variance = np.asarray(
                [
                    np.sum(np.asarray(m.coef_[1:]) ** 2 * np.asarray(m.gamma_[1:]))
                    for m in bundle.models
                ]
            )
            cov = None
            model_stats = dict(mean=mean, std=np.sqrt(variance))
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("Analytic moments are not representable")
        return dict(
            mean=mean,
            std=np.sqrt(variance),
            covariance=cov,
            model_moments=model_stats,
            metadata=dict(
                moment_method="target_orthogonal_pce" if analytic else "constant_output",
                covariance_mode="full" if full else "diagonal",
                target_distribution=np.asarray(poly).tolist(),
                error_source="analytic",
                total_points=0,
                repeats=0,
                **allocation,
            ),
        )

    def model_predictor(models):
        def predict(X):
            inputs = transform.transform(X)
            return np.column_stack([m.predict(inputs) for m in models])

        return predict

    groups = None
    if pca is None and not full:
        groups = [
            model_predictor(bundle.models[start : start + output_block_size])
            for start in range(0, len(bundle.models), output_block_size)
        ]
    return integrate_moments(
        model_predictor(bundle.models),
        poly,
        covariance=full,
        max_covariance_bytes=max_covariance_bytes,
        output_block_size=output_block_size,
        reconstruction=None if pca is None else pca.reconstruction_,
        output_center=None if pca is None else pca.mean_,
        prediction_groups=groups,
        **options,
    )
