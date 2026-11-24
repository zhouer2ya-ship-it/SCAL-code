"""Analyse the saved SRAM and aneurysm surrogate models and responses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import eval_hermitenorm, eval_legendre
from scipy.stats import ks_2samp

from .sampling.theta import standardize_inputs


def design(x: np.ndarray, degrees: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Evaluate Legendre/Hermite products at physical input rows."""
    xi, families = standardize_inputs(x, poly)
    degrees = np.asarray(degrees, dtype=int)
    if degrees.ndim != 2 or degrees.shape[1] != xi.shape[1] or np.any(degrees < 0):
        raise ValueError("Invalid basis degrees")
    result = np.ones((len(xi), len(degrees)))
    for column in range(xi.shape[1]):
        for degree in np.unique(degrees[:, column]):
            if degree:
                evaluator = eval_legendre if families[column] == "legendre" else eval_hermitenorm
                values = evaluator(degree, xi[:, column])
                result[:, degrees[:, column] == degree] *= values[:, None]
    return result


class SRAMAnalysis:
    """Moments, distribution errors and Sobol indices for the stored SRAM models."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as arrays:
            self.data = {key: arrays[key] for key in arrays.files}
        self.coefficients = self.data["coefficients"]
        self.degrees = self.data["basis_degrees"]
        self.constant = int(np.flatnonzero(~np.any(self.degrees, axis=1))[0])
        self.energy = self.coefficients * self.coefficients * self.data["basis_norms"]
        self.energy[:, self.constant] = 0
        self.truth = self.data["truth"]
        self.reference_std = self.truth.std(ddof=0)

    def verify_predictions(self) -> None:
        matrix = design(
            self.data["verification_inputs"], self.degrees, np.tile([1, 0, 1.0], (159, 1))
        )
        prediction = matrix @ self.coefficients[self.data["prediction_budgets"] - 200].T
        np.testing.assert_allclose(
            prediction, self.data["verification_prediction"], rtol=2e-12, atol=2e-14
        )

    def moments(self) -> dict:
        mean = self.coefficients[:, self.constant]
        std = np.sqrt(self.energy.sum(axis=1))
        return {
            "budgets": self.data["budgets"],
            "mean": mean,
            "sd": std,
            "e_mu": abs(mean - self.truth.mean()) / self.reference_std,
            "e_sigma": abs(std - self.reference_std) / self.reference_std,
            "terms": self.data["support"].sum(axis=1),
        }

    def distributions(self) -> list[dict]:
        rows = []
        reference = np.quantile(self.truth, [0.95, 0.99], method="linear")
        for budget, prediction in zip(self.data["prediction_budgets"], self.data["prediction"].T):
            quantiles = np.quantile(prediction, [0.95, 0.99], method="linear")
            rows.append(
                {
                    "N": int(budget),
                    "CDF_distance": float(
                        ks_2samp(prediction, self.truth, method="asymp").statistic
                    ),
                    "q95": quantiles[0],
                    "q99": quantiles[1],
                    "e_q95": abs(quantiles[0] - reference[0]) / self.reference_std,
                    "e_q99": abs(quantiles[1] - reference[1]) / self.reference_std,
                    "RelMSE_direct": float(
                        np.sum((prediction - self.truth) ** 2)
                        / np.sum((self.truth - self.truth.mean()) ** 2)
                    ),
                }
            )
        return rows

    def sobol_indices(self) -> tuple[np.ndarray, np.ndarray]:
        variance = self.energy[-1].sum()
        first_order, total_order = np.zeros(159), np.zeros(159)
        for index, degree in enumerate(self.degrees):
            used = np.flatnonzero(degree)
            total_order[used] += self.energy[-1, index] / variance
            if len(used) == 1:
                first_order[used] += self.energy[-1, index] / variance
        return first_order, total_order

    def calculate(self) -> dict:
        self.verify_predictions()
        first, total = self.sobol_indices()
        return {
            **self.moments(),
            "distribution": self.distributions(),
            "first_order": first,
            "total_order": total,
            "training_observations": len(self.data["training_y"]),
        }


class FieldAnalysis:
    """Reconstruct PCA field predictions and compute case and population moments."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as arrays:
            self.data = {key: arrays[key] for key in arrays.files}

    def predict(self) -> np.ndarray:
        data = self.data
        matrix = design(data["test_inputs"], data["basis_degrees"], data["poly_matrix"])
        prediction = (matrix @ data["coefficients"]) @ data["pca_reconstruction"].T + data[
            "pca_mean"
        ]
        np.testing.assert_allclose(
            prediction[data["verification_case_ids"] - 1],
            data["verification_prediction"],
            rtol=3e-12,
            atol=1e-8,
        )
        return prediction

    def population_moments(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.data
        coefficients, reconstruction = data["coefficients"], data["pca_reconstruction"]
        constant = int(np.flatnonzero(~np.any(data["basis_degrees"], axis=1))[0])
        nonconstant = np.arange(len(coefficients)) != constant
        covariance = coefficients[nonconstant].T @ (
            data["basis_norms"][nonconstant, None] * coefficients[nonconstant]
        )
        mean = data["pca_mean"] + reconstruction @ coefficients[constant]
        std = np.sqrt(
            np.maximum(0, np.einsum("ij,jk,ik->i", reconstruction, covariance, reconstruction))
        )
        return mean, std

    def calculate(self, *, case_ids: tuple[int, int] | None = None) -> dict:
        prediction, truth = self.predict(), self.data["truth"]
        errors = np.mean((prediction - truth) ** 2, axis=1) / truth.var(axis=1)
        reference_mean, reference_std = truth.mean(axis=0), truth.std(axis=0)
        mean, std = prediction.mean(axis=0), prediction.std(axis=0)
        population_mean, population_std = self.population_moments()
        typical = int(np.argmin(abs(errors - np.median(errors)))) + 1
        worst = int(np.argmax(errors)) + 1
        if case_ids is not None:
            candidates = list(self.data["verification_case_ids"])
            if list(case_ids) != candidates or case_ids[1] != worst:
                raise ValueError("Display cases do not match the saved field verification")
            typical, worst = case_ids
        return {
            "case_relmse": errors,
            "mean_relmse": float(errors.mean()),
            "median_relmse": float(np.median(errors)),
            "max_relmse": float(errors.max()),
            "typical_case": typical,
            "worst_case": worst,
            "positions": truth.shape[1],
            "test_cases": len(truth),
            "mean_field_relative_l2": float(
                np.linalg.norm(mean - reference_mean) / np.linalg.norm(reference_mean)
            ),
            "sd_field_relative_l2": float(
                np.linalg.norm(std - reference_std) / np.linalg.norm(reference_std)
            ),
            "population_mean": population_mean,
            "population_sd": population_std,
        }


def sram_statistics(path: str | Path) -> dict:
    """Compatibility entry point; see SRAMAnalysis for individual calculations."""
    return SRAMAnalysis(path).calculate()


def field_statistics(path: str | Path) -> dict:
    """Compatibility entry point; see FieldAnalysis for individual calculations."""
    return FieldAnalysis(path).calculate()
