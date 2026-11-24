"""Load saved polynomial surrogates without pickled Python objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_json
from .solving.utils import construct_matrix_optimized, normalization_all


@dataclass(frozen=True)
class PolynomialSurrogate:
    """A scalar PCE stored as its coefficients, basis and input distributions."""

    coefficients: np.ndarray
    mapping: dict
    poly_matrix: np.ndarray
    mean: float
    std: float

    @classmethod
    def from_run(cls, directory: str | Path, budget: int | None = None) -> PolynomialSurrogate:
        directory = Path(directory)
        if budget is None:
            budget = read_json(directory / "run.json")["final"]
        basis = read_json(directory / "basis.json")
        with np.load(directory / "fits" / f"N{budget:04d}.npz", allow_pickle=False) as values:
            return cls(
                values["coef"].copy(),
                basis["mapping"],
                np.asarray(basis["poly_matrix"]),
                float(values["mean"]),
                float(values["std"]),
            )

    def predict(self, inputs: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
        """Evaluate rows of physical inputs in bounded chunks."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != len(self.poly_matrix):
            raise ValueError(f"Expected inputs with shape (n, {len(self.poly_matrix)})")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        active = np.flatnonzero(self.coefficients != 0)
        # Evaluate the selected columns in chunks to bound memory for large test sets.
        keys = list(self.mapping)
        mapping = {keys[index]: self.mapping[keys[index]] for index in active}
        prediction = np.empty(len(inputs))
        for start in range(0, len(inputs), chunk_size):
            matrix = construct_matrix_optimized(
                mapping, normalization_all(inputs[start : start + chunk_size], self.poly_matrix).T
            )
            prediction[start : start + len(matrix)] = matrix @ self.coefficients[active]
        return prediction
