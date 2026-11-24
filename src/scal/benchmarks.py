"""Analytical response functions in the frozen benchmark protocols."""

import numpy as np


def response(problem, x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if problem == "ishigami":
        return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
    if problem == "rackwitz":
        return x.shape[1] + 0.3 * np.sqrt(x.shape[1]) - x.sum(axis=1)
    if problem == "rosen":
        # The paper uses this weighted quartic response, not textbook Rosenbrock.
        return np.sum((x[:, :-1] - 1) ** 4, axis=1) + np.sum(
            np.sqrt(np.arange(2, x.shape[1] + 1)) * (x[:, 1:] - x[:, :-1] ** 2) ** 2, axis=1
        )
    if problem == "100d":
        if x.shape[1] != 100:
            raise ValueError("100d requires exactly 100 variables")
        i = np.arange(1, 101)
        return (
            3
            - 0.05 * np.sum(i * x, axis=1)
            + 0.01 * np.sum(i * x**3, axis=1)
            + (1 / 300) * np.sum(i * np.log(x**2 + x**4), axis=1)
            + x[:, 0] * x[:, 1] ** 2
            + x[:, 1] * x[:, 3]
            - x[:, 2] * x[:, 4]
            + x[:, 50]
            + x[:, 49] * x[:, 53] ** 2
        )
    raise ValueError(f"Unknown benchmark: {problem}")
