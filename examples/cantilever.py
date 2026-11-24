"""A mixed-input cantilever example with saved models and independent validation.

Edit the settings below and run this file in your Python editor.
Optional terminal command: python examples/cantilever.py
Inputs: load P [N], length L [m], Young's modulus E [Pa], height h [m].
Output: tip displacement [mm]. The rectangular beam width is 0.1 m.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scal import FunctionConfig, FunctionExperiment, PolynomialSurrogate
from scal.reporting import save_report
from scal.io import read_json, write_json, write_npz


# Edit this block, then run the file. Paths stay relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/cantilever"
RESUME = False  # True restores the original settings and completed work in OUTPUT_DIR.
TRAINING = {
    "initial": 24,
    "budget": 60,  # Total training samples, including the initial samples.
    "pool_size": 4096,
    "degree": 4,
    "seed": 42,
}
TEST_SAMPLES = 4096  # Additional independent validation samples; 0 skips validation.


@dataclass(frozen=True)
class Cantilever:
    """Euler-Bernoulli cantilever with a tip load and independent random inputs."""

    width: float = 0.1
    young_mean: float = 200e9
    young_cv: float = 0.05

    @property
    def poly_matrix(self) -> np.ndarray:
        # Convert the physical mean and coefficient of variation of E to log-space parameters.
        log_variance = np.log1p(self.young_cv**2)
        return np.array(
            [
                [1, 1000.0, 100.0],  # P: normal, N
                [0, 1.8, 2.2],  # L: uniform, m
                [2, np.log(self.young_mean) - log_variance / 2, np.sqrt(log_variance)],
                [0, 0.18, 0.22],  # h: uniform, m
            ]
        )

    def response(self, inputs: np.ndarray) -> np.ndarray:
        """Return one deflection per row. Replace this method for your own model."""
        load, length, young, height = inputs.T
        inertia = self.width * height**3 / 12
        # Convert the SI displacement from metres to millimetres.
        return 1000 * load * length**3 / (3 * young * inertia)

    def exact_moments(self) -> tuple[float, float]:
        """Analytic moments under the four independent input distributions."""

        def uniform_moment(lower, upper, power):
            return (upper ** (power + 1) - lower ** (power + 1)) / ((power + 1) * (upper - lower))

        _, log_mean, log_std = self.poly_matrix[2]
        scale = 4000 / self.width
        mean = (
            scale
            * 1000
            * uniform_moment(1.8, 2.2, 3)
            * np.exp(-log_mean + log_std**2 / 2)
            * uniform_moment(0.18, 0.22, -3)
        )
        second = (
            scale**2
            * (1000**2 + 100**2)
            * uniform_moment(1.8, 2.2, 6)
            * np.exp(-2 * log_mean + 2 * log_std**2)
            * uniform_moment(0.18, 0.22, -6)
        )
        return float(mean), float(np.sqrt(second - mean**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--initial", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Independent validation calls after training; 0 skips them",
    )
    args = parser.parse_args()
    case = Cantilever()
    # Restore the training settings and validation count for the same experiment.
    saved = read_json(args.output / "settings.json") if args.resume else None
    options = saved["run"].copy() if saved else TRAINING.copy()
    test_samples = saved["metadata"]["validation_samples"] if saved else TEST_SAMPLES
    if args.test_samples is not None:
        if saved and args.test_samples != test_samples:
            raise ValueError("Keep the saved --test-samples count when resuming")
        test_samples = args.test_samples
    if type(test_samples) is not int or test_samples < 0 or test_samples == 1:
        parser.error("TEST_SAMPLES / --test-samples must be 0 or at least 2")
    for key in ("initial", "budget", "pool_size", "degree", "seed"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    config = FunctionConfig(**options)
    experiment = FunctionExperiment(
        case.response,
        case.poly_matrix,
        config,
        name="cantilever",
        metadata={"validation_samples": test_samples},
    )

    def progress(step):
        if step.budget % 10 == 0 or step.budget == step.final_budget:
            print(f"N={step.budget}/{step.final_budget}  terms={step.selected_terms}")

    result = experiment.run(args.output, progress=progress, resume=args.resume)
    model = PolynomialSurrogate.from_run(result.directory)
    exact_mean, exact_std = case.exact_moments()
    report = {
        "response_unit": "mm",
        "training_calls": result.summary["training_response_evaluations"],
        "validation_calls": test_samples,
        "surrogate_mean": model.mean,
        "surrogate_std": model.std,
        "exact_mean": exact_mean,
        "exact_std": exact_std,
        "mean_standardized_error": abs(model.mean - exact_mean) / exact_std,
        "std_relative_error": abs(model.std - exact_std) / exact_std,
    }
    if test_samples:
        # A separate RNG stream is used only after the final model is fitted.
        test_seed = np.random.SeedSequence([config.seed, 20261101])
        inputs = experiment.space.sample(test_samples, np.random.default_rng(test_seed))
        validation_file = result.directory / "validation.npz"
        if args.resume and validation_file.exists():
            with np.load(validation_file, allow_pickle=False) as values:
                np.testing.assert_array_equal(values["X"], inputs)
                truth = values["y"].copy()
        else:
            truth = experiment.evaluate(inputs)
        prediction = model.predict(inputs)
        report["test_relmse"] = float(
            np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        )
        write_npz(validation_file, X=inputs, y=truth, prediction=prediction)
    write_json(result.directory / "validation.json", report)
    figures = save_report(result.directory)
    print(json.dumps(report, indent=2))
    print(f"Saved model and results: {result.directory}")
    print(f"Summary figure: {figures['figure_png']}")


if __name__ == "__main__":
    main()
