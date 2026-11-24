"""Classic Ishigami example: train, validate, save and reload.

Edit the settings below and run this file in your Python editor.
Optional terminal command: python examples/simple_function.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scal import FunctionConfig, FunctionExperiment, PolynomialSurrogate
from scal.reporting import save_report
from scal.io import read_json, write_json, write_npz


# Edit this block, then run the file. Paths stay relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/ishigami"
RESUME = False  # True restores the original settings and completed work in OUTPUT_DIR.
TRAINING = {
    "initial": 30,
    "budget": 120,  # Total training samples, including the initial samples.
    "pool_size": 16384,
    "degree": 12,
    "seed": 42,
}
TEST_SAMPLES = 4096  # Additional independent validation samples; 0 skips validation.
ISHIGAMI_PARAMETERS = {"a": 7.0, "b": 0.1}
POLY_MATRIX = np.array([[0, -np.pi, np.pi]] * 3)  # Three independent uniform inputs.


def response(inputs: np.ndarray) -> np.ndarray:
    """Evaluate sin(x1) + a sin(x2)^2 + b x3^4 sin(x1)."""
    # Inputs have shape (samples, variables); return one scalar response per row.
    x1, x2, x3 = inputs.T
    a, b = ISHIGAMI_PARAMETERS["a"], ISHIGAMI_PARAMETERS["b"]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--initial", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--test-samples", type=int)
    args = parser.parse_args()
    # 1. Match distribution rows to x1, x2 and x3; each is uniform on [-pi, pi].
    poly = POLY_MATRIX
    saved = read_json(args.output / "settings.json") if args.resume else None
    options = saved["run"].copy() if saved else TRAINING.copy()
    # 2. Restore saved settings on resume; the budget includes the initial observations.
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
    settings = FunctionConfig(**options)
    experiment = FunctionExperiment(
        response,
        poly,
        settings,
        name="ishigami",
        metadata={
            "validation_samples": test_samples,
            "ishigami_parameters": ISHIGAMI_PARAMETERS.copy(),
        },
    )

    def progress(step):
        if step.budget % 10 == 0 or step.budget == step.final_budget:
            print(f"N={step.budget}/{step.final_budget}  terms={step.selected_terms}")

    # 3. Fit, select one point, evaluate its response, append it, and repeat.
    result = experiment.run(args.output, progress=progress, resume=args.resume)
    # Reload the saved surrogate to demonstrate prediction in a separate workflow.
    model = PolynomialSurrogate.from_run(result.directory)
    # Analytic moments for three independent Uniform(-pi, pi) inputs.
    a, b = ISHIGAMI_PARAMETERS["a"], ISHIGAMI_PARAMETERS["b"]
    exact_mean = a / 2
    exact_std = float(np.sqrt(0.5 + a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18))
    report = {
        "training_evaluations": result.summary["training_response_evaluations"],
        "validation_evaluations": test_samples,
        "mean": model.mean,
        "std": model.std,
        "exact_mean": exact_mean,
        "exact_std": exact_std,
        "mean_standardized_error": abs(model.mean - exact_mean) / exact_std,
        "std_relative_error": abs(model.std - exact_std) / exact_std,
    }
    if test_samples:
        # 4. Use a separate random stream for validation after training has finished.
        generator = np.random.default_rng(np.random.SeedSequence([settings.seed, 20261102]))
        test_inputs = experiment.space.sample(test_samples, generator)
        validation_file = result.directory / "validation.npz"
        if args.resume and validation_file.exists():
            with np.load(validation_file, allow_pickle=False) as values:
                np.testing.assert_array_equal(values["X"], test_inputs)
                truth = values["y"].copy()
        else:
            truth = experiment.evaluate(test_inputs)
        prediction = model.predict(test_inputs)
        report["test_relmse"] = float(
            np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        )
        write_npz(validation_file, X=test_inputs, y=truth, prediction=prediction)
        np.savetxt(
            result.directory / "predictions.csv",
            np.column_stack([test_inputs, truth, prediction]),
            delimiter=",",
            header="x1,x2,x3,true_response,prediction",
            comments="",
            fmt="%.17g",
        )
    write_json(result.directory / "validation.json", report)
    figures = save_report(result.directory)
    print(json.dumps(report, indent=2))
    print("Reloaded model prediction at [0.2, 0.5, -0.3]:", model.predict([[0.2, 0.5, -0.3]])[0])
    print(f"Saved model and results: {result.directory}")
    print(f"Summary figure: {figures['figure_png']}")


if __name__ == "__main__":
    main()
