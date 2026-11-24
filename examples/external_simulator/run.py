"""Run SCAL with a separate command-line process for each response.

Edit the settings below and run this file in your Python editor.
Optional terminal command: python examples/external_simulator/run.py
The default command invokes the bundled demonstration solver. An adapter JSON
file can replace that command while retaining the named-parameter JSON format.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from scal import ExternalSimulator, FunctionConfig, FunctionExperiment, PolynomialSurrogate
from scal.external import read_response_json, write_parameter_json
from scal.io import read_json, write_json, write_npz


# Edit this block, then run the file. Paths stay relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/external"
RESUME = False  # True restores the saved adapter, settings and completed responses.
TRAINING = {
    "initial": 24,
    "budget": 60,  # Total training samples, including the initial samples.
    "pool_size": 4096,
    "degree": 4,
    "seed": 42,
}
TEST_SAMPLES = 0  # Each validation sample starts an additional external solver job.
ADAPTER_FILE = Path(__file__).with_name("adapter.json")  # Or adapter_text.json.
TIMEOUT_SECONDS = None  # None uses the adapter's per-job timeout.

# Match these input rows to parameter_names in the adapter.
LOG_VARIANCE = np.log1p(0.05**2)
POLY_MATRIX = np.array(
    [
        [1, 1000, 100],
        [0, 1.8, 2.2],
        [2, np.log(200e9) - LOG_VARIANCE / 2, np.sqrt(LOG_VARIANCE)],
        [0, 0.18, 0.22],
    ]
)


def write_native_input(path: Path, parameters: dict[str, float]) -> None:
    """Edit this writer to produce your solver's native input file."""
    # Fill the solver template with named parameters at full floating-point precision.
    path.write_text(
        "".join(f"{name}={value:.17g}\n" for name, value in parameters.items()), encoding="utf-8"
    )


def read_native_response(path: Path) -> float:
    """Edit this reader to extract one result and convert it to the desired unit."""
    # Read the specified response and unit consistently across all fits.
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["quantity"] == "tip_displacement"]
    if len(rows) != 1 or rows[0]["unit"] != "mm":
        raise ValueError("Expected one tip_displacement result in mm")
    return float(rows[0]["value"])


def make_simulator(
    output: Path, adapter: dict, timeout: float | None, retry_unconfirmed=False
) -> ExternalSimulator:
    substitutions = {
        "{python}": sys.executable,
        "{demo_solver}": str(Path(__file__).with_name("demo_solver.py").resolve()),
    }
    command = []
    # Pass command arguments as a list so paths containing spaces remain intact.
    for argument in adapter["command"]:
        for token, value in substitutions.items():
            argument = argument.replace(token, value)
        command.append(argument)
    file_format = adapter.get("format", "json")
    if file_format not in ("json", "text"):
        raise ValueError("Adapter format must be json or text")
    return ExternalSimulator(
        command,
        adapter["parameter_names"],
        output / "solver_jobs",
        timeout_seconds=timeout if timeout is not None else adapter["timeout_seconds"],
        input_name=adapter.get("input_name", "input.json"),
        output_name=adapter.get("output_name", "output.json"),
        input_writer=write_native_input if file_format == "text" else write_parameter_json,
        output_reader=read_native_response if file_format == "text" else read_response_json,
        model_id=adapter.get("model_id"),
        model_files=adapter.get("model_files", []),
        retry_unconfirmed=retry_unconfirmed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--initial", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume", action="store_true", default=RESUME, help="Continue the saved run in --output"
    )
    parser.add_argument(
        "--retry-unconfirmed",
        action="store_true",
        help="Retry an unconfirmed job only after checking its solver has stopped",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_SECONDS,
        help="Maximum seconds for each solver process",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Additional external validation jobs after training (default: 0)",
    )
    args = parser.parse_args()
    # 1. Read the adapter for a new run, or restore the saved command and file formats.
    saved = read_json(args.output / "settings.json") if args.resume else None
    if args.adapter is not None:
        adapter = read_json(args.adapter)
    elif saved is not None:
        identity = saved["response"]
        adapter = {
            "command": identity["command"],
            "parameter_names": identity["parameters"],
            "input_name": identity["input"],
            "output_name": identity["output"],
            "model_id": identity["model_id"],
            "model_files": list(identity["model_files"]),
            "format": "text"
            if identity["output_reader"]["name"] == "read_native_response"
            else "json",
            "timeout_seconds": saved["response_options"]["timeout_seconds"],
        }
    else:
        adapter = read_json(ADAPTER_FILE)
    simulator = make_simulator(args.output, adapter, args.timeout, args.retry_unconfirmed)
    saved_test_samples = saved["metadata"]["validation_samples"] if saved else TEST_SAMPLES
    test_samples = args.test_samples if args.test_samples is not None else saved_test_samples
    if saved and test_samples != saved_test_samples:
        raise ValueError("Keep the original --test-samples count when resuming")
    if type(test_samples) is not int or test_samples < 0 or test_samples == 1:
        parser.error("TEST_SAMPLES / --test-samples must be 0 or at least 2")
    # 2. Match physical input columns to parameter_names in adapter.json.
    poly = POLY_MATRIX
    options = saved["run"].copy() if saved else TRAINING.copy()
    for key in ("initial", "budget", "pool_size", "degree", "seed"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    settings = FunctionConfig(**options)
    experiment = FunctionExperiment(
        simulator,
        poly,
        settings,
        name="external_cantilever",
        metadata={"validation_samples": test_samples},
    )

    def progress(step):
        if step.budget % 10 == 0 or step.budget == step.final_budget:
            print(f"N={step.budget}/{step.final_budget}  terms={step.selected_terms}")

    # 3. Evaluate each selected point, reusing responses from completed jobs.
    result = experiment.run(args.output, progress=progress, resume=args.resume)
    validation_settings = result.directory / "validation_settings.json"
    write_json(validation_settings, {"test_samples": test_samples})
    model = PolynomialSurrogate.from_run(result.directory)
    report = {
        "training_solver_jobs": result.summary["training_response_evaluations"],
        "validation_solver_jobs": test_samples,
        "surrogate_mean_mm": model.mean,
        "surrogate_std_mm": model.std,
        "nominal_prediction_mm": float(model.predict([[1000, 2, 200e9, 0.2]])[0]),
    }
    if test_samples:
        # 4. Validate after training; the shared response cache also supports validation recovery.
        generator = np.random.default_rng(np.random.SeedSequence([settings.seed, 20261101]))
        inputs = experiment.space.sample(test_samples, generator)
        truth = experiment.evaluate(inputs)
        prediction = model.predict(inputs)
        report["test_relmse"] = float(
            np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        )
        write_npz(result.directory / "validation.npz", X=inputs, y=truth, prediction=prediction)
    jobs = [read_json(path) for path in simulator.work_root.glob("case_*/job.json")]
    # Count training observations separately from solver attempts, including retries.
    report["total_solver_jobs"] = len(jobs)
    report["completed_solver_jobs"] = sum(job["status"] == "completed" for job in jobs)
    report["failed_solver_jobs"] = sum(job["status"] in {"failed", "abandoned"} for job in jobs)
    report["new_solver_jobs_this_invocation"] = simulator.new_jobs
    report["cache_hits_this_invocation"] = simulator.cache_hits
    write_json(result.directory / "external_summary.json", report)
    write_json(result.directory / "adapter_used.json", adapter)
    from scal.reporting import save_report

    figures = save_report(result.directory)
    print(json.dumps(report, indent=2))
    print(f"Solver inputs, outputs and logs: {simulator.work_root}")
    print(f"Summary figure: {figures['figure_png']}")


if __name__ == "__main__":
    main()
