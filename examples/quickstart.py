"""Run the saved Ishigami benchmark with a configured benchmark workspace."""

import argparse
from pathlib import Path

from scal import BenchmarkEvaluator, DatasetRepository, Experiment, PolynomialSurrogate, RunConfig
from scal.verification import ReferenceComparison
from scal.reporting import save_report
from scal.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/quickstart"
FINAL_BUDGET = 120  # From 30 initial observations; use 35 for a shorter run.


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stop", type=int, default=FINAL_BUDGET)
    args = parser.parse_args()
    if not (root / "data/benchmark_inputs.json").is_file():
        parser.error("Use examples/simple_function.py to generate an Ishigami demonstration.")
    repository = DatasetRepository(root)
    experiment = Experiment(repository, RunConfig(problem="ishigami", stop=args.stop))
    result = experiment.run(args.output)
    comparison = ReferenceComparison(root / "data/expected/ishigami").compare(
        result.directory, result.directory / "comparison.json"
    )
    metrics = BenchmarkEvaluator(repository).evaluate(
        result.directory, result.directory / "metrics.csv"
    )
    model = PolynomialSurrogate.from_run(result.directory)
    inputs, truth = repository.test("ishigami")
    figures = save_report(result.directory, validation=(truth, model.predict(inputs)))
    write_json(
        result.directory / "quickstart.json",
        {
            "passed": comparison["passed"],
            "fits": comparison["fits"],
            "candidate_ids_exact": comparison["candidate_ids_exact"],
            "all_numeric_arrays_exact": comparison["all_numeric_arrays_exact"],
            "independent_test_samples": len(truth),
            "final_relmse": float(metrics.iloc[-1].relmse),
            "output": result.directory,
        },
    )
    print("Compared fits:", comparison["fits"])
    print(metrics.tail(1).to_string(index=False))
    print("Prediction at (0, 0, 0):", model.predict([[0.0, 0.0, 0.0]])[0])
    print("Summary figure:", figures["figure_png"])


if __name__ == "__main__":
    main()
