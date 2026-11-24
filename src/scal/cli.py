"""Command-line entry points for installing checks, experiments and analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

from .cohorts import CohortRunner
from .config import RunConfig, SAMPLERS
from .datasets import DatasetRepository
from .evaluation import BenchmarkEvaluator
from .experiment import Experiment, Progress
from .io import json_value, read_json, write_json
from .models import PolynomialSurrogate
from .reporting import save_report
from .solving.registry import SOLVERS
from .verification import ReferenceComparison, verify_release


def _progress(item: Progress) -> None:
    if item.budget % 10 == 0 or item.budget == item.final_budget:
        print(
            f"N={item.budget}/{item.final_budget}  terms={item.selected_terms}",
            file=sys.stderr,
            flush=True,
        )


def _repository(args) -> DatasetRepository:
    return DatasetRepository(args.root, getattr(args, "data_root", None))


def _run(args) -> dict:
    config = RunConfig.from_json(args.config) if args.config else RunConfig()
    overrides = {
        key: getattr(args, key) for key in asdict(config) if getattr(args, key) is not None
    }
    config = replace(config, **overrides)
    output = args.output or Path(
        f"outputs/{config.problem}_{config.solver}_{config.sampler}_seed{config.seed}"
    )
    result = Experiment(_repository(args), config).run(
        output, progress=None if args.quiet else _progress
    )
    return {**result.summary, "output": result.directory, "report": save_report(result.directory)}


def _benchmark_report(repository, directory, output=None):
    run = read_json(Path(directory) / "run.json")
    inputs, truth = repository.test(run["problem"])
    prediction = PolynomialSurrogate.from_run(directory).predict(inputs)
    return save_report(directory, output, validation=(truth, prediction))


def _quickstart(args) -> dict:
    repository = _repository(args)
    config = RunConfig(stop=args.stop)
    result = Experiment(repository, config).run(
        args.output, progress=None if args.quiet else _progress
    )
    comparison = ReferenceComparison(repository.root / "data/expected/ishigami").compare(
        result.directory, result.directory / "comparison.json"
    )
    metrics = BenchmarkEvaluator(repository).evaluate(
        result.directory, result.directory / "metrics.csv"
    )
    report = {
        "passed": True,
        "fits": comparison["fits"],
        "candidate_ids_exact": comparison["candidate_ids_exact"],
        "all_numeric_arrays_exact": comparison["all_numeric_arrays_exact"],
        "supports_exact": comparison["supports_exact"],
        "numeric_tolerances": comparison["numeric_tolerances"],
        "max_absolute_difference_by_array": comparison["max_absolute_difference_by_array"],
        "independent_test_samples": metrics.attrs["independent_test_samples"],
        "final_relmse": float(metrics.iloc[-1].relmse),
        "training_seconds": result.summary["elapsed_seconds"],
        "output": result.directory,
    }
    write_json(result.directory / "quickstart.json", report)
    report["report"] = _benchmark_report(repository, result.directory)
    return report


def _doctor(args) -> dict:
    repository = _repository(args)
    missing, mismatch, optional_missing = [], [], []
    for relative, entry in repository.entries.items():
        try:
            repository.input_path(relative)
        except FileNotFoundError:
            (missing if entry.get("included") else optional_missing).append(relative)
        except ValueError:
            mismatch.append(relative)
    packages = {
        name: version(name)
        for name in ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl", "matplotlib")
    }
    return {
        "passed": not missing and not mismatch,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "root": repository.root,
        "missing_required_inputs": missing,
        "input_size_mismatches": mismatch,
        "optional_training_files_missing": len(optional_missing),
    }


def _list(args) -> dict:
    repository = _repository(args)
    problems = [
        {
            "name": name,
            "dimension": repository.config.problem(name).dimension,
            "initial": repository.config.problem(name).initial,
            "maximum": repository.config.problem(name).maximum,
        }
        for name in repository.config.problem_names
    ]
    tasks = [
        {key: task[key] for key in ("id", "problem", "solver", "sampler")}
        for task in CohortRunner(repository).tasks.values()
    ]
    return {
        "benchmarks": problems,
        "solvers": list(SOLVERS),
        "samplers": list(SAMPLERS),
        "tasks": tasks,
    }


def _cohort(args) -> dict:
    seeds = None if args.all_seeds else [int(value.strip()) for value in args.seeds.split(",")]
    report = CohortRunner(_repository(args)).run(
        args.task,
        args.output,
        seeds=seeds,
        stop=args.stop,
        progress=None if args.quiet else _progress,
    )
    return {
        "task": report["task"],
        "seeds": report["seeds"],
        "completed_runs": len(report["runs"]),
        "output": args.output,
    }


def _evaluate(args) -> dict:
    repository = _repository(args)
    evaluator = BenchmarkEvaluator(repository)
    test_set = args.test_set or ("mc100k" if args.archived else "legacy")
    if args.archived:
        if args.problem is None or args.output is None:
            raise ValueError("Archived fits require --problem and --output")
        run_dir = args.run.parent.parent if args.run.is_file() else args.run
        basis = args.basis or run_dir / "basis.json"
        output = args.output
        frame = evaluator.evaluate_archive(args.run, args.problem, basis, output, test_set=test_set)
    else:
        if args.problem or args.basis:
            raise ValueError("--problem and --basis require --archived")
        filename = "metrics.csv" if test_set == "legacy" else "metrics_mc100k.csv"
        output = args.output or args.run / filename
        frame = evaluator.evaluate(args.run, output, test_set=test_set)
    result = {
        "evaluated_fits": len(frame),
        "independent_test_samples": frame.attrs["independent_test_samples"],
        "test_set_id": frame.attrs["test_set_id"],
        "output": output,
    }
    if not args.archived and test_set == "legacy":
        result["report"] = _benchmark_report(repository, args.run)
    return result


def _report(args) -> dict:
    run = read_json(args.run / "run.json")
    settings = read_json(args.run / "settings.json")
    # A custom response can share a benchmark name; use the saved configuration.
    benchmark = settings.get("run", {}).get("problem") == run["problem"]
    if benchmark and not (args.run / "validation.npz").is_file():
        return _benchmark_report(_repository(args), args.run, args.output)
    return save_report(args.run, args.output)


def _compare(args) -> dict:
    expected = args.expected
    if expected is None:
        run = read_json(args.run / "run.json")
        if run["seed"] != 0 or run["solver"] != "HCBPSolver" or run["sampler"] != "AHSC":
            raise ValueError(
                "Bundled references use seed 0 with HCBPSolver and AHSC; supply --expected for other runs"
            )
        expected = _repository(args).root / "data/expected" / run["problem"]
    report = ReferenceComparison(expected).compare(
        args.run, args.output or args.run / "comparison.json"
    )
    return {key: value for key, value in report.items() if key != "checks"}


def _analysis_scope(value: str) -> str:
    from .data import LEGACY_DATASET_VERSION

    prefix = f"v{LEGACY_DATASET_VERSION}"
    return {prefix: "comparisons", f"{prefix}-field": "field"}.get(value, value)


_CURRENT_DATA_REQUIRED = "Set --data-root to the directory containing source_catalog.json."
_HISTORICAL_ANALYSIS_INPUTS = {
    "paper": "validation_V1_curves.csv",
    "theta_exchange": "source_exchange_primary_curves.csv",
    "mechanism": "mechanism_curves.csv",
}


def _historical_data_required(scope: str) -> str:
    required = _HISTORICAL_ANALYSIS_INPUTS[scope]
    return (
        f"Scope '{scope}' requires the historical SCAL dataset. Set --data-root to its "
        f"data directory with '{required}' listed in source_catalog.json."
    )


def _analysis_target(args) -> tuple[str, Path]:
    from .data import Sources

    scope = args.scope
    if args.data_root is None and scope in ("comparisons", "field"):
        raise ValueError(_CURRENT_DATA_REQUIRED)
    try:
        sources = Sources(args.root, args.data_root)
    except FileNotFoundError as error:
        if scope in _HISTORICAL_ANALYSIS_INPUTS:
            raise ValueError(_historical_data_required(scope)) from error
        raise ValueError(_CURRENT_DATA_REQUIRED) from error

    names = {entry["name"] for entry in sources.catalog["numeric"]}
    if scope is None:
        if "closed_loop_metrics.csv" in names:
            scope = "comparisons"
        elif _HISTORICAL_ANALYSIS_INPUTS["paper"] in names:
            scope = "paper"
        elif args.data_root is not None:
            scope = "comparisons"
        else:
            raise ValueError(_CURRENT_DATA_REQUIRED)

    required = _HISTORICAL_ANALYSIS_INPUTS.get(scope)
    if required is not None and required not in names:
        raise ValueError(_historical_data_required(scope))
    return scope, sources.data_root


def _analyze(args) -> dict:
    from .analysis import PairedEffectAnalysis, PaperAnalysis
    from .analysis.current import CurrentAnalysis
    from .analysis.field import FieldReplay

    scope, data_root = _analysis_target(args)
    output = args.output or Path("outputs") / scope
    if scope in ("comparisons", "field"):
        analysis = FieldReplay if scope == "field" else CurrentAnalysis
        return analysis(data_root).run(output)
    if scope == "paper":
        report = PaperAnalysis(args.root, data_root).run(output)
        return {"passed": report["passed"], "checks": len(report["checks"]), "output": output}
    return PairedEffectAnalysis(args.root, data_root).run(output, scope)


def _verify(args) -> dict:
    return verify_release(_repository(args).root, args.ref)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scal", description="SCAL: sparse polynomial-chaos modelling with HCBP and AHSC."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('scal')}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        help="Research directory containing configs/ and data/ (default: current directory or SCAL_HOME)",
    )
    common.add_argument("--debug", action="store_true", help="Show a traceback on errors")
    commands = parser.add_subparsers(dest="command", required=True)

    quick = commands.add_parser(
        "quickstart", parents=[common], help="Run, compare and evaluate the seed-0 Ishigami example"
    )
    quick.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    quick.add_argument("--stop", type=int, help="Final sample count (30-120; default: 120)")
    quick.add_argument("--quiet", action="store_true")
    quick.set_defaults(handler=_quickstart)

    doctor = commands.add_parser(
        "doctor", parents=[common], help="Check the environment and available benchmark inputs"
    )
    doctor.add_argument("--data-root", type=Path)
    doctor.set_defaults(handler=_doctor)
    listing = commands.add_parser(
        "list", parents=[common], help="List benchmarks, methods and paper task IDs"
    )
    listing.set_defaults(handler=_list)

    run = commands.add_parser("run", parents=[common], help="Fit a single benchmark trajectory")
    run.add_argument(
        "--config", type=Path, help="Run settings in JSON; explicit options override it"
    )
    run.add_argument("--problem", choices=("ishigami", "rackwitz", "rosen", "100d"))
    run.add_argument("--seed", type=int)
    run.add_argument("--solver", choices=tuple(SOLVERS))
    run.add_argument("--sampler", choices=SAMPLERS)
    run.add_argument("--stop", type=int)
    run.add_argument("--output", type=Path)
    run.add_argument("--data-root", type=Path)
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(handler=_run)

    cohort = commands.add_parser(
        "cohort", parents=[common], help="Run selected seeds of a named paper task"
    )
    cohort.add_argument("--task", required=True)
    cohort.add_argument("--output", type=Path, required=True)
    cohort.add_argument("--data-root", type=Path)
    cohort.add_argument("--stop", type=int)
    cohort.add_argument("--quiet", action="store_true")
    seeds = cohort.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", default="0", help="Comma-separated seeds (default: 0)")
    seeds.add_argument("--all-seeds", action="store_true")
    cohort.set_defaults(handler=_cohort)

    evaluate = commands.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate a completed trajectory on independent test samples",
    )
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--data-root", type=Path)
    evaluate.add_argument(
        "--test-set",
        choices=("legacy", "mc100k"),
        help="Default: legacy for runs, mc100k for archives",
    )
    evaluate.add_argument(
        "--archived", action="store_true", help="Read archived fits without run.json"
    )
    evaluate.add_argument(
        "--problem", choices=("ishigami", "rackwitz", "rosen", "100d", "borehole", "weak_parent")
    )
    evaluate.add_argument("--basis", type=Path, help="Frozen basis.json for archived coefficients")
    evaluate.set_defaults(handler=_evaluate)

    compare = commands.add_parser(
        "compare", parents=[common], help="Compare fitted arrays and selected IDs with a saved run"
    )
    compare.add_argument("--run", "--actual", dest="run", type=Path, required=True)
    compare.add_argument("--expected", type=Path)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_compare)

    report = commands.add_parser(
        "report", parents=[common], help="Create summary figures and CSV tables from saved models"
    )
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, help="Report directory (default: the run directory)")
    report.add_argument("--data-root", type=Path)
    report.set_defaults(handler=_report)

    analyze = commands.add_parser(
        "analyze", parents=[common], help="Recompute paper summaries or paired effects"
    )
    analyze.add_argument(
        "--scope",
        type=_analysis_scope,
        choices=("paper", "theta_exchange", "mechanism", "comparisons", "field"),
        help="Default: comparisons for the current dataset, paper for the historical paper catalog",
    )
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--data-root", type=Path, help="Directory containing source_catalog.json")
    analyze.set_defaults(handler=_analyze)
    verify = commands.add_parser("verify", parents=[common], help="Check the tracked Git checkout")
    verify.add_argument("--ref", default="HEAD", help="Git tag or commit to compare with")
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = args.handler(args)
    except (OSError, ValueError, AssertionError, KeyError, RuntimeError) as error:
        if args.debug:
            raise
        print(f"scal: {error}", file=sys.stderr)
        print("Use --help for command options or --debug for a traceback.", file=sys.stderr)
        return 2
    print(json.dumps(json_value(report), indent=2, allow_nan=False))
    return 0 if report.get("passed", True) else 1
