"""Evaluate a completed benchmark trajectory on independent test arrays."""

import sys
from scal.cli import main as cli_main
from scal.datasets import DatasetRepository
from scal.evaluation import BenchmarkEvaluator


def main(root, run_dir, output):
    return BenchmarkEvaluator(DatasetRepository(root)).evaluate(run_dir, output)


if __name__ == "__main__":
    raise SystemExit(cli_main(["evaluate", *sys.argv[1:]]))
