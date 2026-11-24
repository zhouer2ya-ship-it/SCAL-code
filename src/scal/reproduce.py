"""Convenience entry point for benchmark reproduction.

New code can use Experiment and RunConfig, or the scal run command.
"""

from pathlib import Path
import sys

from .acquisition import SamplingPolicy
from .cli import main as cli_main
from .config import RunConfig
from .experiment import Experiment
from .io import digest, json_value, write_json

__all__ = ["run", "select", "digest", "json_value", "write_json"]


def select(method, model, x, y, pool, ids, poly):
    selection = SamplingPolicy(method, poly).select(model, x, y, pool, ids)
    return selection.point, selection.record


def run(
    root,
    problem="ishigami",
    seed=0,
    solver="HCBPSolver",
    sampler="AHSC",
    stop=None,
    output=None,
    data_root=None,
):
    config = RunConfig(problem, seed, solver, sampler, stop)
    output = output or Path(root) / f"outputs/{problem}_{solver}_{sampler}_seed{seed}"
    return Experiment.from_directory(root, config, data_root=data_root).run(output).summary


def main():
    return cli_main(["run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
