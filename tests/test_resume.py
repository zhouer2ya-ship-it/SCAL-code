"""Interrupted expensive jobs resume without changing the adaptive trajectory."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from scal import ExternalSimulator, FunctionConfig, FunctionExperiment, SimulationError
from scal._persistence import FileLock
from scal.io import read_json, write_json

POLY = [[0, -1, 1], [0, -1, 1]]
CONFIG = FunctionConfig(initial=10, budget=13, pool_size=64, degree=2, seed=17)
SOLVER = """import json, math, pathlib, sys, time
input_path, output_path, ledger_path, fail_at, delay = sys.argv[1:]
ledger = pathlib.Path(ledger_path)
entries = ledger.read_text().splitlines() if ledger.exists() else []
p = json.loads(pathlib.Path(input_path).read_text())["parameters"]
attempt = len(entries) + 1
with ledger.open("a") as stream:
    stream.write(json.dumps(p) + "\\n")
time.sleep(float(delay))
if attempt == int(fail_at):
    sys.exit(7)
x, y = p["x"], p["y"]
value = 2 + x*x + .3*y + .2*x*y*y
pathlib.Path(output_path).write_text(json.dumps({"response": value}))
"""


def make_solver(root, fail_at=0, delay=0, timeout=300):
    script = root / "solver.py"
    if not script.exists():
        root.mkdir(parents=True, exist_ok=True)
        script.write_text(SOLVER, encoding="utf-8")
    return ExternalSimulator(
        [
            sys.executable,
            str(script),
            "{input}",
            "{output}",
            str(root / "calls.jsonl"),
            str(fail_at),
            str(delay),
        ],
        ["x", "y"],
        root / "run/jobs",
        timeout_seconds=timeout,
    )


def experiment(simulator):
    return FunctionExperiment(simulator, POLY, CONFIG)


def assert_same_run(left, right):
    assert (
        read_json(left / "run.json")["selected_ids"]
        == read_json(right / "run.json")["selected_ids"]
    )
    for path in sorted((left / "fits").glob("*.npz")) + [left / "observations.npz"]:
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(right / path.relative_to(left), allow_pickle=False) as b,
        ):
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("fail_at", [4, 12])
def test_external_failure_resumes_initial_or_adaptive_point_once(tmp_path, fail_at):
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    experiment(make_solver(continuous)).run(continuous / "run")
    with pytest.raises(SimulationError, match="code 7"):
        experiment(make_solver(interrupted, fail_at)).run(interrupted / "run")
    pending = None
    if fail_at > CONFIG.initial:
        pending = read_json(interrupted / "run/fits/N0011.json")["next"]
        assert "response" not in pending
    resumed = make_solver(interrupted, fail_at)
    result = experiment(resumed).run(interrupted / "run", resume=True)
    assert result.summary["resumed"]
    assert_same_run(continuous / "run", interrupted / "run")
    jobs = [read_json(p) for p in (interrupted / "run/jobs").glob("case_*/job.json")]
    successful = [job["point_sha256"] for job in jobs if job["status"] == "completed"]
    assert len(successful) == len(set(successful)) == CONFIG.budget
    assert len(jobs) == CONFIG.budget + 1
    if pending:
        assert (
            read_json(interrupted / "run/fits/N0011.json")["next"]["candidate_id"]
            == pending["candidate_id"]
        )
    finished = make_solver(interrupted, fail_at)
    experiment(finished).run(interrupted / "run", resume=True)
    assert finished.new_jobs == 0 and finished.completed == []


def test_finished_output_survives_parent_interruption_before_response_commit(tmp_path, monkeypatch):
    simulator = make_solver(tmp_path)

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(simulator, "_finish_result", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment(simulator).run(tmp_path / "run")
    job = tmp_path / "run/jobs/case_000001"
    assert read_json(job / "exit_status.json")["returncode"] == 0
    assert read_json(job / "job.json")["status"] == "submitted"
    restored = make_solver(tmp_path)
    experiment(restored).run(tmp_path / "run", resume=True)
    assert restored.new_jobs == CONFIG.budget - 1
    assert restored.cache_hits == 1
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == CONFIG.budget


def test_resume_after_fit_interruption_does_not_evaluate_old_responses(tmp_path, monkeypatch):
    calls = []

    def response(x):
        calls.extend(x.tolist())
        return 2 + x[:, 0] ** 2 + 0.3 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1] ** 2

    trial = FunctionExperiment(response, POLY, CONFIG)
    from scal.functions import HCBPSolver

    original = HCBPSolver.fit

    def fail(model, inputs, values):
        if len(inputs) == 12:
            raise KeyboardInterrupt()
        return original(model, inputs, values)

    monkeypatch.setattr(HCBPSolver, "fit", fail)
    with pytest.raises(KeyboardInterrupt):
        trial.run(tmp_path / "run")
    assert len(calls) == 12
    monkeypatch.setattr(HCBPSolver, "fit", original)
    trial.run(tmp_path / "run", resume=True)
    assert len(calls) == CONFIG.budget


def test_changed_settings_and_corrupted_cache_are_rejected(tmp_path):
    simulator = make_solver(tmp_path)
    run = tmp_path / "run"
    experiment(simulator).run(run)
    changed = FunctionConfig(initial=10, budget=13, pool_size=64, degree=2, seed=18)
    with pytest.raises(ValueError, match="settings differ"):
        FunctionExperiment(make_solver(tmp_path), POLY, changed).run(run, resume=True)
    with pytest.raises(FileExistsError):
        experiment(simulator).run(run)
    job = simulator.completed[0].directory
    (job / "output.json").write_text('{"response": 999}')
    with pytest.raises(SimulationError, match="checksum"):
        make_solver(tmp_path)([list(read_json(job / "job.json")["parameters"].values())])
    with FileLock(run / ".run.lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            experiment(simulator).run(run, resume=True)


def test_unconfirmed_worker_requires_explicit_retry(tmp_path):
    simulator = make_solver(tmp_path)
    simulator([[0.2, 0.3]])
    job = simulator.completed[0].directory
    record = read_json(job / "job.json")
    record["status"] = "submitted"
    write_json(job / "job.json", record)
    (job / "exit_status.json").rename(job / "saved_exit_status.json")
    with pytest.raises(SimulationError, match="unconfirmed"):
        make_solver(tmp_path)([[0.2, 0.3]])
    retry = make_solver(tmp_path)
    retry.retry_unconfirmed = True
    retry([[0.2, 0.3]])
    assert retry.new_jobs == 1


def test_killed_parent_rejoins_independent_running_solver(tmp_path):
    simulator = make_solver(tmp_path, delay=1.0)
    helper = tmp_path / "parent.py"
    helper.write_text(
        "from scal import ExternalSimulator\n"
        f"s=ExternalSimulator({list(simulator.command)!r}, ['x','y'], {str(simulator.work_root)!r})\n"
        "s([[0.2,0.3]])\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(helper)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "calls.jsonl").exists():
            assert process.poll() is None, "Parent exited before launching solver"
            if time.monotonic() > deadline:
                pytest.fail("Solver did not start")
            time.sleep(0.02)
        process.kill()
        process.wait(timeout=5)
        recovered = make_solver(tmp_path, delay=1.0)
        actual = recovered([[0.2, 0.3]])
        np.testing.assert_allclose(actual, [2 + 0.2**2 + 0.3 * 0.3 + 0.2 * 0.2 * 0.3**2])
        assert recovered.new_jobs == 0 and recovered.cache_hits == 1
        assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("sampler", ["AHSC", "Theta", "SOptimal", "MAXDistance", "MAXGradient"])
def test_warm_start_trajectory_is_exact_for_all_custom_samplers(tmp_path, sampler):
    def response(x):
        return 2 + x[:, 0] ** 2 + 0.3 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1] ** 2

    config = FunctionConfig(initial=10, budget=14, pool_size=64, degree=2, seed=17, sampler=sampler)
    trial = FunctionExperiment(response, POLY, config)
    trial.run(tmp_path / "continuous")

    def stop(step):
        if step.budget == 11:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        trial.run(tmp_path / "interrupted", progress=stop)
    trial.run(tmp_path / "interrupted", resume=True)
    assert_same_run(tmp_path / "continuous", tmp_path / "interrupted")


def test_timeout_can_increase_on_resume(tmp_path):
    with pytest.raises(SimulationError, match="exceeded"):
        experiment(make_solver(tmp_path, delay=0.1, timeout=0.01)).run(tmp_path / "run")
    simulator = make_solver(tmp_path, delay=0.1, timeout=3)
    experiment(simulator).run(tmp_path / "run", resume=True)
    assert simulator.new_jobs == CONFIG.budget


def test_changed_solver_source_is_rejected_before_new_work(tmp_path):
    simulator = make_solver(tmp_path, fail_at=4)
    with pytest.raises(SimulationError):
        experiment(simulator).run(tmp_path / "run")
    with (tmp_path / "solver.py").open("a") as stream:
        stream.write("\n# Changed solver version\n")
    with pytest.raises(ValueError, match="settings differ"):
        experiment(make_solver(tmp_path, fail_at=4)).run(tmp_path / "run", resume=True)
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("adapter_name", ["adapter.json", "adapter_text.json"])
def test_example_cli_recovers_original_options_and_validation(tmp_path, adapter_name):
    examples = Path(__file__).resolve().parents[1] / "examples/external_simulator"
    command = [sys.executable, str(examples / "run.py"), "--output", str(tmp_path / "run")]
    failed = subprocess.run(
        command
        + [
            "--adapter",
            str(examples / adapter_name),
            "--initial",
            "10",
            "--budget",
            "12",
            "--pool-size",
            "64",
            "--degree",
            "2",
            "--seed",
            "19",
            "--test-samples",
            "2",
            "--timeout",
            "0.001",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode != 0 and "exceeded" in failed.stderr
    completed = subprocess.run(
        command + ["--resume", "--timeout", "10"], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    summary = read_json(tmp_path / "run/external_summary.json")
    assert summary["training_solver_jobs"] == 12
    assert summary["validation_solver_jobs"] == 2
    assert summary["total_solver_jobs"] == 15
    assert summary["failed_solver_jobs"] == 1
    repeated = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    summary = read_json(tmp_path / "run/external_summary.json")
    assert summary["new_solver_jobs_this_invocation"] == 0
    assert summary["total_solver_jobs"] == 15
