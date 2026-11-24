"""Exercise real subprocesses, native file adapters and failed solver jobs."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scal import ExternalSimulator, FunctionConfig, FunctionExperiment, SimulationError
from scal.io import read_json

DEMO = Path(__file__).resolve().parents[1] / "examples/external_simulator/demo_solver.py"
NAMES = ["load_N", "length_m", "young_Pa", "height_m"]


def demo_simulator(work_root):
    return ExternalSimulator(
        [sys.executable, str(DEMO), "--input", "{input}", "--output", "{output}"], NAMES, work_root
    )


def test_external_jobs_use_separate_directories_and_physical_parameters(tmp_path):
    simulator = demo_simulator(tmp_path / "job files with spaces")
    values = simulator(np.array([[1000, 2, 200e9, 0.2], [2000, 2, 200e9, 0.2]]))
    np.testing.assert_allclose(values, [0.2, 0.4], rtol=1e-14)
    assert len(simulator.completed) == 2
    assert simulator.completed[0].directory != simulator.completed[1].directory
    job = simulator.completed[0].directory
    assert read_json(job / "input.json")["parameters"]["young_Pa"] == 200e9
    assert read_json(job / "job.json")["status"] == "completed"
    assert "Completed" in (job / "stdout.log").read_text()


def test_external_processes_follow_training_budget(tmp_path):
    output = tmp_path / "training"
    simulator = demo_simulator(output / "jobs")
    poly = [[1, 1000, 100], [0, 1.8, 2.2], [2, np.log(200e9), 0.05], [0, 0.18, 0.22]]
    config = FunctionConfig(initial=10, budget=12, pool_size=64, degree=2)
    result = FunctionExperiment(simulator, poly, config).run(output)
    assert len(simulator.completed) == result.summary["training_response_evaluations"] == 12
    with np.load(output / "observations.npz", allow_pickle=False) as arrays:
        observed = arrays["y"].copy()
    np.testing.assert_array_equal(observed, [job.response for job in simulator.completed])


def test_nonzero_exit_retains_stderr_and_fails(tmp_path):
    simulator = ExternalSimulator(
        [sys.executable, "-c", "import sys; print('solver failure',file=sys.stderr); sys.exit(7)"],
        ["x"],
        tmp_path,
    )
    with pytest.raises(SimulationError, match="code 7") as caught:
        simulator([[1]])
    job = caught.value.directory
    assert "solver failure" in (job / "stderr.log").read_text()
    assert read_json(job / "job.json")["returncode"] == 7
    assert not simulator.completed


def test_old_output_is_never_used_for_a_new_job(tmp_path):
    old = tmp_path / "case_000001"
    old.mkdir()
    (old / "output.json").write_text('{"response": 999}')
    simulator = ExternalSimulator(
        [sys.executable, "-c", "print('no result written')"], ["x"], tmp_path
    )
    with pytest.raises(SimulationError, match="did not produce") as caught:
        simulator([[1]])
    assert caught.value.directory.name == "case_000002"
    assert read_json(old / "output.json")["response"] == 999


@pytest.mark.parametrize("value", [float("nan"), [1, 2], True])
def test_invalid_external_result_stops_the_run(tmp_path, value):
    payload = json.dumps({"response": value})
    code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    simulator = ExternalSimulator(
        [sys.executable, "-c", code, "{output}", payload], ["x"], tmp_path
    )
    with pytest.raises(SimulationError):
        simulator([[1]])
    assert read_json(tmp_path / "case_000001/job.json")["status"] == "failed"


def test_timeout_is_reported_with_job_files(tmp_path):
    simulator = ExternalSimulator(
        [sys.executable, "-c", "import time; time.sleep(3)"], ["x"], tmp_path, timeout_seconds=0.2
    )
    with pytest.raises(SimulationError, match="exceeded") as caught:
        simulator([[1]])
    assert read_json(caught.value.directory / "job.json")["error_type"] == "TimeoutExpired"


def test_native_input_and_output_callbacks(tmp_path):
    def writer(path, parameters):
        path.write_text(str(parameters["x"]), encoding="utf-8")

    def reader(path):
        return float(path.read_text(encoding="utf-8"))

    code = "import pathlib,sys; x=float(pathlib.Path(sys.argv[1]).read_text()); pathlib.Path(sys.argv[2]).write_text(str(x*x))"
    simulator = ExternalSimulator(
        [sys.executable, "-c", code, "{input}", "{output}"],
        ["x"],
        tmp_path,
        input_name="model.dat",
        output_name="result.txt",
        input_writer=writer,
        output_reader=reader,
    )
    np.testing.assert_array_equal(simulator([[3], [4]]), [9, 16])


def test_job_metadata_filenames_are_reserved(tmp_path):
    with pytest.raises(ValueError, match="Reserved"):
        ExternalSimulator([sys.executable], ["x"], tmp_path, input_name="JOB.JSON")
