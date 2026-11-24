"""Exercise the settings a user expects to survive a stop and restart."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from scal import FunctionConfig, FunctionExperiment, __version__
from scal.cli import main
from scal.io import read_json


def load_example(name, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"user_example_{name.replace('/', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_cli_reports_the_installed_package_version(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == f"scal {__version__}"


@pytest.mark.parametrize("name", ["simple_function", "cantilever", "external_simulator/run"])
def test_run_button_uses_editable_settings_and_resumes_from_another_directory(
    tmp_path, monkeypatch, name
):
    module = load_example(name, monkeypatch)
    assert module.OUTPUT_DIR.is_absolute()
    run = tmp_path / "results"
    editor_directory = tmp_path / "editor"
    editor_directory.mkdir()
    monkeypatch.chdir(editor_directory)
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])
    monkeypatch.setattr(module, "OUTPUT_DIR", run)
    monkeypatch.setattr(
        module, "TRAINING", {"initial": 10, "budget": 12, "degree": 2, "pool_size": 64, "seed": 19}
    )
    monkeypatch.setattr(module, "TEST_SAMPLES", 2)
    module.main()
    assert read_json(run / "run.json")["final"] == 12
    settings = read_json(run / "settings.json")
    assert settings["run"]["seed"] == 19
    assert settings["metadata"]["validation_samples"] == 2
    assert (run / "summary.png").is_file()
    fits = {path.name: path.read_bytes() for path in (run / "fits").glob("*.npz")}

    monkeypatch.setattr(module, "RESUME", True)
    monkeypatch.setattr(module, "TRAINING", {"initial": 8, "budget": 9})
    monkeypatch.setattr(module, "TEST_SAMPLES", 0)
    module.main()
    assert read_json(run / "settings.json") == settings
    assert {path.name: path.read_bytes() for path in (run / "fits").glob("*.npz")} == fits
    if name == "external_simulator/run":
        jobs = read_json(run / "external_summary.json")
        assert jobs["total_solver_jobs"] == 14
        assert jobs["new_solver_jobs_this_invocation"] == 0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/benchmark_inputs.json").is_file(),
    reason="requires a separately supplied benchmark workspace",
)
def test_benchmark_run_button_uses_editable_budget(tmp_path, monkeypatch):
    module = load_example("quickstart", monkeypatch)
    assert module.OUTPUT_DIR.is_absolute()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(module, "FINAL_BUDGET", 32)
    module.main()
    assert read_json(tmp_path / "benchmark/run.json")["final"] == 32
    assert read_json(tmp_path / "benchmark/comparison.json")["candidate_ids_exact"]
    assert (tmp_path / "benchmark/summary.png").is_file()


@pytest.mark.parametrize("name", ["simple_function", "cantilever"])
@pytest.mark.parametrize("test_samples", [0, 8])
def test_examples_keep_validation_settings_and_reuse_completed_responses(
    tmp_path, monkeypatch, name, test_samples
):
    module = load_example(name, monkeypatch)
    evaluated = []
    if name == "simple_function":
        original = module.response

        def response(inputs):
            evaluated.extend(inputs.tolist())
            return original(inputs)

        monkeypatch.setattr(module, "response", response)
    else:
        original = module.Cantilever.response

        def response(model, inputs):
            evaluated.extend(inputs.tolist())
            return original(model, inputs)

        monkeypatch.setattr(module.Cantilever, "response", response)
    args = [name, "--output", str(tmp_path / "run")]
    monkeypatch.setattr(
        sys,
        "argv",
        args
        + [
            "--initial",
            "10",
            "--budget",
            "12",
            "--degree",
            "2",
            "--pool-size",
            "64",
            "--seed",
            "19",
            "--test-samples",
            str(test_samples),
        ],
    )
    module.main()
    assert len(evaluated) == 12 + test_samples
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    module.main()
    assert len(evaluated) == 12 + test_samples
    assert (
        read_json(tmp_path / "run/settings.json")["metadata"]["validation_samples"] == test_samples
    )


def test_cantilever_requires_enough_validation_samples_for_relmse(tmp_path, monkeypatch):
    module = load_example("cantilever", monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["cantilever", "--output", str(tmp_path / "run"), "--test-samples", "1"]
    )
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    assert not (tmp_path / "run").exists()


def test_bound_response_parameters_are_checked_on_resume(tmp_path):
    @dataclass(frozen=True)
    class Model:
        scale: float

        def response(self, inputs):
            return self.scale * (1 + inputs[:, 0] ** 2)

    config = FunctionConfig(initial=8, budget=9, pool_size=32, degree=2)
    FunctionExperiment(Model(1).response, [[0, -1, 1]], config).run(tmp_path / "run")
    with pytest.raises(ValueError, match="settings differ"):
        FunctionExperiment(Model(2).response, [[0, -1, 1]], config).run(
            tmp_path / "run", resume=True
        )


def test_documentation_edits_preserve_resume_but_calculation_edits_do_not(tmp_path):
    path = tmp_path / "response.py"

    def response_from(source):
        path.write_text(source, encoding="utf-8")
        module = types.ModuleType("user_response")
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module.response

    config = FunctionConfig(initial=8, budget=10, pool_size=32, degree=2)
    original = response_from("def response(X):\n    return 1 + X[:, 0] ** 2\n")
    first = FunctionExperiment(original, [[0, -1, 1]], config)

    def stop(step):
        if step.budget == 8:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        first.run(tmp_path / "run", progress=stop)
    documented = response_from(
        'def response(X):\n    """A scalar response for each physical input."""\n    # Square the first coordinate.\n    return 1 + X[:, 0] ** 2\n'
    )
    result = FunctionExperiment(documented, [[0, -1, 1]], config).run(tmp_path / "run", resume=True)
    assert result.summary["final"] == 10
    altered = response_from("def response(X):\n    return 2 + X[:, 0] ** 2\n")
    with pytest.raises(ValueError, match="settings differ"):
        FunctionExperiment(altered, [[0, -1, 1]], config).run(tmp_path / "run", resume=True)
