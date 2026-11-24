"""Exercise the public workflow, saved-model loading and common input errors."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scal import DatasetRepository, Experiment, PolynomialSurrogate, RunConfig
from scal.cli import main
from scal import cli
from scal.analysis.tables import TableChecker
from scal.cohorts import CohortRunner
from scal.datasets import find_root
from scal.io import read_json, write_json
from scal.verification import ReferenceComparison

ROOT = Path(__file__).resolve().parents[1]
benchmark_workspace = pytest.mark.skipif(
    not (ROOT / "data/benchmark_inputs.json").is_file(),
    reason="requires a separately supplied benchmark workspace",
)


def test_doctor_requires_bundled_inputs(monkeypatch):
    repository = SimpleNamespace(
        root=ROOT,
        entries={"bundled.npz": {"included": True}, "extra.npz": {"included": False}},
    )

    def missing_input(relative):
        raise FileNotFoundError(relative)

    repository.input_path = missing_input
    monkeypatch.setattr(cli, "_repository", lambda args: repository)
    report = cli._doctor(SimpleNamespace())
    assert report["passed"] is False
    assert report["missing_required_inputs"] == ["bundled.npz"]
    assert report["optional_training_files_missing"] == 1


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    if not (ROOT / "data/benchmark_inputs.json").is_file():
        pytest.skip("requires a separately supplied benchmark workspace")
    directory = tmp_path_factory.mktemp("workflow") / "ishigami"
    repository = DatasetRepository(ROOT)

    # Any attempt to read held-out inputs during training fails this test.
    def forbidden_test_read(*args, **kwargs):
        raise AssertionError("Training attempted to read independent test inputs")

    repository.test = forbidden_test_read
    result = Experiment(repository, RunConfig(stop=32)).run(directory)
    return result


def test_short_run_matches_reference(short_run):
    report = ReferenceComparison(ROOT / "data/expected/ishigami").compare(short_run.directory)
    assert report["fits"] == 3
    assert report["candidate_ids_exact"]
    assert short_run.summary["test_reads_during_training"] == 0


def test_saved_model_predicts_observations(short_run):
    model = PolynomialSurrogate.from_run(short_run.directory)
    with np.load(short_run.directory / "observations.npz", allow_pickle=False) as arrays:
        inputs = arrays["X"]
    one_chunk = model.predict(inputs)
    np.testing.assert_allclose(
        model.predict(inputs, chunk_size=3), one_chunk, rtol=1e-13, atol=1e-13
    )
    assert one_chunk.shape == (32,)
    assert np.isfinite(one_chunk).all()
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((3, 2)))


def test_existing_results_are_preserved(short_run):
    previous = (short_run.directory / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        Experiment.from_directory(ROOT, RunConfig(stop=32)).run(short_run.directory)
    assert (short_run.directory / "run.json").read_bytes() == previous


def test_comparison_rejects_missing_budget(short_run, tmp_path):
    actual = tmp_path / "incomplete"
    shutil.copytree(short_run.directory, actual)
    (actual / "fits/N0031.npz").unlink()
    with pytest.raises(ValueError, match="every budget"):
        ReferenceComparison(ROOT / "data/expected/ishigami").compare(actual)


def test_comparison_rejects_changed_candidate(short_run, tmp_path):
    actual = tmp_path / "changed"
    shutil.copytree(short_run.directory, actual)
    path = actual / "fits/N0030.json"
    record = read_json(path)
    record["next"]["candidate_id"] += 1
    write_json(path, record)
    with pytest.raises(AssertionError, match="candidate"):
        ReferenceComparison(ROOT / "data/expected/ishigami").compare(actual)


def test_comparison_reports_ic_roundoff_without_claiming_exact_equality(short_run, tmp_path):
    actual = tmp_path / "ic_roundoff"
    shutil.copytree(short_run.directory, actual)
    path = actual / "fits/N0030.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["IC"] *= 1 + 4e-12
    np.savez(path, **arrays)
    report = ReferenceComparison(ROOT / "data/expected/ishigami").compare(actual)
    assert report["passed"] and report["supports_exact"] and report["candidate_ids_exact"]
    assert not report["all_numeric_arrays_exact"]
    assert report["max_absolute_difference_by_array"]["IC"] > 0


@pytest.mark.parametrize("key,perturbation", [("coef", 1e-11), ("IC", 1e-8)])
def test_comparison_rejects_changes_beyond_the_field_tolerance(
    short_run, tmp_path, key, perturbation
):
    actual = tmp_path / "numeric_change"
    shutil.copytree(short_run.directory, actual)
    path = actual / "fits/N0030.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {field: saved[field] for field in saved.files}
    arrays[key] *= 1 + perturbation
    np.savez(path, **arrays)
    with pytest.raises(AssertionError, match=key):
        ReferenceComparison(ROOT / "data/expected/ishigami").compare(actual)


@benchmark_workspace
def test_run_settings_reject_unknown_keys_and_types(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"batch_size": 2}))
    with pytest.raises(ValueError, match="Unknown run options"):
        RunConfig.from_json(path)
    with pytest.raises(ValueError, match="stop"):
        RunConfig(stop=True)
    problem = DatasetRepository(ROOT).config.problem("ishigami")
    with pytest.raises(ValueError, match="seed list"):
        RunConfig(seed=999).validate(problem)
    with pytest.raises(ValueError, match="stop"):
        RunConfig(stop=5).validate(problem)


@benchmark_workspace
def test_external_data_overlay_falls_back_to_included_inputs(tmp_path):
    repository = DatasetRepository(ROOT, data_root=tmp_path)
    relative = "benchmarks/ishigami/seed_000/training.npz"
    assert repository.input_path(relative) == ROOT / "data" / relative


@benchmark_workspace
def test_cohort_checks_all_seeds_before_starting(tmp_path):
    output = tmp_path / "cohort"
    with pytest.raises(FileNotFoundError, match="seed_001"):
        CohortRunner(DatasetRepository(ROOT)).run(
            "ishigami_sampling__AHSC", output, seeds=[0, 1], stop=30
        )
    assert not output.exists()


@benchmark_workspace
def test_cli_reports_bad_configuration(capsys, tmp_path):
    assert main(["run", "--root", str(ROOT), "--stop", "1", "--output", str(tmp_path / "run")]) == 2
    assert "stop must" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@benchmark_workspace
def test_cli_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scal", "list", "--root", str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["tasks"]) == 28


@benchmark_workspace
def test_root_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCAL_HOME", str(ROOT))
    assert find_root() == ROOT


def test_table_comparison_uses_published_rows_and_detects_missing_rows(tmp_path):
    expected = pd.DataFrame({"method": ["AHSC"], "error": [0.25]})
    sources = SimpleNamespace(frame=lambda name: expected)
    checker = TableChecker(sources, tmp_path)
    actual = pd.DataFrame({"method": ["AHSC", "AHSC-Original"], "error": [0.25, 0.8]})
    checker.compare("summary.csv", actual, ["method"])
    assert checker.checks[0]["rows_outside_saved_table"] == 1
    assert pd.read_csv(tmp_path / "summary.csv").method.tolist() == ["AHSC"]
    with pytest.raises(ValueError, match="missing"):
        checker.compare("summary.csv", actual.iloc[1:], ["method"])
