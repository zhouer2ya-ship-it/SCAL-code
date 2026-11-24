"""Reports preserve saved experiments and state validation accuracy correctly."""

import csv
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

from scal import FunctionConfig, FunctionExperiment, save_report
from scal.cli import main


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    def response(inputs):
        return 1 + inputs[:, 0] + 0.2 * inputs[:, 0] ** 2

    directory = tmp_path_factory.mktemp("report") / "run"
    experiment = FunctionExperiment(
        response,
        [[0, -1, 1]],
        FunctionConfig(initial=8, budget=10, pool_size=32, degree=2),
        name="ishigami",
    )
    experiment.run(directory)
    return directory


def summary(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return {row["metric"]: row["value"] for row in csv.DictReader(stream)}


def test_report_uses_all_validation_points_and_preserves_run(finished_run, tmp_path):
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in finished_run.rglob("*")
        if p.is_file()
    }
    truth = np.linspace(-1, 1, 6000)
    prediction = truth + 0.25
    report = save_report(finished_run, tmp_path, validation=(truth, prediction))
    values = summary(report["summary_csv"])
    assert int(values["validation_samples"]) == len(truth)
    assert float(values["validation_rmse"]) == pytest.approx(0.25)
    assert float(values["validation_relmse"]) == pytest.approx(0.25**2 / np.var(truth))
    assert float(values["validation_max_absolute_error"]) == pytest.approx(0.25)
    assert Path(report["figure_png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = Path(report["figure_svg"]).read_text(encoding="utf-8")
    assert "Independent validation" in svg and "6,000" in svg and "shown" in svg
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in finished_run.rglob("*")
        if p.is_file()
    }


def test_custom_name_does_not_load_benchmark_data(finished_run, tmp_path, capsys):
    assert main(["report", "--run", str(finished_run), "--output", str(tmp_path)]) == 0
    assert summary(tmp_path / "summary.csv")["validation_samples"] == "0"
    svg = (tmp_path / "summary.svg").read_text(encoding="utf-8")
    assert "Surrogate mean" in svg and "Surrogate standard deviation" in svg
    assert "Independent validation" not in svg
    assert '"validation_samples": 0' in capsys.readouterr().out


def test_saved_constant_validation_has_no_relative_error(finished_run, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(finished_run, run)
    np.savez(run / "validation.npz", y=np.ones(3), prediction=np.zeros(3))
    report = save_report(run)
    values = summary(report["summary_csv"])
    assert values["validation_relmse"] == ""
    assert float(values["validation_rmse"]) == 1


@pytest.mark.parametrize("validation", [([1], [1]), ([1, 2], [1]), ([1, 2], [1, np.nan])])
def test_invalid_validation_preserves_reports(finished_run, tmp_path, validation):
    old = tmp_path / "summary.csv"
    old.write_text("existing report", encoding="utf-8")
    with pytest.raises(ValueError, match="Validation"):
        save_report(finished_run, tmp_path, validation=validation)
    assert old.read_text(encoding="utf-8") == "existing report"


def test_missing_fit_prevents_a_misleading_trajectory(finished_run, tmp_path):
    run = tmp_path / "incomplete"
    shutil.copytree(finished_run, run)
    (run / "fits/N0009.npz").unlink()
    with pytest.raises(ValueError, match="every budget"):
        save_report(run)
    assert not (run / "summary.csv").exists()
