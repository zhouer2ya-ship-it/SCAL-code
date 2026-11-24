"""Regression checks for independent data roots and exact field voting."""

import json
from pathlib import Path

import numpy as np
import pytest

from scal import analysis
from scal.analysis import current, field
from scal.cli import main
from scal.data import Sources
from scal.sampling.ahsc import modal_vote


def test_independent_catalog_root(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("budget,error\n30,0\n", encoding="utf-8")
    catalog = {
        "numeric": [{"name": "metrics", "path": "data/metrics.csv", "bytes": path.stat().st_size}]
    }
    (tmp_path / "source_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    sources = Sources(data_root=tmp_path)
    assert sources.frame("metrics").error.tolist() == [0]
    path.write_text("budget,error\n30,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size differs"):
        sources.path("metrics")


@pytest.mark.parametrize(
    ("former", "current_name"),
    [
        ("Figure21_raw.csv", "fixed_design_fits.csv"),
        ("seed666_mc_effects.csv", "ahsc_theta_mc100k_effects.csv"),
    ],
)
@pytest.mark.parametrize("use_former_name", [False, True])
def test_catalog_names_resolve_in_both_directions(tmp_path, former, current_name, use_former_name):
    path = tmp_path / "record.csv"
    path.write_text("budget,error\n30,0\n", encoding="utf-8")
    name = former if use_former_name else current_name
    _write_catalog(tmp_path, name, path.name)

    sources = Sources(data_root=tmp_path)
    assert sources.path(former) == sources.path(current_name) == path
    assert sources.frame(current_name).error.tolist() == [0]


def _write_catalog(data_root, name, path="records.csv"):
    catalog = {"numeric": [{"name": name, "path": path}]}
    (data_root / "source_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")


@pytest.mark.parametrize(
    ("options", "target_module", "target_class", "scope"),
    [
        ([], current, "CurrentAnalysis", "comparisons"),
        (["--scope", "field"], field, "FieldReplay", "field"),
        (["--scope", "v3"], current, "CurrentAnalysis", "comparisons"),
        (["--scope", "v3-field"], field, "FieldReplay", "field"),
    ],
)
def test_current_analyze_selects_requested_analysis(
    tmp_path, monkeypatch, options, target_module, target_class, scope
):
    _write_catalog(tmp_path, "closed_loop_metrics.csv")

    class Replay:
        def __init__(self, data_root):
            assert data_root == tmp_path

        def run(self, output):
            assert output == Path("outputs") / scope
            return {"passed": True, "scope": scope}

    monkeypatch.setattr(target_module, target_class, Replay)
    assert main(["analyze", "--data-root", str(tmp_path), *options]) == 0


@pytest.mark.parametrize("explicit_data_root", [False, True])
def test_historical_catalog_still_defaults_to_paper(tmp_path, monkeypatch, explicit_data_root):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/paper_benchmarks.json").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "benchmark_inputs.json").touch()
    _write_catalog(data, "validation_V1_curves.csv")

    class PaperReplay:
        def __init__(self, root, data_root):
            assert data_root == data

        def run(self, output):
            assert output == Path("outputs/paper")
            return {"passed": True, "checks": []}

    monkeypatch.setattr(analysis, "PaperAnalysis", PaperReplay)
    options = ["--data-root", str(data)] if explicit_data_root else ["--root", str(tmp_path)]
    assert main(["analyze", *options]) == 0


@pytest.mark.parametrize(
    ("scope", "source"),
    [
        ("paper", "validation_V1_curves.csv"),
        ("theta_exchange", "source_exchange_primary_curves.csv"),
        ("mechanism", "mechanism_curves.csv"),
    ],
)
def test_current_dataset_explains_historical_requirements(tmp_path, capsys, scope, source):
    _write_catalog(tmp_path, "closed_loop_metrics.csv")

    assert main(["analyze", "--scope", scope, "--data-root", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert "historical SCAL dataset" in error
    assert "--data-root" in error
    assert source in error


def test_analyze_without_catalog_explains_data_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCAL_HOME", raising=False)

    assert main(["analyze"]) == 2
    assert (
        "Set --data-root to the directory containing source_catalog.json" in capsys.readouterr().err
    )


def test_modal_vote_abstention_and_ties():
    feasible = np.array([True, True, False])
    ids = np.array([41, 7, 4])
    # Active modes nominate one each. The higher mean score wins the tie.
    scores = [[1.9, 0.2, 0], [1.0, 1.1, 0], [0, 0, 0]]
    assert modal_vote(scores, feasible, ids) == 0
    assert modal_vote([[0, 0, 0]], feasible, ids) == 1
    assert modal_vote([[1, 1, 0]], feasible, ids) == 1
    with pytest.raises(FloatingPointError, match="Invalid scalar"):
        modal_vote([[0, np.nan, 0]], feasible, ids)
