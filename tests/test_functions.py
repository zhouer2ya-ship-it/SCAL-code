"""Custom functions use the stated budget and the same saved-model format."""

import numpy as np
import pytest

from scal import FunctionConfig, FunctionExperiment, HCBPSolver, InputSpace, PolynomialSurrogate
from scal.io import read_json


def test_custom_run_evaluates_only_initial_and_selected_points(tmp_path):
    calls, progress = [], []

    def response(inputs):
        calls.append(inputs.copy())
        return inputs[:, 0] ** 2 + 0.3 * inputs[:, 1]

    config = FunctionConfig(initial=10, budget=13, pool_size=128, degree=2, seed=17)
    experiment = FunctionExperiment(response, [[0, -1, 1], [0, -1, 1]], config)
    result = experiment.run(tmp_path / "run", progress=progress.append)
    assert [len(inputs) for inputs in calls] == [10, 1, 1, 1]
    assert [item.budget for item in progress] == [10, 11, 12, 13]
    assert progress[-1].next_candidate_id is None
    assert result.summary["training_response_evaluations"] == 13
    assert len(set(result.summary["selected_ids"])) == 3
    assert result.summary["candidate_responses_precomputed"] == 0
    with np.load(result.directory / "observations.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["X"], np.vstack(calls))
    model = PolynomialSurrogate.from_run(result.directory)
    points = np.array([[0.2, 0.4], [-0.4, 0.8]])
    np.testing.assert_allclose(
        model.predict(points), points[:, 0] ** 2 + 0.3 * points[:, 1], atol=1e-11
    )
    assert read_json(result.directory / "fits/N0013.json")["next"] is None


def test_custom_run_refuses_existing_output_before_response(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("existing results")

    def response(inputs):
        pytest.fail("Response should not run when the output already exists")

    with pytest.raises(FileExistsError):
        FunctionExperiment(response, [[0, -1, 1]]).run(output)


@pytest.mark.parametrize(
    "response", [lambda x: np.full(len(x), np.nan), lambda x: np.ones((len(x), 2))]
)
def test_invalid_responses_are_reported(response, tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        FunctionExperiment(response, [[0, -1, 1]]).run(output)
    assert read_json(output / "failure.json")["type"] == "ValueError"


def test_mixed_input_space_uses_physical_coordinates():
    space = InputSpace([[0, 2, 3], [1, 100, 0.2], [2, np.log(10), 0.05]])
    inputs = space.sample(100, np.random.default_rng(7))
    assert inputs.shape == (100, 3)
    assert np.all((inputs[:, 0] >= 2) & (inputs[:, 0] <= 3))
    assert np.all(inputs[:, 1] > 90)
    assert np.all(inputs[:, 2] > 0)


@pytest.mark.parametrize("poly", [[[0, 1, -1]], [[1, 0, 0]], [[2, 0, -1]], [[3, 0, 1]]])
def test_invalid_input_distributions_are_rejected(poly):
    with pytest.raises(ValueError):
        InputSpace(poly)


def test_pool_must_cover_the_requested_training_budget():
    with pytest.raises(ValueError, match="pool"):
        FunctionConfig(initial=10, budget=20, pool_size=5)


def test_mixed_inputs_refit_reuses_basis_with_positive_lognormal_probe():
    space = InputSpace([[0, -1, 1], [1, 0, 1], [2, np.log(10), 0.05]])
    inputs = space.sample(30, np.random.default_rng(12))
    responses = 3 + inputs[:, 0] + 0.2 * inputs[:, 1] + np.log(inputs[:, 2])
    previous = HCBPSolver(space.poly_matrix, max_degree=2).fit(inputs[:24], responses[:24])
    model = HCBPSolver(space.poly_matrix, max_degree=2, last_model=previous).fit(inputs, responses)
    assert model.basis_reused_
    assert model.mapping_ == previous.mapping_
    np.testing.assert_array_equal(model.gamma_, previous.gamma_)
    np.testing.assert_allclose(model.predict(inputs), responses, rtol=1e-10, atol=1e-10)
