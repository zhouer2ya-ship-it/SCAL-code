from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scal import HCBPSolver, AHSCSampler
from scal.sampling.ahsc import minmax_component, polynomial_germ_gradient
from scal.sampling.theta import theta_scores, germ_logpdf, select_candidate
from scal.applications import design
from scal.solving.utils import construct_matrix_optimized, normalization_all


def test_constant_and_unused_scores():
    mask = np.array([True, True, False])
    np.testing.assert_array_equal(minmax_component([0, 0, np.nan], mask)[0], [0, 0, 0])
    np.testing.assert_array_equal(minmax_component([7, 7, np.inf], mask)[0], [1, 1, 0])
    np.testing.assert_array_equal(minmax_component([3, 9, -99], mask)[0], [0, 1, 0])


def test_theta_against_direct_euclidean_formula():
    train = np.array([[-0.4, 0.2], [0.5, -0.7], [0.8, 0.9]])
    candidates = np.array([[-0.2, 0.3], [0.2, -0.8], [0.4, 0.4]])
    a = np.column_stack([np.ones(3), train])
    c = np.column_stack([np.ones(3), candidates])
    coef = np.array([7.0, 2.0, -3.0])
    pa = germ_logpdf(train, ["hermite", "hermite"])
    pc = germ_logpdf(candidates, ["hermite", "hermite"])
    result = theta_scores(train, candidates, a, c, coef, existing_logpdf=pa, candidate_logpdf=pc)
    distances = cdist(candidates, train)
    nearest = distances.argmin(axis=1)
    centered_a = a[:, 1:] @ coef[1:]
    centered_c = c[:, 1:] @ coef[1:]
    direct = (
        abs(centered_c * centered_a[nearest])
        * np.exp((pc + pa[nearest]) / 2)
        * distances.min(axis=1) ** 2
    )
    np.testing.assert_allclose(result.theta, direct, rtol=2e-14, atol=1e-15)
    np.testing.assert_array_equal(result.nearest, nearest)


def test_exact_tie_uses_frozen_id():
    train = np.array([[0.0]])
    candidates = np.array([[-1.0], [1.0]])
    result = theta_scores(
        train,
        candidates,
        np.ones((1, 1)),
        np.ones((2, 1)),
        [5.0],
        existing_logpdf=[0.0],
        candidate_logpdf=[0.0, 0.0],
    )
    assert select_candidate(result, candidate_ids=np.array([20, 3])) == 1


@pytest.mark.parametrize(
    "poly",
    [
        np.array([[0, -2, 4], [0, -1, 2]]),
        np.array([[1, 3, 2], [1, -2, 0.5]]),
        np.array([[2, 0.2, 0.5], [2, -0.3, 0.4]]),
        np.array([[0, -2, 4], [2, 0.2, 0.5]]),
    ],
)
def test_germ_gradient_finite_difference(poly):
    rng = np.random.default_rng(2901)
    xi = rng.uniform(-0.7, 0.7, (12, 2))

    def physical(g):
        x = np.empty_like(g)
        for j, (kind, a, b) in enumerate(poly):
            x[:, j] = (g[:, j] + 1) * (b - a) / 2 + a if kind == 0 else a + b * g[:, j]
            if kind == 2:
                x[:, j] = np.exp(x[:, j])
        return x

    x = physical(xi)
    model = HCBPSolver(poly, max_degree=3, q=1).fit(x, x[:, 0] + 0.1 * x[:, 1] ** 2)
    # Exercise a full nonconstant polynomial independent of the chosen sparse fit.
    model.coef_ = rng.normal(size=len(model.mapping_))
    model.support_indices_ = np.arange(len(model.coef_))
    numerical = np.empty_like(x)
    for j in range(2):
        plus = xi.copy()
        minus = xi.copy()
        plus[:, j] += 1e-6
        minus[:, j] -= 1e-6
        numerical[:, j] = (model.predict(physical(plus)) - model.predict(physical(minus))) / (2e-6)
    np.testing.assert_allclose(polynomial_germ_gradient(x, model), numerical, rtol=2e-8, atol=2e-8)


def test_one_response_then_refit_required():
    poly = np.array([[0, -1, 1.0]])
    x = np.linspace(-0.9, 0.9, 12)[:, None]
    y = x[:, 0] ** 2 + 0.3 * x[:, 0]
    model = HCBPSolver(poly, max_degree=3).fit(x, y)
    s = AHSCSampler(poly, verbose=False)
    s.initialize(x, y, np.array([[-0.95], [0.15], [0.95]]), [9, 8, 7])
    s.set_model(model)
    with pytest.raises(ValueError):
        s.select_samples(2)
    assert s.select_samples(1).shape == (1, 1)
    with pytest.raises(RuntimeError):
        s.select_samples(1)
