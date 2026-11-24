"""Fit and sample Ishigami using the solver and sampler directly.

Replace response() with your scalar response function. The input distribution
below is uniform on [-pi, pi] in each of three dimensions.
"""

import numpy as np
from threadpoolctl import threadpool_limits

from scal import HCBPSolver
from scal.acquisition import SamplingPolicy


def response(inputs: np.ndarray) -> np.ndarray:
    x1, x2, x3 = inputs.T
    return np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)


def main():
    rng = np.random.default_rng(42)
    poly = np.array([[0, -np.pi, np.pi]] * 3)
    inputs = rng.uniform(-np.pi, np.pi, (30, 3))
    candidates = rng.uniform(-np.pi, np.pi, (2048, 3))
    candidate_ids = np.arange(len(candidates))
    # Keep candidate IDs fixed; remove selected points by filtering array rows.
    responses = response(inputs)
    policy = SamplingPolicy("AHSC", poly)
    selected = []
    previous = None
    with threadpool_limits(limits=1):
        for budget in range(30, 36):
            # Use the previous support and error in the next HCBP comparison.
            model = HCBPSolver(poly, max_degree=12, q=1, last_model=previous).fit(inputs, responses)
            model.last_model = None
            print(f"N={budget}, mean={model.mean_:.6g}, sd={model.std_:.6g}")
            if budget < 35:
                available = ~np.isin(candidate_ids, selected)
                selection = policy.select(
                    model, inputs, responses, candidates[available], candidate_ids[available]
                )
                # Obtain a response before fitting the next model.
                inputs = np.vstack([inputs, selection.point])
                responses = np.r_[responses, response(selection.point)]
                selected.append(selection.candidate_id)
            previous = model
    print("Selected candidate IDs:", selected)


if __name__ == "__main__":
    main()
