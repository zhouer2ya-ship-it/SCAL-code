"""Use the paper's sampling methods through a common one-point interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SAMPLERS
from .coordinates import TargetGeometry
from .sampling.ahsc import AHSCSampler
from .sampling.base import CandidateFeasibility, compute_gradient_importance
from .sampling.s_optimal import SOptimalSampler
from .sampling.theta import ThetaSampler


@dataclass(frozen=True)
class Selection:
    point: np.ndarray
    record: dict[str, Any]

    @property
    def candidate_id(self) -> int:
        return int(self.record["candidate_id"])


class SamplingPolicy:
    """Select one candidate using a fitted scalar PCE and its observations."""

    def __init__(self, name: str, poly_matrix: np.ndarray):
        if name not in SAMPLERS:
            raise ValueError(f"Unknown sampler: {name}")
        self.name = name
        self.poly_matrix = poly_matrix

    def select(
        self,
        model,
        inputs: np.ndarray,
        responses: np.ndarray,
        candidates: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> Selection:
        classes = {"AHSC": AHSCSampler, "Theta": ThetaSampler, "SOptimal": SOptimalSampler}
        if self.name in classes:
            sampler = classes[self.name](self.poly_matrix, verbose=False)
            sampler.initialize(inputs, responses, candidates, candidate_ids)
            sampler.set_model(model)
            point = sampler.select_samples(1)
            if not len(point):
                raise RuntimeError("No eligible candidate remains")
            return Selection(point, sampler.last_selection_report[0])
        return self._historical_comparator(model, inputs, candidates, candidate_ids)

    def _historical_comparator(self, model, inputs, candidates, candidate_ids) -> Selection:
        """MAXDistance/MAXGradient retain their published z-space radius of 0.1."""
        geometry = TargetGeometry.from_poly(self.poly_matrix)
        state = CandidateFeasibility(
            geometry.transform(inputs), geometry.transform(candidates), ratio=0.1
        )
        active = np.flatnonzero(state.mask)
        if not len(active):
            raise RuntimeError("No feasible comparator candidate remains")
        scores = (
            state.distances[active]
            if self.name == "MAXDistance"
            else compute_gradient_importance(candidates[active], model, geometry=geometry).ravel()
        )
        winner = int(active[np.lexsort((candidate_ids[active], -scores))[0]])
        return Selection(
            candidates[[winner]].copy(),
            {
                "candidate_id": int(candidate_ids[winner]),
                "criterion": self.name,
                "distance_threshold": float(state.radius),
            },
        )
