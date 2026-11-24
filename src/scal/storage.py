"""Save and inspect experiment results using JSON and NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json, write_json, write_npz


@dataclass(frozen=True)
class FitSnapshot:
    budget: int
    values: dict[str, np.ndarray]
    record: dict[str, Any]

    @classmethod
    def read(cls, path: Path, *, require_record: bool = True) -> FitSnapshot:
        with np.load(path, allow_pickle=False) as arrays:
            values = {key: arrays[key].copy() for key in arrays.files}
        metadata = path.with_suffix(".json")
        record = read_json(metadata) if require_record or metadata.is_file() else {}
        return cls(int(path.stem.removeprefix("N")), values, record)


class RunStore:
    """Files belonging to one run. Create it explicitly before writing fits."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()

    def create(self, settings: dict) -> None:
        if self.directory.exists() and any(p.name != ".run.lock" for p in self.directory.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {self.directory}. Choose a new directory."
            )
        (self.directory / "fits").mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "settings.json", settings)

    def save_fit(self, model, record: dict) -> None:
        path = self.directory / "fits" / f"N{record['budget']:04d}"
        write_npz(
            path.with_suffix(".npz"),
            coef=model.coef_,
            support=model.support_indices_,
            mean=float(model.mean_),
            std=float(model.std_),
            error=float(model.error_),
            raw_PRESS=float(model.press_error_),
            IC=float(getattr(model, "ic_", np.nan)),
        )
        # The JSON is the commit marker. Resumable runs include a checksum.
        if "basis_sha256" in record:
            from .io import digest

            record["fit_sha256"] = digest(path.with_suffix(".npz"))
        write_json(path.with_suffix(".json"), record)

    def finish(self, model, inputs, responses, selected_ids, summary: dict) -> None:
        write_json(
            self.directory / "basis.json",
            {
                "mapping": model.mapping_,
                "gamma": model.gamma_,
                "poly_matrix": model.poly_matrix,
            },
        )
        write_npz(
            self.directory / "observations.npz",
            X=inputs,
            y=responses,
            selected_ids=np.asarray(selected_ids, dtype=np.int64),
        )
        write_json(self.directory / "run.json", summary)

    def snapshots(self, *, require_records: bool = True) -> list[FitSnapshot]:
        result = [
            FitSnapshot.read(path, require_record=require_records)
            for path in sorted((self.directory / "fits").glob("N*.npz"))
        ]
        if not result:
            raise ValueError(f"No fitted results found in {self.directory}")
        return result
