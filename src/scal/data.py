"""Resolve versioned numerical source files inside a research directory."""

from pathlib import Path

import pandas as pd

from .datasets import find_root
from .io import data_path, read_json

# The earlier catalog used numbered command and protocol names.
LEGACY_DATASET_VERSION = 3

_SOURCE_ALIASES = {
    "Figure21_raw.csv": "fixed_design_fits.csv",
    "seed666_mc_effects.csv": "ahsc_theta_mc100k_effects.csv",
}
_SOURCE_ALIASES.update({name: former_name for former_name, name in _SOURCE_ALIASES.items()})


class Sources:
    """Look up a numerical source by its published filename."""

    def __init__(self, root: str | Path | None = None, data_root: str | Path | None = None):
        self.data_root = (
            find_root(root) / "data"
            if data_root is None
            else Path(data_root).expanduser().resolve()
        )
        self.catalog = read_json(self.data_root / "source_catalog.json")

    def file(self, name: str) -> Path:
        """Resolve a catalog path inside the data directory, including legacy paths."""
        return data_path(self.data_root, name.removeprefix("data/"))

    def path(self, name: str) -> Path:
        entries = self.catalog["numeric"]
        hits = [entry for entry in entries if entry["name"] == name]
        if not hits and name in _SOURCE_ALIASES:
            alias = _SOURCE_ALIASES[name]
            hits = [entry for entry in entries if entry["name"] == alias]
        if len(hits) != 1:
            raise ValueError(f"Expected one source named {name}, found {len(hits)}")
        entry = hits[0]
        path = self.file(entry["path"])
        if "bytes" in entry and path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source size differs: {name}")
        return path

    def reanalysis_protocol(self) -> dict:
        """Load the statistical protocol by its catalog name."""
        name = "reanalysis_protocol.json"
        names = {entry["name"] for entry in self.catalog["numeric"]}
        previous_name = f"v{LEGACY_DATASET_VERSION}_{name}"
        if name not in names and previous_name in names:
            name = previous_name
        return read_json(self.path(name))

    def frame(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name), float_precision="round_trip", low_memory=False)
