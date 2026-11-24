"""Recompute the paper's tabulated numerical summaries and paired effects."""

from .effects import PairedEffectAnalysis
from .paper import PaperAnalysis

__all__ = ["PaperAnalysis", "PairedEffectAnalysis"]
