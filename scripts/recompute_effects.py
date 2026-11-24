"""Recompute paired effects with the original resampling and comparison families."""

import sys
from scal.analysis import PairedEffectAnalysis
from scal.cli import main as cli_main


def main(root, output, scope):
    return PairedEffectAnalysis(root).run(output, scope)


if __name__ == "__main__":
    raise SystemExit(cli_main(["analyze", "--scope", "theta_exchange", *sys.argv[1:]]))
