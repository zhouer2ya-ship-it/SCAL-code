"""Recompute paper summaries and application statistics."""

import sys
from scal.analysis import PaperAnalysis
from scal.cli import main as cli_main


def main(root, output):
    return PaperAnalysis(root).run(output)


if __name__ == "__main__":
    raise SystemExit(cli_main(["analyze", "--scope", "paper", *sys.argv[1:]]))
