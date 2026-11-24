"""Compare fitted coefficients and candidate IDs with saved experiment results."""

import sys
from scal.cli import main as cli_main
from scal.verification import ReferenceComparison


def verify(expected, actual, output):
    return ReferenceComparison(expected).compare(actual, output)


if __name__ == "__main__":
    raise SystemExit(cli_main(["compare", *sys.argv[1:]]))
