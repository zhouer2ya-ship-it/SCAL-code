"""Run selected seeds of a named paper task."""

import sys
from scal.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["cohort", *sys.argv[1:]]))
