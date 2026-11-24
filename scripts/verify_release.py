"""Check the SCAL checkout against a Git tag or commit."""

import argparse
import json
from pathlib import Path

from scal.verification import verify_release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--scope", choices=["local", "github"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = verify_release(args.root, args.ref)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
