"""Create a local Python environment and install the verified dependencies.

Run from any directory: python path/to/SCAL-code/scripts/setup.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=Path, default=root / ".venv", help="Environment to create or reuse"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Install editable source and the formatter"
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required. Python 3.13 is the verified version.")
    environment = args.env.expanduser().resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        # Keep dependencies local; venv excludes system site-packages by default.
        venv.EnvBuilder(with_pip=True).create(environment)
    # Install the pinned scientific stack before this source tree. --no-deps
    # keeps the second step from selecting different dependency versions.
    commands = [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(root / "requirements-lock.txt"),
        ],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            *(["-e"] if args.dev else []),
            str(root),
        ],
    ]
    if args.dev:
        commands.append([str(python), "-m", "pip", "install", "ruff==0.14.3"])
    for command in commands:
        subprocess.run(command, check=True)
    subprocess.run([str(python), "-m", "scal", "--version"], check=True)
    print("\nInstallation complete. Run the example:")
    prefix = "& " if os.name == "nt" else ""
    print(f'{prefix}"{python}" "{root / "examples/simple_function.py"}" --budget 35')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"Installation step failed (exit {error.returncode}). See the pip output above.",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode)
