"""Standalone demonstration solver. Uses only the Python standard library.

This process reads an input file and writes a scalar result. Replace its
command with a real solver or an adapter that runs the real solver and waits.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args()
    # Read one parameter file and evaluate one sample in this standalone process.
    if args.format == "json":
        parameters = json.loads(args.input.read_text(encoding="utf-8"))["parameters"]
    else:
        parameters = dict(
            line.split("=", 1)
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    load, length, young, height = [
        float(parameters[name]) for name in ("load_N", "length_m", "young_Pa", "height_m")
    ]
    if not all(math.isfinite(value) for value in (load, length, young, height)):
        raise ValueError("All parameters must be finite")
    if min(length, young, height) <= 0:
        raise ValueError("Length, modulus and height must be positive")
    width = 0.1
    inertia = width * height**3 / 12
    displacement_mm = 1000 * load * length**3 / (3 * young * inertia)
    # Write the response and unit for the external adapter to read.
    result = {"response": displacement_mm, "unit": "mm", "model": "cantilever_demo"}
    if args.format == "json":
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    else:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["quantity", "value", "unit"])
            writer.writerow(["tip_displacement", displacement_mm, "mm"])
    print(f"Completed: displacement = {displacement_mm:.12g} mm")


if __name__ == "__main__":
    main()
