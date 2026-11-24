# Getting started

## Installation

Run `python scripts/setup.py` from the checkout with Python 3.13. It
creates `.venv` and installs the dependencies in `requirements-lock.txt`.
Use `--env PATH` to select another environment directory.

| Shell | Activate the environment |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Linux or macOS | `source .venv/bin/activate` |

For manual installation, activate a Python environment and run
`python -m pip install -r requirements-lock.txt` and `python -m pip install .`.

## Run the Ishigami example

```console
python examples/simple_function.py --budget 35
```

The script generates 30 initial observations, adds five observations selected
by AHSC, and evaluates the final surrogate on 4,096 independently generated
points. See `outputs/ishigami/validation.json` for the moments and test
error and `outputs/ishigami/summary.png` for the report.

Omit `--budget 35` to run the full example with 120 observations.
Use `--output PATH` for a separate run directory.

## Run your own function or solver

- [Python response](API.md#custom-problems)
- [Mixed inputs](../examples/cantilever.py)
- [File-based solver](API.md#external-simulators)

To check the code, run `python -m pytest -q -rs`. Tests for reference
runs skip when the separately supplied benchmark workspace is absent.
[Benchmark input formats](DATA.md#benchmark-workspaces)
