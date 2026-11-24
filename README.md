# SCAL

Structure-closed active learning (SCAL) fits sparse polynomial chaos surrogates
to scalar responses from Python functions or external simulators. Heredity-constrained backward pruning (HCBP)
selects basis terms. The adaptive hybrid sampling criterion (AHSC) selects
training points.

[Python API](docs/API.md) · [BSD 3-Clause License](LICENSE)

## Quick start

Install SCAL with Python 3.13.

```console
git clone https://github.com/zhouer2ya-ship-it/SCAL-code.git
cd SCAL-code
python scripts/setup.py
```

[Activate the environment](docs/QUICKSTART.md#installation) and run the Ishigami
example. The script generates training and independent validation inputs and
computes their responses with the analytic Ishigami function.

```console
python examples/simple_function.py --budget 35
```

The run writes fitted models, `validation.json` and `summary.png` to
`outputs/ishigami/`.

## Examples

- [Custom response](docs/API.md#custom-problems) takes one physical-input row per sample.
- [Cantilever displacement](examples/cantilever.py) mixes uniform, normal and lognormal inputs.
- [External simulator](examples/external_simulator/run.py) exchanges inputs and responses through files. [Simulator setup](docs/API.md#external-simulators) · [Resume a run](docs/API.md#resume-a-custom-or-external-run)

[Input formats and benchmarks](docs/DATA.md) · [Software citation](CITATION.cff)
