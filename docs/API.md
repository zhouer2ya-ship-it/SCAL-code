# Python API

## Use your own response

Run `python examples/simple_function.py` for training and validation. The lower-level example in `examples/custom_response.py` uses three operations.

- `HCBPSolver(...).fit(inputs, responses)` to fit the PCE.
- `SamplingPolicy("AHSC", poly_matrix).select(...)` to choose one point.
- `Selection.point` and `Selection.candidate_id` to evaluate and record it.

Append the new response and refit before selecting again. Keep candidate IDs fixed when excluding previously selected rows. Pass the previous model through `last_model` to retain warm-start behavior.

| Array | Shape |
|---|---|
| `inputs` | (observations, dimension) |
| `responses` | (observations,) |
| `candidates` | (candidates, dimension) |
| `candidate_ids` | (candidates,), unique integers |
| `poly_matrix` | (dimension, 3) |

See the [input distribution codes](DATA.md#distributions-and-error-measures) for `poly_matrix`.

## Custom problems

`FunctionExperiment` generates initial inputs and a fixed candidate pool, then saves each fit and the completed observations. It uses HCBP and defaults to AHSC.

```python
import numpy as np
from scal import FunctionConfig, FunctionExperiment, PolynomialSurrogate

def response(X):
    x1, x2, x3 = X.T
    return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)

poly = [[0, -np.pi, np.pi]] * 3
settings = FunctionConfig(initial=30, budget=120, pool_size=16384, degree=12, seed=42)
experiment = FunctionExperiment(response, poly, settings, name="ishigami")
result = experiment.run("outputs/ishigami")
model = PolynomialSurrogate.from_run(result.directory)
```

The callback receives physical inputs of shape `(n, dimension)` and returns an array of shape `(n,)` or `(n, 1)`. Each row is one sample. SCAL sends the initial samples as a batch, then sends one sample per call until it reaches `budget` training points. For independent validation after training, draw inputs with `experiment.space.sample(count, rng)` and evaluate them with `experiment.evaluate(X)`.

`FunctionConfig` exposes `initial`, `budget`, `pool_size`, `degree`, `q`, `seed` and `sampler`. Distributions are independent and the response is scalar. Basis terms can involve up to three variables in problems with fewer than 20 dimensions and up to two variables in higher dimensions.

For a mixed-input engineering example, run `python examples/cantilever.py`. Its default 4,096 validation evaluations are additional to its 60 training evaluations. Use `--test-samples 0` to skip validation.

## External simulators

`ExternalSimulator` is a callable response for `FunctionExperiment`. It evaluates each row in a separate process and working directory. The command is an argument list with `{input}`, `{output}` and `{workdir}` placeholders. Processes run without a shell and must wait for the solver result before exiting.

The constructor takes `command`, `parameter_names`, `work_root`, and optional `timeout_seconds`, `input_name`, `output_name`, `input_writer`, `output_reader`, `model_id`, `model_files` and `retry_unconfirmed`. The default JSON reader expects one scalar `response`. Custom callbacks can write native files and extract a scalar from the solver's output. Register templates, meshes and material files through `model_files` so changes invalidate the cache.

Each job saves its input, output, logs and status. If the solver fails, times out or returns an invalid response, `SimulationError.directory` points to the files for that call. For a background or cluster job, the adapter must wait for the result and handle cancellation of jobs it launched.

Run `python examples/external_simulator/run.py` for the complete file-exchange example. Use `--adapter examples/external_simulator/adapter_text.json` for native text input and CSV output. Add `--test-samples N` to run independent validation. The example defaults to zero extra evaluations.

To connect another solver, set `command` to its executable and arguments, then adapt `write_native_input` and `read_native_response` in [`examples/external_simulator/run.py`](../examples/external_simulator/run.py). The parameter order must match the rows of `poly_matrix`. The reader must return a finite scalar in the units used in your report.

## Resume a custom or external run

```python
result = experiment.run("outputs/ishigami", resume=True)
```

Use the same configuration and output directory to resume. `FunctionExperiment` checks the inputs, response identity and numerical environment, then restores the previous fit, saved responses and any candidate awaiting a response. It returns saved results directly when training is complete. Use `response_id="model-v1"` to identify external data or mutable parameters used by a Python callback. Keep the callback's globals and closure consistent with that identifier. Bound methods of dataclass models also record the instance parameters. Use `metadata={...}` to store settings such as the independent validation count.

The Ishigami, cantilever and external examples also accept `--resume`. Keep the complete output directory. A Python callback interrupted before it returns an initial batch may receive that batch again. For external jobs, SCAL reuses successful results and can recover a job that was still running when SCAL stopped.

If an external job's completion is unconfirmed, inspect its job directory and verify whether it is still running. After confirming the prior job has stopped, use `--resume --retry-unconfirmed` to retry that evaluation if needed.

## Save figures and tables

Call `save_report(result.directory)` to write the figures and tables for a completed run. See [report options and examples](REPORTS.md#python).


## Benchmark workflows

[Benchmark workspaces](DATA.md#benchmark-workspaces) register saved inputs
and validation arrays. Pass the workspace path to `DatasetRepository` and a
`RunConfig` to `Experiment` to run a registered benchmark. Use
`BenchmarkEvaluator` after training to evaluate on the registered test set.

For a fitted surrogate, `PolynomialSurrogate.from_run(path, budget=32)`
can load a particular budget. `predict` takes a two-dimensional array
of physical input rows.
