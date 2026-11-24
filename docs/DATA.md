# Inputs and outputs

The [examples](QUICKSTART.md#run-the-ishigami-example) generate their own
inputs. For a new Python response or an external simulator, construct a
[FunctionExperiment](API.md#custom-problems) with the input distributions,
training budget and scalar response.

## Distributions and error measures

Each row of the distribution matrix describes one independent physical input.

| Row | Distribution |
|---|---|
| `[0, lower, upper]` | Uniform over the given bounds |
| `[1, mean, standard_deviation]` | Normal |
| `[2, mean_log, standard_deviation_log]` | Lognormal, with parameters in log space |

Relative mean squared error is the sum of squared prediction errors divided
by the sum of squared deviations of the reference responses from their mean.

## Benchmark workspaces

To run benchmark commands, supply a workspace that contains
`configs/paper_benchmarks.json` and `data/benchmark_inputs.json`. The
registry lists input paths relative to `data/`. Use `--root PATH` for
the workspace and `--data-root PATH` for an optional input directory.

| Input | Contents |
|---|---|
| `benchmarks/<problem>/seed_<ID>/training.npz` | `initial_X`, `initial_y`, `candidate_pool`, `candidate_ids` |
| `benchmarks/<problem>/test_X.npy`, `test_y.npy` | Independent validation inputs and responses |
| `data/expected/ishigami/` | Reference fits for `scal quickstart` |

Keep candidate IDs unique and fixed across training steps. Run
`python -m scal doctor --root PATH` to check a configured workspace.
`scal cohort` also uses `configs/paper_tasks.json` from that workspace.
An analysis data directory is identified by its `source_catalog.json`.

[Saved-run outputs](REPORTS.md)
