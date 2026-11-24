# Figures and tables

The Ishigami, cantilever and external-simulator examples save figures after the final fit. To generate a report from existing results:

```console
python -m scal report --run outputs/ishigami
python -m scal report --run outputs/external --output outputs/external_figures
```

For a registered benchmark, supply its workspace to evaluate on its independent test set. For custom and external runs, it reads predictions and responses from `validation.npz` when available.

## Output files

| File | Contents |
|---|---|
| `summary.png` | Four-panel figure, 300 dpi |
| `summary.svg` | Scalable figure with editable text and vector scatter points |
| `summary.csv` | One row per final metric |
| `training_history.csv` | One row per fitted model, in training-budget order |

The upper panels show raw PRESS and the number of active polynomial terms. Raw PRESS is the stored leave-one-out cross-validation diagnostic. It uses a logarithmic axis when all finite values are positive. Non-finite diagnostics are omitted from the line and identified in the panel; the table retains their values.

With independent validation, the lower panels show predictions and residuals against reference responses. The diagonal marks exact predictions and the horizontal line marks zero residual. For runs without validation, those panels show the fitted mean and standard deviation at each budget.

Plots with more than 5,000 validation points display a deterministic subset, retaining response and residual extrema. The plot states the total and displayed counts. All validation points contribute to the summary metrics.

## Metrics

The final summary includes the problem, methods, training sample count, number of fitted models, active terms, surrogate mean and standard deviation, raw PRESS and validation sample count. When validation is available it also includes:

- `validation_rmse`: square root of the average squared prediction error, in response units.
- `validation_relmse`: dimensionless [relative mean squared error](DATA.md#distributions-and-error-measures). A blank value indicates constant reference responses.
- `validation_max_absolute_error`: largest absolute prediction error, in response units.

`training_history.csv` records `budget`, `active_terms`, `mean`, `std`, `error` and `raw_PRESS`. The `error` column preserves the solver's own score; for HCBP this is corrected PRESS.

## Python

```python
from scal import save_report

save_report("outputs/my_function")

# Both arrays must refer to the same independent validation inputs.
save_report("outputs/my_function", validation=(truth, prediction))
```

Both validation arrays must have shape `(n,)`, contain finite values and include at least two points. Supply values in the same response units as the saved model. The function returns paths to the four report files. Existing report files in the chosen output directory are replaced; saved fits and observations are preserved.
