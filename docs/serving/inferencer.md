# Serving Guide (Inferencer)

`Inferencer` packages the fitted processors from a cross-validation `Trainer` into a single, self-contained object for deployment. It has no dependency on `Experimenter` or `Trainer` at serve time.

## Exporting from Trainer

After `train()` completes, call `to_inferencer()` to extract the fitted processors:

```python
t.train([Predictor.from_trial(trial, experimenter=e.name)])

inf = t.to_inferencer(
    v=None,   # optional DSL filter on the output columns, as in Trainer.process
)
```

`to_inferencer()` copies the processors out of the Trainer. The result depends on
nothing else — not the Trainer, not the Experimenter, not even a Pipeline, since
only each node's `edges` is needed to wire them up at serve time.

## Saving and Loading

The entire `Inferencer` — node specs, every split's processors, and the output filter — is serialized into a single file.

**Save:**

```python
inferencer.save('./model/inferencer')
# Writes: ./model/inferencer/__inferencer.pkl
```

**Load:**

```python
from mllabs import Inferencer

inferencer = Inferencer.load('./model/inferencer')
```

No training dependencies (Experimenter, Trainer, Collectors) are required to load or run an Inferencer.

## Running Inference

```python
predictions = inferencer.process(test_df, agg='mean')
```

`process()` applies each split's processors to the input data in sequence (Stage transforms → Head prediction), then aggregates across splits.

Input can be any pandas/polars/numpy object that the pipeline was trained on.

### Selecting Predictors

By default `process()` runs every Predictor the Trainer trained and concatenates their outputs column-wise. Pass `nodes` to restrict it to a subset:

```python
# Single head
predictions = inferencer.process(test_df, nodes='lgbm_v1')

# Subset of heads
predictions = inferencer.process(test_df, nodes=['lgbm_v1', 'lgbm_v2'])
```

`nodes` must be a name or list of names from `inferencer.selected_heads`. An unknown name raises `ValueError`.

## Aggregation Strategies

| `agg` | Behaviour | Use when |
|-------|-----------|----------|
| `'mean'` (default) | Element-wise mean across splits | Regression, probability outputs |
| `'mode'` | Element-wise mode across splits | Classification with hard labels |
| `callable` | `agg(results)` where `results` is a list of per-split DataFrames | Custom ensembling logic |
| `None` | Returns a list of per-split results | Debugging, manual aggregation |

When `n_splits == 1`, aggregation is skipped and the single split result is returned directly regardless of `agg`.

**Custom aggregation example:**

```python
import numpy as np

def weighted_mean(results):
    # weight later splits more heavily
    weights = np.linspace(1, 2, len(results))
    stacked = np.stack([r.values for r in results], axis=0)
    return (stacked * weights[:, None, None]).sum(axis=0) / weights.sum()

predictions = inferencer.process(test_df, agg=weighted_mean)
```

## Dependency Requirements

At serve time, only the following are needed:

- `ml-labs` (core package)
- The framework libraries used by the trained processors (e.g., `lightgbm`, `xgboost`)
- **Not** needed: `scikit-learn` splitters, Collectors, Trainer, or any training-only dependencies

This means a serving environment can install a minimal subset:

```bash
pip install ml-labs lightgbm
```
