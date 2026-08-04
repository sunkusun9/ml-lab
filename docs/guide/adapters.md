# Adapters

Adapters bridge the gap between ml-labs' unified interface and each ML framework's specific `fit()` parameter conventions (e.g., `eval_set`, `validation_data`, callbacks).

## Adapter Interface

All adapters extend `ModelAdapter` and may implement three members:

### get_params(params, logger)

Adjusts or filters constructor parameters before the model is instantiated. The default implementation returns `params` unchanged.

LightGBM example — strips `early_stopping` and `eval_metric` from constructor params (they belong in `fit()`):

```python
def get_params(self, params, logger=None):
    return {k: v for k, v in params.items() if k not in ['early_stopping', 'eval_metric']}
```

### get_fit_params(data_dict, params, logger)

Builds the keyword arguments passed to `model.fit()`. Receives a `data_dict` with `(train, train_v)` tuples per key.

Behaviour is controlled by two constructor parameters shared by all adapters:

| Parameter | Values | Effect |
|-----------|--------|--------|
| `eval_mode` | `'none'`/`None` | No eval set |
| | `'valid'` | Pass inner-validation fold only |
| | `'both'` | Pass both train and inner-validation fold |
| `verbose` | `0` | Silent |
| | `0 < v < 1` | Progress every `v*100`% of estimators |
| | `v >= 1` | Every `v` iterations (framework-native) |

### result_objs

A class-level dict of extractable model attributes used by `ModelAttrCollector`:

```python
result_objs = {
    'feature_importances': (get_importance_fn, True),   # (callable, mergeable)
    'trees':               (get_trees_fn,       False),
}
```

`mergeable=True` means the result is a `pd.Series` or `pd.DataFrame` that can be aggregated across folds by `get_attrs_agg()`.

---

## Built-in Adapters

Adapters are registered by model class name and resolved automatically via `get_adapter()`. The default `eval_mode='both'`, `verbose=0.1`.

### DefaultAdapter

Used for any model not in the registry. Passes no extra fit parameters — suitable for standard sklearn transformers and estimators.

### sklearn Adapters

| Adapter | Models | result_objs |
|---------|--------|-------------|
| `LMAdapter` | `LinearRegression`, `LogisticRegression` | `coef` |
| `PCAAdapter` | `PCA` | `explained_variance`, `explained_variance_ratio`, `components` |
| `LDAAdapter` | `LinearDiscriminantAnalysis` | `coef`, `intercept`, `scalings`, `explained_variance_ratio` |
| `DecisionTreeAdapter` | `DecisionTreeClassifier/Regressor` | `feature_importances`✓, `tree`✗, `plot_tree`✗ |

✓ mergeable, ✗ not mergeable

### LightGBMAdapter

Passes `eval_set` to `fit()`. Supports `early_stopping` and `eval_metric` as node `params` (they are moved from constructor to `fit()`).

`early_stopping` is given as a plain dict of `lgb.early_stopping` constructor kwargs, which the adapter turns into the callback. Since params reject live objects, this is the only way to specify it:

```python
Trial('lgbm', 'lightgbm.LGBMClassifier',
      edges={'X': 'scale:(*)', 'y': '{target}'}, method='predict',
      params={
          'n_estimators': 1000,
          'early_stopping': {'stopping_rounds': 50, 'verbose': False},
          'eval_metric': 'auc',
      })
```

Keeping it plain data is also what makes definitions comparable by value: a callback instance would compare by identity, so an unchanged definition could look changed.

`result_objs`: `feature_importances`✓, `evals_result`✓, `trees`✗

### XGBoostAdapter

Passes `eval_set` to `fit()`. Progress callback is added via `get_params()`.

`result_objs`: `feature_importances`✓, `evals_result`✓, `trees`✗

### CatBoostAdapter

Passes `eval_set` to `fit()`. `verbose=0<v<1` falls back to silent (CatBoost callback API is complex).

`result_objs`: `feature_importances_pvc`✓, `feature_importances_interaction`✓, `evals_result`✓, `trees`✗

### KerasAdapter

Uses `validation_data=(X_val, y_val)` instead of `eval_set`. Both `'valid'` and `'both'` eval modes behave identically (Keras only accepts a single validation set).

`result_objs`: none

### NNAdapter

For `NNClassifier` and `NNRegressor`. Passes the inner-validation fold as `eval_set=[(X_val, y_val)]` to `fit()`. Progress is reported via a Keras callback (`_ProgressCallback`) injected into the `callbacks` argument of `fit()`.

`result_objs`: `evals_result`✓

---

## GPU Settings

The `gpu` key in node `params` controls GPU routing and injection. It is stripped before passing to the model constructor.

| `'gpu'` value | Meaning |
|---------------|---------|
| `'auto'` (default) | Framework-specific detection — usually CPU unless a native GPU param is set |
| `'yes'` | Enable GPU: node is routed to a GPU worker and the adapter injects GPU device params |
| `None` | Disable GPU explicitly |

**Adapters inject GPU device params automatically when `'gpu': 'yes'` and a `gpu_id` is assigned by the worker:**

| Adapter | Injected params |
|---------|----------------|
| `XGBoostAdapter` | `device='cuda:{gpu_id}'` |
| `CatBoostAdapter` | `task_type='GPU'`, `devices='{gpu_id}'` |
| `NNAdapter` | `device='/GPU:{gpu_id}'` |
| `LightGBMAdapter` | _(no injection — set `device='gpu'` or `device='cuda'` directly in params)_ |

### LightGBM

LightGBM uses its own `device` param. To enable GPU:

```python
Trial('lgbm_gpu', 'lightgbm.LGBMClassifier', edges, method='predict',
      params={'n_estimators': 5000,
              'device': 'cuda',   # or 'gpu'
              'gpu': 'yes'})      # routes to a GPU worker
```

### XGBoost

```python
Trial('xgb_gpu', 'xgboost.XGBClassifier', edges, method='predict',
      params={'n_estimators': 5000,
              'gpu': 'yes'})   # injects device='cuda:{gpu_id}' automatically
```

### CatBoost

```python
Trial('cb_gpu', 'catboost.CatBoostClassifier', edges, method='predict',
      params={'n_estimators': 5000,
              'gpu': 'yes'})   # injects task_type='GPU', devices='{gpu_id}'
```

### NNClassifier / NNRegressor

```python
Trial('nn_gpu', 'mllabs.nn.NNClassifier', edges, method='predict',
      params={'epochs': 200,
              'gpu': 'yes'})   # injects device='/GPU:{gpu_id}' automatically
```

---

## Custom Adapter

```python
from mllabs.adapter import ModelAdapter, register_adapter

class MyAdapter(ModelAdapter):
    def get_fit_params(self, data_dict, params=None, logger=None):
        train_X, train_v_X = data_dict['X']
        train_y, train_v_y = data_dict['y']
        return {
            'eval_set': [(train_v_X.values, train_v_y.values)],
        }

register_adapter('MyModel', MyAdapter(eval_mode='valid', verbose=0))
```

After registration, `MyAdapter` is resolved automatically whenever a node uses `MyModel` as its processor.

To use an adapter on a specific node or group without registering it globally:

```python
p.set_node('my_node', grp='my_grp', adapter='mypackage.MyAdapter')

# with constructor arguments
p.set_node('my_node', grp='my_grp',
           adapter={'__ref__': 'mypackage.MyAdapter',
                    '__params__': {'eval_mode': 'valid'}})
```

An adapter is named by string or ref spec, never passed as an instance — like
`processor` and `params`, it is resolved only when the processor is built.
