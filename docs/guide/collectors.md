# Collectors

A Collector captures what happens while Trials run — metrics, model attributes, SHAP values, out-of-fold predictions, predictions on external data. Each one uses a `Connector` to decide which Trials it observes.

## The registry

Collectors are registered in a `Collectors` registry, which belongs to an Experimenter — `e.collectors`, stored under `{exp path}/collectors`. Registration builds the instance from its parts and **persists it immediately** — there is no `save()`.

```python
collectors = e.collectors

collectors.set_collector(
    'acc',
    'mllabs.collector.MetricCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {'edges': {'y': '{target}'}}},
    params={'output_var': '-1:',
            'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}},
)
```

The three positional arguments are the name, the Collector class (as a string or a class), and the Connector (as an instance or a ref spec). Everything else goes in `params`.

Constructing the registry *is* restoring it, and the Experimenter constructs one — reopening a run with `load_experimenter` hands back everything registered on it before.

```python
collectors.names()
collectors.get_collector('acc')
collectors.remove_collector('acc')      # deletes the row and its params file
'acc' in collectors
```

A registry belongs to one run on purpose. Everything a Collector writes is keyed by node name and nothing more — `MetricCollector`'s primary key is `(node, idx, inner_idx, split)`, and the file-based ones use `{path}/{node}...` — so the path is the only thing keeping two runs apart. Sharing one registry would have them overwrite each other on every Trial name they had in common, silently, and precisely when the results were worth comparing. Comparing across runs is a read over each run's own store.

!!! note "Registration writes through"
    `e.collectors` is bound to the run's directory, so `set_collector(...)` persists as it is called. A registry created without a path (`Collectors()`) is memory-only and keeps nothing.

### How they persist

A definition is stored as parts to reassemble, never as a pickled instance: a plain-text row (`name`, `collector`, `connector`, `path`) plus the constructor `params` pickled to `{collectors}/__params/{name}.pkl`. Loading calls the same builder `set_collector` does, so registering and restoring take one path.

`params` is the pickled piece because a Collector's arguments can hold things no definition can express — `ProcessCollector(ext_data=df)` takes a DataFrame. The other four stay columns precisely so a registry can be listed and inspected without unpickling anything.

!!! warning "Collector classes must be importable at module level"
    A class defined inside a function cannot be resolved from its reference, and multi-worker runs pickle collectors out to workers.

## Connector — what a Collector attaches to

All three criteria are optional; only the ones you set are checked, and they combine with AND.

```python
from mllabs import Connector

Connector()                                        # everything
Connector(node_query='lgb')                        # regex on the name
Connector(node_query=['lgb1', 'lgb2'])             # exact list
Connector(processor='lightgbm.LGBMClassifier')     # processor, as a string
Connector(edges={'y': '{target}'})                 # exact per-key edge match
```

`processor` is compared as a **plain string**, with no normalisation — pass the same form you gave `set_node`/`Trial`. `edges` matches each named key exactly, not by containment.

## Running with Collectors

```python
e.exp(names)                                 # every Collector on this run
e.exp(names, collectors=['acc', 'shap'])     # a subset, by name
e.exp(names, collectors=[])                  # collect nothing
```

`collectors=` takes **names** out of the run's own registry, the same way `processor` and `adapter` are string refs. An instance is rejected: a Collector this registry does not know has no place in this run to write to, and would quietly deposit its results outside it. An unregistered name raises `KeyError` — silently skipping would be indistinguishable from "collected nothing".

Outcomes always go to that run's `collectors.hist`, whichever selection a call makes. The history belongs to the run, not to one call's choice.

!!! note "Collectors only see Trials that actually run"
    Collection is a side effect of running a Trial. A fold already recorded `'built'` is skipped, so attaching a Collector *after* an experiment and re-running `exp()` collects nothing. Clear the fold history first — see [State Model](../concepts/state-model.md).

## CollectHist — what each Collector did

`collectors.hist` holds one row per `(collector, node, outer_idx, inner_idx)`. There is no experimenter key — the hist sits in one run's registry, so which run a row came from is answered by where the db is, the same way `node_hist` needs no `run_name`.

```python
hist = collectors.hist

pd.DataFrame(hist.get_hist()).groupby(['collector_name', 'status']).size()

for row in hist.get_hist(status='error'):
    print(row['collector_name'], row['node_name'],
          row['info']['phase'], row['info']['type'], row['info']['message'])
```

| `status` | |
|---|---|
| `'collected'` | returned a result |
| `'empty'` | returned `None` without raising |
| `'error'` | raised; `info` carries `{phase, type, message, traceback}` |

The `'empty'` case is the point of the split: a mis-set `output_var` and a crash both produced `None` before, and were indistinguishable. `phase` is one of `'output'`, `'ext'`, `'collect'`, `'push'`.

The fold belongs in the key even though a Collector's own unit is the node: a failure happens in one fold, and which one is where the analysis starts.

This is a log, not a gate — nothing is skipped because of it. A collector failing never interrupts the run.

---

## MetricCollector

Computes a metric against the ground truth for each fold.

```python
collectors.set_collector(
    'logloss', 'mllabs.collector.MetricCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {'edges': {'y': '{target}'}}},
    params={'output_var': '-1:',
            'metric_func': {'__callable__': 'sklearn.metrics.log_loss'},
            'include_train': True},
)
```

`output_var` is a DSL string selecting the prediction columns from the output — `'-1:'` for the last column, `None` for all of them. Results go straight into a `metrics.db` as each inner fold finishes.

```python
mc = collectors.get_collector('logloss')

mc.get_metric('lgb1')                     # per-fold Series
mc.get_metrics(['lgb1', 'lgb2'])          # DataFrame

mean, std = mc.get_metrics_agg(nodes=None, inner_fold=True,
                               outer_fold=True, include_std=True)
```

### ProbToLabel

Wraps a label-based metric so it can score `predict_proba` output. Probability columns follow sorted class order, the sklearn convention.

```python
params={'output_var': None,
        'metric_func': {'__ref__': 'mllabs.collector.ProbToLabel',
                        '__params__': {
                            'metric_func': {'__callable__': 'sklearn.metrics.f1_score'},
                            'var': '{target}',
                            'thresholds': [0.4, 0.5, 0.3]}}}
```

`var` is a DSL string naming the target. `thresholds` is `None` for argmax, a float for a binary threshold, or a per-class list — where the highest-probability class that clears its threshold wins, falling back to argmax when none do. The label classes are read from the Experimenter when the Collector attaches.

---

## StackingCollector

Collects out-of-fold predictions to stack on.

```python
collectors.set_collector(
    'stack', 'mllabs.collector.StackingCollector',
    {'__ref__': 'mllabs.Connector',
     '__params__': {'node_query': ['lgb1', 'xgb1'], 'edges': {'y': '{target}'}}},
    params={'output_var': '1:', 'method': 'mean'},
)
```

The Connector's `'y'` edge is not only for matching — `get_dataset(include_target=True)` reads it to append the target column.

```python
ds = collectors.get_collector('stack').get_dataset(e)
```

`get_dataset` takes the Experimenter, because index, target and fold count come from it at read time. What is stored on disk is the raw per-fold results, so aggregation stays a read-time decision rather than something baked in.

!!! warning "Every outer fold must run in one `exp()` call"
    Results are buffered in memory and written only when all outer folds have arrived. If some folds were already `'built'` and got skipped, the buffer never fills and no file appears. Clear the fold history and run them all.

For full out-of-fold coverage the outer splitter has to partition the data — a `KFold`, not a `ShuffleSplit`.

---

## ModelAttrCollector

Collects a model attribute per fold — feature importances, evaluation curves, whatever the adapter exposes in `result_objs`.

```python
collectors.set_collector(
    'importance', 'mllabs.collector.ModelAttrCollector',
    {'__ref__': 'mllabs.Connector',
     '__params__': {'processor': 'lightgbm.LGBMClassifier', 'edges': {'y': '{target}'}}},
    params={'result_key': 'feature_importances'},
)
```

The adapter is inferred from the Connector's processor. Common keys are `feature_importances`, `evals_result`, `coef`, and CatBoost's `feature_importances_pvc` / `feature_importances_interaction`.

```python
mac = collectors.get_collector('importance')

mac.get_attr('lgb1')                    # [[inner, ...], ...] per outer fold
mac.get_attr('lgb1', idx=0)
mac.get_attrs_agg('lgb1', agg_inner=True, agg_outer=True)   # Series
```

`agg_outer=True` needs `agg_inner=True`; `agg_inner=False` returns the un-aggregated frame. Aggregation only applies to mergeable results.

---

## SHAPCollector

```python
collectors.set_collector(
    'shap', 'mllabs.collector.SHAPCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {'processor': 'lightgbm.LGBMClassifier'}},
    params={'data_filter': {'__ref__': 'mllabs.RandomFilter',
                            '__params__': {'n': 500, 'random_state': 0}}},
)
```

`explainer_cls` defaults to `shap.TreeExplainer`. `data_filter` subsamples train and valid before the values are computed.

```python
sc = collectors.get_collector('shap')

sc.get_feature_importance('lgb1', idx=0)                    # per inner fold
sc.get_feature_importance_agg('lgb1', agg_inner='mean', agg_outer='mean')
```

Setting either aggregation to `None` keeps that axis — `agg_outer=None` gives a DataFrame, `agg_inner=None` a MultiIndex. Multiclass arrays `(n_samples, n_features, n_classes)` are averaged over the class axis first.

---

## OutputCollector

Saves raw outputs per fold.

```python
collectors.set_collector(
    'outputs', 'mllabs.collector.OutputCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {}},
    params={'output_var': None, 'include_target': True},
)
```

```python
oc = collectors.get_collector('outputs')
oc.get_output('lgb1', 0, 0)
oc.get_outputs('lgb1')            # {(outer_idx, inner_idx): entry}
```

---

## ProcessCollector

Predicts on external data — a test set — using the same fitted upstream nodes the fold used.

```python
collectors.set_collector(
    'test_preds', 'mllabs.collector.ProcessCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {'node_query': 'lgb.*'}},
    params={'ext_data': test_df, 'output_var': None, 'method': 'mean'},
)
```

Inner-fold predictions are aggregated per outer fold when collected, outer folds when queried.

```python
pc = collectors.get_collector('test_preds')
pc.get_output(nodes=None, agg='mean')
pc.get_output(nodes=['lgb1'], agg='mean')
```

`ext_data` is a live DataFrame, which is exactly why `params` is the one pickled part of a Collector definition.

## Related

- [Experimenter & Trials](experimenter-trials.md)
- [edges DSL](edges-dsl.md) — `output_var` and Connector `edges` both use it
