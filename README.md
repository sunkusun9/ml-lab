# ml-labs

A structured machine learning experimentation framework for building, managing, and evaluating ML pipelines with cross-validation, caching, and multi-framework support.

## Installation

```bash
pip install ml-labs
```

With optional dependencies:

```bash
pip install ml-labs[xgboost]    # XGBoost support
pip install ml-labs[lightgbm]   # LightGBM support
pip install ml-labs[catboost]   # CatBoost support
pip install ml-labs[shap]       # SHAP value analysis
pip install ml-labs[polars]      # Polars DataFrame support
pip install ml-labs[tensorflow]  # Neural network estimators (NNClassifier, NNRegressor)
pip install ml-labs[all]         # All optional dependencies
```

## Key Features

- **Project**: Owns the directory layout and what is project-wide — pipelines, Collectors, the Trial store, the shared cache
- **PipelineBuilder → Pipeline**: A mutable builder producing an immutable node graph; definitions are declarations, resolved only at execution time
- **Experimenter**: Evaluates **Trials** — candidate models — with nested cross-validation, caching and error resilience
- **Trainer**: Trains **Predictors** — the candidates you chose — and exports a standalone `Inferencer`
- **Collectors**: Extensible collection — metrics, stacking outputs, model attributes, SHAP values, raw outputs — with a per-fold history of what each one did
- **Adapters**: Unified interface for scikit-learn, XGBoost, LightGBM, CatBoost and Keras
- **Data Flexibility**: pandas, polars, cuDF and NumPy arrays

## Architecture Overview

```
Project ─────────── directory layout + pipelines, Collectors, TrialStore, cache
  │
  ├─ PipelineBuilder ──build()──► Pipeline    preprocessing nodes only
  │
  ├─ Experimenter ─── evaluates Trials, fold by fold          exp/{name}/
  │
  └─ Trainer ──────── trains Predictors on full data      trainers/{name}/
        │
        └─ to_inferencer() ──► Inferencer     standalone at serve time
```

Each run keeps its splitter, its adopted Pipeline and its artifacts in its own
directory, and reopens from that path alone — no `Project` required.

**Node State Model:**
```
init ──→ built
  ▲
  └──→ error ──→ (reset_nodes) ──→ init
```

## Quick Start

```python
from mllabs import Project, Trial, Connector

project = Project('exp')

p = project.pipeline_builder('main')
p.set_datasource({'age': 'numerical', 'income': 'numerical', 'target': 'binary'},
                 targets=['target'])
p.set_node('scale', processor='sklearn.preprocessing.StandardScaler',
           method='fit_transform', edges={'X': '{age, income}'})

e = project.experimenter('cv', df, pipeline_name='main',
                         pipeline_version=project.build_pipeline(p).version)
e.build()

collectors = e.collectors
collectors.set_collector(
    'acc', 'mllabs.collector.MetricCollector',
    {'__ref__': 'mllabs.Connector', '__params__': {'edges': {'y': '{target}'}}},
    params={'output_var': '-1:',
            'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}})

trials = [Trial(f'lr_{c}', 'sklearn.linear_model.LogisticRegression',
                {'X': 'scale:(*)', 'y': '{target}'}, method='predict',
                params={'C': c})
          for c in (0.1, 1.0)]

e.exp([(t, 0, 0) for t in trials], project.trials)

print(collectors.get_collector('acc').get_metrics_agg(None)[0])
```

## Documentation

Full documentation is available at **https://sunkusun9.github.io/ml-labs/**

- [Concepts](https://sunkusun9.github.io/ml-labs/concepts/architecture/) — Architecture, Pipeline, State model, Data flow
- [User Guide](https://sunkusun9.github.io/ml-labs/guide/project-pipeline/) — Project & Pipeline, edges DSL, Experimenter & Trials, Trainer & Collectors, Adapters, Processors, Neural Networks
- [Serving Guide](https://sunkusun9.github.io/ml-labs/serving/inferencer/) — Inferencer export and inference
- [API Reference](https://sunkusun9.github.io/ml-labs/reference/index/) — Full API reference

## Requirements

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- cachetools >= 5.0

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free for non-commercial use.
