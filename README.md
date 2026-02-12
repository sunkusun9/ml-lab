# ml-labs

A structured machine learning experimentation framework for building, managing, and evaluating ML pipelines with cross-validation, caching, and multi-framework support.

## Installation

```bash
pip install ml-lab
```

With optional dependencies:

```bash
pip install ml-lab[xgboost]    # XGBoost support
pip install ml-lab[lightgbm]   # LightGBM support
pip install ml-lab[catboost]   # CatBoost support
pip install ml-lab[shap]       # SHAP value analysis
pip install ml-lab[polars]     # Polars DataFrame support
pip install ml-lab[all]        # All optional dependencies
```

## Key Features

- **Pipeline**: DAG-based node graph for defining ML workflows with stages (data transformation) and heads (model prediction)
- **Experimenter**: Experiment execution engine with LRU caching, state management, and error resilience
- **Trainer**: Cross-validation training pipeline with split management
- **Collectors**: Extensible data collection — metrics, stacking outputs, model attributes, SHAP values, raw outputs
- **Adapters**: Unified interface for scikit-learn, XGBoost, LightGBM, CatBoost, and Keras
- **Data Flexibility**: Support for pandas, polars, cuDF, and NumPy arrays

## Architecture Overview

```
Pipeline          Define node graphs (stages + heads) with groups and edges
    │
Experimenter      Execute pipelines, manage cache and state
    │
  ├── ExpObj      Per-node build/experiment objects (StageObj, HeadObj)
  ├── Trainer     Cross-validation training with split management
  └── Collector   Collect metrics, predictions, model attributes, SHAP values
```

**Node State Model:**
```
init ──→ built ──→ finalized
  │
  └──→ error ──→ (reset) ──→ init
```

## Quick Start

```python
from mllab import Experimenter, Connector, MetricCollector

exp = Experimenter(data=df, path="exp/my_experiment")

p = exp.pipeline
p.set_grp("scale", role="stage", processor="StandardScaler")
p.set_grp("model", role="head", processor="RandomForestClassifier",
          parent="scale", edges={"X": [(None, None)], "y": [(None, "target")]})

p.set_node("rf_default", grp="model")

exp.build(["rf_default"])
exp.exp(["rf_default"])

mc = MetricCollector("accuracy", Connector(), output_var="prediction",
                     metric_func=lambda y, pred: (y == pred).mean())
exp.add_collector(mc)
exp.collect(mc)

print(mc.get_metrics(["rf_default"]))
```

## Requirements

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- cachetools >= 5.0

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free for non-commercial use.
