# ml-labs

ML pipeline and experiment management library.

## Overview

ml-labs separates *what* a model workflow is from *when* and *on what data* it runs. You declare preprocessing as a graph of nodes, declare the models you want to compare, and let the library execute them fold by fold, cache what it can, persist what it produces, and record what happened.

A project has five pieces:

| | |
|---|---|
| **Project** | Owns the directory layout and everything that is genuinely project-wide: the pipelines, the Collector registry, the Trial store, the shared cache. |
| **PipelineBuilder → Pipeline** | A mutable builder that produces an immutable node graph. Preprocessing only. |
| **Experimenter** | Evaluates **Trials** — candidate models — with cross-validation. |
| **Trainer** | Trains **Predictors** — the candidates you chose — on full data. |
| **Inferencer** | Applies the trained processors to new data, standalone. |

```python
from mllabs import Project, Trial

project = Project('exp')

p = project.pipeline_builder('main')
p.set_datasource({'age': 'numerical', 'city': 'nominal', 'target': 'binary'})
p.set_node('scale', processor='sklearn.preprocessing.StandardScaler',
           method='fit_transform', edges={'X': '{age}'})

e = project.experimenter('cv', df, pipeline_name='main',
                         pipeline_version=project.build_pipeline(p).version)
e.build()

trial = Trial('lgb', 'lightgbm.LGBMClassifier',
              {'X': 'scale:(*)', 'y': '{target}'}, method='predict')
e.exp([(trial, 0, 0)], project.trials)
```

Three ideas run through all of it:

**Definitions are declarations, not objects.** A node names its processor as `"module.ClassName"`, its inputs as a DSL string, its parameters as plain data. Nothing is imported or instantiated until the moment it runs — so a pipeline is serializable, and importing `mllabs` never drags in TensorFlow.

**A run owns its state.** An Experimenter or Trainer keeps its splitter, its adopted Pipeline and its artifacts in its own directory, and reopens from that path alone — `Experimenter.load_experimenter(path, data)` needs no `Project`.

**Identity is by value.** There are no content hashes or generation counters. Whether a node's artifact is stale is decided by comparing two Pipeline versions field by field.

## Where to go next

- [Concepts](concepts/index.md) — the model behind the API: architecture, the pipeline graph, node states, how data moves
- [User Guide](guide/project-pipeline.md) — building a pipeline, running experiments, training, collecting results
- [edges DSL](guide/edges-dsl.md) — the syntax that wires nodes together
- [API Reference](reference/index.md)

## Installation

```bash
pip install ml-labs
```

Optional dependencies:

```bash
pip install ml-labs[xgboost]
pip install ml-labs[lightgbm]
pip install ml-labs[all]
```
