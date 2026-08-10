# Architecture

```
Project ─────────── owns the directory layout and what is project-wide
  │                 (dataset, pipeline, TrialStore, cache, the runs it manages)
  │
  ├─ PipelineBuilder ──build()──► Pipeline    mutable definition → immutable graph
  │                                           building publishes a version
  ├─ Experimenter ─── evaluates Trials with cross-validation      exp/{name}/
  │
  └─ Trainer ──────── trains Predictors on full data          trainers/{name}/
        │
        └─ to_inferencer() ──► Inferencer     standalone, no Trainer at serve time
```

## Project

`Project` hands out paths from one root and holds the state that spans runs:

- the **dataset** (`data` / `aug_data`) — the one thing a run cannot restore from its own directory
- the pipeline — one per project, with its version lifecycle
- the `TrialStore` — every Trial definition and every fold outcome, from every Experimenter
- the shared `DataCache`
- the Experimenters and Trainers it manages: `list_experimenters()` for the names, `experimenters` for the objects

Collectors are not among them — a registry belongs to the run that writes into it.

Everything else belongs to an individual run. A `Project` is a convenience over the components, not a requirement: each of them takes a path and works standalone.

## A run owns its own state

An `Experimenter` or `Trainer` keeps, inside its own directory, its splitter, the Pipeline it adopted, its node artifacts and its history. So a run reopens from its path alone:

```python
e = Experimenter.load_experimenter('exp/cv', df)     # no Project involved
t = Trainer.load_trainer('trainers/final', df)
```

The `pipeline_version` a run records is **provenance** — it names which project version its copy was taken from, and is not needed to reopen anything.

This is why `Project` indexes run *names* and nothing more: there is no second copy of a run's state to drift out of sync. It does hold the live objects it has opened, so the same name gives back the same one — two instances over a single directory would each carry their own Collectors and node caches.

## Experimenter — evaluates Trials

`Experimenter` splits the data with an outer splitter (`sp`) and an optional inner splitter (`sp_v`), then:

- `build()` runs the Pipeline's nodes fold by fold
- `exp(trial_names)` runs **Trials** — the candidate models — against those node outputs

Trials live outside the Pipeline. A Trial is a candidate *being compared*, which is why its definitions and per-fold outcomes go to the project-wide `TrialStore`: results only mean something next to other results. That ownership is why a run is handed names rather than definitions — it is given the store once, at construction, and adding a Trial to the project is `Project.set_trial`, not a side effect of running one.

**Collectors** attach to Trial runs and capture what happens — metrics, model attributes, SHAP values, out-of-fold predictions for stacking.

## Trainer — trains Predictors

`Trainer` trains on splits of the full dataset, or on all of it when given no splitter. What it trains are **Predictors**, and `train(predictors)` takes them directly.

A `Predictor` is a decision already made, so what matters is not comparability but **provenance** — `Predictor.from_trial(trial, experimenter=...)` records which Trial and which Experimenter justified it. Its registry is per-Trainer for the same reason.

Predictors get their own artifact store under `trainers/{name}/__predictors/`, separate from the Pipeline nodes, so the two are told apart structurally rather than by which history table happened to record them.

## Inferencer

`Inferencer` holds the fitted processors and the specs needed to wire them — not a Pipeline, since only `edges` is actually needed at serve time. Given new data it resolves the graph in memory, per split, and aggregates across splits (`mean`, `mode`, or a callable).

It has no dependency on `Experimenter`, `Trainer` or `Project`, and serializes to a single file.

## Related

- [Pipeline](pipeline.md) — the builder/built split and what a node is
- [State Model](state-model.md) — node states and how staleness is decided
- [Data Flow](data-flow.md) — how a node's inputs get assembled
