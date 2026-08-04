# Concepts

This section explains the model behind the API — how ml-labs describes an ML workflow, decides what has to be re-run, and moves data through the graph.

---

## [Architecture](architecture.md)

`Project` owns what is project-wide; each **run** — an `Experimenter` or a `Trainer` — owns its own state and reopens from its own directory. An Experimenter evaluates **Trials**, a Trainer trains **Predictors**, and an `Inferencer` serves the result standalone.

---

## [Pipeline](pipeline.md)

A mutable `PipelineBuilder` produces an immutable `Pipeline` — preprocessing nodes only, with group inheritance already resolved. Definitions are declarations: a processor is named by string and nothing is instantiated until it runs.

---

## [State Model](state-model.md)

A node is `init`, `built`, or `error`, per fold. What gets skipped is decided from disk and history, not by comparing definitions — so redefining a Trial does not re-run it. Adopting a new Pipeline version is the one thing that invalidates work automatically, by diffing the two versions.

---

## [Data Flow](data-flow.md)

`edges` stays a DSL string everywhere and becomes actual columns only at execution time, against real data. This page covers the `data_dict` a processor receives, the shared cache and how it is keyed, and why inference can run on data with no target column.
