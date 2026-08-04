# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-05

A rework of how a project, a pipeline and a run relate to each other. **Almost
every entry point changed shape**; code written against 0.8.x will not run
unmodified. Three ideas drive the rest:

- **`Project` is the root.** It owns the directory layout and what is genuinely
  project-wide; a run owns its own state and reopens from its own directory.
- **Definitions are declarations.** A processor is named by string, inputs by a
  DSL string, params by plain data — nothing is imported or instantiated until
  it runs.
- **Identity is by value.** No content hashes, no generation counters. Staleness
  is a field-by-field diff between two Pipeline versions.

### Added

- `Project` (`mllabs/_project.py`): owns the directory layout, the pipelines,
  the `Collectors` registry, the `TrialStore`, the shared cache, and an index of
  run names. Factories: `pipeline_builder`, `collectors`, `experimenter`,
  `trainer`, `load_experimenter`, `load_trainer`, `build_pipeline`,
  `load_pipeline`, `list_pipeline_versions`
- `PipelineBuilder` → `Pipeline`: a mutable, SQLite-backed builder and the
  immutable node graph `build()` produces. `Experimenter`/`Trainer`/`Inferencer`
  accept only the built form, so editing a builder cannot leak into a run in
  progress. `Project.build_pipeline()` persists each build as the next version
- `Trial` and `make_trials` (`mllabs/_trial.py`), `TrialStore`: models moved out
  of the Pipeline. A Trial is a candidate being compared, so definitions and
  per-fold outcomes live in a project-wide store
- `Predictor` and `PredictorStore`: the Trainer-side counterpart. A Predictor is
  a decision already made, so it carries provenance (`src_trial`,
  `src_experimenter`) via `Predictor.from_trial()` and its registry is
  per-Trainer
- `ProcessorSpec`: the single execution-unit representation nodes, Trials and
  Predictors all resolve to — `name`, `processor`, `edges`, `method`, `adapter`,
  `params`
- edges DSL (`mllabs/_edge_dsl.py`): `{a, b}` set literals, regex patterns,
  `node:(...)` namespaces, `+`/`-`/`&`, Python slices, and `@selector` suffixes
- Built-in dtype column selectors: `@numeric`, `@categorical`, `@binary`,
  `@float`, `@int`, `@string`
- `Collectors` registry with `CollectorStore`: definitions persist as parts to
  reassemble — a plain-text row plus pickled constructor params — never as a
  pickled instance. `set_collector` writes through
- `CollectHist`: one row per (collector, experimenter, node, outer, inner), with
  `status` split into `'collected'` / `'empty'` / `'error'`. `'empty'` is the
  point — a mis-set `output_var` and a crash both produced `None` before
- `Experimenter.open_os_log()` / `close_os_log()` / `os_log()`: capture
  OS-level (fd 1/2) output from native libraries, per worker when `n_jobs > 1`
- `stack_evals_result` (`mllabs/adapter/_base.py`), shared by all four adapters
- `Trainer` is exported from the package root
- Full documentation rewrite, including a page for the edges DSL

### Changed

- **`edges` is a DSL string**, not a list of `(node_name, var_spec)` tuples, and
  stays one through inheritance, storage and comparison. Columns are resolved
  only at execution time, against real data. `set_grp`/`set_node` validate
  structure alone
- **`processor` must be a `"module.ClassName"` string.** Class objects are
  rejected at definition time. `adapter` takes a string or a `{"__ref__": ...}`
  spec, and `params` takes plain data or ref specs — all resolved only when a
  processor is constructed. This is also a correctness fix: `Connector` compares
  `processor` as a plain string, so a class-defined node could never match a
  string-configured Connector, silently collecting nothing
- LightGBM `early_stopping` must be given as a dict of kwargs; a callback
  instance is no longer accepted. `mllab_sampler` must be a ref spec
- `Experimenter` and `Trainer` take no `Project` — only a `cache`. Each builds
  its own store from its path, keeps its own `pipeline.pkl`, and reopens with
  `Experimenter.load_experimenter(path, data)` / `Trainer.load_trainer(path, data)`.
  `(pipeline_name, pipeline_version)` survives as provenance only
- `set_pipeline(pipeline)` takes an already-built Pipeline object
- `Experimenter.exp(trials, trial_store, ...)` takes explicit
  `(trial, outer_idx, inner_idx)` triples and a required `trial_store`
- `Trainer.train(predictors)` takes what to train directly
- Pipeline definition is stored in SQLite, editable without a live kernel
- `MetricCollector` writes to `metrics.db` as each inner fold completes,
  replacing per-node pickle files (#120)
- `NodeStore` is per run rather than per fold, and owns both artifacts and
  history; `DataCache` is keyed by a per-flow scope id
- `Trainer` accepts a native DataFrame as well as a `DataWrapper`; `wrap()` is
  idempotent

### Removed

- Node `role` and the "stage" vocabulary. A Pipeline holds nodes only
- Head nodes in the Pipeline — they are Trials and Predictors now
- The `finalized` node state, and the Experimenter's open/closed session:
  `finalize`, `reinitialize`, `close_exp`, `reopen_exp`, `open`, `close`,
  `status`
- Node `serial` and `Trial.content_key` — superseded by value comparison
- `Experimenter.create` / `load` / `set_grp` / `set_node` / `add_collector` /
  `get_collector` / `collect` / `get_collect_status` / `add_trainer` /
  `process_ext`
- `Trainer.select_head`, `Trainer.set_predictors`, `Trainer.load`
- `Collector.save` / `load` and `Collector.warnings`, which was excluded from
  neither `__getstate__` nor the worker round trip and so only ever worked in a
  single process
- `Inferencer.selected_heads` → `selected_predictors`

### Fixed

- A worker killed by the OOM killer or a native segfault no longer strands the
  rest of the pool: `wait()` reports a closed pipe as readable, the resulting
  `EOFError` escaped before the stop sentinel was sent, and the surviving
  workers blocked forever holding memory and CUDA contexts. Worker death is an
  outcome now and shutdown moved into a `finally`
- Ragged `evals_result` curves (CatBoost records `eval_metric` on its own
  cadence) no longer break collection; all four adapters shared the same latent
  `pd.DataFrame({metric: list})` construction
- `stack_evals_result` pins whether padded positions survive, which pandas 2.x
  and `future_stack` disagree on — the same code gave different output per
  environment
- Multi-worker executor read completed artifacts from the flow's store instead
  of the one it was passed, failing once a Trainer fed Predictors from the node
  flow while storing them elsewhere
- `Trainer.process()` lost Predictor edges after a reload and silently yielded
  nothing
- Executor errors are keyed by full job identity, so a trial failing on several
  inner folds no longer loses all but one, and the success count is right
- `ProbToLabel` had never been migrated to the DSL and was silently broken
- `Collector` failures are recorded and no longer interrupt a run; an exception
  in the parent's `push` used to orphan the worker pool

## [0.8.0] - 2026-05-16

### Added

- `CrossFitTransformer` (`mllabs/processor/_crossfit.py`): sklearn-compatible OOF meta-feature generator for stacking
  - `fit_transform`: generates OOF predictions via CV splits; fits full estimator on all data
  - `transform`: applies full estimator fitted during `fit_transform`
  - Supports `predict_proba`, `predict`, and any estimator method; output column names derived from estimator class and classes
- `ProbToLabel` (`mllabs/collector/_metric.py`): wraps a label-based metric function with `predict_proba` → label conversion
  - `var` accepts edges-compatible variable spec (`str`, tuple, list) to fetch label classes from experimenter via `on_attach`
  - `thresholds`: `None` = argmax, `float` = binary threshold, `list` = per-class multiclass thresholds with argmax fallback
- `Inferencer.process(nodes=...)`: optional parameter to select a subset of head nodes; preserves ordering, raises `ValueError` on unknown names
- `Experimenter.get_collect_status(collector, nodes=None)`: returns `{node: status}` for collector-matched head nodes; statuses: `'collected'`/`'not_collected'`/`'finalized'`/`'error'`
- `sample_weight` edge key support: passed automatically to `fit()` via adapter pipeline; polars→pandas conversion applied in `LightGBMAdapter` and `CatBoostAdapter`; `NNClassifier`/`NNRegressor` propagate through `_make_tf_dataset`, `fit`, `_split_val`

### Changed

- `Collector` base class: `on_attach(experimenter)` / `_on_attach(experimenter)` lifecycle hooks added
  - `Experimenter.add_collector` and `collect` both call `on_attach`; experimenter identity check skips redundant re-init on repeated calls
  - `_experimenter` excluded from pickle (`save()` and `__getstate__`); reset to `None` on load
- `StackingCollector`: `experimenter` removed from constructor; index/target/data_cls init moved to `_on_attach`

### Fixed

- `MetricCollector`, `ModelAttrCollector`, `SHAPCollector`, `OutputCollector`, `ProcessCollector`: return `None` instead of raising errors when node data is absent (e.g., finalized before collect ran)
- `LMAdapter` multiclass coef concatenation: `intercept_` `expand_dims` axis corrected from 0 to 1 for multiclass `LogisticRegression`

## [0.7.0] - 2026-04-10

### Added

- `DefaultLogger`: new ANSI cursor-movement multi-session progress display — each session occupies one terminal line, redrawn in-place; falls back to plain `print` in non-TTY environments (Jupyter, pipes)
- `ProgressSessionLogger`: renamed from the old `DefaultLogger`; session_cls injection pattern preserved (`TqdmProgressSession` etc.)
- `MetricCollector`, `ModelAttrCollector`, `SHAPCollector`: ad-hoc collection without path — when `path=None`, results accumulate in `_cache` only; all query methods (`get_metric`, `get_attr`, `get_feature_importance`, etc.) work against the cache
- `Experimenter.collect`: calls `_setup(n_outer, n_inner)` before ad-hoc loop so `_flush_outer` triggers correctly for path=None collectors
- `Experimenter.add_collector`: `exist='replace'` mode added
- `NNClassifier`/`NNRegressor`: `device` parameter for explicit GPU assignment (issue #103)
- GPU device profiling and injection interface added to adapters (issue #99)

### Changed

- Collector/Executor architecture fully redesigned for parallel execution (issue #107)
  - Node storage split into `_store.py`, `_flow.py`, `_tracker.py`
  - `DataFlow`/`TrainDataFlow`: encapsulate per-fold data assembly
  - `ArtifactStore`: per-fold artifact read/write abstraction
  - `ExecuteTracker`: progress and state tracking for build/exp runs
  - Collector interface switched to push-based model
- Head node management refactored from `HeadObj` to function-based API (issue #105)
- `Trainer`/`Inferencer` modernized with `TrainDataFlow`/`InferenceDataFlow`

### Fixed

- `SHAPCollector.collect`: second data filter was overwriting `train_data` with filtered `valid_data`; return block was using `train_data['X']` for both train and valid SHAP values
- `Experimenter.remove_collector`: now deletes the collector directory from disk

## [0.6.4] - 2026-03-18

### Added

- `ProcessCollector` (`mllabs/collector/_process.py`): collects predictions on external (test) data during `exp()`
  - Passes ext data through upstream Stage processors via `Experimenter.process_ext()` per outer fold
  - Inner-fold predictions aggregated by `method` (`mean`/`mode`/`simple`); outer-fold aggregated on query
  - `get_output(nodes=None, agg='mean')`: multi-node support with nodes filter (None/list/regex) and column-wise concat
  - Disk-based storage; `save`/`load` roundtrip (ext_data and experimenter not persisted)
- `Experimenter.process_ext(data, node, idx)`: passes external data through upstream Stage fitted processors for a given outer fold, yielding assembled input per inner split
- `Connector(role=...)`: new `role` parameter (`'head'`/`'stage'`) for role-based node filtering; `None` (default) skips the check
- `Pipeline.get_node_attrs`: expose `role` in returned attrs dict

### Fixed

- `Experimenter.load`: prevent `__exp.pkl` corruption when load fails — `__init__` now accepts `_save=False` to skip the initial save; `load()` passes `_save=False`
- `Experimenter.load`: add `OutputCollector` and `ProcessCollector` to `COLLECTOR_TYPES` lookup

## [0.6.3] - 2026-03-14

### Fixed

- `ModelAdapter`: add `get_process_data(data)` — adapters can override to control input type conversion in `process()`; `TransformProcessor` / `PredictProcessor` now use this instead of bare `unwrap()`
- `LightGBMAdapter`: override `get_process_data()` with polars→pandas conversion (mirrors `get_fit_params` behavior)
- `CatBoostAdapter`: version-based polars support (`>=1.3.0`); apply polars→pandas in both `get_fit_params` and `get_process_data` for older versions
- `PolarsWrapper.get_columns()`: handle `pl.Series` correctly (return `.name` instead of `.columns`)
- `TransformProcessor`: handle non-iterable `y_columns` and `str` `result.columns` in `fit_process` / `process`
- `PredictProcessor`: handle non-iterable `y_columns` in `fit`
- `StackingCollector`: wrap `str` `target_columns` in list; use `_data_cls` for `simple` / `mean` / `mode` aggregation

## [0.6.2] - 2026-03-08

### Fixed

- Pipeline: `set_grp()` now recursively invalidates child group attrs cache (`_cascade_clear_attrs`) and node attrs cache — prevents stale adapter/params being resolved when a parent group is updated
- Inferencer: `_get_process_data()` skips `resolve_columns` for DataSource edges (`src_node=None`); uses `var` directly as column spec

## [0.6.1] - 2026-03-07

### Added

- Processor: `TypeConverter(to)` — converts all columns to a target dtype (`'str'`, `'int'`, `'float'`); supports pandas, polars, and numpy
- Pipeline: `desc` attribute on `PipelineGroup` and `PipelineNode` for free-text annotations
  - Not inherited via `get_attrs`, not compared in `diff()` — desc-only changes do not trigger rebuilds
  - Updated silently on `exist='diff'` skip path

### Fixed

- LightGBM adapter: accept `early_stopping` as a plain dict of kwargs; adapter constructs `lgb_early_stopping` internally, eliminating false param-change detection
- Experimenter: collector lifecycle errors (`_start`, `_collect`, `_end_idx`, `_end`) are now caught and stored as warnings instead of propagating exceptions

## [0.6.0] - 2026-03-05

### Added

- `mllabs.nn`: sklearn-compatible neural network estimators (`NNClassifier`, `NNRegressor`) with automatic categorical embedding support
  - Auto-detects categorical columns from pandas `Categorical` / polars `Categorical` dtype
  - Auto-computes embedding dimensions `max(1, min(50, (cardinality+1)//2))`; per-column override via `embedding_dims` dict
  - Modular components: `SimpleConcatHead`, `DenseHidden`, `LogitOutput`, `BinaryLogitOutput`, `RegressionOutput`
  - `hidden` parameter accepts a dict of `DenseHidden` constructor kwargs
  - `fit(X, y, eval_set=None, callbacks=None)` with constructor callbacks and fit callbacks merged; early stopping auto-appended
  - Pickle support via `__getstate__` / `__setstate__` — weights saved only, architecture rebuilt from `col_info_` on load
- `NNAdapter` (`mllabs.adapter`): ml-labs adapter for `NNClassifier` / `NNRegressor`
  - Passes inner-validation fold as `eval_set`
  - Epoch-based progress logging via `_ProgressCallback`
  - `evals_result` exposed as `result_obj` for `ModelAttrCollector`
- `pyproject.toml`: `tensorflow` optional dependency (`pip install ml-labs[tensorflow]`)
- Experimenter: `collect()` accepts `nodes` parameter to limit collection scope
- Docs: nn module user guide (`guide/nn.md`) and API reference (`reference/nn.md`)
- Docs: Concepts index page

### Fixed

- `mllabs.nn._estimator`: guard `import tensorflow` with try/except so tf-free environments can import the package

## [0.5.0] - 2026-02-27

### Added

- Documentation: full MkDocs-based site (Material theme) published to GitHub Pages
  - Concepts: architecture, pipeline, state model, data flow
  - User guides: Pipeline & Experimenter, Trainer & Collectors, Adapters, Processors
  - Serving guide: Inferencer export, save/load, inference
  - API reference: all public classes auto-generated from docstrings via mkdocstrings

### Fixed

- `pyproject.toml`: add `README.md`, fix package description typos

## [0.4.0] - 2026-02-26

### Added
- Processor: `FrequencyEncoder`, `ColSelector`, improved `CatPairCombiner` / `CatConverter` / `CatOOVFilter`
- Processor: X-less `TransformProcessor` support for 1D transformers (e.g. `LabelEncoder`)
- SHAPCollector: `get_feature_importance`, `get_feature_importance_agg` analysis methods
- Pipeline: `exist='diff'` mode for `set_grp` / `set_node` (now default)
- Experimenter: `get_collector`, `remove_collector`, `get_trainer`, `remove_trainer`
- Logger: `rename_progress(title)` method to `BaseLogger` / `DefaultLogger`
- Logger: trailing character cleanup on progress line overwrite
- Examples: Kaggle Playground S6E2 end-to-end notebooks (EDA, feature engineering, modeling)

### Fixed
- Experimenter: `build` `rebuild` parameter not working
- Experimenter: `reopen_exp` losing collector data after `close_exp`
- Experimenter: `close_exp` not persisting status on save
- Experimenter: `exp` error handling not propagating correctly
- Pipeline: `set_grp` not collecting affected nodes when only edges change
- Adapter: safe recursive params comparison and `__eq__` for diff mode
- Processor: `CategoricalPairCombiner` output dtype to Categorical
- ExpObj: error state not persisted to disk (`error.txt`)
- Experimenter: parameter order normalized to `(node, idx)` in `get_node_output` family
- Inferencer: `process` now returns native pandas/numpy (unwrapped)

### Refactoring
- Experimenter: replace error stack trace logging with `show_error_nodes`
- Experimenter: encapsulate internal `collectors` / `trainers` dicts behind accessor methods

## [0.3.0] - 2026-02-20

### Added
- Inferencer: apply trained pipelines to new data with automatic split aggregation
  - `Trainer.to_inferencer(v)` extracts trained Processors into a standalone Inferencer
  - `process(data, agg)` supports mean/mode/callable/None aggregation
  - Single-file save/load, fully independent of Trainer

### Fixed
- StackingCollector: fix `get_dataset` shape mismatch error
- StackingCollector: preserve target data type in `get_dataset`
- MetricCollector: fix FutureWarning by adding `future_stack=True`

### Changed
- StackingCollector: add `experimenter` to constructor, remove `include_target`
- StackingCollector: remove `experimenter` from `get_dataset`, add `include_target`
- StackingCollector: pre-build index and target at construction time, remove `_build_sort_order`

## [0.1.0] - 2026-02-12

### Added
- Pipeline: DAG-based node graph management with groups and edges
- Experimenter: experiment execution engine with LRU caching and state management
- Trainer: cross-validation training pipeline with split management
- ExpObj/TrainObj: per-node build and experiment object lifecycle
- Collectors: MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- Adapters: scikit-learn, XGBoost, LightGBM, CatBoost, Keras
- Processors: categorical encoding, imputation, pandas/polars utilities
- Data support: pandas, polars, cuDF, NumPy via DataWrapper
- Connector: flexible node matching with regex, edges, and processor filters
- Filters: DataFilter, RandomFilter, IndexFilter for data sampling
