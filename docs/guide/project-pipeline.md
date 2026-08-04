# Project & Pipeline

## Creating a project

A `Project` is a directory. It hands out paths and owns what is shared across runs — the pipelines, the Collector registry, the Trial store, the cache.

```python
from mllabs import Project

project = Project('exp')          # created if missing
```

Everything else is reached from it:

```python
project.pipeline_builder('main')       # a PipelineBuilder under pipelines/main/
project.collectors()                   # the Collector registry
project.trials                         # the project-wide TrialStore
project.list_experimenters()           # names only
```

A project can hold several pipelines, each keyed by name with its own version counter. `'pipeline'` is the default name.

## Declaring the DataSource

The DataSource describes the raw input: a variable type per column, and which columns are targets.

```python
p = project.pipeline_builder('main')

p.set_datasource(
    {'age': 'numerical', 'income': 'numerical',
     'city': 'nominal', 'signup': 'datetime', 'target': 'binary'},
    targets=['target'],
)
```

Valid types are `numerical`, `ordinal`, `nominal`, `text`, `binary`, `datetime`.

The schema is what `check_data_compatibility()` verifies a dataset against, and a change to it is what can stale downstream nodes — but only the nodes that actually read the changed columns.

## Nodes

```python
p.set_node('scale',
           processor='sklearn.preprocessing.StandardScaler',
           method='fit_transform',
           edges={'X': '{age, income}'},
           desc='scale the numeric block')
```

`processor` is a **string**, never a class object. Nothing is imported until the node runs — see [Pipeline](../concepts/pipeline.md) for why that matters. `edges` is a [DSL string](edges-dsl.md).

Nodes can read other nodes:

```python
p.set_node('ohe',
           processor='sklearn.preprocessing.OneHotEncoder',
           method='fit_transform',
           edges={'X': '{city}'},
           params={'sparse_output': False, 'handle_unknown': 'ignore'})

p.set_node('poly',
           processor='sklearn.preprocessing.PolynomialFeatures',
           method='fit_transform',
           edges={'X': 'scale:(*)'})
```

Node names cannot contain `__` or any of `/ \ < > : " | ? *`.

## Groups

A group holds configuration several nodes share. Node values win over group values, and a group can have a parent.

```python
p.set_grp('pre', method='transform')
p.set_grp('pre_ft', method='fit_transform', edges={'y': '{target}'})

p.set_node('scale', grp='pre', processor='sklearn.preprocessing.StandardScaler',
           edges={'X': '{age, income}'})
p.set_node('tgt_city', grp='pre_ft', processor='sklearn.preprocessing.TargetEncoder',
           edges={'X': '{city}'})
```

Groups do not survive `build()` — the inheritance is resolved and the original group name is kept only as a display `label`.

!!! warning "`set_grp` writes every field you pass"
    With `exist='diff'` (the default), a detected change assigns **all** the given values, and the ones you omitted become `None`/`{}`. Restate what you want to keep:

    ```python
    p.set_grp('pre', method='transform', params={'with_std': False})
    ```

## `exist` — what happens when the name is taken

| Value | |
|---|---|
| `'diff'` | default for `set_grp`/`set_node`: update only if something differs |
| `'skip'` | leave the existing definition alone |
| `'error'` | raise |
| `'replace'` | overwrite unconditionally |

`desc` is updated even on the `'diff'` skip path, and it never triggers a rebuild.

## Parameters

`params` holds plain data — scalars, lists, dicts — or a reference spec. Live objects are rejected at definition time, and the error message shows the form to use instead.

```python
p.set_node('cat', processor='catboost.CatBoostClassifier',
           params={
               'n_estimators': 1000,
               'cat_features': {'__ref__': 'mllabs.ColSelector',
                                '__params__': {'dsl_string': '*@categorical'}},
           })
```

Two spec forms are understood, both resolved only when the processor is constructed:

- `{"__ref__": "module.Class", "__params__": {...}}` — instantiate
- `{"__callable__": "module.func"}` — the function object itself, uncalled (metric functions, for instance)

## Building a version

```python
pipeline = project.build_pipeline(p)
pipeline.version        # 1, then 2, then 3 …
```

`build_pipeline()` builds the current definition and saves it as the next version. There is no content de-duplication: rebuilding an unchanged builder still mints a new version, which is harmless — a version identical in content stales nothing when adopted.

`p.build()` on its own returns an in-memory Pipeline with `version = None`, useful in tests.

```python
project.list_pipeline_versions('main')
project.load_pipeline('main')        # latest
project.load_pipeline('main', 2)     # a specific version
```

## Inspecting

```python
p.get_node_names()                   # every node
p.get_node_names('tgt_.*')           # by regex
p.get_node_spec('scale')             # the resolved ProcessorSpec
p.compare_nodes(['m1', 'm2'])        # per-processor DataFrames of the differences

from IPython.display import Markdown, display
display(Markdown(p.desc_pipeline()))     # Mermaid diagram of the graph
display(Markdown(p.desc_node('scale')))  # one node and its neighbours
```

`desc_pipeline` and `desc_node` need the group hierarchy, so they are builder-only.

## Editing an existing pipeline

The builder is backed by SQLite and reloads itself, so `project.pipeline_builder('main')` in a fresh session returns what you last defined. `sync()` re-reads the database and reports what changed, including nodes whose *inherited* values moved because their group did.

```python
p.rename_grp('pre', 'preprocess')
p.remove_node('poly')
p.copy_nodes(['scale', 'ohe'])       # into another builder
```

Editing the builder never disturbs a run in progress: runs hold the built Pipeline, and adopting a new version is an explicit `set_pipeline()` call — see [Experimenter & Trials](experimenter-trials.md).

## Related

- [edges DSL](edges-dsl.md)
- [Pipeline concepts](../concepts/pipeline.md)
- [State Model](../concepts/state-model.md) — what adopting a new version invalidates
