# Project & Pipeline

## Creating a project

A `Project` is a directory. It hands out paths and owns what is shared across Experimenters and Trainers — the dataset, the pipeline, the Trial store, the cache.

```python
from mllabs import Project

project = Project('exp', data=df)      # created if missing
```

The dataset belongs here because it is the one thing an Experimenter or Trainer cannot restore from its own directory. Give it once and `add_experimenter` / `add_trainer` and the registries stop asking for a dataframe. `project.set_data(df)` sets it later.

Anything beyond the main dataset — a held-out test set, rows to append to inner train splits (`aug_data`) — goes through `project.ext_data`, a named registry that reads fresh from disk on every access rather than caching:

```python
project.ext_data.register('extra', extra_df)
project.add_experimenter('cv5', aug_data='@ext:extra')   # resolved at open time
```

`'@ext:name'` is accepted anywhere `aug_data=` is: `add_experimenter`, `add_trainer`, and the `Experimenter`/`Trainer` constructors directly.

Everything else is reached from it:

```python
project.pipeline                       # the PipelineBuilder — one per project
project.trials                         # the project-wide TrialStore
project.list_experimenters()           # names only
project.experimenters                  # the objects, opened on demand
```

Collectors are not among them — a registry belongs to the Experimenter that writes into it, as `e.collectors`. See [Collectors](collectors.md).

**One pipeline per project.** What you actually want from several is to refer back to an earlier definition, and that is a version's job rather than a name's — so `pipeline/` holds one builder and its numbered versions.

## Adding and removing Experimenters and Trainers

```python
project.add_experimenter('cv5', sp=..., splitter_params={'y': 'target'})
project.add_trainer('final')

e = project.experimenters['cv5']       # reach an existing one
t = project.trainers['final']

project.remove_experimenter('cv5')     # deletes the directory
```

`add_*` is strictly an addition: a taken name raises. Constructing an Experimenter splits the data afresh and resets its provenance, so doing that over an existing one is damage rather than a reopen — and an accessor that quietly created when the name was free would turn a typo into a second Experimenter instead of an error.

Two different questions decide whether a name is free. `ProjectStore` says what this project *manages*; `ExperimenterStore.stored_at(path)` says whether the directory is *occupied*, possibly by something built outside the project. Either one refuses, with a message saying which.

`remove_experimenter` also drops that name's rows from `experiment_hist`. The name is what keys them, so left behind they would attach to whatever is added under it next and `exp()` would skip folds it had never run. Trial *definitions* stay — those belong to the project.

The registries open only names they are not already holding, so what `add_*` returned is what comes back:

```python
e = project.add_experimenter('cv5', ...)
project.experimenters['cv5'] is e      # True
```

`list_experimenters()` answers the cheaper question — which names exist — without opening anything.

## Declaring the DataSource

The DataSource describes the raw input: a variable type per column, and which columns are targets.

```python
p = project.pipeline

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

**Building publishes.** There is no separate publish step:

```python
pipeline = project.pipeline.build()
pipeline.version        # 1, then 2, then 3 …
pipeline.status         # 'published'
```

A version appears exactly when the definition changes. Build an unchanged builder and you get back the version it already has, not a duplicate of it — so every history row can name the definition it ran on, and there is no unnumbered snapshot to run against.

A project is created with **version 0**: the empty Pipeline, published. That is what "nothing has been built yet" *is* here, rather than an absence every caller has to handle — `load_pipeline()` always returns something, a Trainer can adopt it, and building an untouched builder gives back v0 because nothing was defined and so nothing changed.

| status | |
|---|---|
| `published` | stored under a number. Every version is one, and nothing is ever demoted |
| `draft` | a snapshot that was never stored — `version = None`. Cannot be adopted |

Being stored *is* what publishing means, so the `versions` table has no status column and a draft can never appear in it. The distinction exists at runtime only.

```python
project.list_pipeline_versions()
project.load_pipeline()          # the latest — a read, never a write
project.load_pipeline(2)         # a specific version
project.remove_pipeline_version(1)   # any but the latest
```

The latest is what an omitted version number resolves to, so it cannot be removed — deleting it would silently change what the next `add_experimenter` adopts. Removing an older one breaks nothing that ran against it: every Experimenter and Trainer keeps its own Pipeline copy. What is lost is what a provenance pointer refers to.

### Publishing and propagating

Building alone leaves everything else where it was, and the ordering that matters is easy to miss: `set_trial` stamps a Trial with the **latest published** version, adopting is what gives an Experimenter a version to compare against, and `exp()` refuses a stamp that is not the adopted one. Build without adopting and the Trials you author next carry a version nothing holds — which surfaces as a refusal at `exp()`, some distance from the omission.

`publish_pipeline()` is the pair, and reports what it cost:

```python
report = project.publish_pipeline()
report['version']            # 3
report['experimenters']      # {'phase2': ['scaler', 'tgt_delta']}   nodes dropped
report['trainers']           # {} — not touched by default
```

Building an unchanged definition returns the version it already has, so calling it out of habit costs nothing and moves nothing.

**Trainers stay out unless asked for.** The two sides do not pay the same price: an Experimenter loses node artifacts the next `build()` makes again, while adopting on a Trainer *retires* every Predictor whose inputs changed, and that is terminal. Work moves the other way round anyway — experiment, read the results, then promote into a Trainer.

```python
project.publish_pipeline(dry_run=True, trainers=True)   # price it first
project.publish_pipeline(trainers=True)                 # then commit
```

Leaving a Trainer behind strands nothing. It keeps its own Pipeline copy and its Predictors carry the version they trained against, so it goes on working where it is, and `train()` refuses a mismatch rather than quietly doing the wrong thing.

### Looking without publishing

`draft()` returns the same snapshot `build()` does, unregistered:

```python
snapshot = project.pipeline.draft()
snapshot.version, snapshot.status    # (None, 'draft')

project.stale_nodes()
# {'experimenters': {'phase2': ['scaler']}, 'trainers': {'final': ['scaler']}}
```

This is how a question stays a question — `stale_nodes()` and `publish_pipeline(dry_run=True)` both lean on it, so asking what an edit would cost is not what commits the edit. A draft cannot be adopted: without a number, nothing that ran against it could say what that was.

The two sides answer under separate keys because `exp/{name}` and `trainers/{name}` are separate namespaces — one name can exist in both. Pass `experimenter=` or `trainer=` to narrow to exactly one, and the other side comes back empty. Both report **nodes**, which understates a Trainer: for the retirements too, use `publish_pipeline(dry_run=True, trainers=True)`.

`PipelineBuilder()` constructed without a path has nowhere to publish, so `build()` raises there; `draft()` is how you get the snapshot.

## Inspecting

```python
p.get_node_names()                   # every node
p.get_node_names('tgt_.*')           # by regex
p.get_node_spec('scale')             # the resolved ProcessorSpec

from mllabs import compare_specs
compare_specs({n: p.get_node_spec(n) for n in ['n1', 'n2']})  # common vs. differing, per processor

from IPython.display import Markdown, display
display(Markdown(p.desc_pipeline()))     # Mermaid diagram of the graph
display(Markdown(p.desc_node('scale')))  # one node and its neighbours
```

`desc_pipeline` and `desc_node` need the group hierarchy, so they are builder-only.

## Editing an existing pipeline

The builder is backed by SQLite and reloads itself, so `project.pipeline` in a fresh session returns what you last defined. Asking twice gives the same object rather than a second builder over the same database, which would let two in-memory copies drift apart. `sync()` re-reads the database and reports what changed, including nodes whose *inherited* values moved because their group did.

```python
p.rename_grp('pre', 'preprocess')
p.remove_node('poly')
p.copy_nodes(['scale', 'ohe'])       # into another builder
```

Editing the builder never disturbs an execution in progress: an Experimenter or Trainer holds the built Pipeline, and adopting a new version is an explicit `set_pipeline()` call — see [Experimenter & Trials](experimenter-trials.md).

## Related

- [edges DSL](edges-dsl.md)
- [Pipeline concepts](../concepts/pipeline.md)
- [State Model](../concepts/state-model.md) — what adopting a new version invalidates
