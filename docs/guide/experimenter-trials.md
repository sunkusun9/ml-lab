# Experimenter & Trials

An `Experimenter` runs one cross-validation experiment: it builds the Pipeline's nodes fold by fold, then evaluates **Trials** — the candidate models — against those outputs.

## Creating one

```python
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

e = project.experimenter(
    'cv5', df,
    sp=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    sp_v=StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42),
    splitter_params={'y': 'target'},
    pipeline_name='main',
    pipeline_version=project.build_pipeline(p).version,
)
```

- `sp` — the outer splitter. Its held-out part is the *test* fold of each outer split.
- `sp_v` — the inner splitter, giving each outer fold a train/validation pair for things like early stopping. `None` means one inner fold with no validation set.
- `splitter_params` — maps a splitter keyword to a column, so `{'y': 'target'}` makes the split stratified on that column.

The name is the identity: it is the directory under `exp/` and the key under which every fold outcome is filed in the `TrialStore`.

!!! warning "Constructing means starting over"
    The constructor recomputes the splits and writes fresh state. To continue an existing run, reopen it:

    ```python
    e = project.load_experimenter('cv5', df)
    e = Experimenter.load_experimenter('exp/cv5', df)   # no Project needed
    ```

## Adopting a Pipeline

```python
e.set_pipeline(project.build_pipeline(p))
e.pipeline_version
```

`set_pipeline` takes an already-built `Pipeline` object, not a version number. Moving from one version to another diffs them and deletes exactly the node artifacts the change invalidated — see [State Model](../concepts/state-model.md). Trials are left alone.

The adopted Pipeline is saved into the run's own directory, so reopening restores it without a `Project`.

## Building nodes

```python
e.build()                       # everything, all folds
e.build(nodes=['scale', 'ohe'])
e.build(n_jobs=4, gpu_id_list=[0], logger=logger)
```

Already-built folds are skipped. `n_jobs` is capped at the actual number of jobs, and `rebuild=True` forces a redo.

## Trials

A Trial is one configuration to evaluate. It looks like a node — same `ProcessorSpec` — but lives outside the Pipeline.

```python
from mllabs import Trial, make_trials

trial = Trial('lgb1', 'lightgbm.LGBMClassifier',
              edges={'X': 'scale:(*) + ohe:(*)', 'y': '{target}'},
              method='predict',
              params={'n_estimators': 500, 'learning_rate': 0.05},
              tag=['baseline'])
```

`make_trials` expands a grid, naming the results `{name}_{i}`:

```python
trials = make_trials('lgb', 'lightgbm.LGBMClassifier',
                     edges={'X': 'scale:(*)', 'y': '{target}'},
                     method='predict',
                     params={'random_state': 42},
                     param_grid={'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1]})
```

!!! warning "A Trial's name is its identity"
    `TrialStore` is project-wide and keyed by name, so reusing a name overwrites that definition and its artifacts. Give a new configuration a new name.

## Running

`exp()` takes explicit `(trial, outer_idx, inner_idx)` triples — fold expansion happens in your code, so the executor runs exactly the list it is given.

```python
folds = [(t, o, i)
         for t in trials
         for o in range(e.get_n_splits())
         for i in range(e.get_n_splits_inner())]

e.exp(folds, project.trials, collectors=project.collectors(),
      n_jobs=2, gpu_id_list=[0], logger=logger)
```

`trial_store` is required — it is where definitions are registered, where per-fold outcomes are recorded, and what decides which folds are skipped as already `'built'`.

!!! note "Redefining a Trial does not re-run it"
    A fold recorded as `'built'` is skipped silently. To force it:

    ```python
    project.trials.remove_hist(trial_name='lgb1', experimenter=e.name)
    e.reset_nodes(['lgb1'])
    ```

## Reading results

```python
e.get_status('scale')
e.show_error_nodes(trial_store=project.trials)

from IPython.display import Markdown, display
display(Markdown(e.get_node_info()))

project.trials.get_hist(experimenter=e.name)
```

Node errors live in the run's own history; Trial errors live in the `TrialStore`, which is why `show_error_nodes` takes the store to report both.

To pull data out at a fold:

```python
e.get_train_data({'X': 'scale:(*)'}, o_idx=0, i_idx=0)
e.get_valid_data({'X': 'scale:(*)'})
e.get_test_data({'y': '{target}'})
e.get_objs('scale', outer_idx=0, inner_idx=0)     # the fitted processor
```

## Capturing native output

Progress bars and Python logging appear as usual; what escapes is output written by native libraries straight to file descriptors 1 and 2. `os_log()` redirects those to a file for the duration.

```python
with e.os_log():
    e.build(n_jobs=1)
    e.exp(folds, project.trials, n_jobs=4)

e.get_worker_logs()          # {'master': ..., 0: ..., 1: ...}
```

While the capture is open and `n_jobs > 1`, each worker gets its own log file too.

## Extra training data

`aug_data` appends rows to the inner training split at the DataSource level. It is not persisted — pass it again when reopening.

```python
e = project.load_experimenter('cv5', df, aug_data=extra_df)
```

## Related

- [Project & Pipeline](project-pipeline.md)
- [Trainer & Collectors](trainer-collectors.md) — promoting a Trial and collecting what runs produce
- [State Model](../concepts/state-model.md)
