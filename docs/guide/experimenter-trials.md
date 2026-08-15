# Experimenter & Trials

An `Experimenter` runs one cross-validation experiment: it builds the Pipeline's nodes fold by fold, then evaluates **Trials** — the candidate models — against those outputs.

## Creating one

```python
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

e = project.add_experimenter(
    'cv5',
    sp=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    sp_v=StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42),
    splitter_params={'y': 'target'},
    pipeline_version=p.build().version,
)
```

- `sp` — the outer splitter. Its held-out part is the *test* fold of each outer split.
- `sp_v` — the inner splitter, giving each outer fold a train/validation pair for things like early stopping. `None` means one inner fold with no validation set.
- `splitter_params` — maps a splitter keyword to a column, so `{'y': 'target'}` makes the split stratified on that column.
- `data=` defaults to the project's; `pipeline_version=` defaults to the latest published one. Any published version is adoptable; only a draft is refused.

The name is the identity: it is the directory under `exp/` and the key under which every fold outcome is filed in the `TrialStore`.

!!! warning "`add_*` adds — it never reopens"
    A taken name raises. Constructing an Experimenter recomputes the splits and writes fresh state, so doing that over an existing one restarts it rather than resuming. Reach an existing one through the registry, which is safe to re-run in a notebook cell:

    ```python
    e = project.experimenters['cv5']
    e = Experimenter.load_experimenter('exp/cv5', df)   # no Project needed
    ```

## Adopting a Pipeline

```python
e.set_pipeline(p.build())
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
from mllabs import Trial, GridTrials

trial = Trial('lgb1', 'lightgbm.LGBMClassifier',
              edges={'X': 'scale:(*) + ohe:(*)', 'y': '{target}'},
              method='predict',
              params={'n_estimators': 500, 'learning_rate': 0.05},
              tag=['baseline'])
```

`GridTrials` expands a grid — it only enumerates *content* (processor/method/adapter/params/edges/tag), not names:

```python
gen = GridTrials('lightgbm.LGBMClassifier',
                 edges={'X': 'scale:(*)', 'y': '{target}'},
                 method='predict',
                 params={'random_state': 42},
                 param_grid={'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1]})
```

## Registering

A Trial belongs to the project, so it is added there — separately from running it:

```python
project.set_trial(trial)               # 'lgb1', or None if unchanged
project.make_trials('lgb', gen)        # names + registers all four combos
```

`make_trials` mints each name from `project.trials.next_name('lgb')` — assigned, never derived from the combo's position in the grid, so growing or shrinking the sweep later never renames a sibling combo. Both return only what was **added or changed**, so the return value is the work list for the next run. A definition identical to the stored one is not a change and comes back as `None` / omitted.

!!! warning "A Trial's name is its identity"
    `TrialStore` is project-wide and keyed by name, so a name is what history, results and every run's reference to it all hang on. Give a new configuration a new name.

A name that already ran successfully somewhere is frozen — `set_trial` raises rather than redefine it. The history is keyed by name and a Trial leaves no artifact, so a redefinition would silently leave the old results describing a definition that never produced them. Change one by giving it a new name, or `project.remove_trial(name)` to give up the results with it.

### The Pipeline version is part of the definition

`set_trial` stamps the latest published version onto a Trial that carries none, and a Trial runs only where that version is adopted:

```python
project.set_trial(trial)
trial.pipeline_version                 # 2 — filled in, on the object you still hold

project.set_trial(Trial(..., pipeline_version=1))   # author against an older one
```

An `exp()` whose Experimenter adopted a different version raises rather than run. Collected results are keyed by node name and fold with no version in them, so one name meaning two definitions would overwrite its own metrics with nothing to say which won. Adopt the version the Trial names, or give the new definition a new name.

The check applies only where a job would be created. A Trial whose folds are all already built passes quietly however far the pipeline has moved on — there is nothing left to file, and handing back a finished round has to stay a no-op. One with folds still to run is refused, since that is where new results would land beside old ones.

That also puts teeth on the freeze above: after the pipeline moves on, re-registering a name that already succeeded is a *changed* definition, so it raises instead of quietly re-pointing old results at a version that did not produce them.

## Running

`exp()` takes Trial names. Each one runs on every fold of the run — folds are not something the caller spells out.

```python
e.exp(names, n_jobs=2, gpu_id_list=[0], logger=logger)
```

Folds already recorded `'built'` are dropped, so passing the same names again continues a partial run rather than repeating it.

!!! note "`set_trials()` returns what changed, not what to run"
    Its return value is an authoring diff — a Trial that was already registered and never ran comes back empty. To run a round, pass the names of that round and let `exp()` skip the folds that are done.

The store itself is not an argument: an Experimenter reached through a `Project` already holds it, and a standalone one takes it as `Experimenter(..., trial_store=...)`. An unregistered name raises `KeyError`. Every Collector registered on the run takes part unless `collectors=` narrows it by name — see [Collectors](collectors.md).

!!! note "A built fold is not re-run"
    A fold recorded as `'built'` is skipped silently, whatever the definition says now. To run it again:

    ```python
    e.remove_trial_result('lgb1')   # this run's results and its history
    ```

## Reading results

```python
e.get_status('scale')
e.error_nodes()                                   # this run's Pipeline nodes
project.error_trials(experimenter=e.name)         # its Trials
project.collect_errors(experimenter=e.name)       # its Collectors

from IPython.display import Markdown, display
display(Markdown(e.get_node_info()))

project.trials.get_hist(experimenter=e.name)
```

The split follows the history: node errors are recorded in the run's own store, Trial errors in the project's `TrialStore`, so each is reported by whoever owns them. Both return one line per failed fold, or `None` when nothing failed.

To ask what is still owed a run:

```python
project.pending_trials(experimenter=e.name)       # errored, or never run
```

Both cases need the same thing, so they come back as one list — a filter written by hand usually catches only the second and quietly drops the ones that failed. It is deliberately coarse: a Trial interrupted partway through its folds is not reported, since judging that means comparing history against a fold grid the store does not know. Running the names anyway is safe — `exp()` skips the folds that are done.

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
    e.exp(names, n_jobs=4)

e.get_worker_logs()          # {'master': ..., 0: ..., 1: ...}
```

While the capture is open and `n_jobs > 1`, each worker gets its own log file too.

## Extra training data

`aug_data` appends rows to the inner training split at the DataSource level. It is not persisted — pass it again when reopening.

```python
project.set_aug_data(extra_df)          # or Project(path, aug_data=extra_df)
e = project.experimenters['cv5']
```

## Related

- [Project & Pipeline](project-pipeline.md)
- [Collectors](collectors.md) — capturing what runs produce
- [Trainer & Predictors](trainer-predictors.md) — promoting a Trial to a trained model
- [State Model](../concepts/state-model.md)
