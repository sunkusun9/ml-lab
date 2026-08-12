# Trainer & Predictors

Where an `Experimenter` compares candidates, a `Trainer` trains the ones you chose. What it trains are **Predictors**.

## Creating one

```python
from sklearn.model_selection import KFold

t = project.add_trainer('final', pipeline_version=pipeline.version)

t = project.add_trainer('cv5',
                        splitter=KFold(n_splits=5, shuffle=True, random_state=42),
                        splitter_params={'y': 'target'},
                        pipeline_version=pipeline.version)
```

No splitter means one fold over the whole dataset — the usual choice for a final model. With a splitter you get one fold per split, and `Inferencer` will average across them at serve time.

`pipeline_version=` defaults to the latest published version. Any published version is adoptable — an older one answers "what was this trained against" as well as the newest, and adopting one is how you train a Predictor promoted from a candidate evaluated on it. What `set_pipeline` refuses is a draft, and the default never hands it one.

Like an Experimenter, a Trainer owns its directory and reopens from it:

```python
t = project.trainers['final']
t = Trainer.load_trainer('trainers/final', df)      # no Project needed
```

Reopening replays the **stored split indices** rather than recomputing them, so the folds are exactly the ones that were trained — which also lets a Trainer with no splitter reopen at all.

`aug_data` appends rows to the training split at the DataSource level. Set it on the project (`Project(path, aug_data=...)` / `set_aug_data`) so a reopened Trainer gets it too.

## Predictors

A `Predictor` has the same execution definition as a Trial. What it adds is provenance:

```python
from mllabs import Predictor

pred = Predictor.from_trial(trial, experimenter=e.name)
pred.src_trial          # 'lgb1'
pred.src_experimenter   # 'cv5'
```

`from_trial` copies the definition verbatim and keeps the Trial's name unless you override it — and even then `src_trial` records the original. Keeping the name matters more than it looks: output columns are named after the node, so a Predictor named like its Trial produces columns that line up with what the experiment collected.

Verbatim includes `pipeline_version`, so a promoted Predictor requires the version its candidate was actually evaluated on, and `train()` refuses any other — training a newer pipeline against it would ship something that was never measured. Adopt that version in the Trainer, or re-run the experiment against the newer one. A Predictor declared directly carries no version and takes the Trainer's own, since the definition in front of it is what it is being written against.

Passing a bare `Trial` to `train()` raises `TypeError`. The promotion is deliberate, so the provenance is recorded rather than guessed.

You can also declare one directly:

```python
pred = Predictor('lgb_final', 'lightgbm.LGBMClassifier',
                 edges={'X': 'scale:(*)', 'y': '{target}'},
                 method='predict_proba',
                 params={'n_estimators': 500})
```

## Training

```python
t.train([pred], n_jobs=2, gpu_id_list=[0], logger=logger)
```

`train()` runs the Pipeline nodes those Predictors read — in topological order — and then the Predictors themselves. Only what is not already built on disk runs, so calling it again after adding a Predictor trains just that one.

Registration is an **upsert**, so a second call is additive: previously trained Predictors keep their definitions and their artifacts.

```python
t.train([pred_a])
t.train([pred_b])          # pred_a stays trained and registered
t.train()                  # resume whatever is registered
```

!!! note "Redefining does not retrain"
    Like Trials, a Predictor already built on disk is skipped. Force it with `t.reset_nodes(['lgb_final'])`.

The selection is derived, never stored:

```python
t.predictors            # read from PredictorStore on each access
t.predictor_names()
t.selected_nodes        # the Pipeline nodes those Predictors read
```

## Where a Predictor stands

Each one carries a status, and `train()` writes back what it left on disk:

| | |
|---|---|
| `init` | registered, with at least one split still to train |
| `trained` | every split built — this is what `to_inferencer()` can ship |
| `retired` | ended by a version switch its inputs did not survive. Terminal |
| `error` | at least one split failed. One bad fold is a bad Predictor |

```python
t.predictor_status('lgb_final')     # 'trained'
t.predictor_statuses()              # {name: status}
```

Three of the four could be read off disk, but `retired` could not: retiring drops the artifacts and leaves the history rows standing, which is exactly what a reset leaves behind. Without the status recorded, a retired Predictor looks like one waiting to be trained, and `train()` would build it back into existence.

## Two stores, on purpose

| | |
|---|---|
| `trainers/{name}/` | Pipeline node artifacts and their history |
| `trainers/{name}/__predictors/` | Predictor artifacts, their history, and the definitions |

Separating them tells a node from a Predictor structurally rather than by which history table happened to record it. It is also forced: both are `NodeStore`s, and two in one directory would collide on the `__node_hist.db` filename.

## Checking on it

```python
t.get_status('scale')          # 'built' / 'error' / None / 'inconsistent'
t.get_node_error('lgb_final')  # {type, message, traceback, fold} or None
```

Both look in whichever of the two stores owns the name.

## Applying to new data

`process()` is a generator yielding one result per split:

```python
for split_output in t.process(test_df):
    ...

for split_output in t.process(test_df, v='1:'):   # drop the first output column
    ...
```

`v` is a [DSL string](edges-dsl.md) filtering the Predictor output columns. With several Predictors, their outputs are concatenated column-wise.

## Exporting an Inferencer

```python
inf = t.to_inferencer(v='1:')
inf.save('inferencers/v1')
```

`to_inferencer()` copies the fitted processors out, so the result depends on nothing — no Trainer, no Experimenter, no Project, not even a Pipeline. Every selected node must be `built`, or it raises.

See [Inferencer](../serving/inferencer.md) for serving.

## Resetting, and retiring

Two things drop a Predictor's artifact, and they are not the same thing.

**Resetting** is asking for a rebuild:

```python
t.reset_nodes(['scale'])
```

Downstream nodes go with it, and so does any Predictor reading one of them. The Pipeline definition is untouched, so those Predictors would train to the same model as before — they go back to `init`, and the next `train()` makes them again.

**Retiring** is what a version switch does:

```python
t.set_pipeline(project.load_pipeline(3))
```

Here the definitions really did change. A Predictor whose inputs are gone cannot be trained back to the model it held, so it is retired rather than reset: terminal, skipped by `train()`, and refused if you name it explicitly. Wanting the same specification again means registering it under another name, against the version that can serve it.

Ask before you commit to it:

```python
t.stale_nodes(candidate)            # nodes that would be reset
t.retiring_predictors(candidate)    # Predictors that would end
```

Both run the traversal `set_pipeline` acts on, so a preview cannot disagree with the effect. Afterwards there is nothing left to ask — the artifacts are gone, and the staleness went into deciding to delete them.

A Predictor that comes through the switch is **restamped** onto the adopted version: its artifact survived, which means the nodes it reads are defined identically in both versions, so it would train to the same model there. A retired one keeps naming the version it actually trained against, that being the only true thing left to say about it.

Retiring leaves the definition and the history standing, as the record of what was there. Clear them when you want to:

```python
t.remove_predictor('lgb_final')     # definition, artifacts, history
t.purge_retired()                   # every retired one; returns the names
```

## Samplers

A sampler resamples the training data before each `fit()` — for class imbalance, typically. Set it as the `mllab_sampler` param, which `_node_processor` strips before the remaining params reach the estimator.

Because node params reject live objects, give it as a reference spec:

```python
p.set_node('lgb_smote', grp='model', params={
    'n_estimators': 300,
    'mllab_sampler': {
        '__ref__': 'mllabs.sampler.ImbLearnSampler',
        '__params__': {
            'sampler': {'__ref__': 'imblearn.over_sampling.SMOTE',
                        '__params__': {'random_state': 42}},
        },
    },
})
```

Nested `__ref__` specs resolve recursively, and none of it is instantiated until the processor is constructed.

A custom sampler implements one method:

```python
from mllabs.sampler import Sampler

class MySampler(Sampler):
    def sample(self, fit_params):
        X, y = fit_params['X'], fit_params['y']
        # ... resample ...
        return {**fit_params, 'X': X_resampled, 'y': y_resampled}
```

It must be importable by its dotted path, since that is how the ref spec names it.

## Related

- [Experimenter & Trials](experimenter-trials.md) — where Predictors come from
- [Collectors](collectors.md)
- [Inferencer](../serving/inferencer.md)
