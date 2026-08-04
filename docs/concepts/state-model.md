# State Model

## Node states

```
init ──► built
  ▲
  │
error ──► (reset_nodes) ──► init
```

| State | Disk | Description |
|-------|------|-------------|
| **init** | — | Defined, not yet executed |
| **built** | ✓ | Execution complete; `obj.pkl` and `result.pkl` exist and results can be read |
| **error** | history only | Execution raised; the traceback is recorded, no artifact is written |

State is per node **per fold**. `get_status(name)` reports the state across all folds of a run.

`built` is decided by the artifact on disk — the store simply checks whether `obj.pkl` exists. `error` never appears there, only in the run's history table, which is where `show_error_nodes()` and `get_node_error()` look.

If an upstream node is in `error`, everything downstream fails without any explicit propagation logic — its inputs are simply missing.

### Transitions

| Call | |
|---|---|
| `build(nodes)` | `init → built` for Pipeline nodes |
| `exp(trials, trial_store)` | `init → built` for Trials |
| `train(predictors)` | `init → built` for nodes, then Predictors |
| `reset_nodes(nodes)` | any → `init`, deleting the artifacts |

There is no `finalized` state and no open/closed session. `build()` and `exp()` can be called at any time, and artifacts persist until you delete them with `reset_nodes()`.

## What gets re-run

Being already `built` is what makes a run skip work, and that is decided from disk and history — never by comparing definitions.

- **Nodes** — skipped when that fold's artifact exists.
- **Trials** — skipped when `TrialStore.experiment_hist` records that fold as `'built'`.
- **Predictors** — skipped when the artifact exists in the Trainer's predictor store.

The consequence is worth stating plainly: **redefining a Trial or Predictor does not re-run it.** A fold already marked `'built'` is silently skipped. To force it, remove the record and the artifact:

```python
project.trials.remove_hist(trial_name='lgb', experimenter=e.name)
e.reset_nodes(['lgb'])
```

## Staleness — comparing two Pipelines

The one thing that *does* invalidate work automatically is adopting a new Pipeline version, and it is decided in exactly one place: `set_pipeline()`.

`Pipeline.diff_from(old)` walks from the DataSource in topological order. A node survives only if it exists in the old version under the same name, its definition matches field for field, and every node it reads also survived. Anything else is stale, as is everything downstream — plus any node that no longer exists, so its artifacts get cleaned up rather than orphaned. A DataSource schema change stales a node only if the columns that node actually pulls from the DataSource changed.

No generation counters or content hashes are involved; definitions are compared by value. A useful consequence: **editing a node that a given Trial does not read leaves that Trial's results intact.**

### Trials and Predictors diverge here

- **Trials are left alone.** A Trial's artifact and its `experiment_hist` row document the pipeline version it actually ran against, which stays true after a newer version is adopted. Re-running is a separate, explicit act.
- **Predictors cascade.** A Trainer has no notion of a historical run to preserve, so a Predictor reading a reset node is simply stale, and `reset_nodes()` clears it too.

## Related

- [Pipeline](pipeline.md) — where definitions come from
- [Data Flow](data-flow.md) — what a built artifact holds
