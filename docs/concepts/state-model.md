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

State is per node **per fold**. `get_status(name)` reports the state across all folds.

`built` is decided by the artifact on disk — the store simply checks whether `obj.pkl` exists. `error` never appears there, only in the history table, which is where `error_nodes()` and `get_node_error()` look.

If an upstream node is in `error`, everything downstream fails without any explicit propagation logic — its inputs are simply missing.

### Transitions

| Call | |
|---|---|
| `build(nodes)` | `init → built` for Pipeline nodes |
| `exp(trial_names)` | `init → built` for Trials |
| `train(predictors)` | `init → built` for nodes, then Predictors |
| `reset_nodes(nodes)` | any → `init`, deleting the artifacts |
| `set_pipeline(newer)` | stale nodes → `init`; the Predictors reading them → `retired` |

There is no `finalized` state and no open/closed session. `build()` and `exp()` can be called at any time, and artifacts persist until you delete them with `reset_nodes()`.

## What gets re-run

Being already `built` is what makes execution skip work, and that is decided from disk and history — never by comparing definitions.

- **Nodes** — skipped when that fold's artifact exists.
- **Trials** — skipped when `TrialStore.experiment_hist` records that fold as `'built'`.
- **Predictors** — skipped when the artifact exists in the Trainer's predictor store.

The consequence is worth stating plainly: **a fold already marked `'built'` is silently skipped, whatever the definition says now.** For Trials this is why `Project.set_trial` refuses to redefine a name that has a successful execution behind it — the history would otherwise describe results a definition never produced. Running one again is explicit:

```python
e.remove_trial_result('lgb')      # this Experimenter's results and history
```

For Predictors, which have no such guard, remove the artifact with `reset_nodes`.

### A Predictor carries its own status

Nodes and Trials are read from disk and history; a Predictor is not, because one of its states leaves no trace there.

| | |
|---|---|
| `init` | registered, with at least one split still to train |
| `trained` | every split built — what `to_inferencer()` can ship |
| `retired` | ended by a version switch. Terminal |
| `error` | at least one split failed |

Retiring drops the artifacts and leaves the history rows standing, which is exactly what a reset leaves behind — so disk cannot tell the two apart, and without the status recorded `train()` would build a retired Predictor back into existence.

## Staleness — comparing two Pipelines

The one thing that *does* invalidate work automatically is adopting a new Pipeline version, and it is decided in exactly one place: `set_pipeline()`.

`Pipeline.diff_from(old)` walks from the DataSource in topological order. A node survives only if it exists in the old version under the same name, its definition matches field for field, and every node it reads also survived. Anything else is stale, as is everything downstream — plus any node that no longer exists, so its artifacts get cleaned up rather than orphaned. A DataSource schema change stales a node only if the columns that node actually pulls from the DataSource changed.

No generation counters or content hashes are involved; definitions are compared by value. A useful consequence: **editing a node that a given Trial does not read leaves that Trial's results intact.**

The diff is skipped entirely when what is being replaced is the **empty** Pipeline. That is the state of an Experimenter or Trainer that has adopted nothing yet — a placeholder, not a claim that nothing was built — and reopening one adopts its saved Pipeline over exactly that placeholder. Diffing against it would delete every artifact on disk.

### Trials and Predictors diverge here

- **Trials are left alone.** A Trial leaves no artifact, and its `experiment_hist` row documents the pipeline version it actually ran against, which stays true after a newer version is adopted. Re-running is a separate, explicit act.
- **Predictors are retired.** A Predictor reading a stale node loses the inputs that produced its model, so adopting cannot leave it standing — and cannot rebuild it either. It is retired: terminal, skipped by `train()`, and refused if named. Ask `t.retiring_predictors(candidate)` beforehand; afterwards there is nothing left to ask.

Retiring is not the same motion as `reset_nodes()`. A reset leaves the definitions untouched, so a Predictor caught by that cascade would train to exactly the model it had — it returns to `init` and the next `train()` makes it again. Same files deleted, opposite meaning.

## Related

- [Pipeline](pipeline.md) — where definitions come from
- [Data Flow](data-flow.md) — what a built artifact holds
