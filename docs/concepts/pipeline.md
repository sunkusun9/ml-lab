# Pipeline

A Pipeline is a graph of preprocessing nodes. It holds no data and runs nothing — it is the structure an `Experimenter` or `Trainer` executes.

## Builder and built form

```
PipelineBuilder  ── mutable. groups, nodes, SQLite-backed
      │
      │ .build()
      ▼
Pipeline         ── immutable snapshot. group inheritance already resolved
```

`Experimenter`, `Trainer` and `Inferencer` accept **only the built form**; passing a builder raises `TypeError`. That is the point of the split: editing the builder afterwards cannot leak into a run already in progress.

```python
p = project.pipeline
p.set_node('scale', processor='sklearn.preprocessing.StandardScaler',
           method='fit_transform', edges={'X': '{age, income}'})

pipeline = p.build()                   # published as the next version
e.set_pipeline(pipeline)
```

**Building publishes.** A builder with a database mints a version here, which makes a version appear exactly when the definition changes — build it again unchanged and you get the same number back rather than a duplicate. There is no separate publish step and no unnumbered snapshot to run against, so every history row can name the definition it ran on. A builder with no database (constructed without a path) has nowhere to mint from and stays `open` with `version = None`. See [Project & Pipeline](../guide/project-pipeline.md#building-a-version) for the version lifecycle.

## Nodes

A Pipeline contains nodes and nothing else. There is no role attribute and no separate "stage" concept: models are Trials, and they live outside the Pipeline.

A node has:

| Field | |
|---|---|
| `processor` | `"module.ClassName"` — **a string**, never a class object |
| `edges` | which upstream outputs feed which variable set — see [edges DSL](../guide/edges-dsl.md) |
| `method` | the method to call: `'fit_transform'`, `'transform'`, … |
| `adapter` | optional framework adapter, as a string or `{"__ref__": ...}` spec |
| `params` | constructor parameters — plain data, or ref specs |
| `desc` | free text, display only |

The **DataSource** is the raw input, addressed as the key `None` in the node table and as an unprefixed term in edges. It declares a schema (`{column: var_type}`) and the target columns.

## Groups

A group lets nodes share configuration. Node values override group values; group values override the parent group's.

```python
p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
          method='fit_transform')
p.set_node('scale_num', grp='scale', edges={'X': '{age, income}'})
p.set_node('scale_geo', grp='scale', edges={'X': '{lat, lon}'})
```

Groups exist only in the builder. `build()` resolves the inheritance and drops the hierarchy — the original group name survives on the built node as `label`, for display.

!!! warning "`set_grp` assigns every field you pass"
    When `exist='diff'` (the default) detects a change, the group is updated with **all** the values given, and omitted ones become `None`/`{}`. Restate the fields you want to keep:

    ```python
    # wrong — processor/method are wiped
    p.set_grp('scale', params={'with_std': False})

    # right
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='fit_transform', params={'with_std': False})
    ```

## Definitions are declarations

`processor`, `adapter` and `params` are stored exactly as given and resolved only when a processor is constructed, at execution time. Passing a live class or instance raises `TypeError` at definition time.

```python
p.set_node('lda', processor='sklearn.discriminant_analysis.LinearDiscriminantAnalysis')
```

This keeps a definition serializable and stops a heavyweight import from firing when you merely *describe* a pipeline. It also matters for matching: `Connector` compares `processor` as a plain string, so a node defined with a class object would silently never match a Connector configured with a string.

The same laziness governs `edges`, which are never expanded into column lists except against real data at run time.

## ProcessorSpec

Nodes, Trials and Predictors all resolve to one `ProcessorSpec` — exactly `name`, `processor`, `edges`, `method`, `adapter`, `params` — so the executor treats them identically.

Five of those six are the processor constructor's arguments. `edges` is not: it is input wiring the data flow resolves lazily, which is why the class is a *spec* — an unresolved declaration — rather than a set of attributes.

```python
spec = pipeline.get_node_spec('scale')
spec.processor      # 'sklearn.preprocessing.StandardScaler'
```

The DataSource is deliberately outside this: it is not an execution unit, so it exposes `pipeline.datasource` with `schema`/`targets` instead.
