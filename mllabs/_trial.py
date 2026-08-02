from itertools import product

from ._edge_dsl import referenced_nodes
from ._pipeline import (
    ProcessorSpec, _validate_processor, _validate_adapter, _validate_params,
)


class Trial:
    """One concrete configuration to evaluate — what used to be a Head node.

    A Trial carries the same definition a Head node carried (processor, method,
    adapter, params, edges) but is handed straight to ``Experimenter.exp``
    rather than declared on a :class:`~mllabs.PipelineBuilder`.

    ``name`` is the identity: human-readable, stable, and used as the on-disk
    artifact directory and the :class:`~mllabs.TrialStore` key — the same role
    the Head node name played. Redefining a name overwrites both the artifact
    and its ``TrialStore`` row.

    A Trial's own definition says nothing about the preprocessing it read, and
    it does not live in the pipeline, so a node change cannot cascade into it
    through the Pipeline graph automatically. ``Experimenter.set_pipeline``
    deliberately does not cascade either — a Trial's artifact and its
    ``TrialStore.experiment_hist`` row document the pipeline version it
    actually ran against, which stays valid even after a newer version is
    adopted. Rerunning it is a separate, explicit action.

    The Trainer side makes the opposite call, which is why it has its own
    :class:`~mllabs.Predictor` rather than reusing this class: a Trainer keeps
    no historical runs, so a model trained against a since-changed node is
    just stale and ``Trainer.reset_nodes`` cascades into it.

    Attributes:
        name (str): Identifier, also the artifact directory name.
        processor (str): ``"module.ClassName"`` reference.
        method (str): Processor method (``'predict'``, ``'predict_proba'``, ...).
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict): Constructor params — plain data or ref specs only.
        edges (dict): ``{key: dsl_string}``.
        desc (str): Human-readable description. Display-only, like
            ``PipelineBuilder``'s ``desc`` — never affects matching, diffing,
            or storage identity.
        tag (list[str]): Selection tags.
    """

    __slots__ = ('name', 'processor', 'method', 'adapter', 'params', 'edges',
                 'desc', 'tag')

    def __init__(self, name, processor, edges, method='predict', adapter=None,
                 params=None, desc=None, tag=None):
        self.name = name
        self.processor = processor
        self.edges = dict(edges or {})
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.desc = desc
        self.tag = list(tag or [])

    def get_spec(self):
        """This Trial's :class:`~mllabs._pipeline.ProcessorSpec`.

        The same shape ``Pipeline.get_node_spec`` returns for a Stage, so
        Connectors, the executor, and Collectors treat the two identically.
        ``desc``/``tag`` are display/selection metadata and stay on the Trial
        itself rather than riding along in the spec.
        """
        return ProcessorSpec(
            name=self.name,
            processor=self.processor,
            edges=self.edges,
            method=self.method,
            adapter=self.adapter,
            params=self.params,
        )

    def node_names(self):
        """Names of the Pipeline nodes this Trial's edges read.

        Only direct references are needed: callers intersect this against a
        set of reset/stale node names that is already transitively closed
        (``Pipeline.diff_from`` cascades through ``output_edges`` before
        checking Trials), so the chain is covered without this method walking
        it itself.
        """
        names = set()
        for dsl_string in self.edges.values():
            for name in referenced_nodes(dsl_string):
                if name is not None:
                    names.add(name)
        return names

    def __repr__(self):
        return f"<Trial {self.name!r} processor={self.processor!r}>"


def make_trials(name, processor, edges, method='predict', adapter=None,
                params=None, param_grid=None, tags=None):
    """Build a list of Trials sweeping *param_grid* over one fixed processor.

    Every Trial shares processor/method/adapter/edges; ``params`` holds the
    values constant across the sweep and ``param_grid`` (``{param: [values]}``)
    is expanded as a cartesian product::

        trials = make_trials(
            'lgbm',
            processor='lightgbm.LGBMClassifier',
            edges={'X': 'scaler:(*)', 'y': '{target}'},
            params={'random_state': 42},
            param_grid={'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        )                                    # lgbm_0 .. lgbm_3

    Order is deterministic (grid keys sorted, values in the order given), so the
    same call always produces the same trial names.

    Args:
        name (str): Name prefix. A single trial takes *name* unchanged;
            several get ``{name}_{idx}`` zero-padded.
        processor (str): ``"module.ClassName"`` reference.
        edges (dict): ``{key: dsl_string}`` shared by every trial.
        method (str): Processor method. Default ``'predict'``.
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict, optional): Params fixed across the sweep.
        param_grid (dict, optional): ``{param: [values]}``. Omitted yields a
            single Trial from *params* alone.
        tags (list[str], optional): Tags applied to every Trial.

    Returns:
        list[Trial]
    """
    where = f"make_trials({name!r})"
    _validate_processor(processor, where)
    _validate_adapter(adapter, where)
    _validate_params(params, where)
    _validate_params(param_grid, where)

    if not isinstance(edges, dict) or not edges:
        raise ValueError(f"{where}: edges must be a non-empty {{key: dsl_string}} dict")
    for key, dsl_string in edges.items():
        if not isinstance(dsl_string, str):
            raise TypeError(
                f"{where}: edges[{key!r}] must be a DSL string, got "
                f"{type(dsl_string).__name__}"
            )

    param_grid = dict(param_grid or {})
    for key, values in param_grid.items():
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"{where}: param_grid[{key!r}] must be a list of values, got "
                f"{type(values).__name__}"
            )
        if not values:
            raise ValueError(f"{where}: param_grid[{key!r}] is empty")

    grid_keys = sorted(param_grid)
    combos = [
        dict(zip(grid_keys, values))
        for values in product(*(param_grid[k] for k in grid_keys))
    ]

    width = len(str(len(combos) - 1)) if len(combos) > 1 else 0
    trials = []
    for idx, combo in enumerate(combos):
        merged = dict(params or {})
        merged.update(combo)
        trials.append(Trial(
            name=name if len(combos) == 1 else f"{name}_{idx:0{width}d}",
            processor=processor,
            edges=dict(edges),
            method=method,
            adapter=adapter,
            params=merged,
            tag=list(tags or []),
        ))
    return trials
