from itertools import product

from ._edge_dsl import referenced_nodes
from ._pipeline import (
    ProcessorSpec, _validate_processor, _validate_adapter, _validate_params,
    _combine_edges,
)

_CHAINABLE_FIELDS = frozenset(
    ('processor', 'method', 'adapter', 'params', 'edges', 'desc', 'tag')
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

    ``pipeline_version`` is part of the definition, not a note about it. A
    Trial's own fields say nothing about the preprocessing it reads, and it
    does not live in the pipeline, so the graph cannot tell it that its inputs
    changed; naming the version it was authored against is how it says which
    definition it means. ``Experimenter.exp`` refuses to run it against any
    other version, so the same name can never accumulate results from two
    different pipelines. ``Experimenter.set_pipeline`` still does not cascade
    into Trials — adopting a new version leaves their history alone, and what
    the version stamp adds is that they simply stop running there.

    ``Project.set_trial`` fills it in with the latest published version when it
    is left unset, so authoring one by hand does not mean tracking versions by
    hand.

    The Trainer side makes the opposite call, which is why it has its own
    :class:`~mllabs.Predictor` rather than reusing this class: a Trainer keeps
    no past executions to preserve, so a model trained against a since-changed node is
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
        pipeline_version (int | None): The Pipeline version this was authored
            against. ``None`` means unstamped, which ``Project.set_trial``
            resolves to the latest published version on registration.
        src_trial (str | None): Name of the Trial this one was chained from,
            if any. Reference only — no foreign key, no validation, and no
            staleness: it is never checked against what ``src_trial`` still
            names, or whether it still exists. See :meth:`chain`.
    """

    __slots__ = ('name', 'processor', 'method', 'adapter', 'params', 'edges',
                 'desc', 'tag', 'pipeline_version', 'src_trial')

    def __init__(self, name, processor, edges, method='predict', adapter=None,
                 params=None, desc=None, tag=None, pipeline_version=None,
                 src_trial=None):
        self.name = name
        self.processor = processor
        self.edges = dict(edges or {})
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.desc = desc
        self.tag = list(tag or [])
        self.pipeline_version = pipeline_version
        self.src_trial = src_trial

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

    def chain(self, name, pipeline_version=None, **overrides):
        """A new Trial derived from this one, with ``src_trial`` recorded.

        Every field not named in *overrides* is inherited verbatim, except
        ``params`` and ``edges``, which never fully replace — ``params`` is
        merged over ``self.params`` (only the given keys change) and
        ``edges`` is combined per key the same way a Pipeline node's own
        edges extend its group's: a value starting with ``'+'``/``'-'``
        continues ``self``'s (already-resolved) string for that key, anything
        else replaces it outright, and a key left out of *overrides*
        inherits unchanged.

        Overrides are read by presence, not by value — ``adapter=None`` in
        *overrides* clears the adapter, which a plain keyword default could
        not distinguish from "not given".

        Args:
            name (str): Name for the new Trial. Required: ``self`` and the
                result share a TrialStore, so leaving this to default to
                ``self.name`` would overwrite ``self``'s own row.
            pipeline_version (int, optional): ``None`` (default) leaves the
                result unstamped, same as any newly authored Trial —
                ``Project.set_trial`` fills in the latest published version.
                ``-1`` copies ``self.pipeline_version`` instead. Any other
                value is used as given.
            **overrides: Any of ``processor``/``method``/``adapter``/
                ``params``/``edges``/``desc``/``tag``.

        Returns:
            Trial
        """
        unknown = set(overrides) - _CHAINABLE_FIELDS
        if unknown:
            raise TypeError(
                f"chain() got unexpected keyword argument(s): {sorted(unknown)}"
            )

        def field(key, default):
            return overrides[key] if key in overrides else default

        return Trial(
            name=name,
            processor=field('processor', self.processor),
            edges=_combine_edges(overrides.get('edges', {}), self.edges),
            method=field('method', self.method),
            adapter=field('adapter', self.adapter),
            params={**self.params, **overrides.get('params', {})},
            desc=field('desc', self.desc),
            tag=field('tag', list(self.tag)),
            pipeline_version=(self.pipeline_version if pipeline_version == -1
                              else pipeline_version),
            src_trial=self.name,
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


class GridTrials:
    """Cartesian-product param sweep over one fixed processor.

    A generator for :meth:`~mllabs.Project.make_trials` — it enumerates what
    a batch of Trials should *contain* (processor/method/adapter/params/
    edges/tag), not what they are *named*. Naming is assign-based
    (``TrialStore.next_name``) and lives entirely on the ``Project`` side, on
    purpose: a name derived from grid position — as the old free-function
    ``make_trials`` did — renames every sibling combo when the grid grows,
    or worse, silently repoints a name at a different combo when an axis
    gains a value in the middle of the sort order. Keeping this class
    unaware of names is what makes that defect structurally impossible
    here, not just avoided by convention.

    Every combo shares processor/method/adapter/edges; ``params`` holds the
    values constant across the sweep and ``param_grid`` (``{param: [values]}``)
    is expanded as a cartesian product::

        gen = GridTrials(
            processor='lightgbm.LGBMClassifier',
            edges={'X': 'scaler:(*)', 'y': '{target}'},
            params={'random_state': 42},
            param_grid={'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        )
        project.make_trials('lgbm', gen)

    Order is deterministic (grid keys sorted, values in the order given), so
    the same instance always produces the same sequence of combos.

    Args:
        processor (str): ``"module.ClassName"`` reference.
        edges (dict): ``{key: dsl_string}`` shared by every combo.
        method (str): Processor method. Default ``'predict'``.
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict, optional): Params fixed across the sweep.
        param_grid (dict, optional): ``{param: [values]}``. Omitted yields a
            single combo from *params* alone.
        tags (list[str], optional): Tags applied to every combo.
    """

    def __init__(self, processor, edges, method='predict', adapter=None,
                 params=None, param_grid=None, tags=None):
        where = 'GridTrials'
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

        self.processor = processor
        self.edges = dict(edges)
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.param_grid = param_grid
        self.tags = list(tags or [])

    def combos(self):
        """Every combo as ``Trial`` constructor kwargs, minus ``name``.

        Returns:
            list[dict]: each with ``processor``/``edges``/``method``/
            ``adapter``/``params``/``tag``.
        """
        grid_keys = sorted(self.param_grid)
        result = []
        for values in product(*(self.param_grid[k] for k in grid_keys)):
            merged = dict(self.params)
            merged.update(zip(grid_keys, values))
            result.append(dict(
                processor=self.processor,
                edges=dict(self.edges),
                method=self.method,
                adapter=self.adapter,
                params=merged,
                tag=list(self.tags),
            ))
        return result
