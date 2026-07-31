import json
import hashlib

from itertools import product

from ._serialize import serialize_value
from ._edge_dsl import referenced_nodes
from ._pipeline import _validate_processor, _validate_adapter, _validate_params


class Trial:
    """One concrete configuration to evaluate — what used to be a Head node.

    A Trial carries the same definition a Head node carried (processor, method,
    adapter, params, edges) but is handed straight to ``Experimenter.exp``
    rather than declared on a :class:`~mllabs.PipelineBuilder`.

    Identity is two-part:

    ``name``
        Human-readable, stable, and used as the on-disk artifact directory —
        the same role the Head node name played.
    ``trial_id(pipeline)``
        Content hash of the definition **plus the serials of every Stage node
        the edges reference**. Because Heads no longer live in the pipeline,
        ``_bump_serials`` can no longer cascade a Stage change into them; the
        upstream serials being part of this hash is what replaces that cascade.
        Without it, editing a Stage would silently leave dependent Trials
        looking up-to-date.

    Attributes:
        name (str): Identifier, also the artifact directory name.
        processor (str): ``"module.ClassName"`` reference.
        method (str): Processor method (``'predict'``, ``'predict_proba'``, ...).
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict): Constructor params — plain data or ref specs only.
        edges (dict): ``{key: dsl_string}``.
        label (str): Display-only grouping label (e.g. the Experiment name).
        tag (list[str]): Selection tags.
    """

    __slots__ = ('name', 'processor', 'method', 'adapter', 'params', 'edges',
                 'label', 'tag')

    def __init__(self, name, processor, edges, method='predict', adapter=None,
                 params=None, label=None, tag=None):
        self.name = name
        self.processor = processor
        self.edges = dict(edges or {})
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.label = label
        self.tag = list(tag or [])

    def get_attrs(self):
        """Resolved attributes in the same shape ``Pipeline.get_node_attrs`` returns.

        Lets Connectors, the executor, and Collectors treat a Trial exactly as
        they treat a Head node.
        """
        return {
            'name': self.name,
            'label': self.label,
            'role': 'head',
            'edges': self.edges,
            'processor': self.processor,
            'adapter': self.adapter,
            'params': self.params,
            'method': self.method,
            'tag': self.tag,
        }

    def content_key(self):
        """Deterministic JSON of everything that affects the result.

        ``name``/``label`` are excluded — renaming a Trial does not change what
        it computes. Params are plain data by construction (``_validate_params``
        rejects live objects), which is what makes a stable hash possible at all.
        """
        return json.dumps(
            serialize_value({
                'processor': self.processor,
                'method': self.method,
                'adapter': self.adapter,
                'params': self.params,
                'edges': self.edges,
            }),
            sort_keys=True, ensure_ascii=False, separators=(',', ':'),
        )

    def upstream_serials(self, pipeline):
        """``{stage_name: serial}`` for every Stage node this Trial reads.

        Only direct references are needed: a Stage's own serial is already
        bumped when anything upstream of it changes (``_bump_serials`` walks
        ``output_edges``), so the chain is covered transitively.
        """
        serials = {}
        for dsl_string in self.edges.values():
            for name in referenced_nodes(dsl_string):
                if name is None or name not in pipeline.nodes:
                    continue
                serials[name] = pipeline.nodes[name].serial
        return serials

    def trial_id(self, pipeline):
        """Content hash used the way Head nodes used ``node_serial``."""
        serials = self.upstream_serials(pipeline)
        payload = self.content_key() + '|' + json.dumps(serials, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

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
        name (str): Name prefix, and the Trial ``label``. A single trial takes
            *name* unchanged; several get ``{name}_{idx}`` zero-padded.
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
            label=name,
            tag=list(tags or []),
        ))
    return trials
