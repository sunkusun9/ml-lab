import json
import hashlib

from .._serialize import serialize_value
from .._edge_dsl import referenced_nodes


class Trial:
    """One concrete configuration to evaluate — what used to be a Head node.

    A Trial carries the same definition a Head node carried (processor, method,
    adapter, params, edges) but is produced by an :class:`BaseExperiment` rather
    than declared on a :class:`~mllabs.PipelineBuilder`.

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
