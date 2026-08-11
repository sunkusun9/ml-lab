from ._edge_dsl import referenced_nodes
from ._pipeline import ProcessorSpec


class Predictor:
    """The terminal output node a :class:`~mllabs.Trainer` trains.

    Structurally a Predictor carries the same execution definition a
    :class:`~mllabs.Trial` does (processor, method, adapter, params, edges) —
    both become a :class:`~mllabs._pipeline.ProcessorSpec` the executor
    cannot tell apart. What differs is what the definition *means*:

    - A Trial is a candidate being evaluated. Many are run per Experimenter,
      across folds, and their point is comparison — hence the project-wide
      :class:`~mllabs.TrialStore` collecting definitions and per-fold
      outcomes from every Experimenter in one place.
    - A Predictor is a choice already made. It is trained on the real splits
      to be shipped, so what matters is provenance (``src_trial`` /
      ``src_experimenter`` — which evaluated candidate justified it) rather
      than comparability, and its registry is per-Trainer.

    Keeping them separate is what lets the Trainer side own that provenance
    and its own storage without widening ``Trial``/``TrialStore``, which stay
    exactly as the Experimenter uses them.

    ``name`` is the identity: the artifact directory under the Trainer's
    predictor store and the :class:`~mllabs.PredictorStore` key. Redefining a
    name overwrites both.

    Attributes:
        name (str): Identifier, also the artifact directory name.
        processor (str): ``"module.ClassName"`` reference.
        method (str): Processor method (``'predict'``, ``'predict_proba'``, ...).
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict): Constructor params — plain data or ref specs only.
        edges (dict): ``{key: dsl_string}``.
        desc (str): Human-readable description. Display-only.
        tag (list[str]): Selection tags.
        src_trial (str): Name of the Trial this was promoted from, if any.
        src_experimenter (str): Name of the Experimenter that evaluated
            ``src_trial``, if known.
        pipeline_version (int | None): The Pipeline version this is defined
            against — copied from the Trial by :meth:`from_trial`, so a
            promoted Predictor requires the version its candidate was actually
            evaluated on. ``None`` means unstamped, which ``Trainer.train``
            resolves to the version that Trainer has adopted.
    """

    __slots__ = ('name', 'processor', 'method', 'adapter', 'params', 'edges',
                 'desc', 'tag', 'src_trial', 'src_experimenter',
                 'pipeline_version')

    def __init__(self, name, processor, edges, method='predict', adapter=None,
                 params=None, desc=None, tag=None, src_trial=None,
                 src_experimenter=None, pipeline_version=None):
        self.name = name
        self.processor = processor
        self.edges = dict(edges or {})
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.desc = desc
        self.tag = list(tag or [])
        self.src_trial = src_trial
        self.src_experimenter = src_experimenter
        self.pipeline_version = pipeline_version

    @classmethod
    def from_trial(cls, trial, name=None, experimenter=None):
        """Promote an evaluated *trial* into a Predictor.

        The execution definition is copied verbatim — the promoted Predictor
        trains exactly what was evaluated — and the Trial's name is recorded
        as ``src_trial`` even when *name* overrides the Predictor's own.

        ``pipeline_version`` is part of that verbatim copy, so the Predictor
        requires the version its candidate was evaluated against and
        ``Trainer.train`` refuses to train it under any other. Adopt that
        version in the Trainer — every published version is adoptable, and
        nothing else would be training what was actually measured. No separate
        provenance field records it, because this one already does.

        Args:
            trial (Trial): The evaluated candidate to promote.
            name (str, optional): Name for the Predictor. Defaults to the
                Trial's own name.
            experimenter (str, optional): Name of the Experimenter that
                evaluated *trial*, recorded as ``src_experimenter``.

        Returns:
            Predictor
        """
        return cls(
            name=name if name is not None else trial.name,
            processor=trial.processor,
            edges=dict(trial.edges),
            method=trial.method,
            adapter=trial.adapter,
            params=dict(trial.params),
            desc=trial.desc,
            tag=list(trial.tag),
            src_trial=trial.name,
            src_experimenter=experimenter,
            pipeline_version=trial.pipeline_version,
        )

    def get_spec(self):
        """This Predictor's :class:`~mllabs._pipeline.ProcessorSpec`.

        The same shape ``Pipeline.get_node_spec`` returns for a Pipeline node
        and ``Trial.get_spec`` for a Trial, so the executor treats all three
        identically. ``desc``/``tag`` and the ``src_*`` provenance are
        metadata and stay on the Predictor itself.
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
        """Names of the Pipeline nodes this Predictor's edges read.

        Only direct references, for the same reason as ``Trial.node_names``:
        callers intersect this against an already transitively closed set of
        stale node names.
        """
        names = set()
        for dsl_string in self.edges.values():
            for name in referenced_nodes(dsl_string):
                if name is not None:
                    names.add(name)
        return names

    def __repr__(self):
        return f"<Predictor {self.name!r} processor={self.processor!r}>"
