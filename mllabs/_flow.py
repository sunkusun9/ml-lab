import uuid
from abc import ABC, abstractmethod

from ._edge_dsl import iter_segments, eval_expr, referenced_nodes


class DataSourceProvider(ABC):
    @abstractmethod
    def get_train(self):
        """Returns train_data as DataWrapper."""

    @abstractmethod
    def get_valid(self):
        """Returns valid_data (train-time monitoring, e.g. early stopping) as DataWrapper or None."""

class DataFlow:
    """Single-fold data transformation through stage nodes.

    Wraps a :class:`~mllabs._store.NodeStore` (composition, not inheritance)
    plus this fold's own ``(outer_idx, inner_idx)`` — ``NodeStore`` is shared
    across every fold of a run (constructed once, at that run's base path),
    so both are needed on every call into it. Transforms source data through
    the stage graph given edges. No build functionality.

    Nothing is read from the store until something asks for it. Constructing
    a flow is free, which matters because a run constructs one per fold up
    front: opening a built run used to materialise every fold's processors
    before the caller had asked for anything.

    ``node_objs`` is a cache of ``obj.pkl``, not a place anything lives. The
    store is the only source, so dropping an entry is always safe and there is
    nothing to publish into: a build writes its artifact and whoever reads it
    next takes it off disk.

    Args:
        store (NodeStore): This run's artifact+history store (shared across
            every fold of the run — see :class:`TrainDataFlow`).
        outer_idx, inner_idx: This fold's coordinates within ``store``.
    """

    def __init__(self, store, outer_idx=0, inner_idx=0):
        self.store = store
        self.outer_idx = outer_idx
        self.inner_idx = inner_idx
        self.node_objs = {}    # {name: obj} — cache of this fold's obj.pkl
        self._node_edges = {}  # {name: edges dict}
        self._fold_info = None

    def _fold_edges(self, node_name):
        """The ``edges`` recorded for *node_name* in this fold, or ``None``.

        Neither ``obj.pkl`` nor ``result.pkl`` carries edges, so they come from
        the store's history. One query covers the whole fold and touches no
        artifact, so unlike the pickles it is cheap enough to keep.

        A miss re-queries rather than answering ``None`` from a stale read: a
        flow constructed before a node was built holds a fold snapshot without
        it, and that flow is the one the build then hands work to.
        """
        if self._fold_info is None or node_name not in self._fold_info:
            self._fold_info = self.store.get_fold_info(self.outer_idx, self.inner_idx)
        info = self._fold_info.get(node_name)
        return info.get('edges') if info else None

    def load_obj(self, node_name, edges=None):
        """Read *node_name*'s processor out of the store and cache it here."""
        obj = self.store.get_obj(node_name, self.outer_idx, self.inner_idx)
        self.node_objs[node_name] = obj
        if edges is None:
            edges = self._fold_edges(node_name)
        if edges is not None:
            self._node_edges[node_name] = edges
        return obj

    def _processor(self, node_name):
        """This fold's fitted processor for *node_name*, read on first use.

        Only Pipeline nodes are ever here to read. A Trial persists nothing at
        all, and a Trainer's Predictors go to a store of their own — so neither
        leaves an artifact in the store this flow reads, and no fitted model
        comes into memory from either. A caller holding one from elsewhere
        (:meth:`Trainer.process`) puts it in ``node_objs`` itself, which is
        consulted before the store.

        Raises:
            KeyError: If the node has no artifact in this fold, or one with no
                history row to recover its edges from. Loudly, because the
                alternative is a segment that contributes no columns and says
                nothing about it.
        """
        if node_name in self.node_objs:
            return self.node_objs[node_name]
        if self.store.status(node_name, self.outer_idx, self.inner_idx) != 'built':
            raise KeyError(
                f"node '{node_name}' is not built in fold "
                f"({self.outer_idx}, {self.inner_idx})"
            )
        obj = self.load_obj(node_name)
        if node_name not in self._node_edges:
            raise KeyError(
                f"node '{node_name}' has an artifact in fold "
                f"({self.outer_idx}, {self.inner_idx}) but no recorded edges"
            )
        return obj

    def get_data(self, source_data, edges):
        """Transform source_data through stage nodes per edges.

        Args:
            source_data: DataWrapper — raw input at DataSource level
            edges: {key: dsl_string}

        Returns:
            {key: data} flat dict
        """
        result = {}
        for key, dsl_string in edges.items():
            parts = []
            for node_name, expr in iter_segments(dsl_string):
                data = self._resolve(source_data, node_name)
                if data is None:
                    continue
                cols = eval_expr(expr, data, processor=self.node_objs.get(node_name))
                data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def get_missing_nodes(self, edges):
        """Node names *edges* reads that are not built yet.

        Args:
            edges (dict): ``{key: dsl_string}``.

        Returns:
            list[str]: Referenced stage node names (the DataSource segment,
            ``None``, is always available and excluded) whose disk status is
            not ``'built'`` — empty if everything *edges* needs is ready.
        """
        names = {n for dsl_string in edges.values() for n in referenced_nodes(dsl_string)}
        names.discard(None)
        return sorted(n for n in names if self.status(n) != 'built')

    def _resolve(self, source_data, node_name):
        if source_data is None:
            return None
        if node_name is None:
            return source_data
        obj = self._processor(node_name)
        return obj.process(self.get_data(source_data, self._node_edges[node_name]))

    # -------------------------------------------------------------------------
    # NodeStore delegation — thin, so callers keep treating a flow like a
    # store without DataFlow having to inherit one. This fold's own
    # (outer_idx, inner_idx) is filled in on every call.
    # -------------------------------------------------------------------------

    def status(self, name):
        return self.store.status(name, self.outer_idx, self.inner_idx)

    def get_obj(self, name):
        return self.store.get_obj(name, self.outer_idx, self.inner_idx)

    def get_objs(self, name):
        return self.store.get_objs(name, self.outer_idx, self.inner_idx)

    def get_result(self, name):
        return self.store.get_result(name, self.outer_idx, self.inner_idx)

    def list_nodes(self):
        return self.store.list_nodes(self.outer_idx, self.inner_idx)

    def node_path(self, name):
        return self.store.node_path(name, self.outer_idx, self.inner_idx)

    def reset_node(self, name):
        self.store.reset_node(name, self.outer_idx, self.inner_idx)
        self.node_objs.pop(name, None)
        self._node_edges.pop(name, None)
        if self._fold_info is not None:
            self._fold_info.pop(name, None)


class TrainDataFlow(DataFlow):
    """Single (outer, inner) fold data flow with stage build capability.

    Args:
        store (NodeStore): This run's artifact+history store — the *same*
            instance across every fold of a run (an Experimenter's OuterFolds
            or a Trainer's TrainFolds each construct it once and share it).
            Since it's per-run, an Experimenter's and a Trainer's stores are
            always at different base paths — no coordinate-faking needed to
            keep their artifacts/history apart (contrast the DataCache note
            below, which is a separate, shared-project-wide concern).
        data_source: DataSourceProvider providing train/valid/test raw data
        cache: DataCache shared instance (optional) — keyed by
            ``(self.scope, node, typ)``. ``self.scope`` is a random id this
            instance generates for itself in its own constructor (2026-08-01)
            — since exactly one TrainDataFlow exists per (run, fold), that id
            alone already uniquely identifies this fold; no need to fold
            ``outer_idx``/``inner_idx`` or a store path string into the key.
        outer_idx, inner_idx: This fold's coordinates — the NodeStore key
            (artifact path, history row). Not part of the DataCache key
            anymore (see ``scope`` above).
    """

    def __init__(self, store, data_source, cache=None, outer_idx=0, inner_idx=0):
        self.data_source = data_source
        self.cache = cache
        self.scope = uuid.uuid4().hex
        super().__init__(store, outer_idx=outer_idx, inner_idx=inner_idx)

    def get_train(self, edges):
        """{key: data} train output resolved via edges."""
        return self._get_data_typ(edges, 'train')

    def get_valid(self, edges):
        """{key: data} valid (train-time monitoring) output resolved via edges."""
        return self._get_data_typ(edges, 'valid')

    def get_test(self, edges):
        """{key: data} held-out test output resolved via edges, or ``{}``.

        Kept here rather than on the fold so a TrainDataFlow is self-sufficient:
        a dispatched job then needs nothing but the flow to produce every input
        its Trial reads.
        """
        test_source = self.data_source.get_test()
        if test_source is None:
            return {}
        return self.get_data(test_source, edges)
    
    def _get_data_typ(self, edges, typ):
        result = {}
        for key, dsl_string in edges.items():
            parts = []
            for node_name, expr in iter_segments(dsl_string):
                # Ahead of the resolve: a cache hit returns the output without
                # touching the store, but the column selectors still need the
                # processor, and an unbuilt node still has to fail here.
                obj = None if node_name is None else self._processor(node_name)
                data = self._resolve_typ(node_name, typ)
                if data is None:
                    continue
                cols = eval_expr(expr, data, processor=obj)
                data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result
    
    def _resolve_typ(self, node_name, typ):
        """This fold's *typ* output for *node_name*, via the cache.

        All three outputs go through the one cache now, so all three are under
        its budget. They are recovered differently, though, and the difference
        is not incidental:

        - valid/test are *recomputed* — ``obj.process`` on that split.
        - train is *re-read* from ``result.pkl`` whenever one was stored. It
          has to be. That output is what ``fit_process`` returned, and for a
          cross-fitting node it is not what ``process`` returns for the same
          rows: ``CrossFitTransformer`` yields out-of-fold predictions while
          fitting and full-model predictions afterwards. Recomputing would
          hand downstream nodes different numbers without saying so.

        A node whose method is plain ``fit`` produced no fit-time output, so
        ``result.pkl`` holds ``None`` and ``process`` *is* the definition of
        its train output — the one case where recomputing is the right answer
        rather than a substitute for a lost one.
        """
        if node_name is None:
            if typ == 'train':
                return self.data_source.get_train()
            else:
                return self.data_source.get_valid()
        if self.cache is not None:
            cached = self.cache.get_data(self.scope, node_name, typ)
            if cached is not None:
                return cached
        if typ == 'train':
            data_out = self.store.get_result(node_name, self.outer_idx, self.inner_idx)
            if data_out is None:
                data_out = super()._resolve(self.data_source.get_train(), node_name)
        else:
            data_out = super()._resolve(self.data_source.get_valid(), node_name)
        if self.cache is not None:
            self.cache.put_data(self.scope, node_name, typ, data_out)
        return data_out


class InferenceDataFlow:
    """In-memory graph traversal for Inferencer. No disk or cache dependency.

    Holds one processor per node (single split). Only 'X' edges are resolved —
    'y' / 'sample_weight' edges are training-only and ignored at inference time.
    """

    def __init__(self):
        self.node_objs = {}    # {name: obj}
        self._node_edges = {}  # {name: X-only edges}

    def add_node(self, name, obj, edges):
        self.node_objs[name] = obj
        self._node_edges[name] = {k: v for k, v in edges.items() if k == 'X'}

    def get_data(self, source_data, edges):
        """Resolve edges against source_data through the stage graph.

        Args:
            source_data: DataWrapper — raw input at DataSource level.
            edges: {key: dsl_string} — X-only subset.

        Returns:
            {key: data} flat dict.
        """
        result = {}
        for key, dsl_string in edges.items():
            parts = []
            for node_name, expr in iter_segments(dsl_string):
                data = self._resolve(source_data, node_name)
                if data is None:
                    continue
                obj = self.node_objs.get(node_name)
                cols = eval_expr(expr, data, processor=obj)
                data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def _resolve(self, source_data, node_name):
        if source_data is None:
            return None
        if node_name is None:
            return source_data
        if node_name not in self.node_objs:
            return None
        obj = self.node_objs[node_name]
        return obj.process(self.get_data(source_data, self._node_edges[node_name]))