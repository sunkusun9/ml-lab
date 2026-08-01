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

    Args:
        store (NodeStore): This run's artifact+history store (shared across
            every fold of the run — see :class:`TrainDataFlow`).
        outer_idx, inner_idx: This fold's coordinates within ``store``.
    """

    def __init__(self, store, outer_idx=0, inner_idx=0):
        self.store = store
        self.outer_idx = outer_idx
        self.inner_idx = inner_idx
        self.node_objs = {}    # {name: (obj, result)}
        self._node_edges = {}  # {name: edges dict}
        self.load()

    def load_objs(self, node_name, edges=None):
        obj, result = self.store.get_objs(node_name, self.outer_idx, self.inner_idx)
        self.node_objs[node_name] = (obj, result)
        if edges is not None:
            self._node_edges[node_name] = edges
        return obj, result

    def load(self):
        """Load Stage processors, recovering each one's ``edges`` via the
        store's history (``node_hist``) — the artifact itself
        (obj.pkl/result.pkl) carries neither anymore.

        A node with no matching history row is left unloaded rather than
        guessed at either way. That also covers Trials without needing to
        ask what kind of node this is: a Trial's outcome is only ever
        recorded in ``TrialStore.experiment_hist``, never in this run's
        ``node_hist``, so it always falls into the no-row branch and its
        (potentially large) fitted model is never pulled into memory here.
        """
        fold_info = self.store.get_fold_info(self.outer_idx, self.inner_idx)
        for name in self.store.list_nodes(self.outer_idx, self.inner_idx):
            if self.store.status(name, self.outer_idx, self.inner_idx) != 'built':
                continue
            info = fold_info.get(name)
            if info is None:
                continue
            self.load_objs(name, edges=info.get('edges'))

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
                obj = self.node_objs[node_name][0] if node_name in self.node_objs else None
                cols = eval_expr(expr, data, processor=obj)
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
        if node_name not in self.node_objs or node_name not in self._node_edges:
            return None
        obj, result = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        return obj.process(self.get_data(source_data, edges))

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

    def set_objs(self, node_name, obj, result, info):
        self.node_objs[node_name] = (obj, result)
        if info.get('edges') is not None:
            self._node_edges[node_name] = info['edges']

    def get_available_stages(self, pipeline):
        """Returns stage node names that this DataFlow can produce output for."""
        return [n for n in pipeline.topo_order() if n in self.node_objs]

    def get_missing_stages(self, pipeline):
        """Returns stage node names that are in the pipeline but not yet built in this DataFlow."""
        return [n for n in pipeline.topo_order() if n not in self.node_objs]

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
                data = self._resolve_typ(node_name, typ)
                if data is None:
                    continue
                obj = self.node_objs[node_name][0] if node_name in self.node_objs else None
                cols = eval_expr(expr, data, processor=obj)
                data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result
    
    def _resolve_typ(self, node_name, typ):
        """Returns data for node_name via cache check → process. Used for valid and test."""
        if node_name is None:
            if typ == 'train':
                return self.data_source.get_train()
            else:
                return self.data_source.get_valid()
        if typ == 'train':
            obj, result = self.node_objs[node_name]
            if result is not None:
                return result
        if self.cache is not None:
            cached = self.cache.get_data(self.scope, node_name, typ)
            if cached is not None:
                return cached
        data_out = super()._resolve(
            self.data_source.get_train() if typ == 'train' else self.data_source.get_valid(), node_name
        )
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