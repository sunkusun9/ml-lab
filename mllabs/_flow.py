import os
from abc import ABC, abstractmethod
from pathlib import Path
import shutil

from ._edge_dsl import iter_segments, eval_expr
from ._store import NodeStore


class DataSourceProvider(ABC):
    @abstractmethod
    def get_train(self):
        """Returns train_data as DataWrapper."""

    @abstractmethod
    def get_valid(self):
        """Returns valid_data (train-time monitoring, e.g. early stopping) as DataWrapper or None."""

class DataFlow(NodeStore):
    """Single-fold data transformation through stage nodes.

    Loads one processor per stage node from disk at ``path``.
    Transforms source data through the stage graph given edges.
    No build functionality.
    """

    def __init__(self, path):
        super().__init__(path)
        self.node_objs = {}    # {name: (obj, result, info)}
        self._node_edges = {}  # {name: edges dict}
        self.load()

    def load_objs(self, node_name):
        obj, result, info = self.get_objs(node_name)
        self.node_objs[node_name] = (obj, result, info)
        self._node_edges[node_name] = info['edges']
        return obj, result, info

    def load(self):
        if not self.path.is_dir():
            return
        for node_dir in sorted(self.path.iterdir()):
            if not node_dir.is_dir():
                continue
            if (node_dir / 'obj.pkl').exists():
                self.load_objs(node_dir.name)

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

    def _resolve(self, source_data, node_name):
        if source_data is None:
            return None
        if node_name is None:
            return source_data
        if node_name not in self.node_objs or node_name not in self._node_edges:
            return None
        obj, result, info = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        
        return obj.process(self.get_data(source_data, edges))


class TrainDataFlow(DataFlow):
    """Single (outer, inner) fold data flow with stage build capability.

    Args:
        path: Per-fold storage directory
        data_source: DataSourceProvider providing train/valid/test raw data
        cache: DataCache shared instance (optional)
        cache_key: Key for cache lookups, e.g. (outer_idx, inner_idx)
    """

    def __init__(self, path, data_source, cache=None, outer_idx=0, inner_idx=0):
        self.data_source = data_source
        self.cache = cache
        self.outer_idx = outer_idx
        self.inner_idx = inner_idx
        super().__init__(path)

    def set_objs(self, node_name, obj, result, info):
        self.node_objs[node_name] = (obj, result, info)
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
            obj, result, info = self.node_objs[node_name]
            if result is not None:
                return result
        if self.cache is not None:
            cached = self.cache.get_data(node_name, self.outer_idx, self.inner_idx, typ)
            if cached is not None:
                return cached
        data_out = super()._resolve(
            self.data_source.get_train() if typ == 'train' else self.data_source.get_valid(), node_name
        )
        if self.cache is not None:
            self.cache.put_data(node_name, self.outer_idx, self.inner_idx, typ, data_out)
        return data_out

    def reset_node(self, name):
        super().reset_node(name)
        if name in self.node_objs:
            del self.node_objs[name]


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