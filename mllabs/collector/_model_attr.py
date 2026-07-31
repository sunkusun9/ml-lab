import pickle
import re

import pandas as pd

from ._base import Collector
from ..adapter import get_adapter


class ModelAttrCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict, '_cache': dict}

    def __init__(self, name, connector, result_key, adapter=None, params=None):
        super().__init__(name, connector)
        self.result_key = result_key
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = get_adapter(connector.processor)
            if self.result_key not in self.adapter.result_objs:
                raise RuntimeError("")
        self.params = params or {}
        self._cache = {}  # {node: {(outer_idx, inner_idx): result}}

    def collect(self, context):
        result_func = self.adapter.result_objs[self.result_key][0]
        return result_func(context['processor'], **self.params)

    def _is_mergeable(self):
        if self.adapter is None or self.result_key not in self.adapter.result_objs:
            return False
        return self.adapter.result_objs[self.result_key][1]

    def _flush_outer(self, node, outer_idx, inner_list):
        node_cache = self._cache.setdefault(node, {})
        for inner_idx, r in enumerate(inner_list):
            node_cache[(outer_idx, inner_idx)] = r
        if self.path is not None and self._n_outer is not None and len(node_cache) == self._n_outer * self._n_inner:
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / f'{node}.pkl', 'wb') as f:
                pickle.dump(node_cache, f)

    def _load_results(self, node):
        if node in self._cache:
            return self._cache[node]
        if self.path is None:
            return None
        p = self.path / f'{node}.pkl'
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            result = pickle.load(f)
        self._cache[node] = result
        return result

    def has_node(self, node):
        if node in self._cache:
            return True
        if self.path is None:
            return False
        return (self.path / f'{node}.pkl').exists()

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
        for node in nodes:
            self._cache.pop(node, None)
            if self.path is not None:
                p = self.path / f'{node}.pkl'
                if p.exists():
                    p.unlink()

    def _get_saved_nodes(self):
        if self.path is None:
            return list(self._cache.keys())
        return [f.stem for f in self.path.glob('*.pkl') if f.name != '__config.pkl']

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def get_attr(self, node, idx=None):
        data = self._load_results(node)
        if data is None:
            return None
        outer_idxs = sorted(set(k[0] for k in data.keys()))
        result = []
        for oi in outer_idxs:
            inner_list = [data[(oi, ii)] for ii in sorted(k[1] for k in data if k[0] == oi)]
            result.append(inner_list)
        if idx is not None:
            return result[idx]
        return result

    def get_attrs(self, nodes=None):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        return {node: self.get_attr(node) for node in node_names}

    def get_attrs_agg(self, node, agg_inner=True, agg_outer=True):
        if agg_outer and not agg_inner:
            raise ValueError("agg_outer requires agg_inner to be True")
        if not self._is_mergeable():
            raise ValueError(f"Result '{self.result_key}' is not mergeable across folds")
        results = self.get_attr(node)
        if results is None:
            return None
        l = []
        for no, inner_list in enumerate(results):
            l.append(
                pd.concat([j.rename(no_i) for no_i, j in enumerate(inner_list)], axis=1).stack().rename(no)
            )
        df = pd.concat(l, axis=1)
        if agg_inner:
            df = df.groupby(level=[i for i in range(len(df.index.levels) - 1)]).mean()
            if agg_outer:
                return df.mean(axis=1)
        return df
    
    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': False,
            'need_process_data': False,
        }