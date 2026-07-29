import pickle
import re
import shutil

import numpy as np
import pandas as pd
import shap

from ._base import Collector
from .._data_wrapper import unwrap


class SHAPCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict, '_cache': dict}

    def __init__(self, name, connector, explainer_cls=None, data_filter=None):
        super().__init__(name, connector)
        self.explainer_cls = explainer_cls or shap.TreeExplainer
        self.data_filter = data_filter
        self._cache = {}  # {node: {(outer_idx, inner_idx): result}}

    def collect(self, context):
        train_data, valid_data, test_data = context['input']

        model = context['processor'].obj
        explainer = self.explainer_cls(model)

        if self.data_filter is not None:
            train_data = self.data_filter(train_data)
            if valid_data is not None:
                valid_data = self.data_filter(valid_data)
        return {
            'train': explainer.shap_values(unwrap(train_data['X'])),
            'valid': explainer.shap_values(unwrap(valid_data['X'])) if valid_data is not None and 'X' in valid_data else None,
            'train_index': train_data['X'].get_index(),
            'valid_index': valid_data['X'].get_index() if valid_data is not None and 'X' in valid_data else None,
            'columns': list(context['processor'].X_),
        }

    def push(self, node, outer_idx, inner_idx, result):
        if self.path is None:
            self._cache.setdefault(node, {})[(outer_idx, inner_idx)] = result
        else:
            node_dir = self.path / node
            node_dir.mkdir(parents=True, exist_ok=True)
            with open(node_dir / f'{outer_idx}_{inner_idx}.pkl', 'wb') as f:
                pickle.dump(result, f)

    def has_node(self, node):
        if node in self._cache:
            return True
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
        for node in nodes:
            self._cache.pop(node, None)
            if self.path is not None:
                p = self.path / node
                if p.exists():
                    shutil.rmtree(p)

    def _get_saved_nodes(self):
        if self.path is None:
            return list(self._cache.keys())
        return [p.name for p in self.path.iterdir()
                if p.is_dir() and any(p.glob('*.pkl'))]

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _shap_to_importance(self, shap_vals, columns):
        if isinstance(shap_vals, list):
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            abs_vals = np.abs(shap_vals)
        if abs_vals.ndim > 2:
            abs_vals = abs_vals.mean(axis=-1)
        return pd.Series(abs_vals.mean(axis=0), index=columns)

    def _load_result(self, node, outer_idx, inner_idx):
        if node in self._cache:
            return self._cache[node][(outer_idx, inner_idx)]
        with open(self.path / node / f'{outer_idx}_{inner_idx}.pkl', 'rb') as f:
            return pickle.load(f)

    def _get_inner_idxs(self, node, outer_idx):
        if node in self._cache:
            return sorted(k[1] for k in self._cache[node] if k[0] == outer_idx)
        p = self.path / node
        return sorted(
            int(f.stem.split('_')[1])
            for f in p.glob(f'{outer_idx}_*.pkl')
        )

    def _get_outer_idxs(self, node):
        if node in self._cache:
            return sorted(set(k[0] for k in self._cache[node]))
        p = self.path / node
        return sorted(set(int(f.stem.split('_')[0]) for f in p.glob('*.pkl')))

    def get_feature_importance(self, node, idx):
        result = []
        for inner_idx in self._get_inner_idxs(node, idx):
            data = self._load_result(node, idx, inner_idx)
            result.append(self._shap_to_importance(data['train'], data['columns']).rename(inner_idx))
        return result

    def get_feature_importance_agg(self, node, agg_inner='mean', agg_outer='mean'):
        outer_frames = []
        for idx in self._get_outer_idxs(node):
            inner_list = self.get_feature_importance(node, idx)
            df = pd.concat(inner_list, axis=1)
            if agg_inner is not None:
                df = df.agg(agg_inner, axis=1).rename(idx)
            else:
                df.columns = pd.MultiIndex.from_tuples([(idx, s.name) for s in inner_list])
            outer_frames.append(df)

        if not outer_frames:
            return None
        result = pd.concat(outer_frames, axis=1)
        if agg_outer is not None and agg_inner is not None:
            result = result.agg(agg_outer, axis=1)
        return result

    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': False,
            'need_process_data': False,
        }
