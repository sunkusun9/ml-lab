import pickle
import re
import shutil

import numpy as np

from ._base import Collector
from .._edge_dsl import parse, eval_expr
from .._data_wrapper import DataWrapper


class StackingCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict, '_outer_buf': dict}

    def __init__(self, name, connector, output_var, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._outer_buf = {}  # {node: {outer_idx: aggregated_DataWrapper}}

    def _on_attach(self, experimenter):
        self._data_cls = type(experimenter.data)
        self._index = self._build_index(experimenter)
        self._target, self._target_columns = self._build_target(experimenter)

    def _build_index(self, experimenter):
        all_valid_idx = np.concatenate([
            experimenter.outer_folds[i].test_idx
            for i in range(experimenter.get_n_splits())
        ])
        return experimenter.data.iloc(all_valid_idx).get_index()

    def _build_target(self, experimenter):
        target_vars = self.connector.edges.get('y') if self.connector.edges else None
        if target_vars is None:
            return None, None

        target_list = []
        target_columns = None
        temp_edges = {'_target': target_vars}
        for idx in range(experimenter.get_n_splits()):
            data_dict = experimenter.get_test_data(temp_edges, o_idx=idx, i_idx=0)
            target_data = data_dict['_target']
            if target_columns is None:
                target_columns = target_data.get_columns()
            target_list.append(target_data.to_array())
        if type(target_columns) is str:
            target_columns = [target_columns]
        return np.concatenate(target_list, axis=0), target_columns

    def collect(self, context):
        output_test = context['output_test']
        if self.output_var is None:
            cols = output_test.get_columns()
        else:
            cols = eval_expr(parse(self.output_var), output_test, processor=context['processor'])
        if len(cols) == 0:
            return None
        return output_test.select_columns(cols)

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return self._data_cls.simple(iterator)
        elif self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _flush_outer(self, node, outer_idx, inner_list):
        valid_results = [r for r in inner_list if r is not None]
        if not valid_results:
            return
        aggregated = self._aggregate(iter(valid_results))
        self._outer_buf.setdefault(node, {})[outer_idx] = aggregated
        if self._n_outer is not None and len(self._outer_buf[node]) == self._n_outer:
            self._save_node(node)

    def _save_node(self, node):
        outer_buf = self._outer_buf.pop(node)
        arrays, columns = [], None
        for outer_idx in range(self._n_outer):
            agg = outer_buf[outer_idx]
            if columns is None:
                columns = agg.get_columns()
                if type(columns) is str:
                    columns = [columns]
            arrays.append(agg.to_array())
        all_data = np.concatenate(arrays, axis=0)
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / f'{node}.pkl', 'wb') as f:
            pickle.dump({'data': all_data, 'columns': columns}, f)

    def has_node(self, node):
        if self.path is None:
            return False
        return (self.path / f'{node}.pkl').exists()

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
        node_set = set(nodes)
        self._outer_buf = {k: v for k, v in self._outer_buf.items() if k not in node_set}
        for node in nodes:
            p = self.path / f'{node}.pkl'
            if p.exists():
                p.unlink()

    def _get_saved_nodes(self):
        if self.path is None:
            return []
        return [f.stem for f in self.path.glob('*.pkl') if f.name != '__config.pkl']

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def get_dataset(self, nodes=None, include_target=True):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())

        arrays, columns = [], []
        for node in node_names:
            with open(self.path / f'{node}.pkl', 'rb') as f:
                saved = pickle.load(f)
            arrays.append(saved['data'])
            columns.extend(saved['columns'])

        all_data = np.concatenate(arrays, axis=1)
        wrapped = self._data_cls.from_output(all_data, columns, self._index)

        if include_target and self._target is not None:
            wrapped = self._data_cls.concat([
                wrapped,
                self._data_cls.from_output(self._target, self._target_columns, self._index)
            ], axis=1)

        return wrapped.to_native()

    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': True,
            'need_process_data': False,
        }