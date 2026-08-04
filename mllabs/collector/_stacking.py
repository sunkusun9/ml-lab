import pickle
import re

import numpy as np

from ._base import Collector
from .._edge_dsl import parse, eval_expr


class StackingCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict, '_outer_buf': dict}

    def __init__(self, name, connector, output_var, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._outer_buf = {}  # {node: {outer_idx: [inner results]}}

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

    def _aggregate(self, data_cls, iterator):
        if self.method == 'simple':
            return data_cls.simple(iterator)
        elif self.method == 'mean':
            return data_cls.mean(iterator)
        elif self.method == 'mode':
            return data_cls.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _flush_outer(self, node, outer_idx, inner_list):
        valid_results = [r for r in inner_list if r is not None]
        if not valid_results:
            return
        self._outer_buf.setdefault(node, {})[outer_idx] = valid_results
        if self._n_outer is not None and len(self._outer_buf[node]) == self._n_outer:
            self._save_node(node)

    def _save_node(self, node):
        outer_buf = self._outer_buf.pop(node)
        folds = [outer_buf[outer_idx] for outer_idx in range(self._n_outer)]
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / f'{node}.pkl', 'wb') as f:
            pickle.dump({'folds': folds}, f)

    def has_node(self, node):
        if self.path is None:
            return False
        return (self.path / f'{node}.pkl').exists()

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
        return [f.stem for f in self.path.glob('*.pkl')]

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _load_node(self, node, data_cls, n_splits):
        with open(self.path / f'{node}.pkl', 'rb') as f:
            folds = pickle.load(f)['folds']
        if len(folds) != n_splits:
            raise ValueError(
                f"Collector '{self.name}': node '{node}' was collected over "
                f"{len(folds)} outer fold(s), but this experimenter has {n_splits}"
            )
        arrays, columns = [], None
        for inner_list in folds:
            agg = self._aggregate(data_cls, iter(inner_list))
            if columns is None:
                columns = agg.get_columns()
                if type(columns) is str:
                    columns = [columns]
            arrays.append(agg.to_array())
        return np.concatenate(arrays, axis=0), columns

    def get_dataset(self, experimenter, nodes=None, include_target=True):
        data_cls = type(experimenter.data)
        n_splits = experimenter.get_n_splits()
        index = self._build_index(experimenter)
        node_names = self._get_nodes(nodes, self._get_saved_nodes())

        arrays, columns = [], []
        for node in node_names:
            node_data, node_columns = self._load_node(node, data_cls, n_splits)
            arrays.append(node_data)
            columns.extend(node_columns)

        wrapped = data_cls.from_output(np.concatenate(arrays, axis=1), columns, index)

        target, target_columns = self._build_target(experimenter) if include_target else (None, None)
        if target is not None:
            wrapped = data_cls.concat([
                wrapped,
                data_cls.from_output(target, target_columns, index)
            ], axis=1)

        return wrapped.to_native()

    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': True,
            'need_process_data': False,
        }