import pickle
import re
import shutil

import numpy as np

from ._base import Collector
from .._edge_dsl import parse, eval_expr
from .._data_wrapper import wrap

class ProcessCollector(Collector):
    _SAVE_EXCLUDE = {
        '_buf': dict,
        '_input_cache': dict,
    }

    def __init__(self, name, connector, ext_data, output_var=None, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._ext_data = wrap(ext_data)
        self._data_cls = type(self._ext_data)
        self._input_cache = {}  # {(node, outer_idx): [inner0_input, ...]} transient

    def __getstate__(self):
        exclude = list(self._SAVE_EXCLUDE.keys()) + ['_ext_data']
        return {k: v for k, v in self.__dict__.items() if k not in exclude}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr, factory in self._SAVE_EXCLUDE.items():
            setattr(self, attr, factory())
    
    def collect(self, context):
        output = context['output_ext']
        if output is None:
            return None
        if self.output_var is None:
            cols = output.get_columns()
        else:
            cols = eval_expr(parse(self.output_var), output, processor=context['processor'])
        if not cols:
            return None
        return output.select_columns(cols)

    def _aggregate(self, iterator):
        if self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        elif self.method == 'simple':
            return self._data_cls.simple(iterator)
        raise ValueError(f"Unsupported method: {self.method}")

    def _flush_outer(self, node, outer_idx, inner_list):
        self._input_cache.pop((node, outer_idx), None)
        valid_results = [r for r in inner_list if r is not None]
        if not valid_results:
            return
        aggregated = valid_results[0] if len(valid_results) == 1 else self._aggregate(iter(valid_results))
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f'{outer_idx}.pkl', 'wb') as f:
            pickle.dump({'data': aggregated.to_array(), 'columns': aggregated.get_columns()}, f)

    def has_node(self, node):
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
        node_set = set(nodes)
        self._input_cache = {k: v for k, v in self._input_cache.items() if k[0] not in node_set}
        for node in nodes:
            p = self.path / node
            if p.exists():
                shutil.rmtree(p)

    def _get_saved_nodes(self):
        if self.path is None:
            return []
        return [p.name for p in self.path.iterdir()
                if p.is_dir() and any(p.glob('*.pkl'))]

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _load_node(self, node, agg):
        p = self.path / node
        outer_files = sorted(p.glob('*.pkl'), key=lambda x: int(x.stem))
        arrays, cols = [], None
        for f in outer_files:
            with open(f, 'rb') as fp:
                saved = pickle.load(fp)
            if cols is None:
                cols = saved['columns']
            arrays.append(saved['data'])
        if len(arrays) == 1:
            return self._data_cls.from_output(arrays[0], cols), cols
        wrapped = [self._data_cls.from_output(a, cols) for a in arrays]
        if agg == 'mean':
            return self._data_cls.mean(iter(wrapped)), cols
        elif agg == 'mode':
            return self._data_cls.mode(iter(wrapped)), cols
        elif agg == 'simple':
            return self._data_cls.simple(iter(wrapped)), cols
        raise ValueError(f"Unsupported agg: {agg}")

    def get_output(self, nodes=None, agg='mean'):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        arrays, columns = [], []
        for node in node_names:
            result, cols = self._load_node(node, agg)
            arrays.append(result.to_array())
            columns.extend(cols if cols is not None else [])
        if not arrays:
            return None
        all_data = np.concatenate(arrays, axis=1)
        return self._data_cls.from_output(all_data, columns).to_native()

    def get_ext_data(self):
        return self._ext_data
    
    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': False,
            'need_process_data': True,
        }