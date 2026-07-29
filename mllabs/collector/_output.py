import pickle
import re
import shutil

from ._base import Collector
from .._edge_dsl import parse, eval_expr


class OutputCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict}

    def __init__(self, name, connector, output_var, include_target=True, include_train=False):
        super().__init__(name, connector)
        self.output_var = output_var
        self.include_target = include_target
        self.include_train = include_train

    def collect(self, context):
        if self.output_var is None:
            cols = context['output_test'].get_columns()
        else:
            cols = eval_expr(parse(self.output_var), context['output_test'], processor=context['processor'])
        if len(cols) == 0:
            return None

        output_test = context['output_test'].select_columns(cols)
        if self.include_train:
            output_train = context['output_train']
            if output_train is not None:
                output_train = output_train.select_columns(cols)
            output_valid = context['output_valid']
            if output_valid is not None:
                output_valid = output_valid.select_columns(cols)
        else:
            output_train = None
            output_valid = None        

        return {
            'output_test': output_test.to_array(),
            'output_train': output_train.to_array() if output_train is not None else None,
            'output_valid': output_valid.to_array() if output_valid is not None else None,
            'columns': output_test.get_columns(),
        }

    def push(self, node, outer_idx, inner_idx, result):
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f'{outer_idx}_{inner_idx}.pkl', 'wb') as f:
            pickle.dump(result, f)

    def has_node(self, node):
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
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

    def get_output(self, node, outer_idx, inner_idx):
        p = self.path / node / f'{outer_idx}_{inner_idx}.pkl'
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)

    def get_outputs(self, node):
        p = self.path / node
        if not p.is_dir():
            return {}
        result = {}
        for f in p.glob('*.pkl'):
            outer_idx, inner_idx = map(int, f.stem.split('_'))
            with open(f, 'rb') as fp:
                result[(outer_idx, inner_idx)] = pickle.load(fp)
        return result

    def get_properties(self):
        return {
            'need_output_train': self.include_train,
            'need_output_test': True,
            'need_process_data': False,
        }