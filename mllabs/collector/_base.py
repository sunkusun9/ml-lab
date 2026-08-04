class Collector:
    _SAVE_EXCLUDE = {'_buf': dict}  # {attr: factory} — load 시 factory()로 초기화

    def __init__(self, name, connector):
        self.name = name
        self.connector = connector
        self.path = None
        self._n_outer = None
        self._n_inner = None
        self._buf = {}  # {node: {outer_idx: {inner_idx: result}}}
        self._experimenter = None

    def on_attach(self, experimenter):
        if experimenter is self._experimenter:
            return
        self._on_attach(experimenter)
        self._experimenter = experimenter

    def _on_attach(self, experimenter):
        pass

    def _setup(self, n_outer, n_inner):
        self._n_outer = n_outer
        self._n_inner = n_inner

    def collect(self, context):
        return None

    def push(self, node, outer_idx, inner_idx, result):
        outer_buf = self._buf.setdefault(node, {}).setdefault(outer_idx, {})
        outer_buf[inner_idx] = result
        if self._n_inner is not None and len(outer_buf) == self._n_inner:
            inner_list = [outer_buf.get(i) for i in range(self._n_inner)]
            del self._buf[node][outer_idx]
            self._flush_outer(node, outer_idx, inner_list)

    def _flush_outer(self, node, outer_idx, inner_list):
        pass

    def has_node(self, node):
        return False

    def abort_node(self, node):
        self._buf.pop(node, None)

    def reset_nodes(self, nodes):
        node_set = set(nodes)
        self._buf = {k: v for k, v in self._buf.items() if k not in node_set}

    def __getstate__(self):
        exclude = set(self._SAVE_EXCLUDE.keys()) | {'_experimenter'}
        return {k: v for k, v in self.__dict__.items() if k not in exclude}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr, factory in self._SAVE_EXCLUDE.items():
            setattr(self, attr, factory())
        self._experimenter = None

    def get_properties(self):
        return {
            'need_output_train': False,
            'need_output_test': False,
            'need_process_data': False,
        }