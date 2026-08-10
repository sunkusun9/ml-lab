import sys
from cachetools import LRUCache


def _get_data_size(data):
    if data is None:
        return 0
    if isinstance(data, (list, tuple)):
        return sum(_get_data_size(item) for item in data)
    if hasattr(data, 'nbytes'):
        return data.nbytes
    if hasattr(data, 'memory_usage'):
        return data.memory_usage(deep=True).sum()
    return sys.getsizeof(data)


class DataCache:
    """LRU cache shared by every Experimenter/Trainer in a Project.

    Key is ``(scope, node, typ)`` — no ``outer_idx``/``inner_idx``. ``scope``
    is a random id a ``TrainDataFlow`` generates for itself in its own
    constructor (2026-08-01; was ``str(NodeStore.path)`` before — a real
    collision risk once Experimenter/Trainer became constructible standalone
    with an externally-supplied cache, since a bare ``Path`` string isn't
    resolved to absolute and two different directories could coincide). A
    ``TrainDataFlow`` instance *is* one specific (outer_idx, inner_idx) fold —
    one gets constructed per fold, never shared — so its own scope id already
    uniquely identifies which Experimenter/Trainer and which fold, without needing to fold path
    strings or numeric coordinates into the key at all. The one thing this
    trades away: a fresh reload (new Python objects, e.g. via
    ``Project.load_experimenter``) gets fresh scope ids, so cached entries
    from a previous instance are never hit again — acceptable, since a cache
    miss here just means recomputing, not wrong data.
    """

    def __init__(self, maxsize=4 * 1024 ** 3):
        self.cache_dic = LRUCache(maxsize=maxsize, getsizeof=_get_data_size)

    def get_data(self, scope, node, typ):
        return self.cache_dic.get((scope, node, typ), None)

    def put_data(self, scope, node, typ, data):
        self.cache_dic[(scope, node, typ)] = data

    def clear(self):
        self.cache_dic.clear()

    def clear_nodes(self, nodes):
        node_set = set(nodes)
        for k in [k for k in self.cache_dic if k[1] in node_set]:
            del self.cache_dic[k]
