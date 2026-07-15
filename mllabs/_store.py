import os
import shutil
import pickle as pkl
from pathlib import Path


class NodeStore:
    """On-disk artifact manager for one (outer, inner) fold.

    Node artifacts are stored at {path}/{node_name}/:
      obj.pkl    — processor object
      result.pkl — fit_transform/fit_predict output
      info.pkl   — info dict with 'status' key ('built'/'finalized'/'error')

    write_* are staticmethods that take node_path directly.
    get_* / status / finalize / reset_node are instance methods with lazy info cache.

    Args:
        path: (outer, inner) fold path.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._info_cache = {}  # {node_name: info}

    def _node_path(self, name):
        return self.path / name

    def _load_info(self, name):
        if name not in self._info_cache:
            info_file = self._node_path(name) / 'info.pkl'
            if not info_file.exists():
                return None
            with open(info_file, 'rb') as f:
                self._info_cache[name] = pkl.load(f)
        return self._info_cache[name]

    # -------------------------------------------------------------------------
    # Write (staticmethod — takes node_path)
    # -------------------------------------------------------------------------

    @staticmethod
    def write_objs(node_path, obj, result, info):
        node_path = Path(node_path)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'obj.pkl', 'wb') as f:
            pkl.dump(obj, f)
        with open(node_path / 'result.pkl', 'wb') as f:
            pkl.dump(result, f)
        with open(node_path / 'info.pkl', 'wb') as f:
            pkl.dump({**info, 'status': 'built'}, f)

    @staticmethod
    def write_obj(node_path, obj):
        node_path = Path(node_path)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'obj.pkl', 'wb') as f:
            pkl.dump(obj, f)

    @staticmethod
    def write_result(node_path, result):
        node_path = Path(node_path)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'result.pkl', 'wb') as f:
            pkl.dump(result, f)

    @staticmethod
    def write_info(node_path, info):
        node_path = Path(node_path)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'info.pkl', 'wb') as f:
            pkl.dump(info, f)

    # -------------------------------------------------------------------------
    # Get (instance — lazy cache)
    # -------------------------------------------------------------------------

    def get_objs(self, name):
        node_path = self._node_path(name)
        with open(node_path / 'obj.pkl', 'rb') as f:
            obj = pkl.load(f)
        with open(node_path / 'result.pkl', 'rb') as f:
            result = pkl.load(f)
        info = self._load_info(name)
        return obj, result, info

    def get_obj(self, name):
        with open(self._node_path(name) / 'obj.pkl', 'rb') as f:
            return pkl.load(f)

    def get_result(self, name):
        with open(self._node_path(name) / 'result.pkl', 'rb') as f:
            return pkl.load(f)

    def get_info(self, name):
        return self._load_info(name)

    def list_nodes(self):
        if not self.path.exists():
            return []
        return [p.name for p in self.path.iterdir() if p.is_dir()]

    # -------------------------------------------------------------------------
    # Status / lifecycle (instance)
    # -------------------------------------------------------------------------

    def status(self, name):
        info = self._load_info(name)
        return info.get('status') if info is not None else None

    def finalize(self, name):
        node_path = self._node_path(name)
        info = {**self._load_info(name), 'status': 'finalized'}
        with open(node_path / 'info.pkl', 'wb') as f:
            pkl.dump(info, f)
        self._info_cache[name] = info
        for fname in ('obj.pkl', 'result.pkl'):
            p = node_path / fname
            if p.exists():
                p.unlink()

    def reset_node(self, name):
        node_path = self._node_path(name)
        if os.path.isdir(node_path):
            shutil.rmtree(node_path)
        self._info_cache.pop(name, None)
