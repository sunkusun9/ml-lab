import os
import shutil
import pickle as pkl
import traceback
import warnings
import numpy as np
from pathlib import Path

from ._data_wrapper import wrap, unwrap
from ._trainobj import TrainStageObj, TrainHeadObj
from ._node_processor import resolve_columns


class Trainer:

    def __init__(self, name, pipeline, data, path, splitter, splitter_params, cache, logger):
        self.name = name
        self.pipeline = pipeline
        self.data = data
        self.path = Path(path)
        self.splitter = splitter
        self.splitter_params = splitter_params
        self.cache = cache
        self.logger = logger

        self.selected_stages = []
        self.selected_heads = []

        self.node_objs = {}

        self.split_indices = self._make_splits()
        self.save()

    # ------------------------------------------------------------------
    # node selection
    # ------------------------------------------------------------------

    def select_head(self, nodes):
        node_names = self.pipeline.get_node_names(nodes)

        selected = set(self.selected_stages + self.selected_heads)
        for name in node_names:
            selected.add(name)
            self._collect_upstream(name, selected)

        all_ordered = self.pipeline._get_affected_nodes([None])
        self.selected_stages = []
        self.selected_heads = []
        for n in all_ordered:
            if n is None or n not in selected:
                continue
            grp = self.pipeline.get_grp(self.pipeline.get_node(n).grp)
            if grp.role == 'stage':
                self.selected_stages.append(n)
            elif grp.role == 'head':
                self.selected_heads.append(n)

    def _collect_upstream(self, node_name, selected):
        node_attrs = self.pipeline.get_node_attrs(node_name)
        for key, edge_list in node_attrs['edges'].items():
            for source_node, var in edge_list:
                if source_node is not None and source_node not in selected:
                    selected.add(source_node)
                    self._collect_upstream(source_node, selected)

    def reset_nodes(self, nodes):
        selected_set = set(self.selected_stages + self.selected_heads)
        affected = set(n for n in nodes if n in selected_set)

        queue = list(affected)
        while queue:
            n = queue.pop(0)
            node = self.pipeline.get_node(n)
            for downstream in node.output_edges:
                if downstream in selected_set and downstream not in affected:
                    affected.add(downstream)
                    queue.append(downstream)

        for node_name in affected:
            if node_name in self.node_objs:
                node_obj = self.node_objs.pop(node_name)
                if os.path.isdir(node_obj.path):
                    shutil.rmtree(node_obj.path)

        if self.cache is not None:
            self.cache.clear_nodes(affected)

        self.save()

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(self):
        selected_set = set(self.selected_stages + self.selected_heads)
        stage_set = set(self.selected_stages)

        target_nodes = []
        for name in self.pipeline._get_affected_nodes([None]):
            if name not in selected_set:
                continue
            if name in self.node_objs:
                continue
            target_nodes.append(name)

        for node_name in target_nodes:
            if node_name in stage_set:
                node_obj = TrainStageObj(self._get_node_path(node_name))
            else:
                node_obj = TrainHeadObj(self._get_node_path(node_name))
            node_obj.start_build()
            self.node_objs[node_name] = node_obj

        n_splits = self.get_n_splits()
        self.logger.start_progress("Train", len(target_nodes))
        for ni, node_name in enumerate(target_nodes):
            self.logger.update_progress(ni)
            self.logger._progress[-1][0] = node_name
            node_obj = self.node_objs[node_name]
            node_attrs = self.pipeline.get_node_attrs(node_name)
            self.logger.start_progress("Split", n_splits)
            for split_idx, data_dict in enumerate(self.get_data(node_attrs['edges'])):
                self.logger.update_progress(split_idx)
                if node_obj.status == 'error':
                    break
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_obj.build_split(split_idx, node_attrs, data_dict, self.logger)
                        for w in caught:
                            self.logger.warning(f"[{node_name}] split {split_idx}: {w.category.__name__}: {w.message}")
                except Exception as e:
                    node_obj.status = 'error'
                    node_obj.error = {
                        'type': type(e).__name__,
                        'message': str(e),
                        'traceback': traceback.format_exc(),
                        'split': split_idx,
                    }
                    self.logger.info(f"[{node_name}] Train error at split {split_idx}: {type(e).__name__}: {e}")
                    self.logger.info(traceback.format_exc())
                    self.cache.clear_nodes([node_name])
            self.logger.end_progress(n_splits)
            if node_obj.status != 'error':
                node_obj.end_build()
        self.logger.end_progress(len(target_nodes))

        error_nodes = [n for n in target_nodes if self.node_objs[n].status == 'error']
        if error_nodes:
            self.logger.info(f"Train complete: {len(target_nodes) - len(error_nodes)}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Train complete: {len(target_nodes)} node(s)")

        self.save()

    # ------------------------------------------------------------------
    # data resolution
    # ------------------------------------------------------------------

    def _make_splits(self):
        if self.splitter is None:
            return None
        data_native = unwrap(self.data)
        split_params = {'X': data_native}
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))
        return [
            (train_idx, valid_idx)
            for train_idx, valid_idx in self.splitter.split(**split_params)
        ]

    def get_n_splits(self):
        if self.split_indices is None:
            return 1
        return len(self.split_indices)

    def get_data(self, edges):
        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_output(node_name, var))
            data_dict[key] = zip(*key_data_list)

        for _ in range(self.get_n_splits()):
            result = {}
            for k, it in data_dict.items():
                z = next(it)
                train_sub, valid_sub = [], []
                for train_data, valid_data in z:
                    train_sub.append(train_data)
                    if valid_data is not None:
                        valid_sub.append(valid_data)
                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                if valid_sub:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                else:
                    valid_concat = None
                result[k] = train_concat, valid_concat
            yield result

    def get_node_output(self, node, v=None):
        if node is None:
            if self.split_indices is None:
                data = self.data if v is None else self.data.select_columns(v)
                yield data, None
            else:
                for train_idx, valid_idx in self.split_indices:
                    train_data = self.data.iloc(train_idx)
                    valid_data = self.data.iloc(valid_idx)
                    if v is not None:
                        train_data = train_data.select_columns(v)
                        valid_data = valid_data.select_columns(v)
                    yield train_data, valid_data
            return

        cached = self.cache.get_data(node, "train_all", 0)
        if cached is not None:
            for (train_result, valid_result), (obj, _, _) in zip(cached, self.node_objs[node].get_obj()):
                if v is not None:
                    X = resolve_columns(train_result, v, processor=obj)
                    train_result = train_result.select_columns(X)
                    if valid_result is not None:
                        valid_result = valid_result.select_columns(X)
                yield train_result, valid_result
            return

        node_attrs = self.pipeline.get_node_attrs(node)
        cache_data = []
        for data_dict, (obj, train_, info) in zip(self.get_data(node_attrs['edges']), self.node_objs[node].get_obj()):
            train_X, valid_X = data_dict['X']

            if train_ is None:
                train_result = obj.process(train_X)
            else:
                train_result = train_

            valid_result = obj.process(valid_X) if valid_X is not None else None

            cache_data.append((train_result, valid_result))

            if v is not None:
                X = resolve_columns(train_result, v, processor=obj)
                train_result = train_result.select_columns(X)
                if valid_result is not None:
                    valid_result = valid_result.select_columns(X)

            yield train_result, valid_result
        self.cache.put_data(node, "train_all", 0, cache_data)

    def get_node_data(self, node):
        node_attrs = self.pipeline.get_node_attrs(node)
        return self.get_data(node_attrs['edges'])
    
    def process(self, data, v=None):
        data = wrap(data)
        ordered = [
            name for name in self.pipeline._get_affected_nodes([None])
            if name in set(self.selected_stages + self.selected_heads)
        ]
        stage_set = set(self.selected_stages)

        obj_iters = {name: self.node_objs[name].get_obj() for name in ordered}

        for _ in range(self.get_n_splits()):
            data_dicts = {None: (data, None)}
            head_outputs = []

            for name in ordered:
                obj, _, _ = next(obj_iters[name])
                node_attrs = self.pipeline.get_node_attrs(name)
                output = self._process_node(obj, data_dicts, node_attrs['edges'])

                if name in stage_set:
                    data_dicts[name] = (output, obj)
                else:
                    if v is not None:
                        cols = resolve_columns(output, v, processor=obj)
                        output = output.select_columns(cols)
                    head_outputs.append(output)

            if len(head_outputs) == 1:
                yield head_outputs[0]
            else:
                yield type(head_outputs[0]).concat(head_outputs, axis=1)

    def _process_node(self, obj, data_dicts, edges):
        input_data = self._get_process_data(data_dicts, edges)
        return obj.process(input_data['X'])

    def _get_process_data(self, data_dicts, edges):
        result = {}
        for key, edge_list in edges.items():
            parts = []
            for src_node, var in edge_list:
                src, obj = data_dicts[src_node]
                if var is not None:
                    cols = resolve_columns(src, var, processor=obj)
                    src = src.select_columns(cols)
                parts.append(src)
            if len(parts) == 1:
                result[key] = parts[0]
            else:
                result[key] = type(parts[0]).concat(parts, axis=1)
        return result

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------

    def _get_grp_path(self, grp):
        if isinstance(grp, str):
            grp = self.pipeline.get_grp(grp)
        path_parts = [grp.name]
        current = self.pipeline.get_grp(grp.parent)
        while current is not None:
            path_parts.insert(0, current.name)
            current = self.pipeline.get_grp(current.parent)
        return self.path / '/'.join(path_parts)

    def _get_node_path(self, node_name):
        node = self.pipeline.get_node(node_name)
        return self._get_grp_path(node.grp) / node.name

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self):
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        save_data = {
            'name': self.name,
            'splitter': self.splitter,
            'splitter_params': self.splitter_params,
            'selected_stages': self.selected_stages,
            'selected_heads': self.selected_heads,
            'node_obj_keys': list(self.node_objs.keys()),
            'split_indices': self.split_indices,
        }
        with open(self.path / '__trainer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

    @classmethod
    def _load(cls, path, pipeline, data, cache, logger):
        path = Path(path)
        with open(path / '__trainer.pkl', 'rb') as f:
            save_data = pkl.load(f)

        trainer = object.__new__(cls)
        trainer.name = save_data['name']
        trainer.pipeline = pipeline
        trainer.data = data
        trainer.path = path
        trainer.splitter = save_data['splitter']
        trainer.splitter_params = save_data['splitter_params']
        trainer.cache = cache
        trainer.logger = logger
        trainer.selected_stages = save_data['selected_stages']
        trainer.selected_heads = save_data['selected_heads']
        trainer.split_indices = save_data['split_indices']

        stage_set = set(trainer.selected_stages)
        trainer.node_objs = {}
        for node_name in save_data['node_obj_keys']:
            node_path = trainer._get_node_path(node_name)
            if os.path.isdir(node_path):
                if node_name in stage_set:
                    node_obj = TrainStageObj(node_path)
                else:
                    node_obj = TrainHeadObj(node_path)
                node_obj.load()
                trainer.node_objs[node_name] = node_obj

        return trainer
