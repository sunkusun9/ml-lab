import re
import os
import sys
import uuid
import pickle as pkl
import shutil
import traceback
import warnings
from pathlib import Path

import pandas as pd
from cachetools import LRUCache

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap
from ._expobj import HeadObj, StageObj
from ._describer import desc_spec, desc_status, desc_obj_vars
from ._logger import DefaultLogger

from ._pipeline import Pipeline
from ._node_processor import resolve_columns
from ._connector import Connector
from .collector import Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
from ._trainer import Trainer

def _get_data_size(data):
    if data is None:
        return 0
    if isinstance(data, list):
        return sum(_get_data_size(item) for item in data)
    if isinstance(data, tuple):
        return sum(_get_data_size(item) for item in data)
    if hasattr(data, 'nbytes'):
        return data.nbytes
    if hasattr(data, 'memory_usage'):
        return data.memory_usage(deep=True).sum()
    return sys.getsizeof(data)

class DataCache():
    def __init__(self, maxsize=4 * 1024 ** 3):  # 4GB 기본값
        self.cache_dic = LRUCache(maxsize=maxsize, getsizeof=_get_data_size)

    def get_data(self, node, typ, idx):
        key = (node, typ, idx)
        return self.cache_dic.get(key, None)

    def put_data(self, node, typ, idx, data):
        key = (node, typ, idx)
        self.cache_dic[key] = data

    def clear(self):
        self.cache_dic.clear()

    def clear_nodes(self, nodes):
        node_set = set(nodes)
        keys_to_delete = [k for k in self.cache_dic.keys() if k[0] in node_set]
        for k in keys_to_delete:
            del self.cache_dic[k]

class Experimenter():
    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            logger = DefaultLogger(level=['info', 'progress'])
        ):
        self.cache_maxsize = cache_maxsize
        self.logger = logger
        self.path = Path(path)
        if not os.path.exists(path):
            self.path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {self.path}")
        self.train_idx_list = list()
        self.valid_idx_list = list()
        data_native = data
        self.data = wrap(data)
        # 실험 타이틀 저장
        self.title = title

        # data 식별자 (load 시 검증용)
        self.data_key = data_key

        # splitter 설정 저장
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}

        if data_names is None:
            data_names = self.data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))

        for train_idx, valid_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = self.data.iloc(train_idx)
                train_data_native = unwrap(train_data)

                inner_split_params = {'X': train_data_native}
                for k, v in self.splitter_params.items():
                    inner_split_params[k] = unwrap(train_data.select_columns(v))

                self.train_idx_list.append([
                    (train_idx[train_v_idx], train_idx[valid_v_idx])
                    for train_v_idx, valid_v_idx in sp_v.split(**inner_split_params)
                ])
            else:
                self.train_idx_list.append([
                    (train_idx, None)
                ])
            self.valid_idx_list.append(valid_idx)

        self.pipeline = Pipeline()
        self.node_objs = {}
        self.cache = DataCache(maxsize=cache_maxsize)
        self.grps = {}
        self.collectors = {}
        self.trainers = {}
        self.status = "open"
        self._save()

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def open(self):
        """Experimenter를 open 상태로 변경"""
        self.status = "open"
        self._save()
        self.logger.info("Experimenter status changed to 'open'")

    def close(self):
        """Experimenter를 close 상태로 변경"""
        self.status = "close"
        self._save()
        self.logger.info("Experimenter status changed to 'close'")

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            logger = DefaultLogger(level=['info', 'progress'])):

        if os.path.exists(path):
            raise RuntimeError(f"Exists: {path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params,
            title=title, data_key=data_key, cache_maxsize=cache_maxsize, logger=logger)

    def get_n_splits(self):
        return len(self.train_idx_list)

    def get_n_splits_inner(self):
        return len(self.train_idx_list[0])

    def add_collector(self, collector, exist = 'skip'):
        if collector.name in self.collectors:
            if exist == 'skip':
                return self.collectors[collector.name]
            elif exist == 'error':
                raise RuntimeError("")
        
        self._check_open()
        collector.path = self.path / '__collector' / collector.name
        collector.save()
        self.collectors[collector.name] = collector
        self.collect(collector)
        self._save()
        return collector

    def add_trainer(self, name, data=None, splitter="same", splitter_params=None, exist='skip'):
        if name in self.trainers:
            if exist == 'skip':
                return self.trainers[name]
            elif exist == 'error':
                raise RuntimeError(f"Trainer '{name}' already exists")

        if data is None:
            trainer_data = self.data
        else:
            trainer_data = wrap(data)

        if splitter == 'same':
            if splitter_params is not None:
                raise ValueError("splitter_params must be None when splitter='same'")
            trainer_splitter = self.sp_v
            trainer_splitter_params = self.splitter_params
        else:
            trainer_splitter = splitter
            trainer_splitter_params = splitter_params if splitter_params is not None else {}

        trainer = Trainer(
            name=name,
            pipeline=self.pipeline,
            data=trainer_data,
            path=self.path / '__trainer' / name,
            splitter=trainer_splitter,
            splitter_params=trainer_splitter_params,
            logger=self.logger,
            cache=self.cache
        )
        self.trainers[name] = trainer
        self._save()
        return trainer

    def _validate_name(self, name):
        """Node 또는 NodeGroup 이름 검증

        Args:
            name: 검증할 이름

        Raises:
            ValueError: 이름이 유효하지 않을 경우
        """
        if name is None:
            return

        # '__' 포함 금지
        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        # 파일/폴더명으로 사용 불가한 문자 금지
        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def get_grp_path(self, grp):
        if self.path is None:
            return None
        if isinstance(grp, str):
            grp = self.pipeline.get_grp(grp)
        path_parts = [grp.name]
        current = self.pipeline.get_grp(grp.parent)
        while current is not None:
            path_parts.insert(0, current.name)
            current = self.pipeline.get_grp(current.parent)
        return self.path / '/'.join(path_parts)

    def get_node_path(self, node):
        if isinstance(node, str):
            node = self.pipeline.get_node(node)
        grp_path = self.get_grp_path(node.grp)
        return grp_path / node.name

    def set_grp(self, name, role=None, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, exist = 'skip'):
        self._check_open()
        result_obj = self.pipeline.set_grp(
            name, role, processor, edges, method, parent, adapter, params, exist
        )
        
        affected_nodes = result_obj['affected_nodes']
        self.reset_nodes(affected_nodes)
        new_grp_path = self.get_grp_path(result_obj['grp'])
        if "old_grp" in result_obj:
            old_grp_path = self.get_grp_path(result_obj['old_grp'])
            if old_grp_path != new_grp_path:
                os.makedirs(new_grp_path, exist_ok=True)
                for fname in os.listdir(old_grp_path):
                    src_path = os.path.join(old_grp_path, fname)
                    dst_path = os.path.join(new_grp_path, fname)
                    shutil.move(src_path, dst_path)

            self.logger.info(f"Group '{name}' updated, {len(affected_nodes)} node(s) affected")
        self._save()
        return result_obj

    def rename_grp(self, name_from, name_to):
        self._check_open()
        old_grp_path = self.get_grp_path(name_from)
        self.pipeline.rename_grp(name_from, name_to)
        new_grp_path = self.get_grp_path(name_to)
        if old_grp_path.exists():
            os.makedirs(new_grp_path, exist_ok=True)
            for fname in os.listdir(old_grp_path):
                src_path = os.path.join(old_grp_path, fname)
                dst_path = os.path.join(new_grp_path, fname)
                shutil.move(src_path, dst_path)
            shutil.rmtree(old_grp_path)
        self._save()
    
    def remove_grp(self, name):
        self._check_open()
        self.pipeline.remove_grp(name)

        self.logger.info(f"Group '{name}' removed")
        self._save()


    def remove_node(self, name):
        """노드를 제거

        Args:
            name: 제거할 노드 이름

        Raises:
            ValueError: 노드가 존재하지 않거나, 자식 노드가 있는 경우
        """
        self._check_open()
        self.pipeline.remove_node(name)
        for v in self.collectors.values():
            v.reset_nodes([name])

        self.logger.info(f"Node '{name}' removed")
        self._save()

    def finalize(self, nodes):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        for i in node_names:
            if i is None:
                continue
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'head' and i in self.node_objs:
                node_obj = self.node_objs[i]
                if node_obj.status == 'built':
                    self.logger.info(f"Finalize '{i}'")
                    node_obj.finalize()

    def reinitialize(self, nodes):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        for i in node_names:
            if i is None:
                continue
            if i in self.node_objs:
                node_obj = self.node_objs[i]
                if node_obj.status == 'finalized':
                    self.logger.info(f"reinitialize '{i}'")
                    del self.node_objs[i]

    def close_exp(self):
        if self.status != "open":
            raise RuntimeError("")
        for k, node_obj in self.node_objs.items():
            if node_obj.status == 'built':
                self.logger.info(f"Finalize '{k}'")
                node_obj.finalize()
        self.status = "closed"
    
    def reopen_exp(self):
        if self.status != "closed":
            raise RuntimeError("")
        for k in list(self.node_objs.keys()):
            node = self.pipeline.get_node(k)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage':
                self.logger.info(f"Initialize '{k}'")
                del self.node_objs[k]
        self.build()

    def set_node(
        self, name, grp, processor = None, edges = None,
        method = None, adapter = None, params = None, exist = 'skip'
    ):
        self._check_open()
        result_obj = self.pipeline.set_node(
            name, grp, processor, edges, method, adapter, params, exist
        )

        # 기존 노드를 업데이트한 경우, 하위 노드들도 재빌드
        if len(result_obj['affected_nodes']) > 0:
            affected_nodes = result_obj['affected_nodes']
            self.logger.info(f"Affected {len(affected_nodes)} dependent node(s): {sorted(affected_nodes)}")
            self.reset_nodes(affected_nodes)

        if result_obj['result'] == 'update':
            self.reset_nodes([name])
        self._save()
        return result_obj

    def reset_nodes(self, nodes):
        for i in nodes:
            if i in self.node_objs:
                del self.node_objs[i]

        self.cache.clear_nodes(nodes)

        for v in self.collectors.values():
            v.reset_nodes(nodes)

        for v in self.trainers.values():
            v.reset_nodes(nodes)

    def build(self, nodes = None, rebuild = False):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        target_nodes = list()
        for i in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage' and i not in self.node_objs:
                target_nodes.append(i)

        self.logger.info(f"Building {len(target_nodes)} node(s)")
        for node in target_nodes:
            if node in self.node_objs:
                node_obj = self.node_objs[node]
            else:
                node_obj = StageObj(self.get_node_path(node))
                self.node_objs[node] = node_obj
            node_obj.start_build()
        
        n_splits = self.get_n_splits()
        self.logger.start_progress("Build", n_splits)
        for i in range(n_splits):
            self.logger.update_progress(i)
            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger._progress[-1][0] = node
                node_obj = self.node_objs[node]
                if node_obj.status == 'error':
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_attrs = self.pipeline.get_node_attrs(node)
                        node_obj.build_idx(
                            i, node_attrs, self.get_node_data(node, i), self.logger
                        )
                        for w in caught:
                            self.logger.warning(f"[{node}] fold {i}: {w.category.__name__}: {w.message}")
                except Exception as e:
                    node_obj.status = 'error'
                    node_obj.error = {
                        'type': type(e).__name__,
                        'message': str(e),
                        'traceback': traceback.format_exc(),
                        'fold': i,
                    }
                    self.logger.info(f"[{node}] Build error at fold {i}: {type(e).__name__}: {e}")
                    self.logger.info(traceback.format_exc())
            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        error_nodes = [n for n in target_nodes if self.node_objs[n].status == 'error']
        for node in target_nodes:
            node_obj = self.node_objs[node]
            if node_obj.status != 'error':
                node_obj.end_build()
        if error_nodes:
            self.logger.info(f"Build complete: {len(target_nodes) - len(error_nodes)}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Build complete: {len(target_nodes)} node(s)")
    
    def exp(self, nodes = None):
        self._check_open()
        node_names = set(self.pipeline.get_node_names(nodes))
        target_nodes = list()
        for i in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'head' and i not in self.node_objs and i in node_names:
                target_nodes.append(i)

        self.logger.info(f"Experimenting {len(target_nodes)} node(s)")

        # connector matching
        node_attrs_cache = {n: self.pipeline.get_node_attrs(n) for n in target_nodes}
        matched = {}
        for name, collector in self.collectors.items():
            matched[name] = set(
                n for n in target_nodes
                if collector.connector.match(n, node_attrs_cache[n])
            )

        # start_experiment for all nodes
        for node in target_nodes:
            if node in self.node_objs:
                node_obj = self.node_objs[node]
            else:
                node_obj = HeadObj(self.get_node_path(node))
                self.node_objs[node] = node_obj
            node_obj.start_exp()

        # collector _start
        for name, collector in self.collectors.items():
            for node in target_nodes:
                if node in matched[name]:
                    collector._start(node)

        # experiment loop
        n_splits = self.get_n_splits()
        self.logger.start_progress("Exp", n_splits)
        for i in range(n_splits):
            self.logger.update_progress(i)

            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger._progress[-1][0] = node
                node_obj = self.node_objs[node]
                if node_obj.status == 'error':
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_attrs = node_attrs_cache[node]
                        result_iter = node_obj.exp_idx(
                            i, node_attrs, self.get_node_data(node, i), self.logger
                        )

                        for inner_idx, result_data in enumerate(result_iter):
                            context = {
                                'node_attrs': node_attrs,
                                'processor': result_data['object'],
                                'spec': result_data['spec'],
                                'input': result_data['input'],
                                'output_train': result_data['output_train'],
                                'output_valid': result_data['output_valid'],
                            }
                            for name, collector in self.collectors.items():
                                if node in matched[name]:
                                    collector._collect(node, i, inner_idx, context)

                        for w in caught:
                            self.logger.warning(f"[{node}] fold {i}: {w.category.__name__}: {w.message}")

                    # collector _end_idx
                    for name, collector in self.collectors.items():
                        if node in matched[name]:
                            collector._end_idx(node, i)
                except Exception as e:
                    node_obj.status = 'error'
                    node_obj.error = {
                        'type': type(e).__name__,
                        'message': str(e),
                        'traceback': traceback.format_exc(),
                        'fold': i,
                    }
                    self.logger.info(f"[{node}] Exp error at fold {i}: {type(e).__name__}: {e}")
                    self.logger.info(traceback.format_exc())
                    for name, collector in self.collectors.items():
                        if node in matched[name]:
                            collector.reset_nodes([node])

            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        error_nodes = [n for n in target_nodes if self.node_objs[n].status == 'error']
        # end_experiment for non-error nodes
        for node in target_nodes:
            node_obj = self.node_objs[node]
            if node_obj.status != 'error':
                node_obj.end_exp()

        # collector _end for non-error nodes
        for name, collector in self.collectors.items():
            for node in target_nodes:
                if node in matched[name] and self.node_objs[node].status != 'error':
                    collector._end(node)

        if error_nodes:
            self.logger.info(f"Experimentation complete: {len(target_nodes) - len(error_nodes)}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Experimentation complete: {len(target_nodes)} node(s)")
        self._save()

    def collect(self, collector, exist = 'skip'):
        # built head 노드 중 connector 매칭
        target_nodes = []
        node_attrs_cache = {}
        for name in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(name)
            grp = self.pipeline.get_grp(node.grp)
            if name not in self.node_objs:
                continue
            if exist == 'skip' and collector.has(name):
                continue
            node_obj = self.node_objs[name]
            if node_obj.status != 'built':
                continue
            node_attrs = self.pipeline.get_node_attrs(name)
            if collector.connector.match(name, node_attrs) and not collector.has_node(name):
                target_nodes.append(name)
                node_attrs_cache[name] = node_attrs

        for node in target_nodes:
            collector._start(node)

        n_splits = self.get_n_splits()
        self.logger.start_progress("Collect", n_splits)
        for idx in range(n_splits):
            self.logger.update_progress(idx)
            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger._progress[-1][0] = node
                node_obj = self.node_objs[node]
                node_attrs = node_attrs_cache[node]
                result_iter = node_obj.exp_idx(
                    idx, node_attrs, self.get_node_data(node, idx), self.logger
                )
                for inner_idx, result_data in enumerate(result_iter):
                    context = {
                        'node_attrs': node_attrs,
                        'processor': result_data['object'],
                        'spec': result_data['spec'],
                        'input': result_data['input'],
                        'output_train': result_data['output_train'],
                        'output_valid': result_data['output_valid'],
                    }
                    collector._collect(node, idx, inner_idx, context)
                collector._end_idx(node, idx)
            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        for node in target_nodes:
            collector._end(node)

        return collector

    def get_data(self, idx, edges):
        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)

        for _ in range(self.get_n_splits_inner()):
            result = {}
            for k, it in data_dict.items():
                z = next(it)
                train_sub, valid_sub, outer_valid_sub = list(), list(), list()
                for (train_data, train_v_data), outer_valid_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                    outer_valid_sub.append(outer_valid_data)

                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    result[k] = ((train_concat, valid_concat), outer_concat)
                else:
                    result[k] = ((train_concat, None), outer_concat)
            yield result
    
    def get_data_train(self, idx, edges):
        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_train_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)
        
        for _ in range(self.get_n_splits_inner()):
            result = {}
            for k, it in data_dict.items():
                z = next(it)
                train_sub, valid_sub = list(), list()
                for train_data, train_v_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    result[k] = (train_concat, valid_concat)
                else:
                    result[k] = (train_concat, None)
            yield result
    
    def get_data_valid(self, idx, edges):
        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_valid_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)
        for _ in range(self.get_n_splits_inner()):
            result = {}
            for k, it in data_dict.items():
                z = next(it)
                outer_valid_sub = list()
                for outer_valid_data in z:
                    outer_valid_sub.append(outer_valid_data)
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                result[k] = outer_concat
            yield result

    def get_node_data(self, node, idx):
        node_attrs = self.pipeline.get_node_attrs(node)
        return self.get_data(idx, node_attrs['edges'])
    
    def get_node_data_train(self, node, idx):
        node_attrs = self.pipeline.get_node_attrs(node)
        return self.get_data_train(idx, node_attrs['edges'])

    def get_node_data_valid(self, node, idx):
        node_attrs = self.pipeline.get_node_attrs(node)
        return self.get_data_valid(idx, node_attrs['edges'])

    def split(self, edges):
        for idx in range(len(self.train_idx_list)):
            yield self.get_data(idx, edges)
    
    def get_node_output(self, idx, node, v = None):
        if node is None:
            outer_valid_data = self.data.iloc(self.valid_idx_list[idx])
            if v is not None:
                outer_valid_data = outer_valid_data.select_columns(v)
            
            for train_v_idx, valid_v_idx in self.train_idx_list[idx]:
                if v is None:
                    train_data = self.data.iloc(train_v_idx)
                    train_v_data = self.data.iloc(valid_v_idx) if valid_v_idx is not None else None
                else:
                    train_data = self.data.iloc(train_v_idx).select_columns(v)
                    if valid_v_idx is not None:
                        train_v_data = self.data.iloc(valid_v_idx).select_columns(v)
                    else:
                        train_v_data = None

                yield (train_data, train_v_data), outer_valid_data
            return

        cached = self.cache.get_data(node, "all", idx)
        if cached is not None:
            sub = self.node_objs[node].get_objs(idx)
            for ((train_result, train_v_result), valid_result), (obj, _, _) in zip(cached, sub):
                X = resolve_columns(train_result, v, processor=obj)
                train_result = train_result.select_columns(X)
                if train_v_result is not None:
                    train_v_result = train_v_result.select_columns(X)
                valid_result = valid_result.select_columns(X)
                yield (train_result, train_v_result), valid_result
            return
        it = self.get_node_data(node, idx)
        sub = self.node_objs[node].get_objs(idx)
        use_cache = self.cache_maxsize > 0
        cache_data = list() if use_cache else None
        
        for data_dict, (obj, train_, info) in zip(it, sub):
            # X key로 입력 데이터 가져오기
            (train_X, train_v_X), valid_X = data_dict['X']

            # train data 처리
            if train_ is None:
                train_result = obj.process(train_X)
            else:
                train_result = train_

            # train_v data 처리
            if train_v_X is not None:
                train_v_result = obj.process(train_v_X)
            else:
                train_v_result = None

            # valid data 처리 (외부 fold의 valid)
            valid_result = obj.process(valid_X)
            if use_cache:
                cache_data.append(((train_result, train_v_result), valid_result))
            # 필요하면 컬럼 필터링
            if v is not None:
                X = resolve_columns(train_result, v, processor=obj)
                train_result = train_result.select_columns(X)
                if train_v_result is not None:
                    train_v_result = train_v_result.select_columns(X)
                valid_result = valid_result.select_columns(X)

            yield (train_result, train_v_result), valid_result
        if use_cache:
            self.cache.put_data(node, "all", idx, cache_data)

    def get_node_train_output(self, idx, node, v=None):
        if node is None:
            for train_v_idx, valid_v_idx in self.train_idx_list[idx]:
                if v is None:
                    train_data = self.data.iloc(train_v_idx)
                    train_v_data = self.data.iloc(valid_v_idx) if valid_v_idx is not None else None
                else:
                    train_data = self.data.iloc(train_v_idx).select_columns(v)
                    if valid_v_idx is not None:
                        train_v_data = self.data.iloc(valid_v_idx).select_columns(v)
                    else:
                        train_v_data = None

                yield train_data, train_v_data
            return
        cached = self.cache.get_data(node, "train", idx)
        if cached is not None:
            sub = self.node_objs[node].get_objs(idx)
            for (train_result, train_v_result), (obj, _, _) in zip(cached, sub):
                X = resolve_columns(train_result, v, processor=obj)
                train_result = train_result.select_columns(X)
                if train_v_result is not None:
                    train_v_result = train_v_result.select_columns(X)
                yield train_result, train_v_result
            return

        it = self.get_node_data_train(node, idx)
        sub = self.node_objs[node].get_objs(idx)
        cache_data = list()
        for data_dict, (obj, train_, info) in zip(it, sub):
            train_X, train_v_X = data_dict['X']
            train_result = obj.process(train_X) if train_ is None else train_
            if train_v_X is not None:
                train_v_result = obj.process(train_v_X)
            else:
                train_v_result = None
            cache_data.append((train_result, train_v_result))
            if v is not None:
                X = resolve_columns(train_result, v, processor=obj)
                train_result = train_result.select_columns(X)
                if train_v_result is not None:
                    train_v_result = train_v_result.select_columns(X)
            yld = train_result, train_v_result
            yield yld
        self.cache.put_data(node, "train", idx, cache_data)
    
    def get_node_valid_output(self, idx, node, v=None):
        if node is None:
            outer_valid_data = self.data.iloc(self.valid_idx_list[idx])
            if v is not None:
                outer_valid_data = outer_valid_data.select_columns(v)
            
            for _ in range(self.get_n_splits_inner()):
                yield outer_valid_data

        cached = self.cache.get_data(node, "valid", idx)
        if cached is not None:
            sub = self.node_objs[node].get_objs(idx)
            for valid_result, (obj, _, _) in zip(cached, sub):
                X = resolve_columns(valid_result, v, processor=obj)
                valid_result = valid_result.select_columns(X)
                yield valid_result

        it = self.get_node_data_valid(node, idx)
        sub = self.node_objs[node].get_objs(idx)
        use_cache = self.cache_maxsize > 0
        cache_data = list() if use_cache else None
        for data_dict, (obj, train_, info) in zip(it, sub):
            # X key로 valid 데이터 가져오기
            valid_X = data_dict['X']
            # valid data 처리 (외부 fold의 valid)
            valid_result = obj.process(valid_X)
            if use_cache:
                cache_data.append(valid_result)
            # 필요하면 컬럼 필터링
            if v is not None:
                X = resolve_columns(valid_result, v, processor=obj)
                valid_result = valid_result.select_columns(X)
            yield valid_result
        if use_cache:
            self.cache.put_data(node, "valid", idx, cache_data)
        return ret_func()
    
    def get_node_info(self):
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **DataSource**\n")

        for name in self.pipeline.nodes.keys():
            if name is None:
                continue
            node = self.pipeline.get_node(name)
            node_attrs = node.get_attrs(self.pipeline.grps)
            processor_name = node_attrs['processor'].__name__ if node_attrs['processor'] else 'None'
            edges_info_parts = []
            for key, edge_list in node_attrs['edges'].items():
                edge_strs = [f"{n or 'DataSource'}{f'[{v}]' if v else ''}" for n, v in edge_list]
                edges_info_parts.append(f"{key}: [{', '.join(edge_strs)}]")
            edges_info = ", ".join(edges_info_parts)
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node_attrs['method']}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = self.pipeline._find_descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def get_objs(self, node_name, idx):
        if node_name not in self.node_objs or node_name is None:
            raise ValueError(f"Node '{node_name}' objects not found")

        obj = self.node_objs[node_name]

        # 노드가 빌드되지 않았으면 에러
        if obj.status != 'built':
            raise ValueError(f"Node '{node_name}' status should be built")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        return obj.get_objs(idx)
    
    def get_obj_vars(self, node_name, idx):
        if node_name not in self.node_objs:
            raise ValueError(f"Node '{node_name}' has no object")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        objs_ = self.node_objs[node_name].get_objs(idx)

        # (입력변수 튜플, 출력변수 튜플) -> 내부 fold index 리스트
        var_map = {}
        for inner_idx, (processor, train_v, info) in enumerate(objs_):
            # 입력 변수와 출력 변수 가져오기
            input_vars = tuple(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else ()
            output_vars = tuple(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else ()

            # 튜플 키 생성
            key = (input_vars, output_vars)

            if key not in var_map:
                var_map[key] = []
            var_map[key].append(inner_idx)

        # 결과 리스트 생성: [(입력변수 리스트, 출력변수 리스트, 내부 폴드 index 리스트), ...]
        result = []
        for (input_vars, output_vars), fold_indices in var_map.items():
            result.append((list(input_vars), list(output_vars), fold_indices))

        # 등장 빈도(내부 폴드 개수)의 내림차순으로 정렬
        result.sort(key=lambda x: len(x[2]), reverse=True)
        return result

    def get_edges_var(self, edges):
        class _ColHolder:
            def __init__(self, columns):
                self._columns = columns
            def get_columns(self):
                return self._columns

        var_map = {}

        for idx in range(self.get_n_splits()):
            n_inner = len(self.train_idx_list[idx])
            # edges는 dict: {key: [(node_name, var), ...], ...}
            edge_objs = {}
            for key, edge_list in edges.items():
                edge_objs[key] = []
                for node_name, var in edge_list:
                    if node_name is None:
                        edge_objs[key].append((None, var, None))
                    else:
                        node_obj = self.node_objs[node_name]
                        edge_objs[key].append((node_name, var, node_obj.get_objs(idx)))

            for inner_idx in range(n_inner):
                collected = {}
                for key, objs_list in edge_objs.items():
                    collected[key] = []
                    for node_name, var, objs in objs_list:
                        if node_name is None:
                            cols = self.data.get_columns()
                            proc = None
                        else:
                            obj = next(objs)
                            proc = obj[0]
                            cols = list(proc.output_vars) if proc.output_vars is not None else []

                        if var is not None:
                            cols = resolve_columns(_ColHolder(cols), var, processor=proc)

                        collected[key].extend(cols)

                # key를 정렬된 형태로 튜플화
                key_tuple = tuple((k, tuple(v)) for k, v in sorted(collected.items()))
                if key_tuple not in var_map:
                    var_map[key_tuple] = []
                var_map[key_tuple].append((idx, inner_idx))

        result = []
        for vars_tuple, fold_indices in var_map.items():
            # dict로 복원
            vars_dict = {k: list(v) for k, v in vars_tuple}
            result.append((vars_dict, fold_indices))

        result.sort(key=lambda x: len(x[1]), reverse=True)

        return result

    def _save(self, filepath=None):
        if filepath is None:
            filepath = self.path / '__exp.pkl'

        save_data = {
            'data_key': self.data_key,
            'title': self.title,
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
            'cache_maxsize': self.cache_maxsize,
            'exp_id': self.exp_id,
            'pipeline': self.pipeline,
            'node_obj_keys': list(self.node_objs.keys()),
            'collector_keys': {name: type(c).__name__ for name, c in self.collectors.items()},
            'trainer_keys': list(self.trainers.keys()),
            'status': self.status
        }

        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)

    @staticmethod
    def load(filepath, data, data_key=None):
        COLLECTOR_TYPES = {
            'MetricCollector': MetricCollector,
            'StackingCollector': StackingCollector,
            'ModelAttrCollector': ModelAttrCollector,
            'SHAPCollector': SHAPCollector,
        }

        filepath = Path(filepath)
        with open(filepath / '__exp.pkl', 'rb') as f:
            save_data = pkl.load(f)

        saved_data_key = save_data.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        exp = Experimenter(
            data=data,
            path=filepath,
            sp=save_data['sp'],
            sp_v=save_data['sp_v'],
            splitter_params=save_data['splitter_params'],
            title=save_data['title'],
            data_key=saved_data_key,
            cache_maxsize=save_data.get('cache_maxsize', 4 * 1024 ** 3)
        )
        exp.exp_id = save_data['exp_id']
        exp.pipeline = save_data['pipeline']
        exp.status = save_data['status']

        # node_objs 복원
        for node_name in save_data['node_obj_keys']:
            node = exp.pipeline.get_node(node_name)
            grp = exp.pipeline.get_grp(node.grp)
            node_path = exp.get_node_path(node_name)

            if grp.role == 'stage':
                node_obj = StageObj(node_path)
            else:
                node_obj = HeadObj(node_path)
            node_obj.load()
            exp.node_objs[node_name] = node_obj

        # Collector 복원
        collector_keys = save_data.get('collector_keys', {})
        for coll_name, type_name in collector_keys.items():
            cls = COLLECTOR_TYPES.get(type_name)
            if cls is None:
                continue
            coll_path = filepath / '__collector' / coll_name
            if (coll_path / '__config.pkl').exists():
                collector = cls.load(coll_path)
                exp.collectors[coll_name] = collector

        # Trainer 복원
        from ._trainer import Trainer
        for trainer_name in save_data.get('trainer_keys', []):
            trainer_path = filepath / '__trainer' / trainer_name
            if (trainer_path / '__trainer.pkl').exists():
                trainer = Trainer._load(
                    trainer_path,
                    pipeline=exp.pipeline,
                    data=exp.data,
                    cache=exp.cache,
                    logger=exp.logger,
                )
                exp.trainers[trainer_name] = trainer

        exp.logger.info(f"Loaded: {len(exp.pipeline.nodes) - 1} node(s), {len(exp.pipeline.grps)} group(s), {len(exp.train_idx_list)} fold(s)")
        return exp
    
    def export_pipeline(self):
        return self.pipeline.copy()

    def import_pipeline(self, pipeline):
        if len(self.pipeline.nodes) > 0 or len(self.pipeline.grps) > 0:
            raise RuntimeError("")
        self.pipeline = pipeline.copy()
    
    def desc_status(self):
        return desc_status(self)

    def desc_spec(self):
        return desc_spec(self)

    def desc_obj_vars(self, node_name, idx):
        obj_vars = self.get_obj_vars(node_name, idx)
        return desc_obj_vars(self, obj_vars[0])

    def desc_pipeline(self, max_depth=None, direction='TD'):
        return self.pipeline.desc_pipeline(max_depth, direction)

    def desc_node(self, node_name, direction='TD', show_params=False):
        return self.pipeline.desc_node(node_name, direction, show_params)