import uuid
import pickle as pkl
from .adapter import get_adapter
from ._node_processor import TransformProcessor, PredictProcessor, resolve_columns
import numpy as np
import pandas as pd
import os
import shutil
import time

def _build_sub(node_attrs, data_dict, fit_process, logger):
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        obj = TransformProcessor(node_attrs['name'], node_attrs['processor'], node_attrs['adapter'], node_attrs['params'], logger = logger)
    else:
        obj = PredictProcessor(node_attrs['name'], node_attrs['processor'], method, node_attrs['adapter'], node_attrs['params'], logger = logger)

    # (train, train_v) only — strip valid from ((train, train_v), valid)
    fit_data = {key: val[0] for key, val in data_dict.items()}

    start_time = time.time()
    if fit_process:
        result = obj.fit_process(fit_data)
    else:
        result = None
        obj.fit(fit_data)
    elapsed_time = time.time() - start_time

    train_X, train_v_X = fit_data['X']
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': train_X.get_shape() if train_X is not None else None,
        'train_v_shape': train_v_X.get_shape() if train_v_X is not None else None
    }
    return obj, result, info

def _build_iter(node_attrs, data_dict_it, logger):
    method = node_attrs['method']
    if method in ['transform', 'predict', 'predict_proba']:
        fit_process = False
    elif method in ['fit_transform', 'fit_predict']:
        fit_process = True
    else:
        raise ValueError(f"Unknown processor_type: {method}")
    
    for data_dict in data_dict_it:
        yield _build_sub(node_attrs, data_dict, fit_process, logger), data_dict

def _build_iter_output(node_attrs, data_dict_it, logger):
    method = node_attrs['method']
    if method in ['transform', 'predict', 'predict_proba']:
        fit_process = False
    elif method in ['fit_transform', 'fit_predict']:
        fit_process = True
    else:
        raise ValueError(f"Unknown processor_type: {method}")
    
    for data_dict in data_dict_it:
        obj, result, info =  _build_sub(node_attrs, data_dict, fit_process, logger)
        (train_X, train_v_X), valid_X = data_dict['X']
        if result is None:
            train_result = obj.process(train_X)
        else:
            train_result = result
        if train_v_X is not None:
            train_v_result = obj.process(train_v_X)
        else:
            train_v_result = None
        output_train = (train_result, train_v_result)
        output_valid = obj.process(valid_X)
        yield obj, result, info, data_dict, output_train, output_valid

class StageObj():
    def __init__(self, path):
        self.path = path
        self.status = None
        self.error = None

    def load(self):
        if not os.path.isdir(self.path):
            self.status = 'finalized'
            self.objs_ = None
            return

        self.objs_ = []
        idx = 0
        while True:
            objs = []
            no = 0
            while True:
                filename = self.path / ('obj' + str(idx) + '_' + str(no) + '.pkl')
                if not os.path.isfile(filename):
                    break
                with open(filename, 'rb') as f:
                    objs.append(pkl.load(f))
                no += 1
            if len(objs) == 0:
                break
            self.objs_.append(objs)
            idx += 1

        if len(self.objs_) == 0:
            self.status = 'finalized'
            self.objs_ = None
        else:
            self.status = 'built'

    def start_exp(self, finalize = False):
        if finalize:
            raise ValueError("StageObj cannot be finalized after Experiment")

    def exp_idx(self, idx, node_attrs, data_dict_it, logger, include_input = True, include_output = True):
        if self.status == "built":
            for data_dict, (obj, train_, spec) in zip(data_dict_it, self.objs_[idx]):
                # X key로 데이터 가져오기
                (train_X, train_v_X), valid_X = data_dict['X']
                sub_result = {'spec': spec, 'object': obj}
                if include_output:
                    if train_ is None:
                        train_result = obj.process(train_X)
                    else:
                        train_result = train_
                    if train_v_X is not None:
                        train_v_result = obj.process(train_v_X)
                    else:
                        train_v_result = None
                    if include_output:
                        sub_result['output_train'] = (train_result, train_v_result)
                        sub_result['output_valid'] = obj.process(valid_X)
                    if include_input:
                        sub_result['input'] = data_dict
                    yield sub_result
        elif self.status == "finalized":
            raise RuntimeError(f"Node is finalized and cannot be re-experimented")
        else:
            raise RuntimeError(f"StageObj cannot be experimented unless built")
    
    def end_exp(self):
        pass
    
    def start_build(self):
        self.objs_ = list()
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok = True)
    
    def build_idx(self, idx, node_attrs, data_dict_it, logger):
        if idx != len(self.objs_):
            raise RuntimeError(f"Build sequence is not valid")
        objs = list()
        for no, (obj, _) in enumerate(_build_iter(node_attrs, data_dict_it,logger)):
            filename = self.path / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
            with open(filename, 'wb') as f:
                pkl.dump(obj, f)
            objs.append(obj)
        self.objs_.append(objs)

    def end_build(self):
        self.status = 'built'
    
    def get_objs(self, idx):
        for obj, train_, spec in self.objs_[idx]:
            yield obj, train_, spec
        
    def finalize(self):
        self.status = 'finalized'
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        self.objs_ = None


class HeadObj():
    def __init__(self, path):
        self.path = path
        self.status = None
        self.error = None

    def load(self):
        if not os.path.isdir(self.path):
            self.status = 'finalized'
            return

        idx = 0
        while True:
            filename = self.path / ('obj' + str(idx) + '_0.pkl')
            if not os.path.isfile(filename):
                break
            idx += 1

        if idx == 0:
            self.status = 'finalized'
        else:
            self.status = 'built'

    def start_exp(self, finalize = False):
        self.finalize_after_exp = finalize
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok = True)

    def exp_idx(self, idx, node_attrs, data_dict_it, logger, include_input = True, include_output = True):
        if self.status == "built":
            no = 0
            for data_dict in data_dict_it:
                filename = self.path / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
                if not os.path.isfile(filename):
                    break
                with open(filename, 'rb') as f:
                    obj, train_, spec = pkl.load(f)
                # X key로 데이터 가져오기
                (train_X, train_v_X), valid_X = data_dict['X']
                sub_result = {'spec': spec, 'object': obj}
                if include_output:
                    if train_ is None:
                        train_result = obj.process(train_X)
                    else:
                        train_result = train_
                    if train_v_X is not None:
                        train_v_result = obj.process(train_v_X)
                    else:
                        train_v_result = None
                    if include_output:
                        sub_result['output_train'] = (train_result, train_v_result)
                        sub_result['output_valid'] = obj.process(valid_X)
                    if include_input:
                        sub_result['input'] = data_dict
                    yield sub_result
                no += 1
        elif self.status == "finalized":
            raise RuntimeError(f"Node is finalized and cannot be re-experimented")
        else:
            no = 0
            if include_output:
                objs_iter = _build_iter_output(node_attrs, data_dict_it, logger)
                for obj, train_, spec, data_dict, output_train, output_valid in objs_iter:
                    sub_result = {'spec': spec, 'object': obj}
                    sub_result['output_train'] = output_train
                    sub_result['output_valid'] = output_valid
                    if include_input:
                        sub_result['input'] = data_dict
                    yield sub_result
                    if self.status is None and (not self.finalize_after_exp):
                        filename = self.path / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
                        with open(filename, 'wb') as f:
                            pkl.dump((obj, train_, spec), f)
                    no += 1
            else:
                objs_iter = _build_iter(node_attrs, data_dict_it, logger)
                for obj, train_, spec, data_dict in objs_iter:
                    sub_result = {'spec': spec, 'object': obj}
                    if include_input:
                        sub_result['input'] = data_dict
                    yield sub_result
                    if self.status is None and (not self.finalize_after_exp):
                        filename = self.path / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
                        with open(filename, 'wb') as f:
                            pkl.dump((obj, train_, spec), f)
                    no += 1
    
    def end_exp(self):
        if self.finalize_after_exp:
            self.status = "finalized"
        else:
            self.status = "built"

    def get_objs(self, idx):
        if self.status != 'built':
            raise RuntimeError(f"must be built before accessing objects (status='{self.status}')")
        no = 0
        while True:
            filename = self.path / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
            if not os.path.isfile(filename):
                break
            with open(filename, 'rb') as f:
                obj, train_, spec = pkl.load(f)
            yield obj, train_, spec
            no += 1

    def finalize(self):
        self.status = 'finalized'
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
