import os
import time
import uuid
import pickle as pkl

from ._node_processor import TransformProcessor, PredictProcessor


def _train_build(node_attrs, data_dict, logger):
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        fit_process = method == 'fit_transform'
        obj = TransformProcessor(
            node_attrs['name'], node_attrs['processor'],
            node_attrs['adapter'], node_attrs['params'], logger=logger,
        )
    elif method in ['predict', 'predict_proba', 'fit_predict']:
        fit_process = method == 'fit_predict'
        obj = PredictProcessor(
            node_attrs['name'], node_attrs['processor'],
            method, node_attrs['adapter'], node_attrs['params'], logger=logger,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    start_time = time.time()
    if fit_process:
        result = obj.fit_process(data_dict)
    else:
        obj.fit(data_dict)
        result = None
    elapsed_time = time.time() - start_time

    train_X, train_v_X = data_dict['X']
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': train_X.get_shape() if train_X is not None else None,
        'train_v_shape': train_v_X.get_shape() if train_v_X is not None else None,
    }
    return obj, result, info


class TrainStageObj:

    def __init__(self, path):
        self.path = path
        self.status = None
        self.error = None
        self.objs_ = None

    def load(self):
        if not os.path.isdir(self.path):
            self.objs_ = None
            return
        self.objs_ = {}
        split_idx = 0
        while True:
            filename = self.path / f'obj{split_idx}.pkl'
            if not os.path.isfile(filename):
                break
            with open(filename, 'rb') as f:
                self.objs_[split_idx] = pkl.load(f)
            split_idx += 1
        if self.objs_:
            self.status = 'built'

    def start_build(self):
        self.objs_ = {}
        os.makedirs(self.path, exist_ok=True)

    def build_split(self, split_idx, node_attrs, data_dict, logger):
        obj, result, info = _train_build(node_attrs, data_dict, logger)
        filename = self.path / f'obj{split_idx}.pkl'
        with open(filename, 'wb') as f:
            pkl.dump((obj, result, info), f)
        self.objs_[split_idx] = (obj, result, info)

    def end_build(self):
        self.status = 'built'

    def get_obj(self):
        for i in self.objs_:
            yield self.objs_[i]


class TrainHeadObj:

    def __init__(self, path):
        self.path = path
        self.status = None
        self.error = None

    def load(self):
        if not os.path.isdir(self.path):
            return
        if os.path.isfile(self.path / 'obj0.pkl'):
            self.status = 'built'

    def start_build(self):
        os.makedirs(self.path, exist_ok=True)

    def build_split(self, split_idx, node_attrs, data_dict, logger):
        obj, result, info = _train_build(node_attrs, data_dict, logger)
        filename = self.path / f'obj{split_idx}.pkl'
        with open(filename, 'wb') as f:
            pkl.dump((obj, result, info), f)

    def end_build(self):
        self.status = 'built'

    def get_obj(self):
        no = 0
        while True:
            filename = self.path / f'obj{no}.pkl'
            if not os.path.isfile(filename):
                break
            with open(filename, 'rb') as f:
                yield pkl.load(f)
            no += 1
