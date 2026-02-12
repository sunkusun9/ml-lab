import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllab._experimenter import Experimenter, DataCache
from mllab._expobj import StageObj, HeadObj
from mllab._pipeline import Pipeline
from mllab import Connector, MetricCollector


class BadProcessor:
    __name__ = 'BadProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise ValueError("intentional error")
    def transform(self, X):
        pass


class ErrorProcessor:
    __name__ = 'ErrorProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise TypeError("test error msg")
    def transform(self, X):
        pass


class BadPredictor:
    __name__ = 'BadPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise RuntimeError("predict error")
    def predict(self, X):
        pass


def accuracy_metric(y, pred):
    return (y.values == pred.values).mean()

def dummy_metric(y, pred):
    return 0.5


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
        'target': np.random.randint(0, 2, n),
    })


@pytest.fixture
def exp(tmp_path, sample_data):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    )
    return e


def _setup_stage(e):
    e.set_grp('scale', role='stage', processor=StandardScaler,
              method='transform', edges={'X': [(None, ['f1', 'f2', 'f3'])]})
    e.set_node('scaler', grp='scale')


def _setup_head(e):
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict', edges={'X': [(None, ['f1', 'f2', 'f3'])],
                                        'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt', grp='model')


def _setup_full(e):
    _setup_stage(e)
    _setup_head(e)


class TestDataCache:
    def test_put_get(self):
        c = DataCache(maxsize=1024**3)
        data = np.array([1, 2, 3])
        c.put_data('node1', 'all', 0, data)
        result = c.get_data('node1', 'all', 0)
        assert np.array_equal(result, data)

    def test_get_missing(self):
        c = DataCache(maxsize=1024**3)
        assert c.get_data('no_exist', 'all', 0) is None

    def test_clear_nodes(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 'all', 0, np.array([1]))
        c.put_data('b', 'all', 0, np.array([2]))
        c.clear_nodes(['a'])
        assert c.get_data('a', 'all', 0) is None
        assert c.get_data('b', 'all', 0) is not None

    def test_clear(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 'all', 0, np.array([1]))
        c.clear()
        assert c.get_data('a', 'all', 0) is None


class TestExperimenterInit:
    def test_path_created(self, exp):
        assert exp.path.exists()

    def test_data_wrapped(self, exp):
        assert exp.data is not None

    def test_splits_created(self, exp):
        assert exp.get_n_splits() == 2
        assert len(exp.valid_idx_list) == 2

    def test_no_inner_split(self, exp):
        assert exp.get_n_splits_inner() == 1
        train_idx, valid_v_idx = exp.train_idx_list[0][0]
        assert valid_v_idx is None

    def test_with_inner_split(self, tmp_path, sample_data):
        e = Experimenter(
            data=sample_data,
            path=tmp_path / 'exp_inner',
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        assert e.get_n_splits() == 2
        assert e.get_n_splits_inner() == 3
        train_idx, valid_v_idx = e.train_idx_list[0][0]
        assert valid_v_idx is not None

    def test_create_path_exists(self, tmp_path, sample_data):
        path = tmp_path / 'existing'
        path.mkdir()
        with pytest.raises(RuntimeError):
            Experimenter.create(data=sample_data, path=path)

    def test_status_open(self, exp):
        assert exp.status == 'open'

    def test_pipeline_empty(self, exp):
        assert len(exp.pipeline.grps) == 0

    def test_data_key(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk', data_key='test_key')
        assert e.data_key == 'test_key'


class TestPipelineDelegation:
    def test_set_grp(self, exp):
        r = exp.set_grp('g1', role='stage')
        assert r['result'] == 'new'
        assert 'g1' in exp.pipeline.grps

    def test_set_node(self, exp):
        exp.set_grp('g1', role='stage', processor=StandardScaler,
                    method='transform', edges={'X': [(None, None)]})
        r = exp.set_node('n1', grp='g1')
        assert r['result'] == 'new'
        assert 'n1' in exp.pipeline.nodes

    def test_remove_node(self, exp):
        _setup_stage(exp)
        exp.remove_node('scaler')
        assert 'scaler' not in exp.pipeline.nodes

    def test_remove_grp(self, exp):
        exp.set_grp('g1', role='stage')
        exp.remove_grp('g1')
        assert 'g1' not in exp.pipeline.grps

    def test_rename_grp(self, exp):
        exp.set_grp('old', role='stage')
        exp.rename_grp('old', 'new')
        assert 'new' in exp.pipeline.grps
        assert 'old' not in exp.pipeline.grps

    def test_closed_status_blocks_modification(self, exp):
        exp.close()
        with pytest.raises(RuntimeError):
            exp.set_grp('g1', role='stage')


class TestBuild:
    def test_build_stage(self, exp):
        _setup_stage(exp)
        exp.build()
        assert 'scaler' in exp.node_objs
        assert exp.node_objs['scaler'].status == 'built'

    def test_build_skips_built(self, exp):
        _setup_stage(exp)
        exp.build()
        n_objs_before = len(exp.node_objs)
        exp.build()
        assert len(exp.node_objs) == n_objs_before

    def test_build_error_continues(self, exp):
        exp.set_grp('good', role='stage', processor=StandardScaler,
                    method='transform', edges={'X': [(None, ['f1'])]})
        exp.set_node('good_node', grp='good')
        exp.set_grp('bad', role='stage', processor=BadProcessor,
                    method='transform', edges={'X': [(None, ['f2'])]})
        exp.set_node('bad_node', grp='bad')
        exp.build()
        assert exp.node_objs['good_node'].status == 'built'
        assert exp.node_objs['bad_node'].status == 'error'

    def test_build_error_dict(self, exp):
        exp.set_grp('err', role='stage', processor=ErrorProcessor,
                    method='transform', edges={'X': [(None, ['f1'])]})
        exp.set_node('err_node', grp='err')
        exp.build()
        err = exp.node_objs['err_node'].error
        assert err['type'] == 'TypeError'
        assert 'test error msg' in err['message']
        assert 'traceback' in err
        assert 'fold' in err


class TestExp:
    def test_exp_head(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        assert 'dt' in exp.node_objs
        assert exp.node_objs['dt'].status == 'built'

    def test_exp_skips_built(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        objs_before = dict(exp.node_objs)
        exp.exp()
        assert exp.node_objs['dt'] is objs_before['dt']

    def test_exp_error(self, exp):
        exp.set_grp('bad_model', role='head', processor=BadPredictor,
                    method='predict', edges={'X': [(None, ['f1'])],
                                              'y': [(None, 'target')]})
        exp.set_node('bad_dt', grp='bad_model')
        exp.exp()
        assert exp.node_objs['bad_dt'].status == 'error'

    def test_exp_with_collector(self, exp):
        _setup_full(exp)
        exp.build()
        mc = MetricCollector(
            'acc', Connector(),
            output_var=None,
            metric_func=accuracy_metric,
        )
        exp.add_collector(mc)
        exp.exp()
        assert mc.has('dt')


class TestCollectorManagement:
    def test_add_collector(self, exp):
        mc = MetricCollector('acc', Connector(),
                            output_var=None,
                            metric_func=dummy_metric)
        exp.add_collector(mc)
        assert 'acc' in exp.collectors
        assert mc.path is not None

    def test_add_collector_skip(self, exp):
        mc1 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        mc2 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        exp.add_collector(mc1)
        result = exp.add_collector(mc2, exist='skip')
        assert result is mc1

    def test_add_collector_error(self, exp):
        mc1 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        exp.add_collector(mc1)
        mc2 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        with pytest.raises(RuntimeError):
            exp.add_collector(mc2, exist='error')


class TestResetNodes:
    def test_reset_clears_node_objs(self, exp):
        _setup_stage(exp)
        exp.build()
        assert 'scaler' in exp.node_objs
        exp.reset_nodes(['scaler'])
        assert 'scaler' not in exp.node_objs

    def test_reset_clears_cache(self, exp):
        _setup_stage(exp)
        exp.build()
        exp.cache.put_data('scaler', 'all', 0, np.array([1]))
        exp.reset_nodes(['scaler'])
        assert exp.cache.get_data('scaler', 'all', 0) is None

    def test_set_node_replace_resets(self, exp):
        _setup_stage(exp)
        exp.build()
        assert 'scaler' in exp.node_objs
        exp.set_node('scaler', grp='scale', exist='replace')
        assert 'scaler' not in exp.node_objs


class TestStateManagement:
    def test_open_close(self, exp):
        exp.close()
        assert exp.status == 'close'
        exp.open()
        assert exp.status == 'open'

    def test_finalize_head(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        assert exp.node_objs['dt'].status == 'finalized'

    def test_reinitialize(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        exp.reinitialize(['dt'])
        assert 'dt' not in exp.node_objs


class TestSaveLoad:
    def test_save_creates_file(self, exp):
        _setup_stage(exp)
        assert (exp.path / '__exp.pkl').exists()

    def test_load_restores(self, exp, sample_data):
        _setup_full(exp)
        exp.build()
        exp.exp()
        path = exp.path

        loaded = Experimenter.load(path, sample_data)
        assert set(loaded.pipeline.grps.keys()) == set(exp.pipeline.grps.keys())
        assert 'scaler' in loaded.node_objs
        assert loaded.node_objs['scaler'].status == 'built'
        assert 'dt' in loaded.node_objs

    def test_load_data_key_mismatch(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk',
                        data_key='key_a')
        with pytest.raises(ValueError, match='data_key'):
            Experimenter.load(tmp_path / 'dk', sample_data, data_key='key_b')

    def test_load_preserves_splits(self, exp, sample_data):
        _setup_stage(exp)
        path = exp.path
        loaded = Experimenter.load(path, sample_data)
        assert loaded.get_n_splits() == exp.get_n_splits()
        assert loaded.get_n_splits_inner() == exp.get_n_splits_inner()


class TestPaths:
    def test_get_grp_path(self, exp):
        exp.set_grp('g1', role='stage')
        path = exp.get_grp_path('g1')
        assert path == exp.path / 'g1'

    def test_get_grp_path_nested(self, exp):
        exp.set_grp('parent', role='stage')
        exp.set_grp('child', role='stage', parent='parent')
        path = exp.get_grp_path('child')
        assert path == exp.path / 'parent' / 'child'

    def test_get_node_path(self, exp):
        _setup_stage(exp)
        path = exp.get_node_path('scaler')
        assert path == exp.path / 'scale' / 'scaler'


class TestExpObj:
    def test_stage_obj_lifecycle(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        assert obj.status is None
        obj.start_build()
        assert path.exists()
        obj.end_build()
        assert obj.status == 'built'

    def test_stage_obj_start_exp_finalize_rejected(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        with pytest.raises(ValueError):
            obj.start_exp(finalize=True)

    def test_stage_obj_exp_requires_built(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        with pytest.raises(RuntimeError):
            list(obj.exp_idx(0, {}, iter([]), None))

    def test_stage_obj_exp_finalized_rejected(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        obj.start_build()
        obj.end_build()
        obj.finalize()
        with pytest.raises(RuntimeError):
            list(obj.exp_idx(0, {}, iter([]), None))

    def test_stage_obj_end_exp(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        obj.start_build()
        obj.end_build()
        obj.start_exp()
        obj.end_exp()
        assert obj.status == 'built'

    def test_stage_obj_finalize(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        obj.start_build()
        obj.end_build()
        obj.finalize()
        assert obj.status == 'finalized'
        assert not path.exists()

    def test_head_obj_lifecycle(self, tmp_path):
        path = tmp_path / 'head_node'
        obj = HeadObj(path)
        assert obj.status is None
        obj.start_exp()
        assert path.exists()
        obj.end_exp()
        assert obj.status == 'built'

    def test_head_obj_finalize_after_exp(self, tmp_path):
        path = tmp_path / 'head_node'
        obj = HeadObj(path)
        obj.start_exp(finalize=True)
        obj.end_exp()
        assert obj.status == 'finalized'

    def test_head_obj_get_objs_requires_built(self, tmp_path):
        path = tmp_path / 'head_node'
        obj = HeadObj(path)
        with pytest.raises(RuntimeError):
            list(obj.get_objs(0))

    def test_stage_obj_load(self, tmp_path):
        path = tmp_path / 'stage_node'
        obj = StageObj(path)
        obj.load()
        assert obj.status == 'finalized'

    def test_head_obj_load(self, tmp_path):
        path = tmp_path / 'head_node'
        obj = HeadObj(path)
        obj.load()
        assert obj.status == 'finalized'
