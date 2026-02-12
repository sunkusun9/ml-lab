import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs._trainer import Trainer


class BadProcessor:
    __name__ = 'BadProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise ValueError("intentional error")
    def transform(self, X):
        pass


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
        sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
    )
    e.set_grp('scale', role='stage', processor=StandardScaler,
              method='transform', edges={'X': [(None, ['f1', 'f2', 'f3'])]})
    e.set_node('scaler', grp='scale')
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': [('scaler', None)], 'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt', grp='model')
    e.build()
    e.exp()
    return e


class TestAddTrainer:
    def test_default_splitter_same(self, exp):
        trainer = exp.add_trainer('t1')
        assert trainer.splitter is exp.sp_v
        assert trainer.splitter_params == exp.splitter_params

    def test_add_trainer_skip(self, exp):
        t1 = exp.add_trainer('t1')
        t2 = exp.add_trainer('t1', exist='skip')
        assert t1 is t2

    def test_add_trainer_error(self, exp):
        exp.add_trainer('t1')
        with pytest.raises(RuntimeError):
            exp.add_trainer('t1', exist='error')

    def test_splitter_same_with_params_raises(self, exp):
        with pytest.raises(ValueError):
            exp.add_trainer('t1', splitter='same', splitter_params={'y': 'target'})

    def test_custom_splitter(self, exp):
        sp = KFold(n_splits=5)
        trainer = exp.add_trainer('t1', splitter=sp)
        assert trainer.splitter is sp

    def test_no_splitter(self, exp):
        trainer = exp.add_trainer('t1', splitter=None)
        assert trainer.splitter is None
        assert trainer.split_indices is None
        assert trainer.get_n_splits() == 1


class TestSelectHead:
    def test_select_head_collects_upstream(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        assert 'dt' in trainer.selected_heads
        assert 'scaler' in trainer.selected_stages

    def test_select_head_multiple(self, exp):
        exp.set_grp('model2', role='head', processor=DecisionTreeClassifier,
                    method='predict',
                    edges={'X': [('scaler', None)], 'y': [(None, 'target')]},
                    params={'max_depth': 5, 'random_state': 0})
        exp.set_node('dt2', grp='model2')
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.select_head(['dt2'])
        assert 'dt' in trainer.selected_heads
        assert 'dt2' in trainer.selected_heads
        assert 'scaler' in trainer.selected_stages


class TestTrain:
    def test_train_basic(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        assert 'scaler' in trainer.node_objs
        assert trainer.node_objs['scaler'].status == 'built'
        assert 'dt' in trainer.node_objs
        assert trainer.node_objs['dt'].status == 'built'

    def test_train_skips_built(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        objs_before = dict(trainer.node_objs)
        trainer.train()
        assert trainer.node_objs['scaler'] is objs_before['scaler']
        assert trainer.node_objs['dt'] is objs_before['dt']

    def test_train_no_splitter(self, exp):
        trainer = exp.add_trainer('t_nosplit', splitter=None)
        trainer.select_head(['dt'])
        trainer.train()
        assert trainer.node_objs['scaler'].status == 'built'
        assert trainer.node_objs['dt'].status == 'built'

    def test_train_error(self, exp):
        exp.set_grp('bad', role='head', processor=BadProcessor,
                    method='transform',
                    edges={'X': [(None, ['f1'])]})
        exp.set_node('bad_node', grp='bad')
        trainer = exp.add_trainer('t_err')
        trainer.select_head(['bad_node'])
        trainer.train()
        assert trainer.node_objs['bad_node'].status == 'error'
        err = trainer.node_objs['bad_node'].error
        assert err['type'] == 'ValueError'
        assert 'intentional error' in err['message']

    def test_train_error_continues_other_nodes(self, exp):
        exp.set_grp('bad', role='head', processor=BadProcessor,
                    method='transform',
                    edges={'X': [(None, ['f1'])]})
        exp.set_node('bad_node', grp='bad')
        trainer = exp.add_trainer('t_mixed')
        trainer.select_head(['dt', 'bad_node'])
        trainer.train()
        assert trainer.node_objs['dt'].status == 'built'
        assert trainer.node_objs['bad_node'].status == 'error'

    def test_train_n_splits(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        assert trainer.get_n_splits() == 3
        trainer.train()
        objs = list(trainer.node_objs['dt'].get_obj())
        assert len(objs) == 3


class TestProcess:
    def test_process_yields_per_split(self, exp, sample_data):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        results = list(trainer.process(sample_data))
        assert len(results) == trainer.get_n_splits()

    def test_process_output_shape(self, exp, sample_data):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        for output in trainer.process(sample_data):
            assert output.get_shape()[0] == len(sample_data)


class TestResetNodes:
    def test_reset_clears_node_objs(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        trainer.reset_nodes(['scaler'])
        assert 'scaler' not in trainer.node_objs
        assert 'dt' not in trainer.node_objs

    def test_reset_allows_retrain(self, exp):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        trainer.reset_nodes(['dt'])
        assert 'dt' not in trainer.node_objs
        trainer.train()
        assert trainer.node_objs['dt'].status == 'built'


class TestSaveLoad:
    def test_save_load_roundtrip(self, exp, sample_data):
        trainer = exp.add_trainer('t1')
        trainer.select_head(['dt'])
        trainer.train()
        path = exp.path

        loaded_exp = Experimenter.load(path, sample_data)
        loaded_trainer = loaded_exp.trainers['t1']
        assert loaded_trainer.name == 't1'
        assert set(loaded_trainer.selected_stages) == set(trainer.selected_stages)
        assert set(loaded_trainer.selected_heads) == set(trainer.selected_heads)
        assert 'scaler' in loaded_trainer.node_objs
        assert 'dt' in loaded_trainer.node_objs

    def test_save_creates_file(self, exp):
        trainer = exp.add_trainer('t1')
        assert (trainer.path / '__trainer.pkl').exists()
