import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from mllabs._trainer import Trainer
from mllabs._pipeline import Pipeline
from mllabs._cache import DataCache
from mllabs._data_wrapper import wrap


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
def pipeline(tmp_path):
    p = Pipeline(path=tmp_path / 'pipeline')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', role='stage', processor=StandardScaler,
               method='transform', edges={'X': [(None, ['f1', 'f2', 'f3'])]})
    p.set_node('scaler', grp='scale')
    p.set_grp('model', role='head', processor=DecisionTreeClassifier,
               method='predict',
               edges={'X': [('scaler', None)], 'y': [(None, ['target'])]},
               params={'max_depth': 3, 'random_state': 42})
    p.set_node('dt', grp='model')
    return p


@pytest.fixture
def sp_v():
    return KFold(n_splits=3, shuffle=True, random_state=42)


def _make_trainer(pipeline, sample_data, splitter, name='t1', path=None):
    if path is None:
        path = pipeline._store.db_path.parent / '__trainers' / name
    return Trainer(name=name, data=wrap(sample_data), path=path,
                   splitter=splitter, splitter_params={}, cache=DataCache())


class TestTrainerConstruction:
    def test_no_splitter(self, pipeline, sample_data):
        trainer = _make_trainer(pipeline, sample_data, None)
        assert trainer.splitter is None
        assert trainer.get_n_splits() == 1


class TestSelectHead:
    def test_select_head_collects_upstream(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        assert 'dt' in trainer.selected_heads
        assert 'scaler' in trainer.selected_stages

    def test_select_head_multiple(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('model2', role='head', processor=DecisionTreeClassifier,
                         method='predict',
                         edges={'X': [('scaler', None)], 'y': [(None, ['target'])]},
                         params={'max_depth': 5, 'random_state': 0})
        pipeline.set_node('dt2', grp='model2')
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.select_head(pipeline, ['dt2'])
        assert 'dt' in trainer.selected_heads
        assert 'dt2' in trainer.selected_heads
        assert 'scaler' in trainer.selected_stages


class TestTrain:
    def test_train_basic(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_skips_built(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)

        build_ids = {
            name: [fold.artifact_stores[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            for name in ['scaler', 'dt']
        }
        trainer.train(pipeline)
        for name in ['scaler', 'dt']:
            for i, fold in enumerate(trainer.train_folds):
                assert fold.artifact_stores[0].get_info(name)['build_id'] == build_ids[name][i]

    def test_train_no_splitter(self, pipeline, sample_data):
        trainer = _make_trainer(pipeline, sample_data, None, name='t_nosplit')
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_error(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('bad', role='head', processor=BadProcessor,
                         method='transform',
                         edges={'X': [(None, ['f1'])]})
        pipeline.set_node('bad_node', grp='bad')
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_err')
        trainer.select_head(pipeline, ['bad_node'])
        trainer.train(pipeline)
        assert trainer.get_status('bad_node') == 'error'
        err = trainer.get_node_error('bad_node')
        assert err['type'] == 'ValueError'
        assert 'intentional error' in err['message']

    def test_train_error_continues_other_nodes(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('bad', role='head', processor=BadProcessor,
                         method='transform',
                         edges={'X': [(None, ['f1'])]})
        pipeline.set_node('bad_node', grp='bad')
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_mixed')
        trainer.select_head(pipeline, ['dt', 'bad_node'])
        trainer.train(pipeline)
        assert trainer.get_status('dt') == 'built'
        assert trainer.get_status('bad_node') == 'error'

    def test_train_n_splits(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        assert trainer.get_n_splits() == 3
        trainer.train(pipeline)
        for fold in trainer.train_folds:
            assert fold.artifact_stores[0].status('dt') == 'built'

    def test_serial_in_info(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        expected_serial = pipeline.nodes['dt'].serial
        for fold in trainer.train_folds:
            info = fold.artifact_stores[0].get_info('dt')
            assert info['node_serial'] == expected_serial

    def test_serial_mismatch_triggers_reset(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)

        build_ids_before = [
            fold.artifact_stores[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]

        pipeline.set_grp('model', role='head', processor=DecisionTreeClassifier,
                         method='predict',
                         edges={'X': [('scaler', None)], 'y': [(None, ['target'])]},
                         params={'max_depth': 5, 'random_state': 42})
        trainer.train(pipeline)

        build_ids_after = [
            fold.artifact_stores[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        assert build_ids_before != build_ids_after

    def test_serial_mismatch_stage_cascades(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)

        build_ids_before = {
            name: [fold.artifact_stores[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            for name in ['scaler', 'dt']
        }

        pipeline.set_grp('scale', role='stage', processor=StandardScaler,
                         method='transform',
                         edges={'X': [(None, ['f1', 'f2', 'f3'])]},
                         params={'with_std': False})
        trainer.train(pipeline)

        for name in ['scaler', 'dt']:
            build_ids_after = [fold.artifact_stores[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            assert build_ids_before[name] != build_ids_after

    def test_no_serial_mismatch_skips_rebuild(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)

        build_ids_before = [
            fold.artifact_stores[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        trainer.train(pipeline)

        build_ids_after = [
            fold.artifact_stores[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        assert build_ids_before == build_ids_after


class TestProcess:
    def test_process_yields_per_split(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        results = list(trainer.process(sample_data))
        assert len(results) == trainer.get_n_splits()

    def test_process_output_shape(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        for output in trainer.process(sample_data):
            assert output.get_shape()[0] == len(sample_data)


class TestResetNodes:
    def test_reset_clears_node_objs(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        trainer.reset_nodes(pipeline, ['scaler'])
        assert trainer.get_status('scaler') is None
        assert trainer.get_status('dt') is None

    def test_reset_allows_retrain(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        trainer.reset_nodes(pipeline, ['dt'])
        assert trainer.get_status('dt') is None
        trainer.train(pipeline)
        assert trainer.get_status('dt') == 'built'


class TestSaveLoad:
    def test_save_creates_file(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        assert (trainer.path / '__trainer.pkl').exists()

    def test_save_load_roundtrip(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.select_head(pipeline, ['dt'])
        trainer.train(pipeline)
        path = trainer.path

        loaded = Trainer._load(path, data=None, cache=DataCache())
        assert loaded.name == 't1'
        assert set(loaded.selected_stages) == set(trainer.selected_stages)
        assert set(loaded.selected_heads) == set(trainer.selected_heads)
        assert loaded.get_status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'
