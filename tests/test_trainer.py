import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from mllabs._trainer import Trainer
from mllabs._pipeline import PipelineBuilder
from mllabs import Trial
from mllabs._cache import DataCache
from mllabs._data_wrapper import wrap


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
    p = PipelineBuilder(path=tmp_path / 'pipeline')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1, f2, f3}'})
    p.set_node('scaler', grp='scale')
    return p


DT_EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


def _dt(name='dt', params=None, method='predict'):
    return Trial(name, 'sklearn.tree.DecisionTreeClassifier', DT_EDGES,
                 method=method,
                 params=params if params is not None else {'max_depth': 3, 'random_state': 42})


def _bad_dt(name='bad_dt'):
    """Trial reading the 'bad_node' stage, so that stage gets selected/trained."""
    return Trial(name, 'sklearn.tree.DecisionTreeClassifier',
                 {'X': 'bad_node:(*)', 'y': '{target}'})


def _trials(*trials):
    return list(trials) or [_dt()]


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


class TestSelection:
    def test_stages_default_to_all_without_trials(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        assert trainer.selected_stages == ['scaler']
        assert trainer.trial_names() == []

    def test_set_trials_selects_them(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        assert trainer.trial_names() == ['dt']

    def test_set_trials_collects_upstream_stages(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('extra', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('unused', grp='extra')
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        assert trainer.selected_stages == ['scaler']   # 'unused' is not referenced

    def test_trials_filter_by_name(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials([_dt('b')])
        assert trainer.trial_names() == ['b']

    def test_trial_attrs_carry_trial_id(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        built = pipeline.build()
        assert trainer.trial_attrs()['dt']['serial'] == _dt().trial_id(built)

    def test_no_pipeline_raises_on_train(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            trainer.train()

    def test_no_pipeline_raises_on_to_inferencer(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            trainer.to_inferencer()

    def test_set_pipeline_persists_pkl(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        assert (trainer.path / 'pipeline.pkl').exists()

    def test_set_pipeline_resets_stale_nodes(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        assert trainer.get_status('scaler') == 'built'
        pipeline.set_node('scaler', grp='scale', exist='replace')
        trainer.set_pipeline(pipeline.build())
        assert trainer.get_status('scaler') is None


class TestTrain:
    def test_train_basic(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_skips_built(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        build_ids = {
            name: [fold.train_data_flows[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            for name in ['scaler', 'dt']
        }
        trainer.train()
        for name in ['scaler', 'dt']:
            for i, fold in enumerate(trainer.train_folds):
                assert fold.train_data_flows[0].get_info(name)['build_id'] == build_ids[name][i]

    def test_train_no_splitter(self, pipeline, sample_data):
        trainer = _make_trainer(pipeline, sample_data, None, name='t_nosplit')
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_error(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('bad', processor='mock.BadProcessor',
                         method='transform',
                         edges={'X': '{f1}'})
        pipeline.set_node('bad_node', grp='bad')
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_err')
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials(_bad_dt()))
        trainer.train()
        assert trainer.get_status('bad_node') == 'error'
        err = trainer.get_node_error('bad_node')
        assert err['type'] == 'ValueError'
        assert 'intentional error' in err['message']

    def test_train_error_continues_other_nodes(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('bad', processor='mock.BadProcessor',
                         method='transform',
                         edges={'X': '{f1}'})
        pipeline.set_node('bad_node', grp='bad')
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_mixed')
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials(_dt(), _bad_dt()))
        trainer.train()
        assert trainer.get_status('dt') == 'built'
        assert trainer.get_status('bad_node') == 'error'

    def test_train_n_splits(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        assert trainer.get_n_splits() == 3
        trainer.train()
        for fold in trainer.train_folds:
            assert fold.train_data_flows[0].status('dt') == 'built'

    def test_serial_in_info(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        expected_serial = _dt().trial_id(pipeline.build())
        for fold in trainer.train_folds:
            info = fold.train_data_flows[0].get_info('dt')
            assert info['node_serial'] == expected_serial

    def test_serial_mismatch_triggers_reset(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        build_ids_before = [
            fold.train_data_flows[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]

        trainer.set_trials(_trials(_dt(params={'max_depth': 5, 'random_state': 42})))
        trainer.train()

        build_ids_after = [
            fold.train_data_flows[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        assert build_ids_before != build_ids_after

    def test_serial_mismatch_stage_cascades(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        build_ids_before = {
            name: [fold.train_data_flows[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            for name in ['scaler', 'dt']
        }

        pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                         method='transform',
                         edges={'X': '{f1, f2, f3}'},
                         params={'with_std': False})
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        for name in ['scaler', 'dt']:
            build_ids_after = [fold.train_data_flows[0].get_info(name)['build_id'] for fold in trainer.train_folds]
            assert build_ids_before[name] != build_ids_after

    def test_no_serial_mismatch_skips_rebuild(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        build_ids_before = [
            fold.train_data_flows[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        trainer.train()

        build_ids_after = [
            fold.train_data_flows[0].get_info('dt')['build_id']
            for fold in trainer.train_folds
        ]
        assert build_ids_before == build_ids_after


class TestProcess:
    def test_process_yields_per_split(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        results = list(trainer.process(sample_data))
        assert len(results) == trainer.get_n_splits()

    def test_process_output_shape(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        for output in trainer.process(sample_data):
            assert output.get_shape()[0] == len(sample_data)


class TestResetNodes:
    def test_reset_clears_node_objs(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        trainer.reset_nodes(['scaler'])
        assert trainer.get_status('scaler') is None
        assert trainer.get_status('dt') is None

    def test_reset_allows_retrain(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        trainer.reset_nodes(['dt'])
        assert trainer.get_status('dt') is None
        trainer.train()
        assert trainer.get_status('dt') == 'built'


class TestSaveLoad:
    def test_save_creates_file(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        assert (trainer.path / '__trainer.pkl').exists()

    def test_save_load_roundtrip(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()
        path = trainer.path

        loaded = Trainer._load(path, data=None, cache=DataCache())
        assert loaded.name == 't1'
        assert loaded.get_status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'

    def test_load_does_not_restore_trials(self, pipeline, sample_data, sp_v):
        """Trials are not persisted — re-supply them with set_trials()."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        loaded = Trainer._load(trainer.path, data=None, cache=DataCache())
        assert loaded.trial_names() == []
        loaded.set_trials(_trials())
        assert loaded.trial_names() == ['dt']

    def test_load_restores_pipeline(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.set_trials(_trials())
        trainer.train()

        loaded = Trainer._load(trainer.path, data=None, cache=DataCache())
        assert loaded.pipeline is not None
        assert 'scaler' in loaded.pipeline.nodes
        assert loaded.selected_stages == ['scaler']
        loaded.reset_nodes(['scaler'])  # uses restored pipeline without needing set_pipeline again
        assert loaded.get_status('scaler') is None
