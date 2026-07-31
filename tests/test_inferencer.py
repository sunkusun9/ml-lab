import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._pipeline import PipelineBuilder
from mllabs._experimenter import Experimenter
from mllabs._trainer import Trainer
from mllabs._inferencer import Inferencer
from mllabs._cache import DataCache
from mllabs._data_wrapper import unwrap, wrap


def _make_trainer(pipeline, name, data, path, splitter=None):
    return Trainer(name=name, data=wrap(data), path=path,
                   splitter=splitter, splitter_params={}, cache=DataCache())


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
    p.set_grp('model', processor='sklearn.tree.DecisionTreeClassifier',
              method='predict',
              edges={'X': 'scaler:(*)', 'y': '{target}'},
              params={'max_depth': 3, 'random_state': 42})
    p.set_node('dt', grp='model')
    return p


@pytest.fixture
def exp(tmp_path, sample_data, pipeline):
    exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_main',
                           sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
                           sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
                           pipeline=pipeline.build())
    exp_obj.build()
    exp_obj.exp()
    return exp_obj


@pytest.fixture
def trained_trainer(tmp_path, exp, pipeline, sample_data):
    trainer = _make_trainer(pipeline, 't1', sample_data, tmp_path / 'trainer_t1',
                            splitter=KFold(n_splits=2, shuffle=True, random_state=0))
    trainer.set_pipeline(pipeline.build())
    trainer.train()
    return trainer


class TestToInferencer:
    def test_basic_creation(self, trained_trainer, pipeline):
        inf = trained_trainer.to_inferencer()
        assert isinstance(inf, Inferencer)
        assert inf.n_splits == trained_trainer.get_n_splits()
        assert inf.selected_stages == trained_trainer.selected_stages
        assert inf.selected_heads == trained_trainer.selected_heads

    def test_node_objs_are_processor_lists(self, trained_trainer, pipeline):
        inf = trained_trainer.to_inferencer()
        for name, objs in inf.node_objs.items():
            assert isinstance(objs, list)
            assert len(objs) == inf.n_splits
            assert hasattr(objs[0], 'process')

    def test_minimal_pipeline(self, trained_trainer, pipeline):
        inf = trained_trainer.to_inferencer()
        assert 'scaler' in inf.pipeline.nodes
        assert 'dt' in inf.pipeline.nodes

    def test_not_trained_raises(self, tmp_path, exp, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_no_train', sample_data, tmp_path / 'trainer_t_no_train')
        trainer.set_pipeline(pipeline.build())
        with pytest.raises(RuntimeError, match="not built"):
            trainer.to_inferencer()

    def test_v_stored(self, trained_trainer, pipeline):
        inf = trained_trainer.to_inferencer(v='0')
        assert inf.v == '0'


class TestProcess:
    def test_mean_agg(self, trained_trainer, pipeline, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mean')
        assert result.shape[0] == len(sample_data)

    def test_mode_agg(self, trained_trainer, pipeline, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mode')
        assert result.shape[0] == len(sample_data)

    def test_callable_agg(self, trained_trainer, pipeline, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg=lambda results: results[0])
        assert result.shape[0] == len(sample_data)

    def test_none_agg(self, trained_trainer, pipeline, sample_data):
        inf = trained_trainer.to_inferencer()
        results = inf.process(sample_data, agg=None)
        assert isinstance(results, list)
        assert len(results) == inf.n_splits

    def test_v_parameter(self, tmp_path, exp, pipeline, sample_data):
        pipeline.set_grp('model_proba', processor='sklearn.tree.DecisionTreeClassifier',
                    method='predict_proba',
                    edges={'X': 'scaler:(*)', 'y': '{target}'},
                    params={'max_depth': 3, 'random_state': 42})
        pipeline.set_node('dt_proba', grp='model_proba', tag=['proba'])
        exp.build()
        exp.exp()
        trainer = _make_trainer(pipeline, 't_proba', sample_data, tmp_path / 'trainer_t_proba')
        trainer.set_pipeline(pipeline.build())
        trainer.train()
        inf = trainer.to_inferencer(v='-1:')
        result = inf.process(sample_data)
        assert result.shape[1] == 1

    def test_single_split(self, tmp_path, exp, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_nosplit', sample_data, tmp_path / 'trainer_nosplit', splitter=None)
        trainer.set_pipeline(pipeline.build())
        trainer.train()
        inf = trainer.to_inferencer()
        result = inf.process(sample_data)
        assert result.shape[0] == len(sample_data)

    def test_unknown_agg_raises(self, trained_trainer, pipeline, sample_data):
        inf = trained_trainer.to_inferencer()
        with pytest.raises(ValueError, match="Unknown agg"):
            inf.process(sample_data, agg='unknown')


class TestSaveLoad:
    def test_save_load_roundtrip(self, trained_trainer, pipeline, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.n_splits == inf.n_splits
        assert loaded.selected_stages == inf.selected_stages
        assert loaded.selected_heads == inf.selected_heads
        assert set(loaded.node_objs.keys()) == set(inf.node_objs.keys())

    def test_loaded_process_matches(self, trained_trainer, pipeline, sample_data, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        original = inf.process(sample_data, agg=None)
        loaded_result = loaded.process(sample_data, agg=None)

        assert len(original) == len(loaded_result)
        for orig, load in zip(original, loaded_result):
            np.testing.assert_array_equal(unwrap(orig), unwrap(load))

    def test_save_creates_file(self, trained_trainer, pipeline, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)
        assert (save_path / '__inferencer.pkl').exists()

    def test_save_load_with_v(self, trained_trainer, pipeline, sample_data, tmp_path):
        inf = trained_trainer.to_inferencer(v='0')
        save_path = tmp_path / 'inferencer_v'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.v == '0'
