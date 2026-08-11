import json
import pickle as pkl

import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from mllabs._pipeline import PipelineBuilder
from mllabs._trainer import Trainer
from mllabs._inferencer import Inferencer
from mllabs import Predictor
from mllabs._cache import DataCache
from mllabs._data_wrapper import unwrap, wrap
from mllabs._serialize import _ref_to_obj


DT = 'sklearn.tree.DecisionTreeClassifier'
DT_EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


def _make_trainer(pipeline, name, data, path, splitter=None):
    return Trainer(name=name, data=wrap(data), path=path,
                   splitter=splitter, splitter_params={}, cache=DataCache())


def _dt(name='dt', method='predict'):
    return Predictor(name, DT, DT_EDGES, method=method,
                     params={'max_depth': 3, 'random_state': 42})


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
    """Nodes only — the predicting end is a Predictor, outside the Pipeline."""
    p = PipelineBuilder(path=tmp_path / 'pipeline')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1, f2, f3}'})
    p.set_node('scaler', grp='scale')
    return p


@pytest.fixture
def trained_trainer(tmp_path, pipeline, sample_data):
    trainer = _make_trainer(pipeline, 't1', sample_data, tmp_path / 'trainer_t1',
                            splitter=KFold(n_splits=2, shuffle=True, random_state=0))
    trainer.set_pipeline(pipeline.build())
    trainer.train([_dt()])
    return trainer


class TestToInferencer:
    def test_basic_creation(self, trained_trainer):
        inf = trained_trainer.to_inferencer()
        assert isinstance(inf, Inferencer)
        assert inf.n_splits == trained_trainer.get_n_splits()
        assert inf.selected_nodes == trained_trainer.selected_nodes
        assert inf.selected_predictors == trained_trainer.predictor_names()

    def test_node_objs_are_processor_lists(self, trained_trainer):
        inf = trained_trainer.to_inferencer()
        for name, objs in inf.node_objs.items():
            assert isinstance(objs, list)
            assert len(objs) == inf.n_splits
            assert hasattr(objs[0], 'process')

    def test_carries_specs_not_a_pipeline(self, trained_trainer):
        """Only ``edges`` is needed at serve time, so an Inferencer holds
        ProcessorSpecs for both kinds rather than a whole Pipeline."""
        inf = trained_trainer.to_inferencer()
        assert set(inf.node_specs) == {'scaler', 'dt'}
        assert inf.node_specs['dt'].edges == DT_EDGES
        assert not hasattr(inf, 'pipeline')

    def test_not_trained_raises(self, tmp_path, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_no_train', sample_data, tmp_path / 'trainer_t_no_train')
        trainer.set_pipeline(pipeline.build())
        # Registered but never trained — train() is the only path that does
        # both, so the definition goes in directly.
        trainer.predictor_defs.register_all([_dt()])
        with pytest.raises(RuntimeError, match="not built"):
            trainer.to_inferencer()

    def test_v_stored(self, trained_trainer):
        inf = trained_trainer.to_inferencer(v='0')
        assert inf.v == '0'

    def test_trainer_spec_stamped(self, trained_trainer):
        spec = trained_trainer.to_inferencer().trainer_spec
        assert spec['name'] == 't1'
        assert spec['pipeline_version'] == trained_trainer.pipeline_version
        assert spec['n_splits'] == trained_trainer.get_n_splits()
        assert _ref_to_obj(spec['splitter']) is KFold
        assert spec['splitter_params'] == {}

    def test_trainer_spec_is_json_serializable(self, trained_trainer):
        """Strings and primitives only — a deployed pickle says where it came
        from without carrying a Trainer or a splitter object."""
        spec = trained_trainer.to_inferencer().trainer_spec
        assert json.loads(json.dumps(spec)) == spec

    def test_trainer_spec_without_splitter(self, tmp_path, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_full', sample_data, tmp_path / 'trainer_t_full')
        trainer.set_pipeline(pipeline.build())
        trainer.train([_dt()])
        spec = trainer.to_inferencer().trainer_spec
        assert spec['splitter'] is None
        assert spec['n_splits'] == 1


class TestProcess:
    def test_mean_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mean')
        assert result.shape[0] == len(sample_data)

    def test_mode_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mode')
        assert result.shape[0] == len(sample_data)

    def test_callable_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg=lambda results: results[0])
        assert result.shape[0] == len(sample_data)

    def test_none_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        results = inf.process(sample_data, agg=None)
        assert isinstance(results, list)
        assert len(results) == inf.n_splits

    def test_v_parameter(self, tmp_path, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_proba', sample_data, tmp_path / 'trainer_t_proba')
        trainer.set_pipeline(pipeline.build())
        trainer.train([_dt('dt_proba', method='predict_proba')])
        inf = trainer.to_inferencer(v='-1:')
        result = inf.process(sample_data)
        assert result.shape[1] == 1

    def test_single_split(self, tmp_path, pipeline, sample_data):
        trainer = _make_trainer(pipeline, 't_nosplit', sample_data,
                                tmp_path / 'trainer_nosplit', splitter=None)
        trainer.set_pipeline(pipeline.build())
        trainer.train([_dt()])
        inf = trainer.to_inferencer()
        result = inf.process(sample_data)
        assert result.shape[0] == len(sample_data)

    def test_unknown_agg_raises(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        with pytest.raises(ValueError, match="Unknown agg"):
            inf.process(sample_data, agg='unknown')


class TestSaveLoad:
    def test_save_load_roundtrip(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.n_splits == inf.n_splits
        assert loaded.selected_nodes == inf.selected_nodes
        assert loaded.selected_predictors == inf.selected_predictors
        assert set(loaded.node_objs.keys()) == set(inf.node_objs.keys())

    def test_loaded_process_matches(self, trained_trainer, sample_data, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        original = inf.process(sample_data, agg=None)
        loaded_result = loaded.process(sample_data, agg=None)

        assert len(original) == len(loaded_result)
        for orig, load in zip(original, loaded_result):
            np.testing.assert_array_equal(unwrap(orig), unwrap(load))

    def test_save_creates_file(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)
        assert (save_path / '__inferencer.pkl').exists()

    def test_save_load_with_v(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer(v='0')
        save_path = tmp_path / 'inferencer_v'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.v == '0'

    def test_save_load_with_trainer_spec(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer_spec'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.trainer_spec == inf.trainer_spec

    def test_load_without_trainer_spec(self, trained_trainer, tmp_path):
        """Pickles written before provenance existed still load."""
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer_old'
        inf.save(save_path)
        with open(save_path / '__inferencer.pkl', 'rb') as f:
            save_data = pkl.load(f)
        del save_data['trainer_spec']
        with open(save_path / '__inferencer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

        assert Inferencer.load(save_path).trainer_spec is None
