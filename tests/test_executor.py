import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
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


class BadPredictor:
    __name__ = 'BadPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise RuntimeError("predict error")
    def predict(self, X):
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
def pipeline():
    p = Pipeline()
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    return p


@pytest.fixture
def exp(tmp_path, sample_data, pipeline):
    return Experimenter(
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        pipeline=pipeline,
    )


def _setup_full(pipeline):
    pipeline.set_grp('scale', role='stage', processor=StandardScaler,
                      method='transform', edges={'X': '{f1, f2, f3}'})
    pipeline.set_node('scaler', grp='scale')
    pipeline.set_grp('model', role='head', processor=DecisionTreeClassifier,
                      method='predict', edges={'X': 'scaler:(*)', 'y': '{target}'},
                      params={'max_depth': 3, 'random_state': 42})
    pipeline.set_node('dt', grp='model')


class TestBuildFlowMulti:
    """Exercise _build_flow_multi's worker-pool dispatch (ProcessWorker, n_jobs>1)."""

    def test_builds_across_folds_and_reports_errors(self, exp, pipeline):
        pipeline.set_grp('good', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('bad', role='stage', processor=BadProcessor,
                          method='transform', edges={'X': '{f2}'})
        pipeline.set_node('bad_node', grp='bad')

        exp.build(n_jobs=2)

        assert exp.get_status('good_node') == 'built'
        assert exp.get_status('bad_node') == 'error'
        for outer_fold in exp.outer_folds:
            for flow in outer_fold.train_data_flows:
                assert flow.status('good_node') == 'built'

    def test_skips_already_built_nodes(self, exp, pipeline):
        _setup_full(pipeline)
        exp.build(n_jobs=2)
        build_id = exp.outer_folds[0].train_data_flows[0].get_info('scaler')['build_id']

        exp.build(n_jobs=2)

        assert exp.outer_folds[0].train_data_flows[0].get_info('scaler')['build_id'] == build_id


class TestDataPrepErrors:
    """get_train/get_valid/get_test_data can themselves raise (e.g. a malformed
    regex pattern, or edges pick overlapping/duplicate columns) *before* a node
    ever reaches a worker. This must be reported like a fit-time error (tracker +
    node status='error') instead of crashing the whole build()/exp() run."""

    # 'src_node:([)' is an intentionally malformed regex ('[' unterminated) —
    # re.match raises at column-resolution time, before the node ever fits.
    _BAD_X = "src_node:([)"

    def test_build_single_reports_error_and_continues(self, exp, pipeline):
        pipeline.set_grp('good', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('src', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f2}'})
        pipeline.set_node('src_node', grp='src')
        pipeline.set_grp('bad', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': self._BAD_X})
        pipeline.set_node('bad_node', grp='bad')

        exp.build(n_jobs=1)

        assert exp.get_status('good_node') == 'built'
        assert exp.get_status('src_node') == 'built'
        assert exp.get_status('bad_node') == 'error'

    def test_build_multi_reports_error_and_continues(self, exp, pipeline):
        pipeline.set_grp('good', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('src', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f2}'})
        pipeline.set_node('src_node', grp='src')
        pipeline.set_grp('bad', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': self._BAD_X})
        pipeline.set_node('bad_node', grp='bad')

        exp.build(n_jobs=2)

        assert exp.get_status('good_node') == 'built'
        assert exp.get_status('src_node') == 'built'
        assert exp.get_status('bad_node') == 'error'

    def test_exp_single_reports_error_and_continues(self, exp, pipeline):
        _setup_full(pipeline)
        pipeline.set_grp('bad_model', role='head', processor=DecisionTreeClassifier,
                          method='predict', edges={'X': "scaler:([)", 'y': '{target}'},
                          params={'max_depth': 3, 'random_state': 42})
        pipeline.set_node('bad_dt', grp='bad_model')

        exp.build(n_jobs=1)
        exp.exp(n_jobs=1)

        assert exp.get_status('dt') == 'built'
        assert exp.get_status('bad_dt') == 'error'

    def test_exp_multi_reports_error_and_continues(self, exp, pipeline):
        _setup_full(pipeline)
        pipeline.set_grp('bad_model', role='head', processor=DecisionTreeClassifier,
                          method='predict', edges={'X': "scaler:([)", 'y': '{target}'},
                          params={'max_depth': 3, 'random_state': 42})
        pipeline.set_node('bad_dt', grp='bad_model')

        exp.build(n_jobs=2)
        exp.exp(n_jobs=2)

        assert exp.get_status('dt') == 'built'
        assert exp.get_status('bad_dt') == 'error'
        for outer_fold in exp.outer_folds:
            for store in outer_fold.artifact_stores:
                assert store.status('dt') == 'built'


class TestExperimentMulti:
    """Exercise _experiment_multi's worker-pool dispatch (ProcessWorker, n_jobs>1)."""

    def test_runs_head_across_folds_and_reports_errors(self, exp, pipeline):
        _setup_full(pipeline)
        pipeline.set_grp('bad_model', role='head', processor=BadPredictor,
                          method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pipeline.set_node('bad_dt', grp='bad_model')

        exp.build(n_jobs=2)
        exp.exp(n_jobs=2)

        assert exp.get_status('dt') == 'built'
        assert exp.get_status('bad_dt') == 'error'
        for outer_fold in exp.outer_folds:
            for store in outer_fold.artifact_stores:
                assert store.status('dt') == 'built'


class TestTrainerMulti:
    """Trainer reuses _build_flow_multi/_experiment_multi over a different fold layout
    (TrainFold: single shared train_data_flow/artifact_store per split) — verify it's
    also compatible with n_jobs>1 dispatch."""

    def test_train_stage_and_head_with_n_jobs(self, pipeline, sample_data, tmp_path):
        _setup_full(pipeline)
        t = Trainer(
            name='t1', data=wrap(sample_data), path=tmp_path / 'trainer',
            splitter=KFold(n_splits=2, shuffle=True, random_state=42),
            splitter_params={}, cache=DataCache(),
        )
        t.set_pipeline(pipeline)

        t.train(n_jobs=2)

        assert t.get_status('scaler') == 'built'
        assert t.get_status('dt') == 'built'
