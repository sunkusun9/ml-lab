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
from mllabs._logger import ProgressSessionLogger


class RecordingLogger(ProgressSessionLogger):
    """No-op progress logger that records session create/remove ids."""
    def __init__(self):
        super().__init__(level=[])
        self.created = []
        self.removed = []

    def create_session(self, session_id, **kwargs):
        self.created.append(session_id)
        return super().create_session(session_id, **kwargs)

    def remove_session(self, session_id):
        self.removed.append(session_id)
        return super().remove_session(session_id)


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


class NativeChatterStage:
    """Writes to OS-level fd 1/2 (like a native lib), bypassing Python stdout."""
    __name__ = 'NativeChatterStage'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        import os
        os.write(1, b'NATIVE_STDOUT_XYZ\n')
        os.write(2, b'NATIVE_STDERR_XYZ\n')
        return self
    def transform(self, X):
        return X


class WarnStage:
    """Emits a Python warning during fit (captured into info['warnings'])."""
    __name__ = 'WarnStage'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        import warnings
        warnings.warn("WORKER_WARN_ABC")
        return self
    def transform(self, X):
        return X


def _const_metric(y, p):
    return 0.5


class WarnPredictor:
    """Emits a Python warning during predict (i.e. at collector/process time)."""
    __name__ = 'WarnPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        import warnings
        warnings.warn("PREDICT_WARN_XYZ")
        return np.zeros(len(X), dtype=int)


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


class TestNJobsCap:
    """n_jobs is capped to the actual task count so no idle workers/progress bars."""

    def test_build_caps_worker_sessions_to_total(self, exp, pipeline):
        # 2 folds x 1 stage node = 2 tasks; request far more workers
        pipeline.set_grp('scale', role='stage', processor=StandardScaler,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('scaler', grp='scale')

        logger = RecordingLogger()
        exp.build(n_jobs=8, logger=logger)

        worker_sessions = [s for s in logger.created if s != 0]
        assert worker_sessions == [1, 2]                     # capped to total=2, not 8
        assert sorted(logger.created) == sorted(logger.removed)  # every bar torn down
        for flow in exp.outer_folds[0].train_data_flows:
            assert flow.status('scaler') == 'built'

    def test_exp_caps_worker_sessions_to_total(self, exp, pipeline):
        _setup_full(pipeline)
        exp.build(n_jobs=2)

        logger = RecordingLogger()
        exp.exp(n_jobs=8, logger=logger)                     # 2 folds x 1 head = 2 tasks

        worker_sessions = [s for s in logger.created if s != 0]
        assert worker_sessions == [1, 2]
        assert sorted(logger.created) == sorted(logger.removed)
        assert exp.get_status('dt') == 'built'


class TestWorkerLogCapture:
    """Multi-worker runs capture native stdout/stderr into per-worker log files."""

    def test_native_output_captured_to_worker_logs(self, exp, pipeline, capfd):
        pipeline.set_grp('nc', role='stage', processor=NativeChatterStage,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('nc_node', grp='nc')

        exp.build(n_jobs=2)

        # native chatter went to the worker log files, not the parent console
        out, err = capfd.readouterr()
        assert 'NATIVE_STDOUT_XYZ' not in out
        assert 'NATIVE_STDERR_XYZ' not in err

        logs = exp.get_worker_logs()
        assert logs, "expected per-worker log files"
        combined = '\n'.join(logs.values())
        assert 'NATIVE_STDOUT_XYZ' in combined
        assert 'NATIVE_STDERR_XYZ' in combined
        assert (exp.path / '__worker_logs').exists()

    def test_get_worker_logs_empty_without_multi_run(self, exp, pipeline):
        _setup_full(pipeline)
        exp.build(n_jobs=1)
        assert exp.get_worker_logs() == {}


class TestWorkerWarningVerbosity:
    """Worker warnings route to logger.warning: collected in warning_list and
    gated by the logger's level."""

    def test_warnings_collected_and_silenced_when_level_excludes_warning(self, exp, pipeline, capfd):
        from mllabs._logger import ProgressSessionLogger
        pipeline.set_grp('wn', role='stage', processor=WarnStage,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('wn_node', grp='wn')

        logger = ProgressSessionLogger(level=['info', 'progress'])  # no 'warning'
        exp.build(n_jobs=2, logger=logger)

        # collected in warning_list even though not printed
        assert any('WORKER_WARN_ABC' in w for w in logger.warning_list)
        out, err = capfd.readouterr()
        assert 'WORKER_WARN_ABC' not in out and 'WORKER_WARN_ABC' not in err

    def test_warnings_printed_when_level_includes_warning(self, exp, pipeline, capfd):
        from mllabs._logger import ProgressSessionLogger
        pipeline.set_grp('wn', role='stage', processor=WarnStage,
                          method='transform', edges={'X': '{f1}'})
        pipeline.set_node('wn_node', grp='wn')

        logger = ProgressSessionLogger(level=['info', 'warning', 'progress'])
        exp.build(n_jobs=2, logger=logger)

        out, err = capfd.readouterr()
        assert 'WORKER_WARN_ABC' in (out + err)

    def test_predict_warnings_routed_via_logger(self, exp, pipeline, capfd):
        # predict/collect-time warnings (like XGBoost device mismatch) now flow
        # through the logger with the node prefix, not raw to stderr.
        from mllabs._logger import ProgressSessionLogger
        from mllabs import Connector, MetricCollector
        pipeline.set_grp('wp', role='head', processor=WarnPredictor, method='predict',
                          edges={'X': '{f1, f2, f3}', 'y': '{target}'})
        pipeline.set_node('wp_node', grp='wp')
        exp.build()
        exp.set_collector('m', MetricCollector, Connector(),
                          params={'output_var': None, 'metric_func': _const_metric})

        logger = ProgressSessionLogger(level=['info', 'progress'])  # no 'warning'
        exp.exp(logger=logger)

        assert any('PREDICT_WARN_XYZ' in w for w in logger.warning_list)
        assert any('[wp_node]' in w for w in logger.warning_list)   # node prefix present
        out, err = capfd.readouterr()
        assert 'PREDICT_WARN_XYZ' not in out and 'PREDICT_WARN_XYZ' not in err


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
