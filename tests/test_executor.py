import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs._trainer import Trainer
from mllabs._pipeline import PipelineBuilder
from mllabs import Trial, TrialStore, Predictor
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


def _const_metric(y, p):
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
def pipeline(tmp_path):
    # Given a db, so build() mints a version — a Trainer refuses a Pipeline
    # that has none, since nothing could name what it trained against.
    p = PipelineBuilder(path=tmp_path / 'pipeline')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    return p


@pytest.fixture
def exp(tmp_path, sample_data, pipeline, trial_store):
    e = Experimenter(
        name='e1',
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        trial_store=trial_store,
    )
    e.set_pipeline(pipeline.build())
    return e


@pytest.fixture
def trial_store(tmp_path):
    return TrialStore(tmp_path / 'trials')


def _setup_full(pipeline, exp=None):
    pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                     method='transform', edges={'X': '{f1, f2, f3}'})
    pipeline.set_node('scaler', grp='scale')
    if exp is not None:
        exp.set_pipeline(pipeline.build())


DT_EDGES = {'X': 'scaler:(*)', 'y': '{target}'}
DT_PARAMS = {'max_depth': 3, 'random_state': 42}


def _dt(name='dt', edges=None):
    return Trial(name, 'sklearn.tree.DecisionTreeClassifier',
                 edges or DT_EDGES, params=dict(DT_PARAMS))


def _model():
    """One good Trial over the 'scaler' stage."""
    return [_dt()]


def _bad_edges():
    """A good Trial plus one whose edges fail to resolve at dispatch."""
    return [_dt(), _dt('bad_dt', {'X': 'scaler:([)', 'y': '{target}'})]


def _wp():
    return [
        Trial('wp_node', 'mock.WarnPredictor', {'X': '{f1, f2, f3}', 'y': '{target}'}),
    ]


def _names(trials, exp):
    """Register a bare Trial list and return the names Experimenter.exp takes.

    Each name covers every fold of *exp* — exp() expands the grid itself.
    Stamped with the run's own pipeline version, which Project.set_trial would
    otherwise do: exp() refuses a Trial defined against another version."""
    for t in trials:
        if t.pipeline_version is None:
            t.pipeline_version = exp.pipeline_version
    exp.trial_store.register_all(trials)
    return [t.name for t in trials]


def _stage_errored(exp, node_name):
    """Whether *node_name* (a Stage) has an 'error' row in this run's own
    NodeStore history — NodeStore.status()/Experimenter.get_status() can only
    ever see 'built' or None (obj.pkl existence), never 'error'; that's only
    recorded in node_hist now."""
    return any(r['status'] == 'error' for r in exp.node_store.get_hist(node_name=node_name))


def _trial_errored(trial_store, trial_name, exp):
    """Whether *trial_name* has an 'error' row in TrialStore.experiment_hist
    for this Experimenter — same reasoning as _stage_errored, but Trial
    history lives in TrialStore, not this run's NodeStore."""
    return any(s == 'error' for s in trial_store.get_status(trial_name, exp.name).values())


def _trial_built(trial_store, trial_name, exp):
    """Whether every fold of *exp* recorded *trial_name* as clean.

    A Trial leaves no artifact, so ``Experimenter.get_status``/
    ``NodeStore.status`` (both disk) can never answer this —
    ``experiment_hist`` is the whole record."""
    expected = {(o, i) for o in range(exp.get_n_splits())
                for i in range(exp.get_n_splits_inner())}
    status = trial_store.get_status(trial_name, exp.name)
    return set(status) == expected and set(status.values()) == {'built'}


class TestBuildFlowMulti:
    """Exercise _build_flow_multi's worker-pool dispatch (ProcessWorker, n_jobs>1)."""

    def test_builds_across_folds_and_reports_errors(self, exp, pipeline):
        pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('bad', processor='mock.BadProcessor',
                         method='transform', edges={'X': '{f2}'})
        pipeline.set_node('bad_node', grp='bad')

        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=2)

        assert exp.get_status('good_node') == 'built'
        assert _stage_errored(exp, 'bad_node')
        for outer_fold in exp.outer_folds:
            for flow in outer_fold.train_data_flows:
                assert flow.status('good_node') == 'built'

    def test_skips_already_built_nodes(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build(n_jobs=2)
        build_id = exp.node_store.get_info('scaler')[(0, 0)]['build_id']

        exp.build(n_jobs=2)

        assert exp.node_store.get_info('scaler')[(0, 0)]['build_id'] == build_id


class TestNJobsCap:
    """n_jobs is capped to the actual task count so no idle workers/progress bars."""

    def test_build_caps_worker_sessions_to_total(self, exp, pipeline):
        # 2 folds x 1 stage node = 2 tasks; request far more workers
        pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('scaler', grp='scale')

        logger = RecordingLogger()
        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=8, logger=logger)

        worker_sessions = [s for s in logger.created if s != 0]
        assert worker_sessions == [1, 2]                     # capped to total=2, not 8
        assert sorted(logger.created) == sorted(logger.removed)  # every bar torn down
        for flow in exp.outer_folds[0].train_data_flows:
            assert flow.status('scaler') == 'built'

    def test_exp_caps_worker_sessions_to_total(self, exp, pipeline, trial_store):
        _setup_full(pipeline, exp)
        exp.build(n_jobs=2)

        logger = RecordingLogger()
        exp.exp(_names(_model(), exp), n_jobs=8, logger=logger)  # 2 folds x 1 head = 2 tasks

        worker_sessions = [s for s in logger.created if s != 0]
        assert worker_sessions == [1, 2]
        assert sorted(logger.created) == sorted(logger.removed)
        assert _trial_built(trial_store, 'dt', exp)


class TestWorkerLogCapture:
    """Multi-worker runs capture native stdout/stderr into per-worker log
    files, only while OS log capture is open (open_os_log/os_log)."""

    def test_native_output_captured_to_worker_logs_when_open(self, exp, pipeline, capfd):
        pipeline.set_grp('nc', processor='mock.NativeChatterStage',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('nc_node', grp='nc')

        with exp.os_log():
            exp.set_pipeline(pipeline.build())
            exp.build(n_jobs=2)

        # native chatter went to the worker log files, not the parent console
        out, err = capfd.readouterr()
        assert 'NATIVE_STDOUT_XYZ' not in out
        assert 'NATIVE_STDERR_XYZ' not in err

        logs = exp.get_worker_logs()
        worker_logs = {k: v for k, v in logs.items() if k != 'master'}
        assert worker_logs, "expected per-worker log files"
        combined = '\n'.join(worker_logs.values())
        assert 'NATIVE_STDOUT_XYZ' in combined
        assert 'NATIVE_STDERR_XYZ' in combined
        assert (exp.path / '__worker_logs').exists()

    def test_native_output_not_captured_when_closed(self, exp, pipeline, capfd):
        pipeline.set_grp('nc', processor='mock.NativeChatterStage',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('nc_node', grp='nc')

        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=2)  # no os_log() — capture stays off, matches pre-existing behavior

        out, err = capfd.readouterr()
        assert 'NATIVE_STDOUT_XYZ' in out
        assert 'NATIVE_STDERR_XYZ' in err
        assert exp.get_worker_logs() == {}

    def test_get_worker_logs_empty_without_multi_run(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build(n_jobs=1)
        assert exp.get_worker_logs() == {}

    def test_single_worker_captured_by_master_log_when_open(self, exp, pipeline, capfd):
        pipeline.set_grp('nc', processor='mock.NativeChatterStage',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('nc_node', grp='nc')

        with exp.os_log():
            exp.set_pipeline(pipeline.build())
            exp.build(n_jobs=1)  # single-worker path — no build()-internal dup2, covered by the open master redirect

        out, err = capfd.readouterr()
        assert 'NATIVE_STDOUT_XYZ' not in out
        assert 'NATIVE_STDERR_XYZ' not in err

        master_log = exp.get_worker_logs('master')
        assert 'NATIVE_STDOUT_XYZ' in master_log
        assert 'NATIVE_STDERR_XYZ' in master_log

    def test_open_os_log_twice_raises(self, exp):
        exp.open_os_log()
        try:
            with pytest.raises(RuntimeError):
                exp.open_os_log()
        finally:
            exp.close_os_log()

    def test_close_os_log_without_open_is_noop(self, exp):
        exp.close_os_log()  # must not raise


class TestWorkerWarningVerbosity:
    """Worker warnings route to logger.warning: collected in warning_list and
    gated by the logger's level."""

    def test_warnings_collected_and_silenced_when_level_excludes_warning(self, exp, pipeline, capfd):
        from mllabs._logger import ProgressSessionLogger
        pipeline.set_grp('wn', processor='mock.WarnStage',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('wn_node', grp='wn')

        logger = ProgressSessionLogger(level=['info', 'progress'])  # no 'warning'
        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=2, logger=logger)

        # collected in warning_list even though not printed
        assert any('WORKER_WARN_ABC' in w for w in logger.warning_list)
        out, err = capfd.readouterr()
        assert 'WORKER_WARN_ABC' not in out and 'WORKER_WARN_ABC' not in err

    def test_warnings_printed_when_level_includes_warning(self, exp, pipeline, capfd):
        from mllabs._logger import ProgressSessionLogger
        pipeline.set_grp('wn', processor='mock.WarnStage',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('wn_node', grp='wn')

        logger = ProgressSessionLogger(level=['info', 'warning', 'progress'])
        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=2, logger=logger)

        out, err = capfd.readouterr()
        assert 'WORKER_WARN_ABC' in (out + err)

    def test_predict_warnings_routed_via_logger(self, exp, pipeline, capfd, tmp_path, trial_store):
        # predict/collect-time warnings (like XGBoost device mismatch) now flow
        # through the logger with the node prefix, not raw to stderr.
        from mllabs._logger import ProgressSessionLogger
        from mllabs import Connector, MetricCollector
        exp.set_pipeline(pipeline.build())
        exp.build()
        exp.collectors.set_collector('m', MetricCollector, Connector(),
                                     params={'output_var': None, 'metric_func': _const_metric})
        logger = ProgressSessionLogger(level=['info', 'progress'])  # no 'warning'
        exp.exp(_names(_wp(), exp), ['m'], logger=logger)

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
        pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('src', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f2}'})
        pipeline.set_node('src_node', grp='src')
        pipeline.set_grp('bad', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': self._BAD_X})
        pipeline.set_node('bad_node', grp='bad')

        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=1)

        assert exp.get_status('good_node') == 'built'
        assert exp.get_status('src_node') == 'built'
        assert _stage_errored(exp, 'bad_node')

    def test_build_multi_reports_error_and_continues(self, exp, pipeline):
        pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('src', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f2}'})
        pipeline.set_node('src_node', grp='src')
        pipeline.set_grp('bad', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': self._BAD_X})
        pipeline.set_node('bad_node', grp='bad')

        exp.set_pipeline(pipeline.build())
        exp.build(n_jobs=2)

        assert exp.get_status('good_node') == 'built'
        assert exp.get_status('src_node') == 'built'
        assert _stage_errored(exp, 'bad_node')

    def test_exp_single_reports_error_and_continues(self, exp, pipeline, trial_store):
        _setup_full(pipeline, exp)

        exp.build(n_jobs=1)
        exp.exp(_names(_bad_edges(), exp), n_jobs=1)

        assert _trial_built(trial_store, 'dt', exp)
        assert _trial_errored(trial_store, 'bad_dt', exp)

    def test_exp_multi_reports_error_and_continues(self, exp, pipeline, trial_store):
        _setup_full(pipeline, exp)

        exp.build(n_jobs=2)
        exp.exp(_names(_bad_edges(), exp), n_jobs=2)

        assert _trial_built(trial_store, 'dt', exp)
        assert _trial_errored(trial_store, 'bad_dt', exp)


class TestErrorKeyShape:
    """Both executors key errors by full job identity ``(outer_idx, inner_idx,
    name)``, for Trials as well as nodes. The Trial path used to collapse this
    to ``(outer_idx, name)`` on the way out, which dropped an error whenever
    the same trial failed on more than one inner fold of the same outer fold."""

    @pytest.fixture
    def exp_inner(self, tmp_path, sample_data, pipeline, trial_store):
        """Two outer folds x two inner folds — the layout the collapse lost."""
        e = Experimenter(
            name='e_inner',
            data=sample_data,
            path=tmp_path / 'exp_inner',
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=2),
            trial_store=trial_store,
        )
        e.set_pipeline(pipeline.build())
        return e

    def _bad_jobs(self, exp_inner, pipeline):
        _setup_full(pipeline, exp_inner)
        exp_inner.build(n_jobs=1)
        bad = _dt('bad_dt', {'X': 'scaler:([)', 'y': '{target}'})
        return exp_inner._make_jobs(_names([bad], exp_inner))

    _EXPECTED = {(o, i, 'bad_dt') for o in range(2) for i in range(2)}

    def test_single_keys_every_failed_job(self, exp_inner, pipeline):
        from mllabs._executor import _execute_single
        jobs = self._bad_jobs(exp_inner, pipeline)
        assert len(jobs) == 4
        errors = _execute_single(jobs, exp_inner.node_store, collectors=[])
        assert set(errors) == self._EXPECTED

    def test_multi_keys_every_failed_job(self, exp_inner, pipeline):
        from mllabs._executor import _execute_multi
        jobs = self._bad_jobs(exp_inner, pipeline)
        errors = _execute_multi(jobs, 2, exp_inner.node_store, collectors=[])
        assert set(errors) == self._EXPECTED


class TestExperimentMulti:
    """Exercise _experiment_multi's worker-pool dispatch (ProcessWorker, n_jobs>1)."""

    def test_runs_head_across_folds_and_reports_errors(self, exp, pipeline, trial_store):
        _setup_full(pipeline, exp)

        exp.build(n_jobs=2)
        exp.exp(_names(_bad_edges(), exp), n_jobs=2)

        assert _trial_built(trial_store, 'dt', exp)
        assert _trial_errored(trial_store, 'bad_dt', exp)


class TestTrainerMulti:
    """Trainer reuses _execute_multi over a different fold layout
    (TrainFold: single shared train_data_flow per split) — verify it's
    also compatible with n_jobs>1 dispatch."""

    def test_train_node_and_predictor_with_n_jobs(self, pipeline, sample_data, tmp_path):
        _setup_full(pipeline)
        t = Trainer(
            name='t1', data=wrap(sample_data), path=tmp_path / 'trainer',
            splitter=KFold(n_splits=2, shuffle=True, random_state=42),
            splitter_params={}, cache=DataCache(),
        )
        t.set_pipeline(pipeline.build())
        t.train([Predictor.from_trial(t_) for t_ in _model()], n_jobs=2)

        assert t.get_status('scaler') == 'built'
        assert t.get_status('dt') == 'built'


def _fatal_pipeline(pipeline, exp):
    """A stage whose fit kills its worker, alongside one that behaves."""
    pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                     method='transform', edges={'X': '{f1}'})
    pipeline.set_node('good_node', grp='good')
    pipeline.set_grp('fatal', processor='mock.SuicidalProcessor',
                     method='transform', edges={'X': '{f2}'})
    pipeline.set_node('fatal_node', grp='fatal')
    exp.set_pipeline(pipeline.build())


class TestWorkerDeath:
    """A worker dying used to escape the dispatch loop as an unhandled
    EOFError, which skipped the shutdown at the end of _execute_multi and left
    every other worker blocked in recv() for the life of the parent."""

    def test_the_run_terminates_and_records_the_lost_job(self, exp, pipeline):
        _fatal_pipeline(pipeline, exp)

        exp.build(n_jobs=2)

        assert _stage_errored(exp, 'fatal_node')
        errors = [r for r in exp.node_store.get_hist(node_name='fatal_node')
                  if r['status'] == 'error']
        assert len(errors) == exp.get_n_splits()
        assert any(r['info']['error']['type'] == 'WorkerLost' for r in errors)

    def test_no_worker_outlives_the_call(self, exp, pipeline):
        import multiprocessing

        _fatal_pipeline(pipeline, exp)
        before = {p.pid for p in multiprocessing.active_children()}

        exp.build(n_jobs=2)

        leaked = {p.pid for p in multiprocessing.active_children()} - before
        assert not leaked

    def test_a_clean_run_also_leaves_nothing_behind(self, exp, pipeline, trial_store):
        import multiprocessing

        _setup_full(pipeline, exp)
        before = {p.pid for p in multiprocessing.active_children()}

        exp.build(n_jobs=2)
        exp.exp(_names(_model(), exp), n_jobs=2)

        leaked = {p.pid for p in multiprocessing.active_children()} - before
        assert not leaked
        assert exp.get_status('scaler') == 'built'
