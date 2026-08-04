import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs._cache import DataCache
from mllabs._store import NodeStore
from mllabs._flow import DataFlow
from mllabs import Project, Trial, Connector, MetricCollector, Collectors

TREE = 'sklearn.tree.DecisionTreeClassifier'
DT_EDGES = {'X': '{f1, f2, f3}', 'y': '{target}'}


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
def project(tmp_path):
    return Project(tmp_path / 'proj')


@pytest.fixture
def pipeline(project):
    p = project.pipeline_builder('main')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    return p


@pytest.fixture
def exp(project, sample_data, pipeline):
    version = project.build_pipeline(pipeline).version
    return project.experimenter(
        'e1', sample_data,
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        pipeline_name='main', pipeline_version=version,
    )


def _publish(pipeline, exp):
    """Hand the current definitions to *exp* as a fresh, versioned snapshot.

    Goes through the builder's own version counter (same as
    ``Project.build_pipeline`` — ``builder._store.save_version(pipeline)``,
    no ``project`` object actually needed for that part) rather than an
    in-memory unversioned build: ``set_pipeline`` reads ``pipeline.version``
    straight off the object, and an Experimenter persists
    ``(pipeline_name, pipeline_version)`` as its reload pointer, so an
    unversioned (``version=None``) build would silently break
    ``project.load_experimenter`` for any test that reloads afterward.

    Experimenter holds a built Pipeline, so edits made to the builder after
    construction only take effect once they are built and set again.
    """
    if exp is not None:
        built = pipeline.build()
        pipeline._store.save_version(built)
        exp.set_pipeline(built)


def _setup_stage(pipeline, exp=None):
    pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                      method='transform', edges={'X': '{f1, f2, f3}'})
    pipeline.set_node('scaler', grp='scale')
    _publish(pipeline, exp)


def _dt(name='dt', edges=None):
    return Trial(name, TREE, edges or DT_EDGES, params={'max_depth': 3, 'random_state': 42})


def _folds(trial, exp):
    """(trial, outer_idx, inner_idx) for every fold of *exp* — what Experimenter.exp expects."""
    return [(trial, o, i) for o in range(exp.get_n_splits()) for i in range(exp.get_n_splits_inner())]


def _flow(exp, outer=0, inner=0):
    return exp.outer_folds[outer].train_data_flows[inner]


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


class TestDataCache:
    def test_put_get(self):
        c = DataCache(maxsize=1024**3)
        data = np.array([1, 2, 3])
        c.put_data('scope1', 'node1', 'train', data)
        result = c.get_data('scope1', 'node1', 'train')
        assert np.array_equal(result, data)

    def test_get_missing(self):
        c = DataCache(maxsize=1024**3)
        assert c.get_data('scope1', 'no_exist', 'train') is None

    def test_clear_nodes(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('scope1', 'a', 'train', np.array([1]))
        c.put_data('scope1', 'b', 'train', np.array([2]))
        c.clear_nodes(['a'])
        assert c.get_data('scope1', 'a', 'train') is None
        assert c.get_data('scope1', 'b', 'train') is not None

    def test_clear(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('scope1', 'a', 'train', np.array([1]))
        c.clear()
        assert c.get_data('scope1', 'a', 'train') is None


class TestExperimenterInit:
    def test_path_created(self, exp):
        assert exp.path.exists()

    def test_data_wrapped(self, exp):
        assert exp.data is not None

    def test_splits_created(self, exp):
        assert exp.get_n_splits() == 2
        assert len(exp.outer_folds) == 2
        assert all(of.test_idx is not None for of in exp.outer_folds)

    def test_no_inner_split(self, exp):
        assert exp.get_n_splits_inner() == 1
        flow = _flow(exp)
        assert flow.data_source.valid_idx is None

    def test_with_inner_split(self, tmp_path, sample_data):
        e = Experimenter(
            path=tmp_path / 'exp_inner', name='e_inner', data=sample_data,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        assert e.get_n_splits() == 2
        assert e.get_n_splits_inner() == 3
        flow = e.outer_folds[0].train_data_flows[0]
        assert flow.data_source.valid_idx is not None

    def test_pipeline_empty(self, exp, pipeline):
        user_grps = {k: v for k, v in pipeline.grps.items() if not k.startswith('__')}
        assert len(user_grps) == 0

    def test_data_key(self, tmp_path, sample_data):
        e = Experimenter(path=tmp_path / 'dk', name='e_dk', data=sample_data, data_key='test_key')
        assert e.data_key == 'test_key'

    def test_accepts_already_wrapped_data(self, tmp_path, sample_data):
        """The mirror of Trainer's native-data case: the splitter is fed
        ``data_native``, which has to be unwrapped whichever form came in."""
        from mllabs._data_wrapper import wrap
        e = Experimenter(
            path=tmp_path / 'wrapped', name='e_wrapped', data=wrap(sample_data),
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        )
        assert e.get_n_splits() == 2
        assert e.data.get_shape()[0] == len(sample_data)


class TestBuild:
    def test_build_stage(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'

    def test_build_skips_built(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        build_id = exp.node_store.get_info('scaler')[(0, 0)]['build_id']
        exp.build()
        assert exp.node_store.get_info('scaler')[(0, 0)]['build_id'] == build_id

    def test_build_error_continues(self, exp, pipeline):
        pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('bad', processor='mock.BadProcessor',
                         method='transform', edges={'X': '{f2}'})
        pipeline.set_node('bad_node', grp='bad')
        _publish(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('good_node') == 'built'
        assert _stage_errored(exp, 'bad_node')

    def test_build_error_dict(self, exp, pipeline):
        pipeline.set_grp('err', processor='mock.ErrorProcessor',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('err_node', grp='err')
        _publish(pipeline, exp)
        exp.build()
        info = exp.node_store.get_info('err_node')[(0, 0)]
        err = info['error']
        assert err['type'] == 'TypeError'
        assert 'test error msg' in err['message']
        assert 'traceback' in err

    def test_unknown_column_in_edges_errors_at_build_not_definition(self, exp, pipeline):
        # set_grp only validates DSL structure — an edges string referencing a
        # column that doesn't exist in the schema is accepted at definition
        # time and only surfaces as a node-level error once build() actually
        # resolves it against real data.
        pipeline.set_grp('bad', processor='sklearn.preprocessing.StandardScaler',
                          method='transform', edges={'X': '{unknown_col}'})
        pipeline.set_node('bad_node', grp='bad')
        _publish(pipeline, exp)
        exp.build()
        assert _stage_errored(exp, 'bad_node')
        info = exp.node_store.get_info('bad_node')[(0, 0)]
        assert 'Unknown column' in info['error']['message']


class TestExp:
    def test_exp_head(self, exp, pipeline, project):
        trial = _dt()
        exp.exp(_folds(trial, exp), project.trials)
        assert exp.get_status('dt') == 'built'

    def test_exp_skips_built(self, exp, pipeline, project):
        trial = _dt()
        exp.exp(_folds(trial, exp), project.trials)
        build_id = project.trials.get_info('dt', exp.name)[(0, 0)]['build_id']
        exp.exp(_folds(trial, exp), project.trials)
        assert project.trials.get_info('dt', exp.name)[(0, 0)]['build_id'] == build_id

    def test_exp_error(self, exp, pipeline, project):
        bad_trial = Trial('bad_dt', 'mock.BadPredictor', {'X': '{f1}', 'y': '{target}'})
        exp.exp(_folds(bad_trial, exp), project.trials)
        assert _trial_errored(project.trials, 'bad_dt', exp)

    def test_exp_with_collector(self, exp, pipeline, project):
        trial = _dt()
        mc = project.collectors().set_collector(
            'acc', MetricCollector, Connector(),
            params={'output_var': None, 'metric_func': accuracy_metric},
        )
        exp.exp(_folds(trial, exp), project.trials, collectors=[mc])
        assert mc.has_node('dt')

    def test_set_collector_resolves_callable_metric(self, exp, pipeline, project):
        from sklearn.metrics import balanced_accuracy_score
        trial = _dt()
        mc = project.collectors().set_collector(
            'bacc', MetricCollector, Connector(),
            params={'output_var': None,
                    'metric_func': {'__callable__': 'sklearn.metrics.balanced_accuracy_score'}},
        )
        assert mc.metric_func is balanced_accuracy_score
        exp.exp(_folds(trial, exp), project.trials, collectors=[mc])
        assert mc.has_node('dt')


class TestCollectorsRegistry:
    """Collectors are a project-wide registry now (`mllabs.collector._registry`),
    not owned by Experimenter (no more exp.set_collector/get_collector) — the
    set_collector 'skip'/'error' exist modes are tested directly against it."""

    def test_set_collector(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        mc = registry.set_collector('acc', MetricCollector, Connector(),
                                    params={'output_var': None, 'metric_func': dummy_metric})
        assert registry.get_collector('acc') is not None
        assert mc.path is not None

    def test_set_collector_skip(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        mc1 = registry.set_collector('acc', MetricCollector, Connector(),
                                     params={'output_var': None, 'metric_func': dummy_metric})
        result = registry.set_collector('acc', MetricCollector, Connector(),
                                        params={'output_var': None, 'metric_func': dummy_metric},
                                        exist='skip')
        assert result is mc1

    def test_set_collector_error(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        registry.set_collector('acc', MetricCollector, Connector(),
                               params={'output_var': None, 'metric_func': dummy_metric})
        with pytest.raises(RuntimeError):
            registry.set_collector('acc', MetricCollector, Connector(),
                                   params={'output_var': None, 'metric_func': dummy_metric},
                                   exist='error')


class TestResetNodes:
    def test_reset_removes_node_dir(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        node_path = flow.node_path('scaler')
        assert node_path.exists()
        exp.reset_nodes(['scaler'])
        assert not node_path.exists()
        assert flow.status('scaler') is None

    def test_reset_clears_cache(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        exp.cache.put_data(flow.scope, 'scaler', 'train', np.array([1]))
        exp.reset_nodes(['scaler'])
        assert exp.cache.get_data(flow.scope, 'scaler', 'train') is None

    # Whether Experimenter.reset_nodes() cascades into a Collector's own
    # buffered state is covered in test_collector.py
    # (test_experimenter_reset_nodes_does_not_clear_collectors) — it doesn't,
    # since Collectors live in a separate registry now, not owned by
    # Experimenter.


class TestRebuild:
    def test_build_with_rebuild_true(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        exp.build(rebuild=True)
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert flow.status('scaler') == 'built'
        assert old_obj is not new_obj

    def test_set_node_replace_then_build(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        # Staleness is a value diff now (no serial) — 'replace' with the exact
        # same definition is correctly a no-op, so this needs an actual change.
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        _publish(pipeline, exp)
        exp.build()
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert new_obj is not old_obj
        assert _flow(exp).status('scaler') == 'built'

    def test_exp_rebuilds_non_built_node(self, exp, pipeline, project):
        """Resetting a Trial requires clearing both NodeStore (artifact) and
        TrialStore.experiment_hist (skip decision) — _make_jobs skips purely
        on hist status, so reset_nodes() alone would leave the fold skipped."""
        trial = _dt()
        exp.exp(_folds(trial, exp), project.trials)
        assert exp.get_status('dt') == 'built'
        exp.reset_nodes(['dt'])
        assert exp.get_status('dt') is None
        project.trials.remove_hist(trial_name='dt', experimenter=exp.name)
        exp.exp(_folds(trial, exp), project.trials)
        assert exp.get_status('dt') == 'built'


class TestSetPipeline:
    def test_no_pipeline_raises_on_build(self, tmp_path, sample_data):
        e = Experimenter(path=tmp_path / 'no_pipeline', name='e1', data=sample_data)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            e.build()

    def test_no_pipeline_raises_on_exp(self, tmp_path, sample_data):
        e = Experimenter(path=tmp_path / 'no_pipeline', name='e1', data=sample_data)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            e.exp([], None)

    def test_constructor_pipeline_sets_attribute(self, exp, pipeline):
        from mllabs._pipeline import Pipeline
        assert isinstance(exp.pipeline, Pipeline)
        assert exp.pipeline.pipeline_id == pipeline.pipeline_id

    def test_builder_is_rejected(self, tmp_path, sample_data, pipeline):
        e = Experimenter(path=tmp_path / 'reject', name='e1', data=sample_data)
        with pytest.raises(TypeError, match='built Pipeline'):
            e.set_pipeline(pipeline)

    def test_set_pipeline_keeps_a_copy_in_the_run(self, exp, pipeline):
        """The run owns the Pipeline it works against, so reopening it needs
        only its directory. The (pipeline_name, pipeline_version) pointer is
        recorded alongside as provenance."""
        exp.set_pipeline(pipeline.build())
        assert (exp.path / 'pipeline.pkl').exists()

    def test_pipeline_is_restored_from_the_run_directory(self, exp, pipeline, sample_data):
        """load_experimenter picks the Pipeline back up from the directory —
        nothing resolves a version."""
        _setup_stage(pipeline, exp)
        exp.build()

        reopened = Experimenter.load_experimenter(exp.path, sample_data)
        assert reopened.pipeline is not None
        assert reopened.pipeline.get_node_names() == exp.pipeline.get_node_names()
        assert reopened.get_status('scaler') == 'built'

    def test_a_standalone_run_persists_its_own_pipeline(self, tmp_path, sample_data, pipeline):
        """No Project, no injected store — the run still keeps its Pipeline."""
        e = Experimenter(tmp_path / 'bare', 'bare', sample_data)
        e.set_pipeline(pipeline.build())
        assert (e.path / 'pipeline.pkl').exists()

        reopened = Experimenter.load_experimenter(e.path, sample_data)
        assert reopened.name == 'bare'
        assert reopened.pipeline is not None

    def test_set_pipeline_resets_stale_nodes(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('scaler') == 'built'
        # Staleness is a value diff now (no serial) — 'replace' with the exact
        # same definition is correctly a no-op, so this needs an actual change.
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        exp.set_pipeline(pipeline.build())
        assert flow.status('scaler') is None


class TestSaveLoad:
    def test_load_restores(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        exp.build()
        trial = _dt()
        exp.exp(_folds(trial, exp), project.trials)

        loaded = project.load_experimenter(exp.name, sample_data)
        flow = loaded.outer_folds[0].train_data_flows[0]
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'

    def test_load_restores_pipeline(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        exp.build()

        loaded = project.load_experimenter(exp.name, sample_data)
        assert loaded.pipeline is not None
        assert 'scaler' in loaded.pipeline.nodes
        trial = _dt()
        loaded.exp(_folds(trial, loaded), project.trials)  # uses restored pipeline, no set_pipeline needed
        assert loaded.get_status('dt') == 'built'

    def test_load_data_key_mismatch(self, project, sample_data):
        project.experimenter('dk', sample_data, data_key='key_a')
        with pytest.raises(ValueError, match='data_key'):
            project.load_experimenter('dk', sample_data, data_key='key_b')

    def test_load_preserves_splits(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        loaded = project.load_experimenter(exp.name, sample_data)
        assert loaded.get_n_splits() == exp.get_n_splits()
        assert loaded.get_n_splits_inner() == exp.get_n_splits_inner()

    def test_persistence_layout(self, exp, pipeline):
        """Everything the run needs sits in the run's own directory."""
        _setup_stage(pipeline, exp)
        assert (exp.path / '__exp.db').exists()
        assert (exp.path / 'pipeline.pkl').exists()
        assert not (exp.path / '__splitters.pkl').exists()

    def test_meta_lives_in_the_runs_own_store(self, exp):
        meta = exp._store.fetch()
        assert meta['name'] == exp.name
        assert set(meta.keys()) == {'name', 'data_key', 'title', 'pipeline_name', 'pipeline_version'}

    def test_splitters_live_in_the_store(self, exp):
        """Not a side file — the store owns them, as a blob (sklearn splitters
        are arbitrary objects, so columns are not an option)."""
        splitters = exp._store.load_splitters()
        assert splitters['sp'] is exp.sp or splitters['sp'].get_n_splits() == exp.get_n_splits()
        assert set(splitters) == {'sp', 'sp_v', 'splitter_params'}


class TestGetStatus:
    def test_get_status_none_before_exp(self, exp, pipeline, project):
        assert exp.get_status('dt') is None

    def test_get_status_built_after_exp(self, exp, pipeline, project):
        trial = _dt()
        exp.exp(_folds(trial, exp), project.trials)
        assert exp.get_status('dt') == 'built'

    def test_get_status_error(self, exp, pipeline, project):
        bad_trial = Trial('bad_dt', 'mock.BadPredictor', {'X': '{f1}', 'y': '{target}'})
        exp.exp(_folds(bad_trial, exp), project.trials)
        assert _trial_errored(project.trials, 'bad_dt', exp)


class TestNodeStore:
    """No info.pkl anymore (2026-08-01) — obj.pkl/result.pkl only, and
    ``status`` is purely whether ``obj.pkl`` exists. Build/error history now
    lives in this same store's ``node_hist`` table (merged in from the former
    NodeInfoStore) / TrialStore.experiment_hist for Trials.

    NodeStore is per-run now, not per-fold — every read/write method takes
    ``(name, outer_idx, inner_idx)`` explicitly; ``node_path(...)`` resolves
    ``{path}/{outer_idx}/{inner_idx}/{name}/``.
    """

    def test_write_objs_and_status(self, tmp_path):
        store = NodeStore(tmp_path)
        store.write_objs('node1', 0, 0, object(), np.array([1, 2]))
        assert store.status('node1', 0, 0) == 'built'

    def test_get_objs(self, tmp_path):
        store = NodeStore(tmp_path)
        sc = StandardScaler()
        result = np.array([1.0, 2.0])
        store.write_objs('node1', 0, 0, sc, result)
        got_obj, got_result = store.get_objs('node1', 0, 0)
        assert isinstance(got_obj, StandardScaler)
        assert np.array_equal(got_result, result)

    def test_get_obj_get_result(self, tmp_path):
        store = NodeStore(tmp_path)
        sc = StandardScaler()
        result = np.array([3.0])
        store.write_objs('node1', 0, 0, sc, result)
        assert isinstance(store.get_obj('node1', 0, 0), StandardScaler)
        assert np.array_equal(store.get_result('node1', 0, 0), result)

    def test_status_none_when_missing(self, tmp_path):
        store = NodeStore(tmp_path)
        assert store.status('missing', 0, 0) is None

    def test_reset_node(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = store.node_path('node1', 0, 0)
        store.write_objs('node1', 0, 0, None, None)
        assert node_path.exists()
        store.reset_node('node1', 0, 0)
        assert not node_path.exists()
        assert store.status('node1', 0, 0) is None

    def test_record_and_get_hist(self, tmp_path):
        store = NodeStore(tmp_path)
        store.record('node1', 0, 0, pipeline_version=1, status='built',
                     info={'role': 'stage', 'edges': {'X': '{f1}'}})
        assert store.get_status('node1') == {(0, 0): 'built'}
        assert store.get_info('node1') == {(0, 0): {'role': 'stage', 'edges': {'X': '{f1}'}}}

    def test_dataflow_autoload(self, tmp_path):
        """DataFlow composes a NodeStore now (no inheritance) and needs a
        node_hist row to know a node is a Stage (role) and how to route its
        edges — see DataFlow.load()."""
        store = NodeStore(tmp_path)
        store.write_objs('node1', 0, 0, StandardScaler(), None)
        store.record('node1', 0, 0, status='built',
                     info={'role': 'stage', 'edges': {'X': '{f1}'}})
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert 'node1' in flow.node_objs
        assert flow.status('node1') == 'built'


class TestGetMissingNodes:
    def test_no_missing_when_all_built(self, tmp_path):
        store = NodeStore(tmp_path)
        store.write_objs('n1', 0, 0, StandardScaler(), None)
        store.write_objs('n2', 0, 0, StandardScaler(), None)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert flow.get_missing_nodes({'X': 'n1:(*) + n2:(*)'}) == []

    def test_missing_when_node_not_built(self, tmp_path):
        store = NodeStore(tmp_path)
        store.write_objs('n1', 0, 0, StandardScaler(), None)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert flow.get_missing_nodes({'X': 'n1:(*) + n2:(*)'}) == ['n2']

    def test_missing_when_node_dir_exists_without_obj(self, tmp_path):
        """A directory with no obj.pkl (e.g. left over from a prep-time
        failure) still counts as missing — status is purely obj.pkl existence
        now (no info.pkl 'error' status to distinguish it from 'never ran')."""
        store = NodeStore(tmp_path)
        store.write_objs('n1', 0, 0, StandardScaler(), None)
        store.node_path('n2', 0, 0).mkdir(parents=True)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert flow.get_missing_nodes({'X': 'n1:(*) + n2:(*)'}) == ['n2']

    def test_datasource_segment_is_never_missing(self, tmp_path):
        flow = DataFlow(NodeStore(tmp_path), outer_idx=0, inner_idx=0)
        assert flow.get_missing_nodes({'X': '{f1, f2}', 'y': '{target}'}) == []

    def test_only_referenced_nodes_checked(self, tmp_path):
        store = NodeStore(tmp_path)
        store.write_objs('n1', 0, 0, StandardScaler(), None)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        # n2 was never built, but nothing here references it
        assert flow.get_missing_nodes({'X': 'n1:(*)'}) == []
