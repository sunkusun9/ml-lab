from datetime import datetime

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
def project(tmp_path, sample_data):
    return Project(tmp_path / 'proj', data=sample_data)


@pytest.fixture
def pipeline(project):
    p = project.pipeline
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    return p


@pytest.fixture
def exp(project, pipeline):
    version = pipeline.build().version
    return project.set_experimenter(
        'e1',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), pipeline_version=version,
    )


def _publish(pipeline, exp):
    """Hand the current definitions to *exp* as a fresh, versioned snapshot.

    ``build()`` publishes, so this mints a version whenever the definitions
    actually changed and reuses the current one when they did not.

    Experimenter holds a built Pipeline, so edits made to the builder after
    construction only take effect once they are built and set again.
    """
    if exp is not None:
        exp.set_pipeline(pipeline.build())


def _setup_stage(pipeline, exp=None):
    pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                      method='transform', edges={'X': '{f1, f2, f3}'})
    pipeline.set_node('scaler', grp='scale')
    _publish(pipeline, exp)


def _dt(name='dt', edges=None):
    return Trial(name, TREE, edges or DT_EDGES, params={'max_depth': 3, 'random_state': 42})


def _folds(trial, exp):
    """Register *trial* and return the name list Experimenter.exp expects.

    exp() takes names and reads the definition out of the store, so a test
    Trial has to be in there first. Registering through the store rather than
    Project.set_trial deliberately skips the redefinition guard — several
    tests below redefine a name on purpose to check what a rerun does. It also
    skips the version stamp, so that is done here: exp() refuses a Trial
    defined against another version, and these tests are about everything
    except that."""
    if trial.pipeline_version is None:
        trial.pipeline_version = exp.pipeline_version
    exp.trial_store.register(trial)
    return [trial.name]


def _flow(exp, outer=0, inner=0):
    return exp.outer_folds[outer].train_data_flows[inner]


def _trial_built(trial_store, trial_name, exp):
    """Whether every fold of *exp* recorded *trial_name* as clean.

    A Trial leaves no artifact, so ``Experimenter.get_status``/
    ``NodeStore.status`` (both disk) can never answer this —
    ``experiment_hist`` is the whole record."""
    expected = {(o, i) for o in range(exp.get_n_splits())
                for i in range(exp.get_n_splits_inner())}
    status = trial_store.get_status(trial_name, exp.name)
    return set(status) == expected and set(status.values()) == {'built'}


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
        assert flow.status('scaler') == 'built'

    def test_build_leaves_nothing_in_flow_memory(self, exp, pipeline):
        """A build writes its artifact and hands the flow nothing — whoever
        reads the node next takes it off disk."""
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert 'scaler' not in flow.node_objs
        flow.get_train({'X': 'scaler:(*)'})
        assert 'scaler' in flow.node_objs

    def test_build_skips_built(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        build_id = exp.node_store.get_info('scaler')[(0, 0)]['build_id']
        exp.build()
        assert exp.node_store.get_info('scaler')[(0, 0)]['build_id'] == build_id

    def test_build_records_when_processing_started(self, exp, pipeline):
        """started_at is _process()'s own clock — when it began running, not
        when the row landed in node_hist (that's recorded_at, see TestNodeStore)."""
        _setup_stage(pipeline, exp)
        exp.build()
        started_at = exp.node_store.get_info('scaler')[(0, 0)]['started_at']
        assert started_at is not None
        datetime.fromisoformat(started_at)  # doesn't raise

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
        assert info['started_at'] is not None  # stamped on the error path too

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


class TestExpTakesNames:
    """exp() is handed Trial names and reads the definitions out of the store.

    A Trial belongs to the project, so running one is not also how it gets
    there: registration used to be a side effect of exp(), which meant a Trial
    could not be added without executing it, and one run could redefine a name
    another had already used."""

    def test_a_name_resolves_to_the_stored_definition(self, exp, project):
        project.set_trial(Trial('dt', TREE, DT_EDGES, params={'max_depth': 7}))
        jobs = exp._make_jobs(['dt'])
        assert {j.spec.params['max_depth'] for j in jobs} == {7}

    def test_a_name_covers_every_fold(self, exp, project):
        """A name means the whole grid — the caller does not spell out folds."""
        project.set_trial(_dt())
        jobs = exp._make_jobs(['dt'])
        assert {(j.outer_idx, j.inner_idx) for j in jobs} == {
            (o, i) for o in range(exp.get_n_splits())
            for i in range(exp.get_n_splits_inner())}

    def test_built_folds_are_dropped_so_the_same_names_continue(self, exp, project):
        """Passing the same names after a partial run continues it."""
        project.set_trial(_dt())
        exp.trial_store.record('dt', exp.name, 0, 0, status='built')
        jobs = exp._make_jobs(['dt'])
        assert (0, 0) not in {(j.outer_idx, j.inner_idx) for j in jobs}
        assert (1, 0) in {(j.outer_idx, j.inner_idx) for j in jobs}

    def test_an_unregistered_name_raises(self, exp, project):
        with pytest.raises(KeyError, match='not registered'):
            exp.exp(['nope'])

    def test_a_trial_instance_is_not_accepted(self, exp, project):
        """Caught explicitly — passed through, an instance reaches sqlite as a
        bind parameter and the error says nothing about what the caller did."""
        with pytest.raises(TypeError, match='set_trial'):
            exp.exp([_dt()])

    def test_running_does_not_register(self, exp, project):
        """Authoring is Project.set_trial — exp() only executes what is there."""
        project.set_trial(_dt())
        exp.exp(['dt'])
        assert len(project.trials.list_trials()) == 1

    def test_without_a_trial_store_it_refuses(self, tmp_path, sample_data, pipeline):
        standalone = Experimenter(path=tmp_path / 'solo', name='solo', data=sample_data)
        standalone.set_pipeline(pipeline.build())
        with pytest.raises(RuntimeError, match='trial_store'):
            standalone.exp(['dt'])

    def test_a_project_run_is_given_the_projects_store(self, exp, project):
        assert exp.trial_store is project.trials

    def test_a_reopened_one_is_given_it_too(self, project, sample_data, exp):
        reopened = Project(project.path, data=sample_data)
        assert reopened.experimenters['e1'].trial_store is reopened.trials


class TestExp:
    def test_exp_head(self, exp, pipeline, project):
        trial = _dt()
        exp.exp(_folds(trial, exp))
        assert _trial_built(project.trials, 'dt', exp)

    def test_exp_skips_built(self, exp, pipeline, project):
        trial = _dt()
        exp.exp(_folds(trial, exp))
        build_id = project.trials.get_info('dt', exp.name)[(0, 0)]['build_id']
        exp.exp(_folds(trial, exp))
        assert project.trials.get_info('dt', exp.name)[(0, 0)]['build_id'] == build_id

    def test_exp_error(self, exp, pipeline, project):
        bad_trial = Trial('bad_dt', 'mock.BadPredictor', {'X': '{f1}', 'y': '{target}'})
        exp.exp(_folds(bad_trial, exp))
        assert _trial_errored(project.trials, 'bad_dt', exp)

    def test_exp_with_collector(self, exp, pipeline, project):
        trial = _dt()
        mc = exp.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.accuracy_metric'}},
        )
        exp.exp(_folds(trial, exp), collectors=['acc'])
        assert mc.has_node('dt')

    def test_set_collector_resolves_callable_metric(self, exp, pipeline, project):
        from sklearn.metrics import balanced_accuracy_score
        trial = _dt()
        mc = exp.collectors.set_collector(
            'bacc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None,
                    'metric_func': {'__callable__': 'sklearn.metrics.balanced_accuracy_score'}},
        )
        assert mc.metric_func is balanced_accuracy_score
        exp.exp(_folds(trial, exp), collectors=['bacc'])
        assert mc.has_node('dt')

    def test_the_run_restores_its_own_collectors(self, exp, pipeline, project, sample_data):
        """The registry is the run's, so reopening it brings the Collectors
        back — a standalone Experimenter is complete on its own directory."""
        exp.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.accuracy_metric'}},
        )
        reopened = Experimenter.load_experimenter(exp.path, sample_data)
        assert reopened.collectors.names() == ['acc']


class TestCollectorsRegistry:
    """A registry belongs to one run (`mllabs.collector._registry`); an
    Experimenter builds its own and hands it out as `.collectors`. The
    set_collector 'skip'/'error' exist modes are tested directly against it."""

    def test_set_collector(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        mc = registry.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                                    params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.dummy_metric'}})
        assert registry.get_collector('acc') is not None
        assert mc.path is not None

    def test_set_collector_skip(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        mc1 = registry.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                                     params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.dummy_metric'}})
        result = registry.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                                        params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.dummy_metric'}},
                                        exist='skip')
        assert result is mc1

    def test_set_collector_error(self, tmp_path):
        registry = Collectors(tmp_path / 'coll')
        registry.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                               params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.dummy_metric'}})
        with pytest.raises(RuntimeError):
            registry.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                                   params={'output_var': None, 'metric_func': {'__callable__': 'test_experimenter.dummy_metric'}},
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


class TestTrialLeavesNoArtifact:
    """A Trial is a candidate being measured, so only its outcome is kept:
    experiment_hist plus whatever its Collectors collected. Nothing is
    written to the run's NodeStore and nothing is pinned in the flow."""

    def test_nothing_on_disk_and_nothing_in_the_flow(self, exp, project):
        trial = _dt()
        exp.exp(_folds(trial, exp))
        flow = _flow(exp)
        assert not flow.node_path('dt').exists()
        assert 'dt' not in flow.list_nodes()
        assert 'dt' not in flow.node_objs
        assert flow.status('dt') is None
        assert exp.get_status('dt') is None
        assert exp.node_store.get_hist(node_name='dt') == []

    def test_the_outcome_is_still_recorded(self, exp, project):
        trial = _dt()
        exp.exp(_folds(trial, exp))
        status = project.trials.get_status('dt', exp.name)
        assert status
        assert set(status.values()) == {'built'}

    def test_nodes_are_unaffected(self, exp, pipeline, project):
        _setup_stage(pipeline, exp)
        exp.build()
        exp.exp(_folds(_dt(), exp))
        flow = _flow(exp)
        assert flow.status('scaler') == 'built'
        assert flow.get_obj('scaler') is not None

    def test_a_trial_reads_nodes_built_in_the_same_process(self, exp, pipeline, project):
        """A Trial reading a node through a namespace segment loads it out of
        the store — build() published nothing into the flow."""
        _setup_stage(pipeline, exp)
        exp.build()
        trial = _dt(edges={'X': 'scaler:(*)', 'y': '{target}'})
        exp.exp(_folds(trial, exp))
        assert set(project.trials.get_status('dt', exp.name).values()) == {'built'}

    def test_rerun_needs_only_remove_hist(self, exp, project):
        trial = _dt()
        exp.exp(_folds(trial, exp))
        project.trials.remove_hist(trial_name='dt', experimenter=exp.name)
        exp.exp(_folds(trial, exp))
        assert set(project.trials.get_status('dt', exp.name).values()) == {'built'}

    def test_a_failed_trial_records_the_error_and_leaves_nothing(self, exp, project):
        bad = Trial('dt', 'mock.BadPredictor', DT_EDGES)
        exp.exp(_folds(bad, exp))
        flow = _flow(exp)
        assert not flow.node_path('dt').exists()
        info = project.trials.get_info('dt', exp.name)[(0, 0)]
        assert set(project.trials.get_status('dt', exp.name).values()) == {'error'}
        assert info['error']['type'] == 'RuntimeError'
        assert 'traceback' in info['error']

    def test_a_failed_fold_is_retried_on_the_next_exp(self, exp, project):
        """'error' is not 'built', so the fold gets a job again — no reset
        needed, and nothing stale can be left behind to confuse it."""
        exp.exp(_folds(Trial('dt', 'mock.BadPredictor', DT_EDGES), exp))
        assert set(project.trials.get_status('dt', exp.name).values()) == {'error'}
        exp.exp(_folds(_dt(), exp))
        assert _trial_built(project.trials, 'dt', exp)

    def test_reset_nodes_on_a_trial_name_is_inert(self, exp, project):
        """It has nothing to remove, so it cannot leave history asserting a
        result whose artifact is gone (#127)."""
        trial = _dt()
        exp.exp(_folds(trial, exp))
        exp.reset_nodes(['dt'])
        assert set(project.trials.get_status('dt', exp.name).values()) == {'built'}


class TestRebuild:
    # build_id, not object identity: nothing is held in flow memory across a
    # build anymore, so two reads of the same artifact are two objects and
    # `is not` would pass without anything having been rebuilt.
    def _build_id(self, exp):
        return exp.node_store.get_info('scaler')[(0, 0)]['build_id']

    def test_build_with_rebuild_true(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        old_id = self._build_id(exp)
        exp.build(rebuild=True)
        assert _flow(exp).status('scaler') == 'built'
        assert self._build_id(exp) != old_id

    def test_set_node_replace_then_build(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        old_id = self._build_id(exp)
        # Staleness is a value diff now (no serial) — 'replace' with the exact
        # same definition is correctly a no-op, so this needs an actual change.
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        _publish(pipeline, exp)
        exp.build()
        assert self._build_id(exp) != old_id
        assert _flow(exp).status('scaler') == 'built'

    def test_exp_skips_folds_already_recorded_built(self, exp, pipeline, project):
        """experiment_hist is the whole skip decision — a Trial persists
        nothing that could disagree with it."""
        trial = _dt()
        exp.exp(_folds(trial, exp))
        first = project.trials.get_hist(trial_name='dt', experimenter=exp.name)
        exp.exp(_folds(trial, exp))
        second = project.trials.get_hist(trial_name='dt', experimenter=exp.name)
        assert [r['info']['build_id'] for r in first] == [r['info']['build_id'] for r in second]


class TestSetPipeline:
    def test_no_pipeline_is_nothing_to_build(self, tmp_path, sample_data):
        """Having no Pipeline is a state, not a fault: the default is the empty
        one, so there is simply nothing to build."""
        e = Experimenter(path=tmp_path / 'no_pipeline', name='e1', data=sample_data)
        assert e.pipeline.is_empty
        assert e.pipeline_version is None
        e.build()

    def test_no_pipeline_still_runs_trials(self, tmp_path, sample_data):
        """A Trial reads the DataSource directly, so an Experimenter with no
        nodes at all is still a working one — no preprocessing, raw columns."""
        from mllabs import TrialStore, Trial
        store = TrialStore(tmp_path)
        e = Experimenter(path=tmp_path / 'no_pipeline', name='e1', data=sample_data,
                         trial_store=store)
        trial = Trial('dt', 'sklearn.tree.DecisionTreeClassifier',
                      edges={'X': '{f1, f2}', 'y': '{target}'},
                      params={'max_depth': 2, 'random_state': 0})
        store.register(trial)
        e.exp(['dt'])
        assert e.trial_store.get_status('dt', 'e1') == {(0, 0): 'built'}

    def test_constructor_pipeline_sets_attribute(self, exp, pipeline):
        from mllabs._pipeline import Pipeline
        assert isinstance(exp.pipeline, Pipeline)
        assert exp.pipeline.pipeline_id == pipeline.pipeline_id

    def test_builder_is_rejected(self, tmp_path, sample_data, pipeline):
        e = Experimenter(path=tmp_path / 'reject', name='e1', data=sample_data)
        with pytest.raises(TypeError, match='built Pipeline'):
            e.set_pipeline(pipeline)

    def test_a_draft_is_rejected(self, tmp_path, sample_data, pipeline):
        """The same gate the Trainer has. Experimenting against a definition
        under edit used to be allowed, but a Trial names the version it was
        authored against and an adopted draft has no number to check it
        against."""
        e = Experimenter(path=tmp_path / 'draft', name='e1', data=sample_data)
        with pytest.raises(ValueError, match='draft'):
            e.set_pipeline(pipeline.draft())

    def test_set_pipeline_keeps_a_copy_in_the_run(self, exp, pipeline):
        """The run owns the Pipeline it works against, so reopening it needs
        only its directory. The pipeline_version pointer is
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


class TestStaleNodes:
    """Adoption spends the answer — it derives staleness and turns it straight
    into deletions — so before adopting is the only moment it can be read."""

    def test_nothing_adopted_is_nothing_stale(self, tmp_path, sample_data, pipeline):
        """The empty Pipeline means 'no prior definition', not 'nothing was
        built'; diffing against it would name every node."""
        e = Experimenter(path=tmp_path / 'fresh', name='e1', data=sample_data)
        assert e.stale_nodes(pipeline.build()) == []

    def test_unchanged_definition_is_nothing_stale(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        assert exp.stale_nodes(pipeline.build()) == []

    def test_changed_node_is_named(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        assert exp.stale_nodes(pipeline.build()) == ['scaler']

    def test_preview_is_exactly_what_adoption_resets(self, exp, pipeline):
        """Why the two share one implementation: what the preview names is what
        loses its artifact, with nothing else touched."""
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        candidate = pipeline.build()

        predicted = exp.stale_nodes(candidate)
        assert predicted and all(flow.status(n) == 'built' for n in predicted)
        exp.set_pipeline(candidate)
        assert all(flow.status(n) is None for n in predicted)

    def test_asking_changes_nothing(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        adopted = exp.pipeline_version
        pipeline.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        exp.stale_nodes(pipeline.build())
        assert exp.pipeline_version == adopted
        assert _flow(exp).status('scaler') == 'built'

    def test_builder_is_rejected(self, exp, pipeline):
        with pytest.raises(TypeError, match='built Pipeline'):
            exp.stale_nodes(pipeline)


class TestSaveLoad:
    def test_load_restores(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        exp.build()
        trial = _dt()
        exp.exp(_folds(trial, exp))

        loaded = Experimenter.load_experimenter(exp.path, sample_data,
                                               trial_store=project.trials)
        flow = loaded.outer_folds[0].train_data_flows[0]
        # Reopening reads no artifact: what it recovers is the ability to.
        assert flow.node_objs == {}
        assert flow.status('scaler') == 'built'
        assert flow.get_train({'X': 'scaler:(*)'})['X'].get_shape()[0] > 0
        assert _trial_built(project.trials, 'dt', loaded)

    def test_load_restores_pipeline(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        exp.build()

        loaded = Experimenter.load_experimenter(exp.path, sample_data,
                                               trial_store=project.trials)
        assert loaded.pipeline is not None
        assert 'scaler' in loaded.pipeline.nodes
        trial = _dt()
        loaded.exp(_folds(trial, loaded))  # uses restored pipeline, no set_pipeline needed
        assert _trial_built(project.trials, 'dt', loaded)

    def test_load_data_key_mismatch(self, project, sample_data):
        e = project.set_experimenter('dk', data_key='key_a')
        with pytest.raises(ValueError, match='data_key'):
            Experimenter.load_experimenter(e.path, sample_data, data_key='key_b')

    def test_load_preserves_splits(self, project, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        loaded = Experimenter.load_experimenter(exp.path, sample_data)
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
        assert set(meta.keys()) == {'name', 'data_key', 'title', 'pipeline_version'}

    def test_splitters_live_in_the_store(self, exp):
        """Not a side file — the store owns them, as a blob (sklearn splitters
        are arbitrary objects, so columns are not an option)."""
        splitters = exp._store.load_splitters()
        assert splitters['sp'] is exp.sp or splitters['sp'].get_n_splits() == exp.get_n_splits()
        assert set(splitters) == {'sp', 'sp_v', 'splitter_params'}


class TestGetStatus:
    def test_get_status_none_before_build(self, exp, pipeline, project):
        _setup_stage(pipeline, exp)
        assert exp.get_status('scaler') is None

    def test_get_status_built_after_build(self, exp, pipeline, project):
        _setup_stage(pipeline, exp)
        exp.build()
        assert exp.get_status('scaler') == 'built'

    def test_get_status_is_nodes_only(self, exp, pipeline, project):
        """A Trial persists nothing, so it never shows here however often it
        has run — its status is TrialStore's to answer."""
        trial = _dt()
        exp.exp(_folds(trial, exp))
        assert exp.get_status('dt') is None
        assert _trial_built(project.trials, 'dt', exp)

    def test_get_status_error(self, exp, pipeline, project):
        bad_trial = Trial('bad_dt', 'mock.BadPredictor', {'X': '{f1}', 'y': '{target}'})
        exp.exp(_folds(bad_trial, exp))
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

    def test_recorded_at_is_stamped_automatically(self, tmp_path):
        store = NodeStore(tmp_path)
        store.record('node1', 0, 0, status='built')
        recorded_at = store.get_hist(node_name='node1')[0]['recorded_at']
        assert recorded_at is not None
        datetime.fromisoformat(recorded_at)  # doesn't raise

    def test_recorded_at_can_be_given_explicitly(self, tmp_path):
        store = NodeStore(tmp_path)
        store.record('node1', 0, 0, status='built',
                     recorded_at='2026-01-01T00:00:00+00:00')
        assert store.get_hist(node_name='node1')[0]['recorded_at'] == '2026-01-01T00:00:00+00:00'

    def test_record_and_get_hist(self, tmp_path):
        store = NodeStore(tmp_path)
        store.record('node1', 0, 0, pipeline_version=1, status='built',
                     info={'role': 'stage', 'edges': {'X': '{f1}'}})
        assert store.get_status('node1') == {(0, 0): 'built'}
        assert store.get_info('node1') == {(0, 0): {'role': 'stage', 'edges': {'X': '{f1}'}}}

    def test_dataflow_reads_nothing_until_asked(self, tmp_path):
        """Constructing a flow touches no artifact — a run makes one per fold
        up front, so anything read here is read for every fold of the grid."""
        store = NodeStore(tmp_path)
        store.write_objs('node1', 0, 0, StandardScaler(), None)
        store.record('node1', 0, 0, status='built',
                     info={'edges': {'X': '{f1}'}})
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert flow.node_objs == {}
        assert flow.status('node1') == 'built'

    def test_dataflow_recovers_edges_on_first_use(self, tmp_path):
        """DataFlow composes a NodeStore (no inheritance) and neither pickle
        carries edges, so routing comes from the node_hist row."""
        store = NodeStore(tmp_path)
        store.write_objs('node1', 0, 0, StandardScaler(), None)
        store.record('node1', 0, 0, status='built',
                     info={'edges': {'X': '{f1}'}})
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        assert flow._processor('node1') is not None
        assert flow._node_edges['node1'] == {'X': '{f1}'}

    def test_dataflow_raises_for_a_node_that_is_not_built(self, tmp_path):
        """Loudly — the old answer was a segment that contributed no columns
        and said nothing about it."""
        flow = DataFlow(NodeStore(tmp_path), outer_idx=0, inner_idx=0)
        with pytest.raises(KeyError, match='not built'):
            flow._processor('node1')

    def test_dataflow_raises_for_an_artifact_with_no_history(self, tmp_path):
        store = NodeStore(tmp_path)
        store.write_objs('node1', 0, 0, StandardScaler(), None)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        with pytest.raises(KeyError, match='no recorded edges'):
            flow._processor('node1')

    def test_dataflow_sees_a_node_built_after_it_read_the_fold(self, tmp_path):
        """The fold's history is cached, but a build hands the flow nothing —
        so a miss has to re-query rather than answer from the old snapshot."""
        store = NodeStore(tmp_path)
        flow = DataFlow(store, outer_idx=0, inner_idx=0)
        with pytest.raises(KeyError):
            flow._processor('node1')

        store.write_objs('node1', 0, 0, StandardScaler(), None)
        store.record('node1', 0, 0, status='built',
                     info={'edges': {'X': '{f1}'}})
        assert flow._processor('node1') is not None


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
