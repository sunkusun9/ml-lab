import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from mllabs import (Project, TrialStore, Trial, Predictor, PipelineBuilder, make_trials,
                    Connector, MetricCollector)


TREE = 'sklearn.tree.DecisionTreeClassifier'
EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / 'proj')


@pytest.fixture
def builder(project):
    p = project.pipeline_builder('main')
    p.set_datasource({'f1': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1}'})
    p.set_node('scaler', grp='scale')
    return p


@pytest.fixture
def store(tmp_path):
    return TrialStore(tmp_path / 'ts')


def _trial(name='dt', params=None):
    return Trial(name, TREE, EDGES, params=params or {'max_depth': 3})


def dummy_metric(y, pred):
    return 0.5


class TestProjectLayout:
    def test_root_created(self, project, tmp_path):
        assert (tmp_path / 'proj').is_dir()

    def test_trial_store_created(self, project):
        assert (project.path / 'trials.db').exists()

    def test_project_store_created(self, project):
        assert (project.path / 'project.db').exists()

    def test_paths_are_under_root(self, project):
        for p in (project.pipeline_path('a'), project.exp_path('b'),
                  project.trainer_path('c'), project.inferencer_path('d'),
                  project.collectors_path()):
            assert project.path in p.parents or p.parent == project.path

    def test_paths_are_created(self, project):
        assert project.exp_path('r1').is_dir()

    def test_exp_path_is_under_exp_folder(self, project):
        assert project.exp_path('run_a') == project.path / 'exp' / 'run_a'

    def test_pipeline_builder_stored_under_project(self, project):
        p = project.pipeline_builder('main')
        assert p._store.db_path == project.pipeline_path('main') / 'main.db'

    def test_pipeline_builder_persists(self, project):
        p = project.pipeline_builder('main')
        p.set_grp('g', processor=TREE, method='predict', edges={'X': '{f1}'})
        again = project.pipeline_builder('main')
        assert 'g' in again.grps

    def test_collectors_registry_rooted_in_project(self, project):
        assert project.collectors().path == project.collectors_path()


class TestPipelineVersions:
    def test_first_build_is_v1(self, project, builder):
        assert project.build_pipeline(builder).version == 1

    def test_unbuilt_pipeline_has_no_version(self, builder):
        assert builder.build().version is None

    def test_rebuild_always_bumps(self, project, builder):
        """No content dedup — even an unchanged rebuild mints a new version."""
        a = project.build_pipeline(builder)
        b = project.build_pipeline(builder)
        assert a.build_id != b.build_id
        assert b.version == a.version + 1

    def test_edit_bumps_version(self, project, builder):
        v1 = project.build_pipeline(builder).version
        builder.set_node('scaler', grp='scale', exist='replace')
        assert project.build_pipeline(builder).version == v1 + 1

    def test_versions_are_per_name(self, project, builder):
        project.build_pipeline(builder)
        other = project.pipeline_builder('other')
        other.set_datasource({'f1': 'numerical', 'target': 'binary'})
        other.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                      method='transform', edges={'X': '{f1}'})
        other.set_node('scaler', grp='scale')
        assert project.build_pipeline(other).version == 1

    def test_build_pipeline_requires_a_path(self, project):
        with pytest.raises(ValueError, match='no db path'):
            project.build_pipeline(PipelineBuilder())

    def test_load_latest(self, project, builder):
        project.build_pipeline(builder)
        builder.set_node('scaler', grp='scale', exist='replace')
        latest = project.build_pipeline(builder)
        assert project.load_pipeline('main').build_id == latest.build_id

    def test_load_specific_version(self, project, builder):
        first = project.build_pipeline(builder)
        builder.set_node('scaler', grp='scale', exist='replace')
        project.build_pipeline(builder)
        assert project.load_pipeline('main', 1).build_id == first.build_id

    def test_loaded_pipeline_is_usable(self, project, builder):
        project.build_pipeline(builder)
        loaded = project.load_pipeline('main')
        assert loaded.topo_order() == ['scaler']
        assert loaded.get_node_spec('scaler').processor == \
            'sklearn.preprocessing.StandardScaler'

    def test_load_unknown_raises(self, project):
        with pytest.raises(KeyError):
            project.load_pipeline('nope')

    def test_list_versions(self, project, builder):
        project.build_pipeline(builder)
        builder.set_node('scaler', grp='scale', exist='replace')
        project.build_pipeline(builder)
        assert [r['version'] for r in project.list_pipeline_versions('main')] == [1, 2]


class TestTrialRegistration:
    def test_same_definition_registers_once(self, store):
        store.register(_trial())
        store.register(_trial())
        assert len(store.list_trials()) == 1

    def test_redefine_replaces_the_row(self, store):
        """name is the primary key — redefining it overwrites its row,
        exactly like redefining a Trial overwrites its on-disk artifact."""
        store.register(_trial(params={'max_depth': 3}))
        store.register(_trial(params={'max_depth': 5}))
        rows = [t for t in store.list_trials() if t['name'] == 'dt']
        assert len(rows) == 1
        assert rows[0]['params'] == {'max_depth': 5}

    def test_name_does_not_split_identity(self, store):
        """Two Trials with an otherwise-identical definition but different
        names still land in two separate rows — storage keys purely on name,
        with no notion of "same definition" collapsing them.
        """
        store.register(_trial('a'))
        store.register(_trial('b'))
        assert len(store.list_trials()) == 2

    def test_register_all_registers_every_trial(self, store):
        trials = make_trials('dt', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': [3, 5]})
        store.register_all(trials)
        assert {t['name'] for t in store.list_trials()} == {'dt_0', 'dt_1'}

    def test_has_before_and_after(self, store):
        assert not store.has(_trial())
        store.register(_trial())
        assert store.has(_trial())

    def test_has_false_after_redefinition(self, store):
        store.register(_trial(params={'max_depth': 3}))
        assert not store.has(_trial(params={'max_depth': 5}))

    def test_param_order_does_not_affect_the_comparison(self, store):
        """has() compares decoded values, not the stored JSON text — so the
        key order params happened to be written in cannot make an identical
        definition look changed."""
        store.register(_trial(params={'a': 1, 'b': 2}))
        assert store.has(_trial(params={'b': 2, 'a': 1}))

    def test_get_by_name_roundtrip(self, store):
        store.register(_trial(params={'max_depth': 7}))
        got = store.get_by_name('dt')
        assert got['processor'] == TREE
        assert got['params'] == {'max_depth': 7}
        assert got['edges'] == EDGES

    def test_get_by_name_unknown(self, store):
        assert store.get_by_name('nope') is None

    def test_list_trials(self, store):
        store.register(_trial('a'))
        store.register(_trial('b', params={'max_depth': 9}))
        assert len(store.list_trials()) == 2

    def test_survives_reopen(self, store, tmp_path):
        store.register(_trial())
        assert TrialStore(tmp_path / 'ts').get_by_name('dt') is not None


class TestExperimentHist:
    def test_record_and_read(self, store):
        store.record('dt', 'exp-1', 0, 0, pipeline_version='pk1', status='built')
        rows = store.get_hist(trial_name='dt')
        assert len(rows) == 1
        assert rows[0]['status'] == 'built'
        assert rows[0]['pipeline_version'] == 'pk1'

    def test_fold_coordinates_are_part_of_the_key(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-1', 0, 1, status='built')
        store.record('dt', 'exp-1', 1, 0, status='error')
        assert len(store.get_hist(trial_name='dt')) == 3

    def test_rerun_same_fold_overwrites(self, store):
        """Redefining a name overwrites its artifact, so it overwrites its row."""
        store.record('dt', 'exp-1', 0, 0, status='error')
        store.record('dt', 'exp-1', 0, 0, status='built')
        rows = store.get_hist(trial_name='dt')
        assert len(rows) == 1
        assert rows[0]['status'] == 'built'

    def test_same_name_across_experimenters_is_separate(self, store):
        store.record('dt', 'exp-1', 0, 0, pipeline_version='pk1', status='built')
        store.record('dt', 'exp-2', 0, 0, pipeline_version='pk2', status='built')
        assert {r['pipeline_version'] for r in store.get_hist(trial_name='dt')} == {'pk1', 'pk2'}

    def test_filter_by_experimenter(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-2', 0, 0, status='built')
        assert len(store.get_hist(experimenter='exp-2')) == 1

    def test_filter_by_pipeline_version(self, store):
        store.record('dt', 'exp-1', 0, 0, pipeline_version='pk1', status='built')
        store.record('dt', 'exp-2', 0, 0, pipeline_version='pk2', status='built')
        assert len(store.get_hist(pipeline_version='pk2')) == 1

    def test_get_status_map(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-1', 1, 0, status='error')
        assert store.get_status('dt', 'exp-1') == {(0, 0): 'built', (1, 0): 'error'}

    def test_remove_hist_by_experimenter(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-2', 0, 0, status='built')
        store.remove_hist(experimenter='exp-1')
        assert len(store.get_hist(trial_name='dt')) == 1

    def test_remove_hist_keeps_the_definition(self, store):
        store.register(_trial())
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.remove_hist(trial_name='dt')
        assert store.get_by_name('dt') is not None


class TestProjectEndToEnd:
    def test_hist_records_the_pipeline_version(self, project, builder):
        built = project.build_pipeline(builder)
        project.trials.register(_trial())
        project.trials.record('dt', 'exp-1', 0, 0,
                              pipeline_version=built.version, status='built')

        row = project.trials.get_hist(trial_name='dt')[0]
        assert row['pipeline_version'] == built.version
        assert project.trials.get_by_name('dt')['processor'] == TREE


@pytest.fixture
def sample_data():
    np.random.seed(0)
    n = 60
    return pd.DataFrame({'f1': np.random.randn(n),
                         'target': np.random.randint(0, 2, n)})


class TestExperimenterUnderProject:
    def _exp(self, project, builder, sample_data, name='run_a', version=None):
        if version is None:
            version = project.build_pipeline(builder).version
        return project.experimenter(
            name, sample_data,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0),
            pipeline_name='main', pipeline_version=version,
        )

    def test_name_is_the_directory(self, project, builder, sample_data):
        e = self._exp(project, builder, sample_data)
        assert e.path == project.path / 'exp' / 'run_a'
        assert e.name == 'run_a'

    def test_pipeline_comes_from_the_version(self, project, builder, sample_data):
        built = project.build_pipeline(builder)
        e = self._exp(project, builder, sample_data, version=built.version)
        assert e.pipeline_version == built.version
        assert e.pipeline.build_id == built.build_id

    def test_pipeline_is_kept_beside_the_run(self, project, builder, sample_data):
        """The run owns its Pipeline copy; the version stays as provenance."""
        e = self._exp(project, builder, sample_data)
        assert (e.path / 'pipeline.pkl').exists()
        assert e._store.fetch()['pipeline_version'] == e.pipeline_version

    def test_reopens_without_a_project(self, project, builder, sample_data):
        """The directory alone is enough — no Project, no version resolution,
        and the splitters come back with it."""
        from mllabs._experimenter import Experimenter
        e = self._exp(project, builder, sample_data)
        e.build()

        reopened = Experimenter.load_experimenter(e.path, sample_data)
        assert reopened.name == 'run_a'
        assert reopened.get_n_splits() == e.get_n_splits()
        assert reopened.pipeline is not None
        assert reopened.pipeline_version == e.pipeline_version
        assert reopened.pipeline.get_node_names() == e.pipeline.get_node_names()
        assert reopened.get_status('scaler') == 'built'

    def test_reload_survives_the_version_being_gone(self, project, builder, sample_data):
        """The run reads its own copy, so it reopens even if the project's
        stored version can no longer be loaded."""
        e = self._exp(project, builder, sample_data)
        e.build()
        for f in (project.pipeline_path('main')).glob('v*.pkl'):
            f.unlink()

        loaded = project.load_experimenter('run_a', sample_data)
        assert loaded.pipeline is not None
        assert loaded.get_status('scaler') == 'built'

    def test_no_pipeline_until_a_version_is_set(self, project, sample_data):
        e = project.experimenter('bare', sample_data)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            e.build()

    def test_switching_version_resets_stale_nodes(self, project, builder, sample_data):
        e = self._exp(project, builder, sample_data)
        e.build()
        assert e.get_status('scaler') == 'built'

        # Staleness is a value diff now (no serial) — 'replace' with the exact
        # same definition is correctly a no-op, so this needs an actual change.
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        v2 = project.build_pipeline(builder)
        e.set_pipeline(v2)
        assert e.get_status('scaler') is None

    def test_build_respects_stage_dependency_order(self, project, sample_data):
        """Exercises DataFlow.get_missing_nodes via _build_flow_single's
        readiness loop — s2 depends on s1 and can only build once s1 is."""
        p = project.pipeline_builder('chain')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = project.build_pipeline(p).version
        e = project.experimenter(
            'chained', sample_data,
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0),
            pipeline_name='chain', pipeline_version=version,
        )
        e.build()
        assert e.get_status('s1') == 'built'
        assert e.get_status('s2') == 'built'

    def test_build_respects_stage_dependency_order_multi_worker(self, project, sample_data):
        """Same as above but through _build_flow_multi's _collect_ready."""
        p = project.pipeline_builder('chain2')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = project.build_pipeline(p).version
        e = project.experimenter(
            'chained_multi', sample_data,
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0),
            pipeline_name='chain2', pipeline_version=version,
        )
        e.build(n_jobs=2)
        assert e.get_status('s1') == 'built'
        assert e.get_status('s2') == 'built'

    def test_multiple_experimenters_per_project(self, project, builder, sample_data):
        version = project.build_pipeline(builder).version
        self._exp(project, builder, sample_data, name='a', version=version)
        self._exp(project, builder, sample_data, name='b', version=version)
        assert project.list_experimenters() == ['a', 'b']

    def test_project_indexes_names_run_holds_the_rest(self, project, builder, sample_data):
        """ProjectStore answers 'which runs exist'; everything about a run
        lives in the run's own store."""
        e = self._exp(project, builder, sample_data)
        assert project.store.list_experimenters() == ['run_a']
        assert (e.path / '__exp.db').exists()
        assert e._store.fetch()['pipeline_version'] == e.pipeline_version

    def test_load_unknown_name_raises(self, project, sample_data):
        with pytest.raises(KeyError, match='No experimenter'):
            project.load_experimenter('nope', sample_data)

    def test_load_unknown_name_leaves_no_directory_behind(self, project, sample_data):
        with pytest.raises(KeyError):
            project.load_experimenter('nope', sample_data)
        assert not (project.path / 'exp' / 'nope' / '__exp.db').exists()

    def test_unknown_meta_column_rejected(self, project, builder, sample_data):
        e = self._exp(project, builder, sample_data)
        with pytest.raises(ValueError, match='Unknown experimenter meta column'):
            e._store.save({'name': 'run_a', 'bogus': 1})

    def test_remove_experimenter_from_the_index(self, project, builder, sample_data):
        self._exp(project, builder, sample_data)
        project.store.remove_experimenter('run_a')
        assert project.list_experimenters() == []

    def test_trainers_are_indexed_too(self, project, builder, sample_data):
        from mllabs._data_wrapper import wrap
        version = project.build_pipeline(builder).version
        project.trainer('t1', wrap(sample_data),
                        pipeline_name='main', pipeline_version=version)
        assert project.list_trainers() == ['t1']
        assert project.list_experimenters() == []

    def test_reload_restores_name_and_version(self, project, builder, sample_data):
        e = self._exp(project, builder, sample_data)
        e.build()
        loaded = project.load_experimenter('run_a', sample_data)
        assert loaded.name == 'run_a'
        assert loaded.pipeline_version == e.pipeline_version
        assert loaded.get_status('scaler') == 'built'

    def test_reload_checks_data_key(self, project, builder, sample_data):
        project.build_pipeline(builder)
        project.experimenter('keyed', sample_data, data_key='k1')
        with pytest.raises(ValueError, match='data_key mismatch'):
            project.load_experimenter('keyed', sample_data, data_key='wrong')

    def test_trials_run_and_land_in_hist(self, project, builder, sample_data):
        e = self._exp(project, builder, sample_data)
        e.build()
        trial = make_trials('dt', processor=TREE, edges=EDGES,
                             params={'max_depth': 3, 'random_state': 0})[0]
        e.exp([(trial, 0, 0), (trial, 1, 0)], project.trials)
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}

        # history is keyed by the two names, matching the layout on disk
        row = project.trials.get_hist(experimenter='run_a')[0]
        assert row['trial_name'] == 'dt'
        assert row['pipeline_version'] == e.pipeline_version

    def test_exp_skips_fold_already_built_in_hist(self, project, builder, sample_data):
        """_make_jobs consults TrialStore.experiment_hist, not the on-disk
        artifact — a fold recorded 'built' there is skipped without dispatch."""
        e = self._exp(project, builder, sample_data)
        e.build()
        trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0})
        e.exp([(trial, 0, 0)], project.trials)
        build_id_1 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']

        e.exp([(trial, 0, 0)], project.trials)
        build_id_2 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 == build_id_1

    def test_redefined_trial_with_built_hist_is_not_rerun(self, project, builder, sample_data):
        """Redefining a Trial no longer forces a rerun of folds the hist
        already marks 'built' — rerunning is an explicit action now."""
        e = self._exp(project, builder, sample_data)
        e.build()
        trial_v1 = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0})
        e.exp([(trial_v1, 0, 0)], project.trials)
        build_id_1 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']

        trial_v2 = Trial('dt', TREE, EDGES, params={'max_depth': 5, 'random_state': 0})
        e.exp([(trial_v2, 0, 0)], project.trials)
        build_id_2 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 == build_id_1

    def test_error_fold_is_retried(self, project, builder, sample_data):
        """NodeStore.status() can no longer see 'error' at all (obj.pkl was
        never written) — TrialStore.experiment_hist is the only place a
        Trial's error status/detail is recorded now."""
        e = self._exp(project, builder, sample_data)
        e.build()
        bad_trial = Trial('bad', 'mock.BadPredictor', EDGES)
        e.exp([(bad_trial, 0, 0)], project.trials)
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'
        build_id_1 = project.trials.get_info('bad', 'run_a')[(0, 0)]['build_id']

        e.exp([(bad_trial, 0, 0)], project.trials)
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'
        build_id_2 = project.trials.get_info('bad', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 != build_id_1

    def test_exp_multi_worker(self, project, builder, sample_data):
        """Experimenter.exp() through _execute_multi (n_jobs>1) — the merged
        Stage/Trial worker-pool executor, exercised here on its Trial/
        collectors path (unlike test_train_multi_worker, which only covers
        Trainer). One good and one failing trial across two folds so both
        the 'done' and 'error' worker messages are covered."""
        e = self._exp(project, builder, sample_data)
        e.build()
        trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0})
        bad_trial = Trial('bad', 'mock.BadPredictor', EDGES)
        e.exp([(trial, 0, 0), (trial, 1, 0), (bad_trial, 0, 0)], project.trials, n_jobs=2)
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'


class TestRemoveTrial:
    """A Trial leaves no artifact, so everything it produced sits in stores
    that don't know about each other — TrialStore (definition + history),
    CollectHist (per-fold collect outcomes) and each Collector's own data.
    Project is the only thing that sees all three."""

    def _run(self, project, builder, sample_data):
        version = project.build_pipeline(builder).version
        e = project.experimenter(
            'run_a', sample_data,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0),
            pipeline_name='main', pipeline_version=version,
        )
        e.build()
        collectors = project.collectors()
        collectors.set_collector('acc', MetricCollector, Connector(),
                                 params={'output_var': None, 'metric_func': dummy_metric})
        trial = _trial()
        e.exp([(trial, 0, 0), (trial, 1, 0)], project.trials, collectors=collectors)
        return e, collectors

    def test_store_remove_drops_the_definition_only(self, store):
        store.register(_trial())
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.remove('dt')
        assert store.get_by_name('dt') is None
        assert store.list_trials() == []
        assert store.get_hist(trial_name='dt')  # what ran stays readable

    def test_removing_an_unregistered_name_is_a_no_op(self, project):
        project.remove_trial('never_registered')

    def test_removes_definition_history_and_collected_data(self, project, builder, sample_data):
        e, collectors = self._run(project, builder, sample_data)
        assert project.trials.get_by_name('dt') is not None
        assert project.trials.get_status('dt', 'run_a')
        assert collectors.hist.get_hist(node_name='dt')
        assert collectors.get_collector('acc').get_metric('dt') is not None

        project.remove_trial('dt')

        assert project.trials.get_by_name('dt') is None
        assert project.trials.get_status('dt', 'run_a') == {}
        assert collectors.hist.get_hist(node_name='dt') == []
        assert project.collectors().get_collector('acc').get_metric('dt') is None

    def test_leaves_other_trials_alone(self, project, builder, sample_data):
        e, collectors = self._run(project, builder, sample_data)
        other = _trial('dt2')
        e.exp([(other, 0, 0), (other, 1, 0)], project.trials, collectors=collectors)

        project.remove_trial('dt')

        assert project.trials.get_by_name('dt2') is not None
        assert project.trials.get_status('dt2', 'run_a')
        assert collectors.hist.get_hist(node_name='dt2')
        assert project.collectors().get_collector('acc').get_metric('dt2') is not None

    def test_cleans_the_registry_it_is_given(self, project, builder, sample_data):
        """collectors() builds a fresh registry each call, and a Collector may
        answer from an in-memory cache — so the caller's own registry has to be
        cleanable, not just whatever Project happens to construct."""
        e, collectors = self._run(project, builder, sample_data)
        acc = collectors.get_collector('acc')
        assert acc.has_node('dt')
        project.remove_trial('dt', collectors=collectors)
        assert not acc.has_node('dt')

    def test_a_removed_trial_runs_again_from_scratch(self, project, builder, sample_data):
        e, collectors = self._run(project, builder, sample_data)
        project.remove_trial('dt')
        trial = _trial()
        e.exp([(trial, 0, 0), (trial, 1, 0)], project.trials, collectors=collectors)
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}


class TestTrainerUnderProject:
    """Exercises Trainer.train() -> Job/_execute_* through the same job-based
    path Experimenter.build() uses."""

    def test_train_respects_node_dependency_order(self, project, sample_data):
        from mllabs._data_wrapper import wrap

        p = project.pipeline_builder('trainer_chain')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = project.build_pipeline(p).version

        t = project.trainer('trained_chain', wrap(sample_data),
                            pipeline_name='trainer_chain', pipeline_version=version)
        t.train([Predictor('dt', TREE, {'X': 's2:(*)', 'y': '{target}'},
                           params={'max_depth': 3, 'random_state': 0})])
        assert t.get_status('s1') == 'built'
        assert t.get_status('s2') == 'built'
        assert t.get_status('dt') == 'built'

    def test_train_multi_worker(self, project, sample_data):
        from mllabs._data_wrapper import wrap

        p = project.pipeline_builder('trainer_chain_multi')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = project.build_pipeline(p).version

        t = project.trainer('trained_chain_multi', wrap(sample_data),
                            pipeline_name='trainer_chain_multi', pipeline_version=version)
        t.train([Predictor('dt', TREE, {'X': 's2:(*)', 'y': '{target}'},
                           params={'max_depth': 3, 'random_state': 0})], n_jobs=2)
        assert t.get_status('s1') == 'built'
        assert t.get_status('s2') == 'built'
        assert t.get_status('dt') == 'built'

    def test_reload_recovers_built_nodes_via_history(self, project, sample_data):
        """A reloaded Trainer reattaches the nodes it already built: load()
        recovers each one's processor *and* its edges from the store's
        history, so the flow can route data through the chain again.
        """
        from mllabs._data_wrapper import wrap

        p = project.pipeline_builder('trainer_reload_chain')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = project.build_pipeline(p).version

        predictor = Predictor('dt', TREE, {'X': 's2:(*)', 'y': '{target}'},
                              params={'max_depth': 3, 'random_state': 0})
        t = project.trainer('trainer_reload', wrap(sample_data),
                            pipeline_name='trainer_reload_chain', pipeline_version=version)
        t.train([predictor])

        loaded = project.load_trainer('trainer_reload', wrap(sample_data))
        flow = loaded.train_folds[0].train_data_flows[0]
        assert 's1' in flow.node_objs
        assert 's2' in flow.node_objs

        # The selection comes back from PredictorStore, so nothing needs
        # re-supplying and nothing is reset by reopening.
        assert loaded.predictor_names() == ['dt']
        assert loaded.selected_nodes == ['s1', 's2']
        assert loaded.get_status('dt') == 'built'

        # Proves _node_edges (recovered from the store's history) actually
        # routes data through s1 -> s2, not just that obj.pkl happened to load.
        train_data = flow.get_train({'X': 's2:(*)', 'y': '{target}'})
        assert train_data['X'].get_shape()[0] > 0

    def test_trainer_reopens_without_a_project(self, project, sample_data):
        """Trainer.load_trainer() reads the Pipeline the Trainer itself saved
        — the caller resolves no version."""
        from mllabs._data_wrapper import wrap
        from mllabs._trainer import Trainer

        p = project.pipeline_builder('trainer_standalone')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        version = project.build_pipeline(p).version

        t = project.trainer('t_standalone', wrap(sample_data),
                            pipeline_name='trainer_standalone', pipeline_version=version)
        t.train([Predictor('dt', TREE, {'X': 's1:(*)', 'y': '{target}'},
                           params={'max_depth': 3, 'random_state': 0})])
        assert (t.path / 'pipeline.pkl').exists()

        reopened = Trainer.load_trainer(t.path, wrap(sample_data))
        assert reopened.pipeline is not None
        assert reopened.pipeline_version == version
        assert reopened.selected_nodes == ['s1']
        assert reopened.get_status('dt') == 'built'
