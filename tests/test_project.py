import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from mllabs._experimenter import Experimenter
from mllabs import (Project, TrialStore, Trial, GridTrials, Predictor, PipelineBuilder,
                    Collectors, Connector, MetricCollector)


TREE = 'sklearn.tree.DecisionTreeClassifier'
EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


@pytest.fixture
def sample_data():
    np.random.seed(0)
    n = 60
    return pd.DataFrame({'f1': np.random.randn(n),
                         'target': np.random.randint(0, 2, n)})


@pytest.fixture
def project(tmp_path, sample_data):
    return Project(tmp_path / 'proj', data=sample_data)


@pytest.fixture
def builder(project):
    p = project.pipeline
    p.set_datasource({'f1': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1}'})
    p.set_node('scaler', grp='scale')
    return p


@pytest.fixture
def store(tmp_path):
    return TrialStore(tmp_path / 'ts')


def _trial(name='dt', params=None, pipeline_version=None, processor=TREE,
           desc=None, tag=None):
    return Trial(name, processor, EDGES, params=params or {'max_depth': 3},
                 pipeline_version=pipeline_version, desc=desc, tag=tag)


def _named_grid(names, **grid_kw):
    """A GridTrials sweep wrapped into Trials under explicit *names* — for
    tests that need a fixed, predictable batch. Real naming (next_name) is
    Project.make_trials's job, exercised in TestMakeTrials."""
    combos = GridTrials(processor=TREE, edges=EDGES, **grid_kw).combos()
    return [Trial(name=n, **c) for n, c in zip(names, combos)]


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
        for path in (project.pipeline_path, project.exp_path('b'),
                     project.trainer_path('c')):
            assert project.path in path.parents or path.parent == project.path

    def test_no_inferencer_path(self, project):
        """An Inferencer is reassembled from the Trainer on demand, so the
        project neither stores one nor owns a path for it."""
        assert not hasattr(project, 'inferencer_path')

    def test_paths_are_created(self, project):
        assert project.exp_path('r1').is_dir()

    def test_exp_path_is_under_exp_folder(self, project):
        assert project.exp_path('run_a') == project.path / 'exp' / 'run_a'

    def test_one_pipeline_per_project(self, project):
        assert project.pipeline._store.db_path == project.pipeline_path / 'pipeline.db'

    def test_the_same_builder_comes_back(self, project):
        """Two builders over one db would let two in-memory copies of the
        definitions drift apart."""
        assert project.pipeline is project.pipeline

    def test_the_builder_persists(self, project, sample_data):
        project.pipeline.set_grp('g', processor=TREE, method='predict', edges={'X': '{f1}'})
        again = Project(project.path, data=sample_data)
        assert 'g' in again.pipeline.grps

    def test_the_dataset_is_the_projects(self, project, sample_data, tmp_path):
        """The one thing an Experimenter cannot restore from its own directory,
        so without it here every question needs the caller to bring a frame."""
        reopened = Project(project.path)
        assert reopened.data.shape == sample_data.shape

    def test_no_data_is_an_error_not_a_guess(self, tmp_path):
        empty = Project(tmp_path / 'empty')
        assert empty.data is None
        with pytest.raises(ValueError, match='no data'):
            empty.set_experimenter('e')

    def test_project_owns_no_collector_registry(self, project, sample_data):
        """A registry belongs to the Experimenter that writes into it — Collector
        data is keyed by node name alone, so a project-wide one would have two
        Experimenters overwrite each other on any Trial name they share."""
        assert not hasattr(project, 'collectors')
        assert not (project.path / 'collectors').exists()

        e = project.set_experimenter('run_a')
        assert e.collectors.path == e.path / 'collectors'


class TestPipelineVersions:
    """Building publishes. A version appears exactly when the definition changes."""

    def test_first_build_is_v1(self, project, builder):
        built = builder.build()
        assert (built.version, built.status) == (1, 'published')

    def test_an_unchanged_rebuild_is_the_same_version(self, project, builder):
        a = builder.build()
        b = builder.build()
        assert b.version == a.version
        assert a.build_id != b.build_id      # a different build, the same definition
        assert len(project.list_pipeline_versions()) == 2   # v0 and it

    def test_an_edit_mints_the_next_version(self, project, builder):
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        assert builder.build().version == 2

    def test_publishing_demotes_nothing(self, project, builder):
        """Every stored version stays published and stays adoptable; the newest
        is only what an omitted version number resolves to."""
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        builder.build()
        assert [r['version'] for r in project.list_pipeline_versions()] == [0, 1, 2]
        assert all(project.pipeline._store.get_status(v) == 'published'
                   for v in (0, 1, 2))

    def test_a_builder_with_no_db_cannot_build(self):
        """Nowhere to publish, so build() says so rather than quietly handing
        back something unnumbered. draft() is how you get the snapshot."""
        bare = PipelineBuilder()
        bare.set_datasource({'f1': 'numerical'})
        with pytest.raises(ValueError, match='no db'):
            bare.build()
        drafted = bare.draft()
        assert (drafted.version, drafted.status) == (None, 'draft')

    def test_a_new_project_starts_published_at_v0(self, project):
        """The empty Pipeline is a real published row, not an absence — so
        load_pipeline() always has something to return."""
        assert [r['version'] for r in project.list_pipeline_versions()] == [0]
        assert project.load_pipeline().is_empty

    def test_building_an_untouched_working_copy_stays_at_v0(self, project):
        """Nothing was defined, so nothing changed, so no version appears."""
        assert project.pipeline.build().version == 0
        assert len(project.list_pipeline_versions()) == 1

    def test_load_without_a_version_is_the_published_one(self, project, builder):
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        assert project.load_pipeline().version == 1   # the edit is not published yet
        assert builder.build().version == 2
        assert project.load_pipeline().version == 2

    def test_load_does_not_mint(self, project, builder):
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        before = len(project.list_pipeline_versions())
        project.load_pipeline()
        assert len(project.list_pipeline_versions()) == before

    def test_a_datasource_only_change_still_mints(self, project, builder):
        """diff_from answers 'what artifacts to reset' and so returns node
        names only; a schema change no node reads stales nothing but is still
        a different definition."""
        p = PipelineBuilder(path=project.pipeline_path, name='pipeline')
        p.set_datasource({'f1': 'numerical'})
        first = p.build().version
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical'})
        assert p.build().version == first + 1

    def test_load_specific_version(self, project, builder):
        first = builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        builder.build()
        assert project.load_pipeline(1).build_id == first.build_id

    def test_loaded_pipeline_is_usable(self, project, builder):
        builder.build()
        loaded = project.load_pipeline(1)
        assert loaded.topo_order() == ['scaler']
        assert loaded.get_node_spec('scaler').processor == \
            'sklearn.preprocessing.StandardScaler'

    def test_load_unknown_raises(self, project, builder):
        builder.build()
        with pytest.raises(KeyError):
            project.load_pipeline(99)

    def test_an_older_version_can_be_removed(self, project, builder):
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        builder.build()
        project.remove_pipeline_version(1)
        assert [r['version'] for r in project.list_pipeline_versions()] == [0, 2]

    def test_the_latest_version_cannot_be_removed(self, project, builder):
        """It is what an omitted version resolves to, so removing it would move
        that pointer with nothing in the call to say so."""
        builder.build()
        with pytest.raises(ValueError, match='latest'):
            project.remove_pipeline_version(1)


class TestSetTrial:
    """Authoring a Trial is a project-level act, separate from running one.

    Registration used to be a side effect of Experimenter.exp(), so a Trial
    could not enter the project without being executed, and a rerun could
    redefine a name whose history described the old definition."""

    def test_adding_returns_the_name(self, project):
        assert project.set_trial(_trial()) == 'dt'
        assert project.trials.get_by_name('dt') is not None

    def test_an_unchanged_definition_returns_none(self, project):
        project.set_trial(_trial())
        assert project.set_trial(_trial()) is None

    def test_changing_an_unrun_trial_is_allowed(self, project):
        project.set_trial(_trial(params={'max_depth': 3}))
        assert project.set_trial(_trial(params={'max_depth': 5})) == 'dt'
        assert project.trials.get_by_name('dt').params == {'max_depth': 5}

    def test_a_trial_with_a_successful_run_is_frozen(self, project):
        project.set_trial(_trial(params={'max_depth': 3}))
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        with pytest.raises(ValueError, match='run_a'):
            project.set_trial(_trial(params={'max_depth': 5}))
        assert project.trials.get_by_name('dt').params == {'max_depth': 3}

    def test_resetting_the_same_definition_stays_allowed(self, project):
        """The guard is about the definition changing, not about touching it."""
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        assert project.set_trial(_trial()) is None

    def test_a_failed_run_does_not_freeze_it(self, project):
        """Only a success is a result worth protecting; an error leaves nothing
        the history would misdescribe."""
        project.set_trial(_trial(params={'max_depth': 3}))
        project.trials.record('dt', 'run_a', 0, 0, status='error')
        assert project.set_trial(_trial(params={'max_depth': 5})) == 'dt'

    def test_removing_it_lifts_the_freeze(self, project):
        project.set_trial(_trial(params={'max_depth': 3}))
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        project.remove_trial('dt')
        assert project.set_trial(_trial(params={'max_depth': 5})) == 'dt'

    def test_set_trials_returns_only_what_changed(self, project):
        trials = _named_grid(['dt_0', 'dt_1'], param_grid={'max_depth': [3, 5]})
        assert project.set_trials(trials) == ['dt_0', 'dt_1']
        assert project.set_trials(trials) == []

    def test_a_frozen_name_leaves_the_whole_batch_unwritten(self, project):
        """Checked before anything is written, so a batch does not land half
        registered — the returned work list would then be a lie."""
        project.set_trial(_trial('dt_0', params={'max_depth': 3}))
        project.trials.record('dt_0', 'run_a', 0, 0, status='built')
        trials = _named_grid(['dt_0', 'dt_1'], param_grid={'max_depth': [9, 11]})
        with pytest.raises(ValueError, match='dt_0'):
            project.set_trials(trials)
        assert project.trials.get_by_name('dt_1') is None
        assert project.trials.get_by_name('dt_0').params == {'max_depth': 3}


class TestSetTrialStampsTheVersion:
    """A Trial is defined against a Pipeline version, and this is where it is
    filled in — the only place that both knows the registry and owns the
    definition."""

    def test_an_unstamped_trial_gets_the_latest_published(self, project, builder):
        assert builder.build().version == 1
        assert project.set_trial(_trial()) == 'dt'
        assert project.trials.get_by_name('dt').pipeline_version == 1

    def test_a_new_project_stamps_v0(self, project):
        """Nothing built yet, so the empty Pipeline is what a Trial is against."""
        project.set_trial(_trial())
        assert project.trials.get_by_name('dt').pipeline_version == 0

    def test_the_caller_s_object_is_stamped_too(self, project, builder):
        """Mutated rather than copied, so the Trial still in hand says the same
        thing the row does."""
        builder.build()
        trial = _trial()
        project.set_trial(trial)
        assert trial.pipeline_version == 1

    def test_an_explicit_older_version_is_kept(self, project, builder):
        builder.build()
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        assert builder.build().version == 2
        project.set_trial(_trial(pipeline_version=1))
        assert project.trials.get_by_name('dt').pipeline_version == 1

    def test_an_unpublished_version_is_refused(self, project, builder):
        builder.build()
        with pytest.raises(ValueError, match='has not.*published'):
            project.set_trial(_trial(pipeline_version=7))
        assert project.trials.get_by_name('dt') is None

    def test_a_bumped_version_makes_a_succeeded_name_a_redefinition(self, project, builder):
        """The stamp is part of the definition, so after the pipeline moves on
        the freeze catches a re-registration instead of quietly re-pointing
        results at a version that did not produce them."""
        builder.build()
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='built')

        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        builder.build()
        with pytest.raises(ValueError, match='run_a'):
            project.set_trial(_trial())
        assert project.trials.get_by_name('dt').pipeline_version == 1

    def test_a_sweep_can_be_authored_against_one_version(self, project, builder):
        builder.build()
        gen = GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]})
        project.make_trials('dt', gen, pipeline_version=0)
        assert [t.pipeline_version for t in project.trials.list_trials()] == [0, 0]


class TestMakeTrials:
    """Project.make_trials — a generator's combos, named and registered.

    The naming defect this replaces: the old free-function make_trials
    derived names from grid position (index + a count-dependent zero-pad
    width), so growing a sweep silently renamed every sibling, or worse,
    repointed a name at a different combo. Assign-based naming
    (TrialStore.next_name) makes that structurally impossible — a name is
    never a function of anything but "give me the next one."
    """

    def test_registers_every_combo(self, project):
        gen = GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]})
        names = project.make_trials('dt', gen)
        assert len(names) == 2
        assert {project.trials.get_by_name(n).params['max_depth'] for n in names} == {3, 5}

    def test_names_are_minted_not_derived(self, project):
        gen = GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]})
        names = project.make_trials('dt', gen)
        assert all(n.startswith('dt') for n in names)
        assert len(set(names)) == 2

    def test_growing_the_grid_never_touches_earlier_names(self, project):
        first = project.make_trials(
            'dt', GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]}))
        project.trials.record(first[0], 'run_a', 0, 0, status='built')
        before = project.trials.get_by_name(first[0]).params

        project.make_trials(
            'dt', GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5, 7]}))

        assert project.trials.get_by_name(first[0]).params == before
        assert project.trials.get_hist(trial_name=first[0], status='built')

    def test_pipeline_version_applied_to_every_trial(self, project, builder):
        builder.build()
        gen = GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]})
        names = project.make_trials('dt', gen, pipeline_version=0)
        assert {project.trials.get_by_name(n).pipeline_version for n in names} == {0}

    def test_unstamped_pipeline_version_gets_latest_published(self, project, builder):
        builder.build()
        gen = GridTrials(processor=TREE, edges=EDGES, param_grid={'max_depth': [3, 5]})
        names = project.make_trials('dt', gen)
        assert {project.trials.get_by_name(n).pipeline_version for n in names} == {1}


class TestSetCollector:
    """Project.set_collector — a proxy onto the named Experimenter's own
    registry, not a Collectors registry Project owns (see
    TestProjectLayout.test_project_owns_no_collector_registry)."""

    def test_registers_on_the_named_experimenter(self, project, sample_data):
        e = project.set_experimenter('run_a')
        mc = project.set_collector(
            'run_a', 'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None,
                    'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}})
        assert e.collectors.get_collector('acc') is mc

    def test_unknown_experimenter_raises(self, project):
        with pytest.raises(KeyError, match='nope'):
            project.set_collector('nope', 'acc', 'mllabs.MetricCollector',
                                  'mllabs._connector.Connector')


class TestGetCollector:
    """Project.get_collector — the read-side counterpart to set_collector,
    same proxy relationship (no registry of its own)."""

    def test_reads_back_the_registered_one(self, project, sample_data):
        project.set_experimenter('run_a')
        mc = project.set_collector(
            'run_a', 'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None,
                    'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}})
        assert project.get_collector('run_a', 'acc') is mc

    def test_unregistered_name_returns_none(self, project, sample_data):
        project.set_experimenter('run_a')
        assert project.get_collector('run_a', 'nope') is None

    def test_unknown_experimenter_raises(self, project):
        with pytest.raises(KeyError, match='nope'):
            project.get_collector('nope', 'acc')


class TestProjectBuildProxy:
    """Project.build — a thin proxy onto Experimenter.build, sparing the
    project.experimenters[name].build(...) two-hop reach."""

    def test_builds_the_named_experimenter(self, project, builder, sample_data):
        version = builder.build().version
        e = project.set_experimenter('run_a', pipeline_version=version)
        project.build('run_a')
        assert e.get_status('scaler') == 'built'

    def test_unknown_experimenter_raises(self, project):
        with pytest.raises(KeyError, match='nope'):
            project.build('nope')


class TestProjectExpProxy:
    """Project.exp — a thin proxy onto Experimenter.exp, sparing the
    project.experimenters[name].exp(...) two-hop reach."""

    def test_runs_trials_on_the_named_experimenter(self, project, builder, sample_data):
        version = builder.build().version
        e = project.set_experimenter(
            'run_a', sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0),
            pipeline_version=version,
        )
        e.build()
        project.set_trial(_trial())
        project.exp('run_a', ['dt'])
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built'}

    def test_unknown_experimenter_raises(self, project):
        with pytest.raises(KeyError, match='nope'):
            project.exp('nope', ['dt'])


class TestChainTrial:
    """Project.chain_trial — chain() plus registration, in one call."""

    def test_registers_the_derived_trial(self, project):
        project.set_trial(_trial())
        assert project.chain_trial('dt', 'dt_stk') == 'dt_stk'
        assert project.trials.get_by_name('dt_stk').src_trial == 'dt'

    def test_unknown_source_raises(self, project):
        with pytest.raises(KeyError, match='nope'):
            project.chain_trial('nope', 'x')

    def test_name_left_unset_is_minted(self, project):
        project.set_trial(_trial())
        name = project.chain_trial('dt')
        assert name is not None and name != 'dt'
        assert project.trials.get_by_name(name).src_trial == 'dt'

    def test_overrides_pass_through(self, project):
        project.set_trial(_trial(params={'max_depth': 3}))
        project.chain_trial('dt', 'dt_stk', params={'max_depth': 9})
        assert project.trials.get_by_name('dt_stk').params == {'max_depth': 9}

    def test_pipeline_version_defaults_to_latest_published(self, project, builder):
        builder.build()
        project.set_trial(_trial(pipeline_version=0))
        project.chain_trial('dt', 'dt_stk')
        assert project.trials.get_by_name('dt_stk').pipeline_version == 1

    def test_pipeline_version_minus_one_inherits_the_source(self, project, builder):
        builder.build()
        project.set_trial(_trial(pipeline_version=0))
        project.chain_trial('dt', 'dt_stk', pipeline_version=-1)
        assert project.trials.get_by_name('dt_stk').pipeline_version == 0

    def test_freeze_gate_still_applies(self, project):
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        project.set_trial(_trial('dt_stk', params={'max_depth': 9}))
        project.trials.record('dt_stk', 'run_a', 0, 0, status='built')
        with pytest.raises(ValueError, match='dt_stk'):
            project.chain_trial('dt', 'dt_stk', params={'max_depth': 3})


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
        rows = [t for t in store.list_trials() if t.name == 'dt']
        assert len(rows) == 1
        assert rows[0].params == {'max_depth': 5}

    def test_name_does_not_split_identity(self, store):
        """Two Trials with an otherwise-identical definition but different
        names still land in two separate rows — storage keys purely on name,
        with no notion of "same definition" collapsing them.
        """
        store.register(_trial('a'))
        store.register(_trial('b'))
        assert len(store.list_trials()) == 2

    def test_register_all_registers_every_trial(self, store):
        trials = _named_grid(['dt_0', 'dt_1'], param_grid={'max_depth': [3, 5]})
        store.register_all(trials)
        assert {t.name for t in store.list_trials()} == {'dt_0', 'dt_1'}

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
        assert got.processor == TREE
        assert got.params == {'max_depth': 7}
        assert got.edges == EDGES

    def test_get_by_name_unknown(self, store):
        assert store.get_by_name('nope') is None

    def test_list_trials(self, store):
        store.register(_trial('a'))
        store.register(_trial('b', params={'max_depth': 9}))
        assert len(store.list_trials()) == 2

    def test_survives_reopen(self, store, tmp_path):
        store.register(_trial())
        assert TrialStore(tmp_path / 'ts').get_by_name('dt') is not None

    def test_src_trial_roundtrips(self, store):
        chained = _trial().chain('dt_stk')
        store.register(chained)
        assert store.get_by_name('dt_stk').src_trial == 'dt'

    def test_src_trial_is_not_part_of_the_definition(self, store):
        """Provenance, not identity — has() ignores it like desc/tag."""
        store.register(_trial())
        with_src = _trial()
        with_src.src_trial = 'somewhere-else'
        assert store.has(with_src)


class TestListTrialsFiltering:
    """list_trials() filters, AND-combined, all done in SQL."""

    LGBM = 'lightgbm.LGBMClassifier'

    def test_no_filters_returns_everything(self, store):
        store.register(_trial('a'))
        store.register(_trial('b', processor=self.LGBM))
        assert {t.name for t in store.list_trials()} == {'a', 'b'}

    def test_filter_by_processor(self, store):
        store.register(_trial('a'))
        store.register(_trial('b', processor=self.LGBM))
        assert [t.name for t in store.list_trials(processor=self.LGBM)] == ['b']

    def test_filter_by_name(self, store):
        store.register(_trial('a'))
        store.register(_trial('b'))
        assert [t.name for t in store.list_trials(name='a')] == ['a']

    def test_filter_by_pipeline_version(self, store):
        store.register(_trial('a', pipeline_version=0))
        store.register(_trial('b', pipeline_version=1))
        assert [t.name for t in store.list_trials(pipeline_version=1)] == ['b']

    def test_filter_by_src_trial(self, store):
        store.register(_trial('a'))
        store.register(_trial('a').chain('a_stk'))
        assert [t.name for t in store.list_trials(src_trial='a')] == ['a_stk']

    def test_filter_by_desc_is_a_substring_match(self, store):
        store.register(_trial('a', desc='xgboost baseline'))
        store.register(_trial('b', desc='lgbm tuned'))
        assert [t.name for t in store.list_trials(desc='base')] == ['a']

    def test_filter_by_tag_is_exact_membership(self, store):
        """tag='b' must not match a Trial tagged 'ab' — a raw substring
        search on the stored JSON text would."""
        store.register(_trial('a', tag=['ab']))
        store.register(_trial('b', tag=['b', 'other']))
        assert [t.name for t in store.list_trials(tag='b')] == ['b']

    def test_filters_combine_with_and(self, store):
        store.register(_trial('a', processor=self.LGBM, pipeline_version=0))
        store.register(_trial('b', processor=self.LGBM, pipeline_version=1))
        store.register(_trial('c', processor=TREE, pipeline_version=1))
        assert [t.name for t in store.list_trials(processor=self.LGBM, pipeline_version=1)] == ['b']

    def test_no_match_returns_empty_list(self, store):
        store.register(_trial('a'))
        assert store.list_trials(processor=self.LGBM) == []


class TestTrialStoreNaming:
    def test_next_seq_increases(self, store):
        first = store.next_seq()
        assert store.next_seq() == first + 1

    def test_next_seq_persists_across_reopen(self, store, tmp_path):
        store.next_seq()
        store.next_seq()
        reopened = TrialStore(tmp_path / 'ts')
        assert reopened.next_seq() == 3

    def test_next_name_keeps_the_prefix_of_a_full_name(self, store):
        assert store.next_name('lgb5').startswith('lgb')

    def test_next_name_accepts_a_bare_prefix(self, store):
        assert store.next_name('lgb').startswith('lgb')

    def test_next_name_appends_the_sequence_value(self, store):
        seq = store.next_seq()
        assert store.next_name('lgb') == f'lgb{seq + 1}'

    def test_next_name_never_repeats(self, store):
        assert store.next_name('lgb') != store.next_name('lgb')

    def test_next_name_does_not_derive_from_existing_rows(self, store):
        """A naive 'highest existing number + 1' would return 'lgb10' here —
        the actual value comes from the store's own global counter, which
        knows nothing about what is registered under the prefix."""
        store.register(_trial('lgb9'))
        assert store.next_name('lgb') != 'lgb10'


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
        built = builder.build()
        project.trials.register(_trial())
        project.trials.record('dt', 'exp-1', 0, 0,
                              pipeline_version=built.version, status='built')

        row = project.trials.get_hist(trial_name='dt')[0]
        assert row['pipeline_version'] == built.version
        assert project.trials.get_by_name('dt').processor == TREE


class TestExperimenterUnderProject:
    def _exp(self, project, builder, name='run_a', version=None):
        if version is None:
            version = builder.build().version
        return project.set_experimenter(
            name,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0), pipeline_version=version,
        )

    def test_name_is_the_directory(self, project, builder, sample_data):
        e = self._exp(project, builder)
        assert e.path == project.path / 'exp' / 'run_a'
        assert e.name == 'run_a'

    def test_pipeline_comes_from_the_version(self, project, builder, sample_data):
        built = builder.build()
        e = self._exp(project, builder, version=built.version)
        assert e.pipeline_version == built.version
        assert e.pipeline.build_id == built.build_id

    def test_pipeline_is_kept_beside_the_run(self, project, builder, sample_data):
        """The run owns its Pipeline copy; the version stays as provenance."""
        e = self._exp(project, builder)
        assert (e.path / 'pipeline.pkl').exists()
        assert e._store.fetch()['pipeline_version'] == e.pipeline_version

    def test_reopens_without_a_project(self, project, builder, sample_data):
        """The directory alone is enough — no Project, no version resolution,
        and the splitters come back with it."""
        from mllabs._experimenter import Experimenter
        e = self._exp(project, builder)
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
        e = self._exp(project, builder)
        e.build()
        for f in project.pipeline_path.glob('v*.pkl'):
            f.unlink()

        loaded = Project(project.path, data=sample_data).experimenters['run_a']
        assert loaded.pipeline is not None
        assert loaded.get_status('scaler') == 'built'

    def test_adding_without_a_version_adopts_the_published_one(self, project, builder,
                                                              sample_data):
        """The published version, not the working copy — adding an
        Experimenter is not an occasion to mint one."""
        published = builder.build().version
        e = project.set_experimenter('bare')
        assert e.pipeline_version == published
        assert len(project.list_pipeline_versions()) == 2   # v0 and this one

    def test_adding_before_anything_is_built_adopts_v0(self, project, sample_data):
        """Version 0 is the empty Pipeline every project is created with, so
        there is always something to adopt and nothing to special-case."""
        e = project.set_experimenter('bare')
        assert e.pipeline_version == 0
        assert e.pipeline.is_empty
        e.build()                       # nothing to build, and that is not an error

    def test_a_trainer_takes_the_published_one_too(self, project, builder, sample_data):
        """Published is frozen, which is exactly what a Trainer is allowed to
        adopt — the default never hands it the working copy."""
        published = builder.build().version
        t = project.set_trainer('bare_t')
        assert t.pipeline_version == published

    def test_switching_version_resets_stale_nodes(self, project, builder, sample_data):
        e = self._exp(project, builder)
        e.build()
        assert e.get_status('scaler') == 'built'

        # Staleness is a value diff now (no serial) — 'replace' with the exact
        # same definition is correctly a no-op, so this needs an actual change.
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        v2 = builder.build()
        e.set_pipeline(v2)
        assert e.get_status('scaler') is None

    def test_build_respects_stage_dependency_order(self, project, sample_data):
        """Exercises DataFlow.get_missing_nodes via _build_flow_single's
        readiness loop — s2 depends on s1 and can only build once s1 is."""
        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = p.build().version
        e = project.set_experimenter(
            'chained',
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0), pipeline_version=version,
        )
        e.build()
        assert e.get_status('s1') == 'built'
        assert e.get_status('s2') == 'built'

    def test_build_respects_stage_dependency_order_multi_worker(self, project, sample_data):
        """Same as above but through _build_flow_multi's _collect_ready."""
        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = p.build().version
        e = project.set_experimenter(
            'chained_multi',
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0), pipeline_version=version,
        )
        e.build(n_jobs=2)
        assert e.get_status('s1') == 'built'
        assert e.get_status('s2') == 'built'

    def test_multiple_experimenters_per_project(self, project, builder, sample_data):
        version = builder.build().version
        self._exp(project, builder, name='a', version=version)
        self._exp(project, builder, name='b', version=version)
        assert project.list_experimenters() == ['a', 'b']

    def test_project_indexes_names_run_holds_the_rest(self, project, builder, sample_data):
        """ProjectStore answers 'which runs exist'; everything about a run
        lives in the run's own store."""
        e = self._exp(project, builder)
        assert project.store.list_experimenters() == ['run_a']
        assert (e.path / '__exp.db').exists()
        assert e._store.fetch()['pipeline_version'] == e.pipeline_version

    def test_an_unmanaged_name_is_not_in_the_registry(self, project, sample_data):
        with pytest.raises(KeyError):
            project.experimenters['nope']

    def test_the_added_one_is_the_one_the_registry_gives_back(self, project, builder,
                                                              sample_data):
        """Two live Experimenters over one directory would each hold their own
        Collectors and node caches, so a change through one would be invisible
        to the other."""
        e = self._exp(project, builder)
        assert project.experimenters['run_a'] is e

    def test_the_registry_opens_only_what_it_is_not_holding(self, project, builder,
                                                            sample_data):
        e = self._exp(project, builder, name='a')
        reopened = Project(project.path, data=sample_data)
        b = reopened.set_experimenter('b')
        assert reopened.experimenters['b'] is b
        assert reopened.experimenters['a'].name == 'a'
        assert reopened.experimenters['a'] is not e

    def test_asking_for_one_leaves_no_directory_behind(self, project, sample_data):
        """Reading the registry must not bring into being what it is asked about."""
        assert 'nope' not in project.experimenters
        assert not (project.path / 'exp' / 'nope' / '__exp.db').exists()

    def test_unknown_meta_column_rejected(self, project, builder, sample_data):
        e = self._exp(project, builder)
        with pytest.raises(ValueError, match='Unknown experimenter meta column'):
            e._store.save({'name': 'run_a', 'bogus': 1})

    def test_remove_experimenter(self, project, builder, sample_data):
        self._exp(project, builder)
        project.remove_experimenter('run_a')
        assert project.list_experimenters() == []
        assert not (project.path / 'exp' / 'run_a').exists()

    def test_remove_experimenter_takes_its_trial_history_with_it(self, project, builder,
                                                                sample_data):
        """The name keys the history, so leaving it would hand the rows to
        whatever is added under that name next."""
        e = self._exp(project, builder)
        project.trials.record('dt', 'run_a', 0, 0, 1, 'built')
        project.remove_experimenter('run_a')
        assert project.trials.get_hist(experimenter='run_a') == []

    def test_removing_an_unmanaged_name_raises(self, project):
        with pytest.raises(KeyError):
            project.remove_experimenter('nope')

    def test_a_taken_name_cannot_be_added_over(self, project, builder, sample_data):
        self._exp(project, builder)
        with pytest.raises(ValueError, match='already exists'):
            project.set_experimenter('run_a')

    def test_trainers_are_indexed_too(self, project, builder, sample_data):
        version = builder.build().version
        project.set_trainer('t1', pipeline_version=version)
        assert project.list_trainers() == ['t1']
        assert project.list_experimenters() == []

    def test_reload_restores_name_and_version(self, project, builder, sample_data):
        e = self._exp(project, builder)
        e.build()
        loaded = Project(project.path, data=sample_data).experimenters['run_a']
        assert loaded.name == 'run_a'
        assert loaded.pipeline_version == e.pipeline_version
        assert loaded.get_status('scaler') == 'built'

    def test_reload_checks_data_key(self, project, builder, sample_data):
        builder.build()
        e = project.set_experimenter('keyed', data_key='k1')
        with pytest.raises(ValueError, match='data_key mismatch'):
            Experimenter.load_experimenter(e.path, sample_data, data_key='wrong')

    def test_trials_run_and_land_in_hist(self, project, builder, sample_data):
        e = self._exp(project, builder)
        e.build()
        trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0})
        assert project.set_trial(trial) == 'dt'
        e.exp(['dt'])
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}

        # history is keyed by the two names, matching the layout on disk
        row = project.trials.get_hist(experimenter='run_a')[0]
        assert row['trial_name'] == 'dt'
        assert row['pipeline_version'] == e.pipeline_version

    def test_exp_skips_fold_already_built_in_hist(self, project, builder, sample_data):
        """_make_jobs consults TrialStore.experiment_hist, not the on-disk
        artifact — a fold recorded 'built' there is skipped without dispatch."""
        e = self._exp(project, builder)
        e.build()
        project.set_trial(Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0}))
        e.exp(['dt'])
        build_id_1 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']

        e.exp(['dt'])
        build_id_2 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 == build_id_1

    def test_redefined_trial_with_built_hist_is_not_rerun(self, project, builder, sample_data):
        """A fold the hist marks 'built' is skipped whatever the definition now
        says. set_trial refuses such a redefinition, so this goes through the
        store directly — the skip rule has to hold on its own, since it is what
        makes the freeze necessary rather than merely tidy."""
        e = self._exp(project, builder)
        e.build()
        project.set_trial(Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0}))
        e.exp(['dt'])
        build_id_1 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']

        project.trials.register(Trial('dt', TREE, EDGES, params={'max_depth': 5, 'random_state': 0},
                                      pipeline_version=e.pipeline_version))
        e.exp(['dt'])
        build_id_2 = project.trials.get_info('dt', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 == build_id_1

    def test_error_fold_is_retried(self, project, builder, sample_data):
        """NodeStore.status() can no longer see 'error' at all (obj.pkl was
        never written) — TrialStore.experiment_hist is the only place a
        Trial's error status/detail is recorded now."""
        e = self._exp(project, builder)
        e.build()
        project.set_trial(Trial('bad', 'mock.BadPredictor', EDGES))
        e.exp(['bad'])
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'
        build_id_1 = project.trials.get_info('bad', 'run_a')[(0, 0)]['build_id']

        e.exp(['bad'])
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'
        build_id_2 = project.trials.get_info('bad', 'run_a')[(0, 0)]['build_id']
        assert build_id_2 != build_id_1

    def test_exp_multi_worker(self, project, builder, sample_data):
        """Experimenter.exp() through _execute_multi (n_jobs>1) — the merged
        Stage/Trial worker-pool executor, exercised here on its Trial/
        collectors path (unlike test_train_multi_worker, which only covers
        Trainer). One good and one failing trial across two folds so both
        the 'done' and 'error' worker messages are covered."""
        e = self._exp(project, builder)
        e.build()
        project.set_trials([
            Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 0}),
            Trial('bad', 'mock.BadPredictor', EDGES),
        ])
        e.exp(['dt', 'bad'], n_jobs=2)
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}
        assert project.trials.get_status('bad', 'run_a')[(0, 0)] == 'error'


class TestSetExperimenterExist:
    """set_experimenter/set_trainer's exist= gate — 'error' (default) is
    covered by test_a_taken_name_cannot_be_added_over above; this covers
    'skip' and 'replace', for both a name this project manages and a
    directory that is merely occupied (built outside the project's index)."""

    def test_unknown_exist_mode_raises(self, project, builder, sample_data):
        """Only checked once a name is actually taken — exist= only ever
        matters when there is something to decide about."""
        version = builder.build().version
        project.set_experimenter('run_a', pipeline_version=version)
        with pytest.raises(ValueError, match='Unknown exist mode'):
            project.set_experimenter('run_a', exist='bogus')

    def test_skip_returns_the_managed_one_unexamined(self, project, builder, sample_data):
        version = builder.build().version
        e = project.set_experimenter('run_a', pipeline_version=version)
        e.build()
        again = project.set_experimenter('run_a', exist='skip')
        assert again is e
        assert e.get_status('scaler') == 'built'   # untouched, not recreated

    def test_skip_on_an_occupied_but_unmanaged_directory_raises(self, project, builder,
                                                                sample_data):
        """Nothing indexed means 'skip' has no existing object to hand back —
        this is not the same situation as a name this project manages."""
        Experimenter(project.exp_path('outsider'), 'outsider', sample_data)
        assert 'outsider' not in project.list_experimenters()
        with pytest.raises(ValueError, match='already holds one'):
            project.set_experimenter('outsider', exist='skip')

    def test_replace_recreates_a_managed_one(self, project, builder, sample_data):
        version = builder.build().version
        e = project.set_experimenter(
            'run_a', sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0),
            pipeline_version=version,
        )
        e.build()
        assert e.get_status('scaler') == 'built'

        replaced = project.set_experimenter('run_a', pipeline_version=version, exist='replace')
        assert replaced is not e
        assert project.experimenters['run_a'] is replaced
        assert replaced.get_status('scaler') is None   # fresh directory, nothing built yet

    def test_replace_clears_an_occupied_but_unmanaged_directory(self, project, builder,
                                                                sample_data):
        version = builder.build().version
        Experimenter(project.exp_path('outsider'), 'outsider', sample_data)

        e = project.set_experimenter('outsider', pipeline_version=version, exist='replace')
        assert project.list_experimenters() == ['outsider']
        assert e.pipeline_version == version

    def test_trainer_replace_recreates_a_managed_one(self, project, builder, sample_data):
        version = builder.build().version
        t = project.set_trainer('bare_t', pipeline_version=version)
        t.train()   # no Predictors -> just builds every Pipeline node
        assert t.get_status('scaler') == 'built'

        replaced = project.set_trainer('bare_t', pipeline_version=version, exist='replace')
        assert project.trainers['bare_t'] is replaced
        assert replaced.get_status('scaler') is None

    def test_trainer_skip_returns_the_managed_one(self, project, builder, sample_data):
        version = builder.build().version
        t = project.set_trainer('bare_t', pipeline_version=version)
        again = project.set_trainer('bare_t', exist='skip')
        assert again is t


class TestExpChecksTheVersion:
    """A Trial runs only where it was authored. Collected results are keyed by
    node name and fold with no version in them, so two versions of one name
    would overwrite each other's metrics with nothing to say which won."""

    def _exp(self, project, builder, name='run_a', version=None):
        return project.set_experimenter(
            name,
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0),
            pipeline_version=version,
        )

    def test_a_matching_version_runs(self, project, builder, sample_data):
        builder.build()
        e = self._exp(project, builder)
        e.build()
        project.set_trial(_trial())
        e.exp(['dt'])
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built'}

    def test_a_trial_from_an_older_version_is_refused(self, project, builder, sample_data):
        builder.build()
        project.set_trial(_trial())                      # stamped v1
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        assert builder.build().version == 2

        e = self._exp(project, builder)     # adopts the latest, v2
        e.build()
        with pytest.raises(ValueError, match='version 1'):
            e.exp(['dt'])
        assert project.trials.get_status('dt', 'run_a') == {}

    def test_a_finished_trial_passes_after_a_version_bump(self, project, builder, sample_data):
        """Nothing left to file, so nothing to refuse — re-handing a finished
        round stays a no-op rather than becoming an error."""
        builder.build()
        e = self._exp(project, builder)
        e.build()
        project.set_trial(_trial())
        e.exp(['dt'])

        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        e.set_pipeline(project.load_pipeline(builder.build().version))
        e.build()
        e.exp(['dt'])                                    # no jobs, no complaint
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built'}

    def test_a_half_run_trial_is_still_refused_after_a_bump(self, project, builder, sample_data):
        """Some folds left means new results would land beside the old ones."""
        builder.build()
        e = project.set_experimenter(
            'run_b',
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0))
        e.build()
        project.set_trial(_trial())
        e.exp(['dt'])
        project.trials.remove_hist(trial_name='dt')      # as if one fold never ran
        project.trials.record('dt', 'run_b', 0, 0, status='built')

        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        e.set_pipeline(project.load_pipeline(builder.build().version))
        with pytest.raises(ValueError, match='version 1'):
            e.exp(['dt'])

    def test_adopting_that_version_lets_it_run(self, project, builder, sample_data):
        """Nothing is demoted, so the older version is still adoptable — which
        is the way out of the refusal above."""
        builder.build()
        project.set_trial(_trial())
        builder.set_node('scaler', grp='scale', edges={'X': '{target}'}, exist='replace')
        builder.build()

        e = self._exp(project, builder, version=1)
        e.build()
        e.exp(['dt'])
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built'}


class TestTrialErrorReporting:
    """Who answers for a failure follows who owns the history it sits in: node
    history is the run's (Experimenter.error_nodes), Trial history is the
    project's, keyed by experimenter."""

    def test_error_trials_reports_the_failed_fold(self, project):
        project.trials.record('bad', 'run_a', 0, 1, status='error',
                              info={'error': {'type': 'ValueError', 'message': 'boom',
                                              'traceback': 'TB'}})
        rows = project.error_trials()
        assert rows == [{'trial_name': 'bad', 'experimenter': 'run_a',
                         'outer_idx': 0, 'inner_idx': 1, 'pipeline_version': None,
                         'type': 'ValueError', 'message': 'boom', 'traceback': 'TB'}]

    def test_a_row_without_a_payload_still_answers(self, project):
        """The failure fields are always there, so a caller never has to check
        whether info carried them."""
        project.trials.record('bad', 'run_a', 0, 0, status='error')
        assert project.error_trials()[0]['type'] is None

    def test_nothing_failed_is_empty(self, project):
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        assert project.error_trials() == []

    def test_narrowed_to_one_run(self, project):
        project.trials.record('bad', 'run_a', 0, 0, status='error')
        project.trials.record('bad', 'run_b', 0, 0, status='built')
        assert project.error_trials(experimenter='run_a')
        assert project.error_trials(experimenter='run_b') == []

    def test_node_errors_stay_out_of_it(self, project, builder, sample_data):
        """error_nodes is Pipeline nodes only — the two halves used to come back
        from one call, indistinguishable in the output."""
        version = builder.build().version
        e = project.set_experimenter('run_a',
                                 pipeline_version=version)
        project.trials.record('bad', 'run_a', 0, 0, status='error')
        assert e.error_nodes() == []


class TestStaleNodes:
    """What an edit would cost, asked before adopting it — the only moment the
    answer exists, since set_pipeline turns it straight into deletions."""

    @staticmethod
    def _built_exp(project, builder, name='run_a'):
        e = project.set_experimenter(name, pipeline_version=builder.build().version)
        e.build()
        return e

    def test_untouched_working_copy_costs_nothing(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        assert project.stale_nodes() == {'experimenters': {'run_a': []}, 'trainers': {}}
        assert e.get_status('scaler') == 'built'

    def test_edit_names_the_node_it_would_drop(self, project, builder, sample_data):
        self._built_exp(project, builder)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        assert project.stale_nodes()['experimenters'] == {'run_a': ['scaler']}

    def test_trainers_answer_too(self, project, builder, sample_data):
        """Under their own key: exp/{name} and trainers/{name} are separate
        namespaces, so one flat mapping would drop a shared name silently."""
        self._built_exp(project, builder, 'shared')
        project.set_trainer('shared', pipeline_version=builder.build().version)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        assert project.stale_nodes() == {
            'experimenters': {'shared': ['scaler']},
            'trainers': {'shared': ['scaler']},
        }

    def test_asking_does_not_publish(self, project, builder, sample_data):
        """The working copy is built through draft(), which mints nothing — a
        preview that minted a version would change what set_experimenter()
        adopts next."""
        self._built_exp(project, builder)
        before = project.list_pipeline_versions()
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        project.stale_nodes()
        assert project.list_pipeline_versions() == before

    def test_narrowed_to_one_experimenter(self, project, builder, sample_data):
        """Naming one narrows to exactly that: the other side comes back empty
        rather than quietly answering in full."""
        self._built_exp(project, builder, 'run_a')
        self._built_exp(project, builder, 'run_b')
        project.set_trainer('t1', pipeline_version=builder.build().version)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        assert project.stale_nodes(experimenter='run_a') == {
            'experimenters': {'run_a': ['scaler']}, 'trainers': {}}

    def test_narrowed_to_one_trainer(self, project, builder, sample_data):
        self._built_exp(project, builder, 'run_a')
        project.set_trainer('t1', pipeline_version=builder.build().version)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        assert project.stale_nodes(trainer='t1') == {
            'experimenters': {}, 'trainers': {'t1': ['scaler']}}

    def test_published_pipeline_asks_the_other_direction(self, project, builder, sample_data):
        """How far behind an Experimenter is: it adopted v0, the project has
        since published the node."""
        e = project.set_experimenter('run_a',
                                     pipeline_version=builder.build().version)
        e.build()
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})
        builder.build()
        assert project.stale_nodes(project.load_pipeline())['experimenters'] == {'run_a': ['scaler']}


class TestSetData:
    """set_data(data) is a commit point like set_pipeline: it resets whatever
    the new data invalidates for every Experimenter/Trainer this project
    manages (#148), before writing the new data."""

    @staticmethod
    def _built_exp(project, builder, name='run_a'):
        e = project.set_experimenter(name, pipeline_version=builder.build().version)
        e.build()
        return e

    def test_first_set_data_stales_nothing(self, tmp_path, sample_data):
        """No prior data means nothing to have diverged from — the same
        reasoning as diffing against the empty Pipeline."""
        project = Project(tmp_path / 'fresh')
        project.set_data(sample_data)
        assert project.data.equals(sample_data)

    def test_unrelated_column_added_leaves_the_node_built(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        assert e.get_status('scaler') == 'built'

        new_data = sample_data.assign(extra=1)
        project.set_data(new_data)
        assert project.experimenters['run_a'].get_status('scaler') == 'built'

    def test_removed_referenced_column_resets_the_node(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        assert e.get_status('scaler') == 'built'

        renamed = sample_data.rename(columns={'f1': 'f1_renamed'})
        project.set_data(renamed)
        assert project.experimenters['run_a'].get_status('scaler') is None

    def test_trainers_are_reset_too(self, project, builder, sample_data):
        """Checked on the already-held Trainer directly, not through a
        reopen — reopening re-adopts the Pipeline, and its still-declared
        schema requiring 'f1' is a separate concern (check_data_compatibility)
        from the column-level staleness this test is about."""
        t = project.set_trainer('t1', pipeline_version=builder.build().version)
        t.train()
        assert t.get_status('scaler') == 'built'

        renamed = sample_data.rename(columns={'f1': 'f1_renamed'})
        project.set_data(renamed)
        assert t.get_status('scaler') is None

    def test_stale_nodes_preview_matches_what_set_data_would_do(self, project, builder, sample_data):
        self._built_exp(project, builder)
        renamed = sample_data.rename(columns={'f1': 'f1_renamed'})

        preview = project.stale_nodes(data=renamed)
        assert preview['experimenters'] == {'run_a': ['scaler']}

        project.set_data(renamed)
        assert project.experimenters['run_a'].get_status('scaler') is None

    def test_reopens_already_held_runs_against_the_new_data(self, project, builder, sample_data):
        """Resetting artifacts is not enough on its own — an Experimenter
        already held in memory still has the old data's splits until it is
        reopened."""
        self._built_exp(project, builder)
        held_before = project.experimenters['run_a']

        project.set_data(sample_data.assign(extra=1))
        assert project.experimenters['run_a'] is not held_before


class TestPublishPipeline:
    """Build, propagate, report — one call, because the ordering it enforces
    spans every Experimenter and no single one can see it."""

    @staticmethod
    def _built_exp(project, builder, name='run_a'):
        e = project.set_experimenter(name, pipeline_version=builder.build().version)
        e.build()
        return e

    def test_publishing_moves_the_experimenters_onto_the_new_version(
            self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        report = project.publish_pipeline()
        assert report['version'] == project.load_pipeline().version
        assert e.pipeline_version == report['version']

    def test_the_report_names_what_it_cost(self, project, builder, sample_data):
        """Not just the number. Afterwards there is nothing left to ask —
        the artifacts are gone and the staleness went into deleting them."""
        e = self._built_exp(project, builder)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        report = project.publish_pipeline()
        assert report['experimenters'] == {'run_a': ['scaler']}
        assert e.get_status('scaler') is None

    def test_an_unchanged_definition_moves_nothing(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        before = project.list_pipeline_versions()

        report = project.publish_pipeline()
        assert project.list_pipeline_versions() == before
        assert report['experimenters'] == {'run_a': []}
        assert e.get_status('scaler') == 'built'

    def test_a_dry_run_prices_it_without_publishing(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        before = project.list_pipeline_versions()
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        report = project.publish_pipeline(dry_run=True)
        assert report['version'] is None          # a draft carries no number
        assert report['experimenters'] == {'run_a': ['scaler']}
        assert project.list_pipeline_versions() == before
        assert e.get_status('scaler') == 'built'  # adopted nothing

    def test_trainers_stay_out_unless_asked_for(self, project, builder, sample_data):
        """An Experimenter loses artifacts that rebuild; a Trainer loses
        trained models for good. The default declines to spend the second."""
        self._built_exp(project, builder)
        t = project.set_trainer('t1', pipeline_version=builder.build().version)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        report = project.publish_pipeline()
        assert report['trainers'] == {}
        assert t.pipeline_version == 1                    # left where it was
        assert project.experimenters['run_a'].pipeline_version == report['version']

        report = project.publish_pipeline(trainers=True)
        # Names, not a claim that an artifact stood there — the same reading
        # Experimenter.stale_nodes has.
        assert report['trainers'] == {'t1': {'nodes': ['scaler'], 'retired': []}}
        assert t.pipeline_version == report['version']

    def test_experimenters_can_be_declined_too(self, project, builder, sample_data):
        e = self._built_exp(project, builder)
        builder.set_node('scaler', grp='scale', exist='replace', params={'with_std': False})

        report = project.publish_pipeline(experimenters=False)
        assert report['experimenters'] == {}
        assert e.get_status('scaler') == 'built'
        assert e.pipeline_version == 1        # the version was still minted
        assert project.load_pipeline().version == report['version']


class TestPendingTrials:
    """Registered Trials that errored or never ran — one list, because both
    still owe a run. Hand-written, this filter tends to catch only the second
    (see the notebook it came from)."""

    def test_a_never_run_trial_is_pending(self, project):
        project.set_trial(_trial())
        assert project.pending_trials() == ['dt']

    def test_an_errored_trial_is_pending(self, project):
        """The case a hand-written 'has no history' filter drops."""
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='error')
        assert project.pending_trials() == ['dt']

    def test_a_built_trial_is_not(self, project):
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        assert project.pending_trials() == []

    def test_scoped_to_one_run(self, project):
        project.set_trial(_trial())
        project.trials.record('dt', 'run_a', 0, 0, status='built')
        assert project.pending_trials(experimenter='run_a') == []
        assert project.pending_trials(experimenter='run_b') == ['dt']

    def test_only_registered_trials_are_considered(self, project):
        """History alone does not make a Trial — a removed definition is not
        work still owed."""
        project.trials.record('gone', 'run_a', 0, 0, status='error')
        assert project.pending_trials() == []

    def test_feeds_straight_into_exp(self, project, builder, sample_data):
        version = builder.build().version
        e = project.set_experimenter('run_a',
                                 sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0), pipeline_version=version)
        e.build()
        project.set_trial(_trial())
        e.exp(project.pending_trials(experimenter='run_a'))
        assert project.pending_trials(experimenter='run_a') == []


class TestRemoveTrial:
    """A Trial leaves no artifact, so everything it produced sits in stores
    that don't know about each other — the project's TrialStore (definition +
    history), and inside every run that ran it, the collected data and the
    CollectHist rows for it. Project is the only thing that sees them all."""

    def _run(self, project, builder, name='run_a'):
        version = builder.build().version
        e = project.set_experimenter(
            name,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=0), pipeline_version=version,
        )
        e.build()
        e.collectors.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
                                   params={'output_var': None, 'metric_func': {'__callable__': 'test_project.dummy_metric'}})
        project.set_trial(_trial())
        e.exp(['dt'])
        return e, e.collectors

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
        e, collectors = self._run(project, builder)
        assert project.trials.get_by_name('dt') is not None
        assert project.trials.get_status('dt', 'run_a')
        assert collectors.hist.get_hist(node_name='dt')
        assert collectors.get_collector('acc').get_metric('dt') is not None

        project.remove_trial('dt')

        assert project.trials.get_by_name('dt') is None
        assert project.trials.get_status('dt', 'run_a') == {}
        assert collectors.hist.get_hist(node_name='dt') == []
        assert Collectors(e.path / 'collectors').get_collector('acc').get_metric('dt') is None

    def test_leaves_other_trials_alone(self, project, builder, sample_data):
        e, collectors = self._run(project, builder)
        project.set_trial(_trial('dt2'))
        e.exp(['dt2'])

        project.remove_trial('dt')

        assert project.trials.get_by_name('dt2') is not None
        assert project.trials.get_status('dt2', 'run_a')
        assert collectors.hist.get_hist(node_name='dt2')
        assert collectors.get_collector('acc').get_metric('dt2') is not None

    def test_reaches_every_run_that_collected_it(self, project, builder, sample_data):
        """The registries are per-run now, so this is a pass over the runs
        rather than one store — a run left out would keep its results."""
        e_a, coll_a = self._run(project, builder, name='run_a')
        e_b, coll_b = self._run(project, builder, name='run_b')
        assert coll_a.get_collector('acc').has_node('dt')
        assert coll_b.get_collector('acc').has_node('dt')

        project.remove_trial('dt')

        for e in (e_a, e_b):
            reopened = Collectors(e.path / 'collectors')
            assert reopened.get_collector('acc').get_metric('dt') is None
            assert reopened.hist.get_hist(node_name='dt') == []

    def test_cleans_the_live_experimenter_the_caller_holds(self, project, builder, sample_data):
        """A registry reopened from disk is not the one the caller holds, and a
        Collector may answer from an in-memory cache — so removal has to go
        through the Experimenter object, which the project is holding."""
        e, collectors = self._run(project, builder)
        acc = collectors.get_collector('acc')
        assert acc.has_node('dt')
        project.remove_trial('dt')
        assert not acc.has_node('dt')
        assert project.experimenters['run_a'] is e

    def test_a_removed_trial_runs_again_from_scratch(self, project, builder, sample_data):
        """Removal takes the definition too, so re-running means authoring it
        again — which is also what lifts the redefinition freeze."""
        e, collectors = self._run(project, builder)
        project.remove_trial('dt')
        assert project.set_trial(_trial()) == 'dt'
        e.exp(['dt'])
        assert project.trials.get_status('dt', 'run_a') == {(0, 0): 'built', (1, 0): 'built'}


class TestTrainerUnderProject:
    """Exercises Trainer.train() -> Job/_execute_* through the same job-based
    path Experimenter.build() uses."""

    def test_train_respects_node_dependency_order(self, project, sample_data):
        from mllabs._data_wrapper import wrap

        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = p.build().version

        t = project.set_trainer('trained_chain', pipeline_version=version)
        t.train([Predictor('dt', TREE, {'X': 's2:(*)', 'y': '{target}'},
                           params={'max_depth': 3, 'random_state': 0})])
        assert t.get_status('s1') == 'built'
        assert t.get_status('s2') == 'built'
        assert t.get_status('dt') == 'built'

    def test_train_multi_worker(self, project, sample_data):
        from mllabs._data_wrapper import wrap

        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = p.build().version

        t = project.set_trainer('trained_chain_multi', pipeline_version=version)
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

        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        p.set_grp('scale2', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': 's1:(*)'})
        p.set_node('s2', grp='scale2')
        version = p.build().version

        predictor = Predictor('dt', TREE, {'X': 's2:(*)', 'y': '{target}'},
                              params={'max_depth': 3, 'random_state': 0})
        t = project.set_trainer('trainer_reload', pipeline_version=version)
        t.train([predictor])

        loaded = Project(project.path, data=sample_data).trainers['trainer_reload']
        flow = loaded.train_folds[0].train_data_flows[0]
        assert flow.node_objs == {}

        # The selection comes back from PredictorStore, so nothing needs
        # re-supplying and nothing is reset by reopening.
        assert loaded.predictor_names() == ['dt']
        assert loaded.selected_nodes == ['s1', 's2']
        assert loaded.get_status('dt') == 'built'

        # Proves _node_edges (recovered from the store's history) actually
        # routes data through s1 -> s2, not just that obj.pkl happened to load.
        train_data = flow.get_train({'X': 's2:(*)', 'y': '{target}'})
        assert train_data['X'].get_shape()[0] > 0
        assert 's1' in flow.node_objs
        assert 's2' in flow.node_objs

    def test_trainer_reopens_without_a_project(self, project, sample_data):
        """Trainer.load_trainer() reads the Pipeline the Trainer itself saved
        — the caller resolves no version."""
        from mllabs._data_wrapper import wrap
        from mllabs._trainer import Trainer

        p = project.pipeline
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{f1}'})
        p.set_node('s1', grp='scale')
        version = p.build().version

        t = project.set_trainer('t_standalone', pipeline_version=version)
        t.train([Predictor('dt', TREE, {'X': 's1:(*)', 'y': '{target}'},
                           params={'max_depth': 3, 'random_state': 0})])
        assert (t.path / 'pipeline.pkl').exists()

        reopened = Trainer.load_trainer(t.path, wrap(sample_data))
        assert reopened.pipeline is not None
        assert reopened.pipeline_version == version
        assert reopened.selected_nodes == ['s1']
        assert reopened.get_status('dt') == 'built'
