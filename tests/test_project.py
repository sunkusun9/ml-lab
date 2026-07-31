import pytest

from mllabs import Project, TrialStore, Trial, PipelineBuilder, make_trials


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


class TestProjectLayout:
    def test_root_created(self, project, tmp_path):
        assert (tmp_path / 'proj').is_dir()

    def test_project_db_created(self, project):
        assert (project.path / 'project.db').exists()

    def test_trial_store_created(self, project):
        assert (project.path / 'trials.db').exists()

    def test_paths_are_under_root(self, project):
        for p in (project.pipeline_path('a'), project.run_path('b'),
                  project.trainer_path('c'), project.inferencer_path('d'),
                  project.collectors_path()):
            assert project.path in p.parents or p.parent == project.path

    def test_paths_are_created(self, project):
        assert project.run_path('r1').is_dir()

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
    def test_first_save_is_v1(self, project, builder):
        assert project.save_pipeline(builder.build(), 'main') == 1

    def test_resave_same_content_does_not_bump(self, project, builder):
        v1 = project.save_pipeline(builder.build(), 'main')
        assert project.save_pipeline(builder.build(), 'main') == v1

    def test_rebuild_alone_does_not_bump(self, project, builder):
        """build() mints a fresh build_id each call; that must not be a version."""
        a, b = builder.build(), builder.build()
        assert a.build_id != b.build_id
        assert project.save_pipeline(a, 'main') == project.save_pipeline(b, 'main')

    def test_edit_bumps_version(self, project, builder):
        v1 = project.save_pipeline(builder.build(), 'main')
        builder.set_node('scaler', grp='scale', exist='replace')
        assert project.save_pipeline(builder.build(), 'main') == v1 + 1

    def test_versions_are_per_name(self, project, builder):
        project.save_pipeline(builder.build(), 'main')
        assert project.save_pipeline(builder.build(), 'other') == 1

    def test_load_latest(self, project, builder):
        project.save_pipeline(builder.build(), 'main')
        builder.set_node('scaler', grp='scale', exist='replace')
        latest = builder.build()
        project.save_pipeline(latest, 'main')
        assert project.load_pipeline('main').content_key() == latest.content_key()

    def test_load_specific_version(self, project, builder):
        first = builder.build()
        project.save_pipeline(first, 'main')
        builder.set_node('scaler', grp='scale', exist='replace')
        project.save_pipeline(builder.build(), 'main')
        assert project.load_pipeline('main', 1).content_key() == first.content_key()

    def test_loaded_pipeline_is_usable(self, project, builder):
        project.save_pipeline(builder.build(), 'main')
        loaded = project.load_pipeline('main')
        assert loaded.topo_order() == ['scaler']
        assert loaded.get_node_attrs('scaler')['role'] == 'stage'

    def test_load_unknown_raises(self, project):
        with pytest.raises(KeyError):
            project.load_pipeline('nope')

    def test_get_pipeline_version(self, project, builder):
        built = builder.build()
        v = project.save_pipeline(built, 'main')
        assert project.get_pipeline_version(built, 'main') == v

    def test_get_pipeline_version_unsaved(self, project, builder):
        assert project.get_pipeline_version(builder.build(), 'main') is None

    def test_list_versions(self, project, builder):
        project.save_pipeline(builder.build(), 'main')
        builder.set_node('scaler', grp='scale', exist='replace')
        project.save_pipeline(builder.build(), 'main')
        assert [r['version'] for r in project.list_pipeline_versions('main')] == [1, 2]

    def test_resolve_version_from_content_key(self, project, builder):
        built = builder.build()
        v = project.save_pipeline(built, 'main')
        assert project.resolve_version(built.content_key(), 'main') == v


class TestTrialRegistration:
    def test_register_returns_content_key(self, store):
        assert store.register(_trial()) == _trial().content_key()

    def test_same_definition_registers_once(self, store):
        store.register(_trial())
        store.register(_trial())
        assert len(store.list_trials()) == 1

    def test_name_does_not_split_identity(self, store):
        """content_key ignores the name — renaming is not a new definition."""
        assert store.register(_trial('a')) == store.register(_trial('b'))

    def test_different_params_different_key(self, store):
        assert store.register(_trial(params={'max_depth': 3})) != \
               store.register(_trial(params={'max_depth': 5}))

    def test_register_all_maps_name_to_key(self, store):
        trials = make_trials('dt', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': [3, 5]})
        mapping = store.register_all(trials)
        assert set(mapping) == {'dt_0', 'dt_1'}
        assert mapping['dt_0'] == trials[0].content_key()

    def test_has_before_and_after(self, store):
        assert not store.has(_trial())
        store.register(_trial())
        assert store.has(_trial())

    def test_get_definition_roundtrip(self, store):
        key = store.register(_trial(params={'max_depth': 7}))
        got = store.get_definition(key)
        assert got['processor'] == TREE
        assert got['params'] == {'max_depth': 7}
        assert got['edges'] == EDGES

    def test_get_definition_unknown(self, store):
        assert store.get_definition('nope') is None

    def test_list_trials(self, store):
        store.register(_trial('a'))
        store.register(_trial('b', params={'max_depth': 9}))
        assert len(store.list_trials()) == 2

    def test_survives_reopen(self, store, tmp_path):
        key = store.register(_trial())
        assert TrialStore(tmp_path / 'ts').get_definition(key) is not None


class TestExperimentHist:
    def test_record_and_read(self, store):
        store.record('dt', 'exp-1', 0, 0, content_key='ck1',
                     pipeline_version='pk1', status='built')
        rows = store.get_hist(trial_name='dt')
        assert len(rows) == 1
        assert rows[0]['status'] == 'built'
        assert rows[0]['pipeline_version'] == 'pk1'
        assert rows[0]['content_key'] == 'ck1'

    def test_fold_coordinates_are_part_of_the_key(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-1', 0, 1, status='built')
        store.record('dt', 'exp-1', 1, 0, status='error')
        assert len(store.get_hist(trial_name='dt')) == 3

    def test_rerun_same_fold_overwrites(self, store):
        """Redefining a name overwrites its artifact, so it overwrites its row."""
        store.record('dt', 'exp-1', 0, 0, content_key='ck1', status='error')
        store.record('dt', 'exp-1', 0, 0, content_key='ck2', status='built')
        rows = store.get_hist(trial_name='dt')
        assert len(rows) == 1
        assert rows[0]['status'] == 'built' and rows[0]['content_key'] == 'ck2'

    def test_same_name_across_experimenters_is_separate(self, store):
        store.record('dt', 'exp-1', 0, 0, pipeline_version='pk1', status='built')
        store.record('dt', 'exp-2', 0, 0, pipeline_version='pk2', status='built')
        assert {r['pipeline_version'] for r in store.get_hist(trial_name='dt')} == {'pk1', 'pk2'}

    def test_filter_by_experimenter(self, store):
        store.record('dt', 'exp-1', 0, 0, status='built')
        store.record('dt', 'exp-2', 0, 0, status='built')
        assert len(store.get_hist(experimenter_id='exp-2')) == 1

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
        store.remove_hist(experimenter_id='exp-1')
        assert len(store.get_hist(trial_name='dt')) == 1

    def test_remove_hist_keeps_the_definition(self, store):
        key = store.register(_trial())
        store.record('dt', 'exp-1', 0, 0, content_key=key, status='built')
        store.remove_hist(trial_name='dt')
        assert store.get_definition(key) is not None


class TestProjectEndToEnd:
    def test_hist_pipeline_version_resolves_to_a_number(self, project, builder):
        built = builder.build()
        version = project.save_pipeline(built, 'main')
        key = project.trials.register(_trial())
        project.trials.record('dt', 'exp-1', 0, 0, content_key=key,
                              pipeline_version=built.content_key(), status='built')

        row = project.trials.get_hist(trial_name='dt')[0]
        assert project.resolve_version(row['pipeline_version'], 'main') == version
        assert project.trials.get_definition(row['content_key'])['processor'] == TREE
