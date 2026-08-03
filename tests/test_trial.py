import pytest

from mllabs import CollectorStore, Collectors, Connector, PipelineBuilder
from mllabs import Trial, make_trials


LGBM = 'lightgbm.LGBMClassifier'
TREE = 'sklearn.tree.DecisionTreeClassifier'
EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


@pytest.fixture
def pipeline():
    p = PipelineBuilder()
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1, f2}'})
    p.set_node('scaler', grp='scale')
    return p


@pytest.fixture
def swept():
    return make_trials(
        'lgbm', processor=LGBM, edges=EDGES,
        params={'random_state': 42},
        param_grid={'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    )


class TestMakeTrialsGrid:
    def test_trial_count_is_cartesian_product(self, swept):
        assert len(swept) == 4

    def test_no_grid_yields_single_trial(self):
        trials = make_trials('one', processor=TREE, edges=EDGES, params={'max_depth': 3})
        assert len(trials) == 1
        assert trials[0].name == 'one'

    def test_fixed_params_merged_into_every_trial(self, swept):
        assert all(t.params['random_state'] == 42 for t in swept)

    def test_grid_params_cover_all_combinations(self, swept):
        combos = {(t.params['max_depth'], t.params['learning_rate'])
                  for t in swept}
        assert combos == {(3, 0.05), (3, 0.1), (5, 0.05), (5, 0.1)}

    def test_grid_overrides_fixed_param(self):
        trials = make_trials('x', processor=TREE, edges=EDGES,
                             params={'max_depth': 1}, param_grid={'max_depth': [7]})
        assert trials[0].params['max_depth'] == 7

    def test_shared_fields_are_identical(self, swept):
        for t in swept:
            assert t.processor == LGBM
            assert t.method == 'predict'
            assert t.edges == EDGES

    def test_edges_not_shared_between_trials(self, swept):
        a, b = swept[:2]
        a.edges['X'] = 'mutated'
        assert b.edges['X'] == 'scaler:(*)'

    def test_names_are_unique_and_padded(self, swept):
        assert [t.name for t in swept] == ['lgbm_0', 'lgbm_1', 'lgbm_2', 'lgbm_3']

    def test_wide_index_padding(self):
        trials = make_trials('w', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': list(range(12))})
        assert trials[0].name == 'w_00'

    def test_order_is_deterministic(self):
        def mk():
            return make_trials('d', processor=TREE, edges=EDGES,
                               param_grid={'b': [1, 2], 'a': ['x', 'y']})
        assert [t.params for t in mk()] == [t.params for t in mk()]

    def test_tags_propagate_to_trials(self):
        trials = make_trials('t', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': [1, 2]}, tags=['final'])
        assert all(t.tag == ['final'] for t in trials)


class TestMakeTrialsValidation:
    def test_processor_class_rejected(self):
        from sklearn.tree import DecisionTreeClassifier
        with pytest.raises(TypeError, match='processor must be'):
            make_trials('x', processor=DecisionTreeClassifier, edges=EDGES)

    def test_adapter_instance_rejected(self):
        from mllabs.adapter import DefaultAdapter
        with pytest.raises(TypeError, match='adapter must be'):
            make_trials('x', processor=TREE, edges=EDGES, adapter=DefaultAdapter())

    def test_live_object_in_params_rejected(self):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match='must be plain data'):
            make_trials('x', processor=TREE, edges=EDGES,
                             params={'cat_features': ColSelector('*')})

    def test_live_object_in_grid_rejected(self):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match='must be plain data'):
            make_trials('x', processor=TREE, edges=EDGES,
                             param_grid={'cat_features': [ColSelector('*')]})

    def test_empty_edges_rejected(self):
        with pytest.raises(ValueError, match='non-empty'):
            make_trials('x', processor=TREE, edges={})

    def test_non_string_edge_rejected(self):
        with pytest.raises(TypeError, match='DSL string'):
            make_trials('x', processor=TREE, edges={'X': ['a', 'b']})

    def test_scalar_grid_value_rejected(self):
        with pytest.raises(TypeError, match='must be a list'):
            make_trials('x', processor=TREE, edges=EDGES, param_grid={'max_depth': 3})

    def test_empty_grid_value_rejected(self):
        with pytest.raises(ValueError, match='is empty'):
            make_trials('x', processor=TREE, edges=EDGES, param_grid={'max_depth': []})


class TestTrialSpec:
    """What a Trial resolves to for the executor, and what it reads.

    Identity is no longer derived here: a Trial's name *is* its identity and
    "is this the stored definition" is a plain value comparison, both covered
    by ``TestTrialRegistration`` in test_project.py. The content hash that
    used to live on this class (``trial_id``, folding in upstream node
    serials) is gone along with serials themselves.
    """

    def test_spec_shape_matches_a_node_spec(self, pipeline, swept):
        """A Trial must look like a node to Connector/executor/Collector."""
        trial_spec = swept[0].get_spec()
        node_spec = pipeline.build().get_node_spec('scaler')
        assert type(trial_spec) is type(node_spec)
        assert trial_spec.__slots__ == node_spec.__slots__

    def test_spec_carries_the_definition(self, swept):
        spec = swept[0].get_spec()
        trial = swept[0]
        assert (spec.name, spec.processor, spec.method) == (trial.name, LGBM, 'predict')
        assert spec.edges == EDGES
        assert spec.params == trial.params

    def test_spec_drops_display_only_fields(self, swept):
        """desc/tag stay on the Trial — they never reach the executor."""
        spec = swept[0].get_spec()
        assert not hasattr(spec, 'desc')
        assert not hasattr(spec, 'tag')

    def test_node_names_lists_referenced_nodes(self):
        assert Trial('a', TREE, EDGES).node_names() == {'scaler'}

    def test_node_names_excludes_the_datasource(self):
        """``{target}`` reads the DataSource, which is not a node reference."""
        assert Trial('a', TREE, {'y': '{target}'}).node_names() == set()

    def test_node_names_spans_every_edge_key(self):
        trial = Trial('a', TREE, {'X': 'scaler:(*)', 'y': 'labeller:(*)'})
        assert trial.node_names() == {'scaler', 'labeller'}


class TestCollectorsRegistry:
    def _reg(self, tmp_path):
        return Collectors(tmp_path)

    def _set(self, reg, name='m', **kw):
        # Unrestricted — Trial.get_spec() no longer carries a 'role' key to
        # match on (2026-08-01, role was dead weight: Collectors only ever
        # run against Trial jobs anyway, never Stage ones).
        return reg.set_collector(
            name, 'mllabs.MetricCollector', Connector(),
            params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'},
                    'output_var': '*'},
            **kw,
        )

    def test_registers_and_returns(self, tmp_path):
        reg = self._reg(tmp_path)
        c = self._set(reg)
        assert reg.get_collector('m') is c

    def test_path_defaults_under_registry(self, tmp_path):
        assert self._set(self._reg(tmp_path)).path == tmp_path / 'm'

    def test_explicit_path_wins(self, tmp_path):
        c = self._set(self._reg(tmp_path), path=tmp_path / 'elsewhere')
        assert c.path == tmp_path / 'elsewhere'

    def test_pathless_registry_requires_path(self):
        with pytest.raises(ValueError, match='no base path'):
            self._set(Collectors())

    def test_callable_ref_resolved(self, tmp_path):
        from sklearn.metrics import accuracy_score
        assert self._set(self._reg(tmp_path)).metric_func is accuracy_score

    def test_skip_returns_existing(self, tmp_path):
        reg = self._reg(tmp_path)
        assert self._set(reg) is self._set(reg)

    def test_error_raises(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg)
        with pytest.raises(RuntimeError, match='already registered'):
            self._set(reg, exist='error')

    def test_replace_rebuilds(self, tmp_path):
        reg = self._reg(tmp_path)
        first = self._set(reg)
        assert self._set(reg, exist='replace') is not first

    def test_unknown_exist_mode(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg)
        with pytest.raises(ValueError, match='Unknown exist mode'):
            self._set(reg, exist='bogus')

    def test_remove(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg)
        reg.remove_collector('m')
        assert reg.get_collector('m') is None and 'm' not in reg

    def test_names_and_len(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a'); self._set(reg, 'b')
        assert reg.names() == ['a', 'b'] and len(reg) == 2

    def test_resolve_by_names(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a'); self._set(reg, 'b')
        assert [c.name for c in reg.resolve(['b'])] == ['b']

    def test_resolve_none_returns_all(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a'); self._set(reg, 'b')
        assert len(reg.resolve(None)) == 2

    def test_resolve_unknown_name_raises(self, tmp_path):
        """A silent miss would look exactly like 'collected nothing'."""
        with pytest.raises(KeyError, match='nope'):
            self._reg(tmp_path).resolve(['nope'])

    def test_match_unrestricted_connector(self, swept, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg)
        spec = swept[0].get_spec()
        assert [c.name for c in reg.match(spec)] == ['m']

    def test_match_filters_by_connector(self, swept, tmp_path):
        reg = self._reg(tmp_path)
        reg.set_collector('nope', 'mllabs.MetricCollector', Connector(node_query='^zzz'),
                          params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'},
                                  'output_var': '*'})
        assert reg.match(swept[0].get_spec()) == []

    def test_match_restricted_to_names(self, swept, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a'); self._set(reg, 'b')
        spec = swept[0].get_spec()
        assert [c.name for c in reg.match(spec, names=['a'])] == ['a']

    def test_registration_persists_without_an_explicit_save(self, tmp_path):
        """set_collector writes through — there is no Collectors.save()."""
        self._set(self._reg(tmp_path), 'a')
        loaded = Collectors(tmp_path)
        assert loaded.names() == ['a']
        assert loaded.get_collector('a').path == tmp_path / 'a'

    def test_empty_store_is_an_empty_registry(self, tmp_path):
        assert Collectors(tmp_path / 'nothing').names() == []

    def test_remove_clears_the_stored_row(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a')
        reg.remove_collector('a')
        assert Collectors(tmp_path).names() == []

    def test_pathless_registry_persists_nothing(self, tmp_path):
        reg = Collectors()
        reg.set_collector('m', 'mllabs.MetricCollector', Connector(),
                          path=tmp_path / 'm', params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}, 'output_var': '*'})
        assert reg.names() == ['m']
        assert not (tmp_path / 'collectors.db').exists()


class TestCollectorStore:
    """The entity columns describe a row without unpickling its instance —
    that is the whole reason they are columns and not part of the blob."""

    def _reg(self, tmp_path):
        reg = Collectors(tmp_path)
        reg.set_collector('m', 'mllabs.MetricCollector',
                          Connector(node_query='^dt', processor='mock.DummyHead'),
                          params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}, 'output_var': '*'})
        return reg

    def test_entity_keeps_the_string_form_as_given(self, tmp_path):
        self._reg(tmp_path)
        entity = CollectorStore(tmp_path).get_entity('m')
        assert entity.collector == 'mllabs.MetricCollector'
        assert entity.path == str(tmp_path / 'm')

    def test_entity_describes_the_connector(self, tmp_path):
        self._reg(tmp_path)
        entity = CollectorStore(tmp_path).get_entity('m')
        assert entity.connector == {
            '__ref__': 'mllabs._connector.Connector',
            '__params__': {'node_query': '^dt', 'edges': None,
                           'processor': 'mock.DummyHead'},
        }

    def test_a_class_argument_is_recorded_as_its_ref(self, tmp_path):
        from mllabs import MetricCollector
        reg = Collectors(tmp_path)
        reg.set_collector('c', MetricCollector, Connector(), params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}, 'output_var': '*'})
        assert CollectorStore(tmp_path).get_entity('c').collector == \
            'mllabs.collector._metric.MetricCollector'

    def test_list_entities_is_registration_ordered(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.set_collector('z', 'mllabs.MetricCollector', Connector(), params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}, 'output_var': '*'})
        assert [e.name for e in CollectorStore(tmp_path).list_entities()] == ['m', 'z']

    def test_unknown_name_reads_as_none(self, tmp_path):
        store = CollectorStore(tmp_path)
        assert store.build('nope') is None and store.get_entity('nope') is None

    def test_build_reassembles_from_entity_and_params(self, tmp_path):
        reg = self._reg(tmp_path)
        rebuilt = CollectorStore(tmp_path).build('m')
        assert type(rebuilt) is type(reg.get_collector('m'))
        assert rebuilt.output_var == '*'
        assert rebuilt.connector.node_query == '^dt'
        assert rebuilt.path == tmp_path / 'm'

    def test_params_are_stored_as_given(self, tmp_path):
        """The ref spec is what is written — resolution happens on build,
        the same way it happens on set_collector."""
        self._reg(tmp_path)
        assert CollectorStore(tmp_path).get_params('m')['metric_func'] == \
            {'__callable__': 'sklearn.metrics.accuracy_score'}

    def test_no_instance_state_survives(self, tmp_path):
        """Nothing about a live Collector is stored — only the two halves it
        was built from — so run-time state cannot leak into the next build."""
        reg = self._reg(tmp_path)
        reg.get_collector('m')._buf['dt'] = {0: {0: 'x'}}
        assert CollectorStore(tmp_path).build('m')._buf == {}

    def test_remove_drops_the_params_file(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.remove_collector('m')
        store = CollectorStore(tmp_path)
        assert store.get_params('m') is None and store.build('m') is None
