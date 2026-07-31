import pytest

from mllabs import Collectors, Connector, PipelineBuilder
from mllabs.experiment import Trial, BaseExperiment, SimpleExperiment


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
def simple():
    return SimpleExperiment(
        'lgbm', processor=LGBM, edges=EDGES,
        params={'random_state': 42},
        param_grid={'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    )


class TestSimpleExperimentGrid:
    def test_trial_count_is_cartesian_product(self, simple):
        assert simple.get_trial_nums() == 4

    def test_no_grid_yields_single_trial(self):
        e = SimpleExperiment('one', processor=TREE, edges=EDGES, params={'max_depth': 3})
        assert e.get_trial_nums() == 1
        assert e.get_next_trial().name == 'one'

    def test_fixed_params_merged_into_every_trial(self, simple):
        assert all(t.params['random_state'] == 42 for t in simple.get_trials())

    def test_grid_params_cover_all_combinations(self, simple):
        combos = {(t.params['max_depth'], t.params['learning_rate'])
                  for t in simple.get_trials()}
        assert combos == {(3, 0.05), (3, 0.1), (5, 0.05), (5, 0.1)}

    def test_grid_overrides_fixed_param(self):
        e = SimpleExperiment('x', processor=TREE, edges=EDGES,
                             params={'max_depth': 1}, param_grid={'max_depth': [7]})
        assert e.get_next_trial().params['max_depth'] == 7

    def test_shared_fields_are_identical(self, simple):
        for t in simple.get_trials():
            assert t.processor == LGBM
            assert t.method == 'predict'
            assert t.edges == EDGES
            assert t.label == 'lgbm'

    def test_edges_not_shared_between_trials(self, simple):
        a, b = simple.get_trials()[:2]
        a.edges['X'] = 'mutated'
        assert b.edges['X'] == 'scaler:(*)'

    def test_tags_propagate_to_trials(self):
        e = SimpleExperiment('t', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': [1, 2]}, tags=['final'])
        assert all(t.tag == ['final'] for t in e.get_trials())


class TestSimpleExperimentSequence:
    def test_names_are_unique_and_zero_padded(self, simple):
        assert [t.name for t in simple.get_trials()] == [
            'lgbm_0', 'lgbm_1', 'lgbm_2', 'lgbm_3']

    def test_wide_index_padding(self):
        e = SimpleExperiment('w', processor=TREE, edges=EDGES,
                             param_grid={'max_depth': list(range(12))})
        assert e.get_trials()[0].name == 'w_00'

    def test_order_is_deterministic(self):
        def mk():
            return SimpleExperiment('d', processor=TREE, edges=EDGES,
                                    param_grid={'b': [1, 2], 'a': ['x', 'y']})
        assert ([t.params for t in mk().get_trials()]
                == [t.params for t in mk().get_trials()])

    def test_cursor_advances(self, simple):
        assert simple.get_next_trial().name == 'lgbm_0'
        assert simple.get_next_trial().name == 'lgbm_1'

    def test_exhausted_cursor_raises(self):
        e = SimpleExperiment('s', processor=TREE, edges=EDGES)
        e.get_next_trial()
        with pytest.raises(StopIteration):
            e.get_next_trial()

    def test_reset_rewinds(self, simple):
        simple.get_next_trial()
        simple.reset()
        assert simple.get_next_trial().name == 'lgbm_0'

    def test_get_trials_is_repeatable(self, simple):
        assert [t.name for t in simple.get_trials()] == [t.name for t in simple.get_trials()]

    def test_get_trial_does_not_move_cursor(self, simple):
        assert simple.get_trial(2).name == 'lgbm_2'
        assert simple.get_next_trial().name == 'lgbm_0'

    def test_get_trial_out_of_range(self, simple):
        with pytest.raises(IndexError):
            simple.get_trial(4)


class TestSimpleExperimentValidation:
    def test_processor_class_rejected(self):
        from sklearn.tree import DecisionTreeClassifier
        with pytest.raises(TypeError, match='processor must be'):
            SimpleExperiment('x', processor=DecisionTreeClassifier, edges=EDGES)

    def test_adapter_instance_rejected(self):
        from mllabs.adapter import DefaultAdapter
        with pytest.raises(TypeError, match='adapter must be'):
            SimpleExperiment('x', processor=TREE, edges=EDGES, adapter=DefaultAdapter())

    def test_live_object_in_params_rejected(self):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match='must be plain data'):
            SimpleExperiment('x', processor=TREE, edges=EDGES,
                             params={'cat_features': ColSelector('*')})

    def test_live_object_in_grid_rejected(self):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match='must be plain data'):
            SimpleExperiment('x', processor=TREE, edges=EDGES,
                             param_grid={'cat_features': [ColSelector('*')]})

    def test_empty_edges_rejected(self):
        with pytest.raises(ValueError, match='non-empty'):
            SimpleExperiment('x', processor=TREE, edges={})

    def test_non_string_edge_rejected(self):
        with pytest.raises(TypeError, match='DSL string'):
            SimpleExperiment('x', processor=TREE, edges={'X': ['a', 'b']})

    def test_scalar_grid_value_rejected(self):
        with pytest.raises(TypeError, match='must be a list'):
            SimpleExperiment('x', processor=TREE, edges=EDGES, param_grid={'max_depth': 3})

    def test_empty_grid_value_rejected(self):
        with pytest.raises(ValueError, match='is empty'):
            SimpleExperiment('x', processor=TREE, edges=EDGES, param_grid={'max_depth': []})


class TestTrialIdentity:
    def test_get_attrs_shape_matches_node_attrs(self, pipeline, simple):
        """A Trial must look like a node to Connector/executor/Collector.

        It carries ``tag`` (selection lives on the Experiment side now) and no
        ``serial`` — its identity is ``trial_id(pipeline)`` instead.
        """
        trial_attrs = simple.get_next_trial().get_attrs()
        node_attrs = pipeline.build().get_node_attrs('scaler')
        assert set(trial_attrs) - {'tag'} == set(node_attrs) - {'serial'}

    def test_role_is_head(self, simple):
        assert simple.get_next_trial().get_attrs()['role'] == 'head'

    def test_same_definition_same_id(self, pipeline):
        built = pipeline.build()
        a = Trial('a', TREE, EDGES, params={'max_depth': 3})
        b = Trial('b', TREE, EDGES, params={'max_depth': 3})
        assert a.trial_id(built) == b.trial_id(built)

    def test_name_does_not_affect_id(self, pipeline):
        built = pipeline.build()
        assert (Trial('one', TREE, EDGES).trial_id(built)
                == Trial('two', TREE, EDGES).trial_id(built))

    def test_param_change_changes_id(self, pipeline):
        built = pipeline.build()
        assert (Trial('a', TREE, EDGES, params={'max_depth': 3}).trial_id(built)
                != Trial('a', TREE, EDGES, params={'max_depth': 5}).trial_id(built))

    def test_processor_change_changes_id(self, pipeline):
        built = pipeline.build()
        assert Trial('a', TREE, EDGES).trial_id(built) != Trial('a', LGBM, EDGES).trial_id(built)

    def test_param_order_does_not_affect_id(self, pipeline):
        built = pipeline.build()
        assert (Trial('a', TREE, EDGES, params={'a': 1, 'b': 2}).trial_id(built)
                == Trial('a', TREE, EDGES, params={'b': 2, 'a': 1}).trial_id(built))

    def test_upstream_stage_serial_is_part_of_id(self, pipeline):
        """A Stage edit must invalidate dependent Trials — Heads left the
        pipeline, so _bump_serials can no longer cascade into them."""
        trial = Trial('a', TREE, EDGES, params={'max_depth': 3})
        before = trial.trial_id(pipeline.build())

        pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1, f2}'},
                         params={'with_std': False})
        assert trial.trial_id(pipeline.build()) != before

    def test_unrelated_stage_edit_does_not_change_id(self, pipeline):
        pipeline.set_grp('other', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('other_node', grp='other')
        trial = Trial('a', TREE, EDGES)
        before = trial.trial_id(pipeline.build())

        pipeline.set_grp('other', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'}, params={'with_mean': False})
        assert trial.trial_id(pipeline.build()) == before

    def test_upstream_serials_lists_referenced_stages(self, pipeline):
        assert set(Trial('a', TREE, EDGES).upstream_serials(pipeline.build())) == {'scaler'}


class TestBaseExperimentContract:
    def test_subclass_must_implement(self):
        e = BaseExperiment('x')
        with pytest.raises(NotImplementedError):
            e.get_trial_nums()
        with pytest.raises(NotImplementedError):
            e.get_next_trial()

    def test_reset_is_noop_by_default(self):
        assert BaseExperiment('x').reset() is None


class TestExperimentCollectorNames:
    """An Experiment records collector *names*; instances live in a registry."""

    def test_no_names_by_default(self, simple):
        assert simple.collector_names == []

    def test_constructor_names(self):
        e = SimpleExperiment('x', processor=TREE, edges=EDGES, collectors=['m'])
        assert e.collector_names == ['m']

    def test_use_collector_appends(self, simple):
        simple.use_collector('a', 'b')
        assert simple.collector_names == ['a', 'b']

    def test_use_collector_is_idempotent(self, simple):
        simple.use_collector('a').use_collector('a')
        assert simple.collector_names == ['a']

    def test_drop_collector(self, simple):
        simple.use_collector('a', 'b').drop_collector('a')
        assert simple.collector_names == ['b']

    def test_holds_no_live_collector(self, simple):
        """An Experiment must stay pure definition — nothing live inside it."""
        simple.use_collector('m')
        assert not hasattr(simple, 'collectors')


class TestCollectorsRegistry:
    def _reg(self, tmp_path):
        return Collectors(tmp_path)

    def _set(self, reg, name='m', **kw):
        return reg.set_collector(
            name, 'mllabs.MetricCollector', Connector(role='head'),
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

    def test_match_by_role(self, simple, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg)
        attrs = simple.get_next_trial().get_attrs()
        assert [c.name for c in reg.match(attrs)] == ['m']

    def test_match_filters_by_connector(self, simple, tmp_path):
        reg = self._reg(tmp_path)
        reg.set_collector('nope', 'mllabs.MetricCollector', Connector(node_query='^zzz'),
                          params={'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'},
                                  'output_var': '*'})
        assert reg.match(simple.get_next_trial().get_attrs()) == []

    def test_match_restricted_to_names(self, simple, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a'); self._set(reg, 'b')
        attrs = simple.get_next_trial().get_attrs()
        assert [c.name for c in reg.match(attrs, names=['a'])] == ['a']

    def test_save_load_roundtrip(self, tmp_path):
        reg = self._reg(tmp_path)
        self._set(reg, 'a')
        reg.save()
        loaded = Collectors.load(tmp_path)
        assert loaded.names() == ['a']
        assert loaded.get_collector('a').path == tmp_path / 'a'

    def test_load_missing_index_is_empty(self, tmp_path):
        assert Collectors.load(tmp_path / 'nothing').names() == []

    def test_save_without_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match='no path'):
            Collectors().save()
