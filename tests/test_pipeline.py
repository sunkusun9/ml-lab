import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mllabs._pipeline import PipelineBuilder, _PipelineGroup, _PipelineNode, _DataSourceNode, VAR_TYPES


def _dummy_metric(y, p):
    return 0.5


# processor= below is always a "module.ClassName" string (PipelineBuilder never
# resolves it) — 'mock.DummyStage' etc. reference the classes in mock.py,
# used here as opaque, structurally-distinct identifiers (these pipeline-graph
# tests never actually build/fit a node).
DummyStage = 'mock.DummyStage'
DummyHead = 'mock.DummyHead'
AnotherProcessor = 'mock.AnotherProcessor'


@pytest.fixture
def p():
    return PipelineBuilder()


@pytest.fixture
def sp():
    p = PipelineBuilder()
    p.set_grp('stage1', processor=DummyStage, method='transform',
              edges={'X': '{x1}'})
    p.set_node('s1', grp='stage1')
    p.set_grp('head1', processor=DummyHead, method='predict',
              edges={'X': '{x1}', 'y': '{target}'})
    p.set_node('h1', grp='head1')
    return p


class TestInit:
    def test_datasource_exists(self, p):
        assert None in p.nodes
        assert p.nodes[None].name == 'Data_Source'

    def test_datasource_is_datasource_node(self, p):
        assert isinstance(p.nodes[None], _DataSourceNode)


class TestSetGrp:

    def test_with_parent(self, p):
        p.set_grp('parent')
        p.set_grp('child', parent='parent')
        assert 'child' in p.grps['parent'].children
        assert p.grps['child'].parent == 'parent'

    def test_parent_not_found(self, p):
        with pytest.raises(ValueError):
            p.set_grp('g1', parent='no_exist')

    def test_name_conflicts_with_node(self, sp):
        with pytest.raises(ValueError):
            sp.set_grp('s1')

    def test_exist_skip(self, p):
        p.set_grp('g1', processor=DummyStage)
        r = p.set_grp('g1', processor=AnotherProcessor, exist='skip')
        assert r['result'] == 'skip'
        assert p.grps['g1'].processor == DummyStage

    def test_exist_error(self, p):
        p.set_grp('g1')
        with pytest.raises(ValueError):
            p.set_grp('g1', exist='error')

    def test_exist_replace(self, p):
        p.set_grp('g1', processor=DummyStage)
        r = p.set_grp('g1', processor=AnotherProcessor, exist='replace')
        assert r['result'] == 'update'
        assert p.grps['g1'].processor == AnotherProcessor

    def test_replace_parent_change(self, p):
        p.set_grp('p1')
        p.set_grp('p2')
        p.set_grp('child', parent='p1')
        assert 'child' in p.grps['p1'].children
        p.set_grp('child', parent='p2', exist='replace')
        assert 'child' not in p.grps['p1'].children
        assert 'child' in p.grps['p2'].children

    def test_replace_affected_nodes_with_group_edges(self, p):
        # Control: group has edges → affected_nodes should include group nodes (works before fix)
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        r = p.set_grp('g1', processor=AnotherProcessor, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_replace_affected_nodes_without_group_edges(self, p):
        # Bug: group has no edges, node has own edges → affected_nodes incorrectly empty
        p.set_grp('g1', processor=DummyStage, method='transform')
        p.set_node('n1', grp='g1', edges={'X': '{x1}'})
        r = p.set_grp('g1', processor=AnotherProcessor, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_replace_affected_nodes_child_grp_when_parent_has_no_edges(self, p):
        # Bug: parent group has no edges, child group has nodes → parent update misses child nodes
        p.set_grp('parent')
        p.set_grp('child', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        r = p.set_grp('parent', params={'a': 1}, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_with_all_attrs(self, p):
        p.set_grp('g1', processor=DummyStage,
                  edges={'X': '{x1}'}, method='transform',
                  params={'n': 10})
        g = p.grps['g1']
        assert g.processor == DummyStage
        assert g.edges == {'X': '{x1}'}
        assert g.method == 'transform'
        assert g.params == {'n': 10}


class TestSetNode:
    def test_new_node(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        r = p.set_node('n1', grp='g1')
        assert r['result'] == 'new'
        assert 'n1' in p.nodes
        assert 'n1' in p.grps['g1'].nodes

    def test_grp_not_found(self, p):
        with pytest.raises(ValueError):
            p.set_node('n1', grp='no_exist')

    def test_name_conflicts_with_group(self, p):
        p.set_grp('g1')
        with pytest.raises(ValueError):
            p.set_node('g1', grp='g1')

    def test_processor_required(self, p):
        p.set_grp('g1', method='transform', edges={'X': '{x1}'})
        with pytest.raises(ValueError, match='processor'):
            p.set_node('n1', grp='g1')

    def test_method_required(self, p):
        p.set_grp('g1', processor=DummyStage, edges={'X': '{x1}'})
        with pytest.raises(ValueError, match='method'):
            p.set_node('n1', grp='g1')

    def test_edges_required(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='edges'):
            p.set_node('n1', grp='g1')

    def test_output_edges_updated(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        assert 's2' in sp.nodes['s1'].output_edges

    def test_exist_skip(self, sp):
        r = sp.set_node('s1', grp='stage1', processor=AnotherProcessor, exist='skip')
        assert r['result'] == 'skip'

    def test_exist_error(self, sp):
        with pytest.raises(ValueError):
            sp.set_node('s1', grp='stage1', exist='error')

    def test_exist_replace(self, sp):
        sp.set_node('s1', grp='stage1', processor=AnotherProcessor, exist='replace')
        assert sp.nodes['s1'].processor == AnotherProcessor

    def test_replace_changes_group(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_node('n1', grp='g2', exist='replace')
        assert 'n1' not in p.grps['g1'].nodes
        assert 'n1' in p.grps['g2'].nodes
        assert p.nodes['n1'].grp == 'g2'

    def test_replace_returns_affected_nodes(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        r = sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in r['affected_nodes']

    def test_replace_preserves_output_edges(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in sp.nodes['s1'].output_edges

    def test_with_node_level_params(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1})
        p.set_node('n1', grp='g1', params={'b': 2})
        assert p.nodes['n1'].params == {'b': 2}

class TestProcessorStringRef:
    """processor is always passed/stored as a "module.ClassName" string —
    PipelineBuilder never resolves it (see resolve_processor in _node_processor.py,
    the only place it becomes a real class)."""

    def test_set_grp_stores_string_unresolved(self, p):
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        assert p.grps['g1'].processor == 'sklearn.preprocessing.StandardScaler'

    def test_set_grp_string_resolved_at_use(self, p):
        from mllabs._serialize import resolve_processor
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        assert resolve_processor(p.grps['g1'].processor) is StandardScaler

    def test_set_node_stores_string_unresolved(self, p):
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', processor='sklearn.tree.DecisionTreeClassifier')
        assert p.nodes['n1'].processor == 'sklearn.tree.DecisionTreeClassifier'

    def test_set_node_string_resolved_at_use(self, p):
        from mllabs._serialize import resolve_processor
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', processor='sklearn.tree.DecisionTreeClassifier')
        assert resolve_processor(p.nodes['n1'].processor) is DecisionTreeClassifier

    def test_diff_skips_on_identical_string(self, p):
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        r = p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler',
                       method='transform', edges={'X': '{x1}'})
        assert r['result'] == 'skip'

    def test_invalid_string_does_not_raise_at_set_time(self, p):
        # Structure-only — no import happens until resolve_processor runs at
        # point of use (_node_processor.py), not at set_grp.
        r = p.set_grp('g1', processor='not.a.real.module.Thing',
                       method='transform', edges={'X': '{x1}'})
        assert r['result'] == 'new'

    def test_invalid_string_raises_when_resolved(self, p):
        from mllabs._serialize import resolve_processor
        p.set_grp('g1', processor='not.a.real.module.Thing',
                  method='transform', edges={'X': '{x1}'})
        with pytest.raises(Exception):
            resolve_processor(p.grps['g1'].processor)


class TestAdapterStringRef:
    """set_grp/set_node no longer eagerly instantiate ``adapter`` — the spec
    (str / {'__ref__':...} dict) is stored as-is on the Node/Grp, and only
    resolved to an instance at point of use via ``resolve_node_adapter``."""

    def test_set_grp_string_ref_stored_unresolved(self, p):
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter='mllabs.adapter._default.DefaultAdapter')
        assert p.grps['g1'].adapter == 'mllabs.adapter._default.DefaultAdapter'

    def test_set_grp_string_ref_resolved_at_use(self, p):
        from mllabs.adapter import DefaultAdapter, resolve_node_adapter
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter='mllabs.adapter._default.DefaultAdapter')
        adapter = resolve_node_adapter('sklearn.tree.DecisionTreeClassifier', p.grps['g1'].adapter)
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'both'

    def test_set_grp_ref_dict_stored_unresolved(self, p):
        spec = {'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                '__params__': {'eval_mode': 'valid', 'verbose': 0.25}}
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter=spec)
        assert p.grps['g1'].adapter == spec

    def test_set_grp_ref_dict_resolved_at_use(self, p):
        from mllabs.adapter import DefaultAdapter, resolve_node_adapter
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                           '__params__': {'eval_mode': 'valid', 'verbose': 0.25}})
        adapter = resolve_node_adapter('sklearn.tree.DecisionTreeClassifier', p.grps['g1'].adapter)
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'valid'
        assert adapter.verbose == pytest.approx(0.25)

    def test_set_node_ref_dict_resolved_at_use(self, sp):
        from mllabs.adapter import DefaultAdapter, resolve_node_adapter
        sp.set_node('h1', grp='head1',
                    adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                             '__params__': {'eval_mode': 'none'}}, exist='replace')
        node_adapter = sp.nodes['h1'].adapter
        assert node_adapter == {'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                                 '__params__': {'eval_mode': 'none'}}
        adapter = resolve_node_adapter(sp.nodes['h1'].processor, node_adapter)
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'none'

    def test_instance_rejected(self, p):
        from mllabs.adapter import DefaultAdapter
        with pytest.raises(TypeError, match='adapter must be'):
            p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      adapter=DefaultAdapter(eval_mode='valid'))

    def test_instance_rejection_suggests_ref_form(self, p):
        from mllabs.adapter import DefaultAdapter
        with pytest.raises(TypeError, match='mllabs.adapter._default.DefaultAdapter'):
            p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      adapter=DefaultAdapter())

    def test_class_rejected_for_processor(self, p):
        from sklearn.tree import DecisionTreeClassifier
        with pytest.raises(TypeError, match='processor must be'):
            p.set_grp('g1', processor=DecisionTreeClassifier,
                      method='predict', edges={'X': '{x1}', 'y': '{target}'})

    def test_ref_dict_diff_skips_on_identical_spec(self, p):
        spec = {'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                '__params__': {'eval_mode': 'valid'}}
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'}, adapter=spec)
        r = p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      adapter=dict(spec))
        assert r['result'] == 'skip'

    def test_ref_dict_diff_detects_param_change(self, p):
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                           '__params__': {'eval_mode': 'valid'}})
        r = p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                               '__params__': {'eval_mode': 'both'}})
        assert r['result'] == 'update'


class TestParamsRefDict:
    """set_grp/set_node no longer eagerly resolve {'__ref__':...}/
    {'__callable__':...} entries inside params — stored as-is, resolved only
    at point of use (_node_processor.py's _resolve_params, via
    resolve_ref_values)."""

    def test_set_grp_stores_colselector_ref_unresolved(self, p):
        spec = {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '*@categorical'}}
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'cat_features': spec, 'max_depth': 3})
        params = p.grps['g1'].params
        assert params['cat_features'] == spec
        assert params['max_depth'] == 3

    def test_set_grp_colselector_ref_resolved_at_use(self, p):
        from mllabs import ColSelector
        from mllabs._serialize import resolve_ref_values
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'cat_features': {'__ref__': 'mllabs.ColSelector',
                                           '__params__': {'dsl_string': '*@categorical'}},
                          'max_depth': 3})
        sel = resolve_ref_values(p.grps['g1'].params['cat_features'])
        assert isinstance(sel, ColSelector)
        assert sel.dsl_string == '*@categorical'

    def test_set_node_stores_colselector_ref_unresolved(self, p):
        spec = {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '^cat_'}}
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'cat_features': spec})
        assert p.nodes['n1'].params['cat_features'] == spec

    def test_set_node_colselector_ref_resolved_at_use(self, p):
        from mllabs import ColSelector
        from mllabs._serialize import resolve_ref_values
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1',
                   params={'cat_features': {'__ref__': 'mllabs.ColSelector',
                                            '__params__': {'dsl_string': '^cat_'}}})
        sel = resolve_ref_values(p.nodes['n1'].params['cat_features'])
        assert isinstance(sel, ColSelector)
        assert sel.dsl_string == '^cat_'

    def test_ref_dict_diff_skips_on_identical_spec(self, p):
        spec = {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '*@categorical'}}
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'cat_features': spec})
        r = p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      params={'cat_features': dict(spec)})
        assert r['result'] == 'skip'

    def test_instance_rejected(self, p):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match="params\\['cat_features'\\] must be plain data"):
            p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      params={'cat_features': ColSelector('*@categorical')})

    def test_nested_instance_rejected(self, p):
        from mllabs import ColSelector
        with pytest.raises(TypeError, match="params\\['a'\\]\\['b'\\]\\[0\\]"):
            p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      params={'a': {'b': [ColSelector('*')]}})

    def test_callable_rejected_with_callable_hint(self, p):
        import math
        with pytest.raises(TypeError, match='__callable__'):
            p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      params={'metric': math.sqrt})

    def test_plain_data_accepted(self, p):
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'n': 3, 'f': 0.5, 'flag': True, 'name': 'x', 'none': None,
                          'seq': [1, 2], 'tup': (1, 2), 'nested': {'a': [1, {'b': 2}]}})
        assert p.grps['g1'].params['nested'] == {'a': [1, {'b': 2}]}

    def test_numpy_scalar_accepted(self, p):
        import numpy as np
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'lr': np.float64(0.1), 'n': np.int64(5)})
        assert p.grps['g1'].params['n'] == 5

    def test_plain_string_param_untouched(self, p):
        p.set_grp('g1', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'eval_metric': 'AUC', 'criterion': 'gini'})
        params = p.grps['g1'].params
        assert params['eval_metric'] == 'AUC'
        assert params['criterion'] == 'gini'


class TestGroupHierarchy:
    def test_edges_full_override_by_default(self, p):
        p.set_grp('parent', edges={'X': '{a}'})
        p.set_grp('child', parent='parent', edges={'X': '{b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{b}'

    def test_edges_plus_continues_from_parent(self, p):
        p.set_grp('parent', edges={'X': '{a}'})
        p.set_grp('child', parent='parent', edges={'X': '+ {b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{a} + {b}'

    def test_edges_minus_continues_from_parent(self, p):
        p.set_datasource({'a': 'numerical', 'b': 'numerical'})
        p.set_grp('parent', edges={'X': '{a, b}'})
        p.set_grp('child', parent='parent', edges={'X': '- {b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{a, b} - {b}'

    def test_edges_plus_without_parent_value_raises(self, p):
        p.set_grp('parent')
        with pytest.raises(ValueError, match='no parent value'):
            p.set_grp('child', parent='parent', edges={'X': '+ {b}'})

    def test_params_no_override(self, p):
        p.set_grp('parent', params={'a': 1, 'b': 2})
        p.set_grp('child', parent='parent', params={'b': 3, 'c': 4})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['params'] == {'a': 1, 'b': 3, 'c': 4}

    def test_processor_inherited(self, p):
        p.set_grp('parent', processor=DummyStage)
        p.set_grp('child', parent='parent')
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == DummyStage

    def test_processor_overridden(self, p):
        p.set_grp('parent', processor=DummyStage)
        p.set_grp('child', parent='parent', processor=AnotherProcessor)
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == AnotherProcessor

    def test_method_inherited(self, p):
        p.set_grp('parent', method='transform')
        p.set_grp('child', parent='parent')
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['method'] == 'transform'

    def test_three_level_hierarchy(self, p):
        p.set_grp('gp', processor=DummyStage, edges={'X': '{a}'},
                  params={'x': 1})
        p.set_grp('par', parent='gp', method='transform',
                  edges={'X': '+ {b}'}, params={'y': 2})
        p.set_grp('child', parent='par',
                  edges={'X': '+ {c}'}, params={'z': 3})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == DummyStage
        assert attrs['method'] == 'transform'
        assert attrs['edges']['X'] == '{a} + {b} + {c}'
        assert attrs['params'] == {'x': 1, 'y': 2, 'z': 3}

    def test_attrs_caching(self, p):
        p.set_grp('g1', processor=DummyStage)
        attrs1 = p.grps['g1'].get_attrs(p.grps)
        attrs2 = p.grps['g1'].get_attrs(p.grps)
        assert attrs1 is attrs2

    def test_update_attrs_invalidates_cache(self, p):
        p.set_grp('g1', processor=DummyStage)
        p.grps['g1'].get_attrs(p.grps)
        assert p.grps['g1'].attrs is not None
        p.grps['g1'].update_attrs()
        assert p.grps['g1'].attrs is None


class TestNodeAttrs:
    def test_merges_from_group(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1})
        p.set_node('n1', grp='g1')
        spec = p.get_node_spec('n1')
        assert spec.processor == DummyStage
        assert spec.method == 'transform'
        assert spec.params == {'a': 1}

    def test_node_overrides_processor(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', processor=AnotherProcessor)
        spec = p.get_node_spec('n1')
        assert spec.processor == AnotherProcessor

    def test_node_edges_full_override_by_default(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{a}'})
        p.set_node('n1', grp='g1', edges={'X': '{b}'})
        spec = p.get_node_spec('n1')
        assert spec.edges['X'] == '{b}'

    def test_node_edges_plus_continues_from_group(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{a}'})
        p.set_node('n1', grp='g1', edges={'X': '+ {b}'})
        spec = p.get_node_spec('n1')
        assert spec.edges['X'] == '{a} + {b}'

    def test_node_edges_minus_continues_from_group(self, p):
        p.set_datasource({'a': 'numerical', 'b': 'numerical'})
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{a, b}'})
        p.set_node('n1', grp='g1', edges={'X': '- {b}'})
        spec = p.get_node_spec('n1')
        assert spec.edges['X'] == '{a, b} - {b}'

    def test_node_edges_plus_without_group_value_raises(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='no parent value'):
            p.set_node('n1', grp='g1', edges={'X': '+ {b}'})

    def test_node_params_no_override(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1, 'b': 2})
        p.set_node('n1', grp='g1', params={'b': 3, 'c': 4})
        spec = p.get_node_spec('n1')
        assert spec.params == {'a': 1, 'b': 3, 'c': 4}

    def test_adapter_left_unresolved_when_unspecified(self, p):
        # By-processor-class default resolution is deferred to point of use
        # (resolve_node_adapter), not decided at pipeline-definition time.
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        spec = p.get_node_spec('n1')
        assert spec.adapter is None

    def test_adapter_auto_detect_at_use(self, p):
        from mllabs.adapter import resolve_node_adapter
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        spec = p.get_node_spec('n1')
        adapter = resolve_node_adapter(spec.processor, spec.adapter)
        assert adapter is not None

    def test_node_attrs_caching(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        a1 = p.nodes['n1'].get_spec(p.grps)
        a2 = p.nodes['n1'].get_spec(p.grps)
        assert a1 is a2


class TestNameValidation:
    @pytest.mark.parametrize('name', [
        'test__name', 'a/b', 'a\\b', 'a\0b', 'a<b', 'a>b',
        'a:b', 'a"b', 'a|b', 'a?b', 'a*b'
    ])
    def test_invalid_names_rejected(self, p, name):
        with pytest.raises(ValueError):
            p.set_grp(name)

    def test_valid_names_accepted(self, p):
        for name in ['test', 'test_name', 'test-name', 'test123']:
            p.set_grp(name)
            assert name in p.grps


class TestEdgeValidation:
    def test_edge_node_not_found(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='does not reference an existing node'):
            p.set_node('n1', grp='g1', edges={'X': 'no_exist:(*)'})

class TestDataSourceEdgeRequiresList:
    def test_set_grp_list_var_spec_accepted(self, p):
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_stage_to_stage_edge_still_allows_non_list(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{f1}'})
        p.set_node('a', grp='g1')
        r = p.set_grp('g2', processor=DummyStage, method='transform',
                       edges={'X': 'a:(some_pattern)'})
        assert r['result'] == 'new'


class TestDataSourceEdgeSchemaMembership:
    """set_grp/set_node only validate DSL *structure* — column/schema
    membership is resolved lazily, at process time, against real data (see
    _edge_dsl.validate_edges). So an edges string referencing a column that
    doesn't exist in the schema is accepted here regardless."""

    def test_no_schema_no_check(self, p):
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_columns_in_schema_accepted(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_unknown_column_not_rejected_at_definition_time(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                       edges={'X': '{f1, unknown_col}'})
        assert r['result'] == 'new'

    def test_set_node_unknown_column_not_rejected_at_definition_time(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        p.set_grp('g1', processor=DummyStage, method='transform')
        r = p.set_node('n1', grp='g1', edges={'X': '{unknown_col}'})
        assert r['result'] == 'new'


class TestCycleDetection:
    def test_direct_cycle(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        with pytest.raises(ValueError, match='cycle'):
            sp.set_node('s1', grp='stage1', edges={'X': 's2:(*)'}, exist='replace')

    def test_indirect_cycle(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        with pytest.raises(ValueError, match='cycle'):
            p.set_node('a', grp='g1', edges={'X': 'c:(*)'}, exist='replace')

    def test_no_cycle_chain(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        assert 'c' in p.nodes

    def test_diamond_no_cycle(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('c', grp='g3')
        p.set_grp('g4', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*) + c:(*)'})
        p.set_node('d', grp='g4')
        assert 'd' in p.nodes


class TestRenameGrp:
    def test_basic_rename(self, p):
        p.set_grp('old')
        p.rename_grp('old', 'new')
        assert 'old' not in p.grps
        assert 'new' in p.grps
        assert p.grps['new'].name == 'new'

    def test_updates_parent_children(self, p):
        p.set_grp('parent')
        p.set_grp('old', parent='parent')
        p.rename_grp('old', 'new')
        assert 'new' in p.grps['parent'].children
        assert 'old' not in p.grps['parent'].children

    def test_updates_child_parent(self, p):
        p.set_grp('old')
        p.set_grp('child', parent='old')
        p.rename_grp('old', 'new')
        assert p.grps['child'].parent == 'new'

    def test_updates_node_grp(self, sp):
        sp.rename_grp('stage1', 'renamed')
        assert sp.nodes['s1'].grp == 'renamed'

    def test_invalidates_node_cache(self, sp):
        sp.nodes['s1'].get_spec(sp.grps)
        sp.rename_grp('stage1', 'renamed')
        assert sp.nodes['s1'].spec is None

    def test_source_not_found(self, p):
        with pytest.raises(ValueError):
            p.rename_grp('no_exist', 'new')

    def test_target_exists(self, p):
        p.set_grp('a')
        p.set_grp('b')
        with pytest.raises(ValueError):
            p.rename_grp('a', 'b')


class TestRemoveGrp:
    def test_remove_empty(self, p):
        p.set_grp('g1')
        p.remove_grp('g1')
        assert 'g1' not in p.grps

    def test_updates_parent(self, p):
        p.set_grp('parent')
        p.set_grp('child', parent='parent')
        p.remove_grp('child')
        assert 'child' not in p.grps['parent'].children

    def test_not_found(self, p):
        with pytest.raises(ValueError):
            p.remove_grp('no_exist')

    def test_has_children(self, p):
        p.set_grp('parent')
        p.set_grp('child', parent='parent')
        with pytest.raises(ValueError, match='child'):
            p.remove_grp('parent')

    def test_has_nodes(self, sp):
        with pytest.raises(ValueError, match='node'):
            sp.remove_grp('stage1')


class TestRemoveNode:
    def test_remove_leaf(self, sp):
        sp.remove_node('h1')
        assert 'h1' not in sp.nodes
        assert 'h1' not in sp.grps['head1'].nodes

    def test_updates_output_edges(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        assert 's2' in sp.nodes['s1'].output_edges
        sp.remove_node('s2')
        assert 's2' not in sp.nodes['s1'].output_edges

    def test_not_found(self, p):
        with pytest.raises(ValueError):
            p.remove_node('no_exist')

    def test_cannot_remove_datasource(self, p):
        with pytest.raises(ValueError):
            p.remove_node(None)

    def test_has_descendants(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        with pytest.raises(ValueError, match='dependent'):
            sp.remove_node('s1')


class TestGetNodeNames:
    def test_none_returns_all(self, sp):
        names = sp.get_node_names(None)
        assert None in names
        assert 's1' in names
        assert 'h1' in names

    def test_list_filter(self, sp):
        names = sp.get_node_names(['s1', 'no_exist'])
        assert names == ['s1']

    def test_regex(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': '{x1}'})
        sp.set_node('s2', grp='stage2')
        names = sp.get_node_names('s\\d')
        assert 's1' in names
        assert 's2' in names
        assert 'h1' not in names

    def test_regex_excludes_none(self, sp):
        names = sp.get_node_names('.*')
        assert None not in names

    def test_invalid_type(self, sp):
        with pytest.raises(ValueError):
            sp.get_node_names(123)


class TestCopy:
    def test_independent_copy(self, sp):
        cp = sp.copy()
        assert set(cp.nodes.keys()) == set(sp.nodes.keys())
        assert set(cp.grps.keys()) == set(sp.grps.keys())
        cp.set_grp('new_grp')
        assert 'new_grp' not in sp.grps

    def test_preserves_output_edges(self, sp):
        sp.set_grp('stage2', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        cp = sp.copy()
        assert 's2' in cp.nodes['s1'].output_edges


class TestCopyNodes:
    def test_includes_dependencies(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['c'])
        assert 'a' in cp.nodes
        assert 'b' in cp.nodes
        assert 'c' in cp.nodes

    def test_excludes_unrelated(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('b', grp='g2')
        cp = p.copy_nodes(['a'])
        assert 'a' in cp.nodes
        assert 'b' not in cp.nodes

    def test_includes_required_groups(self, p):
        p.set_grp('parent')
        p.set_grp('child', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        cp = p.copy_nodes(['n1'])
        assert 'child' in cp.grps
        assert 'parent' in cp.grps

    def test_adjusts_output_edges(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['b'])
        assert 'c' not in cp.nodes['a'].output_edges

    def test_empty_list(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        cp = p.copy_nodes([])
        assert len(cp.nodes) == 1
        assert None in cp.nodes

    def test_datasource_preserved(self, sp):
        cp = sp.copy_nodes(['s1'])
        assert None in cp.nodes


class TestCompareNodes:
    def test_param_differences(self, p):
        p.set_grp('g1', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1, 'b': 2})
        p.set_node('n2', grp='g1', params={'a': 1, 'b': 3})
        result = p.compare_nodes(['n1', 'n2'])
        df = result[DummyHead]
        assert ('params', 'b') in df.columns
        assert ('params', 'a') not in df.columns

    def test_groups_by_processor(self, p):
        p.set_grp('g1', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_grp('g2', processor=AnotherProcessor, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n2', grp='g2', params={'a': 2})
        result = p.compare_nodes(['n1', 'n2'])
        assert DummyHead in result
        assert AnotherProcessor in result

    def test_edge_differences(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('s1', grp='g1')
        p.set_grp('g2', processor=DummyHead, method='predict',
                  edges={'y': '{target}'})
        p.set_node('n1', grp='g2', edges={'X': 's1:({a, b})'})
        p.set_node('n2', grp='g2', edges={'X': 's1:({a, c})'})
        result = p.compare_nodes(['n1', 'n2'])
        df = result[DummyHead]
        x_cols = [c for c in df.columns if c[0] == 'X']
        assert len(x_cols) > 0

    def test_identical_nodes_empty_columns(self, p):
        p.set_grp('g1', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_node('n2', grp='g1', params={'a': 1})
        result = p.compare_nodes(['n1', 'n2'])
        df = result[DummyHead]
        assert len(df.columns) == 0


class TestParamsEqual:
    """params/adapter hold plain data or ref specs only, so comparison is ``==``."""

    def test_identical_ref_specs_equal(self):
        from mllabs._pipeline import _params_equal
        spec = {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '*'}}
        assert _params_equal({'a': dict(spec)}, {'a': dict(spec)})

    def test_different_ref_specs_not_equal(self):
        from mllabs._pipeline import _params_equal
        assert not _params_equal(
            {'a': {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '*'}}},
            {'a': {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '^c_'}}},
        )

    def test_set_grp_diff_skips_on_same_adapter_spec(self, p):
        spec = {'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                '__params__': {'eval_mode': 'valid'}}
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=dict(spec))
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, adapter=dict(spec), exist='diff')
        assert r['result'] == 'skip'

    def test_set_node_diff_skips_on_same_adapter_spec(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', adapter='mllabs.adapter._default.DefaultAdapter')
        r = p.set_node('n1', grp='g1', adapter='mllabs.adapter._default.DefaultAdapter',
                       exist='diff')
        assert r['result'] == 'skip'

    def test_set_grp_diff_skips_on_same_ref_params(self, p):
        spec = {'__ref__': 'mllabs.ColSelector', '__params__': {'dsl_string': '*@categorical'}}
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'cat_features': dict(spec)})
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, params={'cat_features': dict(spec)}, exist='diff')
        assert r['result'] == 'skip'

    def test_set_grp_diff_detects_different_params(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'n': 50})
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, params={'n': 100}, exist='diff')
        assert r['result'] == 'update'


class TestGetParents:
    def test_node_parents(self, p):
        p.set_grp('gp')
        p.set_grp('par', parent='gp')
        p.set_grp('child', parent='par', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        result = p.get_parents('n1')
        assert result == ['child', 'par', 'gp']

    def test_datasource(self, p):
        assert p.get_parents(None) == []

    def test_not_found(self, p):
        assert p.get_parents('no_exist') == []


class TestAdapterAttrsCacheInvalidation:
    """Changing an adapter spec anywhere up the group chain must invalidate the
    cached resolved attrs of every group/node below it."""

    @staticmethod
    def _adapter(mode):
        return {'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                '__params__': {'eval_mode': mode}}

    def test_direct_grp_adapter_change_detected(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self._adapter('valid'))
        r = p.set_grp('g1', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, adapter=self._adapter('both'), exist='diff')
        assert r['result'] == 'update'

    def test_direct_grp_adapter_change_applied(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self._adapter('valid'))
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self._adapter('both'), exist='diff')
        assert p.grps['g1'].adapter['__params__']['eval_mode'] == 'both'

    def test_parent_grp_adapter_change_clears_child_grp_cache(self, p):
        p.set_grp('parent', adapter=self._adapter('valid'))
        p.set_grp('child', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        _ = p.grps['child'].get_attrs(p.grps)
        p.set_grp('parent', adapter=self._adapter('both'), exist='replace')
        assert p.grps['child'].attrs is None

    def test_parent_grp_adapter_change_reflected_in_node(self, p):
        p.set_grp('parent', adapter=self._adapter('valid'))
        p.set_grp('child', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        _ = p.nodes['n1'].get_spec(p.grps)
        p.set_grp('parent', adapter=self._adapter('both'), exist='replace')
        spec = p.nodes['n1'].get_spec(p.grps)
        assert spec.adapter['__params__']['eval_mode'] == 'both'

    def test_grandchild_grp_attrs_cleared(self, p):
        p.set_grp('gp', adapter=self._adapter('valid'))
        p.set_grp('par', parent='gp')
        p.set_grp('child', parent='par', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        _ = p.nodes['n1'].get_spec(p.grps)
        p.set_grp('gp', adapter=self._adapter('both'), exist='replace')
        spec = p.nodes['n1'].get_spec(p.grps)
        assert spec.adapter['__params__']['eval_mode'] == 'both'


SCHEMA_SIMPLE = {'f1': 'numerical', 'f2': 'nominal', 'target': 'binary'}


class TestDataSourceNode:
    def test_get_attrs_works(self, p):
        attrs = p.datasource.get_attrs()
        assert attrs['name'] == 'Data_Source'
        assert 'schema' in attrs
        assert 'targets' in attrs

    def test_datasource_property(self, p):
        assert p.datasource is p.nodes[None]
        assert isinstance(p.datasource, _DataSourceNode)

    def test_set_datasource_basic(self, p):
        result = p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        assert result == 'update'
        assert p.datasource.schema == SCHEMA_SIMPLE
        assert p.datasource.targets == ['target']

    def test_set_datasource_skip_when_unchanged(self, p):
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        old_attrs = p.datasource.get_attrs()
        result = p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        assert result == 'skip'
        assert p.datasource.get_attrs() is old_attrs

    def test_set_datasource_invalid_type(self, p):
        with pytest.raises(ValueError, match='Invalid type'):
            p.set_datasource({'col': 'unknown_type'})

    def test_set_datasource_target_not_in_schema(self, p):
        with pytest.raises(ValueError, match='not in schema'):
            p.set_datasource(SCHEMA_SIMPLE, targets=['missing_col'])

    def test_all_var_types_accepted(self, p):
        schema = {t: t for t in VAR_TYPES}
        result = p.set_datasource(schema)
        assert result == 'update'

    def test_attrs_cache_invalidated_after_set_datasource(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        old_attrs = p.datasource.get_attrs()
        p.set_datasource({**SCHEMA_SIMPLE, 'f3': 'datetime'})
        new_attrs = p.datasource.get_attrs()
        assert new_attrs is not old_attrs
        assert 'f3' in new_attrs['schema']

    def test_copy_preserves_schema_and_targets(self, p):
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        cp = p.copy()
        assert cp.datasource.schema == SCHEMA_SIMPLE
        assert cp.datasource.targets == ['target']

    def test_copy_is_independent(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        cp = p.copy()
        cp.set_datasource({**SCHEMA_SIMPLE, 'extra': 'text'})
        assert 'extra' not in p.datasource.schema


class TestCheckDataCompatibility:
    def test_no_schema_no_check(self, p):
        from mllabs._data_wrapper import wrap
        data = wrap(pd.DataFrame({'a': [1, 2]}))
        p.check_data_compatibility(data)

    def test_missing_column_raises(self, p):
        from mllabs._data_wrapper import wrap
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        data = wrap(pd.DataFrame({'f1': [1, 2], 'f2': ['a', 'b']}))
        with pytest.raises(ValueError, match='target'):
            p.check_data_compatibility(data)

    def test_matching_columns_no_error(self, p):
        from mllabs._data_wrapper import wrap
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        data = wrap(pd.DataFrame({'f1': [1, 2], 'f2': ['a', 'b'], 'target': [0, 1]}))
        p.check_data_compatibility(data)

    @pytest.fixture
    def sample_data(self):
        import numpy as np
        np.random.seed(0)
        n = 20
        return pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.random.randint(0, 2, n),
        })

    def test_experimenter_build_raises_on_missing_column(self, tmp_path, sample_data):
        from mllabs._experimenter import Experimenter
        pl = PipelineBuilder(path=tmp_path / 'pipeline')
        pl.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'missing_col': 'numerical'})
        pl.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                   method='transform', edges={'X': '{f1, f2}'})
        pl.set_node('scaler', grp='scale')
        e = Experimenter(name='e1', data=sample_data, path=tmp_path / 'exp', pipeline=pl.build())
        with pytest.raises(ValueError, match='missing_col'):
            e.build()

    def test_trainer_select_head_raises_on_missing_column(self, tmp_path, sample_data):
        from mllabs._trainer import Trainer
        from mllabs._cache import DataCache
        from mllabs._data_wrapper import wrap
        pl = PipelineBuilder(path=tmp_path / 'pipeline')
        pl.set_datasource({'f1': 'numerical', 'target': 'binary', 'missing_col': 'numerical'})
        pl.set_grp('model', processor='sklearn.tree.DecisionTreeClassifier',
                   method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pl.set_node('dt', grp='model')
        t = Trainer(name='t1', data=wrap(sample_data), path=tmp_path / 'trainer',
                    splitter=None, splitter_params={}, cache=DataCache())
        with pytest.raises(ValueError, match='missing_col'):
            t.set_pipeline(pl.build())


class TestPipelineSQLite:
    def test_init_creates_db(self, tmp_path):
        p = PipelineBuilder(path=tmp_path, name='test')
        assert (tmp_path / 'test.db').exists()

    def test_path_none_no_db(self):
        p = PipelineBuilder()
        assert p._store is None

    def test_copy_no_db(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        cp = p.copy()
        assert cp._store is None

    def test_set_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'scale' in p2.grps
        assert p2.grps['scale'].processor == 'sklearn.preprocessing.StandardScaler'
        assert p2.grps['scale'].method == 'transform'

    def test_set_node_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'scaler' in p2.nodes
        assert p2.nodes['scaler'].grp == 'scale'
        assert 'scaler' in p2.grps['scale'].nodes

    def test_set_datasource_persists(self, tmp_path):
        p = PipelineBuilder(path=tmp_path, name='test')
        schema = {'f1': 'numerical', 'f2': 'nominal', 'target': 'binary'}
        p.set_datasource(schema, targets=['target'])
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert p2.datasource.schema == schema
        assert p2.datasource.targets == ['target']


    def test_remove_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.remove_grp('scale')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'scale' not in p2.grps

    def test_remove_node_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p.remove_node('scaler')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'scaler' not in p2.nodes
        assert 'scaler' not in p2.grps['scale'].nodes

    def test_rename_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p.rename_grp('scale', 'scaler_grp')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'scale' not in p2.grps
        assert 'scaler_grp' in p2.grps
        assert p2.nodes['scaler'].grp == 'scaler_grp'
        assert 'scaler' in p2.grps['scaler_grp'].nodes

    def test_output_edges_reconstructed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('s1', grp='g1')
        p.set_grp('g2', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': 's1:(*)'})
        p.set_node('s2', grp='g2')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 's2' in p2.nodes['s1'].output_edges

    def test_children_reconstructed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('parent')
        p.set_grp('child', parent='parent', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert 'child' in p2.grps['parent'].children

    def test_edges_with_list_var_roundtrip(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{f1, f2}'})
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert p2.grps['g1'].edges == {'X': '{f1, f2}'}

    def test_params_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'}, params={'with_std': False})
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert p2.grps['g1'].params == {'with_std': False}

    def test_set_grp_update_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert p2.grps['g1'].params == {'with_std': False}

    def test_parent_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('parent')
        p.set_grp('child', parent='parent', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        p2 = PipelineBuilder(path=tmp_path, name='test')
        assert p2.grps['child'].parent == 'parent'
        assert p2.nodes['n1'].grp == 'child'


class TestPipelineSync:
    def _make(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = PipelineBuilder(path=tmp_path, name='test')
        p.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        return p

    def test_sync_no_db_raises(self):
        p = PipelineBuilder()
        with pytest.raises(ValueError):
            p.sync()

    def test_sync_no_change(self, tmp_path):
        p = self._make(tmp_path)
        result = p.sync()
        assert result['datasource'] == 'skip'
        assert result['grps'] == {'added': [], 'removed': [], 'updated': []}
        assert result['nodes'] == {'added': [], 'removed': [], 'updated': []}

    def test_sync_datasource_updated(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        schema = {'f1': 'numerical', 'target': 'binary'}
        # B changes datasource
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_datasource(schema, targets=['target'])
        # A syncs
        result = p.sync()
        assert result['datasource'] == 'updated'
        assert p.datasource.schema == schema
        assert p.datasource.targets == ['target']

    def test_sync_grp_added(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds a new group
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_grp('g2', processor='sklearn.preprocessing.StandardScaler', method='transform',
                   edges={'X': 'n1:(*)'})
        # A syncs
        result = p.sync()
        assert 'g2' in result['grps']['added']
        assert 'g2' in p.grps
        assert p.grps['g2'].processor == 'sklearn.preprocessing.StandardScaler'

    def test_sync_grp_removed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B removes the group
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.remove_node('n1')
        p2.remove_grp('g1')
        # A syncs
        result = p.sync()
        assert 'g1' in result['grps']['removed']
        assert 'g1' not in p.grps

    def test_sync_grp_updated(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B updates params
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                   edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        # A syncs
        result = p.sync()
        assert 'g1' in result['grps']['updated']
        assert p.grps['g1'].params == {'with_std': False}

    def test_sync_node_added(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds a node
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_node('n2', grp='g1')
        # A syncs
        result = p.sync()
        assert 'n2' in result['nodes']['added']
        assert 'n2' in p.nodes

    def test_sync_node_removed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B removes the node
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.remove_node('n1')
        # A syncs
        result = p.sync()
        assert 'n1' in result['nodes']['removed']
        assert 'n1' not in p.nodes

    def test_sync_node_updated(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B updates grp
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_grp('g1', processor='sklearn.preprocessing.StandardScaler', method='transform',
                   edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        # A syncs
        result = p.sync()
        assert 'n1' in result['nodes']['updated']
        assert p.nodes['n1'].get_spec(p.grps).params == {'with_std': False}

    def test_sync_rebuilds_output_edges(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds g2/n2 that references n1
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_grp('g2', processor='sklearn.preprocessing.StandardScaler', method='transform',
                   edges={'X': 'n1:(*)'})
        p2.set_node('n2', grp='g2')
        # A syncs
        p.sync()
        assert 'n2' in p.nodes['n1'].output_edges

    def test_sync_rebuilds_children(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds child group
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_grp('g1child', parent='g1', processor='sklearn.preprocessing.StandardScaler',
                   method='transform', edges={'X': '{x1}'})
        # A syncs
        p.sync()
        assert 'g1child' in p.grps['g1'].children

    def test_sync_rebuilds_grp_nodes(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds another node to g1
        p2 = PipelineBuilder(path=tmp_path, name='test')
        p2.set_node('n2', grp='g1')
        # A syncs
        p.sync()
        assert 'n2' in p.grps['g1'].nodes

class TestBuild:
    """PipelineBuilder.build() -> immutable Pipeline structure."""

    def test_returns_pipeline(self, sp):
        from mllabs._pipeline import Pipeline
        assert isinstance(sp.build(), Pipeline)

    def test_carries_pipeline_id_and_new_build_id(self, sp):
        b1, b2 = sp.build(), sp.build()
        assert b1.pipeline_id == b2.pipeline_id == sp.pipeline_id
        assert b1.build_id != b2.build_id

    def test_resolves_group_inheritance(self, p):
        p.set_grp('g1', processor=DummyStage, method='transform',
                  edges={'X': '{a}'}, params={'a': 1, 'b': 2})
        p.set_node('n1', grp='g1', edges={'X': '+ {b}'}, params={'b': 3, 'c': 4})
        node = p.build().nodes['n1']
        assert node.processor == DummyStage
        assert node.method == 'transform'
        assert node.edges['X'] == '{a} + {b}'
        assert node.params == {'a': 1, 'b': 3, 'c': 4}

    def test_group_name_kept_as_label_only(self, sp):
        node = sp.build().get_node('s1')
        assert node.label == 'stage1'

    def test_node_attrs_shape_matches_builder(self, sp):
        built = sp.build().get_node_spec('h1')
        from_builder = sp.get_node_spec('h1')
        assert built == from_builder

    def test_datasource_snapshot(self, p):
        p.set_datasource({'a': 'numerical', 'b': 'binary'}, targets=['b'])
        ds = p.build().datasource
        assert ds.schema == {'a': 'numerical', 'b': 'binary'}
        assert ds.targets == ['b']

    def test_datasource_is_none_key(self, sp):
        built = sp.build()
        assert built.nodes[None] is built.datasource

class TestBuildIsolation:
    """A built Pipeline is a snapshot — later builder edits must not reach it."""

    def test_node_edit_does_not_affect_built(self, sp):
        built = sp.build()
        sp.set_node('s1', grp='stage1', edges={'X': '{x9}'})
        assert built.nodes['s1'].edges['X'] == '{x1}'

    def test_group_edit_does_not_affect_built(self, sp):
        built = sp.build()
        sp.set_grp('stage1', processor=AnotherProcessor,
                   method='transform', edges={'X': '{x1}'})
        assert built.nodes['s1'].processor == DummyStage

    def test_new_node_does_not_appear_in_built(self, sp):
        built = sp.build()
        sp.set_node('s2', grp='stage1')
        assert 's2' not in built.nodes

    def test_removed_node_stays_in_built(self, sp):
        built = sp.build()
        sp.remove_node('s1')
        assert 's1' in built.nodes

    def test_datasource_edit_does_not_affect_built(self, p):
        p.set_datasource({'a': 'numerical'}, targets=['a'])
        built = p.build()
        p.set_datasource({'a': 'numerical', 'b': 'numerical'}, targets=['b'])
        assert built.datasource.schema == {'a': 'numerical'}
        assert built.datasource.targets == ['a']


class TestBuiltPipelineQueries:
    @pytest.fixture
    def chain(self):
        p = PipelineBuilder()
        p.set_datasource({'x1': 'numerical', 'target': 'binary'})
        p.set_grp('g_stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('s1', grp='g_stage')
        p.set_node('s2', grp='g_stage', edges={'X': 's1:(*)'})
        p.set_grp('g_head', processor=DummyHead, method='predict',
                  edges={'X': 's2:(*)', 'y': '{target}'})
        p.set_node('h1', grp='g_head')
        return p.build()

    def test_topo_order(self, chain):
        order = chain.topo_order()
        assert order.index('s1') < order.index('s2') < order.index('h1')
        assert None not in order

    def test_topo_order_returns_copy(self, chain):
        chain.topo_order().append('bogus')
        assert 'bogus' not in chain.topo_order()

    def test_descendants(self, chain):
        assert chain.descendants('s1') == {'s2', 'h1'}
        assert chain.descendants('h1') == set()

    def test_get_node_names_all(self, chain):
        assert set(chain.get_node_names()) == {None, 's1', 's2', 'h1'}

    def test_get_node_names_regex(self, chain):
        assert set(chain.get_node_names('^s')) == {'s1', 's2'}

    def test_get_node_names_list(self, chain):
        assert chain.get_node_names(['s1', 'nope']) == ['s1']

    def test_get_node_names_bad_query(self, chain):
        with pytest.raises(ValueError, match='must be None, list, or str'):
            chain.get_node_names(123)

    def test_get_node(self, chain):
        assert chain.get_node('s2').name == 's2'

    def test_check_data_compatibility_ok(self, chain):
        from mllabs._data_wrapper import wrap
        chain.check_data_compatibility(wrap(pd.DataFrame({'x1': [1], 'target': [0]})))

    def test_check_data_compatibility_missing(self, chain):
        from mllabs._data_wrapper import wrap
        with pytest.raises(ValueError, match='missing columns'):
            chain.check_data_compatibility(wrap(pd.DataFrame({'x1': [1]})))

    def test_subset_pulls_ancestors(self, chain):
        sub = chain.subset(['h1'])
        assert set(sub.get_node_names()) == {None, 's1', 's2', 'h1'}

    def test_subset_drops_unrelated(self, chain):
        sub = chain.subset(['s1'])
        assert set(sub.get_node_names()) == {None, 's1'}
        assert sub.nodes['s1'].output_edges == []

    def test_subset_is_independent(self, chain):
        sub = chain.subset(['h1'])
        sub.nodes['s1'].edges['X'] = '{mutated}'
        assert chain.nodes['s1'].edges['X'] == '{x1}'


class TestDiffFromDataSourceChange:
    """A DataSource schema/targets change should only stale nodes whose own
    edges actually pull different DataSource columns before vs. after —
    not every node in the pipeline."""

    def _pipeline(self, schema, targets=None, x_edge='{x1}'):
        p = PipelineBuilder()
        p.set_datasource(schema, targets=targets)
        p.set_grp('g_stage', processor=DummyStage, method='transform', edges={'X': x_edge})
        p.set_node('s1', grp='g_stage')
        return p.build()

    def test_unrelated_column_added_does_not_stale(self):
        old = self._pipeline({'x1': 'numerical', 'target': 'binary'})
        new = self._pipeline({'x1': 'numerical', 'target': 'binary', 'x2': 'numerical'})
        assert new.diff_from(old) == set()

    def test_referenced_column_removed_stales(self):
        old = self._pipeline({'x1': 'numerical', 'x2': 'numerical', 'target': 'binary'},
                              x_edge='{x1, x2}')
        new = self._pipeline({'x1': 'numerical', 'target': 'binary'}, x_edge='{x1, x2}')
        assert new.diff_from(old) == {'s1'}

    def test_var_type_change_alone_does_not_stale(self):
        old = self._pipeline({'x1': 'numerical', 'target': 'binary'})
        new = self._pipeline({'x1': 'nominal', 'target': 'binary'})
        assert new.diff_from(old) == set()

    def test_targets_only_change_does_not_stale(self):
        old = self._pipeline({'x1': 'numerical', 'x2': 'numerical', 'target': 'binary'},
                              targets=['target'])
        new = self._pipeline({'x1': 'numerical', 'x2': 'numerical', 'target': 'binary'},
                              targets=[])
        assert new.diff_from(old) == set()

    def test_star_segment_stales_on_any_schema_change(self):
        old = self._pipeline({'x1': 'numerical', 'target': 'binary'}, x_edge='*')
        new = self._pipeline({'x1': 'numerical', 'target': 'binary', 'x2': 'numerical'}, x_edge='*')
        assert new.diff_from(old) == {'s1'}

    def test_downstream_of_stale_ds_node_also_stales(self):
        def build(schema):
            p = PipelineBuilder()
            p.set_datasource(schema)
            p.set_grp('g_stage', processor=DummyStage, method='transform', edges={'X': '{x1, x2}'})
            p.set_node('s1', grp='g_stage')
            p.set_grp('g_stage2', processor=DummyStage, method='transform', edges={'X': 's1:(*)'})
            p.set_node('s2', grp='g_stage2')
            return p.build()

        old = build({'x1': 'numerical', 'x2': 'numerical', 'target': 'binary'})
        new = build({'x1': 'numerical', 'target': 'binary'})
        assert new.diff_from(old) == {'s1', 's2'}
