import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mllabs._pipeline import Pipeline, PipelineGroup, PipelineNode, DataSourceNode, VAR_TYPES


def _dummy_metric(y, p):
    return 0.5


class DummyStage:
    __name__ = 'DummyStage'

class DummyHead:
    __name__ = 'DummyHead'

class AnotherProcessor:
    __name__ = 'AnotherProcessor'


@pytest.fixture
def p():
    return Pipeline()


@pytest.fixture
def sp():
    p = Pipeline()
    p.set_grp('stage1', role='stage', processor=DummyStage, method='transform',
              edges={'X': '{x1}'})
    p.set_node('s1', grp='stage1')
    p.set_grp('head1', role='head', processor=DummyHead, method='predict',
              edges={'X': '{x1}', 'y': '{target}'})
    p.set_node('h1', grp='head1')
    return p


class TestInit:
    def test_datasource_exists(self, p):
        assert None in p.nodes
        assert p.nodes[None].name == 'Data_Source'

    def test_datasource_grp_exists(self, p):
        assert '__datasource__' in p.grps
        assert p.grps['__datasource__'].role == 'datasource'

    def test_datasource_is_datasource_node(self, p):
        assert isinstance(p.nodes[None], DataSourceNode)


class TestSetGrp:
    def test_new_stage(self, p):
        r = p.set_grp('g1', role='stage')
        assert r['result'] == 'new'
        assert 'g1' in p.grps
        assert p.grps['g1'].role == 'stage'

    def test_new_head(self, p):
        p.set_grp('g1', role='head')
        assert p.grps['g1'].role == 'head'

    def test_with_parent(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('child', parent='parent')
        assert 'child' in p.grps['parent'].children
        assert p.grps['child'].parent == 'parent'

    def test_parent_role_inherited(self, p):
        p.set_grp('parent', role='head')
        p.set_grp('child', parent='parent')
        assert p.grps['child'].role == 'head'

    def test_parent_not_found(self, p):
        with pytest.raises(ValueError):
            p.set_grp('g1', role='stage', parent='no_exist')

    def test_invalid_role(self, p):
        with pytest.raises(ValueError):
            p.set_grp('g1', role='invalid')

    def test_role_required(self, p):
        with pytest.raises(ValueError):
            p.set_grp('g1')

    def test_name_conflicts_with_node(self, sp):
        with pytest.raises(ValueError):
            sp.set_grp('s1', role='stage')

    def test_exist_skip(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage)
        r = p.set_grp('g1', role='stage', processor=AnotherProcessor, exist='skip')
        assert r['result'] == 'skip'
        assert p.grps['g1'].processor == DummyStage

    def test_exist_error(self, p):
        p.set_grp('g1', role='stage')
        with pytest.raises(ValueError):
            p.set_grp('g1', role='stage', exist='error')

    def test_exist_replace(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage)
        r = p.set_grp('g1', role='stage', processor=AnotherProcessor, exist='replace')
        assert r['result'] == 'update'
        assert p.grps['g1'].processor == AnotherProcessor

    def test_replace_role_change_fails(self, p):
        p.set_grp('g1', role='stage')
        with pytest.raises(ValueError):
            p.set_grp('g1', role='head', exist='replace')

    def test_replace_parent_change(self, p):
        p.set_grp('p1', role='stage')
        p.set_grp('p2', role='stage')
        p.set_grp('child', role='stage', parent='p1')
        assert 'child' in p.grps['p1'].children
        p.set_grp('child', role='stage', parent='p2', exist='replace')
        assert 'child' not in p.grps['p1'].children
        assert 'child' in p.grps['p2'].children

    def test_replace_affected_nodes_with_group_edges(self, p):
        # Control: group has edges → affected_nodes should include group nodes (works before fix)
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        r = p.set_grp('g1', role='stage', processor=AnotherProcessor, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_replace_affected_nodes_without_group_edges(self, p):
        # Bug: group has no edges, node has own edges → affected_nodes incorrectly empty
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        p.set_node('n1', grp='g1', edges={'X': '{x1}'})
        r = p.set_grp('g1', role='stage', processor=AnotherProcessor, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_replace_affected_nodes_child_grp_when_parent_has_no_edges(self, p):
        # Bug: parent group has no edges, child group has nodes → parent update misses child nodes
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        r = p.set_grp('parent', role='stage', params={'a': 1}, exist='replace')
        assert 'n1' in r['affected_nodes']

    def test_with_all_attrs(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage,
                  edges={'X': '{x1}'}, method='transform',
                  params={'n': 10})
        g = p.grps['g1']
        assert g.processor == DummyStage
        assert g.edges == {'X': '{x1}'}
        assert g.method == 'transform'
        assert g.params == {'n': 10}


class TestSetNode:
    def test_new_node(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        r = p.set_node('n1', grp='g1')
        assert r['result'] == 'new'
        assert 'n1' in p.nodes
        assert 'n1' in p.grps['g1'].nodes

    def test_grp_not_found(self, p):
        with pytest.raises(ValueError):
            p.set_node('n1', grp='no_exist')

    def test_name_conflicts_with_group(self, p):
        p.set_grp('g1', role='stage')
        with pytest.raises(ValueError):
            p.set_node('g1', grp='g1')

    def test_processor_required(self, p):
        p.set_grp('g1', role='stage', method='transform', edges={'X': '{x1}'})
        with pytest.raises(ValueError, match='processor'):
            p.set_node('n1', grp='g1')

    def test_method_required(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, edges={'X': '{x1}'})
        with pytest.raises(ValueError, match='method'):
            p.set_node('n1', grp='g1')

    def test_edges_required(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='edges'):
            p.set_node('n1', grp='g1')

    def test_output_edges_updated(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
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
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_node('n1', grp='g2', exist='replace')
        assert 'n1' not in p.grps['g1'].nodes
        assert 'n1' in p.grps['g2'].nodes
        assert p.nodes['n1'].grp == 'g2'

    def test_replace_returns_affected_nodes(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        r = sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in r['affected_nodes']

    def test_replace_preserves_output_edges(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in sp.nodes['s1'].output_edges

    def test_with_node_level_params(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1})
        p.set_node('n1', grp='g1', params={'b': 2})
        assert p.nodes['n1'].params == {'b': 2}

    def test_tag_default_empty(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        assert p.nodes['n1'].tag == []

    def test_tag_set_on_creation(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', tag=['cv', 'holdout'])
        assert p.nodes['n1'].tag == ['cv', 'holdout']

    def test_tag_change_alone_no_serial_bump(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        r = p.set_node('n1', grp='g1', tag=['new_tag'])
        assert r['result'] == 'skip'
        assert p.nodes['n1'].serial == serial_before

    def test_tag_updated_in_diff_skip_path(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', tag=['old'])
        p.set_node('n1', grp='g1', tag=['new'])
        assert p.nodes['n1'].tag == ['new']


class TestProcessorStringRef:
    def test_set_grp_resolves_string(self, p):
        p.set_grp('g1', role='stage', processor='sklearn.preprocessing.StandardScaler',
                  method='transform', edges={'X': '{x1}'})
        assert p.grps['g1'].processor is StandardScaler

    def test_set_node_resolves_string(self, p):
        p.set_grp('g1', role='stage', processor=StandardScaler,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', processor='sklearn.tree.DecisionTreeClassifier')
        assert p.nodes['n1'].processor is DecisionTreeClassifier

    def test_string_and_class_equivalent_for_diff(self, p):
        p.set_grp('g1', role='stage', processor=StandardScaler,
                  method='transform', edges={'X': '{x1}'})
        r = p.set_grp('g1', role='stage', processor='sklearn.preprocessing.StandardScaler',
                       method='transform', edges={'X': '{x1}'})
        assert r['result'] == 'skip'

    def test_invalid_string_raises(self, p):
        with pytest.raises(Exception):
            p.set_grp('g1', role='stage', processor='not.a.real.module.Thing',
                       method='transform', edges={'X': '{x1}'})


class TestAdapterStringRef:
    def test_set_grp_string_ref_instantiates(self, p):
        from mllabs.adapter import DefaultAdapter
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter='mllabs.adapter._default.DefaultAdapter')
        adapter = p.grps['g1'].adapter
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'both'

    def test_set_grp_ref_dict_with_params(self, p):
        from mllabs.adapter import DefaultAdapter
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                           '__params__': {'eval_mode': 'valid', 'verbose': 0.25}})
        adapter = p.grps['g1'].adapter
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'valid'
        assert adapter.verbose == pytest.approx(0.25)

    def test_set_node_ref_dict_with_params(self, sp):
        from mllabs.adapter import DefaultAdapter
        sp.set_node('h1', grp='head1',
                    adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                             '__params__': {'eval_mode': 'none'}}, exist='replace')
        adapter = sp.nodes['h1'].adapter
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.eval_mode == 'none'

    def test_instance_passthrough(self, p):
        from mllabs.adapter import DefaultAdapter
        inst = DefaultAdapter(eval_mode='valid')
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'}, adapter=inst)
        assert p.grps['g1'].adapter is inst

    def test_ref_dict_and_instance_equivalent_for_diff(self, p):
        from mllabs.adapter import DefaultAdapter
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  adapter=DefaultAdapter(eval_mode='valid'))
        r = p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      adapter={'__ref__': 'mllabs.adapter._default.DefaultAdapter',
                               '__params__': {'eval_mode': 'valid'}})
        assert r['result'] == 'skip'


class TestParamsRefDict:
    def test_set_grp_resolves_colselector_ref(self, p):
        from mllabs import ColSelector
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'cat_features': {'__ref__': 'mllabs.ColSelector',
                                           '__params__': {'dsl_string': '*@categorical'}},
                          'max_depth': 3})
        params = p.grps['g1'].params
        assert isinstance(params['cat_features'], ColSelector)
        assert params['cat_features'].dsl_string == '*@categorical'
        assert params['max_depth'] == 3

    def test_set_node_resolves_colselector_ref(self, p):
        from mllabs import ColSelector
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1',
                   params={'cat_features': {'__ref__': 'mllabs.ColSelector',
                                            '__params__': {'dsl_string': '^cat_'}}})
        sel = p.nodes['n1'].params['cat_features']
        assert isinstance(sel, ColSelector)
        assert sel.dsl_string == '^cat_'

    def test_ref_dict_and_instance_equivalent_for_diff(self, p):
        from mllabs import ColSelector
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'cat_features': ColSelector('*@categorical')})
        r = p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                      method='predict', edges={'X': '{x1}', 'y': '{target}'},
                      params={'cat_features': {'__ref__': 'mllabs.ColSelector',
                                               '__params__': {'dsl_string': '*@categorical'}}})
        assert r['result'] == 'skip'

    def test_plain_string_param_untouched(self, p):
        p.set_grp('g1', role='head', processor=DecisionTreeClassifier,
                  method='predict', edges={'X': '{x1}', 'y': '{target}'},
                  params={'eval_metric': 'AUC', 'criterion': 'gini'})
        params = p.grps['g1'].params
        assert params['eval_metric'] == 'AUC'
        assert params['criterion'] == 'gini'


class TestAddRemoveTag:
    def test_add_tag_basic(self, sp):
        sp.add_tag('h1', 'cv')
        assert 'cv' in sp.nodes['h1'].tag

    def test_add_multiple_tags(self, sp):
        sp.add_tag('h1', 'cv', 'holdout')
        assert sp.nodes['h1'].tag == ['cv', 'holdout']

    def test_add_tag_no_duplicate(self, sp):
        sp.add_tag('h1', 'cv')
        sp.add_tag('h1', 'cv')
        assert sp.nodes['h1'].tag.count('cv') == 1

    def test_add_tag_no_serial_bump(self, sp):
        serial_before = sp.nodes['h1'].serial
        sp.add_tag('h1', 'cv')
        assert sp.nodes['h1'].serial == serial_before

    def test_add_tag_invalidates_attrs_cache(self, sp):
        sp.nodes['h1'].get_attrs(sp.grps)
        sp.add_tag('h1', 'cv')
        assert sp.nodes['h1'].attrs is None

    def test_add_tag_node_not_found(self, sp):
        with pytest.raises(ValueError):
            sp.add_tag('no_exist', 'cv')

    def test_remove_tag_basic(self, sp):
        sp.add_tag('h1', 'cv', 'holdout')
        sp.remove_tag('h1', 'cv')
        assert 'cv' not in sp.nodes['h1'].tag
        assert 'holdout' in sp.nodes['h1'].tag

    def test_remove_multiple_tags(self, sp):
        sp.add_tag('h1', 'cv', 'holdout')
        sp.remove_tag('h1', 'cv', 'holdout')
        assert sp.nodes['h1'].tag == []

    def test_remove_tag_nonexistent_ignored(self, sp):
        sp.remove_tag('h1', 'no_such_tag')
        assert sp.nodes['h1'].tag == []

    def test_remove_tag_no_serial_bump(self, sp):
        sp.add_tag('h1', 'cv')
        serial_before = sp.nodes['h1'].serial
        sp.remove_tag('h1', 'cv')
        assert sp.nodes['h1'].serial == serial_before

    def test_remove_tag_node_not_found(self, sp):
        with pytest.raises(ValueError):
            sp.remove_tag('no_exist', 'cv')

    def test_add_remove_tag_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.add_tag('n1', 'cv', 'holdout')
        p.remove_tag('n1', 'holdout')
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['n1'].tag == ['cv']


class TestGroupHierarchy:
    def test_edges_full_override_by_default(self, p):
        p.set_grp('parent', role='stage', edges={'X': '{a}'})
        p.set_grp('child', role='stage', parent='parent', edges={'X': '{b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{b}'

    def test_edges_plus_continues_from_parent(self, p):
        p.set_grp('parent', role='stage', edges={'X': '{a}'})
        p.set_grp('child', role='stage', parent='parent', edges={'X': '+ {b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{a} + {b}'

    def test_edges_minus_continues_from_parent(self, p):
        p.set_datasource({'a': 'numerical', 'b': 'numerical'})
        p.set_grp('parent', role='stage', edges={'X': '{a, b}'})
        p.set_grp('child', role='stage', parent='parent', edges={'X': '- {b}'})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == '{a, b} - {b}'

    def test_edges_plus_without_parent_value_raises(self, p):
        p.set_grp('parent', role='stage')
        with pytest.raises(ValueError, match='no parent value'):
            p.set_grp('child', role='stage', parent='parent', edges={'X': '+ {b}'})

    def test_params_no_override(self, p):
        p.set_grp('parent', role='stage', params={'a': 1, 'b': 2})
        p.set_grp('child', role='stage', parent='parent', params={'b': 3, 'c': 4})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['params'] == {'a': 1, 'b': 3, 'c': 4}

    def test_processor_inherited(self, p):
        p.set_grp('parent', role='stage', processor=DummyStage)
        p.set_grp('child', role='stage', parent='parent')
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == DummyStage

    def test_processor_overridden(self, p):
        p.set_grp('parent', role='stage', processor=DummyStage)
        p.set_grp('child', role='stage', parent='parent', processor=AnotherProcessor)
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == AnotherProcessor

    def test_method_inherited(self, p):
        p.set_grp('parent', role='stage', method='transform')
        p.set_grp('child', role='stage', parent='parent')
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['method'] == 'transform'

    def test_three_level_hierarchy(self, p):
        p.set_grp('gp', role='stage', processor=DummyStage, edges={'X': '{a}'},
                  params={'x': 1})
        p.set_grp('par', role='stage', parent='gp', method='transform',
                  edges={'X': '+ {b}'}, params={'y': 2})
        p.set_grp('child', role='stage', parent='par',
                  edges={'X': '+ {c}'}, params={'z': 3})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == DummyStage
        assert attrs['method'] == 'transform'
        assert attrs['edges']['X'] == '{a} + {b} + {c}'
        assert attrs['params'] == {'x': 1, 'y': 2, 'z': 3}

    def test_attrs_caching(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage)
        attrs1 = p.grps['g1'].get_attrs(p.grps)
        attrs2 = p.grps['g1'].get_attrs(p.grps)
        assert attrs1 is attrs2

    def test_update_attrs_invalidates_cache(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage)
        p.grps['g1'].get_attrs(p.grps)
        assert p.grps['g1'].attrs is not None
        p.grps['g1'].update_attrs()
        assert p.grps['g1'].attrs is None


class TestNodeAttrs:
    def test_merges_from_group(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1})
        p.set_node('n1', grp='g1')
        attrs = p.get_node_attrs('n1')
        assert attrs['processor'] == DummyStage
        assert attrs['method'] == 'transform'
        assert attrs['params'] == {'a': 1}

    def test_node_overrides_processor(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', processor=AnotherProcessor)
        attrs = p.get_node_attrs('n1')
        assert attrs['processor'] == AnotherProcessor

    def test_node_edges_full_override_by_default(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{a}'})
        p.set_node('n1', grp='g1', edges={'X': '{b}'})
        attrs = p.get_node_attrs('n1')
        assert attrs['edges']['X'] == '{b}'

    def test_node_edges_plus_continues_from_group(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{a}'})
        p.set_node('n1', grp='g1', edges={'X': '+ {b}'})
        attrs = p.get_node_attrs('n1')
        assert attrs['edges']['X'] == '{a} + {b}'

    def test_node_edges_minus_continues_from_group(self, p):
        p.set_datasource({'a': 'numerical', 'b': 'numerical'})
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{a, b}'})
        p.set_node('n1', grp='g1', edges={'X': '- {b}'})
        attrs = p.get_node_attrs('n1')
        assert attrs['edges']['X'] == '{a, b} - {b}'

    def test_node_edges_plus_without_group_value_raises(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='no parent value'):
            p.set_node('n1', grp='g1', edges={'X': '+ {b}'})

    def test_node_params_no_override(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'a': 1, 'b': 2})
        p.set_node('n1', grp='g1', params={'b': 3, 'c': 4})
        attrs = p.get_node_attrs('n1')
        assert attrs['params'] == {'a': 1, 'b': 3, 'c': 4}

    def test_adapter_auto_detect(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        attrs = p.get_node_attrs('n1')
        assert attrs['adapter'] is not None

    def test_node_attrs_caching(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        a1 = p.nodes['n1'].get_attrs(p.grps)
        a2 = p.nodes['n1'].get_attrs(p.grps)
        assert a1 is a2

    def test_attrs_includes_serial(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        attrs = p.get_node_attrs('n1')
        assert 'serial' in attrs
        assert attrs['serial'] == p.nodes['n1'].serial

    def test_attrs_serial_updated_after_bump(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        old_serial = p.get_node_attrs('n1')['serial']
        p._bump_serials(['n1'])
        new_serial = p.get_node_attrs('n1')['serial']
        assert new_serial != old_serial

    def test_attrs_includes_tag(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', tag=['cv'])
        attrs = p.get_node_attrs('n1')
        assert attrs['tag'] == ['cv']

    def test_attrs_tag_is_copy(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', tag=['cv'])
        attrs = p.get_node_attrs('n1')
        attrs['tag'].append('mutated')
        assert p.nodes['n1'].tag == ['cv']


class TestNameValidation:
    @pytest.mark.parametrize('name', [
        'test__name', 'a/b', 'a\\b', 'a\0b', 'a<b', 'a>b',
        'a:b', 'a"b', 'a|b', 'a?b', 'a*b'
    ])
    def test_invalid_names_rejected(self, p, name):
        with pytest.raises(ValueError):
            p.set_grp(name, role='stage')

    def test_valid_names_accepted(self, p):
        for name in ['test', 'test_name', 'test-name', 'test123']:
            p.set_grp(name, role='stage')
            assert name in p.grps


class TestEdgeValidation:
    def test_edge_node_not_found(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='does not reference an existing node'):
            p.set_node('n1', grp='g1', edges={'X': 'no_exist:(*)'})

    def test_edge_must_be_stage(self, sp):
        with pytest.raises(ValueError, match='stage'):
            sp.set_grp('g2', role='head', edges={'X': 'h1:(*)'})


class TestDataSourceEdgeRequiresList:
    def test_set_grp_list_var_spec_accepted(self, p):
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_stage_to_stage_edge_still_allows_non_list(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{f1}'})
        p.set_node('a', grp='g1')
        r = p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                       edges={'X': 'a:(some_pattern)'})
        assert r['result'] == 'new'


class TestDataSourceEdgeSchemaMembership:
    """set_grp/set_node only validate DSL *structure* — column/schema
    membership is resolved lazily, at process time, against real data (see
    _edge_dsl.validate_edges). So an edges string referencing a column that
    doesn't exist in the schema is accepted here regardless."""

    def test_no_schema_no_check(self, p):
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_columns_in_schema_accepted(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                       edges={'X': '{f1, f2}'})
        assert r['result'] == 'new'

    def test_unknown_column_not_rejected_at_definition_time(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                       edges={'X': '{f1, unknown_col}'})
        assert r['result'] == 'new'

    def test_set_node_unknown_column_not_rejected_at_definition_time(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        r = p.set_node('n1', grp='g1', edges={'X': '{unknown_col}'})
        assert r['result'] == 'new'


class TestCycleDetection:
    def test_direct_cycle(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        with pytest.raises(ValueError, match='cycle'):
            sp.set_node('s1', grp='stage1', edges={'X': 's2:(*)'}, exist='replace')

    def test_indirect_cycle(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        with pytest.raises(ValueError, match='cycle'):
            p.set_node('a', grp='g1', edges={'X': 'c:(*)'}, exist='replace')

    def test_no_cycle_chain(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        assert 'c' in p.nodes

    def test_diamond_no_cycle(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('c', grp='g3')
        p.set_grp('g4', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*) + c:(*)'})
        p.set_node('d', grp='g4')
        assert 'd' in p.nodes


class TestRenameGrp:
    def test_basic_rename(self, p):
        p.set_grp('old', role='stage')
        p.rename_grp('old', 'new')
        assert 'old' not in p.grps
        assert 'new' in p.grps
        assert p.grps['new'].name == 'new'

    def test_updates_parent_children(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('old', role='stage', parent='parent')
        p.rename_grp('old', 'new')
        assert 'new' in p.grps['parent'].children
        assert 'old' not in p.grps['parent'].children

    def test_updates_child_parent(self, p):
        p.set_grp('old', role='stage')
        p.set_grp('child', role='stage', parent='old')
        p.rename_grp('old', 'new')
        assert p.grps['child'].parent == 'new'

    def test_updates_node_grp(self, sp):
        sp.rename_grp('stage1', 'renamed')
        assert sp.nodes['s1'].grp == 'renamed'

    def test_invalidates_node_cache(self, sp):
        sp.nodes['s1'].get_attrs(sp.grps)
        sp.rename_grp('stage1', 'renamed')
        assert sp.nodes['s1'].attrs is None

    def test_source_not_found(self, p):
        with pytest.raises(ValueError):
            p.rename_grp('no_exist', 'new')

    def test_target_exists(self, p):
        p.set_grp('a', role='stage')
        p.set_grp('b', role='stage')
        with pytest.raises(ValueError):
            p.rename_grp('a', 'b')


class TestRemoveGrp:
    def test_remove_empty(self, p):
        p.set_grp('g1', role='stage')
        p.remove_grp('g1')
        assert 'g1' not in p.grps

    def test_updates_parent(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent')
        p.remove_grp('child')
        assert 'child' not in p.grps['parent'].children

    def test_not_found(self, p):
        with pytest.raises(ValueError):
            p.remove_grp('no_exist')

    def test_has_children(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent')
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
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
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
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
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
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
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


class TestGetAffectedNodes:
    def test_chain(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        result = p._get_affected_nodes(['a'])
        assert result.index('a') < result.index('b') < result.index('c')

    def test_diamond(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('c', grp='g3')
        p.set_grp('g4', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*) + c:(*)'})
        p.set_node('d', grp='g4')
        result = p._get_affected_nodes(['a'])
        assert result.index('a') < result.index('d')
        assert result.index('b') < result.index('d')
        assert result.index('c') < result.index('d')

    def test_excludes_none(self, sp):
        result = sp._get_affected_nodes(['s1'])
        assert None not in result

    def test_leaf_node(self, sp):
        result = sp._get_affected_nodes(['h1'])
        assert result == ['h1']


class TestCopy:
    def test_independent_copy(self, sp):
        cp = sp.copy()
        assert set(cp.nodes.keys()) == set(sp.nodes.keys())
        assert set(cp.grps.keys()) == set(sp.grps.keys())
        cp.set_grp('new_grp', role='stage')
        assert 'new_grp' not in sp.grps

    def test_preserves_output_edges(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        cp = sp.copy()
        assert 's2' in cp.nodes['s1'].output_edges


class TestCopyStage:
    def test_excludes_head(self, sp):
        cp = sp.copy_stage()
        assert 's1' in cp.nodes
        assert 'h1' not in cp.nodes
        assert 'stage1' in cp.grps
        assert 'head1' not in cp.grps

    def test_adjusts_output_edges(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': 's1:(*)'})
        sp.set_node('s2', grp='stage2')
        cp = sp.copy_stage()
        assert 'h1' not in cp.nodes['s1'].output_edges if cp.nodes['s1'].output_edges else True

    def test_datasource_preserved(self, sp):
        cp = sp.copy_stage()
        assert None in cp.nodes


class TestCopyNodes:
    def test_includes_dependencies(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'b:(*)'})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['c'])
        assert 'a' in cp.nodes
        assert 'b' in cp.nodes
        assert 'c' in cp.nodes

    def test_excludes_unrelated(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('b', grp='g2')
        cp = p.copy_nodes(['a'])
        assert 'a' in cp.nodes
        assert 'b' not in cp.nodes

    def test_includes_required_groups(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        cp = p.copy_nodes(['n1'])
        assert 'child' in cp.grps
        assert 'parent' in cp.grps

    def test_adjusts_output_edges(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'a:(*)'})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['b'])
        assert 'c' not in cp.nodes['a'].output_edges

    def test_empty_list(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
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
        p.set_grp('g1', role='head', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1, 'b': 2})
        p.set_node('n2', grp='g1', params={'a': 1, 'b': 3})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        assert ('params', 'b') in df.columns
        assert ('params', 'a') not in df.columns

    def test_groups_by_processor(self, p):
        p.set_grp('g1', role='head', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_grp('g2', role='head', processor=AnotherProcessor, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n2', grp='g2', params={'a': 2})
        result = p.compare_nodes(['n1', 'n2'])
        assert 'DummyHead' in result
        assert 'AnotherProcessor' in result

    def test_edge_differences(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('s1', grp='g1')
        p.set_grp('g2', role='head', processor=DummyHead, method='predict',
                  edges={'y': '{target}'})
        p.set_node('n1', grp='g2', edges={'X': 's1:({a, b})'})
        p.set_node('n2', grp='g2', edges={'X': 's1:({a, c})'})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        x_cols = [c for c in df.columns if c[0] == 'X']
        assert len(x_cols) > 0

    def test_identical_nodes_empty_columns(self, p):
        p.set_grp('g1', role='head', processor=DummyHead, method='predict',
                  edges={'X': '{x1}', 'y': '{target}'})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_node('n2', grp='g1', params={'a': 1})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        assert len(df.columns) == 0


class TestAdapterEq:
    def test_same_type_same_attrs(self):
        from mllabs.adapter._base import ModelAdapter
        from mllabs._pipeline import _params_equal
        class DummyAdapter(ModelAdapter):
            pass
        assert _params_equal(DummyAdapter(), DummyAdapter())

    def test_same_type_different_attrs(self):
        from mllabs.adapter._base import ModelAdapter
        from mllabs._pipeline import _params_equal
        class DummyAdapter(ModelAdapter):
            pass
        assert not _params_equal(DummyAdapter(eval_mode='none'), DummyAdapter(eval_mode='both'))

    def test_different_types(self):
        from mllabs.adapter._base import ModelAdapter
        from mllabs._pipeline import _params_equal
        class AdapterA(ModelAdapter):
            pass
        class AdapterB(ModelAdapter):
            pass
        assert not _params_equal(AdapterA(), AdapterB())

    def test_set_grp_diff_skips_on_same_adapter_instance(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        from mllabs.adapter._base import ModelAdapter
        class DummyAdapter(ModelAdapter):
            pass
        p.set_grp('g1', role='stage', adapter=DummyAdapter(), exist='replace')
        r = p.set_grp('g1', role='stage', adapter=DummyAdapter(), exist='diff')
        assert r['result'] == 'skip'

    def test_set_node_diff_skips_on_same_adapter_instance(self, p):
        from mllabs.adapter._base import ModelAdapter
        class DummyAdapter(ModelAdapter):
            pass
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', adapter=DummyAdapter())
        r = p.set_node('n1', grp='g1', adapter=DummyAdapter(), exist='diff')
        assert r['result'] == 'skip'

    def test_set_grp_diff_skips_on_non_eq_params(self, p):
        class NoEqObj:
            pass
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'cb': NoEqObj()})
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, params={'cb': NoEqObj()}, exist='diff')
        assert r['result'] == 'skip'

    def test_set_grp_diff_detects_different_params(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, params={'n': 50})
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, params={'n': 100}, exist='diff')
        assert r['result'] == 'update'


class TestGetParents:
    def test_node_parents(self, p):
        p.set_grp('gp', role='stage')
        p.set_grp('par', role='stage', parent='gp')
        p.set_grp('child', role='stage', parent='par', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        result = p.get_parents('n1')
        assert result == ['child', 'par', 'gp']

    def test_datasource(self, p):
        assert p.get_parents(None) == []

    def test_not_found(self, p):
        assert p.get_parents('no_exist') == []


class TestAdapterAttrsCacheInvalidation:
    class DummyAdapter:
        def __init__(self, mode):
            self.mode = mode
        def __eq__(self, other):
            return type(self) is type(other) and self.__dict__ == other.__dict__
        def __hash__(self):
            return id(self)

    def test_direct_grp_adapter_change_detected(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self.DummyAdapter('a'))
        r = p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                      edges={'X': '{x1}'}, adapter=self.DummyAdapter('b'), exist='diff')
        assert r['result'] == 'update'

    def test_direct_grp_adapter_change_applied(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self.DummyAdapter('a'))
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'}, adapter=self.DummyAdapter('b'), exist='diff')
        assert p.grps['g1'].adapter.mode == 'b'

    def test_parent_grp_adapter_change_clears_child_grp_cache(self, p):
        p.set_grp('parent', role='stage', adapter=self.DummyAdapter('a'))
        p.set_grp('child', role='stage', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        _ = p.grps['child'].get_attrs(p.grps)
        p.set_grp('parent', role='stage', adapter=self.DummyAdapter('b'), exist='replace')
        assert p.grps['child'].attrs is None

    def test_parent_grp_adapter_change_reflected_in_node(self, p):
        p.set_grp('parent', role='stage', adapter=self.DummyAdapter('a'))
        p.set_grp('child', role='stage', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        _ = p.nodes['n1'].get_attrs(p.grps)
        p.set_grp('parent', role='stage', adapter=self.DummyAdapter('b'), exist='replace')
        attrs = p.nodes['n1'].get_attrs(p.grps)
        assert attrs['adapter'].mode == 'b'

    def test_grandchild_grp_attrs_cleared(self, p):
        p.set_grp('gp', role='stage', adapter=self.DummyAdapter('a'))
        p.set_grp('par', role='stage', parent='gp')
        p.set_grp('child', role='stage', parent='par', processor=DummyStage,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        _ = p.nodes['n1'].get_attrs(p.grps)
        p.set_grp('gp', role='stage', adapter=self.DummyAdapter('b'), exist='replace')
        attrs = p.nodes['n1'].get_attrs(p.grps)
        assert attrs['adapter'].mode == 'b'


class TestNodeSerial:
    def _is_uuid(self, s):
        import uuid
        try:
            uuid.UUID(s)
            return True
        except (ValueError, AttributeError):
            return False

    def test_node_has_serial_at_creation(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        assert self._is_uuid(p.nodes['n1'].serial)

    def test_datasource_has_serial(self, p):
        assert self._is_uuid(p.nodes[None].serial)

    def test_different_nodes_have_different_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_node('n2', grp='g1')
        assert p.nodes['n1'].serial != p.nodes['n2'].serial

    def test_copy_preserves_serial(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        cp = p.copy()
        assert cp.nodes['n1'].serial == serial_before

    def test_set_node_no_change_preserves_serial(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        p.set_node('n1', grp='g1')  # exist='diff', no change
        assert p.nodes['n1'].serial == serial_before

    def test_set_node_update_changes_serial(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        p.set_node('n1', grp='g1', params={'with_std': False})
        assert p.nodes['n1'].serial != serial_before

    def test_set_node_update_bumps_descendant_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'n1:(*)'})
        p.set_node('n2', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'n2:(*)'})
        p.set_node('n3', grp='g3')
        serial_n2 = p.nodes['n2'].serial
        serial_n3 = p.nodes['n3'].serial
        p.set_node('n1', grp='g1', params={'with_std': False})
        assert p.nodes['n2'].serial != serial_n2
        assert p.nodes['n3'].serial != serial_n3

    def test_set_node_skip_preserves_descendant_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'n1:(*)'})
        p.set_node('n2', grp='g2')
        serial_n1 = p.nodes['n1'].serial
        serial_n2 = p.nodes['n2'].serial
        p.set_node('n1', grp='g1', exist='skip')
        assert p.nodes['n1'].serial == serial_n1
        assert p.nodes['n2'].serial == serial_n2

    def test_set_grp_no_change_preserves_node_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})  # exist='diff', no change
        assert p.nodes['n1'].serial == serial_before

    def test_set_grp_update_bumps_node_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        p.set_grp('g1', role='stage', processor=AnotherProcessor, method='transform',
                  edges={'X': '{x1}'}, exist='replace')
        assert p.nodes['n1'].serial != serial_before

    def test_set_grp_update_bumps_descendant_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': 'n1:(*)'})
        p.set_node('n2', grp='g2')
        serial_n2 = p.nodes['n2'].serial
        p.set_grp('g1', role='stage', processor=AnotherProcessor, method='transform',
                  edges={'X': '{x1}'}, exist='replace')
        assert p.nodes['n2'].serial != serial_n2


SCHEMA_SIMPLE = {'f1': 'numerical', 'f2': 'nominal', 'target': 'binary'}


class TestDataSourceNode:
    def test_get_node_attrs_works(self, p):
        attrs = p.get_node_attrs(None)
        assert attrs['role'] == 'datasource'
        assert attrs['name'] == 'Data_Source'
        assert 'serial' in attrs
        assert 'schema' in attrs
        assert 'targets' in attrs

    def test_datasource_property(self, p):
        assert p.datasource is p.nodes[None]
        assert isinstance(p.datasource, DataSourceNode)

    def test_set_datasource_basic(self, p):
        result = p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        assert result == 'update'
        assert p.datasource.schema == SCHEMA_SIMPLE
        assert p.datasource.targets == ['target']

    def test_set_datasource_skip_when_unchanged(self, p):
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        serial_before = p.datasource.serial
        result = p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        assert result == 'skip'
        assert p.datasource.serial == serial_before

    def test_set_datasource_bumps_serial_on_schema_change(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        serial_before = p.datasource.serial
        p.set_datasource({**SCHEMA_SIMPLE, 'f3': 'ordinal'})
        assert p.datasource.serial != serial_before

    def test_set_datasource_bumps_serial_on_targets_change(self, p):
        p.set_datasource(SCHEMA_SIMPLE, targets=[])
        serial_before = p.datasource.serial
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        assert p.datasource.serial != serial_before

    def test_set_datasource_bumps_downstream_node_serials(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        serial_before = p.nodes['n1'].serial
        p.set_datasource(SCHEMA_SIMPLE)
        assert p.nodes['n1'].serial != serial_before

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

    def test_attrs_serial_reflects_datasource_serial(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        attrs = p.get_node_attrs(None)
        assert attrs['serial'] == p.datasource.serial

    def test_attrs_cache_invalidated_after_set_datasource(self, p):
        p.set_datasource(SCHEMA_SIMPLE)
        old_attrs = p.get_node_attrs(None)
        p.set_datasource({**SCHEMA_SIMPLE, 'f3': 'datetime'})
        new_attrs = p.get_node_attrs(None)
        assert new_attrs is not old_attrs
        assert 'f3' in new_attrs['schema']

    def test_copy_preserves_schema_and_targets(self, p):
        p.set_datasource(SCHEMA_SIMPLE, targets=['target'])
        cp = p.copy()
        assert cp.datasource.schema == SCHEMA_SIMPLE
        assert cp.datasource.targets == ['target']
        assert cp.datasource.serial == p.datasource.serial

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
        pl = Pipeline(path=tmp_path / 'pipeline')
        pl.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'missing_col': 'numerical'})
        pl.set_grp('scale', role='stage', processor=StandardScaler,
                   method='transform', edges={'X': '{f1, f2}'})
        pl.set_node('scaler', grp='scale')
        e = Experimenter(data=sample_data, path=tmp_path / 'exp', pipeline=pl)
        with pytest.raises(ValueError, match='missing_col'):
            e.build()

    def test_experimenter_exp_raises_on_missing_column(self, tmp_path, sample_data):
        from mllabs._experimenter import Experimenter
        pl = Pipeline(path=tmp_path / 'pipeline')
        pl.set_datasource({'f1': 'numerical', 'target': 'binary', 'missing_col': 'numerical'})
        pl.set_grp('model', role='head', processor=DecisionTreeClassifier,
                   method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pl.set_node('dt', grp='model')
        e = Experimenter(data=sample_data, path=tmp_path / 'exp', pipeline=pl)
        with pytest.raises(ValueError, match='missing_col'):
            e.exp()

    def test_trainer_select_head_raises_on_missing_column(self, tmp_path, sample_data):
        from mllabs._trainer import Trainer
        from mllabs._cache import DataCache
        from mllabs._data_wrapper import wrap
        pl = Pipeline(path=tmp_path / 'pipeline')
        pl.set_datasource({'f1': 'numerical', 'target': 'binary', 'missing_col': 'numerical'})
        pl.set_grp('model', role='head', processor=DecisionTreeClassifier,
                   method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pl.set_node('dt', grp='model')
        t = Trainer(name='t1', data=wrap(sample_data), path=tmp_path / 'trainer',
                    splitter=None, splitter_params={}, cache=DataCache())
        with pytest.raises(ValueError, match='missing_col'):
            t.set_pipeline(pl)


class TestPipelineSQLite:
    def test_init_creates_db(self, tmp_path):
        p = Pipeline(path=tmp_path, name='test')
        assert (tmp_path / 'test.db').exists()

    def test_path_none_no_db(self):
        p = Pipeline()
        assert p._store is None

    def test_copy_no_db(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        cp = p.copy()
        assert cp._store is None

    def test_set_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'scale' in p2.grps
        assert p2.grps['scale'].role == 'stage'
        assert p2.grps['scale'].processor is StandardScaler
        assert p2.grps['scale'].method == 'transform'

    def test_set_node_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'scaler' in p2.nodes
        assert p2.nodes['scaler'].grp == 'scale'
        assert 'scaler' in p2.grps['scale'].nodes

    def test_node_serial_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        serial = p.nodes['scaler'].serial
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['scaler'].serial == serial

    def test_set_datasource_persists(self, tmp_path):
        p = Pipeline(path=tmp_path, name='test')
        schema = {'f1': 'numerical', 'f2': 'nominal', 'target': 'binary'}
        p.set_datasource(schema, targets=['target'])
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.datasource.schema == schema
        assert p2.datasource.targets == ['target']

    def test_datasource_serial_persists(self, tmp_path):
        p = Pipeline(path=tmp_path, name='test')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        serial = p.datasource.serial
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.datasource.serial == serial

    def test_remove_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.remove_grp('scale')
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'scale' not in p2.grps

    def test_remove_node_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p.remove_node('scaler')
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'scaler' not in p2.nodes
        assert 'scaler' not in p2.grps['scale'].nodes

    def test_rename_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('scale', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('scaler', grp='scale')
        p.rename_grp('scale', 'scaler_grp')
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'scale' not in p2.grps
        assert 'scaler_grp' in p2.grps
        assert p2.nodes['scaler'].grp == 'scaler_grp'
        assert 'scaler' in p2.grps['scaler_grp'].nodes

    def test_output_edges_reconstructed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('s1', grp='g1')
        p.set_grp('g2', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': 's1:(*)'})
        p.set_node('s2', grp='g2')
        p2 = Pipeline(path=tmp_path, name='test')
        assert 's2' in p2.nodes['s1'].output_edges

    def test_children_reconstructed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent', processor=StandardScaler,
                  method='transform', edges={'X': '{x1}'})
        p2 = Pipeline(path=tmp_path, name='test')
        assert 'child' in p2.grps['parent'].children

    def test_bump_serial_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p._bump_serials(['n1'])
        new_serial = p.nodes['n1'].serial
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['n1'].serial == new_serial

    def test_edges_with_list_var_roundtrip(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{f1, f2}'})
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.grps['g1'].edges == {'X': '{f1, f2}'}

    def test_params_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'}, params={'with_std': False})
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.grps['g1'].params == {'with_std': False}

    def test_node_tag_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1', tag=['cv', 'holdout'])
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['n1'].tag == ['cv', 'holdout']

    def test_node_tag_default_empty_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['n1'].tag == []

    def test_set_grp_update_serial_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        serial = p.nodes['n1'].serial
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.nodes['n1'].serial == serial

    def test_parent_grp_persists(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent', processor=StandardScaler,
                  method='transform', edges={'X': '{x1}'})
        p.set_node('n1', grp='child')
        p2 = Pipeline(path=tmp_path, name='test')
        assert p2.grps['child'].parent == 'parent'
        assert p2.nodes['n1'].grp == 'child'


class TestPipelineSync:
    def _make(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = Pipeline(path=tmp_path, name='test')
        p.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                  edges={'X': '{x1}'})
        p.set_node('n1', grp='g1')
        return p

    def test_sync_no_db_raises(self):
        p = Pipeline()
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
        p2 = Pipeline(path=tmp_path, name='test')
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
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g2', role='stage', processor=StandardScaler, method='transform',
                   edges={'X': 'n1:(*)'})
        # A syncs
        result = p.sync()
        assert 'g2' in result['grps']['added']
        assert 'g2' in p.grps
        assert p.grps['g2'].processor is StandardScaler

    def test_sync_grp_removed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B removes the group
        p2 = Pipeline(path=tmp_path, name='test')
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
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                   edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        # A syncs
        result = p.sync()
        assert 'g1' in result['grps']['updated']
        assert p.grps['g1'].params == {'with_std': False}

    def test_sync_node_added(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds a node
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_node('n2', grp='g1')
        # A syncs
        result = p.sync()
        assert 'n2' in result['nodes']['added']
        assert 'n2' in p.nodes

    def test_sync_node_removed(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B removes the node
        p2 = Pipeline(path=tmp_path, name='test')
        p2.remove_node('n1')
        # A syncs
        result = p.sync()
        assert 'n1' in result['nodes']['removed']
        assert 'n1' not in p.nodes

    def test_sync_node_updated(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        old_serial = p.nodes['n1'].serial
        # B updates grp (bumps n1 serial)
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                   edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        # A syncs
        result = p.sync()
        assert 'n1' in result['nodes']['updated']
        assert p.nodes['n1'].serial != old_serial
        assert p.nodes['n1'].serial == p2.nodes['n1'].serial

    def test_sync_rebuilds_output_edges(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds g2/n2 that references n1
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g2', role='stage', processor=StandardScaler, method='transform',
                   edges={'X': 'n1:(*)'})
        p2.set_node('n2', grp='g2')
        # A syncs
        p.sync()
        assert 'n2' in p.nodes['n1'].output_edges

    def test_sync_rebuilds_children(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds child group
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g1child', role='stage', parent='g1', processor=StandardScaler,
                   method='transform', edges={'X': '{x1}'})
        # A syncs
        p.sync()
        assert 'g1child' in p.grps['g1'].children

    def test_sync_rebuilds_grp_nodes(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        # B adds another node to g1
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_node('n2', grp='g1')
        # A syncs
        p.sync()
        assert 'n2' in p.grps['g1'].nodes

    def test_sync_node_added_with_tag(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_node('n2', grp='g1', tag=['cv'])
        p.sync()
        assert p.nodes['n2'].tag == ['cv']

    def test_sync_node_updated_tag_applied(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        p = self._make(tmp_path)
        p2 = Pipeline(path=tmp_path, name='test')
        p2.set_grp('g1', role='stage', processor=StandardScaler, method='transform',
                   edges={'X': '{x1}'}, params={'with_std': False}, exist='replace')
        p2.nodes['n1'].tag = ['cv']
        p2._db_write(lambda conn: conn.execute(
            "UPDATE nodes SET tag = ? WHERE name = ?", ('["cv"]', 'n1')
        ))
        p.sync()
        assert p.nodes['n1'].tag == ['cv']


class TestExperimenterTags:
    @pytest.fixture
    def sample_data(self):
        import numpy as np
        np.random.seed(0)
        n = 60
        return pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.random.randint(0, 2, n),
        })

    @pytest.fixture
    def pipeline(self, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        p = Pipeline(path=tmp_path / 'pipeline')
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 0})
        p.set_node('dt', grp='model', tag=['cv', 'final'])
        p.set_node('dt2', grp='model', tag=['final'])
        return p

    def _make_exp(self, sample_data, tmp_path, pipeline, tags=None):
        from mllabs._experimenter import Experimenter
        return Experimenter(data=sample_data, path=tmp_path / 'exp1', tags=tags, pipeline=pipeline)

    def test_tags_selects_matching_heads(self, pipeline, sample_data, tmp_path):
        e = self._make_exp(sample_data, tmp_path, pipeline, tags=['cv'])
        e.exp()
        assert e.get_status('dt') == 'built'
        assert e.get_status('dt2') is None

    def test_tags_no_match_nothing_run(self, pipeline, sample_data, tmp_path):
        e = self._make_exp(sample_data, tmp_path, pipeline, tags=['nonexistent'])
        e.exp()
        assert e.get_status('dt') is None
        assert e.get_status('dt2') is None

    def test_no_tags_runs_all_heads(self, pipeline, sample_data, tmp_path):
        e = self._make_exp(sample_data, tmp_path, pipeline, tags=None)
        e.exp()
        assert e.get_status('dt') == 'built'
        assert e.get_status('dt2') == 'built'

    def test_explicit_nodes_overrides_tags(self, pipeline, sample_data, tmp_path):
        e = self._make_exp(sample_data, tmp_path, pipeline, tags=['cv'])
        e.exp(nodes=['dt2'])
        assert e.get_status('dt') is None
        assert e.get_status('dt2') == 'built'


class TestTrainerTags:
    @pytest.fixture
    def sample_data(self):
        import numpy as np
        np.random.seed(0)
        n = 60
        return pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.random.randint(0, 2, n),
        })

    @pytest.fixture
    def pipeline(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        p = Pipeline(path=tmp_path / 'pipeline')
        p.set_grp('scale', role='stage', processor=StandardScaler,
                  method='transform', edges={'X': '{f1, f2}'})
        p.set_node('scaler', grp='scale')
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': 'scaler:(*)', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 0})
        p.set_node('dt', grp='model', tag=['cv', 'final'])
        p.set_node('dt2', grp='model', tag=['final'])
        return p

    def _make_trainer(self, sample_data, tmp_path, tags=None, name='t1'):
        from sklearn.model_selection import KFold
        from mllabs._trainer import Trainer
        from mllabs._cache import DataCache
        from mllabs._data_wrapper import wrap
        return Trainer(name=name, data=wrap(sample_data),
                       path=tmp_path / name, splitter=KFold(n_splits=3),
                       splitter_params={}, cache=DataCache(), tags=tags)

    def test_tags_selects_matching_heads(self, pipeline, sample_data, tmp_path):
        trainer = self._make_trainer(sample_data, tmp_path, tags=['cv'])
        trainer.set_pipeline(pipeline)
        assert 'dt' in trainer.selected_heads
        assert 'dt2' not in trainer.selected_heads

    def test_tags_no_match_empty_selection(self, pipeline, sample_data, tmp_path):
        trainer = self._make_trainer(sample_data, tmp_path, tags=['nonexistent'])
        trainer.set_pipeline(pipeline)
        assert trainer.selected_heads == []

    def test_no_tags_selects_all_heads(self, pipeline, sample_data, tmp_path):
        trainer = self._make_trainer(sample_data, tmp_path, tags=None)
        trainer.set_pipeline(pipeline)
        assert 'dt' in trainer.selected_heads
        assert 'dt2' in trainer.selected_heads

    def test_tags_multiple_match(self, pipeline, sample_data, tmp_path):
        trainer = self._make_trainer(sample_data, tmp_path, tags=['final'])
        trainer.set_pipeline(pipeline)
        assert 'dt' in trainer.selected_heads
        assert 'dt2' in trainer.selected_heads
