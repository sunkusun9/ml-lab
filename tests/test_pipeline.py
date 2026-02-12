import pytest
import pandas as pd
from mllab._pipeline import Pipeline, PipelineGroup, PipelineNode


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
              edges={'X': [(None, None)]})
    p.set_node('s1', grp='stage1')
    p.set_grp('head1', role='head', processor=DummyHead, method='predict',
              edges={'X': [(None, None)], 'y': [(None, 'target')]})
    p.set_node('h1', grp='head1')
    return p


class TestInit:
    def test_datasource_exists(self, p):
        assert None in p.nodes
        assert p.nodes[None].name == 'Data_Source'

    def test_empty_grps(self, p):
        assert p.grps == {}


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

    def test_replace_params_merge(self, p):
        p.set_grp('g1', role='stage', params={'a': 1})
        p.set_grp('g1', role='stage', params={'b': 2}, exist='replace')
        assert p.grps['g1'].params == {'a': 1, 'b': 2}

    def test_with_all_attrs(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage,
                  edges={'X': [(None, None)]}, method='transform',
                  params={'n': 10})
        g = p.grps['g1']
        assert g.processor == DummyStage
        assert g.edges == {'X': [(None, None)]}
        assert g.method == 'transform'
        assert g.params == {'n': 10}


class TestSetNode:
    def test_new_node(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
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
        p.set_grp('g1', role='stage', method='transform', edges={'X': [(None, None)]})
        with pytest.raises(ValueError, match='processor'):
            p.set_node('n1', grp='g1')

    def test_method_required(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, edges={'X': [(None, None)]})
        with pytest.raises(ValueError, match='method'):
            p.set_node('n1', grp='g1')

    def test_edges_required(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform')
        with pytest.raises(ValueError, match='edges'):
            p.set_node('n1', grp='g1')

    def test_output_edges_updated(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': [('s1', None)]})
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
                  edges={'X': [(None, None)]})
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('n1', grp='g1')
        p.set_node('n1', grp='g2', exist='replace')
        assert 'n1' not in p.grps['g1'].nodes
        assert 'n1' in p.grps['g2'].nodes
        assert p.nodes['n1'].grp == 'g2'

    def test_replace_returns_affected_nodes(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': [('s1', None)]})
        sp.set_node('s2', grp='stage2')
        r = sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in r['affected_nodes']

    def test_replace_preserves_output_edges(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': [('s1', None)]})
        sp.set_node('s2', grp='stage2')
        sp.set_node('s1', grp='stage1', exist='replace')
        assert 's2' in sp.nodes['s1'].output_edges

    def test_with_node_level_params(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]}, params={'a': 1})
        p.set_node('n1', grp='g1', params={'b': 2})
        assert p.nodes['n1'].params == {'b': 2}


class TestGroupHierarchy:
    def test_edges_extend(self, p):
        p.set_grp('parent', role='stage', edges={'X': [(None, 'a')]})
        p.set_grp('child', role='stage', parent='parent', edges={'X': [(None, 'b')]})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['edges']['X'] == [(None, 'b'), (None, 'a')]

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
        p.set_grp('gp', role='stage', processor=DummyStage, edges={'X': [(None, 'a')]},
                  params={'x': 1})
        p.set_grp('par', role='stage', parent='gp', method='transform',
                  edges={'X': [(None, 'b')]}, params={'y': 2})
        p.set_grp('child', role='stage', parent='par',
                  edges={'X': [(None, 'c')]}, params={'z': 3})
        attrs = p.grps['child'].get_attrs(p.grps)
        assert attrs['processor'] == DummyStage
        assert attrs['method'] == 'transform'
        assert attrs['edges']['X'] == [(None, 'c'), (None, 'b'), (None, 'a')]
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
                  edges={'X': [(None, None)]}, params={'a': 1})
        p.set_node('n1', grp='g1')
        attrs = p.get_node_attrs('n1')
        assert attrs['processor'] == DummyStage
        assert attrs['method'] == 'transform'
        assert attrs['params'] == {'a': 1}

    def test_node_overrides_processor(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('n1', grp='g1', processor=AnotherProcessor)
        attrs = p.get_node_attrs('n1')
        assert attrs['processor'] == AnotherProcessor

    def test_node_edges_extend_group(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, 'a')]})
        p.set_node('n1', grp='g1', edges={'X': [(None, 'b')]})
        attrs = p.get_node_attrs('n1')
        assert attrs['edges']['X'] == [(None, 'b'), (None, 'a')]

    def test_node_params_no_override(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]}, params={'a': 1, 'b': 2})
        p.set_node('n1', grp='g1', params={'b': 3, 'c': 4})
        attrs = p.get_node_attrs('n1')
        assert attrs['params'] == {'a': 1, 'b': 3, 'c': 4}

    def test_adapter_auto_detect(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('n1', grp='g1')
        attrs = p.get_node_attrs('n1')
        assert attrs['adapter'] is not None

    def test_node_attrs_caching(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('n1', grp='g1')
        a1 = p.nodes['n1'].get_attrs(p.grps)
        a2 = p.nodes['n1'].get_attrs(p.grps)
        assert a1 is a2


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
        with pytest.raises(ValueError, match='not found'):
            p.set_node('n1', grp='g1', edges={'X': [('no_exist', None)]})

    def test_edge_must_be_stage(self, sp):
        with pytest.raises(ValueError, match='stage'):
            sp.set_grp('g2', role='head', edges={'X': [('h1', None)]})


class TestCycleDetection:
    def test_direct_cycle(self, sp):
        sp.set_grp('stage2', role='stage', processor=DummyStage, method='transform',
                   edges={'X': [('s1', None)]})
        sp.set_node('s2', grp='stage2')
        with pytest.raises(ValueError, match='cycle'):
            sp.set_node('s1', grp='stage1', edges={'X': [('s2', None)]}, exist='replace')

    def test_indirect_cycle(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None)]})
        p.set_node('c', grp='g3')
        with pytest.raises(ValueError, match='cycle'):
            p.set_node('a', grp='g1', edges={'X': [('c', None)]}, exist='replace')

    def test_no_cycle_chain(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None)]})
        p.set_node('c', grp='g3')
        assert 'c' in p.nodes

    def test_diamond_no_cycle(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('c', grp='g3')
        p.set_grp('g4', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None), ('c', None)]})
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
                   edges={'X': [('s1', None)]})
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
                   edges={'X': [('s1', None)]})
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
                   edges={'X': [(None, None)]})
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
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None)]})
        p.set_node('c', grp='g3')
        result = p._get_affected_nodes(['a'])
        assert result.index('a') < result.index('b') < result.index('c')

    def test_diamond(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('c', grp='g3')
        p.set_grp('g4', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None), ('c', None)]})
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
                   edges={'X': [('s1', None)]})
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
                   edges={'X': [('s1', None)]})
        sp.set_node('s2', grp='stage2')
        cp = sp.copy_stage()
        assert 'h1' not in cp.nodes['s1'].output_edges if cp.nodes['s1'].output_edges else True

    def test_datasource_preserved(self, sp):
        cp = sp.copy_stage()
        assert None in cp.nodes


class TestCopyNodes:
    def test_includes_dependencies(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('b', None)]})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['c'])
        assert 'a' in cp.nodes
        assert 'b' in cp.nodes
        assert 'c' in cp.nodes

    def test_excludes_unrelated(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('b', grp='g2')
        cp = p.copy_nodes(['a'])
        assert 'a' in cp.nodes
        assert 'b' not in cp.nodes

    def test_includes_required_groups(self, p):
        p.set_grp('parent', role='stage')
        p.set_grp('child', role='stage', parent='parent', processor=DummyStage,
                  method='transform', edges={'X': [(None, None)]})
        p.set_node('n1', grp='child')
        cp = p.copy_nodes(['n1'])
        assert 'child' in cp.grps
        assert 'parent' in cp.grps

    def test_adjusts_output_edges(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('a', grp='g1')
        p.set_grp('g2', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('b', grp='g2')
        p.set_grp('g3', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [('a', None)]})
        p.set_node('c', grp='g3')
        cp = p.copy_nodes(['b'])
        assert 'c' not in cp.nodes['a'].output_edges

    def test_empty_list(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
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
                  edges={'X': [(None, None)], 'y': [(None, 'target')]})
        p.set_node('n1', grp='g1', params={'a': 1, 'b': 2})
        p.set_node('n2', grp='g1', params={'a': 1, 'b': 3})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        assert ('params', 'b') in df.columns
        assert ('params', 'a') not in df.columns

    def test_groups_by_processor(self, p):
        p.set_grp('g1', role='head', processor=DummyHead, method='predict',
                  edges={'X': [(None, None)], 'y': [(None, 'target')]})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_grp('g2', role='head', processor=AnotherProcessor, method='predict',
                  edges={'X': [(None, None)], 'y': [(None, 'target')]})
        p.set_node('n2', grp='g2', params={'a': 2})
        result = p.compare_nodes(['n1', 'n2'])
        assert 'DummyHead' in result
        assert 'AnotherProcessor' in result

    def test_edge_differences(self, p):
        p.set_grp('g1', role='stage', processor=DummyStage, method='transform',
                  edges={'X': [(None, None)]})
        p.set_node('s1', grp='g1')
        p.set_grp('g2', role='head', processor=DummyHead, method='predict',
                  edges={'y': [(None, 'target')]})
        p.set_node('n1', grp='g2', edges={'X': [('s1', ['a', 'b'])]})
        p.set_node('n2', grp='g2', edges={'X': [('s1', ['a', 'c'])]})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        x_cols = [c for c in df.columns if c[0] == 'X']
        assert len(x_cols) > 0

    def test_identical_nodes_empty_columns(self, p):
        p.set_grp('g1', role='head', processor=DummyHead, method='predict',
                  edges={'X': [(None, None)], 'y': [(None, 'target')]})
        p.set_node('n1', grp='g1', params={'a': 1})
        p.set_node('n2', grp='g1', params={'a': 1})
        result = p.compare_nodes(['n1', 'n2'])
        df = result['DummyHead']
        assert len(df.columns) == 0


class TestGetParents:
    def test_node_parents(self, p):
        p.set_grp('gp', role='stage')
        p.set_grp('par', role='stage', parent='gp')
        p.set_grp('child', role='stage', parent='par', processor=DummyStage,
                  method='transform', edges={'X': [(None, None)]})
        p.set_node('n1', grp='child')
        result = p.get_parents('n1')
        assert result == ['child', 'par', 'gp']

    def test_datasource(self, p):
        assert p.get_parents(None) == []

    def test_not_found(self, p):
        assert p.get_parents('no_exist') == []
