import pandas as pd
import pytest

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mllabs._edge_dsl import (
    parse, eval_expr, validate_edges, referenced_nodes, iter_segments, unparse,
    Star, SetLiteral, Pattern, Namespace, BinOp,
)
from mllabs._pipeline import PipelineBuilder
from mllabs._data_wrapper import wrap


def _data(columns):
    """eval_expr takes a DataWrapper (data.get_columns()), not a bare list."""
    return wrap(pd.DataFrame({c: [] for c in columns}))


class FakeProcessor:
    def __init__(self, obj, X_, name='node'):
        self.obj = obj
        self.X_ = X_
        self.name = name


class TestParse:
    def test_star(self):
        assert parse('*') == Star()

    def test_set_literal(self):
        assert parse('{Num1, Num2, Num3 }') == SetLiteral(['Num1', 'Num2', 'Num3'])

    def test_empty_set_literal(self):
        assert parse('{}') == SetLiteral([])

    def test_plain_pattern(self):
        assert parse('A.*|B.*') == Pattern('A.*|B.*', None)

    def test_pattern_with_selector(self):
        assert parse('A.*@ohe_drop_first') == Pattern('A.*', '@ohe_drop_first')

    def test_pattern_with_empty_selector_call(self):
        node = parse('C.*@subset_poly()')
        assert node == Pattern('C.*', '@subset_poly()')

    def test_bare_selector_no_pattern_prefix(self):
        # '@name' with nothing before it == "all columns, then select".
        assert parse('@ohe_drop_first()') == Pattern('', '@ohe_drop_first()')

    def test_star_with_selector(self):
        # '@selector' is a general postfix — also attaches directly to '*'.
        assert parse('*@ohe_drop_first') == Star('@ohe_drop_first')

    def test_set_literal_with_selector(self):
        # ...and to a set_literal, e.g. to dtype-filter an explicit column list.
        assert parse('{a, b}@int') == SetLiteral(['a', 'b'], '@int')

    def test_selector_with_arguments_raises(self):
        # selectors take no arguments (see col.col_selector) — pattern-based
        # narrowing before '@' replaces the old explicit-args form.
        with pytest.raises(ValueError, match='arguments are not supported'):
            parse("C.*@subset_poly(['v1', 'v2'])")

    def test_namespace(self):
        node = parse('ohe:(A.*)')
        assert node == Namespace('ohe', Pattern('A.*', None))

    def test_difference(self):
        node = parse('* - {N}')
        assert node == BinOp('-', Star(), SetLiteral(['N']))

    def test_union_chain(self):
        node = parse('{a} + {b} + {c}')
        assert node == BinOp('+', BinOp('+', SetLiteral(['a']), SetLiteral(['b'])), SetLiteral(['c']))

    def test_parens_group(self):
        node = parse('(* - {N}) + {a}')
        assert node == BinOp('+', BinOp('-', Star(), SetLiteral(['N'])), SetLiteral(['a']))

    def test_full_example(self):
        node = parse(
            "(* - {N}) + {Num1, Num2, Num3} "
            "+ ohe:(A.*|B.*@ohe_drop_first) "
            "+ poly:(C.*|D.*@subset_poly)"
        )
        assert isinstance(node, BinOp) and node.op == '+'

    def test_trailing_garbage_raises(self):
        with pytest.raises(ValueError):
            parse('{a} )')

    def test_unbalanced_paren_raises(self):
        with pytest.raises(ValueError):
            parse('ohe:(A.*')

    def test_regex_quantifier_plus_is_literal(self):
        # 'Num[a-z]+' has no space before its '+' -> stays part of the pattern,
        # not the DSL union operator.
        assert parse('Num[a-z]+') == Pattern('Num[a-z]+', None)

    def test_regex_quantifier_plus_glued_to_union(self):
        node = parse('Num[a-z]+ + {Other}')
        assert node == BinOp('+', Pattern('Num[a-z]+', None), SetLiteral(['Other']))

    def test_regex_char_class_dash_is_literal(self):
        assert parse('[a-z]+') == Pattern('[a-z]+', None)

    def test_regex_brace_quantifier_is_literal(self):
        assert parse('Num[a-z]{2,3}') == Pattern('Num[a-z]{2,3}', None)

    def test_regex_group_mid_pattern_is_literal(self):
        assert parse('prefix_(a|b)_suffix') == Pattern('prefix_(a|b)_suffix', None)

    def test_operator_needs_surrounding_whitespace(self):
        # No space around '-' at all -> the whole thing is one literal pattern.
        assert parse('a-b') == Pattern('a-b', None)

    def test_slice_start_only(self):
        assert parse('-1:') == slice(-1, None)

    def test_slice_stop_only(self):
        assert parse(':-1') == slice(None, -1)

    def test_slice_start_and_stop(self):
        assert parse('1:2') == slice(1, 2)

    def test_slice_bare_colon(self):
        assert parse(':') == slice(None, None)

    def test_slice_inside_namespace(self):
        assert parse('ohe:(-1:)') == Namespace('ohe', slice(-1, None))

    def test_digit_leading_pattern_without_colon_unaffected(self):
        assert parse('2024.*') == Pattern('2024.*', None)


class TestEvalExpr:
    def test_star(self):
        assert eval_expr(Star(), _data(['a', 'b'])) == ['a', 'b']

    def test_set_literal(self):
        assert eval_expr(SetLiteral(['b', 'a']), _data(['a', 'b', 'c'])) == ['b', 'a']

    def test_set_literal_missing_raises(self):
        with pytest.raises(ValueError, match='Unknown column'):
            eval_expr(SetLiteral(['z']), _data(['a', 'b']))

    def test_pattern(self):
        assert eval_expr(Pattern('^a', None), _data(['aa', 'ab', 'ba'])) == ['aa', 'ab']

    def test_empty_pattern_matches_all(self):
        assert eval_expr(Pattern('', None), _data(['aa', 'ab', 'ba'])) == ['aa', 'ab', 'ba']

    def test_pattern_with_selector(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        columns = ['ohe__color_blue', 'ohe__color_red']
        node = Pattern('.*', '@ohe_drop_first()')
        assert eval_expr(node, _data(columns), processor) == ['ohe__color_red']

    def test_pattern_selector_without_processor_raises(self):
        node = Pattern('.*', '@ohe_drop_first()')
        with pytest.raises(ValueError, match='requires a processor'):
            eval_expr(node, _data(['a']), processor=None)

    def test_star_with_selector(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        columns = ['ohe__color_blue', 'ohe__color_red']
        assert eval_expr(Star('@ohe_drop_first'), _data(columns), processor) == ['ohe__color_red']

    def test_set_literal_with_selector(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        columns = ['ohe__color_blue', 'ohe__color_red']
        node = SetLiteral(columns, '@ohe_drop_first')
        assert eval_expr(node, _data(columns), processor) == ['ohe__color_red']

    def test_union(self):
        node = BinOp('+', SetLiteral(['a', 'b']), SetLiteral(['b', 'c']))
        assert eval_expr(node, _data(['a', 'b', 'c'])) == ['a', 'b', 'c']

    def test_difference(self):
        node = BinOp('-', Star(), SetLiteral(['b']))
        assert eval_expr(node, _data(['a', 'b', 'c'])) == ['a', 'c']

    def test_intersection(self):
        node = BinOp('&', SetLiteral(['a', 'b', 'c']), SetLiteral(['b', 'c', 'd']))
        assert eval_expr(node, _data(['a', 'b', 'c', 'd'])) == ['b', 'c']

    def test_namespace_inside_expr_raises(self):
        with pytest.raises(ValueError, match='Namespace'):
            eval_expr(Namespace('ohe', Star()), _data(['a']))

    def test_slice_last_only(self):
        assert eval_expr(slice(-1, None), _data(['a', 'b', 'c'])) == ['c']

    def test_slice_all_but_last(self):
        assert eval_expr(slice(None, -1), _data(['a', 'b', 'c'])) == ['a', 'b']

    def test_slice_range(self):
        assert eval_expr(slice(1, 2), _data(['a', 'b', 'c'])) == ['b']


@pytest.fixture
def pipeline():
    p = PipelineBuilder()
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'N': 'numerical', 'target': 'binary'})
    p.set_grp('ohe_grp', processor='sklearn.preprocessing.OneHotEncoder',
              method='fit_transform', edges={'X': '{f1}'})
    p.set_node('ohe', grp='ohe_grp')
    return p


class TestValidateEdges:
    """validate_edges is structural-only — it never resolves columns/schema."""

    def test_datasource_forms_are_never_resolved(self, pipeline):
        # None of these touch the schema at all — validate_edges just needs
        # them to parse; actual column resolution happens later, at process time.
        validate_edges('* - {N}', pipeline)
        validate_edges('{f1, f2}', pipeline)
        validate_edges('{does_not_exist}', pipeline)  # not a schema error here
        validate_edges('*', pipeline)

    def test_no_schema_does_not_matter(self):
        p = PipelineBuilder()  # no set_datasource() call
        validate_edges('* - {N}', p)  # still just a structural check

    def test_namespace_segment_ok(self, pipeline):
        validate_edges('ohe:(A.*|B.*@ohe_drop_first)', pipeline)

    def test_unknown_namespace_raises(self, pipeline):
        with pytest.raises(ValueError, match='does not reference an existing node'):
            validate_edges('bogus:(A.*)', pipeline)

    def test_multiple_segments_ok(self, pipeline):
        validate_edges('{f1} + ohe:(A.*)', pipeline)

    def test_cross_namespace_minus_raises(self, pipeline):
        with pytest.raises(ValueError, match='not supported at the top level'):
            validate_edges('ohe:(A.*) - {f1}', pipeline)


class TestReferencedNodesAndUnparse:
    def test_referenced_nodes_datasource(self):
        assert referenced_nodes('{f1, f2}') == {None}

    def test_referenced_nodes_namespace(self):
        assert referenced_nodes('ohe:(A.*)') == {'ohe'}

    def test_referenced_nodes_multiple(self):
        assert referenced_nodes('{f1} + ohe:(A.*) + poly:(B.*)') == {None, 'ohe', 'poly'}

    def test_iter_segments(self):
        segs = list(iter_segments('{f1} + ohe:(A.*)'))
        assert segs == [(None, SetLiteral(['f1'])), ('ohe', Pattern('A.*', None))]

    def test_unparse_roundtrip_pieces(self):
        assert unparse(Star()) == '*'
        assert unparse(SetLiteral(['a', 'b'])) == '{a, b}'
        assert unparse(Pattern('A.*', None)) == 'A.*'
        assert unparse(Pattern('A.*', '@ohe_drop_first()')) == 'A.*@ohe_drop_first()'
        assert unparse(Star('@numeric')) == '*@numeric'
        assert unparse(SetLiteral(['a', 'b'], '@int')) == '{a, b}@int'
        assert unparse(Namespace('ohe', Pattern('A.*', None))) == 'ohe:(A.*)'
        assert unparse(BinOp('-', Star(), SetLiteral(['N']))) == '* - {N}'
        assert unparse(slice(-1, None)) == '-1:'
        assert unparse(slice(None, -1)) == ':-1'
        assert unparse(slice(1, 2)) == '1:2'
        assert unparse(slice(None, None)) == ':'

    def test_unparse_matches_reparse(self):
        text = "(* - {N}) + {Num1, Num2} + ohe:(A.*|B.*@ohe_drop_first)"
        assert parse(unparse(parse(text))) == parse(text)

    def test_slice_unparse_matches_reparse(self):
        assert parse(unparse(slice(-1, None))) == slice(-1, None)
