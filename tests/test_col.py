import pandas as pd
import pytest

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from mllabs.col import parse_selector_ref, resolve_selector, ohe_drop_first, subset_poly
from mllabs._edge_dsl import parse, eval_expr
from mllabs._data_wrapper import wrap


class FakeProcessor:
    def __init__(self, obj, X_, name='node'):
        self.obj = obj
        self.X_ = X_
        self.name = name


class FakeData:
    def __init__(self, columns):
        self._columns = columns

    def get_columns(self):
        return self._columns

    def select_columns(self, columns):
        return FakeData(list(columns))


class TestParseSelectorRef:
    def test_bare_name(self):
        assert parse_selector_ref('@ohe_drop_first') == 'ohe_drop_first'

    def test_empty_call(self):
        assert parse_selector_ref('@ohe_drop_first()') == 'ohe_drop_first'

    def test_non_empty_call_raises(self):
        with pytest.raises(ValueError, match='Invalid column selector reference'):
            parse_selector_ref("@subset_poly(['v1'])")

    def test_invalid_identifier_raises(self):
        with pytest.raises(ValueError, match='Invalid column selector reference'):
            parse_selector_ref('@1foo')


class TestResolveSelector:
    def test_unknown_name_raises(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'])
        with pytest.raises(ValueError, match='Unknown column selector'):
            resolve_selector('@not_registered', FakeData(['a', 'b']), processor)

    def test_processor_type_mismatch_raises(self):
        processor = FakeProcessor(StandardScaler(), ['f1', 'f2'], name='scaler')
        with pytest.raises(ValueError, match='OneHotEncoder'):
            resolve_selector('@ohe_drop_first', FakeData(['scaler__f1']), processor)

    def test_ohe_drop_first_selects_via_at_syntax(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        columns = ['ohe__color_blue', 'ohe__color_red']
        result = resolve_selector('@ohe_drop_first', FakeData(columns), processor)
        assert result == ['ohe__color_red']

    def test_ohe_drop_first_matches_direct_call(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        columns = ['ohe__color_blue', 'ohe__color_red']
        mask = ohe_drop_first(FakeData(columns), processor)
        direct = [col for col, keep in zip(columns, mask) if keep]
        assert resolve_selector('@ohe_drop_first', FakeData(columns), processor) == direct

    def test_subset_poly_via_at_syntax(self):
        # subset_poly no longer takes a 'vars' argument — the caller (a preceding
        # regex pattern) narrows candidates first, and subset_poly derives which
        # origin var(s) they mention, then snaps to the consistent full expansion.
        processor = FakeProcessor(PolynomialFeatures(degree=2, include_bias=False), ['v1', 'v2'], name='poly')
        candidates = ['poly__v1', 'poly__v1^2']  # e.g. matched via '^poly__v1$|^poly__v1\\^2$'
        result = resolve_selector('@subset_poly', FakeData(candidates), processor)
        assert result == ['poly__v1', 'poly__v1^2']

    def test_subset_poly_drops_inconsistent_candidate(self):
        processor = FakeProcessor(PolynomialFeatures(degree=1, include_bias=False), ['v1', 'v2'], name='poly')
        # 'poly__v1^2' isn't part of the degree=1 expansion for v1, so it's dropped
        # even though it was passed in as a candidate.
        candidates = ['poly__v1', 'poly__v1^2']
        result = resolve_selector('@subset_poly', FakeData(candidates), processor)
        assert result == ['poly__v1']


class TestDtypeSelectors:
    """@numeric / @categorical / @binary / @float / @int / @string — dtype-based,
    no processor required. Uses a real DataWrapper (pandas) since these need
    actual dtype info, not just column names."""

    @staticmethod
    def _df():
        return wrap(pd.DataFrame({
            'f': pd.array([1.0, 2.0], dtype='float64'),
            'i': pd.array([1, 2], dtype='int64'),
            'b': pd.array([True, False], dtype='bool'),
            's': pd.array(['x', 'y'], dtype='object'),
            'c': pd.Categorical(['x', 'y']),
        }))

    def test_float(self):
        assert resolve_selector('@float', self._df(), processor=None) == ['f']

    def test_int(self):
        assert resolve_selector('@int', self._df(), processor=None) == ['i']

    def test_binary(self):
        assert resolve_selector('@binary', self._df(), processor=None) == ['b']

    def test_string(self):
        assert resolve_selector('@string', self._df(), processor=None) == ['s']

    def test_categorical(self):
        assert resolve_selector('@categorical', self._df(), processor=None) == ['c']

    def test_numeric(self):
        # pandas' is_numeric_dtype (used by select_by_dtype('numeric')) counts
        # bool as numeric too — pre-existing behavior, not specific to @numeric.
        assert resolve_selector('@numeric', self._df(), processor=None) == ['f', 'i', 'b']

    def test_via_star_selector(self):
        assert eval_expr(parse('*@numeric'), self._df()) == ['f', 'i', 'b']

    def test_via_set_literal_selector(self):
        assert eval_expr(parse('{f, i, b}@int'), self._df()) == ['i']


def _resolve(data, dsl_string, processor=None):
    """eval_expr(parse(...)) is what every resolve_columns caller does now."""
    return eval_expr(parse(dsl_string), data, processor=processor)


class TestEvalExprAtSyntax:
    def test_bare_selector_dispatches_over_all_columns(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        data = FakeData(['ohe__color_blue', 'ohe__color_red'])
        result = _resolve(data, '@ohe_drop_first', processor)
        assert result == ['ohe__color_red']

    def test_plain_string_is_still_a_regex(self):
        data = FakeData(['ohe__color_blue', 'std__f1'])
        result = _resolve(data, '^ohe')
        assert result == ['ohe__color_blue']

    def test_composes_with_union(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        data = FakeData(['ohe__color_blue', 'ohe__color_red', 'std__f1'])
        result = _resolve(data, '@ohe_drop_first + ^std', processor)
        assert result == ['ohe__color_red', 'std__f1']

    def test_pattern_then_selector(self):
        # pattern filters candidates *before* the selector runs, so a variable
        # excluded elsewhere doesn't affect this variable's own first-seen tracking.
        processor = FakeProcessor(OneHotEncoder(), ['color', 'size'], name='ohe')
        data = FakeData(['ohe__color_blue', 'ohe__color_red', 'ohe__size_S', 'ohe__size_M'])
        result = _resolve(data, '^ohe__color@ohe_drop_first', processor)
        assert result == ['ohe__color_red']

    def test_star_selector_all_columns(self):
        processor = FakeProcessor(OneHotEncoder(), ['color'], name='ohe')
        data = FakeData(['ohe__color_blue', 'ohe__color_red'])
        result = _resolve(data, '*@ohe_drop_first', processor)
        assert result == ['ohe__color_red']
