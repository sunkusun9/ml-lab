import pytest
import numpy as np
from mllabs._data_wrapper import (
    DataWrapper, PandasWrapper, NumpyWrapper, wrap, unwrap
)

pd = pytest.importorskip("pandas", reason="pandas not installed")

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

skip_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
skip_cudf = pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")


# ── fixtures ──

@pytest.fixture
def pdf():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

@pytest.fixture
def pdf1():
    return pd.DataFrame({'x': [10, 20, 30]})

@pytest.fixture
def narr():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def narr1():
    return np.array([[10], [20], [30]])

@pytest.fixture
def pldf():
    if not HAS_POLARS:
        return None
    return pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

@pytest.fixture
def pldf1():
    if not HAS_POLARS:
        return None
    return pl.DataFrame({'x': [10, 20, 30]})


# ── wrap / unwrap ──

class TestWrapUnwrap:
    def test_wrap_pandas(self, pdf):
        w = wrap(pdf)
        assert isinstance(w, PandasWrapper)

    def test_wrap_numpy(self, narr):
        w = wrap(narr)
        assert isinstance(w, NumpyWrapper)

    @skip_polars
    def test_wrap_polars(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = wrap(pldf)
        assert isinstance(w, PolarsWrapper)

    def test_wrap_none(self):
        assert wrap(None) is None

    def test_unwrap_none(self):
        assert unwrap(None) is None

    def test_unwrap_native(self, pdf):
        assert unwrap(pdf) is pdf

    def test_unwrap_wrapper(self, pdf):
        w = PandasWrapper(pdf)
        assert unwrap(w) is pdf

    def test_wrap_unsupported(self):
        with pytest.raises(TypeError):
            wrap("not a data object")


# ── PandasWrapper ──

class TestPandasWrapper:
    def test_iloc_list(self, pdf):
        w = PandasWrapper(pdf)
        result = w.iloc([0, 2])
        assert isinstance(result, PandasWrapper)
        assert result.get_shape() == (2, 3)
        assert result.to_native().iloc[0]['a'] == 1

    def test_iloc_slice(self, pdf):
        w = PandasWrapper(pdf)
        result = w.iloc(slice(1, 3))
        assert result.get_shape() == (2, 3)

    def test_select_columns_single(self, pdf):
        w = PandasWrapper(pdf)
        result = w.select_columns('a')
        assert isinstance(result.to_native(), pd.Series)

    def test_select_columns_list(self, pdf):
        w = PandasWrapper(pdf)
        result = w.select_columns(['a', 'c'])
        assert result.get_columns() == ['a', 'c']

    def test_get_columns(self, pdf):
        w = PandasWrapper(pdf)
        assert w.get_columns() == ['a', 'b', 'c']

    def test_get_columns_series(self):
        s = pd.Series([1, 2, 3], name='x')
        w = PandasWrapper(s)
        assert w.get_columns() == ['x']

    def test_get_shape(self, pdf):
        w = PandasWrapper(pdf)
        assert w.get_shape() == (3, 3)

    def test_get_index(self, pdf):
        w = PandasWrapper(pdf)
        idx = w.get_index()
        assert list(idx) == [0, 1, 2]

    def test_concat_axis1(self, pdf, pdf1):
        w1 = PandasWrapper(pdf)
        w2 = PandasWrapper(pdf1)
        result = PandasWrapper.concat([w1, w2], axis=1)
        assert result.get_shape() == (3, 4)
        assert 'x' in result.get_columns()

    def test_concat_axis0(self, pdf):
        w1 = PandasWrapper(pdf)
        w2 = PandasWrapper(pdf.copy())
        result = PandasWrapper.concat([w1, w2], axis=0)
        assert result.get_shape() == (6, 3)

    def test_to_native(self, pdf):
        w = PandasWrapper(pdf)
        assert w.to_native() is pdf

    def test_to_array(self, pdf):
        w = PandasWrapper(pdf)
        arr = w.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)

    def test_getitem_str(self, pdf):
        w = PandasWrapper(pdf)
        result = w['a']
        assert isinstance(result, PandasWrapper)

    def test_getitem_list(self, pdf):
        w = PandasWrapper(pdf)
        result = w[['a', 'b']]
        assert result.get_columns() == ['a', 'b']

    # from_output
    def test_from_output_none(self):
        assert PandasWrapper.from_output(None) is None

    def test_from_output_dataframe(self, pdf):
        result = PandasWrapper.from_output(pdf)
        assert isinstance(result, PandasWrapper)
        assert result.get_columns() == ['a', 'b', 'c']

    def test_from_output_dataframe_rename(self, pdf):
        result = PandasWrapper.from_output(pdf, column_names=['x', 'y', 'z'])
        assert result.get_columns() == ['x', 'y', 'z']

    def test_from_output_ndarray_2d(self, narr):
        result = PandasWrapper.from_output(narr, column_names=['a', 'b', 'c'])
        assert isinstance(result, PandasWrapper)
        assert result.get_columns() == ['a', 'b', 'c']

    def test_from_output_ndarray_1d(self):
        arr = np.array([1, 2, 3])
        result = PandasWrapper.from_output(arr)
        assert result.get_shape() == (3, 1)

    def test_from_output_unsupported(self):
        with pytest.raises(TypeError):
            PandasWrapper.from_output("bad")

    # squeeze
    def test_squeeze_single_col(self, pdf1):
        w = PandasWrapper(pdf1)
        result = w.squeeze()
        assert isinstance(result, PandasWrapper)
        native = result.to_native()
        assert isinstance(native, pd.Series)
        assert native.name == 'x'

    def test_squeeze_multi_col(self, pdf):
        w = PandasWrapper(pdf)
        result = w.squeeze()
        native = result.to_native()
        assert isinstance(native, pd.DataFrame)
        assert native.shape == (3, 3)

    def test_squeeze_series(self):
        s = pd.Series([1, 2, 3], name='y')
        w = PandasWrapper(s)
        result = w.squeeze()
        assert isinstance(result.to_native(), pd.Series)

    # aggregation
    def test_simple(self, pdf):
        w = PandasWrapper(pdf)
        result = DataWrapper.simple(iter([w]))
        assert result is w

    def test_mean(self, pdf):
        w1 = PandasWrapper(pdf.copy())
        w2 = PandasWrapper(pdf.copy() * 3)
        result = DataWrapper.mean(iter([w1, w2]))
        native = unwrap(result)
        assert native['a'].tolist() == [2.0, 4.0, 6.0]

    def test_mode(self):
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 4]})
        df3 = pd.DataFrame({'a': [1, 5, 4]})
        result = PandasWrapper.mode(iter([wrap(df1), wrap(df2), wrap(df3)]))
        native = unwrap(result)
        assert native['a'].iloc[0] == 1


# ── NumpyWrapper ──

class TestNumpyWrapper:
    def test_iloc_list(self, narr):
        w = NumpyWrapper(narr)
        result = w.iloc([0, 2])
        assert isinstance(result, NumpyWrapper)
        assert result.get_shape() == (2, 3)

    def test_iloc_slice(self, narr):
        w = NumpyWrapper(narr)
        result = w.iloc(slice(0, 2))
        assert result.get_shape() == (2, 3)

    def test_select_columns_single(self, narr):
        w = NumpyWrapper(narr, columns=['a', 'b', 'c'])
        result = w.select_columns('b')
        np.testing.assert_array_equal(result.to_native(), np.array([2, 5, 8]))

    def test_select_columns_multi(self, narr):
        w = NumpyWrapper(narr, columns=['a', 'b', 'c'])
        result = w.select_columns(['a', 'c'])
        assert result.get_shape() == (3, 2)
        assert result.get_columns() == ['a', 'c']

    def test_select_columns_not_found(self, narr):
        w = NumpyWrapper(narr, columns=['a', 'b', 'c'])
        with pytest.raises(KeyError):
            w.select_columns('z')

    def test_get_columns_default(self, narr):
        w = NumpyWrapper(narr)
        assert w.get_columns() == [0, 1, 2]

    def test_get_columns_1d(self):
        w = NumpyWrapper(np.array([1, 2, 3]))
        assert w.get_columns() == [0]

    def test_get_shape(self, narr):
        w = NumpyWrapper(narr)
        assert w.get_shape() == (3, 3)

    def test_get_index(self, narr):
        w = NumpyWrapper(narr)
        assert list(w.get_index()) == [0, 1, 2]

    def test_concat_axis1(self, narr):
        w1 = NumpyWrapper(narr, columns=['a', 'b', 'c'])
        w2 = NumpyWrapper(np.array([[10], [20], [30]]), columns=['d'])
        result = NumpyWrapper.concat([w1, w2], axis=1)
        assert result.get_shape() == (3, 4)
        assert result.get_columns() == ['a', 'b', 'c', 'd']

    def test_concat_axis0(self, narr):
        w1 = NumpyWrapper(narr)
        w2 = NumpyWrapper(narr.copy())
        result = NumpyWrapper.concat([w1, w2], axis=0)
        assert result.get_shape() == (6, 3)

    def test_to_native(self, narr):
        w = NumpyWrapper(narr)
        assert w.to_native() is narr

    def test_to_array(self, narr):
        w = NumpyWrapper(narr)
        assert w.to_array() is narr

    # from_output
    def test_from_output_none(self):
        assert NumpyWrapper.from_output(None) is None

    def test_from_output_ndarray(self, narr):
        result = NumpyWrapper.from_output(narr)
        assert isinstance(result, NumpyWrapper)
        assert result.get_columns() == [0, 1, 2]

    def test_from_output_1d(self):
        arr = np.array([1, 2, 3])
        result = NumpyWrapper.from_output(arr)
        assert result.get_columns() == [0]

    def test_from_output_dataframe(self, pdf):
        result = NumpyWrapper.from_output(pdf)
        assert isinstance(result, NumpyWrapper)
        assert result.get_columns() == ['a', 'b', 'c']

    def test_from_output_unsupported(self):
        with pytest.raises(TypeError):
            NumpyWrapper.from_output("bad")

    # squeeze
    def test_squeeze_2d_single_col(self, narr1):
        w = NumpyWrapper(narr1)
        result = w.squeeze()
        assert isinstance(result, NumpyWrapper)
        assert result.to_native().ndim == 1
        np.testing.assert_array_equal(result.to_native(), [10, 20, 30])

    def test_squeeze_2d_multi_col(self, narr):
        w = NumpyWrapper(narr)
        result = w.squeeze()
        assert result.to_native().shape == (3, 3)

    def test_squeeze_1d(self):
        arr = np.array([1, 2, 3])
        w = NumpyWrapper(arr)
        result = w.squeeze()
        assert result.to_native().ndim == 1

    # aggregation
    def test_mean(self, narr):
        w1 = NumpyWrapper(narr.copy())
        w2 = NumpyWrapper(narr.copy() * 3)
        result = DataWrapper.mean(iter([w1, w2]))
        native = unwrap(result)
        np.testing.assert_array_equal(native[0], [2.0, 4.0, 6.0])

    def test_mode(self):
        a1 = np.array([1, 2, 3])
        a2 = np.array([1, 2, 4])
        a3 = np.array([1, 5, 4])
        result = NumpyWrapper.mode(iter([wrap(a1), wrap(a2), wrap(a3)]))
        native = unwrap(result)
        assert native[0] == 1
        assert native[2] == 4


# ── PolarsWrapper ──

@skip_polars
class TestPolarsWrapper:
    def test_iloc_list(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        result = w.iloc([0, 2])
        assert isinstance(result, PolarsWrapper)
        assert result.get_shape() == (2, 3)

    def test_iloc_slice(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        result = w.iloc(slice(1, 3))
        assert result.get_shape() == (2, 3)

    def test_select_columns(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        result = w.select_columns(['a', 'c'])
        assert result.get_columns() == ['a', 'c']

    def test_get_columns(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        assert w.get_columns() == ['a', 'b', 'c']

    def test_get_shape(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        assert w.get_shape() == (3, 3)

    def test_get_index(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        assert list(w.get_index()) == [0, 1, 2]

    def test_concat_axis1(self, pldf, pldf1):
        from mllabs._data_wrapper import PolarsWrapper
        w1 = PolarsWrapper(pldf)
        w2 = PolarsWrapper(pldf1)
        result = PolarsWrapper.concat([w1, w2], axis=1)
        assert result.get_shape() == (3, 4)

    def test_concat_axis0(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w1 = PolarsWrapper(pldf)
        w2 = PolarsWrapper(pldf.clone())
        result = PolarsWrapper.concat([w1, w2], axis=0)
        assert result.get_shape() == (6, 3)

    def test_to_array(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        arr = w.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)

    # from_output
    def test_from_output_none(self):
        from mllabs._data_wrapper import PolarsWrapper
        assert PolarsWrapper.from_output(None) is None

    def test_from_output_polars(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        result = PolarsWrapper.from_output(pldf)
        assert isinstance(result, PolarsWrapper)

    def test_from_output_ndarray_2d(self):
        from mllabs._data_wrapper import PolarsWrapper
        arr = np.array([[1, 2], [3, 4]])
        result = PolarsWrapper.from_output(arr, column_names=['a', 'b'])
        assert result.get_columns() == ['a', 'b']

    def test_from_output_ndarray_1d(self):
        from mllabs._data_wrapper import PolarsWrapper
        arr = np.array([1, 2, 3])
        result = PolarsWrapper.from_output(arr)
        assert result.get_shape() == (3, 1)

    def test_from_output_unsupported(self):
        from mllabs._data_wrapper import PolarsWrapper
        with pytest.raises(TypeError):
            PolarsWrapper.from_output("bad")

    # squeeze
    def test_squeeze_single_col(self, pldf1):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf1)
        result = w.squeeze()
        native = result.to_native()
        assert isinstance(native, pl.Series)
        assert native.to_list() == [10, 20, 30]

    def test_squeeze_multi_col(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w = PolarsWrapper(pldf)
        result = w.squeeze()
        assert isinstance(result.to_native(), pl.DataFrame)

    def test_squeeze_series(self):
        from mllabs._data_wrapper import PolarsWrapper
        s = pl.Series('y', [1, 2, 3])
        w = PolarsWrapper(s)
        result = w.squeeze()
        assert result is w

    # aggregation
    def test_mean(self, pldf):
        from mllabs._data_wrapper import PolarsWrapper
        w1 = PolarsWrapper(pldf.clone())
        w2 = PolarsWrapper(pldf.clone() * 3)
        result = PolarsWrapper.mean(iter([w1, w2]))
        native = unwrap(result)
        assert native['a'].to_list() == [2.0, 4.0, 6.0]

    def test_mode(self):
        from mllabs._data_wrapper import PolarsWrapper
        df1 = pl.DataFrame({'a': [1, 2, 3]})
        df2 = pl.DataFrame({'a': [1, 2, 4]})
        df3 = pl.DataFrame({'a': [1, 5, 4]})
        result = PolarsWrapper.mode(iter([wrap(df1), wrap(df2), wrap(df3)]))
        native = unwrap(result)
        assert native['a'].to_list()[0] == 1


# ── CudfWrapper ──

@skip_cudf
class TestCudfWrapper:
    @pytest.fixture
    def cdf(self):
        return cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

    @pytest.fixture
    def cdf1(self):
        return cudf.DataFrame({'x': [10, 20, 30]})

    def test_iloc(self, cdf):
        from mllabs._data_wrapper import CudfWrapper
        w = CudfWrapper(cdf)
        result = w.iloc([0, 2])
        assert result.get_shape() == (2, 3)

    def test_select_columns(self, cdf):
        from mllabs._data_wrapper import CudfWrapper
        w = CudfWrapper(cdf)
        result = w.select_columns(['a', 'c'])
        assert result.get_columns() == ['a', 'c']

    def test_get_columns(self, cdf):
        from mllabs._data_wrapper import CudfWrapper
        w = CudfWrapper(cdf)
        assert w.get_columns() == ['a', 'b', 'c']

    def test_get_shape(self, cdf):
        from mllabs._data_wrapper import CudfWrapper
        w = CudfWrapper(cdf)
        assert w.get_shape() == (3, 3)

    def test_concat_axis1(self, cdf, cdf1):
        from mllabs._data_wrapper import CudfWrapper
        w1 = CudfWrapper(cdf)
        w2 = CudfWrapper(cdf1)
        result = CudfWrapper.concat([w1, w2], axis=1)
        assert result.get_shape() == (3, 4)

    def test_from_output_none(self):
        from mllabs._data_wrapper import CudfWrapper
        assert CudfWrapper.from_output(None) is None

    def test_squeeze_single_col(self, cdf1):
        from mllabs._data_wrapper import CudfWrapper
        w = CudfWrapper(cdf1)
        result = w.squeeze()
        assert result.to_native().ndim == 1
