import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from mllabs import ColSelector
from mllabs._data_wrapper import PandasWrapper, NumpyWrapper

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

from mllabs.processor import (
    CatConverter,
    CatPairCombiner,
    CatOOVFilter,
    FrequencyEncoder,
    TypeConverter,
)

if HAS_POLARS:
    from mllabs.processor import (
        PolarsLoader,
        ExprProcessor,
        PandasConverter,
    )


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2.5,x\n2,3.5,y\n3,4.5,z\n")
    return str(csv_path)


@pytest.fixture
def sample_csv_list(tmp_path):
    csv1 = tmp_path / "sample1.csv"
    csv2 = tmp_path / "sample2.csv"
    csv1.write_text("a,b,c\n1,2.5,x\n2,3.5,y\n")
    csv2.write_text("a,b,c\n3,4.5,z\n4,5.5,w\n")
    return [str(csv1), str(csv2)]


@pytest.fixture
def sample_polars_df():
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    return pl.DataFrame({
        "a": [1, 2, 3],
        "b": [2.5, 3.5, 4.5],
        "c": ["x", "y", "z"],
    })


@pytest.fixture
def sample_pandas_df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2.5, 3.5, 4.5],
        "c": ["x", "y", "z"],
    })


@requires_polars
class TestPolarsLoader:
    def test_fit_single_csv(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        assert hasattr(loader, 'df_type_')
        assert hasattr(loader, 'pl_type_')
        assert loader.fitted_ is True

    def test_fit_csv_list(self, sample_csv_list):
        loader = PolarsLoader()
        loader.fit(sample_csv_list)
        assert loader.fitted_ is True

    def test_transform_single_csv(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        result = loader.transform(sample_csv)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result.columns == ["a", "b", "c"]

    def test_transform_csv_list(self, sample_csv_list):
        loader = PolarsLoader()
        loader.fit(sample_csv_list)
        result = loader.transform(sample_csv_list)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 4

    def test_get_feature_names_out(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        names = loader.get_feature_names_out()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_get_params(self):
        loader = PolarsLoader(predefined_types={"a": pl.Int32}, read_method="read_csv")
        params = loader.get_params()
        assert params["predefined_types"] == {"a": pl.Int32}
        assert params["read_method"] == "read_csv"

    def test_predefined_types(self, sample_csv):
        loader = PolarsLoader(predefined_types={"a": pl.Int32})
        loader.fit(sample_csv)
        result = loader.transform(sample_csv)
        assert result["a"].dtype == pl.Int32


@requires_polars
class TestExprProcessor:
    def test_with_columns_true(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=True)
        proc.fit(sample_polars_df)
        result = proc.transform(sample_polars_df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert "d" in result.columns
        assert result["d"].to_list() == [2, 4, 6]

    def test_with_columns_false(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=False)
        proc.fit(sample_polars_df)
        result = proc.transform(sample_polars_df)
        assert "d" in result.columns
        assert len(result.columns) == 1

    def test_get_feature_names_out(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=True)
        proc.fit(sample_polars_df)
        names = proc.get_feature_names_out()
        assert "a" in names
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_get_params(self):
        expr = {"d": pl.col("a") * 2}
        proc = ExprProcessor(dict_expr=expr, with_columns=True)
        params = proc.get_params()
        assert params["dict_expr"] == expr
        assert params["with_columns"] is True


@requires_polars
class TestPandasConverter:
    def test_basic_conversion(self, sample_polars_df):
        conv = PandasConverter()
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]

    def test_with_index_col(self, sample_polars_df):
        conv = PandasConverter(index_col="a")
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "a"
        assert "a" not in result.columns

    def test_get_feature_names_out(self, sample_polars_df):
        conv = PandasConverter(index_col="a")
        conv.fit(sample_polars_df)
        names = conv.get_feature_names_out()
        assert "a" not in names
        assert "b" in names
        assert "c" in names

    def test_get_params(self):
        conv = PandasConverter(index_col="id")
        params = conv.get_params()
        assert params["index_col"] == "id"


class TestCatConverter:
    def test_pandas_all_columns(self, sample_pandas_df):
        conv = CatConverter()
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        for col in result.columns:
            assert result[col].dtype.name == "category"

    def test_pandas_specific_columns(self, sample_pandas_df):
        conv = CatConverter(columns=["c"])
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        assert result["c"].dtype.name == "category"
        assert result["a"].dtype != "category"

    @requires_polars
    def test_polars_all_columns(self):
        df = pl.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        conv = CatConverter()
        conv.fit(df)
        result = conv.transform(df)
        for col in result.columns:
            assert result[col].dtype == pl.Categorical

    @requires_polars
    def test_polars_specific_columns(self, sample_polars_df):
        conv = CatConverter(columns=["c"])
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert result["c"].dtype == pl.Categorical
        assert result["a"].dtype != pl.Categorical

    def test_numpy(self):
        arr = np.array([[1, "a"], [2, "b"], [3, "c"]], dtype=object)
        conv = CatConverter(columns=[1])
        conv.fit(arr)
        result = conv.transform(arr)
        assert result.dtype == object

    def test_get_feature_names_out(self, sample_pandas_df):
        conv = CatConverter(columns=["a", "c"])
        conv.fit(sample_pandas_df)
        names = conv.get_feature_names_out()
        assert "a" in names
        assert "c" in names

    def test_column_index(self, sample_pandas_df):
        conv = CatConverter(columns=[2])
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        assert result["c"].dtype.name == "category"


class TestCatPairCombiner:
    @pytest.fixture
    def pair_pandas_df(self):
        return pd.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
            "val": [1, 2, 3, 4, 5],
        })

    @pytest.fixture
    def pair_polars_df(self):
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        return pl.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
            "val": [1, 2, 3, 4, 5],
        })

    def test_pandas_basic(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert list(result.columns) == ["cat1__cat2"]
        assert result["cat1__cat2"].tolist() == ["a__x", "a__x", "b__y", "b__y", "c__z"]

    @requires_polars
    def test_polars_basic(self, pair_polars_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(pair_polars_df)
        result = comb.transform(pair_polars_df)
        assert result.columns == ["cat1__cat2"]
        assert result["cat1__cat2"].to_list() == ["a__x", "a__x", "b__y", "b__y", "c__z"]

    def test_custom_sep(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")], sep="-")
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert list(result.columns) == ["cat1-cat2"]
        assert result["cat1-cat2"].tolist()[0] == "a-x"

    def test_custom_new_col_names(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")], new_col_names=["combined"])
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert list(result.columns) == ["combined"]

    def test_missing_none_passthrough(self):
        df = pd.DataFrame({
            "cat1": ["a", None, "b"],
            "cat2": ["x", "y", None],
        })
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(df)
        result = comb.transform(df)
        values = result["cat1__cat2"].tolist()
        assert values[0] == "a__x"
        assert pd.isna(values[1])
        assert pd.isna(values[2])

    def test_numpy_basic(self):
        arr = np.array([["a", "x"], ["a", "x"], ["b", "y"]], dtype=object)
        comb = CatPairCombiner(pairs=[(0, 1)])
        comb.fit(arr)
        result = comb.transform(arr)
        assert result.shape == (3, 1)
        assert result[0, 0] == "a__x"

    def test_multiple_pairs(self, pair_pandas_df):
        df = pair_pandas_df.copy()
        df["cat3"] = ["p", "q", "p", "q", "p"]
        comb = CatPairCombiner(pairs=[("cat1", "cat2"), ("cat1", "cat3")])
        comb.fit(df)
        result = comb.transform(df)
        assert list(result.columns) == ["cat1__cat2", "cat1__cat3"]

    def test_not_fitted_error(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        with pytest.raises(RuntimeError):
            comb.transform(pair_pandas_df)

    def test_pandas_output_dtype_categorical(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert result["cat1__cat2"].dtype.name == "category"

    @requires_polars
    def test_polars_output_dtype_categorical(self, pair_polars_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(pair_polars_df)
        result = comb.transform(pair_polars_df)
        assert result["cat1__cat2"].dtype == pl.Categorical

    def test_get_feature_names_out(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(pair_pandas_df)
        assert comb.get_feature_names_out() == ["cat1__cat2"]

    def test_get_feature_names_out_multiple_pairs(self, pair_pandas_df):
        df = pair_pandas_df.copy()
        df["cat3"] = ["p", "q", "p", "q", "p"]
        comb = CatPairCombiner(pairs=[("cat1", "cat2"), ("cat1", "cat3")])
        comb.fit(df)
        assert comb.get_feature_names_out() == ["cat1__cat2", "cat1__cat3"]

    def test_get_feature_names_out_custom_names(self, pair_pandas_df):
        comb = CatPairCombiner(pairs=[("cat1", "cat2")], new_col_names=["combined"])
        comb.fit(pair_pandas_df)
        assert comb.get_feature_names_out() == ["combined"]

    def test_get_feature_names_out_numpy(self):
        arr = np.array([["a", "x"], ["b", "y"]], dtype=object)
        comb = CatPairCombiner(pairs=[(0, 1)])
        comb.fit(arr)
        assert comb.get_feature_names_out() == ["0__1"]

    def test_triple_pandas(self, pair_pandas_df):
        df = pair_pandas_df.copy()
        df["cat3"] = ["p", "q", "p", "q", "p"]
        comb = CatPairCombiner(pairs=[("cat1", "cat2", "cat3")])
        comb.fit(df)
        result = comb.transform(df)
        assert list(result.columns) == ["cat1__cat2__cat3"]
        assert result["cat1__cat2__cat3"].tolist()[0] == "a__x__p"

    @requires_polars
    def test_triple_polars(self, pair_polars_df):
        df = pair_polars_df.with_columns(pl.Series("cat3", ["p", "q", "p", "q", "p"]))
        comb = CatPairCombiner(pairs=[("cat1", "cat2", "cat3")])
        comb.fit(df)
        result = comb.transform(df)
        assert result.columns == ["cat1__cat2__cat3"]
        assert result["cat1__cat2__cat3"].to_list()[0] == "a__x__p"

    def test_triple_missing(self, pair_pandas_df):
        df = pd.DataFrame({
            "cat1": ["a", None, "b"],
            "cat2": ["x", "y", None],
            "cat3": ["p", "q", "r"],
        })
        comb = CatPairCombiner(pairs=[("cat1", "cat2", "cat3")])
        comb.fit(df)
        result = comb.transform(df)
        values = result["cat1__cat2__cat3"].tolist()
        assert values[0] == "a__x__p"
        assert pd.isna(values[1])
        assert pd.isna(values[2])

    def test_triple_numpy(self):
        arr = np.array([["a", "x", "p"], ["b", "y", "q"]], dtype=object)
        comb = CatPairCombiner(pairs=[(0, 1, 2)])
        comb.fit(arr)
        result = comb.transform(arr)
        assert result.shape == (2, 1)
        assert result[0, 0] == "a__x__p"


class TestCatOOVFilter:
    @pytest.fixture
    def oov_pandas_df(self):
        return pd.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
        })

    @pytest.fixture
    def oov_polars_df(self):
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        return pl.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
        })

    def test_pandas_basic(self, oov_pandas_df):
        filt = CatOOVFilter()
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        assert result["cat1"].tolist() == ["a", "a", "b", "b", "c"]

    def test_pandas_output_dtype_categorical(self, oov_pandas_df):
        filt = CatOOVFilter()
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        assert result["cat1"].dtype.name == "category"
        assert result["cat2"].dtype.name == "category"

    def test_pandas_min_frequency(self, oov_pandas_df):
        # "c", "z" 각각 1회 등장 → min_frequency=1이면 >1이어야 살아남으므로 OOV
        filt = CatOOVFilter(min_frequency=1)
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        values = result["cat1"].tolist()
        assert values[:4] == ["a", "a", "b", "b"]
        assert pd.isna(values[4])

    def test_pandas_missing_value(self, oov_pandas_df):
        filt = CatOOVFilter(min_frequency=1, missing_value="OOV")
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        assert result["cat1"].tolist()[-1] == "OOV"

    def test_pandas_fixed_categories(self, oov_pandas_df):
        filt = CatOOVFilter(min_frequency=1, missing_value="OOV")
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        cats = result["cat1"].cat.categories.tolist()
        assert "OOV" in cats
        assert "a" in cats
        assert "b" in cats
        assert "c" not in cats

    def test_pandas_missing_value_in_categories(self, oov_pandas_df):
        filt = CatOOVFilter(missing_value="OOV")
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        cats = result["cat1"].cat.categories.tolist()
        assert "OOV" in cats

    def test_pandas_unseen_in_transform(self, oov_pandas_df):
        filt = CatOOVFilter(missing_value="OOV")
        filt.fit(oov_pandas_df)
        test_df = pd.DataFrame({"cat1": ["a", "UNSEEN"], "cat2": ["x", "UNSEEN"]})
        result = filt.transform(test_df)
        assert result["cat1"].tolist() == ["a", "OOV"]

    def test_pandas_none_input(self, oov_pandas_df):
        filt = CatOOVFilter(missing_value="OOV")
        filt.fit(oov_pandas_df)
        test_df = pd.DataFrame({"cat1": [None, "a"], "cat2": ["x", None]})
        result = filt.transform(test_df)
        assert result["cat1"].tolist()[0] == "OOV"
        assert result["cat2"].tolist()[1] == "OOV"

    def test_pandas_empty_string_as_missing(self, oov_pandas_df):
        filt = CatOOVFilter(missing_value="OOV", treat_empty_string_as_missing=True)
        filt.fit(oov_pandas_df)
        test_df = pd.DataFrame({"cat1": ["", "a"], "cat2": ["x", ""]})
        result = filt.transform(test_df)
        assert result["cat1"].tolist()[0] == "OOV"
        assert result["cat2"].tolist()[1] == "OOV"

    def test_pandas_specific_columns(self, oov_pandas_df):
        filt = CatOOVFilter(columns=["cat1"])
        filt.fit(oov_pandas_df)
        result = filt.transform(oov_pandas_df)
        assert result["cat1"].dtype.name == "category"
        assert result["cat2"].dtype.name != "category"

    @requires_polars
    def test_polars_basic(self, oov_polars_df):
        filt = CatOOVFilter()
        filt.fit(oov_polars_df)
        result = filt.transform(oov_polars_df)
        assert result["cat1"].to_list() == ["a", "a", "b", "b", "c"]

    @requires_polars
    def test_polars_output_dtype_categorical(self, oov_polars_df):
        filt = CatOOVFilter()
        filt.fit(oov_polars_df)
        result = filt.transform(oov_polars_df)
        assert result["cat1"].dtype == pl.Enum(filt.categories_["cat1"])
        assert result["cat2"].dtype == pl.Enum(filt.categories_["cat2"])

    @requires_polars
    def test_polars_oov_filter(self, oov_polars_df):
        filt = CatOOVFilter(min_frequency=1, missing_value="OOV")
        filt.fit(oov_polars_df)
        result = filt.transform(oov_polars_df)
        assert result["cat1"].to_list()[4] == "OOV"

    def test_numpy_basic(self):
        arr = np.array([["a", "x"], ["a", "x"], ["b", "y"]], dtype=object)
        filt = CatOOVFilter(columns=[0, 1])
        filt.fit(arr)
        result = filt.transform(arr)
        assert result[0, 0] == "a"
        assert result[0, 1] == "x"

    def test_numpy_oov_filter(self):
        arr = np.array([["a", "x"], ["a", "x"], ["b", "y"]], dtype=object)
        # "b" 1번 등장, min_frequency=1 → >1이어야 살아남으므로 OOV
        filt = CatOOVFilter(columns=[0], min_frequency=1, missing_value="OOV")
        filt.fit(arr)
        result = filt.transform(arr)
        assert result[0, 0] == "a"
        assert result[2, 0] == "OOV"

    def test_get_feature_names_out(self, oov_pandas_df):
        filt = CatOOVFilter(columns=["cat1"])
        filt.fit(oov_pandas_df)
        assert filt.get_feature_names_out() == ["cat1"]

    def test_not_fitted_error(self, oov_pandas_df):
        filt = CatOOVFilter()
        with pytest.raises(RuntimeError):
            filt.transform(oov_pandas_df)

    def test_pipeline_with_pair_combiner(self):
        df = pd.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
        })
        comb = CatPairCombiner(pairs=[("cat1", "cat2")])
        comb.fit(df)
        combined = comb.transform(df)
        filt = CatOOVFilter(columns=["cat1__cat2"], min_frequency=1, missing_value="OOV")
        filt.fit(combined)
        result = filt.transform(combined)
        values = result["cat1__cat2"].tolist()
        assert values[0] == "a__x"
        assert values[4] == "OOV"
        assert result["cat1__cat2"].dtype.name == "category"


class TestFrequencyEncoder:
    @pytest.fixture
    def freq_pandas_df(self):
        return pd.DataFrame({'ST depression': [0.0, 0.0, 0.0, 1.2, 1.2, 3.5]})

    @pytest.fixture
    def freq_polars_df(self):
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        return pl.DataFrame({'ST depression': [0.0, 0.0, 0.0, 1.2, 1.2, 3.5]})

    def test_pandas_output_columns(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert list(result.columns) == ['ST depression_freq']

    def test_pandas_normalize(self, freq_pandas_df):
        fe = FrequencyEncoder(normalize=True)
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert abs(result['ST depression_freq'].iloc[0] - 3/6) < 1e-6
        assert abs(result['ST depression_freq'].iloc[3] - 2/6) < 1e-6
        assert abs(result['ST depression_freq'].iloc[5] - 1/6) < 1e-6

    def test_pandas_count(self, freq_pandas_df):
        fe = FrequencyEncoder(normalize=False)
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert result['ST depression_freq'].iloc[0] == 3
        assert result['ST depression_freq'].iloc[3] == 2
        assert result['ST depression_freq'].iloc[5] == 1

    def test_pandas_unseen_value(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        unseen = pd.DataFrame({'ST depression': [99.9]})
        result = fe.transform(unseen)
        assert result['ST depression_freq'].iloc[0] == 0.0

    def test_pandas_multiple_columns(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 3]})
        fe = FrequencyEncoder()
        fe.fit(df)
        result = fe.transform(df)
        assert list(result.columns) == ['a_freq', 'b_freq']

    def test_get_feature_names_out(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        names = fe.get_feature_names_out()
        assert names == ['ST depression_freq']

    def test_get_feature_names_out_with_input(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        names = fe.get_feature_names_out(['ST depression'])
        assert names == ['ST depression_freq']

    @requires_polars
    def test_polars_output_columns(self, freq_polars_df):
        fe = FrequencyEncoder()
        fe.fit(freq_polars_df)
        result = fe.transform(freq_polars_df)
        assert result.columns == ['ST depression_freq']

    @requires_polars
    def test_polars_normalize(self, freq_polars_df):
        fe = FrequencyEncoder(normalize=True)
        fe.fit(freq_polars_df)
        result = fe.transform(freq_polars_df)
        assert abs(result['ST depression_freq'][0] - 3/6) < 1e-6
        assert abs(result['ST depression_freq'][3] - 2/6) < 1e-6

    @requires_polars
    def test_polars_unseen_value(self, freq_polars_df):
        fe = FrequencyEncoder()
        fe.fit(freq_polars_df)
        unseen = pl.DataFrame({'ST depression': [99.9]})
        result = fe.transform(unseen)
        assert result['ST depression_freq'][0] == 0.0

    def test_numpy_output_shape(self):
        arr = np.array([[0.0], [0.0], [0.0], [1.2], [1.2], [3.5]])
        fe = FrequencyEncoder()
        fe.fit(arr)
        result = fe.transform(arr)
        assert result.shape == (6, 1)

    def test_numpy_normalize(self):
        arr = np.array([[0.0], [0.0], [0.0], [1.2], [1.2], [3.5]])
        fe = FrequencyEncoder(normalize=True)
        fe.fit(arr)
        result = fe.transform(arr)
        assert abs(result[0, 0] - 3/6) < 1e-6
        assert abs(result[3, 0] - 2/6) < 1e-6

    def test_numpy_unseen_value(self):
        arr = np.array([[0.0], [1.0], [2.0]])
        fe = FrequencyEncoder()
        fe.fit(arr)
        unseen = np.array([[99.0]])
        result = fe.transform(unseen)
        assert result[0, 0] == 0.0


class TestColSelector:
    @pytest.fixture
    def pandas_df(self):
        return pd.DataFrame({
            'cat_col': pd.Categorical(['a', 'b', 'a']),
            'int_col': pd.array([1, 2, 3], dtype='int32'),
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['x', 'y', 'z'],
            'cat_col2': pd.Categorical(['p', 'q', 'p']),
        })

    @pytest.fixture
    def numpy_int(self):
        return NumpyWrapper(np.array([[1, 2], [3, 4]], dtype=np.int32))

    @pytest.fixture
    def numpy_float(self):
        return NumpyWrapper(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    @pytest.fixture
    def numpy_object(self):
        return NumpyWrapper(np.array([['a', 'b'], ['c', 'd']], dtype=object))

    # --- ColSelector 기본 (dsl_string 기반, eval_expr로 resolve) ---

    def test_col_selector_init(self):
        cs = ColSelector('cat.*@categorical')
        assert cs.dsl_string == 'cat.*@categorical'

    def test_col_selector_defaults(self):
        cs = ColSelector()
        assert cs.dsl_string == '*'

    def _resolve(self, wrapper, dsl_string):
        from mllabs._edge_dsl import parse, eval_expr
        return eval_expr(parse(dsl_string), wrapper)

    # --- PandasWrapper.select_by_dtype / ColSelector resolution ---

    def test_pandas_no_filter(self, pandas_df):
        w = PandasWrapper(pandas_df)
        assert self._resolve(w, ColSelector().dsl_string) == list(pandas_df.columns)

    def test_pandas_category(self, pandas_df):
        w = PandasWrapper(pandas_df)
        assert set(w.select_by_dtype('category')) == {'cat_col', 'cat_col2'}

    def test_pandas_int(self, pandas_df):
        w = PandasWrapper(pandas_df)
        result = w.select_by_dtype('int')
        assert 'int_col' in result
        assert 'float_col' not in result

    def test_pandas_float(self, pandas_df):
        w = PandasWrapper(pandas_df)
        result = w.select_by_dtype('float')
        assert 'float_col' in result
        assert 'int_col' not in result

    def test_pandas_numeric(self, pandas_df):
        w = PandasWrapper(pandas_df)
        result = w.select_by_dtype('numeric')
        assert 'int_col' in result
        assert 'float_col' in result
        assert 'str_col' not in result
        assert 'cat_col' not in result

    def test_pandas_str(self, pandas_df):
        w = PandasWrapper(pandas_df)
        result = w.select_by_dtype('str')
        assert 'str_col' in result
        assert 'int_col' not in result

    def test_pandas_pattern(self, pandas_df):
        w = PandasWrapper(pandas_df)
        assert set(self._resolve(w, 'cat.*')) == {'cat_col', 'cat_col2'}

    def test_pandas_col_type_and_pattern(self, pandas_df):
        w = PandasWrapper(pandas_df)
        assert self._resolve(w, '.*2$@categorical') == ['cat_col2']

    # --- PolarsWrapper.select_by_dtype ---

    @requires_polars
    def test_polars_category(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'cat_col': pl.Series(['a', 'b', 'a'], dtype=pl.Categorical),
            'int_col': pl.Series([1, 2, 3], dtype=pl.Int32),
            'float_col': pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            'str_col': pl.Series(['x', 'y', 'z'], dtype=pl.Utf8),
        })
        w = PolarsWrapper(df)
        assert w.select_by_dtype('category') == ['cat_col']

    @requires_polars
    def test_polars_int(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'int_col': pl.Series([1, 2, 3], dtype=pl.Int32),
            'float_col': pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            'str_col': pl.Series(['x', 'y', 'z'], dtype=pl.Utf8),
        })
        w = PolarsWrapper(df)
        assert w.select_by_dtype('int') == ['int_col']

    @requires_polars
    def test_polars_float(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'int_col': pl.Series([1, 2, 3], dtype=pl.Int32),
            'float_col': pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        })
        w = PolarsWrapper(df)
        assert w.select_by_dtype('float') == ['float_col']

    @requires_polars
    def test_polars_numeric(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'int_col': pl.Series([1, 2, 3], dtype=pl.Int32),
            'float_col': pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            'str_col': pl.Series(['x', 'y', 'z'], dtype=pl.Utf8),
        })
        w = PolarsWrapper(df)
        assert set(w.select_by_dtype('numeric')) == {'int_col', 'float_col'}

    @requires_polars
    def test_polars_str(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'int_col': pl.Series([1, 2, 3], dtype=pl.Int32),
            'str_col': pl.Series(['x', 'y', 'z'], dtype=pl.Utf8),
        })
        w = PolarsWrapper(df)
        assert w.select_by_dtype('str') == ['str_col']

    @requires_polars
    def test_polars_pattern_and_category(self):
        from mllabs._data_wrapper import PolarsWrapper
        df = pl.DataFrame({
            'cat_a': pl.Series(['a', 'b'], dtype=pl.Categorical),
            'cat_b': pl.Series(['p', 'q'], dtype=pl.Categorical),
            'int_col': pl.Series([1, 2], dtype=pl.Int32),
        })
        w = PolarsWrapper(df)
        assert self._resolve(w, 'cat_a@categorical') == ['cat_a']

    # --- NumpyWrapper.select_by_dtype / ColSelector resolution ---

    def test_numpy_int(self, numpy_int):
        assert numpy_int.select_by_dtype('int') == [0, 1]

    def test_numpy_int_no_match_float(self, numpy_float):
        assert numpy_float.select_by_dtype('int') == []

    def test_numpy_float(self, numpy_float):
        assert numpy_float.select_by_dtype('float') == [0, 1]

    def test_numpy_numeric_int(self, numpy_int):
        assert numpy_int.select_by_dtype('numeric') == [0, 1]

    def test_numpy_numeric_float(self, numpy_float):
        assert numpy_float.select_by_dtype('numeric') == [0, 1]

    def test_numpy_str(self, numpy_object):
        assert numpy_object.select_by_dtype('str') == [0, 1]

    def test_numpy_category_empty(self, numpy_object):
        assert numpy_object.select_by_dtype('category') == []

    def test_numpy_no_filter(self, numpy_int):
        assert self._resolve(numpy_int, ColSelector().dsl_string) == [0, 1]

    def test_numpy_pattern(self, numpy_int):
        assert self._resolve(numpy_int, '^0$') == [0]


class _DummyProc:
    def fit_transform(self, X, y=None):
        return X


class TestPipelineDesc:
    @pytest.fixture
    def pipeline(self):
        from mllabs._pipeline import Pipeline
        p = Pipeline()
        p.set_grp('g1', role='stage')
        p.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform')
        return p

    # --- 기본값 ---

    def test_grp_desc_default_none(self, pipeline):
        assert pipeline.grps['g1'].desc is None

    def test_node_desc_default_none(self, pipeline):
        assert pipeline.nodes['n1'].desc is None

    # --- set_grp / set_node ---

    def test_set_grp_desc(self):
        from mllabs._pipeline import Pipeline
        p = Pipeline()
        p.set_grp('g1', role='stage', desc='my group')
        assert p.grps['g1'].desc == 'my group'

    def test_set_node_desc(self, pipeline):
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='my node', exist='replace')
        assert pipeline.nodes['n1'].desc == 'my node'

    # --- get_attrs에 desc 포함 안 됨 (상속 없음) ---

    def test_desc_not_in_get_attrs(self, pipeline):
        pipeline.set_grp('g1', role='stage', desc='group desc', exist='replace')
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='node desc', exist='replace')
        attrs = pipeline.get_node_attrs('n1')
        assert 'desc' not in attrs

    def test_node_desc_not_inherited_from_grp(self, pipeline):
        pipeline.set_grp('g1', role='stage', desc='group desc', exist='replace')
        assert pipeline.nodes['n1'].desc is None

    # --- copy ---

    def test_grp_copy_preserves_desc(self):
        from mllabs._pipeline import Pipeline
        p = Pipeline()
        p.set_grp('g1', role='stage', desc='group desc')
        assert p.grps['g1'].copy().desc == 'group desc'

    def test_node_copy_preserves_desc(self, pipeline):
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='node desc', exist='replace')
        assert pipeline.nodes['n1'].copy().desc == 'node desc'

    # --- desc 변경은 affected_nodes에 영향 없음 ---

    def test_set_grp_desc_only_change_returns_skip(self, pipeline):
        pipeline.set_grp('g1', role='stage', desc='old', exist='replace')
        result = pipeline.set_grp('g1', role='stage', desc='new')
        assert result['result'] == 'skip'
        assert result['affected_nodes'] == []

    def test_set_grp_desc_only_change_updates_desc(self, pipeline):
        pipeline.set_grp('g1', role='stage', desc='old', exist='replace')
        pipeline.set_grp('g1', role='stage', desc='new')
        assert pipeline.grps['g1'].desc == 'new'

    def test_set_node_desc_only_change_returns_skip(self, pipeline):
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='old', exist='replace')
        result = pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='new')
        assert result['result'] == 'skip'
        assert result['affected_nodes'] == []

    def test_set_node_desc_only_change_updates_desc(self, pipeline):
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='old', exist='replace')
        pipeline.set_node('n1', 'g1', processor=_DummyProc, edges={'X': '{x1}'}, method='fit_transform', desc='new')
        assert pipeline.nodes['n1'].desc == 'new'

    # --- 모델 속성 변경 시 affected_nodes 정상 동작 확인 ---

    def test_set_grp_model_change_returns_update(self, pipeline):
        from mllabs.processor import CatConverter
        result = pipeline.set_grp('g1', role='stage', processor=CatConverter)
        assert result['result'] == 'update'


class TestTypeConverter:
    @pytest.fixture
    def df_pd(self):
        return pd.DataFrame({
            'a': pd.Categorical(['x', 'y', 'z']),
            'b': pd.Categorical([1, 2, 3]),
            'c': [1.0, 2.0, 3.0],
        })

    def test_pandas_to_str(self, df_pd):
        tc = TypeConverter(to='str').fit(df_pd)
        out = tc.transform(df_pd)
        assert pd.api.types.is_string_dtype(out['a'])
        assert pd.api.types.is_string_dtype(out['b'])
        assert pd.api.types.is_string_dtype(out['c'])

    def test_pandas_to_float(self, df_pd):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        tc = TypeConverter(to='float').fit(df)
        out = tc.transform(df)
        assert out['a'].dtype == np.float64

    def test_pandas_to_int(self, df_pd):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        tc = TypeConverter(to='int').fit(df)
        out = tc.transform(df)
        assert out['a'].dtype == np.int64

    def test_numpy_to_str(self):
        arr = np.array([[1, 2], [3, 4]])
        tc = TypeConverter(to='str').fit(arr)
        out = tc.transform(arr)
        assert out.dtype.kind == 'U'

    def test_numpy_to_float(self):
        arr = np.array([[1, 2], [3, 4]])
        tc = TypeConverter(to='float').fit(arr)
        out = tc.transform(arr)
        assert out.dtype == np.float64

    def test_get_feature_names_out(self, df_pd):
        tc = TypeConverter(to='str').fit(df_pd)
        assert tc.get_feature_names_out() == ['a', 'b', 'c']

    def test_get_feature_names_out_with_input(self, df_pd):
        tc = TypeConverter(to='str').fit(df_pd)
        assert tc.get_feature_names_out(['x', 'y', 'z']) == ['x', 'y', 'z']

    @requires_polars
    def test_polars_to_str(self):
        df = pl.DataFrame({'a': pl.Series(['x', 'y']).cast(pl.Categorical), 'b': [1, 2]})
        tc = TypeConverter(to='str').fit(df)
        out = tc.transform(df)
        assert out['a'].dtype == pl.Utf8
        assert out['b'].dtype == pl.Utf8

    @requires_polars
    def test_polars_to_int(self):
        df = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        tc = TypeConverter(to='int').fit(df)
        out = tc.transform(df)
        assert out['a'].dtype == pl.Int64

    @requires_polars
    def test_polars_unsupported_to(self):
        df = pl.DataFrame({'a': [1, 2]})
        tc = TypeConverter(to='bool').fit(df)
        with pytest.raises(ValueError):
            tc.transform(df)
