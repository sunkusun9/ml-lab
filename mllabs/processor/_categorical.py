import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import polars as pl
except Exception:
    pl = None

from ._util import detect_kind, is_nan


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self.freq_ = {}

        if kind == "pandas_df":
            self.columns_ = list(X.columns)
            for col in self.columns_:
                self.freq_[col] = X[col].value_counts(normalize=self.normalize).to_dict()
        elif kind == "polars_df":
            self.columns_ = list(X.columns)
            for col in self.columns_:
                vc = X.get_column(col).value_counts(normalize=self.normalize)
                prop_col = "proportion" if self.normalize else "count"
                self.freq_[col] = dict(zip(vc.get_column(col).to_list(), vc.get_column(prop_col).to_list()))
        elif kind == "numpy":
            arr = X if X.ndim > 1 else X.reshape(-1, 1)
            self.columns_ = list(range(arr.shape[1]))
            for i in self.columns_:
                unique, counts = np.unique(arr[:, i], return_counts=True)
                vals = (counts / counts.sum()).tolist() if self.normalize else counts.tolist()
                self.freq_[i] = dict(zip(unique.tolist(), vals))

        return self

    def transform(self, X):
        kind = detect_kind(X)
        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _transform_pandas(self, X):
        return pd.DataFrame(
            {str(col) + '_freq': X[col].map(self.freq_[col]).fillna(0) for col in self.columns_},
            index=X.index
        )

    def _transform_polars(self, X):
        exprs = []
        for col in self.columns_:
            freq_map = self.freq_[col]
            exprs.append(
                pl.col(col).map_elements(
                    lambda x, fm=freq_map: fm.get(x, 0.0),
                    return_dtype=pl.Float64
                ).alias(str(col) + '_freq')
            )
        return X.select(exprs)

    def _transform_numpy(self, X):
        arr = X if X.ndim > 1 else X.reshape(-1, 1)
        result = np.zeros((arr.shape[0], len(self.columns_)), dtype=float)
        for j in range(len(self.columns_)):
            freq_map = self.freq_[self.columns_[j]]
            for i in range(arr.shape[0]):
                result[i, j] = freq_map.get(arr[i, j], 0.0)
        return result

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else [str(c) for c in self.columns_]
        return [col + '_freq' for col in cols]

    def set_output(self, transform=None):
        pass


class CatPairCombiner(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        pairs,
        *,
        sep="__",
        treat_empty_string_as_missing=True,
        new_col_names=None,
    ):
        self.pairs = list(pairs)
        self.sep = sep
        self.treat_empty_string_as_missing = treat_empty_string_as_missing
        self.new_col_names = list(new_col_names) if new_col_names is not None else None

        self._resolved_pairs_ = []
        self._new_names_ = []

    # ----------------- helpers -----------------

    def _is_missing(self, v):
        if v is None:
            return True
        if is_nan(v):
            return True
        if self.treat_empty_string_as_missing and isinstance(v, str) and v == "":
            return True
        return False

    def _combine(self, *vals):
        if any(self._is_missing(v) for v in vals):
            return None
        return self.sep.join(str(v) for v in vals)

    def _default_new_name(self, *labels):
        return self.sep.join(str(l) for l in labels)

    def _resolve_pair(self, X, p):
        if isinstance(X, np.ndarray):
            if not all(isinstance(e, int) for e in p):
                raise ValueError("For numpy input, pair elements should be integer column indices.")
            return [(e, str(e)) for e in p]

        cols = list(X.columns)
        result = []
        for e in p:
            if isinstance(e, int):
                name = cols[e]
            else:
                name = e
            result.append((name, str(name)))
        return result

    def _make_new_names(self):
        if self.new_col_names is not None:
            if len(self.new_col_names) != len(self._resolved_pairs_):
                raise ValueError("new_col_names length must match pairs length.")
            return list(self.new_col_names)
        names = []
        for group in self._resolved_pairs_:
            names.append(self._default_new_name(*[label for _, label in group]))
        return names

    # ----------------- sklearn API -----------------

    def fit(self, X, y=None):
        self._resolved_pairs_ = []
        for p in self.pairs:
            self._resolved_pairs_.append(self._resolve_pair(X, p))
        self._new_names_ = self._make_new_names()
        return self

    def transform(self, X):
        if not self._resolved_pairs_:
            raise RuntimeError("This transformer is not fitted yet. Call fit() first.")

        kind = detect_kind(X)

        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)

        raise TypeError(f"Unsupported input type: {type(X)}")

    # ----------------- transforms -----------------

    def _transform_pandas(self, X):
        new_cols = {}
        for group, new_name in zip(self._resolved_pairs_, self._new_names_):
            cols = [X[name] for name, _ in group]
            if self.treat_empty_string_as_missing:
                missing = [c.isna() | (c.astype(str) == '') for c in cols]
            else:
                missing = [c.isna() for c in cols]
            any_missing = missing[0]
            for m in missing[1:]:
                any_missing = any_missing | m
            combined = cols[0].astype(str)
            for c in cols[1:]:
                combined = combined + self.sep + c.astype(str)
            new_cols[new_name] = pd.Categorical(combined.where(~any_missing))
        return pd.DataFrame(new_cols, index=X.index)

    def _transform_polars(self, X):
        exprs = []
        for group, new_name in zip(self._resolved_pairs_, self._new_names_):
            names = [name for name, _ in group]
            utf8_cols = [pl.col(n).cast(pl.Utf8) for n in names]
            if self.treat_empty_string_as_missing:
                missing_exprs = [pl.col(n).is_null() | (pl.col(n).cast(pl.Utf8) == "") for n in names]
            else:
                missing_exprs = [pl.col(n).is_null() for n in names]
            any_missing = missing_exprs[0]
            for m in missing_exprs[1:]:
                any_missing = any_missing | m
            combined = utf8_cols[0]
            for u in utf8_cols[1:]:
                combined = combined + pl.lit(self.sep) + u
            exprs.append(
                pl.when(any_missing)
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(combined)
                .cast(pl.Categorical)
                .alias(new_name)
            )
        return X.select(exprs)

    def get_feature_names_out(self, input_features=None):
        return list(self._new_names_)

    def set_output(self, transform=None):
        pass

    def _transform_numpy(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        vcombine = np.vectorize(self._combine, otypes=[object])
        new_cols = [
            vcombine(*[arr[:, idx] for idx, _ in group]).reshape(-1, 1)
            for group, _ in zip(self._resolved_pairs_, self._new_names_)
        ]
        return np.concatenate(new_cols, axis=1)


class CatConverter(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, safe_cast=True):
        self.columns = columns
        self.safe_cast = safe_cast

    def _resolve_columns(self, X, kind):
        if self.columns is None:
            if kind == "numpy":
                return list(range(X.shape[1]))
            return list(X.columns)
        cols = []
        for c in self.columns:
            if isinstance(c, int) and kind != "numpy":
                cols.append(X.columns[c])
            else:
                cols.append(c)
        return cols

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self.columns_ = self._resolve_columns(X, kind)
        self.kind_ = kind
        return self

    def transform(self, X):
        kind = detect_kind(X)
        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _transform_pandas(self, X):
        X_out = X.copy()
        for col in self.columns_:
            if col in X_out.columns:
                s = X_out[col]
                if self.safe_cast and pd.api.types.is_float_dtype(s):
                    # xgboost rejects category names of float dtype (only str/int are
                    # supported) -- route through str so the category names are str.
                    s = s.where(s.isna(), s.astype(str))
                X_out[col] = s.astype('category')
        return X_out

    def _transform_polars(self, X):
        exprs = []
        for col in self.columns_:
            if col in X.columns:
                expr = pl.col(col)
                if self.safe_cast and X.schema[col].is_numeric():
                    # polars refuses to cast numeric dtypes to Categorical directly
                    # (int and float alike) -- Utf8 is the only accepted source.
                    expr = expr.cast(pl.Utf8)
                exprs.append(expr.cast(pl.Categorical))
        if exprs:
            return X.with_columns(exprs)
        return X

    def _transform_numpy(self, X):
        arr = X.copy()
        for col_idx in self.columns_:
            arr[:, col_idx] = arr[:, col_idx].astype(str)
        return arr.astype(object)

    def get_feature_names_out(self, X=None):
        return list(self.columns_) if hasattr(self, 'columns_') else None

    def set_output(self, transform=None):
        pass


class CatOOVFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns=None,
        min_frequency=0,
        missing_value=None,
        treat_empty_string_as_missing=True,
    ):
        self.columns = columns
        self.min_frequency = min_frequency
        self.missing_value = missing_value
        self.treat_empty_string_as_missing = treat_empty_string_as_missing

    def _is_missing(self, v):
        if v is None:
            return True
        if is_nan(v):
            return True
        if self.treat_empty_string_as_missing and isinstance(v, str) and v == "":
            return True
        return False

    def _resolve_columns(self, X, kind):
        if self.columns is None:
            if kind == "numpy":
                return list(range(X.shape[1]))
            return list(X.columns)
        cols = []
        for c in self.columns:
            if isinstance(c, int) and kind != "numpy":
                cols.append(X.columns[c])
            else:
                cols.append(c)
        return cols

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self.columns_ = self._resolve_columns(X, kind)
        self.categories_ = {}
        if kind == "numpy":
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        for col in self.columns_:
            if kind == "pandas_df":
                col_series = X[col]
                if self.treat_empty_string_as_missing:
                    valid = ~(col_series.isna() | (col_series.astype(str) == ''))
                else:
                    valid = ~col_series.isna()
                vc = col_series[valid].astype(str).value_counts()
                self.categories_[col] = sorted(k for k, c in vc.items() if c > self.min_frequency)
            elif kind == "polars_df":
                col_series = X.get_column(col)
                col_utf8 = col_series.cast(pl.Utf8)
                if self.treat_empty_string_as_missing:
                    valid = col_series.is_not_null() & (col_utf8 != "")
                else:
                    valid = col_series.is_not_null()
                vc = col_utf8.filter(valid).value_counts()
                self.categories_[col] = sorted(
                    k for k, c in zip(vc.get_column(col).to_list(), vc.get_column("count").to_list())
                    if c > self.min_frequency
                )
            elif kind == "numpy":
                col_data = arr[:, col]
                valid = np.array([not self._is_missing(v) for v in col_data])
                unique, counts = np.unique(col_data[valid].astype(str), return_counts=True)
                self.categories_[col] = sorted(k for k, c in zip(unique.tolist(), counts.tolist()) if c > self.min_frequency)
            else:
                col_data = X[col].tolist()
                freq = {}
                for v in col_data:
                    if self._is_missing(v):
                        continue
                    v_str = str(v)
                    freq[v_str] = freq.get(v_str, 0) + 1
                self.categories_[col] = sorted(k for k, c in freq.items() if c > self.min_frequency)
        return self

    def _map_value(self, v, allowed_set):
        if self._is_missing(v):
            return self.missing_value
        v_str = str(v)
        if v_str in allowed_set:
            return v_str
        return self.missing_value

    def _get_dtype_categories(self, col):
        cats = list(self.categories_[col])
        mv = self.missing_value
        if mv is not None and not is_nan(mv):
            mv_str = str(mv)
            if mv_str not in cats:
                cats = [mv_str] + cats
        return cats

    def transform(self, X):
        if not hasattr(self, "categories_"):
            raise RuntimeError("This transformer is not fitted yet. Call fit() first.")
        kind = detect_kind(X)
        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _transform_pandas(self, X):
        X_out = X.copy()
        for col in self.columns_:
            allowed = set(self.categories_[col])
            cats = self._get_dtype_categories(col)
            values = [self._map_value(v, allowed) for v in X_out[col].tolist()]
            X_out[col] = pd.Categorical(values, categories=cats)
        return X_out

    def _transform_polars(self, X):
        exprs = []
        for col in self.columns_:
            allowed = set(self.categories_[col])
            cats = self._get_dtype_categories(col)
            mv = self.missing_value
            tes = self.treat_empty_string_as_missing

            def _map(v, _allowed=allowed, _mv=mv, _tes=tes):
                if v is None:
                    return _mv
                if _tes and isinstance(v, str) and v == "":
                    return _mv
                v_str = str(v)
                if v_str in _allowed:
                    return v_str
                return _mv

            exprs.append(
                pl.col(col)
                .map_elements(_map, return_dtype=pl.Utf8)
                .cast(pl.Enum(cats))
                .alias(col)
            )
        return X.with_columns(exprs)

    def _transform_numpy(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = arr.astype(object, copy=True)
        for col in self.columns_:
            allowed = set(self.categories_[col])
            for i in range(out.shape[0]):
                out[i, col] = self._map_value(out[i, col], allowed)
        return out

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "columns_"):
            return [str(c) for c in self.columns_]
        if input_features is not None:
            return list(input_features)
        return None

    def set_output(self, transform=None):
        pass
