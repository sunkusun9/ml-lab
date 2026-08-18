from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
from ._dproc import get_type_df, get_type_pl, merge_type_df

class PolarsLoader(TransformerMixin, BaseEstimator):
    def __init__(self, predefined_types = {}, read_method = 'read_csv', infer_schema_length = 100, safe_cast = True):
        self.predefined_types = predefined_types
        self.read_method = read_method
        self.infer_schema_length = infer_schema_length
        self.safe_cast = safe_cast

    def fit(self, X, y = None):
        if type(X) is list:
            self.df_type_ = merge_type_df([
                pl.scan_csv(i, infer_schema_length=self.infer_schema_length).pipe(get_type_df) for i in X
            ])
        else:
            self.df_type_ = pl.scan_csv(X, infer_schema_length=self.infer_schema_length).pipe(get_type_df)
        self.pl_type_ = get_type_pl(
            self.df_type_, self.predefined_types
        )
        # columns forced to Categorical/Enum by predefined_types whose column is
        # natively numeric (Int/Float) go through a two-stage read: parsed as their
        # natural numeric dtype first, then cast via Utf8. Reading the raw CSV text
        # straight into Categorical would key categories off the raw text tokens, so
        # formatting drift in the source (e.g. "1" vs "1.0" for the same value) would
        # silently split into distinct categories -- see CatConverter's safe_cast for
        # the analogous cast()-time issue.
        self._numeric_to_cat_cols_ = []
        self._read_type_ = dict(self.pl_type_)
        if self.safe_cast:
            natural_type_ = get_type_pl(self.df_type_, {})
            for col, ty in self.pl_type_.items():
                if col not in self.predefined_types:
                    continue
                base = ty.base_type() if hasattr(ty, 'base_type') else ty
                if base not in (pl.Categorical, pl.Enum):
                    continue
                natural = natural_type_.get(col)
                if natural is not None and natural.is_numeric():
                    self._numeric_to_cat_cols_.append(col)
                    self._read_type_[col] = natural
        self.fitted_ = True
        return self

    def _read(self, X):
        df = getattr(pl, self.read_method)(X, schema_overrides=self._read_type_)
        cols = [c for c in self._numeric_to_cat_cols_ if c in df.columns]
        if cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Utf8).cast(self.pl_type_[c]) for c in cols
            ])
        return df

    def transform(self, X):
        if type(X) is list:
            with pl.StringCache():
                return pl.concat([self._read(i) for i in X])
        else:
            return self._read(X)

    def get_params(self, deep = True):
        return {
            'predefined_types': self.predefined_types,
            'read_method': self.read_method,
            'infer_schema_length': self.infer_schema_length,
            'safe_cast': self.safe_cast,
        }
        
    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return [i for i in self.pl_type_.keys()]


class ExprProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, dict_expr, with_columns = True):
        self.dict_expr = dict_expr
        self.with_columns = with_columns
        self.columns = []

    def fit(self, X, y = None):
        if self.with_columns:
            self.columns = X.columns
        self.fitted_ = True
        return self
    
    def transform(self, X):
        if self.with_columns:
            return X.with_columns(**{
                k:v for k, v in self.dict_expr.items()
            })
        else:
            return X.select(**self.dict_expr)
    
    def get_params(self, deep=True):
        return {
            'dict_expr': self.dict_expr,
            'with_columns': self.with_columns
        }

    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return self.columns + list(self.dict_expr.keys())