# Processors

ml-labs provides a set of built-in sklearn-compatible processors for common preprocessing tasks.

## Built-in Processors

### CatConverter

Converts specified columns to a categorical dtype (pandas `category` / polars `Categorical`).

```python
from mllabs.processor import CatConverter

CatConverter(columns=None)   # None → convert all columns
CatConverter(columns=['city', 'gender'])
```

Output columns are the same as input columns with dtype changed. Supports pandas, polars, and numpy inputs.

### CatOOVFilter

Filters out-of-vocabulary (OOV) values in categorical columns. Values not seen during `fit()` (or seen fewer than `min_frequency` times) are replaced with `missing_value`.

```python
from mllabs.processor import CatOOVFilter

CatOOVFilter(
    columns=None,                    # None → all columns
    min_frequency=0,                 # values with count <= this are treated as OOV
    missing_value=None,              # replacement for OOV/missing values
    treat_empty_string_as_missing=True,
)
```

Fitted attribute: `categories_` — dict of `{column: [allowed_values]}`.

### CatPairCombiner

Combines groups of categorical columns into new interaction columns by concatenating their string representations. Supports N-way (2 or more columns per group).

```python
from mllabs.processor import CatPairCombiner

# pair (2-way)
CatPairCombiner(
    pairs=[('city', 'gender'), ('age_bin', 'city')],
    sep='__',
    treat_empty_string_as_missing=True,
    new_col_names=None,   # None → 'city__gender', 'age_bin__city'
)

# N-way (3 or more columns)
CatPairCombiner(
    pairs=[('city', 'gender', 'age_bin')],
    sep='__',
)
# output: 'city__gender__age_bin'
```

Each group element can be a column name (str) or integer index (numpy). If any value in a group is missing, the output is `None`. Supports pandas, polars, and numpy inputs.

### TypeConverter

Converts all columns to a target type. Useful for normalizing mixed-type columns before feeding into processors that require a specific dtype.

```python
from mllabs.processor import TypeConverter

TypeConverter(to='str')    # str, int, or float
TypeConverter(to='float')
```

| Backend | Behavior |
|---------|----------|
| pandas | `df.astype(to)` |
| polars | `cast` to `pl.Utf8` / `pl.Int64` / `pl.Float64` |
| numpy | `array.astype(to)` |

`get_feature_names_out` is supported — column names are preserved after conversion.

### FrequencyEncoder

Replaces each categorical value with its frequency (proportion or count) observed during `fit()`.

```python
from mllabs.processor import FrequencyEncoder

FrequencyEncoder(normalize=True)   # True → proportion, False → count
```

Output column names are `{col}_freq`. Unseen values at transform time receive `0`.

### CrossFitTransformer

Generates out-of-fold (OOF) predictions from a classifier or regressor during `fit_transform`, and uses a full-data fitted model during `transform`. Useful for building stacking meta-features inside a pipeline node.

```python
from mllabs.processor import CrossFitTransformer
from lightgbm import LGBMClassifier

CrossFitTransformer(
    estimator=LGBMClassifier(),
    cv=5,                    # number of folds (int) or a CV splitter instance
    method='predict_proba',  # 'predict_proba', 'predict', or any estimator method name
    stratified=True,         # use StratifiedKFold when cv is int (ignored for regression)
)
```

**Behaviour:**

| Call | What happens |
|------|-------------|
| `fit_transform(X, y)` | Fits full estimator on all data; generates OOF predictions via CV splits; returns OOF array |
| `transform(X)` | Applies the full estimator fitted during `fit_transform` |
| `fit(X, y)` | Fits full estimator only (no OOF — use `fit_transform` in training pipelines) |

**Output column names** are derived from the estimator class name and output type:

```python
# predict_proba with classes [0, 1, 2]  →  ['lgbmclassifier_0', 'lgbmclassifier_1', 'lgbmclassifier_2']
# predict                                →  ['lgbmclassifier_pred']
```

Column names are accessible via `get_feature_names_out()`.

**Usage as a node:**

```python
p.set_node('meta_lgbm',
           processor='mllabs.processor.CrossFitTransformer',
           method='fit_transform',
           edges={'X': 'scale:(*)', 'y': '{target}'},
           params={'estimator': {'__ref__': 'lightgbm.LGBMClassifier',
                                 '__params__': {'n_estimators': 100}},
                   'cv': 5, 'method': 'predict_proba'})
```

The node produces OOF features in an `Experimenter` and full-model predictions in a `Trainer` or `Inferencer` — the same definition, behaving correctly on both sides.

---

## Polars-Optional Processors

Available only when `polars` is installed (`pip install ml-labs[polars]`).

### PolarsLoader

Reads one or more CSV files into a Polars DataFrame with automatically optimized dtypes. Uses `get_type_df` internally to analyze data ranges.

```python
from mllabs.processor import PolarsLoader

PolarsLoader(
    predefined_types={'id': pl.Int64},   # override specific column types
    read_method='read_csv',              # any pl.read_* method name
)

# fit: accepts a file path (str) or list of file paths
# transform: returns pl.DataFrame with optimized schema
```

### ExprProcessor

Applies a dict of Polars expressions to a DataFrame via `with_columns` or `select`.

```python
from mllabs.processor import ExprProcessor
import polars as pl

ExprProcessor(
    dict_expr={
        'log_income': pl.col('income').log1p(),
        'age_sq': pl.col('age') ** 2,
    },
    with_columns=True,   # True → add/replace columns; False → select only these columns
)
```

When `with_columns=True`, output includes all original columns plus the new/replaced ones. When `False`, output contains only the expressions defined in `dict_expr`.

### PandasConverter

Converts a Polars DataFrame to pandas. Useful as the final stage in a Polars-based pipeline.

```python
from mllabs.processor import PandasConverter

PandasConverter(index_col=None)   # optionally set a column as the DataFrame index
```

---

## Type Utilities

Utilities for analyzing column types and generating optimal dtype mappings. Useful for preprocessing pipelines that need to minimize memory usage.

```python
from mllabs.processor import get_type_df, get_type_pl, get_type_pd, merge_type_df
```

### get_type_df(df)

Analyzes a Polars DataFrame (or LazyFrame) and returns a `pd.DataFrame` with per-column statistics and dtype-fit flags.

```python
df_type = pl.scan_csv('train.csv').pipe(get_type_df)
# Columns: min, max, na, count, n_unique, dtype, f32, i32, i16, i8
# f32/i32/i16/i8: True if numeric values fit within that type's range
```

### get_type_pl(df_type, ...) / get_type_pd(df_type, ...)

Convert the `df_type` analysis into a type mapping dict for Polars or pandas loading.

```python
pl_types = get_type_pl(
    df_type,
    predefine={'id': pl.Int64},  # always override these
    f32=True,                    # use Float32 where possible
    i64=False,                   # use smallest int type that fits
    cat_max=50,                  # columns with ≤50 unique values → Categorical
    txt_cols=['description'],    # force these columns to String
)

pd_types = get_type_pd(df_type, predefine={'id': 'int64'}, cat_max=50)
```

### merge_type_df(dfs)

Merges `df_type` results from multiple files (e.g., sharded datasets) into a single summary. Takes the global min/max across files to determine safe dtype ranges.

```python
from glob import glob

df_type = merge_type_df([
    pl.scan_csv(f).pipe(get_type_df) for f in glob('data/*.csv')
])
```
