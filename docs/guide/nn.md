# Neural Network Models

`mllabs.nn` provides sklearn-compatible neural network estimators built on TensorFlow/Keras. They handle categorical embeddings automatically and integrate with the ml-labs pipeline system.

```
pip install ml-labs tensorflow
```

---

## NNClassifier / NNRegressor

Both estimators share the same constructor and follow the standard sklearn `fit` / `predict` interface.

```python
from mllabs.nn import NNClassifier, NNRegressor

clf = NNClassifier(
    cat_cols=None,           # None → auto-detect from pandas Categorical / polars Categorical dtype
    embedding_dims=None,     # {col: dim} overrides; None → auto (min(50, (cardinality+1)//2))
    head=None,               # Head class or None → SimpleConcatHead
    head_params=None,        # dict of kwargs passed to head factory on construction
    hidden=None,             # NNHidden instance, dict of DenseHidden params, or None → DenseHidden()
    output=None,             # NNOutput instance or None → auto-selected by task
    epochs=100,
    batch_size=1024,
    learning_rate=1e-3,
    early_stopping=10,       # patience; 0/False → disabled; dict → passed to EarlyStopping(**dict)
    validation_fraction=0.1, # internal val split when eval_set is not given (0 → disabled)
    shuffle_buffer=-1,       # -1 → full shuffle; 0 → no shuffle; N → buffer size
    callbacks=None,          # keras Callbacks added every fit() call
    loss=None,               # None → from output component
    metrics=None,            # None → from output component
    random_state=None,
)
```

`NNRegressor` defaults to `validation_fraction=0.0`.

---

## Column Handling

### Categorical columns

Columns passed as `cat_cols` (or auto-detected) are encoded via embedding layers. Three input types are supported:

| Column type | Encoding |
|-------------|----------|
| Ordinal integer (0, 1, 2, …, N-1) | Direct `Embedding` lookup |
| Arbitrary integer | `IntegerLookup` → `Embedding` |
| String / object | `StringLookup` → `Embedding` |

Vocabulary is fitted from training data. Unknown values at inference time are handled by the lookup layer's OOV token.

### Embedding dimensions

Auto-computed as `max(1, min(50, (cardinality + 1) // 2))`. Override per column:

```python
NNClassifier(embedding_dims={'city': 8, 'product_id': 32})
```

### Continuous columns

All non-categorical columns are concatenated as `float32` and passed through unchanged.

---

## Components

The model is assembled in three stages: **Head → Hidden → Output**.

### Head

Merges categorical embeddings and continuous inputs into a single tensor.

```python
from mllabs.nn import SimpleConcatHead

# default: concatenate all embeddings (optionally with dropout) and continuous inputs
NNClassifier(head=SimpleConcatHead)

# with embedding dropout via head_params
NNClassifier(head=SimpleConcatHead, head_params={'emb_dropout': 0.1})
```

`head_params` is a dict of keyword arguments passed to the head factory at construction time. This avoids lambda wrappers when passing custom head options.

#### FTTransformerHead

A Feature Tokenizer + Transformer head for tabular data, following the [FT-Transformer](https://arxiv.org/abs/2106.11959) architecture.

```python
from mllabs.nn import FTTransformerHead

NNClassifier(
    head=FTTransformerHead,
    head_params={
        'd_model': 192,
        'n_heads': 8,
        'n_layers': 3,
        'ffn_factor': 4/3,
        'attention_dropout': 0.2,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.0,
    },
)
```

Each feature is tokenized into a `d_model`-dimensional vector:
- **Categorical**: embedding projected to `d_model` via a linear layer (if `emb_dim != d_model`)
- **Continuous**: per-feature learned weight and bias, producing a `d_model` token

A CLS token is prepended, then `n_layers` transformer blocks (pre-LayerNorm, MHA + FFN/GELU, residual dropout) are applied. The CLS token output is returned and fed into the hidden/output layers.

#### Custom Head

Implement a custom head by subclassing `NNHead`:

```python
from mllabs.nn._head import NNHead

class MyHead(NNHead):
    def call(self, inputs):
        processed = self.input_model(inputs)
        # combine processed['cat_col'], processed['__cont__'], etc.
        ...

NNClassifier(head=MyHead)
```

### Hidden

Applies dense layers after the head. Pass `DenseHidden` constructor parameters as a dict:

```python
from mllabs.nn import DenseHidden

# equivalent forms
NNClassifier(hidden={'units': (512, 256, 128), 'dropout': 0.4, 'batch_norm': True})
NNClassifier(hidden=DenseHidden(units=(512, 256, 128), dropout=0.4, batch_norm=True))
```

`DenseHidden` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `units` | `(256, 128)` | Tuple of layer sizes |
| `dropout` | `0.3` | Dropout rate per layer (0 → disabled) |
| `activation` | `'relu'` | Activation function |
| `batch_norm` | `False` | BatchNormalization after each Dense |

### Output

Selected automatically based on task:

| Task | Output |
|------|--------|
| Binary classification | `BinaryLogitOutput` — sigmoid, `binary_crossentropy` |
| Multi-class | `LogitOutput` — softmax, `sparse_categorical_crossentropy` |
| Regression | `RegressionOutput` — linear, `mse` |

Override with a custom output:

```python
from mllabs.nn import BinaryLogitOutput

NNClassifier(output=BinaryLogitOutput(), loss='focal_crossentropy')
```

---

## fit

```python
clf.fit(X, y, eval_set=None, callbacks=None)
```

| Parameter | Description |
|-----------|-------------|
| `eval_set` | `[(X_val, y_val)]` — explicit validation set; overrides `validation_fraction` |
| `callbacks` | Additional Keras callbacks for this call (e.g., progress monitors from `NNAdapter`) |

`callbacks` is merged with the constructor's `callbacks`. Early stopping is appended automatically when validation data is available.

After fit, `evals_result_` holds training history:

```python
clf.evals_result_
# {'train': {'loss': [...], 'accuracy': [...]},
#  'valid': {'loss': [...], 'accuracy': [...]}}
```

---

## Predict

```python
# NNClassifier
clf.predict(X)          # class labels
clf.predict_proba(X)    # shape (n, n_classes)

# NNRegressor
reg.predict(X)          # shape (n,)
```

---

## Pickle

Both estimators support `pickle` and `joblib`. The Keras model is saved by weights only; the architecture is reconstructed from the fitted column information on load.

```python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
```

---

## Integration with ml-labs

`NNClassifier` and `NNRegressor` work directly as node processors in an `Experimenter`.

```python
from mllabs import Trial

Trial('nn_clf', 'mllabs.nn.NNClassifier',
      edges={'X': 'scale:(*) + ohe:(*)', 'y': '{target}'},
      method='predict_proba',
      params={
          'epochs': 200,
          'hidden': {'units': (256, 128), 'dropout': 0.3},
          'early_stopping': 20,
      })
```

Naming the processor by string is what keeps `import mllabs` from pulling in
TensorFlow — the class is imported only when the Trial actually runs.

### NNAdapter

`NNAdapter` is registered automatically for `NNClassifier` and `NNRegressor`. It handles:

- Passing the inner-validation fold as `eval_set` (controlled by `eval_mode`)
- Progress logging via `logger` (controlled by `verbose`)

```python
Trial('nn_clf', 'mllabs.nn.NNClassifier', edges, method='predict_proba',
      adapter={'__ref__': 'mllabs.adapter.NNAdapter',
               '__params__': {'eval_mode': 'valid', 'verbose': 0.2}},
      params={'epochs': 200})
```

### ModelAttrCollector

`NNAdapter.result_objs` exposes `evals_result` for collection:

```python
collectors.set_collector(
    'nn_evals', 'mllabs.collector.ModelAttrCollector',
    {'__ref__': 'mllabs.Connector',
     '__params__': {'processor': 'mllabs.nn.NNClassifier'}},
    params={'result_key': 'evals_result'},
)
```
