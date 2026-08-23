import contextlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._input import _analyze_cols, _auto_emb_dim, _make_tf_dataset, _make_input_model, _make_input_model_from_col_info
from ._head import SimpleConcatHead
from ._hidden import DenseHidden
from ._output import LogitOutput, BinaryLogitOutput, RegressionOutput

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def _iloc(X, indices):
    if pd is not None and isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[list(indices)]
    if isinstance(X, np.ndarray):
        return X[indices]
    raise TypeError(f"Unsupported data type: {type(X)}")


class _NNBase(BaseEstimator):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        head_params=None,
        hidden=None,
        output=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping=10,
        validation_fraction=0.1,
        shuffle_buffer=-1,
        callbacks=None,
        loss=None,
        metrics=None,
        random_state=None,
        device=None,
    ):
        self.cat_cols = cat_cols
        self.embedding_dims = embedding_dims
        self.head = head
        self.head_params = head_params
        self.hidden = hidden
        self.output = output
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.shuffle_buffer = shuffle_buffer
        self.callbacks = callbacks
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.device = device

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _auto_cat_cols(self, X):
        if pd is not None and isinstance(X, pd.DataFrame):
            return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        if pl is not None and isinstance(X, pl.DataFrame):
            return [c for c in X.columns
                    if X[c].dtype == pl.Categorical
                    or (hasattr(pl, 'Enum') and isinstance(X[c].dtype, pl.Enum))]
        raise ValueError("cat_cols must be specified for numpy input")

    def _all_cols(self, X):
        if pd is not None and isinstance(X, pd.DataFrame):
            return list(X.columns)
        if pl is not None and isinstance(X, pl.DataFrame):
            return list(X.columns)
        if isinstance(X, np.ndarray):
            return list(range(X.shape[1]))
        raise TypeError(f"Unsupported data type: {type(X)}")

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _fit_encoder(self, X):
        cat_cols = self._auto_cat_cols(X) if self.cat_cols is None else list(self.cat_cols)
        self.cat_cols_ = cat_cols
        self.cont_cols_ = [c for c in self._all_cols(X) if c not in set(cat_cols)]
        self.col_info_ = _analyze_cols(X, cat_cols) if cat_cols else {}
        self.var_specs_ = []
        self.embedding_dims_ = self._resolve_embedding_dims()
        for col in self.cat_cols_:
            t   = self.col_info_[col]['type']
            ext = 'int' if t in ('ordinal_int', 'int') else 'str'
            dim = self.embedding_dims_[col]
            self.var_specs_.append((col, [col], ('Embedding', dim, ext)))
        if self.cont_cols_:
            self.var_specs_.append(('__cont__', self.cont_cols_, 'num'))
        

    def _resolve_embedding_dims(self):
        if not self.cat_cols_:
            return {}
        override = self.embedding_dims or {}
        return {
            col: override.get(col) or _auto_emb_dim(self.col_info_[col]['cardinality'])
            for col in self.cat_cols_
        }

    def _resolve_loss(self):
        l = self.loss if self.loss is not None else self.output_.get_loss()
        return tf.keras.losses.get(l) if isinstance(l, str) else l

    def _resolve_metrics(self):
        return self.metrics if self.metrics is not None else self.output_.get_metrics()

    def _make_early_stopping(self):
        es = self.early_stopping
        if not es:
            return None
        if isinstance(es, dict):
            return tf.keras.callbacks.EarlyStopping(**es)
        return tf.keras.callbacks.EarlyStopping(
            patience=es, restore_best_weights=True, monitor='val_loss',
        )

    def _shuffled(self, ds, n):
        if self.shuffle_buffer == 0:
            return ds
        buf = n if self.shuffle_buffer == -1 else self.shuffle_buffer
        return ds.shuffle(buf, seed=self.random_state)

    def _split_val(self, X, y_encoded, sample_weight=None):
        n = len(y_encoded)
        n_val = max(1, int(n * self.validation_fraction))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        sw_tr = sw_val = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight)
            sw_tr = sw_arr[train_idx]
            sw_val = sw_arr[val_idx]
        return (
            _iloc(X, train_idx), y_encoded[train_idx],
            _iloc(X, val_idx),   y_encoded[val_idx],
            sw_tr, sw_val,
        )

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------

    def _build_model(self, X, n_output):
        if hasattr(self, 'model_'):
            del self.model_
        self.input_model_ = _make_input_model(X, self.var_specs_)

        head_factory = self.head or SimpleConcatHead
        self.head_ = head_factory(self.input_model_, **(self.head_params or {}))

        inputs_dict = self.input_model_.make_inputs()
        x = self.head_(inputs_dict)

        hidden = DenseHidden(**(self.hidden if isinstance(self.hidden, dict) else {})) if not self.hidden or isinstance(self.hidden, dict) else self.hidden
        x = hidden(x)

        self.output_ = self._resolve_output()
        self.output_.set_output_dim(n_output)
        output = self.output_(x)

        return tf.keras.Model(inputs=list(inputs_dict.values()), outputs=output)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, eval_set=None, callbacks=None, sample_weight=None, eval_sample_weight=None):
        if tf is None:
            scope = contextlib.nullcontext()
        elif self.device == 'mirror':
            scope = tf.distribute.MirroredStrategy().scope()
        elif self.device:
            scope = tf.device(self.device)
        else:
            scope = contextlib.nullcontext()

        with scope:
            return self._fit(X, y, eval_set=eval_set, callbacks=callbacks,
                             sample_weight=sample_weight, eval_sample_weight=eval_sample_weight)

    def _fit(self, X, y, eval_set=None, callbacks=None, sample_weight=None, eval_sample_weight=None):
        self._fit_encoder(X)
        y_encoded = self._prepare_target(y)
        self.model_ = self._build_model(X, self._n_output())
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self._resolve_loss(),
            metrics=self._resolve_metrics(),
        )

        if eval_set is not None:
            X_val, y_val = eval_set[0]
            y_val = self._encode_y(y_val)
            train_ds = self._shuffled(
                _make_tf_dataset(X, self.var_specs_, y_encoded, sample_weight=sample_weight), len(y_encoded)
            ).batch(self.batch_size)
            val_ds = _make_tf_dataset(
                X_val, self.var_specs_, y_val, sample_weight=eval_sample_weight
            ).batch(self.batch_size)
        elif self.validation_fraction > 0:
            X_tr, y_tr, X_val, y_val, sw_tr, sw_val = self._split_val(X, y_encoded, sample_weight=sample_weight)
            train_ds = self._shuffled(
                _make_tf_dataset(X_tr, self.var_specs_, y_tr, sample_weight=sw_tr), len(y_tr)
            ).batch(self.batch_size)
            val_ds = _make_tf_dataset(X_val, self.var_specs_, y_val, sample_weight=sw_val).batch(self.batch_size)
        else:
            train_ds = self._shuffled(
                _make_tf_dataset(X, self.var_specs_, y_encoded, sample_weight=sample_weight), len(y_encoded)
            ).batch(self.batch_size)
            val_ds = None

        all_callbacks = list(self.callbacks) if self.callbacks else []
        if callbacks:
            all_callbacks.extend(callbacks)
        if val_ds is not None:
            cb = self._make_early_stopping()
            if cb is not None:
                all_callbacks.append(cb)

        history = self.model_.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=all_callbacks,
            verbose=0,
        )

        h = history.history
        self.evals_result_ = {}
        train_keys = [k for k in h if not k.startswith('val_')]
        val_keys   = [k for k in h if k.startswith('val_')]
        if train_keys:
            self.evals_result_['train'] = {k: h[k] for k in train_keys}
        if val_keys:
            self.evals_result_['valid'] = {k[4:]: h[k] for k in val_keys}
        self.best_epoch_ = len(h['loss']) - 1

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_raw(self, X):
        check_is_fitted(self)
        ds = _make_tf_dataset(X, self.var_specs_).batch(self.batch_size)
        return self.model_.predict(ds, verbose=0)

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        weights = None
        if 'model_' in state:
            weights = state.pop('model_').get_weights()
            state.pop('head_', None)
            state.pop('input_model_', None)
            state.pop('output_', None)
        state['_model_weights'] = weights
        return state

    def __setstate__(self, state):
        weights = state.pop('_model_weights', None)
        self.__dict__.update(state)
        if weights is not None:
            self.input_model_ = _make_input_model_from_col_info(self.col_info_, self.var_specs_)
            head_factory = self.head or SimpleConcatHead
            self.head_ = head_factory(self.input_model_, **(self.head_params or {}))
            inputs_dict = self.input_model_.make_inputs()
            x = self.head_(inputs_dict)
            hidden = DenseHidden(**(self.hidden if isinstance(self.hidden, dict) else {})) if not self.hidden or isinstance(self.hidden, dict) else self.hidden
            x = hidden(x)
            self.output_ = self._resolve_output()
            self.output_.set_output_dim(self._n_output())
            output = self.output_(x)
            self.model_ = tf.keras.Model(inputs=list(inputs_dict.values()), outputs=output)
            self.model_.set_weights(weights)

    # ------------------------------------------------------------------
    # Abstract (subclass)
    # ------------------------------------------------------------------

    def _prepare_target(self, y):
        raise NotImplementedError

    def _encode_y(self, y):
        raise NotImplementedError

    def _n_output(self):
        raise NotImplementedError

    def _resolve_output(self):
        raise NotImplementedError


# ======================================================================
# NNClassifier
# ======================================================================

class NNClassifier(_NNBase, ClassifierMixin):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        head_params=None,
        hidden=None,
        output=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping=10,
        validation_fraction=0.1,
        shuffle_buffer=-1,
        callbacks=None,
        loss=None,
        metrics=None,
        random_state=None,
        device = None
    ):
        super().__init__(
            cat_cols=cat_cols, embedding_dims=embedding_dims,
            head=head, head_params=head_params, hidden=hidden, output=output,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            shuffle_buffer=shuffle_buffer, callbacks=callbacks,
            loss=loss, metrics=metrics, random_state=random_state, device = device
        )

    def _prepare_target(self, y):
        self.classes_ = np.unique(y)
        return self._encode_y(y)

    def _encode_y(self, y):
        return np.searchsorted(self.classes_, np.asarray(y)).astype(np.int32)

    def _n_output(self):
        return len(self.classes_)

    def _resolve_output(self):
        if self.output is not None:
            return self.output
        return BinaryLogitOutput() if len(self.classes_) == 2 else LogitOutput()

    def predict_proba(self, X):
        raw = self._predict_raw(X)
        if raw.shape[1] == 1:
            return np.hstack([1 - raw, raw])
        return raw

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ======================================================================
# NNRegressor
# ======================================================================

class NNRegressor(_NNBase, RegressorMixin):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        head_params=None,
        hidden=None,
        output=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping=10,
        validation_fraction=0.0,
        shuffle_buffer=-1,
        callbacks=None,
        loss=None,
        metrics=None,
        random_state=None,
        device = None
    ):
        super().__init__(
            cat_cols=cat_cols, embedding_dims=embedding_dims,
            head=head, head_params=head_params, hidden=hidden, output=output,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            shuffle_buffer=shuffle_buffer, callbacks=callbacks,
            loss=loss, metrics=metrics, random_state=random_state,
            device = device
        )

    def _prepare_target(self, y):
        y_arr = np.asarray(y, dtype=np.float32)
        return y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    def _encode_y(self, y):
        y_arr = np.asarray(y, dtype=np.float32)
        return y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    def _n_output(self):
        return 1

    def _resolve_output(self):
        return self.output if self.output is not None else RegressionOutput()

    def predict(self, X):
        return self._predict_raw(X).ravel()
