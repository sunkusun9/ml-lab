import pytest
import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def pd_df():
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'color':  pd.Categorical(np.random.choice(['red', 'green', 'blue'], n)),
        'grade':  pd.Categorical(np.random.choice(['A', 'B', 'C', 'D'], n)),
        'num1':   np.random.randn(n).astype(np.float32),
        'num2':   np.random.randn(n).astype(np.float32),
    })


@pytest.fixture
def clf_target():
    np.random.seed(0)
    return np.random.choice(['cat', 'dog', 'bird'], 200)


@pytest.fixture
def bin_target():
    np.random.seed(0)
    return np.random.choice([0, 1], 200)


@pytest.fixture
def reg_target():
    np.random.seed(0)
    return np.random.randn(200).astype(np.float32)


# ======================================================================
# _make_tf_dataset
# ======================================================================

class TestMakeTfDataset:

    @requires_tf
    def test_num_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('feats', ['num1', 'num2'], 'num')]
        ds = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].dtype == tf.float32
        assert batch['feats'].shape == (10, 2)

    @requires_tf
    def test_int_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        df = pd_df.copy()
        df['code'] = np.arange(len(df))
        specs = [('code', ['code'], 'int')]
        ds = _make_tf_dataset(df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['code'].dtype == tf.int32

    @requires_tf
    def test_str_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('color', ['color'], 'str')]
        ds = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['color'].dtype == tf.string

    @requires_tf
    def test_cat_dtype_as_str(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('color', ['color'], 'str')]
        ds = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(5)))
        assert batch['color'].shape == (5, 1)

    @requires_tf
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars(self):
        from mllabs.nn._input import _make_tf_dataset
        df = pl.DataFrame({
            'a': np.random.randn(50).astype(np.float32),
            'b': np.random.randn(50).astype(np.float32),
        })
        ds = _make_tf_dataset(df, [('feats', ['a', 'b'], 'num')])
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].shape == (10, 2)

    @requires_tf
    def test_numpy(self):
        from mllabs.nn._input import _make_tf_dataset
        arr = np.random.randn(50, 3).astype(np.float32)
        ds = _make_tf_dataset(arr, [('feats', [0, 1, 2], 'num')])
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].shape == (10, 3)

    @requires_tf
    def test_with_label(self, pd_df, bin_target):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('feats', ['num1', 'num2'], 'num')]
        ds = _make_tf_dataset(pd_df, specs, y=bin_target)
        batch = next(iter(ds.batch(10)))
        x_batch, y_batch = batch
        assert x_batch['feats'].shape == (10, 2)
        assert y_batch.shape == (10,)

    @requires_tf
    def test_embedding_tensor_dtype_str(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('color', ['color'], ('Embedding', 4, 'str'))]
        ds = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(5)))
        assert batch['color'].dtype == tf.string

    @requires_tf
    def test_embedding_tensor_dtype_int(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        df = pd_df.copy()
        df['code'] = np.arange(len(df), dtype=np.int32)
        specs = [('code', ['code'], ('Embedding', 4, 'int'))]
        ds = _make_tf_dataset(df, specs)
        batch = next(iter(ds.batch(5)))
        assert batch['code'].dtype == tf.int32

    @requires_tf
    def test_unknown_type_raises(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        with pytest.raises(ValueError):
            _make_tf_dataset(pd_df, [('x', ['num1'], 'unknown')])


# ======================================================================
# _analyze_cols / build_embedding_models
# ======================================================================

class TestEmbeddingEncoder:

    def test_analyze_string_col(self, pd_df):
        from mllabs.nn._input import _analyze_cols
        col_info = _analyze_cols(pd_df, ['color', 'grade'])
        assert col_info['color']['type'] == 'string'
        assert col_info['grade']['type'] == 'string'
        assert col_info['color']['cardinality'] == 3
        assert col_info['grade']['cardinality'] == 4

    def test_analyze_ordinal_int(self):
        from mllabs.nn._input import _analyze_cols
        df = pd.DataFrame({'x': [0, 1, 2, 3, 0, 1]})
        col_info = _analyze_cols(df, ['x'])
        assert col_info['x']['type'] == 'ordinal_int'
        assert col_info['x']['cardinality'] == 4

    def test_analyze_non_ordinal_int(self):
        from mllabs.nn._input import _analyze_cols
        df = pd.DataFrame({'x': [0, 5, 10, 0, 5]})
        col_info = _analyze_cols(df, ['x'])
        assert col_info['x']['type'] == 'int'

    @requires_tf
    def test_build_returns_dict_of_models(self, pd_df):
        from mllabs.nn._input import build_embedding_models
        cat_specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('grade', ['grade'], ('Embedding', 3, 'str')),
        ]
        models = build_embedding_models(pd_df, cat_specs)
        assert set(models.keys()) == {'color', 'grade'}
        for model in models.values():
            assert isinstance(model, tf.keras.Sequential)

    @requires_tf
    def test_build_output_shape(self, pd_df):
        from mllabs.nn._input import build_embedding_models, _analyze_cols
        cat_specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('grade', ['grade'], ('Embedding', 3, 'str')),
        ]
        models = build_embedding_models(pd_df, cat_specs)
        col_info = _analyze_cols(pd_df, ['color', 'grade'])
        for col, dim in [('color', 4), ('grade', 3)]:
            t = col_info[col]['type']
            dtype = 'int32' if t in ('ordinal_int', 'int') else 'string'
            inp = tf.keras.Input(shape=(), dtype=dtype, name=col)
            out = models[col](inp)
            assert out.shape[-1] == dim

    @requires_tf
    def test_build_auto_dim(self, pd_df):
        from mllabs.nn._input import build_embedding_models, _analyze_cols, _auto_emb_dim
        cat_specs = [('color', ['color'], ('Embedding', None, 'str'))]
        models = build_embedding_models(pd_df, cat_specs)
        col_info = _analyze_cols(pd_df, ['color'])
        inp = tf.keras.Input(shape=(), dtype='string', name='color')
        out = models['color'](inp)
        expected_dim = _auto_emb_dim(col_info['color']['cardinality'])
        assert out.shape[-1] == expected_dim


# ======================================================================
# _make_input_model
# ======================================================================

class TestMakeInputModel:

    @requires_tf
    def test_cont_only(self, pd_df):
        from mllabs.nn._input import _make_input_model, _DatasetInputModel
        specs = [('feats', ['num1', 'num2'], 'num')]
        model = _make_input_model(pd_df, specs)
        assert isinstance(model, _DatasetInputModel)
        inp = tf.keras.Input(shape=(2,), name='feats')
        out = model({'feats': inp})
        assert out['feats'].shape[-1] == 2

    @requires_tf
    def test_with_embedding(self, pd_df):
        from mllabs.nn._input import _make_input_model, _DatasetInputModel
        specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('feats', ['num1', 'num2'], 'num'),
        ]
        model = _make_input_model(pd_df, specs)
        assert isinstance(model, _DatasetInputModel)
        cat_inp  = tf.keras.Input(shape=(1,), dtype='string', name='color')
        cont_inp = tf.keras.Input(shape=(2,), name='feats')
        out = model({'color': cat_inp, 'feats': cont_inp})
        assert out['color'].shape[-1] == 4
        assert out['feats'].shape[-1] == 2

    @requires_tf
    def test_cat_only(self, pd_df):
        from mllabs.nn._input import _make_input_model, _DatasetInputModel
        specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('grade', ['grade'], ('Embedding', 3, 'str')),
        ]
        model = _make_input_model(pd_df, specs)
        assert isinstance(model, _DatasetInputModel)
        assert model.make_inputs().keys() == {'color', 'grade'}

    @requires_tf
    def test_make_inputs_dtypes(self, pd_df):
        from mllabs.nn._input import _make_input_model
        specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('feats', ['num1', 'num2'], 'num'),
        ]
        model = _make_input_model(pd_df, specs)
        inputs = model.make_inputs()
        assert inputs['color'].dtype == tf.string
        assert inputs['feats'].dtype == tf.float32


# ======================================================================
# SimpleConcatHead
# ======================================================================

class TestSimpleConcatHead:

    @requires_tf
    def test_concat_emb_and_cont(self):
        from mllabs.nn._head import SimpleConcatHead
        from mllabs.nn._input import _DatasetInputModel

        emb_models = {
            'color': tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=['red', 'green', 'blue'], mask_token=None),
                tf.keras.layers.Embedding(4, 4),
            ]),
            'grade': tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=['A', 'B', 'C', 'D'], mask_token=None),
                tf.keras.layers.Embedding(5, 3),
            ]),
        }
        cat_specs = [('color', ['color'], ('Embedding', 4, 'str')),
                     ('grade', ['grade'], ('Embedding', 3, 'str'))]
        cont_specs = [('__cont__', ['x1', 'x2'], 'num')]
        input_model = _DatasetInputModel(cat_specs, emb_models, cont_specs)

        head = SimpleConcatHead(input_model)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        # (4 + 3 + 2) = 9
        assert out.shape[-1] == 9

    @requires_tf
    def test_no_cont(self):
        from mllabs.nn._head import SimpleConcatHead
        from mllabs.nn._input import _DatasetInputModel

        emb_models = {
            'c1': tf.keras.Sequential([tf.keras.layers.Embedding(5, 4)]),
            'c2': tf.keras.Sequential([tf.keras.layers.Embedding(5, 3)]),
        }
        cat_specs = [('c1', ['c1'], ('Embedding', 4, 'int')),
                     ('c2', ['c2'], ('Embedding', 3, 'int'))]
        input_model = _DatasetInputModel(cat_specs, emb_models, [])

        head = SimpleConcatHead(input_model)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        assert out.shape[-1] == 7

    @requires_tf
    def test_emb_dropout(self):
        from mllabs.nn._head import SimpleConcatHead
        from mllabs.nn._input import _DatasetInputModel

        emb_models = {'c': tf.keras.Sequential([tf.keras.layers.Embedding(5, 4)])}
        cat_specs = [('c', ['c'], ('Embedding', 4, 'int'))]
        input_model = _DatasetInputModel(cat_specs, emb_models, [])

        head = SimpleConcatHead(input_model, emb_dropout=0.2)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        assert out.shape[-1] == 4


# ======================================================================
# DenseBody
# ======================================================================

class TestDenseHidden:

    @requires_tf
    def test_basic(self):
        from mllabs.nn._hidden import DenseHidden
        inp = tf.keras.Input(shape=(16,))
        hidden = DenseHidden(units=(32, 16))
        out = hidden(inp)
        assert out.shape[-1] == 16

    @requires_tf
    def test_batch_norm(self):
        from mllabs.nn._hidden import DenseHidden
        inp = tf.keras.Input(shape=(16,))
        hidden = DenseHidden(units=(32,), batch_norm=True)
        out = hidden(inp)
        assert out.shape[-1] == 32

    @requires_tf
    def test_no_dropout(self):
        from mllabs.nn._hidden import DenseHidden
        inp = tf.keras.Input(shape=(8,))
        hidden = DenseHidden(units=(16,), dropout=0.0)
        out = hidden(inp)
        assert out.shape[-1] == 16

    @requires_tf
    def test_is_keras_model(self):
        from mllabs.nn._hidden import DenseHidden
        hidden = DenseHidden(units=(32,))
        assert isinstance(hidden, tf.keras.Model)


# ======================================================================
# Outputs
# ======================================================================

class TestOutputs:

    @requires_tf
    def test_logit_output(self):
        from mllabs.nn._output import LogitOutput
        inp = tf.keras.Input(shape=(32,))
        output = LogitOutput()
        output.set_output_dim(3)
        out = output(inp)
        assert out.shape[-1] == 3
        assert output.get_loss() == 'sparse_categorical_crossentropy'
        assert 'accuracy' in output.get_metrics()

    @requires_tf
    def test_binary_logit_output(self):
        from mllabs.nn._output import BinaryLogitOutput
        inp = tf.keras.Input(shape=(32,))
        output = BinaryLogitOutput()
        output.set_output_dim(1)
        out = output(inp)
        assert out.shape[-1] == 1
        assert output.get_loss() == 'binary_crossentropy'

    @requires_tf
    def test_regression_output(self):
        from mllabs.nn._output import RegressionOutput
        inp = tf.keras.Input(shape=(32,))
        output = RegressionOutput()
        output.set_output_dim(1)
        out = output(inp)
        assert out.shape[-1] == 1
        assert output.get_loss() == 'mse'
        assert 'mae' in output.get_metrics()

    @requires_tf
    def test_is_keras_model(self):
        from mllabs.nn._output import LogitOutput, BinaryLogitOutput, RegressionOutput
        for cls in (BinaryLogitOutput,):
            assert isinstance(cls(), tf.keras.Model)
        for cls in (LogitOutput, RegressionOutput):
            obj = cls()
            obj.set_output_dim(2)
            assert isinstance(obj, tf.keras.Model)


# ======================================================================
# _split_val
# ======================================================================

class TestSplitVal:
    """_split_val's auto-split path (validation_fraction > 0, no explicit
    eval_set) used to slice sample_weight for the train half only and drop
    the held-out validation half's weight on the floor."""

    @requires_tf
    def test_val_weight_matches_val_indices(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(validation_fraction=0.2, random_state=0)
        clf._fit_encoder(pd_df)
        y_encoded = clf._prepare_target(clf_target)
        weight = np.arange(len(clf_target), dtype=np.float64)

        X_tr, y_tr, X_val, y_val, sw_tr, sw_val = clf._split_val(
            pd_df, y_encoded, sample_weight=weight)

        assert len(sw_tr) == len(y_tr)
        assert len(sw_val) == len(y_val)
        assert len(sw_tr) + len(sw_val) == len(clf_target)

    @requires_tf
    def test_no_weight_returns_none_for_both(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(validation_fraction=0.2, random_state=0)
        clf._fit_encoder(pd_df)
        y_encoded = clf._prepare_target(clf_target)

        *_, sw_tr, sw_val = clf._split_val(pd_df, y_encoded)
        assert sw_tr is None
        assert sw_val is None


# ======================================================================
# NNClassifier
# ======================================================================

class TestNNClassifier:

    @requires_tf
    def test_multiclass_fit_predict(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(pd_df, clf_target)
        preds = clf.predict(pd_df)
        assert preds.shape == (200,)
        assert set(preds).issubset(set(clf_target))

    @requires_tf
    def test_binary_fit_predict(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(pd_df, bin_target)
        preds = clf.predict(pd_df)
        assert set(preds).issubset({0, 1})

    @requires_tf
    def test_predict_proba_shape(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(pd_df, clf_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @requires_tf
    def test_binary_predict_proba_shape(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(pd_df, bin_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 2)

    @requires_tf
    def test_eval_set(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(pd_df, clf_target, eval_set=[(pd_df, clf_target)])
        assert 'valid' in clf.evals_result_
        assert 'train' in clf.evals_result_

    @requires_tf
    def test_eval_set_sample_weight(self, pd_df, clf_target):
        """eval_sample_weight threads into the validation tf.data.Dataset as
        a weighted (inputs, target, weight) triple rather than being silently
        dropped — Keras applies it to val_loss/val_metrics the same way
        sample_weight does for the training set."""
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        weights = np.ones(len(clf_target))
        clf.fit(pd_df, clf_target, eval_set=[(pd_df, clf_target)],
               eval_sample_weight=weights)
        assert 'valid' in clf.evals_result_

    @requires_tf
    def test_validation_fraction(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.2,
                           early_stopping=0)
        clf.fit(pd_df, clf_target)
        assert 'valid' in clf.evals_result_

    @requires_tf
    def test_early_stopping(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=50, batch_size=64, validation_fraction=0.2,
                           early_stopping=2)
        clf.fit(pd_df, clf_target)
        assert 'valid' in clf.evals_result_
        assert isinstance(clf.best_epoch_, int)

    @requires_tf
    def test_early_stopping_dict(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(
            epochs=50, batch_size=64, validation_fraction=0.2,
            early_stopping={'patience': 2, 'monitor': 'val_loss', 'restore_best_weights': True},
        )
        clf.fit(pd_df, clf_target)
        assert 'valid' in clf.evals_result_
        assert isinstance(clf.best_epoch_, int)

    @requires_tf
    def test_early_stopping_zero(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=3, batch_size=64, validation_fraction=0.2,
                           early_stopping=0)
        clf.fit(pd_df, clf_target)
        assert len(clf.evals_result_['train']['loss']) == 3

    @requires_tf
    def test_evals_result_structure(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=3, batch_size=64, validation_fraction=0.2,
                           early_stopping=0)
        clf.fit(pd_df, clf_target)
        assert 'loss' in clf.evals_result_['train']
        assert 'loss' in clf.evals_result_['valid']
        assert len(clf.evals_result_['train']['loss']) == 3

    @requires_tf
    def test_custom_hidden(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier, DenseHidden
        clf = NNClassifier(
            hidden=DenseHidden(units=(32,), dropout=0.1),
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    def test_shuffle_buffer_zero(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0, shuffle_buffer=0)
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    def test_shuffle_buffer_fixed(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0, shuffle_buffer=50)
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    def test_callbacks_lr_schedule(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.9 ** epoch)
        clf = NNClassifier(
            epochs=3, batch_size=64, validation_fraction=0.0,
            early_stopping=0, callbacks=[lr_scheduler],
        )
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    def test_no_cat_cols(self, reg_target):
        from mllabs.nn import NNClassifier
        df = pd.DataFrame({'a': np.random.randn(200), 'b': np.random.randn(200)})
        y = np.random.choice(['x', 'y'], 200)
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping=0)
        clf.fit(df, y)
        assert clf.predict(df).shape == (200,)

    @requires_tf
    def test_explicit_embedding_dims(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(
            embedding_dims={'color': 8, 'grade': 2},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, clf_target)
        assert clf.embedding_dims_['color'] == 8
        assert clf.embedding_dims_['grade'] == 2


# ======================================================================
# NNRegressor
# ======================================================================

class TestNNRegressor:

    @requires_tf
    def test_fit_predict(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.0,
                          early_stopping=0)
        reg.fit(pd_df, reg_target)
        preds = reg.predict(pd_df)
        assert preds.shape == (200,)
        assert preds.dtype == np.float32

    @requires_tf
    def test_eval_set(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.0,
                          early_stopping=0)
        reg.fit(pd_df, reg_target, eval_set=[(pd_df, reg_target)])
        assert 'valid' in reg.evals_result_
        assert 'mae' in reg.evals_result_['valid']

    @requires_tf
    def test_custom_loss_string(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.0,
                          early_stopping=0, loss='mae')
        reg.fit(pd_df, reg_target)
        assert reg.predict(pd_df).shape == (200,)

    @requires_tf
    def test_custom_metrics_string(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.2,
                          early_stopping=0, metrics=['mae', 'mse'])
        reg.fit(pd_df, reg_target)
        assert 'mae' in reg.evals_result_['valid']
        assert 'mse' in reg.evals_result_['valid']


# ======================================================================
# FTTransformerHead
# ======================================================================

class TestFTTransformerHead:

    @requires_tf
    def test_ft_block_output_shape(self):
        from mllabs.nn._fttransformer import FTBlock as _FTBlock
        d_model, n_heads, n_tokens = 32, 4, 5
        block = _FTBlock(d_model, n_heads, ffn_factor=4/3,
                         attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0)
        x = tf.keras.Input(shape=(n_tokens, d_model))
        out = block(x)
        assert out.shape == (None, n_tokens, d_model)

    @requires_tf
    def test_mixed_output_shape(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        d_model = 64
        var_specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('grade', ['grade'], ('Embedding', 4, 'str')),
            ('__cont__', ['num1', 'num2'], 'num'),
        ]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=2)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        assert out.shape[-1] == d_model

    @requires_tf
    def test_cat_only_output_shape(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        d_model = 32
        var_specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('grade', ['grade'], ('Embedding', 4, 'str')),
        ]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        assert out.shape[-1] == d_model

    @requires_tf
    def test_cont_only_output_shape(self):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        df = pd.DataFrame({
            'a': np.random.randn(50).astype(np.float32),
            'b': np.random.randn(50).astype(np.float32),
            'c': np.random.randn(50).astype(np.float32),
        })
        d_model = 32
        var_specs = [('__cont__', ['a', 'b', 'c'], 'num')]
        input_model = _make_input_model(df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        inputs_dict = input_model.make_inputs()
        out = head(inputs_dict)
        assert out.shape[-1] == d_model

    @requires_tf
    def test_cat_projection_applied(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        d_model = 64
        var_specs = [('color', ['color'], ('Embedding', 4, 'str'))]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        assert 'color' in head._cat_projections

    @requires_tf
    def test_cat_projection_skipped_when_dim_matches(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        d_model = 32
        var_specs = [('color', ['color'], ('Embedding', d_model, 'str'))]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        assert 'color' not in head._cat_projections

    @requires_tf
    def test_cont_weights_shape(self):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        df = pd.DataFrame({'a': np.ones(10, dtype=np.float32), 'b': np.ones(10, dtype=np.float32)})
        d_model = 32
        var_specs = [('__cont__', ['a', 'b'], 'num')]
        input_model = _make_input_model(df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        assert head._cont_w.shape == (2, d_model)
        assert head._cont_b.shape == (2, d_model)

    @requires_tf
    def test_cls_token_shape(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        d_model = 48
        var_specs = [('__cont__', ['num1', 'num2'], 'num')]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=d_model, n_heads=4, n_layers=1)
        assert head._cls_token.shape == (1, 1, d_model)

    @requires_tf
    def test_no_cont_no_cont_weights(self, pd_df):
        from mllabs.nn import FTTransformerHead
        from mllabs.nn._input import _make_input_model
        var_specs = [('color', ['color'], ('Embedding', 16, 'str'))]
        input_model = _make_input_model(pd_df, var_specs)
        head = FTTransformerHead(input_model, d_model=16, n_heads=4, n_layers=1)
        assert head._n_cont == 0
        assert not hasattr(head, '_cont_w')

    @requires_tf
    def test_classifier_binary(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier, FTTransformerHead
        clf = NNClassifier(
            head=FTTransformerHead,
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, bin_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 2)

    @requires_tf
    def test_classifier_multiclass(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier, FTTransformerHead
        clf = NNClassifier(
            head=FTTransformerHead,
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, clf_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 3)

    @requires_tf
    def test_regressor(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor, FTTransformerHead
        reg = NNRegressor(
            head=FTTransformerHead,
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        reg.fit(pd_df, reg_target)
        preds = reg.predict(pd_df)
        assert preds.shape == (200,)

    @requires_tf
    def test_head_params(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier, FTTransformerHead
        clf = NNClassifier(
            head=FTTransformerHead,
            head_params={'d_model': 64, 'n_heads': 4, 'n_layers': 2},
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, bin_target)
        assert clf.predict(pd_df).shape == (200,)
        assert clf.head_.d_model == 64
        assert clf.head_.n_layers == 2

    @requires_tf
    def test_pickle_roundtrip(self, pd_df, bin_target):
        import pickle
        from mllabs.nn import NNClassifier, FTTransformerHead
        clf = NNClassifier(
            head=FTTransformerHead,
            head_params={'d_model': 48, 'n_heads': 4, 'n_layers': 1},
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, bin_target)
        expected = clf.predict_proba(pd_df)
        restored = pickle.loads(pickle.dumps(clf))
        result = restored.predict_proba(pd_df)
        assert np.allclose(expected, result, atol=1e-5)

    @requires_tf
    def test_with_dense_hidden(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier, FTTransformerHead, DenseHidden
        clf = NNClassifier(
            head=FTTransformerHead,
            hidden=DenseHidden(units=(64,), dropout=0.1),
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_input(self, bin_target):
        from mllabs.nn import NNClassifier, FTTransformerHead
        np.random.seed(0)
        df = pl.DataFrame({
            'cat1': pl.Series(np.random.choice(['a', 'b', 'c'], 200)).cast(pl.Categorical),
            'num1': np.random.randn(200).astype(np.float32),
            'num2': np.random.randn(200).astype(np.float32),
        })
        clf = NNClassifier(
            head=FTTransformerHead,
            hidden={'units': ()},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping=0,
        )
        clf.fit(df, bin_target)
        assert clf.predict_proba(df).shape == (200, 2)
