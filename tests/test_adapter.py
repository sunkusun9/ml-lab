import numpy as np
import pandas as pd
import pytest

from mllabs.adapter._base import stack_evals_result


class TestStackEvalsResult:
    """CatBoost records a metric like AUC on a different cadence than the loss,
    so two curves in the same split can differ in length — which used to raise
    ValueError('All arrays must be of the same length') out of the collect
    call, killing that Collector for that fold."""

    def test_equal_length_curves(self):
        s = stack_evals_result({
            'learn': {'MultiClass': [1.0, 0.9, 0.8]},
            'validation': {'MultiClass': [1.1, 1.0, 0.95]},
        })
        assert s[(0, 'MultiClass', 'learn')] == 1.0
        assert s[(2, 'MultiClass', 'validation')] == 0.95
        assert len(s) == 6

    def test_ragged_curves_in_one_split(self):
        """The padded tail is dropped rather than left as NaN — whether
        .stack() itself drops it depends on the pandas version, so the helper
        settles it instead of inheriting that default."""
        s = stack_evals_result({
            'learn': {'MultiClass': [1.0, 0.9, 0.8]},
            'validation': {'MultiClass': [1.1, 1.0, 0.95], 'AUC': [0.5, 0.6]},
        })
        assert s[(1, 'AUC', 'validation')] == 0.6
        assert s[(2, 'MultiClass', 'validation')] == 0.95
        assert (2, 'AUC', 'validation') not in s.index
        assert not s.isna().any()

    def test_metric_absent_from_a_split_leaves_no_row(self):
        """CatBoost records AUC on validation only, so (iter, 'AUC', 'learn')
        is NaN for every iteration. Those rows were dropped before this helper
        existed, and get_attrs_agg's groupby/mean still assumes they are."""
        s = stack_evals_result({
            'learn': {'MultiClass': [1.0, 0.9]},
            'validation': {'MultiClass': [1.1, 1.0], 'AUC': [0.5, 0.6]},
        })
        assert [k for k in s.index if k[1] == 'AUC'] == [
            (0, 'AUC', 'validation'), (1, 'AUC', 'validation')
        ]

    def test_ragged_result_matches_the_equal_length_path(self):
        """Padding must not invent values: the rows a short curve does have are
        the same ones it would have had on its own."""
        ragged = stack_evals_result({'validation': {'a': [1.0, 2.0, 3.0], 'b': [9.0]}})
        alone = stack_evals_result({'validation': {'b': [9.0]}})
        assert ragged[(0, 'b', 'validation')] == alone[(0, 'b', 'validation')]

    def test_differing_metric_sets_between_splits(self):
        s = stack_evals_result({
            'learn': {'MultiClass': [1.0, 0.9]},
            'validation': {'MultiClass': [1.1, 1.0], 'AUC': [0.5, 0.6]},
        })
        assert np.isnan(s.get((0, 'AUC', 'learn'), np.nan))
        assert s[(0, 'AUC', 'validation')] == 0.5

    def test_empty_is_an_empty_series(self):
        s = stack_evals_result({})
        assert isinstance(s, pd.Series) and len(s) == 0


class TestAdaptersShareTheHelper:
    """All four adapters built the same frame the same way, so the ragged case
    was latent in every one of them, not just CatBoost's."""

    @pytest.mark.parametrize('module,cls_name,attr', [
        ('mllabs.adapter._xgboost', 'XGBoostAdapter', 'xgboost'),
        ('mllabs.adapter._lightgbm', 'LightGBMAdapter', 'lightgbm'),
        ('mllabs.adapter._catboost', 'CatBoostAdapter', 'catboost'),
    ])
    def test_evals_result_survives_ragged_curves(self, module, cls_name, attr):
        pytest.importorskip(attr, reason=f'{attr} not installed')
        import importlib
        adapter_cls = getattr(importlib.import_module(module), cls_name)

        class FakeEstimator:
            def evals_result(self):
                return self._er

            def get_evals_result(self):
                return self._er

            _er = {'learn': {'loss': [1.0, 0.9]},
                   'validation': {'loss': [1.1, 1.0], 'AUC': [0.5]}}
            evals_result_ = _er

        class FakeProcessor:
            obj = FakeEstimator()

        s = adapter_cls.result_objs['evals_result'][0](FakeProcessor())
        assert s[(0, 'AUC', 'validation')] == 0.5


class TestEvalWeightList:
    """ModelAdapter._eval_weight_list — positions eval weights to match the
    eval_set list eval_mode builds, for LightGBM/XGBoost (list-of-weights
    adapters; CatBoost folds weight into a Pool instead, tested per-adapter
    below)."""

    def test_neither_side_returns_none(self):
        from mllabs.adapter._base import ModelAdapter
        assert ModelAdapter._eval_weight_list(None, None, 'both') is None

    def test_valid_mode_returns_valid_weight_only(self):
        from mllabs.adapter._base import ModelAdapter
        assert ModelAdapter._eval_weight_list('train_w', 'valid_w', 'valid') == ['valid_w']

    def test_both_mode_returns_train_then_valid(self):
        from mllabs.adapter._base import ModelAdapter
        assert ModelAdapter._eval_weight_list('train_w', 'valid_w', 'both') == ['train_w', 'valid_w']

    def test_missing_side_is_none_not_dropped(self):
        """None stays in its position (uniform weight for just that eval
        set) rather than shrinking the list, which would desync it from
        eval_set's own length."""
        from mllabs.adapter._base import ModelAdapter
        assert ModelAdapter._eval_weight_list('train_w', None, 'both') == ['train_w', None]
        assert ModelAdapter._eval_weight_list(None, 'valid_w', 'both') == [None, 'valid_w']

    def test_unknown_mode_returns_none(self):
        from mllabs.adapter._base import ModelAdapter
        assert ModelAdapter._eval_weight_list('t', 'v', 'none') is None


class TestXGBoostEvalWeight:
    def _adapter(self, eval_mode='both'):
        pytest.importorskip('xgboost', reason='xgboost not installed')
        from mllabs.adapter._xgboost import XGBoostAdapter
        return XGBoostAdapter(eval_mode=eval_mode, verbose=0)

    def test_eval_sample_weight_both_mode(self):
        adapter = self._adapter('both')
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0]),
                      'sample_weight': pd.Series([1.0, 1.0, 2.0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1]),
                      'sample_weight': pd.Series([1.0, 2.0, 3.0])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        weights = fit_params['sample_weight_eval_set']
        assert len(weights) == 2
        assert list(weights[0]) == [1.0, 1.0, 2.0]
        assert list(weights[1]) == [1.0, 2.0, 3.0]

    def test_no_weight_key_without_weight_edge(self):
        adapter = self._adapter('both')
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        assert 'sample_weight_eval_set' not in fit_params


class TestCatBoostEvalWeight:
    def _adapter(self, eval_mode='both'):
        pytest.importorskip('catboost', reason='catboost not installed')
        from mllabs.adapter._catboost import CatBoostAdapter
        return CatBoostAdapter(eval_mode=eval_mode, verbose=0)

    def test_eval_set_becomes_pool_when_weighted(self):
        from catboost import Pool
        adapter = self._adapter('both')
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0]),
                      'sample_weight': pd.Series([1.0, 1.0, 2.0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1]),
                      'sample_weight': pd.Series([1.0, 2.0, 3.0])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        eval_set = fit_params['eval_set']
        assert len(eval_set) == 2
        assert all(isinstance(p, Pool) for p in eval_set)

    def test_eval_set_stays_plain_tuples_without_weight(self):
        adapter = self._adapter('both')
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        eval_set = fit_params['eval_set']
        assert len(eval_set) == 2
        assert all(isinstance(p, tuple) for p in eval_set)


class TestNNAdapterEvalWeight:
    """NNAdapter only ever builds one eval_set entry (no LightGBM/XGBoost
    'both' second slot for train), so its eval weight is a single array,
    not a positional list — no _eval_weight_list involved."""

    def _adapter(self, eval_mode='valid'):
        pytest.importorskip('tensorflow', reason='tensorflow not installed')
        from mllabs.adapter._nn import NNAdapter
        return NNAdapter(eval_mode=eval_mode, verbose=0)

    def test_eval_sample_weight_set_when_valid_has_weight(self):
        adapter = self._adapter()
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1]),
                      'sample_weight': pd.Series([1.0, 2.0, 3.0])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        assert list(fit_params['eval_sample_weight']) == [1.0, 2.0, 3.0]

    def test_no_eval_sample_weight_without_weight_edge(self):
        adapter = self._adapter()
        train_data = {'X': pd.DataFrame({'a': [1, 2, 3]}), 'y': pd.Series([0, 1, 0])}
        valid_data = {'X': pd.DataFrame({'a': [4, 5, 6]}), 'y': pd.Series([1, 0, 1])}
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        assert 'eval_sample_weight' not in fit_params
