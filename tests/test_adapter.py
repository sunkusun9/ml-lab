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
        s = stack_evals_result({
            'learn': {'MultiClass': [1.0, 0.9, 0.8]},
            'validation': {'MultiClass': [1.1, 1.0, 0.95], 'AUC': [0.5, 0.6]},
        })
        assert s[(1, 'AUC', 'validation')] == 0.6
        assert s[(2, 'MultiClass', 'validation')] == 0.95
        assert (2, 'AUC', 'validation') not in s.index

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
