import pytest
import numpy as np
import pandas as pd

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")
lgb_early_stopping = lgb.early_stopping

from mllabs.adapter._lightgbm import LightGBMAdapter
from mllabs._pipeline import _params_equal


@pytest.fixture
def adapter():
    return LightGBMAdapter(eval_mode='none', verbose=0)


def make_train_data(weight=None):
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    data = {'X': X, 'y': y}
    if weight is not None:
        data['sample_weight'] = pd.Series(weight)
    return data


def make_valid_data(weight=None):
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    data = {'X': X, 'y': y}
    if weight is not None:
        data['sample_weight'] = pd.Series(weight)
    return data


class TestGetParams:
    def test_filters_early_stopping(self, adapter):
        params = {'n_estimators': 100, 'early_stopping': {'stopping_rounds': 50}}
        result = adapter.get_params(params)
        assert 'early_stopping' not in result
        assert result['n_estimators'] == 100

    def test_filters_eval_metric(self, adapter):
        params = {'n_estimators': 100, 'eval_metric': 'logloss'}
        result = adapter.get_params(params)
        assert 'eval_metric' not in result

    def test_passes_other_params(self, adapter):
        params = {'n_estimators': 200, 'learning_rate': 0.05}
        result = adapter.get_params(params)
        assert result == params


class TestGetFitParamsEarlyStopping:
    def test_dict_early_stopping_creates_callback(self, adapter):
        train_data = make_train_data()
        params = {'early_stopping': {'stopping_rounds': 50, 'verbose': False}}
        fit_params = adapter.get_fit_params(train_data, params=params)
        assert 'callbacks' in fit_params
        assert len(fit_params['callbacks']) == 1
        cb = fit_params['callbacks'][0]
        assert hasattr(cb, 'stopping_rounds') or callable(cb)

    def test_dict_early_stopping_stopping_rounds(self, adapter):
        train_data = make_train_data()
        params = {'early_stopping': {'stopping_rounds': 30, 'verbose': False}}
        fit_params = adapter.get_fit_params(train_data, params=params)
        cb = fit_params['callbacks'][0]
        assert cb.stopping_rounds == 30

    def test_instance_early_stopping_passthrough(self, adapter):
        train_data = make_train_data()
        es_instance = lgb_early_stopping(stopping_rounds=20, verbose=False)
        params = {'early_stopping': es_instance}
        fit_params = adapter.get_fit_params(train_data, params=params)
        assert fit_params['callbacks'][0] is es_instance

    def test_no_early_stopping_no_callbacks(self, adapter):
        train_data = make_train_data()
        fit_params = adapter.get_fit_params(train_data, params={})
        assert 'callbacks' not in fit_params

    def test_eval_metric_passed_to_fit(self, adapter):
        train_data = make_train_data()
        params = {'eval_metric': 'auc'}
        fit_params = adapter.get_fit_params(train_data, params=params)
        assert fit_params.get('eval_metric') == 'auc'

    def test_eval_set_with_valid_data(self):
        adapter = LightGBMAdapter(eval_mode='valid', verbose=0)
        train_data = make_train_data()
        valid_data = make_valid_data()
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        assert 'eval_set' in fit_params
        assert len(fit_params['eval_set']) == 1

    def test_eval_set_both_mode(self):
        adapter = LightGBMAdapter(eval_mode='both', verbose=0)
        train_data = make_train_data()
        valid_data = make_valid_data()
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        assert 'eval_set' in fit_params
        assert len(fit_params['eval_set']) == 2

    def test_no_eval_set_without_valid_data(self):
        adapter = LightGBMAdapter(eval_mode='valid', verbose=0)
        train_data = make_train_data()
        fit_params = adapter.get_fit_params(train_data, params={})
        assert 'eval_set' not in fit_params


class TestGetFitParamsEvalWeight:
    def test_eval_sample_weight_both_mode(self):
        adapter = LightGBMAdapter(eval_mode='both', verbose=0)
        train_data = make_train_data(weight=[1.0, 1.0, 2.0])
        valid_data = make_valid_data(weight=[1.0, 2.0, 3.0])
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        weights = fit_params['eval_sample_weight']
        assert len(weights) == 2
        assert list(weights[0]) == [1.0, 1.0, 2.0]
        assert list(weights[1]) == [1.0, 2.0, 3.0]

    def test_eval_sample_weight_valid_mode_is_valid_weight_only(self):
        adapter = LightGBMAdapter(eval_mode='valid', verbose=0)
        train_data = make_train_data(weight=[1.0, 1.0, 2.0])
        valid_data = make_valid_data(weight=[1.0, 2.0, 3.0])
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        weights = fit_params['eval_sample_weight']
        assert len(weights) == 1
        assert list(weights[0]) == [1.0, 2.0, 3.0]

    def test_no_eval_sample_weight_without_weight_edge(self):
        adapter = LightGBMAdapter(eval_mode='both', verbose=0)
        fit_params = adapter.get_fit_params(make_train_data(), make_valid_data(), params={})
        assert 'eval_sample_weight' not in fit_params

    def test_missing_valid_weight_is_padded_with_none(self):
        """train_data carries a weight but valid_data doesn't (edges applied
        oddly, or a hand-built call) — the valid entry stays None (uniform)
        rather than the whole param being dropped or the list shrinking."""
        adapter = LightGBMAdapter(eval_mode='both', verbose=0)
        train_data = make_train_data(weight=[1.0, 1.0, 2.0])
        valid_data = make_valid_data()
        fit_params = adapter.get_fit_params(train_data, valid_data, params={})
        weights = fit_params['eval_sample_weight']
        assert list(weights[0]) == [1.0, 1.0, 2.0]
        assert weights[1] is None


class TestParamsEqualWithDictEarlyStopping:
    def test_same_dict_is_equal(self):
        a = {'n_estimators': 100, 'early_stopping': {'stopping_rounds': 50, 'verbose': False}}
        b = {'n_estimators': 100, 'early_stopping': {'stopping_rounds': 50, 'verbose': False}}
        assert _params_equal(a, b)

    def test_different_dict_not_equal(self):
        a = {'early_stopping': {'stopping_rounds': 50}}
        b = {'early_stopping': {'stopping_rounds': 30}}
        assert not _params_equal(a, b)

    def test_two_instances_not_equal(self):
        # 인스턴스 두 개는 _params_equal에서 같다고 보장되지 않음 (이것이 이슈의 근본)
        a_dict = {'early_stopping': {'stopping_rounds': 50}}
        b_dict = {'early_stopping': {'stopping_rounds': 50}}
        assert _params_equal(a_dict, b_dict)
