import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._pipeline import Pipeline
from mllabs._experimenter import Experimenter
from mllabs import Connector, MetricCollector, StackingCollector, ModelAttrCollector, OutputCollector, ProcessCollector

Built = namedtuple('Built', ['e', 'p'])


class FailPredictor:
    __name__ = 'FailPredictor'
    def __init__(self, **kwargs): pass
    def fit(self, X, y=None): raise RuntimeError("fail")
    def predict(self, X): pass


def accuracy_metric(y, pred):
    return (y.values == pred.values).mean()


def dummy_metric(y, pred):
    return 0.5


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
        'target': np.random.randint(0, 2, n),
    })


@pytest.fixture
def built_exp(tmp_path, sample_data):
    p = Pipeline(path=tmp_path / 'pipeline_built')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', role='stage', processor=StandardScaler,
              method='transform', edges={'X': '{f1, f2, f3}'})
    p.set_node('scaler', grp='scale')
    p.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': 'scaler:(*)', 'y': '{target}'},
              params={'max_depth': 3, 'random_state': 42})
    p.set_node('dt', grp='model')
    exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_built',
                           sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
    exp_obj.set_pipeline(p)
    exp_obj.build()
    exp_obj.exp()
    return Built(e=exp_obj, p=p)


@pytest.fixture
def built_exp_inner(tmp_path, sample_data):
    p = Pipeline(path=tmp_path / 'pipeline_inner')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': '{f1, f2, f3}', 'y': '{target}'},
              params={'max_depth': 3, 'random_state': 42})
    p.set_node('dt', grp='model')
    exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_inner',
                           sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
                           sp_v=KFold(n_splits=3, shuffle=True, random_state=42))
    exp_obj.set_pipeline(p)
    exp_obj.build()
    exp_obj.exp()
    return Built(e=exp_obj, p=p)


@pytest.fixture
def multi_head_exp(tmp_path, sample_data):
    p = Pipeline(path=tmp_path / 'pipeline_multi')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': '{f1, f2, f3}', 'y': '{target}'},
              params={'max_depth': 3, 'random_state': 42})
    p.set_node('dt1', grp='model')
    p.set_node('dt2', grp='model', params={'max_depth': 5})
    exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_multi',
                           sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
    exp_obj.set_pipeline(p)
    exp_obj.build()
    exp_obj.exp()
    return Built(e=exp_obj, p=p)


class TestConnector:
    def test_match_all(self):
        c = Connector()
        assert c.match({'name': 'any_node'}) is True

    def test_match_node_query_str(self):
        c = Connector(node_query='dt')
        assert c.match({'name': 'dt1'}) is True
        assert c.match({'name': 'scaler'}) is False

    def test_match_node_query_regex(self):
        c = Connector(node_query='^dt')
        assert c.match({'name': 'dt1'}) is True
        assert c.match({'name': 'my_dt'}) is False

    def test_match_node_query_list(self):
        c = Connector(node_query=['dt1', 'dt2'])
        assert c.match({'name': 'dt1'}) is True
        assert c.match({'name': 'dt3'}) is False

    def test_match_processor(self):
        c = Connector(processor=DecisionTreeClassifier)
        assert c.match({'name': 'dt', 'processor': DecisionTreeClassifier}) is True
        assert c.match({'name': 'dt', 'processor': StandardScaler}) is False

    def test_match_processor_by_ref_string(self):
        c = Connector(processor='sklearn.tree.DecisionTreeClassifier')
        assert c.match({'name': 'dt', 'processor': DecisionTreeClassifier}) is True
        assert c.match({'name': 'dt', 'processor': StandardScaler}) is False

    def test_match_edges(self):
        c = Connector(edges={'X': '{f1}'})
        node_attrs = {'name': 'dt', 'edges': {'X': '{f1}', 'y': '{target}'}}
        assert c.match(node_attrs) is True

    def test_match_edges_different_value(self):
        c = Connector(edges={'X': '{f1}'})
        node_attrs = {'name': 'dt', 'edges': {'X': '{f1, f2}'}}
        assert c.match(node_attrs) is False

    def test_match_edges_missing_key(self):
        c = Connector(edges={'z': '{f1}'})
        assert c.match({'name': 'dt', 'edges': {'X': '{f1}'}}) is False

    def test_match_combined(self):
        c = Connector(node_query='dt', processor=DecisionTreeClassifier)
        assert c.match({'name': 'dt1', 'processor': DecisionTreeClassifier}) is True
        assert c.match({'name': 'dt1', 'processor': StandardScaler}) is False
        assert c.match({'name': 'scaler', 'processor': DecisionTreeClassifier}) is False


class TestMetricCollector:
    def test_collect_basic(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        assert mc.has('dt')

    def test_get_metric(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        result = mc.get_metric('dt')
        assert isinstance(result, pd.Series)
        assert result.name == 'dt'
        assert len(result) > 0
        assert all(0 <= v <= 1 for v in result.values)

    def test_get_metrics(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        result = mc.get_metrics()
        assert isinstance(result, pd.DataFrame)
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' in result.index.get_level_values(0)

    def test_get_metrics_with_node_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        result = mc.get_metrics(nodes=['dt1'])
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' not in result.index.get_level_values(0)

    def test_get_metrics_regex(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        result = mc.get_metrics(nodes='dt1')
        assert len(result) > 0

    def test_get_metrics_agg(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        mean, std = mc.get_metrics_agg()
        assert isinstance(mean, pd.DataFrame)
        assert std is None

    def test_get_metrics_agg_with_std(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        mean, std = mc.get_metrics_agg(include_std=True)
        assert isinstance(mean, pd.DataFrame)
        assert isinstance(std, pd.DataFrame)

    def test_get_metrics_agg_inner_only(self, built_exp_inner):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp_inner.e.add_collector(mc)
        built_exp_inner.e.collect(mc)
        mean, std = mc.get_metrics_agg(inner_fold=True, outer_fold=False)
        assert isinstance(mean, pd.DataFrame)

    def test_get_metrics_agg_no_fold(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        result = mc.get_metrics_agg(inner_fold=False, outer_fold=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_metrics_agg_invalid(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        with pytest.raises(ValueError):
            mc.get_metrics_agg(inner_fold=False, outer_fold=True)

    def test_include_train(self, built_exp):
        mc = MetricCollector('acc_train', Connector(), output_var=None,
                             metric_func=accuracy_metric, include_train=True)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        result = mc.get_metric('dt')
        assert 'train' in result.index.get_level_values(-1)

    def test_inner_split_metrics(self, built_exp_inner):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp_inner.e.add_collector(mc)
        built_exp_inner.e.collect(mc)
        result = mc.get_metric('dt')
        assert len(result) > 2

    def test_connector_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(node_query=['dt1']),
                             output_var=None, metric_func=accuracy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        assert mc.has('dt1')
        assert not mc.has('dt2')

    def test_reset_nodes(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        assert mc.has('dt')
        mc.reset_nodes(['dt'])
        assert not mc.has('dt')

    def test_save_load(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        loaded = MetricCollector.load(mc.path)
        assert loaded.has('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_ad_hoc_collect(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        mc2 = MetricCollector('acc2', Connector(), output_var=None,
                              metric_func=dummy_metric)
        built_exp.e.add_collector(mc2)
        built_exp.e.collect(mc2)
        assert mc2.has('dt')
        result = mc2.get_metric('dt')
        assert all(v == 0.5 for v in result.values)


class TestStackingCollector:
    def test_collect_basic(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        assert sc.has_node('dt')

    def test_get_dataset(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': '{target}'}
        ), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        ds = sc.get_dataset()
        assert isinstance(ds, pd.DataFrame)
        assert len(ds) == int(len(built_exp.e.data.data) * 0.2) * 2 # ShuffleSplit, n_splits = 2, test_size = 0.2
        assert 'target' in ds.columns

    def test_get_dataset_no_target(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        ds = sc.get_dataset(include_target=False)
        assert isinstance(ds, pd.DataFrame)
        assert 'target' not in ds.columns

    def test_get_dataset_multi_nodes(self, multi_head_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': '{target}'}
        ), output_var=None)
        multi_head_exp.e.add_collector(sc)
        multi_head_exp.e.collect(sc)
        ds = sc.get_dataset()
        assert ds.shape[1] > 2

    def test_get_dataset_node_filter(self, multi_head_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': '{target}'}
        ), output_var=None)
        multi_head_exp.e.add_collector(sc)
        multi_head_exp.e.collect(sc)
        ds = sc.get_dataset(nodes=['dt1'])
        assert isinstance(ds, pd.DataFrame)

    def test_method_mean(self, built_exp_inner):
        sc = StackingCollector('stk_mean', Connector(
            edges={'y': '{target}'}
        ), output_var=None, method='mean')
        built_exp_inner.e.add_collector(sc)
        built_exp_inner.e.collect(sc)
        assert sc.has_node('dt')

    def test_reset_nodes(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        assert sc.has_node('dt')
        sc.reset_nodes(['dt'])
        assert not sc.has_node('dt')

    def test_save_load(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': '{target}'}
        ), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        loaded = StackingCollector.load(sc.path)
        assert loaded.has_node('dt')
        ds_orig = sc.get_dataset()
        ds_loaded = loaded.get_dataset()
        pd.testing.assert_frame_equal(ds_orig, ds_loaded)

    def test_index_preserved(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': '{target}'}
        ), output_var=None)
        built_exp.e.add_collector(sc)
        built_exp.e.collect(sc)
        ds = sc.get_dataset()
        all_valid_idx = np.concatenate([
            built_exp.e.outer_folds[i].test_idx
            for i in range(built_exp.e.get_n_splits())
        ])
        expected_index = built_exp.e.data.data.index[all_valid_idx]
        pd.testing.assert_index_equal(ds.index, expected_index)


class TestModelAttrCollector:
    def test_collect_basic(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        assert mac.has('dt')

    def test_get_attr(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        result = mac.get_attr('dt')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_attr_idx(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        result = mac.get_attr('dt', idx=0)
        assert isinstance(result, list)

    def test_get_attrs(self, multi_head_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        multi_head_exp.e.add_collector(mac)
        multi_head_exp.e.collect(mac)
        result = mac.get_attrs()
        assert 'dt1' in result
        assert 'dt2' in result

    def test_get_attrs_agg(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        result = mac.get_attrs_agg('dt')
        assert isinstance(result, pd.Series)

    def test_get_attrs_agg_inner_only(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        result = mac.get_attrs_agg('dt', agg_inner=True, agg_outer=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_attrs_agg_invalid(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        with pytest.raises(ValueError):
            mac.get_attrs_agg('dt', agg_inner=False, agg_outer=True)

    def test_not_mergeable(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('tree', Connector(processor=DecisionTreeClassifier),
                                result_key='tree',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        with pytest.raises(ValueError, match='not mergeable'):
            mac.get_attrs_agg('dt')

    def test_reset_nodes(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        mac.reset_nodes(['dt'])
        assert not mac.has('dt')

    def test_save_load(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        loaded = ModelAttrCollector.load(mac.path)
        assert loaded.has('dt')

    def test_auto_adapter(self):
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances')
        assert mac.adapter is not None

    def test_auto_adapter_invalid_key(self):
        with pytest.raises(RuntimeError):
            ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                               result_key='nonexistent_key')


class TestOutputCollector:
    def test_collect_basic(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        assert oc.has_node('dt')

    def test_get_output(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        result = oc.get_output('dt', 0, 0)
        print(result)
        assert 'output_test' in result
        assert 'output_train' in result
        assert 'columns' in result

    def test_get_output_structure(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        result = oc.get_output('dt', 0, 0)
        assert isinstance(result['output_test'], np.ndarray)
        assert result['output_train'] is None or isinstance(result['output_train'], np.ndarray)
        assert result['output_valid'] is None or isinstance(result['output_valid'], np.ndarray)

    def test_get_outputs(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        results = oc.get_outputs('dt')
        assert isinstance(results, dict)
        assert len(results) == built_exp.e.get_n_splits() * built_exp.e.get_n_splits_inner()
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_get_outputs_inner_split(self, built_exp_inner):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp_inner.e.add_collector(oc)
        built_exp_inner.e.collect(oc)
        results = oc.get_outputs('dt')
        n_expected = built_exp_inner.e.get_n_splits() * built_exp_inner.e.get_n_splits_inner()
        assert len(results) == n_expected

    def test_get_output_not_found(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        assert oc.get_output('dt', 99, 99) is None

    def test_get_outputs_node_not_found(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        assert oc.get_outputs('nonexistent') == {}

    def test_reset_nodes(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        assert oc.has_node('dt')
        oc.reset_nodes(['dt'])
        assert not oc.has_node('dt')

    def test_save_load(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        loaded = OutputCollector.load(oc.path)
        assert loaded.has_node('dt')
        result_orig = oc.get_output('dt', 0, 0)
        result_loaded = loaded.get_output('dt', 0, 0)
        np.testing.assert_array_equal(result_orig['output_valid'],
                                      result_loaded['output_valid'])

    def test_saved_nodes(self, multi_head_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        multi_head_exp.e.add_collector(oc)
        multi_head_exp.e.collect(oc)
        saved = oc._get_saved_nodes()
        assert 'dt1' in saved
        assert 'dt2' in saved


class TestCollectorWithExperimenter:
    def test_collect_skip_existing(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        metric_before = mc.get_metric('dt').copy()
        built_exp.e.collect(mc, exist='skip')
        metric_after = mc.get_metric('dt')
        pd.testing.assert_series_equal(metric_before, metric_after)

    def test_experimenter_save_load_with_collectors(self, built_exp, sample_data):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        path = built_exp.e.path

        loaded = Experimenter.load(path, sample_data)
        assert loaded.get_collector('acc') is not None
        loaded_mc = loaded.get_collector('acc')
        assert loaded_mc.has('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded_mc.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_reset_nodes_clears_collectors(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        assert mc.has('dt')
        built_exp.e.reset_nodes(['dt'])
        assert not mc.has('dt')

    def test_multiple_collectors(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        oc = OutputCollector('out', Connector(), output_var=None)
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        built_exp.e.add_collector(oc)
        built_exp.e.collect(oc)
        built_exp.e.add_collector(mac)
        built_exp.e.collect(mac)
        assert mc.has('dt')
        assert oc.has_node('dt')
        assert mac.has('dt')


class TestSHAPCollector:
    @pytest.fixture(autouse=True)
    def skip_if_no_shap(self):
        pytest.importorskip('shap')

    def _make_sc(self, exp):
        from mllabs import SHAPCollector
        sc = SHAPCollector('shap', Connector(processor=DecisionTreeClassifier))
        exp.e.add_collector(sc)
        exp.e.collect(sc)
        return sc

    def test_collect_basic(self, built_exp):
        sc = self._make_sc(built_exp)
        assert sc.has_node('dt')

    def test_get_feature_importance_returns_list(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance('dt', 0)
        assert isinstance(result, list)
        assert len(result) == 1  # no inner split → 1 inner fold

    def test_get_feature_importance_series_structure(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance('dt', 0)
        s = result[0]
        assert isinstance(s, pd.Series)
        assert len(s.index) == 3
        assert (s >= 0).all()

    def test_get_feature_importance_inner_order(self, built_exp_inner):
        sc = self._make_sc(built_exp_inner)
        result = sc.get_feature_importance('dt', 0)
        assert len(result) == 3  # KFold n_splits=3
        assert [s.name for s in result] == [0, 1, 2]

    def test_get_feature_importance_agg_default_returns_series(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt')
        assert isinstance(result, pd.Series)
        assert len(result.index) == 3
        assert (result >= 0).all()

    def test_get_feature_importance_agg_outer_none_returns_dataframe(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt', agg_outer=None)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)  # 3 features x 2 outer folds

    def test_get_feature_importance_agg_inner_none_multiindex(self, built_exp_inner):
        sc = self._make_sc(built_exp_inner)
        result = sc.get_feature_importance_agg('dt', agg_inner=None)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.shape == (3, 6)  # 3 features x (2 outer * 3 inner)

    def test_get_feature_importance_agg_callable(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt', agg_inner=np.mean, agg_outer=np.mean)
        assert isinstance(result, pd.Series)

    def test_reset_nodes(self, built_exp):
        sc = self._make_sc(built_exp)
        assert sc.has_node('dt')
        sc.reset_nodes(['dt'])
        assert not sc.has_node('dt')

    def test_save_load(self, built_exp):
        sc = self._make_sc(built_exp)
        from mllabs import SHAPCollector
        loaded = SHAPCollector.load(sc.path)
        assert loaded.has_node('dt')
        orig = sc.get_feature_importance_agg('dt')
        loaded_result = loaded.get_feature_importance_agg('dt')
        pd.testing.assert_series_equal(orig, loaded_result)


class TestBaseCollector:
    def test_get_nodes_none(self):
        c = MetricCollector('test', Connector(), output_var=None, metric_func=dummy_metric)
        result = c._get_nodes(None, ['a', 'b', 'c'])
        assert result == ['a', 'b', 'c']

    def test_get_nodes_list(self):
        c = MetricCollector('test', Connector(), output_var=None, metric_func=dummy_metric)
        result = c._get_nodes(['a', 'c'], ['a', 'b', 'c'])
        assert result == ['a', 'c']

    def test_get_nodes_list_filter(self):
        c = MetricCollector('test', Connector(), output_var=None, metric_func=dummy_metric)
        result = c._get_nodes(['a', 'x'], ['a', 'b', 'c'])
        assert result == ['a']

    def test_get_nodes_regex(self):
        c = MetricCollector('test', Connector(), output_var=None, metric_func=dummy_metric)
        result = c._get_nodes('dt', ['dt1', 'dt2', 'scaler'])
        assert result == ['dt1', 'dt2']

    def test_get_nodes_invalid_type(self):
        c = MetricCollector('test', Connector(), output_var=None, metric_func=dummy_metric)
        with pytest.raises((ValueError, TypeError)):
            c._get_nodes(123, ['a', 'b'])


class TestCollectorErrorHandling:
    @pytest.fixture
    def pre_exp(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_pre')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_pre',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        return Built(e=e, p=p)

    def _make_broken_collector(self):
        from mllabs.collector._base import Collector

        class BrokenCollector(Collector):
            def collect(self, context):
                raise RuntimeError("collect error")

        return BrokenCollector('broken', Connector())

    def test_exp_warning_contains_traceback(self, pre_exp):
        bc = self._make_broken_collector()
        pre_exp.e.add_collector(bc)
        pre_exp.e.exp()
        w = bc.warnings[0]
        assert 'traceback' in w
        assert 'RuntimeError' in w['traceback']
        assert 'collect error' in w['traceback']

    def test_exp_continues_other_collectors_after_error(self, pre_exp):
        bc = self._make_broken_collector()
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=accuracy_metric)
        pre_exp.e.add_collector(bc)
        pre_exp.e.add_collector(mc)
        pre_exp.e.exp()
        assert mc.has('dt')
        assert len(bc.warnings) > 0

class TestProcessCollector:
    @pytest.fixture
    def ext_data(self, sample_data):
        return sample_data.iloc[:20].reset_index(drop=True)

    def test_collect_basic(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        assert pc.has_node('dt')

    def test_get_output_shape(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_get_output_nodes_none(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output(nodes=None)
        assert isinstance(result, pd.DataFrame)

    def test_get_output_nodes_list(self, multi_head_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        multi_head_exp.e.add_collector(pc)
        multi_head_exp.e.collect(pc)
        result_dt1 = pc.get_output(nodes=['dt1'])
        result_all = pc.get_output(nodes=None)
        assert isinstance(result_dt1, pd.DataFrame)
        assert result_dt1.shape[1] < result_all.shape[1]

    def test_get_output_nodes_regex(self, multi_head_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        multi_head_exp.e.add_collector(pc)
        multi_head_exp.e.collect(pc)
        result = pc.get_output(nodes='dt1')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_with_upstream_stage(self, built_exp, ext_data):
        # built_exp: scaler(stage) -> dt(head), ext_data goes through scaler first
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output()
        assert len(result) == 20

    def test_agg_mean(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data, method='mean')
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output(agg='mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_agg_mode(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output(agg='mode')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_agg_simple(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data, method='simple')
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        result = pc.get_output(agg='simple')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_with_inner_splits(self, built_exp_inner, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp_inner.e.add_collector(pc)
        built_exp_inner.e.collect(pc)
        result = pc.get_output()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_multi_head_columns_concat(self, multi_head_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        multi_head_exp.e.add_collector(pc)
        multi_head_exp.e.collect(pc)
        result_all = pc.get_output(nodes=None)
        result_dt1 = pc.get_output(nodes=['dt1'])
        result_dt2 = pc.get_output(nodes=['dt2'])
        assert result_all.shape[1] == result_dt1.shape[1] + result_dt2.shape[1]

    def test_connector_filter(self, multi_head_exp, ext_data):
        pc = ProcessCollector('proc', Connector(node_query=['dt1']), ext_data=ext_data)
        multi_head_exp.e.add_collector(pc)
        multi_head_exp.e.collect(pc)
        assert pc.has_node('dt1')
        assert not pc.has_node('dt2')

    def test_reset_nodes(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        assert pc.has_node('dt')
        pc.reset_nodes(['dt'])
        assert not pc.has_node('dt')

    def test_get_saved_nodes(self, multi_head_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        multi_head_exp.e.add_collector(pc)
        multi_head_exp.e.collect(pc)
        saved = pc._get_saved_nodes()
        assert 'dt1' in saved
        assert 'dt2' in saved

    def test_save_load(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        loaded = ProcessCollector.load(pc.path)
        assert loaded.has_node('dt')
        result_orig = pc.get_output()
        result_loaded = loaded.get_output()
        pd.testing.assert_frame_equal(result_orig, result_loaded)

    def test_invalid_agg(self, built_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        built_exp.e.add_collector(pc)
        built_exp.e.collect(pc)
        with pytest.raises(ValueError):
            pc.get_output(agg='invalid')

    @pytest.fixture
    def proba_exp(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_proba')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict_proba',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_proba',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        e.exp()
        return Built(e=e, p=p)

    def test_output_var_none_returns_all_columns(self, proba_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data, output_var=None)
        proba_exp.e.add_collector(pc)
        proba_exp.e.collect(pc)
        result = pc.get_output()
        assert result.shape == (20, 2)

    def test_output_var_list_selects_column(self, proba_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data, output_var='{dt__target_0}')
        proba_exp.e.add_collector(pc)
        proba_exp.e.collect(pc)
        result = pc.get_output()
        assert list(result.columns) == ['dt__target_0']
        assert result.shape == (20, 1)

    def test_output_var_regex_selects_column(self, proba_exp, ext_data):
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data, output_var='dt__target_1')
        proba_exp.e.add_collector(pc)
        proba_exp.e.collect(pc)
        result = pc.get_output()
        assert list(result.columns) == ['dt__target_1']
        assert result.shape == (20, 1)


class TestFinalizedBeforeCollect:
    """Collector query methods return None/{} when node was finalized before collect."""

    @pytest.fixture
    def finalized_exp(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_fin')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_fin',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        e.exp()
        e.finalize(['dt'])
        return Built(e=e, p=p)

    def test_metric_get_metric_returns_none(self, finalized_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        finalized_exp.e.add_collector(mc)
        finalized_exp.e.collect(mc)
        assert mc.get_metric('dt') is None

    def test_metric_get_metrics_returns_none(self, finalized_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        finalized_exp.e.add_collector(mc)
        finalized_exp.e.collect(mc)
        assert mc.get_metrics() is None

    def test_model_attr_get_attr_returns_none(self, finalized_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                 result_key='feature_importances',
                                 adapter=DecisionTreeAdapter())
        finalized_exp.e.add_collector(mac)
        finalized_exp.e.collect(mac)
        assert mac.get_attr('dt') is None

    def test_model_attr_get_attrs_agg_returns_none(self, finalized_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                 result_key='feature_importances',
                                 adapter=DecisionTreeAdapter())
        finalized_exp.e.add_collector(mac)
        finalized_exp.e.collect(mac)
        assert mac.get_attrs_agg('dt') is None

    def test_output_get_output_returns_none(self, finalized_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        finalized_exp.e.add_collector(oc)
        finalized_exp.e.collect(oc)
        assert oc.get_output('dt', 0, 0) is None

    def test_output_get_outputs_returns_empty(self, finalized_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        finalized_exp.e.add_collector(oc)
        finalized_exp.e.collect(oc)
        assert oc.get_outputs('dt') == {}

    def test_process_get_output_returns_none(self, finalized_exp, sample_data):
        ext_data = sample_data.iloc[:20].reset_index(drop=True)
        pc = ProcessCollector('proc', Connector(), ext_data=ext_data)
        finalized_exp.e.add_collector(pc)
        finalized_exp.e.collect(pc)
        assert pc.get_output() is None

    def test_shap_get_feature_importance_agg_returns_none(self, finalized_exp):
        pytest.importorskip('shap')
        from mllabs import SHAPCollector
        sc = SHAPCollector('shap', Connector(processor=DecisionTreeClassifier))
        finalized_exp.e.add_collector(sc)
        finalized_exp.e.collect(sc)
        assert sc.get_feature_importance_agg('dt') is None


class TestGetCollectStatus:

    @pytest.fixture
    def not_exp_exp(self, tmp_path, sample_data):
        """Stage built but head exp() not called."""
        p = Pipeline(path=tmp_path / 'pipeline_notexp')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_notexp',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        return Built(e=e, p=p)

    @pytest.fixture
    def finalized_exp(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_fin2')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_fin2',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        e.exp()
        e.finalize(['dt'])
        return Built(e=e, p=p)

    @pytest.fixture
    def error_exp(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_err')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=FailPredictor,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'})
        p.set_node('dt', grp='model')
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_err',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        e.exp()
        return Built(e=e, p=p)

    def test_status_collected(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        assert built_exp.e.get_collect_status(mc)['dt'] == 'collected'

    def test_status_not_collected(self, not_exp_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        not_exp_exp.e.add_collector(mc)
        not_exp_exp.e.collect(mc)
        assert not_exp_exp.e.get_collect_status(mc)['dt'] == 'not_collected'

    def test_status_finalized(self, finalized_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        finalized_exp.e.add_collector(mc)
        finalized_exp.e.collect(mc)
        assert finalized_exp.e.get_collect_status(mc)['dt'] == 'finalized'

    def test_status_error(self, error_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        error_exp.e.add_collector(mc)
        error_exp.e.collect(mc)
        assert error_exp.e.get_collect_status(mc)['dt'] == 'error'

    def test_by_collector_name(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect(mc)
        assert built_exp.e.get_collect_status('acc')['dt'] == 'collected'

    def test_nodes_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        status = multi_head_exp.e.get_collect_status(mc, nodes=['dt1'])
        assert 'dt1' in status
        assert 'dt2' not in status

    def test_connector_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(node_query=['dt1']),
                             output_var=None, metric_func=dummy_metric)
        multi_head_exp.e.add_collector(mc)
        multi_head_exp.e.collect(mc)
        status = multi_head_exp.e.get_collect_status(mc)
        assert 'dt1' in status
        assert 'dt2' not in status

    def test_mixed_status(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_mixed')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt1', grp='model')
        p.set_node('dt2', grp='model', params={'max_depth': 5})
        exp_obj = Experimenter(data=sample_data, path=tmp_path / 'exp_mixed',
                               sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e = exp_obj
        e.set_pipeline(p)
        e.build()
        e.exp('dt1')
        e.exp('dt2')
        e.finalize(['dt2'])
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        e.add_collector(mc)
        e.collect(mc)
        status = e.get_collect_status(mc)
        assert status['dt1'] == 'collected'
        assert status['dt2'] == 'finalized'


class TestCollectMissing:

    def test_collects_not_collected_node(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc)
        result = built_exp.e.collect_missing(mc)
        assert result == {'acc': ['dt']}
        assert built_exp.e.get_collect_status(mc)['dt'] == 'collected'

    def test_skips_already_collected(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc)
        built_exp.e.collect_missing(mc)
        result = built_exp.e.collect_missing(mc)
        assert result == {'acc': []}

    def test_all_collectors_when_none_given(self, built_exp):
        mc1 = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        mc2 = MetricCollector('acc2', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc1)
        built_exp.e.add_collector(mc2)
        result = built_exp.e.collect_missing()
        assert result == {'acc': ['dt'], 'acc2': ['dt']}
        assert built_exp.e.get_collect_status(mc1)['dt'] == 'collected'
        assert built_exp.e.get_collect_status(mc2)['dt'] == 'collected'

    def test_by_collector_name(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        built_exp.e.add_collector(mc)
        result = built_exp.e.collect_missing('acc')
        assert result == {'acc': ['dt']}

    def test_nodes_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        multi_head_exp.e.add_collector(mc)
        result = multi_head_exp.e.collect_missing(mc, nodes=['dt1'])
        assert result == {'acc': ['dt1']}
        status = multi_head_exp.e.get_collect_status(mc)
        assert status['dt1'] == 'collected'
        assert status['dt2'] == 'not_collected'

    def test_excludes_finalized_and_error(self, tmp_path, sample_data):
        p = Pipeline(path=tmp_path / 'pipeline_cm_mixed')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
        p.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt_ok', grp='model')
        p.set_node('dt_finalized', grp='model')
        p.set_grp('bad_model', role='head', processor=FailPredictor,
                  method='predict',
                  edges={'X': '{f1, f2, f3}', 'y': '{target}'})
        p.set_node('dt_error', grp='bad_model')
        e = Experimenter(data=sample_data, path=tmp_path / 'exp_cm_mixed',
                         sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42))
        e.set_pipeline(p)
        e.build()
        e.exp()
        e.finalize(['dt_finalized'])

        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=dummy_metric)
        e.add_collector(mc)
        result = e.collect_missing(mc)
        assert result == {'acc': ['dt_ok']}
        status = e.get_collect_status(mc)
        assert status['dt_ok'] == 'collected'
        assert status['dt_finalized'] == 'finalized'
        assert status['dt_error'] == 'error'
