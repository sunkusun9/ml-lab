import pytest
import numpy as np
import pandas as pd
from collections import namedtuple

from sklearn.model_selection import ShuffleSplit, KFold

from mllabs import (
    Project, Trial, Connector, ProcessorSpec,
    MetricCollector, StackingCollector, ModelAttrCollector, OutputCollector,
    ProcessCollector, ProbToLabel,
)

TREE = 'sklearn.tree.DecisionTreeClassifier'
EDGES = {'X': 'scaler:(*)', 'y': '{target}'}

Built = namedtuple('Built', ['project', 'e', 'trial'])
MultiBuilt = namedtuple('MultiBuilt', ['project', 'e', 'trial1', 'trial2'])


def accuracy_metric(y, pred):
    return (y.values == pred.values).mean()


def dummy_metric(y, pred):
    return 0.5


def boom_metric(y, pred):
    raise ValueError('boom')


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


def _pipeline_version(project):
    """A single-stage Pipeline (StandardScaler) — the model itself is a Trial,
    not a pipeline node, since Pipeline is stage-only under the current
    Project/Trial split."""
    p = project.pipeline
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1, f2, f3}'})
    p.set_node('scaler', grp='scale')
    return p.build().version


def _register(trial, e):
    """Register *trial* and return the name — exp() takes names, reads the
    definition out of the store, and covers every fold of *e* itself.

    Stamped with the run's own pipeline version, which Project.set_trial would
    otherwise do: exp() refuses a Trial defined against another version, and
    these tests are about Collectors."""
    if trial.pipeline_version is None:
        trial.pipeline_version = e.pipeline_version
    e.trial_store.register(trial)
    return trial.name


def _ext(built, data, name='ext'):
    """Registers *data* into *built*'s Project.ext_data and returns the
    '@ext:name' reference a Collector's params should hold instead of the
    live object — set_collector's _validate_params gate now rejects a raw
    DataFrame there, same rule PipelineBuilder enforces for node params."""
    built.project.ext_data.register(name, data)
    return f'@ext:{name}'


def _run(built, *trials, collectors=None, n_jobs=1):
    """Runs every fold of the given trials (default: the fixture's own).

    *collectors* is a list of names out of the run's own registry, or None for
    every one registered on it — which is what most tests want, since each
    registers exactly the Collector it is testing into a fresh fixture.

    Collection happens during exp() dispatch only — a fold already recorded
    'built' in TrialStore.experiment_hist is skipped without dispatch, so a
    Collector attached to an already-exp()'d Trial never sees it. Fixtures
    below therefore only build() the Stage graph; each test runs its own
    Trial(s) through exp() together with whatever Collector it's testing.
    """
    trials = trials or (built.trial,)
    names = [_register(t, built.e) for t in trials]
    built.e.exp(names, collectors=collectors, n_jobs=n_jobs)


@pytest.fixture
def built_exp(tmp_path, sample_data):
    """Stage built; Trial 'dt' defined but not yet exp()'d."""
    project = Project(tmp_path / 'proj_built', data=sample_data)
    version = _pipeline_version(project)
    e = project.add_experimenter('exp_built',
                             sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), pipeline_version=version)
    e.build()
    trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 42})
    return Built(project=project, e=e, trial=trial)


@pytest.fixture
def built_exp_inner(tmp_path, sample_data):
    """Same as built_exp, plus an inner CV split (KFold, 3 folds)."""
    project = Project(tmp_path / 'proj_inner', data=sample_data)
    version = _pipeline_version(project)
    e = project.add_experimenter('exp_inner',
                             sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
                             sp_v=KFold(n_splits=3, shuffle=True, random_state=42), pipeline_version=version)
    e.build()
    trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 42})
    return Built(project=project, e=e, trial=trial)


@pytest.fixture
def multi_head_exp(tmp_path, sample_data):
    """Two Trials ('dt1', 'dt2') reading the same stage, neither exp()'d yet."""
    project = Project(tmp_path / 'proj_multi', data=sample_data)
    version = _pipeline_version(project)
    e = project.add_experimenter('exp_multi',
                             sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), pipeline_version=version)
    e.build()
    trial1 = Trial('dt1', TREE, EDGES, params={'max_depth': 3, 'random_state': 42})
    trial2 = Trial('dt2', TREE, EDGES, params={'max_depth': 5, 'random_state': 42})
    return MultiBuilt(project=project, e=e, trial1=trial1, trial2=trial2)


class TestConnector:
    """Connector matches against a ProcessorSpec now, not a plain attrs dict."""

    @staticmethod
    def _spec(name, processor=None, edges=None):
        return ProcessorSpec(name=name, processor=processor, edges=edges or {})

    def test_match_all(self):
        c = Connector()
        assert c.match(self._spec('any_node')) is True

    def test_match_node_query_str(self):
        c = Connector(node_query='dt')
        assert c.match(self._spec('dt1')) is True
        assert c.match(self._spec('scaler')) is False

    def test_match_node_query_regex(self):
        c = Connector(node_query='^dt')
        assert c.match(self._spec('dt1')) is True
        assert c.match(self._spec('my_dt')) is False

    def test_match_node_query_list(self):
        c = Connector(node_query=['dt1', 'dt2'])
        assert c.match(self._spec('dt1')) is True
        assert c.match(self._spec('dt3')) is False

    def test_match_processor(self):
        # Connector.processor is a "module.ClassName" string, compared
        # directly (string equality) against spec.processor, which
        # PipelineBuilder/Trial also always store as that same string form.
        c = Connector(processor='sklearn.tree.DecisionTreeClassifier')
        assert c.match(self._spec('dt', 'sklearn.tree.DecisionTreeClassifier')) is True
        assert c.match(self._spec('dt', 'sklearn.preprocessing.StandardScaler')) is False

    def test_match_edges(self):
        c = Connector(edges={'X': '{f1}'})
        assert c.match(self._spec('dt', edges={'X': '{f1}', 'y': '{target}'})) is True

    def test_match_edges_different_value(self):
        c = Connector(edges={'X': '{f1}'})
        assert c.match(self._spec('dt', edges={'X': '{f1, f2}'})) is False

    def test_match_edges_missing_key(self):
        c = Connector(edges={'z': '{f1}'})
        assert c.match(self._spec('dt', edges={'X': '{f1}'})) is False

    def test_match_combined(self):
        c = Connector(node_query='dt', processor='sklearn.tree.DecisionTreeClassifier')
        assert c.match(self._spec('dt1', 'sklearn.tree.DecisionTreeClassifier')) is True
        assert c.match(self._spec('dt1', 'sklearn.preprocessing.StandardScaler')) is False
        assert c.match(self._spec('scaler', 'sklearn.tree.DecisionTreeClassifier')) is False


class TestMetricCollector:
    def test_collect_basic(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        assert mc.has_node('dt')

    def test_get_metric(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        result = mc.get_metric('dt')
        assert isinstance(result, pd.Series)
        assert result.name == 'dt'
        assert len(result) > 0
        assert all(0 <= v <= 1 for v in result.values)

    def test_get_metrics(self, multi_head_exp):
        mc = multi_head_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[mc.name])
        result = mc.get_metrics()
        assert isinstance(result, pd.DataFrame)
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' in result.index.get_level_values(0)

    def test_get_metrics_with_node_filter(self, multi_head_exp):
        mc = multi_head_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[mc.name])
        result = mc.get_metrics(nodes=['dt1'])
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' not in result.index.get_level_values(0)

    def test_get_metrics_regex(self, multi_head_exp):
        mc = multi_head_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[mc.name])
        result = mc.get_metrics(nodes='dt1')
        assert len(result) > 0

    def test_get_metrics_agg(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        mean, std = mc.get_metrics_agg()
        assert isinstance(mean, pd.DataFrame)
        assert std is None

    def test_get_metrics_agg_with_std(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        mean, std = mc.get_metrics_agg(include_std=True)
        assert isinstance(mean, pd.DataFrame)
        assert isinstance(std, pd.DataFrame)

    def test_get_metrics_agg_inner_only(self, built_exp_inner):
        mc = built_exp_inner.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp_inner, collectors=[mc.name])
        mean, std = mc.get_metrics_agg(inner_fold=True, outer_fold=False)
        assert isinstance(mean, pd.DataFrame)

    def test_get_metrics_agg_no_fold(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        result = mc.get_metrics_agg(inner_fold=False, outer_fold=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_metrics_agg_invalid(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        with pytest.raises(ValueError):
            mc.get_metrics_agg(inner_fold=False, outer_fold=True)

    def test_include_train(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc_train', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}, 'include_train': True})
        _run(built_exp, collectors=[mc.name])
        result = mc.get_metric('dt')
        assert 'train' in result.index.get_level_values(-1)

    def test_inner_split_metrics(self, built_exp_inner):
        mc = built_exp_inner.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp_inner, collectors=[mc.name])
        result = mc.get_metric('dt')
        assert len(result) > 2

    def test_connector_filter(self, multi_head_exp):
        mc = multi_head_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector',
            {'__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': ['dt1']}},
            params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[mc.name])
        assert mc.has_node('dt1')
        assert not mc.has_node('dt2')

    def test_reset_nodes(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        assert mc.has_node('dt')
        mc.reset_nodes(['dt'])
        assert not mc.has_node('dt')

    def test_save_load(self, built_exp):
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        loaded = built_exp.e.collectors.get_collector('acc')
        assert loaded.has_node('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_second_collector_runs_on_rerun(self, built_exp):
        """A Collector attached after a Trial is already 'built' does not see
        it retroactively — collection only happens during exp() dispatch, and
        ``_make_jobs`` skips a fold purely on ``TrialStore.experiment_hist``
        status, not on-disk state. So ``Experimenter.reset_nodes`` (which only
        touches NodeStore/cache) is not enough to force a rerun by itself —
        the hist row also has to go; ``_make_jobs`` then resets the NodeStore
        entry itself once it decides a job is needed."""
        collectors = built_exp.e.collectors
        mc = collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        assert mc.has_node('dt')

        mc2 = collectors.set_collector(
            'acc2', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.dummy_metric'}})
        built_exp.project.trials.remove_hist(trial_name='dt', experimenter=built_exp.e.name)
        _run(built_exp, collectors=[mc2.name])
        assert mc2.has_node('dt')
        result = mc2.get_metric('dt')
        assert all(v == 0.5 for v in result.values)


class TestProbToLabel:
    """var is a DSL string (e.g. '{target}'), resolved via Experimenter.get_test_data —
    same lazy-resolution path as MetricCollector.output_var."""

    def test_on_attach_sets_classes(self, built_exp):
        ptl = ProbToLabel(dummy_metric, '{target}')
        ptl.on_attach(built_exp.e)
        assert list(ptl._classes) == [0, 1]

    def test_convert_binary_argmax(self, built_exp):
        ptl = ProbToLabel(dummy_metric, '{target}')
        ptl.on_attach(built_exp.e)
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8]])
        assert list(ptl._convert(y_prob)) == [0, 1]

    def test_call_uses_metric_func(self, built_exp):
        calls = {}

        def capture_metric(y_true, y_pred):
            calls['y_true'] = y_true
            calls['y_pred'] = y_pred
            return 0.75

        ptl = ProbToLabel(capture_metric, '{target}')
        ptl.on_attach(built_exp.e)
        y_true = np.array([0, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8]])
        assert ptl(y_true, y_prob) == 0.75
        assert list(calls['y_pred']) == [0, 1]


class TestStackingCollector:
    def test_collect_basic(self, built_exp):
        sc = built_exp.e.collectors.set_collector('stk', 'mllabs.StackingCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        assert sc.has_node('dt')

    def test_get_dataset(self, built_exp):
        sc = built_exp.e.collectors.set_collector(
            'stk', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        ds = sc.get_dataset(built_exp.e)
        assert isinstance(ds, pd.DataFrame)
        assert len(ds) == int(len(built_exp.e.data.data) * 0.2) * 2  # ShuffleSplit, n_splits = 2, test_size = 0.2
        assert 'target' in ds.columns

    def test_get_dataset_no_target(self, built_exp):
        sc = built_exp.e.collectors.set_collector('stk', 'mllabs.StackingCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        ds = sc.get_dataset(built_exp.e, include_target=False)
        assert isinstance(ds, pd.DataFrame)
        assert 'target' not in ds.columns

    def test_get_dataset_multi_nodes(self, multi_head_exp):
        sc = multi_head_exp.e.collectors.set_collector(
            'stk', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[sc.name])
        ds = sc.get_dataset(multi_head_exp.e)
        assert ds.shape[1] > 2

    def test_get_dataset_node_filter(self, multi_head_exp):
        sc = multi_head_exp.e.collectors.set_collector(
            'stk', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[sc.name])
        ds = sc.get_dataset(multi_head_exp.e, nodes=['dt1'])
        assert isinstance(ds, pd.DataFrame)

    def test_method_mean(self, built_exp_inner):
        sc = built_exp_inner.e.collectors.set_collector(
            'stk_mean', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None, 'method': 'mean'})
        _run(built_exp_inner, collectors=[sc.name])
        assert sc.has_node('dt')

    def test_reset_nodes(self, built_exp):
        sc = built_exp.e.collectors.set_collector('stk', 'mllabs.StackingCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        assert sc.has_node('dt')
        sc.reset_nodes(['dt'])
        assert not sc.has_node('dt')

    def test_save_load(self, built_exp):
        sc = built_exp.e.collectors.set_collector(
            'stk', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        loaded = built_exp.e.collectors.get_collector('stk')
        assert loaded.has_node('dt')
        ds_orig = sc.get_dataset(built_exp.e)
        ds_loaded = loaded.get_dataset(built_exp.e)
        pd.testing.assert_frame_equal(ds_orig, ds_loaded)

    def test_index_preserved(self, built_exp):
        sc = built_exp.e.collectors.set_collector(
            'stk', 'mllabs.StackingCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'edges': {'y': '{target}'}}}, params={'output_var': None})
        _run(built_exp, collectors=[sc.name])
        ds = sc.get_dataset(built_exp.e)
        all_valid_idx = np.concatenate([
            built_exp.e.outer_folds[i].test_idx
            for i in range(built_exp.e.get_n_splits())
        ])
        expected_index = built_exp.e.data.data.index[all_valid_idx]
        pd.testing.assert_index_equal(ds.index, expected_index)


class TestModelAttrCollector:
    def test_collect_basic(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        assert mac.has_node('dt')

    def test_get_attr(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        result = mac.get_attr('dt')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_attr_idx(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        result = mac.get_attr('dt', idx=0)
        assert isinstance(result, list)

    def test_get_attrs(self, multi_head_exp):
        mac = multi_head_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[mac.name])
        result = mac.get_attrs()
        assert 'dt1' in result
        assert 'dt2' in result

    def test_get_attrs_agg(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        result = mac.get_attrs_agg('dt')
        assert isinstance(result, pd.Series)

    def test_get_attrs_agg_inner_only(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        result = mac.get_attrs_agg('dt', agg_inner=True, agg_outer=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_attrs_agg_invalid(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        with pytest.raises(ValueError):
            mac.get_attrs_agg('dt', agg_inner=False, agg_outer=True)

    def test_not_mergeable(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'tree', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'tree', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        with pytest.raises(ValueError, match='not mergeable'):
            mac.get_attrs_agg('dt')

    def test_reset_nodes(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        mac.reset_nodes(['dt'])
        assert not mac.has_node('dt')

    def test_save_load(self, built_exp):
        mac = built_exp.e.collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mac.name])
        loaded = built_exp.e.collectors.get_collector('fi')
        assert loaded.has_node('dt')

    def test_auto_adapter(self):
        mac = ModelAttrCollector('fi', Connector(processor='sklearn.tree.DecisionTreeClassifier'),
                                result_key='feature_importances')
        assert mac.adapter is not None

    def test_auto_adapter_invalid_key(self):
        with pytest.raises(RuntimeError):
            ModelAttrCollector('fi', Connector(processor='sklearn.tree.DecisionTreeClassifier'),
                               result_key='nonexistent_key')


class TestOutputCollector:
    def test_collect_basic(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        assert oc.has_node('dt')

    def test_get_output(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        result = oc.get_output('dt', 0, 0)
        assert 'output_test' in result
        assert 'output_train' in result
        assert 'columns' in result

    def test_get_output_structure(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        result = oc.get_output('dt', 0, 0)
        assert isinstance(result['output_test'], np.ndarray)
        assert result['output_train'] is None or isinstance(result['output_train'], np.ndarray)
        assert result['output_valid'] is None or isinstance(result['output_valid'], np.ndarray)

    def test_get_outputs(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        results = oc.get_outputs('dt')
        assert isinstance(results, dict)
        assert len(results) == built_exp.e.get_n_splits() * built_exp.e.get_n_splits_inner()
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_get_outputs_inner_split(self, built_exp_inner):
        oc = built_exp_inner.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp_inner, collectors=[oc.name])
        results = oc.get_outputs('dt')
        n_expected = built_exp_inner.e.get_n_splits() * built_exp_inner.e.get_n_splits_inner()
        assert len(results) == n_expected

    def test_get_output_not_found(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        assert oc.get_output('dt', 99, 99) is None

    def test_get_outputs_node_not_found(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        assert oc.get_outputs('nonexistent') == {}

    def test_reset_nodes(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        assert oc.has_node('dt')
        oc.reset_nodes(['dt'])
        assert not oc.has_node('dt')

    def test_save_load(self, built_exp):
        oc = built_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(built_exp, collectors=[oc.name])
        loaded = built_exp.e.collectors.get_collector('out')
        assert loaded.has_node('dt')
        result_orig = oc.get_output('dt', 0, 0)
        result_loaded = loaded.get_output('dt', 0, 0)
        np.testing.assert_array_equal(result_orig['output_valid'],
                                      result_loaded['output_valid'])

    def test_saved_nodes(self, multi_head_exp):
        oc = multi_head_exp.e.collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[oc.name])
        saved = oc._get_saved_nodes()
        assert 'dt1' in saved
        assert 'dt2' in saved


class TestCollectorWithExperimenter:
    def test_collect_skip_existing(self, built_exp):
        """A fold already recorded 'built' is skipped by exp() without
        dispatch, so a second exp() call with the same collector leaves its
        result unchanged — this is exp()'s own skip logic, not a separate
        exist='skip' collect() call (which no longer exists)."""
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        metric_before = mc.get_metric('dt').copy()
        _run(built_exp, collectors=[mc.name])
        metric_after = mc.get_metric('dt')
        pd.testing.assert_series_equal(metric_before, metric_after)

    def test_collectors_reload_from_the_store(self, built_exp):
        """Collector state lives in the project's Collectors registry, not the
        Experimenter. Registration writes through, so there is nothing to save."""
        collectors = built_exp.e.collectors
        mc = collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])

        reloaded = built_exp.e.collectors
        loaded_mc = reloaded.get_collector('acc')
        assert loaded_mc is not None
        assert loaded_mc.has_node('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded_mc.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_experimenter_reset_nodes_does_not_clear_collectors(self, built_exp):
        """Collectors are no longer owned by Experimenter — they live in the
        project's separate Collectors registry — so Experimenter.reset_nodes
        (NodeStore + cache only) has no way to cascade into one. Clearing a
        Collector's own state is a separate, explicit reset_nodes call on it."""
        mc = built_exp.e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp, collectors=[mc.name])
        assert mc.has_node('dt')
        built_exp.e.reset_nodes(['dt'])
        assert mc.has_node('dt')
        mc.reset_nodes(['dt'])
        assert not mc.has_node('dt')

    def test_multiple_collectors(self, built_exp):
        collectors = built_exp.e.collectors
        mc = collectors.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        oc = collectors.set_collector('out', 'mllabs.OutputCollector', 'mllabs._connector.Connector', params={'output_var': None})
        mac = collectors.set_collector(
            'fi', 'mllabs.ModelAttrCollector', {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}},
            params={'result_key': 'feature_importances', 'adapter': {'__ref__': 'mllabs.adapter._sklearn.DecisionTreeAdapter'}})
        _run(built_exp, collectors=[mc.name, oc.name, mac.name])
        assert mc.has_node('dt')
        assert oc.has_node('dt')
        assert mac.has_node('dt')


class TestSHAPCollector:
    @pytest.fixture(autouse=True)
    def skip_if_no_shap(self):
        pytest.importorskip('shap')

    def _make_sc(self, built):
        from mllabs import SHAPCollector
        sc = built.e.collectors.set_collector(
            'shap', 'mllabs.SHAPCollector',
            {'__ref__': 'mllabs._connector.Connector', '__params__': {'processor': 'sklearn.tree.DecisionTreeClassifier'}})
        _run(built, collectors=[sc.name])
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
        loaded = built_exp.e.collectors.get_collector('shap')
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
    """A failing Collector is recorded, never raised: it must not take down an
    experiment whose Trials already ran, and in a worker the exception object
    never reaches the parent — only what the hist row carries."""

    def test_collect_error_is_recorded_with_its_traceback(self, built_exp):
        collectors = built_exp.e.collectors
        collectors.set_collector('broken', 'mock.BrokenCollector', 'mllabs._connector.Connector')
        _run(built_exp)

        rows = collectors.hist.get_hist(collector_name='broken')
        assert rows and all(r['status'] == 'error' for r in rows)
        info = rows[0]['info']
        assert info['phase'] == 'collect'
        assert info['type'] == 'RuntimeError'
        assert 'collect error' in info['traceback']

    def test_exp_continues_other_collectors_after_error(self, built_exp):
        collectors = built_exp.e.collectors
        collectors.set_collector('broken', 'mock.BrokenCollector', 'mllabs._connector.Connector')
        mc = collectors.set_collector('acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector', params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        _run(built_exp)

        assert mc.has_node('dt')
        assert collectors.hist.get_hist(collector_name='broken', status='error')
        assert collectors.hist.get_hist(collector_name='acc', status='collected')

    def test_push_failure_is_recorded_as_its_own_phase(self, built_exp):
        """A Collector whose collect() works but whose store step blows up —
        the fold that triggered the flush is the one that carries the error."""
        collectors = built_exp.e.collectors
        collectors.set_collector('badpush', 'mock.BrokenPushCollector', 'mllabs._connector.Connector')
        _run(built_exp)

        rows = collectors.hist.get_hist(collector_name='badpush')
        assert rows and all(r['status'] == 'error' for r in rows)
        assert rows[0]['info']['phase'] == 'push'


class TestCollectHist:
    """One row per (collector, node, outer, inner) — the fold keys are what
    make an error locatable, which is the whole point of recording an outcome
    the Collector itself no longer keeps. No experimenter key: the hist
    belongs to one run's registry, so the run is the store it is in."""

    @staticmethod
    def _folds(e):
        return {(o, i) for o in range(e.get_n_splits())
                for i in range(e.get_n_splits_inner())}

    def test_a_row_per_collector_and_fold(self, built_exp_inner):
        collectors = built_exp_inner.e.collectors
        collectors.set_collector('count', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp_inner)

        rows = collectors.hist.get_hist(collector_name='count')
        assert {(r['outer_idx'], r['inner_idx']) for r in rows} == self._folds(built_exp_inner.e)
        assert all(r['node_name'] == 'dt' and r['status'] == 'collected' for r in rows)

    def test_row_carries_the_stamp_of_the_run(self, built_exp):
        collectors = built_exp.e.collectors
        collectors.set_collector('count', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp)

        row = collectors.hist.get_hist(collector_name='count')[0]
        assert 'experimenter' not in row
        assert row['pipeline_version'] == built_exp.e.pipeline_version
        assert row['collect_date'] is not None
        assert row['elapsed'] >= 0

    def test_the_hist_lives_in_the_run_directory(self, built_exp):
        """Which run a row came from is answered by where the db is, not by a
        column — the same way node_hist needs no run_name."""
        assert built_exp.e.collectors.hist.db_path.parent == built_exp.e.path / 'collectors'

    def test_collecting_nothing_is_not_an_error(self, built_exp):
        """The base Collector returns None — that used to be indistinguishable
        from a failed collect, since both produced None."""
        collectors = built_exp.e.collectors
        collectors.set_collector('noop', 'mllabs.Collector', 'mllabs._connector.Connector')
        _run(built_exp)

        rows = collectors.hist.get_hist(collector_name='noop')
        assert rows and all(r['status'] == 'empty' and r['info'] is None for r in rows)

    def test_only_matched_collectors_get_rows(self, built_exp):
        collectors = built_exp.e.collectors
        collectors.set_collector('count', 'mock.CountingCollector',
                                 {'__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': '^dt'}})
        collectors.set_collector('other', 'mock.CountingCollector',
                                 {'__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': '^nope'}})
        _run(built_exp)

        assert collectors.hist.get_hist(collector_name='count')
        assert collectors.hist.get_hist(collector_name='other') == []

    def test_get_status_is_keyed_by_node_and_fold(self, built_exp_inner):
        collectors = built_exp_inner.e.collectors
        collectors.set_collector('count', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp_inner)

        status = collectors.hist.get_status('count')
        assert set(status) == {('dt', o, i) for o, i in self._folds(built_exp_inner.e)}
        assert set(status.values()) == {'collected'}

    def test_multi_worker_records_the_same_rows(self, built_exp):
        """_run_collectors runs in the worker, so the outcome has to travel
        back over the pipe — only the parent ever writes the hist."""
        collectors = built_exp.e.collectors
        collectors.set_collector('count', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp, n_jobs=2)

        rows = collectors.hist.get_hist(collector_name='count')
        assert {(r['outer_idx'], r['inner_idx']) for r in rows} == self._folds(built_exp.e)
        assert all(r['status'] == 'collected' for r in rows)

    def test_multi_worker_records_a_collect_error(self, built_exp):
        collectors = built_exp.e.collectors
        collectors.set_collector('broken', 'mock.BrokenCollector', 'mllabs._connector.Connector')
        _run(built_exp, n_jobs=2)

        rows = collectors.hist.get_hist(collector_name='broken')
        assert rows and all(r['status'] == 'error' for r in rows)
        assert 'collect error' in rows[0]['info']['traceback']

    def test_a_named_subset_is_recorded_too(self, built_exp):
        """Narrowing to a list is a choice about this call, not about where the
        history goes — that is the run's, whichever form is passed."""
        collectors = built_exp.e.collectors
        c = collectors.set_collector('count', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp, collectors=[c.name])

        assert collectors.hist.get_hist(collector_name='count')


class TestCollectorSelection:
    """exp(collectors=) names Collectors on this run's registry — the registry
    is the run's, so anything it does not know has no place here to write."""

    def test_none_uses_every_collector_registered_on_the_run(self, built_exp):
        c1 = built_exp.e.collectors.set_collector('c1', 'mock.CountingCollector', 'mllabs._connector.Connector')
        c2 = built_exp.e.collectors.set_collector('c2', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp)

        hist = built_exp.e.collectors.hist
        assert hist.get_hist(collector_name=c1.name)
        assert hist.get_hist(collector_name=c2.name)

    def test_a_name_selects_only_that_one(self, built_exp):
        built_exp.e.collectors.set_collector('c1', 'mock.CountingCollector', 'mllabs._connector.Connector')
        built_exp.e.collectors.set_collector('c2', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp, collectors=['c1'])

        hist = built_exp.e.collectors.hist
        assert hist.get_hist(collector_name='c1')
        assert hist.get_hist(collector_name='c2') == []

    def test_empty_list_collects_nothing(self, built_exp):
        built_exp.e.collectors.set_collector('c1', 'mock.CountingCollector', 'mllabs._connector.Connector')
        _run(built_exp, collectors=[])

        assert built_exp.e.collectors.hist.get_hist() == []

    def test_an_unregistered_name_raises(self, built_exp):
        """A silent miss reads exactly like 'this Collector produced nothing'."""
        with pytest.raises(KeyError, match='nope'):
            _run(built_exp, collectors=['nope'])

    def test_an_instance_is_rejected(self, built_exp):
        c = built_exp.e.collectors.set_collector('c1', 'mock.CountingCollector', 'mllabs._connector.Connector')
        with pytest.raises(TypeError, match='names'):
            _run(built_exp, collectors=[c])


class TestRegistryIsPerRun:
    """Two runs of the same Trial name must not overwrite each other's results.

    Everything a Collector writes is keyed by node name and nothing more, so
    the registry's path is the only thing keeping two runs apart — which is
    why the registry belongs to an Experimenter and not to the Project.
    """

    @staticmethod
    def _run_one(project, name, version, trial):
        e = project.add_experimenter(
            name,
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), pipeline_version=version)
        e.build()
        mc = e.collectors.set_collector(
            'acc', 'mllabs.MetricCollector', 'mllabs._connector.Connector',
            params={'output_var': None, 'metric_func': {'__callable__': 'test_collector.accuracy_metric'}})
        e.exp([_register(trial, e)])
        return e, mc

    @pytest.fixture
    def two_runs(self, tmp_path, sample_data):
        project = Project(tmp_path / 'proj_two', data=sample_data)
        version = _pipeline_version(project)
        trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 42})
        a = self._run_one(project, 'run_a', version, trial)
        b = self._run_one(project, 'run_b', version, trial)
        return project, a, b

    def test_each_run_stores_under_its_own_directory(self, two_runs):
        _, (e_a, mc_a), (e_b, mc_b) = two_runs
        assert mc_a.path != mc_b.path
        assert mc_a.path.is_relative_to(e_a.path)
        assert mc_b.path.is_relative_to(e_b.path)
        assert e_a.collectors.hist.db_path != e_b.collectors.hist.db_path

    def test_the_same_trial_name_is_collected_by_both(self, two_runs):
        _, (_, mc_a), (_, mc_b) = two_runs
        assert mc_a.get_metric('dt') is not None
        assert mc_b.get_metric('dt') is not None
        assert len(mc_a.get_metric('dt')) == len(mc_b.get_metric('dt'))

    def test_each_run_records_its_own_hist(self, two_runs):
        _, (e_a, _), (e_b, _) = two_runs
        rows_a = e_a.collectors.hist.get_hist(collector_name='acc')
        rows_b = e_b.collectors.hist.get_hist(collector_name='acc')
        assert rows_a and len(rows_a) == len(rows_b)

    def test_removing_from_one_run_leaves_the_other(self, two_runs):
        """The overwrite this arrangement prevents, seen from the other end:
        one run dropping its result must not reach into the other's."""
        _, (e_a, mc_a), (e_b, mc_b) = two_runs
        e_a.remove_trial_result('dt')

        assert mc_a.get_metric('dt') is None
        assert e_a.collectors.hist.get_hist(node_name='dt') == []
        assert mc_b.get_metric('dt') is not None
        assert e_b.collectors.hist.get_hist(node_name='dt')

    def test_removing_the_result_reruns_the_trial(self, two_runs):
        """History is the only gate on a fold, so dropping it is what makes the
        next exp() dispatch the Trial again."""
        project, (e_a, mc_a), _ = two_runs
        trial = Trial('dt', TREE, EDGES, params={'max_depth': 3, 'random_state': 42})
        e_a.remove_trial_result('dt')

        assert project.trials.get_status('dt', e_a.name) == {}
        e_a.exp([_register(trial, e_a)])
        assert mc_a.get_metric('dt') is not None


class TestProcessCollector:
    @pytest.fixture
    def ext_data(self, sample_data):
        return sample_data.iloc[:20].reset_index(drop=True)

    def test_collect_basic(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        assert pc.has_node('dt')

    def test_get_output_shape(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_get_output_nodes_none(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output(nodes=None)
        assert isinstance(result, pd.DataFrame)

    def test_get_output_nodes_list(self, multi_head_exp, ext_data):
        pc = multi_head_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(multi_head_exp, ext_data)})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[pc.name])
        result_dt1 = pc.get_output(nodes=['dt1'])
        result_all = pc.get_output(nodes=None)
        assert isinstance(result_dt1, pd.DataFrame)
        assert result_dt1.shape[1] < result_all.shape[1]

    def test_get_output_nodes_regex(self, multi_head_exp, ext_data):
        pc = multi_head_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(multi_head_exp, ext_data)})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[pc.name])
        result = pc.get_output(nodes='dt1')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_with_upstream_stage(self, built_exp, ext_data):
        # built_exp: scaler(stage) -> dt(Trial), ext_data goes through scaler first
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output()
        assert len(result) == 20

    def test_agg_mean(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data), 'method': 'mean'})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output(agg='mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_agg_mode(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output(agg='mode')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_agg_simple(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data), 'method': 'simple'})
        _run(built_exp, collectors=[pc.name])
        result = pc.get_output(agg='simple')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_with_inner_splits(self, built_exp_inner, ext_data):
        pc = built_exp_inner.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp_inner, ext_data)})
        _run(built_exp_inner, collectors=[pc.name])
        result = pc.get_output()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_multi_head_columns_concat(self, multi_head_exp, ext_data):
        pc = multi_head_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(multi_head_exp, ext_data)})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[pc.name])
        result_all = pc.get_output(nodes=None)
        result_dt1 = pc.get_output(nodes=['dt1'])
        result_dt2 = pc.get_output(nodes=['dt2'])
        assert result_all.shape[1] == result_dt1.shape[1] + result_dt2.shape[1]

    def test_connector_filter(self, multi_head_exp, ext_data):
        pc = multi_head_exp.e.collectors.set_collector(
            'proc', 'mllabs.ProcessCollector',
            {'__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': ['dt1']}},
            params={'ext_data': _ext(multi_head_exp, ext_data)})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[pc.name])
        assert pc.has_node('dt1')
        assert not pc.has_node('dt2')

    def test_reset_nodes(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        assert pc.has_node('dt')
        pc.reset_nodes(['dt'])
        assert not pc.has_node('dt')

    def test_get_saved_nodes(self, multi_head_exp, ext_data):
        pc = multi_head_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(multi_head_exp, ext_data)})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2, collectors=[pc.name])
        saved = pc._get_saved_nodes()
        assert 'dt1' in saved
        assert 'dt2' in saved

    def test_save_load(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        loaded = built_exp.e.collectors.get_collector('proc')
        assert loaded.has_node('dt')
        result_orig = pc.get_output()
        result_loaded = loaded.get_output()
        pd.testing.assert_frame_equal(result_orig, result_loaded)

    def test_invalid_agg(self, built_exp, ext_data):
        pc = built_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(built_exp, ext_data)})
        _run(built_exp, collectors=[pc.name])
        with pytest.raises(ValueError):
            pc.get_output(agg='invalid')

    @pytest.fixture
    def proba_exp(self, tmp_path, sample_data):
        project = Project(tmp_path / 'proj_proba', data=sample_data)
        version = _pipeline_version(project)
        e = project.add_experimenter('exp_proba',
                                 sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), pipeline_version=version)
        e.build()
        trial = Trial('dt', TREE, EDGES, method='predict_proba', params={'max_depth': 3, 'random_state': 42})
        return Built(project=project, e=e, trial=trial)

    def test_output_var_none_returns_all_columns(self, proba_exp, ext_data):
        pc = proba_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(proba_exp, ext_data), 'output_var': None})
        _run(proba_exp, collectors=[pc.name])
        result = pc.get_output()
        assert result.shape == (20, 2)

    def test_output_var_list_selects_column(self, proba_exp, ext_data):
        pc = proba_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(proba_exp, ext_data), 'output_var': '{dt__target_0}'})
        _run(proba_exp, collectors=[pc.name])
        result = pc.get_output()
        assert list(result.columns) == ['dt__target_0']
        assert result.shape == (20, 1)

    def test_output_var_regex_selects_column(self, proba_exp, ext_data):
        pc = proba_exp.e.collectors.set_collector('proc', 'mllabs.ProcessCollector', 'mllabs._connector.Connector', params={'ext_data': _ext(proba_exp, ext_data), 'output_var': 'dt__target_1'})
        _run(proba_exp, collectors=[pc.name])
        result = pc.get_output()
        assert list(result.columns) == ['dt__target_1']
        assert result.shape == (20, 1)


class TestCollectState:
    """Reading what was collected — the one question that spans stores."""

    def _acc(self, e, name='acc', connector=None, metric=accuracy_metric):
        return e.collectors.set_collector(
            name, 'mllabs.MetricCollector', connector or 'mllabs._connector.Connector',
            params={'output_var': None,
                    'metric_func': {'__callable__': f'test_collector.{metric.__name__}'}})

    def test_uncollected_empty_when_everything_collected(self, built_exp):
        mc = self._acc(built_exp.e)
        _run(built_exp, collectors=[mc.name])
        assert built_exp.e.uncollected_trials() == {'acc': []}

    def test_uncollected_reports_collector_attached_after_the_run(self, built_exp):
        """The case the history is there to expose: folds already recorded
        'built' are skipped, so a Collector attached afterwards never sees them
        and exp() reports nothing wrong."""
        _run(built_exp, collectors=[])
        self._acc(built_exp.e)
        assert built_exp.e.uncollected_trials() == {'acc': ['dt']}

    def test_uncollected_ignores_a_trial_that_never_ran(self, built_exp):
        """Not run is not a collection failure — that is pending_trials. Getting
        this wrong invites remove_trial_result on history that was fine."""
        self._acc(built_exp.e)
        _register(built_exp.trial, built_exp.e)
        assert built_exp.e.uncollected_trials() == {'acc': []}

    def test_uncollected_ignores_a_trial_the_connector_skips(self, multi_head_exp):
        mc = self._acc(multi_head_exp.e, connector={
            '__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': 'dt1'}})
        _run(multi_head_exp, multi_head_exp.trial1, multi_head_exp.trial2,
             collectors=[mc.name])
        assert multi_head_exp.e.uncollected_trials() == {'acc': []}

    def test_uncollected_counts_a_failed_collect(self, built_exp):
        """An error row and a missing row mean the same thing here — nothing
        was kept — so both land in the list."""
        mc = self._acc(built_exp.e, name='boom', metric=boom_metric)
        _run(built_exp, collectors=[mc.name])
        assert built_exp.e.uncollected_trials() == {'boom': ['dt']}

    def test_uncollected_selects_by_name(self, built_exp):
        good = self._acc(built_exp.e)
        bad = self._acc(built_exp.e, name='boom', metric=boom_metric)
        _run(built_exp, collectors=[good.name, bad.name])
        assert built_exp.e.uncollected_trials('boom') == {'boom': ['dt']}

    def test_collect_errors_rows(self, built_exp):
        """The failure is merged in flat — phase says which of the four points
        broke, and nothing has to be dug out of a nested info."""
        mc = self._acc(built_exp.e, name='boom', metric=boom_metric)
        _run(built_exp, collectors=[mc.name])
        rows = built_exp.e.collect_errors()
        assert len(rows) == built_exp.e.get_n_splits()
        assert {r['collector_name'] for r in rows} == {'boom'}
        assert rows[0]['node_name'] == 'dt'
        assert rows[0]['phase'] == 'collect'
        assert rows[0]['type'] == 'ValueError'
        assert rows[0]['message'] == 'boom'

    def test_collect_errors_empty_when_clean(self, built_exp):
        mc = self._acc(built_exp.e)
        _run(built_exp, collectors=[mc.name])
        assert built_exp.e.collect_errors() == []

    def test_project_collect_errors_adds_the_experimenter(self, built_exp):
        """collect_hist has no experimenter column on purpose — which one it is
        gets answered by whose db it is, so the fan-out supplies the key."""
        mc = self._acc(built_exp.e, name='boom', metric=boom_metric)
        _run(built_exp, collectors=[mc.name])
        rows = built_exp.project.collect_errors()
        assert len(rows) == built_exp.e.get_n_splits()
        assert rows[0]['experimenter'] == 'exp_built'
        assert rows[0]['collector_name'] == 'boom'
        assert rows[0]['phase'] == 'collect'

    def test_project_collect_errors_empty_when_clean(self, built_exp):
        mc = self._acc(built_exp.e)
        _run(built_exp, collectors=[mc.name])
        assert built_exp.project.collect_errors() == []

    def test_project_collect_errors_by_experimenter(self, built_exp):
        mc = self._acc(built_exp.e, name='boom', metric=boom_metric)
        _run(built_exp, collectors=[mc.name])
        assert built_exp.project.collect_errors('exp_built')
        assert built_exp.project.collect_errors('exp_built', collectors='boom')

    def test_project_uncollected_trials_is_nested_by_experimenter(self, built_exp):
        _run(built_exp, collectors=[])
        self._acc(built_exp.e)
        assert built_exp.project.uncollected_trials() == {'exp_built': {'acc': ['dt']}}
