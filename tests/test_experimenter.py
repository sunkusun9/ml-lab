import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs._cache import DataCache
from mllabs._store import NodeStore
from mllabs._flow import DataFlow
from mllabs._pipeline import PipelineBuilder
from mllabs import Connector, MetricCollector


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
def pipeline():
    p = PipelineBuilder()
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    return p


@pytest.fixture
def exp(tmp_path, sample_data, pipeline):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        pipeline=pipeline.build(),
    )
    return e


def _publish(pipeline, exp):
    """Hand the current definitions to *exp* as a fresh snapshot.

    Experimenter holds a built Pipeline, so edits made to the builder after
    construction only take effect once they are built and set again.
    """
    if exp is not None:
        exp.set_pipeline(pipeline.build())


def _setup_stage(pipeline, exp=None):
    pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                      method='transform', edges={'X': '{f1, f2, f3}'})
    pipeline.set_node('scaler', grp='scale')
    _publish(pipeline, exp)


def _setup_head(pipeline, exp=None):
    pipeline.set_grp('model', processor='sklearn.tree.DecisionTreeClassifier',
                      method='predict', edges={'X': '{f1, f2, f3}', 'y': '{target}'},
                      params={'max_depth': 3, 'random_state': 42})
    pipeline.set_node('dt', grp='model')
    _publish(pipeline, exp)


def _setup_full(pipeline, exp=None):
    _setup_stage(pipeline)
    _setup_head(pipeline)
    _publish(pipeline, exp)


def _flow(exp, outer=0, inner=0):
    return exp.outer_folds[outer].train_data_flows[inner]

def _store(exp, outer=0, inner=0):
    return exp.outer_folds[outer].train_data_flows[inner]


class TestDataCache:
    def test_put_get(self):
        c = DataCache(maxsize=1024**3)
        data = np.array([1, 2, 3])
        c.put_data('node1', 0, 0, 'train', data)
        result = c.get_data('node1', 0, 0, 'train')
        assert np.array_equal(result, data)

    def test_get_missing(self):
        c = DataCache(maxsize=1024**3)
        assert c.get_data('no_exist', 0, 0, 'train') is None

    def test_clear_nodes(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 0, 0, 'train', np.array([1]))
        c.put_data('b', 0, 0, 'train', np.array([2]))
        c.clear_nodes(['a'])
        assert c.get_data('a', 0, 0, 'train') is None
        assert c.get_data('b', 0, 0, 'train') is not None

    def test_clear(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 0, 0, 'train', np.array([1]))
        c.clear()
        assert c.get_data('a', 0, 0, 'train') is None


class TestExperimenterInit:
    def test_path_created(self, exp):
        assert exp.path.exists()

    def test_data_wrapped(self, exp):
        assert exp.data is not None

    def test_splits_created(self, exp):
        assert exp.get_n_splits() == 2
        assert len(exp.outer_folds) == 2
        assert all(of.test_idx is not None for of in exp.outer_folds)

    def test_no_inner_split(self, exp):
        assert exp.get_n_splits_inner() == 1
        flow = _flow(exp)
        assert flow.data_source.valid_idx is None

    def test_with_inner_split(self, tmp_path, sample_data):
        e = Experimenter(
            data=sample_data,
            path=tmp_path / 'exp_inner',
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        assert e.get_n_splits() == 2
        assert e.get_n_splits_inner() == 3
        flow = e.outer_folds[0].train_data_flows[0]
        assert flow.data_source.valid_idx is not None

    def test_create_path_exists(self, tmp_path, sample_data):
        path = tmp_path / 'existing'
        path.mkdir()
        with pytest.raises(RuntimeError):
            Experimenter.create(data=sample_data, path=path)

    def test_status_open(self, exp):
        assert exp.status == 'open'

    def test_pipeline_empty(self, exp, pipeline):
        user_grps = {k: v for k, v in pipeline.grps.items() if not k.startswith('__')}
        assert len(user_grps) == 0

    def test_data_key(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk', data_key='test_key')
        assert e.data_key == 'test_key'


class TestBuild:
    def test_build_stage(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'

    def test_build_skips_built(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        build_id = flow.get_info('scaler')['build_id']
        exp.build()
        assert flow.get_info('scaler')['build_id'] == build_id

    def test_build_error_continues(self, exp, pipeline):
        pipeline.set_grp('good', processor='sklearn.preprocessing.StandardScaler',
                             method='transform', edges={'X': '{f1}'})
        pipeline.set_node('good_node', grp='good')
        pipeline.set_grp('bad', processor='mock.BadProcessor',
                             method='transform', edges={'X': '{f2}'})
        pipeline.set_node('bad_node', grp='bad')
        _publish(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('good_node') == 'built'
        assert flow.status('bad_node') == 'error'

    def test_build_error_dict(self, exp, pipeline):
        pipeline.set_grp('err', processor='mock.ErrorProcessor',
                             method='transform', edges={'X': '{f1}'})
        pipeline.set_node('err_node', grp='err')
        _publish(pipeline, exp)
        exp.build()
        info = _flow(exp).get_info('err_node')
        err = info['error']
        assert err['type'] == 'TypeError'
        assert 'test error msg' in err['message']
        assert 'traceback' in err

    def test_unknown_column_in_edges_errors_at_build_not_definition(self, exp, pipeline):
        # set_grp only validates DSL structure — an edges string referencing a
        # column that doesn't exist in the schema is accepted at definition
        # time and only surfaces as a node-level error once build() actually
        # resolves it against real data.
        pipeline.set_grp('bad', processor='sklearn.preprocessing.StandardScaler',
                          method='transform', edges={'X': '{unknown_col}'})
        pipeline.set_node('bad_node', grp='bad')
        _publish(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('bad_node') == 'error'
        assert 'Unknown column' in flow.get_info('bad_node')['error']['message']

    def test_build_info_contains_node_serial(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        expected_serial = pipeline.nodes['scaler'].serial
        info = _flow(exp).get_info('scaler')
        assert info['node_serial'] == expected_serial

    def test_build_error_info_contains_node_serial(self, exp, pipeline):
        pipeline.set_grp('err', processor='mock.ErrorProcessor',
                             method='transform', edges={'X': '{f1}'})
        pipeline.set_node('err_node', grp='err')
        _publish(pipeline, exp)
        exp.build()
        expected_serial = pipeline.nodes['err_node'].serial
        info = _flow(exp).get_info('err_node')
        assert info['node_serial'] == expected_serial


class TestExp:
    def test_exp_head(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'

    def test_exp_skips_built(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        store = _store(exp)
        build_id = store.get_info('dt')['build_id']
        exp.exp()
        assert store.get_info('dt')['build_id'] == build_id

    def test_exp_error(self, exp, pipeline):
        pipeline.set_grp('bad_model', processor='mock.BadPredictor',
                             method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pipeline.set_node('bad_dt', grp='bad_model')
        _publish(pipeline, exp)
        exp.exp()
        assert exp.get_status('bad_dt') == 'error'

    def test_exp_with_collector(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        mc = exp.set_collector(
            'acc', MetricCollector, Connector(),
            params={'output_var': None, 'metric_func': accuracy_metric},
        )
        exp.exp()
        assert mc.has_node('dt')

    def test_set_collector_resolves_callable_metric(self, exp, pipeline):
        from sklearn.metrics import balanced_accuracy_score
        _setup_full(pipeline, exp)
        exp.build()
        mc = exp.set_collector(
            'bacc', MetricCollector, Connector(),
            params={'output_var': None,
                    'metric_func': {'__callable__': 'sklearn.metrics.balanced_accuracy_score'}},
        )
        assert mc.metric_func is balanced_accuracy_score
        exp.exp()
        assert mc.has_node('dt')


class TestCollectorManagement:
    def test_set_collector(self, exp, pipeline):
        mc = exp.set_collector('acc', MetricCollector, Connector(),
                               params={'output_var': None, 'metric_func': dummy_metric})
        assert exp.get_collector('acc') is not None
        assert mc.path is not None

    def test_set_collector_skip(self, exp, pipeline):
        mc1 = exp.set_collector('acc', MetricCollector, Connector(),
                                params={'output_var': None, 'metric_func': dummy_metric})
        result = exp.set_collector('acc', MetricCollector, Connector(),
                                   params={'output_var': None, 'metric_func': dummy_metric},
                                   exist='skip')
        assert result is mc1

    def test_set_collector_error(self, exp, pipeline):
        exp.set_collector('acc', MetricCollector, Connector(),
                          params={'output_var': None, 'metric_func': dummy_metric})
        with pytest.raises(RuntimeError):
            exp.set_collector('acc', MetricCollector, Connector(),
                              params={'output_var': None, 'metric_func': dummy_metric},
                              exist='error')


class TestResetNodes:
    def test_reset_removes_node_dir(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        node_path = flow._node_path('scaler')
        assert node_path.exists()
        exp.reset_nodes(['scaler'])
        assert not node_path.exists()
        assert flow.status('scaler') is None

    def test_reset_clears_cache(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        exp.cache.put_data('scaler', 0, 0, 'train', np.array([1]))
        exp.reset_nodes(['scaler'])
        assert exp.cache.get_data('scaler', 0, 0, 'train') is None

    def test_pipeline_set_node_replace_then_build_resets(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('scaler') == 'built'
        pipeline.set_node('scaler', grp='scale', exist='replace')
        # artifact still on disk until build() triggers serial mismatch reset
        exp.build()
        assert flow.status('scaler') == 'built'


class TestRebuild:
    def test_build_with_rebuild_true(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        exp.build(rebuild=True)
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert flow.status('scaler') == 'built'
        assert old_obj is not new_obj

    def test_set_node_replace_then_build(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        pipeline.set_node('scaler', grp='scale', exist='replace')
        _publish(pipeline, exp)
        exp.build()
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert new_obj is not old_obj
        assert _flow(exp).status('scaler') == 'built'

    def test_exp_rebuilds_non_built_node(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'
        exp.reset_nodes(['dt'])
        assert exp.get_status('dt') is None
        exp.exp()
        assert exp.get_status('dt') == 'built'

    def test_build_auto_resets_on_serial_mismatch_stage(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        old_build_id = _flow(exp).get_info('scaler')['build_id']
        # Simulate pipeline node change by bumping serial directly
        pipeline._bump_serials(['scaler'])
        _publish(pipeline, exp)
        exp.build()
        new_build_id = _flow(exp).get_info('scaler')['build_id']
        assert new_build_id != old_build_id

    def test_build_skips_when_serial_matches_stage(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        build_id = _flow(exp).get_info('scaler')['build_id']
        exp.build()  # serial unchanged — should skip
        assert _flow(exp).get_info('scaler')['build_id'] == build_id

    def test_exp_auto_resets_on_serial_mismatch_head(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        old_build_id = exp.outer_folds[0].train_data_flows[0].get_info('dt')['build_id']
        pipeline._bump_serials(['dt'])
        _publish(pipeline, exp)
        exp.exp()
        new_build_id = exp.outer_folds[0].train_data_flows[0].get_info('dt')['build_id']
        assert new_build_id != old_build_id

    def test_exp_skips_when_serial_matches_head(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        build_id = exp.outer_folds[0].train_data_flows[0].get_info('dt')['build_id']
        exp.exp()  # serial unchanged, already built — should skip
        assert exp.outer_folds[0].train_data_flows[0].get_info('dt')['build_id'] == build_id


class TestStateManagement:
    def test_open_close(self, exp):
        exp.close()
        assert exp.status == 'close'
        exp.open()
        assert exp.status == 'open'

    def test_finalize_head(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        assert exp.get_status('dt') == 'finalized'

    def test_reinitialize(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        exp.reinitialize(['dt'])
        assert exp.get_status('dt') is None

    def test_reopen_exp_status(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        exp.close_exp()
        assert exp.status == 'closed'
        exp.reopen_exp()
        assert exp.status == 'open'

    def test_reopen_exp_collector_data_valid(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        mc = exp.set_collector('acc', MetricCollector, Connector(edges={'y': '{target}'}),
                               params={'output_var': None, 'metric_func': accuracy_metric})
        exp.exp()
        assert mc.has_node('dt')
        first_result = mc.get_metrics_agg(None)[0]

        exp.close_exp()
        exp.reopen_exp()
        exp.exp()

        assert mc.has_node('dt')
        second_result = mc.get_metrics_agg(None)[0]
        assert second_result.shape == first_result.shape

    def test_reset_nodes_clears_collector_sub(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        mc = exp.set_collector('acc', MetricCollector, Connector(),
                               params={'output_var': None, 'metric_func': accuracy_metric})
        exp.exp()

        mc._buf['dt'] = [{'valid': 0.9}]
        exp.reset_nodes(['dt'])

        assert 'dt' not in mc._buf

    def test_close_exp_saves_status(self, exp, pipeline, sample_data):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        exp.close_exp()

        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.status == 'closed'

    def test_reopen_exp_after_save_load(self, exp, pipeline, sample_data):
        _setup_full(pipeline, exp)
        exp.build()
        mc = exp.set_collector('acc', MetricCollector, Connector(edges={'y': '{target}'}),
                               params={'output_var': None, 'metric_func': accuracy_metric})
        exp.exp()
        first_result = mc.get_metrics_agg(None)[0]
        exp.close_exp()

        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.status == 'closed'
        loaded.reopen_exp()
        assert loaded.status == 'open'
        loaded.exp()

        mc2 = loaded.get_collector('acc')
        assert mc2.has_node('dt')
        second_result = mc2.get_metrics_agg(None)[0]
        assert second_result.shape == first_result.shape


class TestSetPipeline:
    def test_no_pipeline_raises_on_build(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'no_pipeline')
        with pytest.raises(RuntimeError, match='set_pipeline'):
            e.build()

    def test_no_pipeline_raises_on_exp(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'no_pipeline')
        with pytest.raises(RuntimeError, match='set_pipeline'):
            e.exp()

    def test_constructor_pipeline_sets_attribute(self, exp, pipeline):
        from mllabs._pipeline import Pipeline
        assert isinstance(exp.pipeline, Pipeline)
        assert exp.pipeline.pipeline_id == pipeline.pipeline_id

    def test_builder_is_rejected(self, tmp_path, sample_data, pipeline):
        with pytest.raises(TypeError, match='built Pipeline'):
            Experimenter(data=sample_data, path=tmp_path / 'reject', pipeline=pipeline)

    def test_set_pipeline_persists_pkl(self, exp, pipeline):
        assert (exp.path / 'pipeline.pkl').exists()

    def test_set_pipeline_resets_stale_nodes(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('scaler') == 'built'
        pipeline.set_node('scaler', grp='scale', exist='replace')
        exp.set_pipeline(pipeline.build())
        assert flow.status('scaler') is None


class TestSaveLoad:
    def test_save_creates_file(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        assert (exp.path / '__exp.db').exists()

    def test_load_restores(self, exp, pipeline, sample_data):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        path = exp.path

        loaded = Experimenter.load(path, sample_data)
        assert set(pipeline.grps.keys()) == set(pipeline.grps.keys())
        flow = loaded.outer_folds[0].train_data_flows[0]
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'

    def test_load_restores_pipeline(self, exp, pipeline, sample_data):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()

        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.pipeline is not None
        assert 'scaler' in loaded.pipeline.nodes
        assert 'dt' in loaded.pipeline.nodes
        loaded.exp()  # uses restored pipeline without needing set_pipeline again
        assert loaded.get_status('dt') == 'built'

    def test_load_data_key_mismatch(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk', data_key='key_a')
        with pytest.raises(ValueError, match='data_key'):
            Experimenter.load(tmp_path / 'dk', sample_data, data_key='key_b')

    def test_load_preserves_splits(self, exp, pipeline, sample_data):
        _setup_stage(pipeline, exp)
        path = exp.path
        loaded = Experimenter.load(path, sample_data)
        assert loaded.get_n_splits() == exp.get_n_splits()
        assert loaded.get_n_splits_inner() == exp.get_n_splits_inner()

    def test_persistence_layout(self, exp, pipeline):
        _setup_stage(pipeline, exp)
        assert (exp.path / '__exp.db').exists()
        assert (exp.path / '__splitters.pkl').exists()
        assert not (exp.path / '__exp.pkl').exists()

    def test_meta_table_holds_only_simple_values(self, exp):
        meta = exp._store.fetch_meta()
        assert meta['status'] == 'open'
        assert 'exp_id' in meta and 'status' in meta
        assert 'sp' not in meta and 'splitter_params' not in meta

    def test_load_restores_collector_via_ref(self, exp, pipeline, sample_data):
        _setup_full(pipeline, exp)
        exp.build()
        exp.set_collector('acc', MetricCollector, Connector(edges={'y': '{target}'}),
                          params={'output_var': None, 'metric_func': accuracy_metric})
        exp.exp()

        loaded = Experimenter.load(exp.path, sample_data)
        restored = loaded.get_collector('acc')
        assert type(restored) is MetricCollector
        assert restored.has_node('dt')

    def test_collectors_table_stores_class_ref(self, exp, pipeline):
        exp.set_collector('acc', MetricCollector, Connector(),
                          params={'output_var': None, 'metric_func': dummy_metric})
        collectors = exp._store.fetch_collectors()
        assert collectors['acc'] is MetricCollector

    def test_remove_collector_drops_db_row(self, exp, pipeline, sample_data):
        exp.set_collector('acc', MetricCollector, Connector(),
                          params={'output_var': None, 'metric_func': dummy_metric})
        exp.remove_collector('acc')
        assert 'acc' not in exp._store.fetch_collectors()
        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.get_collector('acc') is None

    def test_set_status_updates_only_status_row(self, exp, sample_data):
        exp.title = 'mutated_not_saved'
        exp.set_status('closed')
        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.status == 'closed'
        assert loaded.title != 'mutated_not_saved'


class TestGetStatus:
    def test_get_status_none_before_exp(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        assert exp.get_status('dt') is None

    def test_get_status_built_after_exp(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'

    def test_get_status_finalized(self, exp, pipeline):
        _setup_full(pipeline, exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        assert exp.get_status('dt') == 'finalized'

    def test_get_status_error(self, exp, pipeline):
        pipeline.set_grp('bad_model', processor='mock.BadPredictor',
                             method='predict', edges={'X': '{f1}', 'y': '{target}'})
        pipeline.set_node('bad_dt', grp='bad_model')
        _publish(pipeline, exp)
        exp.exp()
        assert exp.get_status('bad_dt') == 'error'


class TestNodeStore:
    def test_write_objs_and_status(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, object(), np.array([1, 2]), {'build_id': 'x'})
        assert store.status('node1') == 'built'

    def test_get_objs(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        sc = StandardScaler()
        result = np.array([1.0, 2.0])
        NodeStore.write_objs(node_path, sc, result, {'build_id': 'abc'})
        got_obj, got_result, got_info = store.get_objs('node1')
        assert isinstance(got_obj, StandardScaler)
        assert np.array_equal(got_result, result)
        assert got_info['build_id'] == 'abc'
        assert got_info['status'] == 'built'

    def test_get_info(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {'build_id': 'xyz'})
        info = store.get_info('node1')
        assert info['status'] == 'built'
        assert info['build_id'] == 'xyz'

    def test_get_obj_get_result(self, tmp_path):
        store = NodeStore(tmp_path)
        sc = StandardScaler()
        result = np.array([3.0])
        NodeStore.write_objs(tmp_path / 'node1', sc, result, {})
        assert isinstance(store.get_obj('node1'), StandardScaler)
        assert np.array_equal(store.get_result('node1'), result)

    def test_status_none_when_missing(self, tmp_path):
        store = NodeStore(tmp_path)
        assert store.status('missing') is None

    def test_write_info_error_status(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        error_info = {
            'status': 'error',
            'build_id': 'e1',
            'error': {'type': 'ValueError', 'message': 'oops', 'traceback': '...'},
        }
        NodeStore.write_info(node_path, error_info)
        assert store.status('node1') == 'error'
        assert store.get_info('node1')['error']['type'] == 'ValueError'

    def test_finalize(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, None, None, {'build_id': 'y'})
        store.finalize('node1')
        assert store.status('node1') == 'finalized'
        assert not (node_path / 'obj.pkl').exists()
        assert not (node_path / 'result.pkl').exists()
        assert (node_path / 'info.pkl').exists()

    def test_reset_node(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, None, None, {})
        assert node_path.exists()
        store.reset_node('node1')
        assert not node_path.exists()
        assert store.status('node1') is None

    def test_info_cache_lazy(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {'build_id': 'cached'})
        info1 = store.get_info('node1')
        info2 = store.get_info('node1')
        assert info1 is info2

    def test_reset_clears_cache(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {})
        store.get_info('node1')  # populate cache
        store.reset_node('node1')
        assert store.status('node1') is None  # cache cleared, disk gone

    def test_dataflow_autoload(self, tmp_path):
        NodeStore.write_objs(tmp_path / 'node1', StandardScaler(), None, {'build_id': 'dl', 'edges': {'X': (None, ['X1', 'X2'])}})
        flow = DataFlow(tmp_path)
        assert 'node1' in flow.node_objs
        assert flow.status('node1') == 'built'
