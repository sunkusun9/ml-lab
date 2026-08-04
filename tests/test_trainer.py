import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from mllabs._trainer import Trainer
from mllabs._pipeline import PipelineBuilder
from mllabs import Trial, Predictor
from mllabs._cache import DataCache
from mllabs._data_wrapper import wrap


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
def pipeline(tmp_path):
    p = PipelineBuilder(path=tmp_path / 'pipeline')
    p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'f3': 'numerical', 'target': 'binary'})
    p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
              method='transform', edges={'X': '{f1, f2, f3}'})
    p.set_node('scaler', grp='scale')
    return p


DT = 'sklearn.tree.DecisionTreeClassifier'
DT_EDGES = {'X': 'scaler:(*)', 'y': '{target}'}


def _dt(name='dt', params=None, method='predict'):
    return Predictor(name, DT, DT_EDGES, method=method,
                     params=params if params is not None else {'max_depth': 3, 'random_state': 42})


def _bad_dt(name='bad_dt'):
    """Predictor reading the 'bad_node' node, so that node gets selected/trained."""
    return Predictor(name, DT, {'X': 'bad_node:(*)', 'y': '{target}'})


def _predictors(*predictors):
    return list(predictors) or [_dt()]


@pytest.fixture
def sp_v():
    return KFold(n_splits=3, shuffle=True, random_state=42)


def _trainer_path(pipeline, name):
    return pipeline._store.db_path.parent / '__trainers' / name


def _make_trainer(pipeline, sample_data, splitter, name='t1', path=None):
    if path is None:
        path = _trainer_path(pipeline, name)
    return Trainer(name=name, data=wrap(sample_data), path=path,
                   splitter=splitter, splitter_params={}, cache=DataCache())


def _build_ids(trainer, name):
    """Per-split ``build_id`` for *name*, from whichever store holds it.

    History is per-store now, so a Predictor's rows live in
    ``predictor_store`` and a node's in ``node_store`` — the flow itself
    carries no history.
    """
    info = trainer._store_for(name).get_info(name)
    return [info[(fold.split_idx, 0)]['build_id'] for fold in trainer.train_folds]


def _add_bad_node(pipeline):
    pipeline.set_grp('bad', processor='mock.BadProcessor', method='transform',
                     edges={'X': '{f1}'})
    pipeline.set_node('bad_node', grp='bad')


class TestTrainerConstruction:
    def test_no_splitter(self, pipeline, sample_data):
        trainer = _make_trainer(pipeline, sample_data, None)
        assert trainer.splitter is None
        assert trainer.get_n_splits() == 1

    def test_accepts_native_data(self, pipeline, sample_data):
        """``Project.trainer()`` passes the caller's DataFrame straight through,
        so the constructor has to wrap it — ``Experimenter`` already does."""
        trainer = Trainer(name='native', data=sample_data,
                          path=_trainer_path(pipeline, 'native'),
                          splitter=None, cache=DataCache())
        assert trainer.get_n_splits() == 1
        assert trainer.data.get_shape()[0] == len(sample_data)

    def test_accepts_native_data_with_splitter(self, pipeline, sample_data, sp_v):
        """The splitter path reads ``self.data.select_columns`` for
        splitter_params, so it needs the wrapper just as much."""
        trainer = Trainer(name='native_sp', data=sample_data,
                          path=_trainer_path(pipeline, 'native_sp'),
                          splitter=sp_v, splitter_params={'y': 'target'},
                          cache=DataCache())
        assert trainer.get_n_splits() == sp_v.get_n_splits()

    def test_accepts_already_wrapped_data(self, pipeline, sample_data):
        trainer = Trainer(name='wrapped', data=wrap(sample_data),
                          path=_trainer_path(pipeline, 'wrapped'),
                          splitter=None, cache=DataCache())
        assert trainer.data.get_shape()[0] == len(sample_data)

    def test_two_stores_at_separate_paths(self, pipeline, sample_data, sp_v):
        """Predictors get their own NodeStore — same class, own directory, so
        the two ``__node_hist.db`` files cannot collide."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        assert trainer.predictor_store.path == trainer.node_store.path / '__predictors'
        assert trainer.predictor_store.db_path != trainer.node_store.db_path


class TestSelection:
    def test_nodes_default_to_all_without_predictors(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        assert trainer.selected_nodes == ['scaler']
        assert trainer.predictor_names() == []

    def test_training_registers_the_predictors(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.predictor_names() == ['dt']

    def test_selection_collects_upstream_nodes(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('extra', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('unused', grp='extra')
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.selected_nodes == ['scaler']   # 'unused' is not referenced
        assert trainer.get_status('unused') is None   # ...so it is never trained

    def test_predictors_filter_by_name(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train([_dt('b')])
        assert trainer.predictor_names() == ['b']

    def test_predictor_specs_carry_the_definition(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        spec = trainer.predictor_specs()['dt']
        assert spec.processor == DT
        assert spec.edges == DT_EDGES
        assert spec.params == {'max_depth': 3, 'random_state': 42}

    def test_train_rejects_a_trial(self, pipeline, sample_data, sp_v):
        """A Trial must be promoted explicitly, so the provenance it came
        from is recorded rather than guessed."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        with pytest.raises(TypeError, match='from_trial'):
            trainer.train([Trial('dt', DT, DT_EDGES)])

    def test_no_pipeline_raises_on_train(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            trainer.train()

    def test_no_pipeline_raises_on_to_inferencer(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        with pytest.raises(RuntimeError, match='set_pipeline'):
            trainer.to_inferencer()

    def test_set_pipeline_keeps_a_copy_and_the_pointer(self, pipeline, sample_data, sp_v):
        """The Trainer owns the Pipeline it trains against; the pointer stays
        in its own store as provenance."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        built = pipeline.build()
        trainer.set_pipeline(built, pipeline_name='pipeline')
        assert (trainer.path / 'pipeline.pkl').exists()
        assert trainer._store.fetch()['pipeline_version'] == built.version

    def test_load_restores_the_pipeline_without_being_given_one(
            self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        built = pipeline.build()
        trainer.set_pipeline(built)
        trainer.train(_predictors())

        reopened = Trainer.load_trainer(trainer.path, wrap(sample_data))
        assert reopened.pipeline is not None
        assert reopened.pipeline.get_node_names() == built.get_node_names()
        assert reopened.selected_nodes == trainer.selected_nodes

    def test_set_pipeline_resets_stale_nodes(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.get_status('scaler') == 'built'
        pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1, f2, f3}'},
                         params={'with_std': False})
        trainer.set_pipeline(pipeline.build())
        assert trainer.get_status('scaler') is None


class TestTrain:
    def test_train_basic(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_skips_built(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())

        before = {name: _build_ids(trainer, name) for name in ['scaler', 'dt']}
        trainer.train(_predictors())
        for name in ['scaler', 'dt']:
            assert _build_ids(trainer, name) == before[name]

    def test_train_without_predictors_resumes_the_registered_ones(
            self, pipeline, sample_data, sp_v):
        """Omitting the argument means 'carry on with what is registered',
        not 'train nothing'."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        trainer.reset_nodes(['dt'])

        trainer.train()
        assert trainer.get_status('dt') == 'built'

    def test_adding_a_predictor_leaves_the_earlier_one_trained(
            self, pipeline, sample_data, sp_v):
        """Registration is an upsert, so a second train() call is additive —
        it trains the new Predictor and keeps the first one's artifacts."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train([_dt('a')])
        before = _build_ids(trainer, 'a')

        trainer.train([_dt('b')])
        assert trainer.predictor_names() == ['a', 'b']
        assert trainer.get_status('b') == 'built'
        assert _build_ids(trainer, 'a') == before

    def test_train_no_splitter(self, pipeline, sample_data):
        trainer = _make_trainer(pipeline, sample_data, None, name='t_nosplit')
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_with_native_data(self, pipeline, sample_data):
        trainer = Trainer(name='native_train', data=sample_data,
                          path=_trainer_path(pipeline, 'native_train'),
                          splitter=None, cache=DataCache())
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        assert trainer.get_status('scaler') == 'built'
        assert trainer.get_status('dt') == 'built'

    def test_train_error(self, pipeline, sample_data, sp_v):
        """A failed node has no artifact, so its *status* is None — 'error'
        exists only in history, which is what get_node_error reads."""
        _add_bad_node(pipeline)
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_err')
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors(_bad_dt()))
        assert trainer.get_status('bad_node') is None
        err = trainer.get_node_error('bad_node')
        assert err['type'] == 'ValueError'
        assert 'intentional error' in err['message']

    def test_train_records_predictor_errors(self, pipeline, sample_data, sp_v):
        """Predictor failures are recorded too, now that they have a store of
        their own — a Trainer used to have nowhere to put them."""
        _add_bad_node(pipeline)
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_perr')
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors(_bad_dt()))
        assert trainer.get_status('bad_dt') is None
        assert trainer.get_node_error('bad_dt') is not None

    def test_train_error_continues_other_nodes(self, pipeline, sample_data, sp_v):
        _add_bad_node(pipeline)
        trainer = _make_trainer(pipeline, sample_data, sp_v, name='t_mixed')
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors(_dt(), _bad_dt()))
        assert trainer.get_status('dt') == 'built'
        assert trainer.get_status('bad_node') is None
        assert trainer.get_node_error('bad_node') is not None

    def test_train_n_splits(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        assert trainer.get_n_splits() == 3
        trainer.train(_predictors())
        for fold in trainer.train_folds:
            assert trainer.predictor_store.status('dt', fold.split_idx, 0) == 'built'

    def test_history_records_the_pipeline_version(self, pipeline, sample_data, sp_v):
        """What a run was made against is recorded as the Pipeline's integer
        version — there is no serial/hash anymore."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        built = pipeline.build()
        trainer.set_pipeline(built)
        trainer.train(_predictors())
        for store in (trainer.node_store, trainer.predictor_store):
            for row in store.get_hist():
                assert row['pipeline_version'] == built.version

    def test_redefining_a_predictor_does_not_retrain_it(self, pipeline, sample_data, sp_v):
        """Skipping is disk-based: a built split is a built split, whatever the
        definition now says. Redefining updates the stored definition but does
        not by itself discard the artifact — reset_nodes() is the explicit
        way to force the rerun."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        before = _build_ids(trainer, 'dt')

        redefined = _dt(params={'max_depth': 5, 'random_state': 42})
        trainer.train([redefined])
        assert _build_ids(trainer, 'dt') == before
        assert trainer.predictor_defs.get_by_name('dt').params == redefined.params

        trainer.reset_nodes(['dt'])
        trainer.train([redefined])
        assert _build_ids(trainer, 'dt') != before

    def test_node_edit_cascades_into_the_predictor(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        before = {name: _build_ids(trainer, name) for name in ['scaler', 'dt']}

        pipeline.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1, f2, f3}'},
                         params={'with_std': False})
        trainer.set_pipeline(pipeline.build())
        assert trainer.get_status('scaler') is None
        assert trainer.get_status('dt') is None      # cascaded via node_names()
        trainer.train()

        for name in ['scaler', 'dt']:
            assert _build_ids(trainer, name) != before[name]

    def test_unrelated_node_edit_leaves_the_predictor_alone(self, pipeline, sample_data, sp_v):
        pipeline.set_grp('other', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'})
        pipeline.set_node('unused', grp='other')
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        before = _build_ids(trainer, 'dt')

        pipeline.set_grp('other', processor='sklearn.preprocessing.StandardScaler',
                         method='transform', edges={'X': '{f1}'},
                         params={'with_mean': False})
        trainer.set_pipeline(pipeline.build())
        trainer.train()
        assert _build_ids(trainer, 'dt') == before


class TestPredictorStorage:
    def test_artifacts_land_in_the_predictor_store(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        for fold in trainer.train_folds:
            assert trainer.predictor_store.status('dt', fold.split_idx, 0) == 'built'
            assert trainer.node_store.status('dt', fold.split_idx, 0) is None
            assert trainer.node_store.status('scaler', fold.split_idx, 0) == 'built'

    def test_predictor_is_not_loaded_into_the_flow(self, pipeline, sample_data, sp_v):
        """Its history is in the other store, so flow.load() never sees a row
        for it — which is what keeps the fitted model out of memory."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())

        flow = trainer.train_folds[0].train_data_flows[0]
        flow.node_objs.clear()
        flow.load()
        assert 'scaler' in flow.node_objs
        assert 'dt' not in flow.node_objs

    def test_definitions_are_persisted(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        stored = trainer.predictor_defs.list_predictors()
        assert [p.name for p in stored] == ['dt']
        assert stored[0].edges == DT_EDGES
        assert stored[0].params == {'max_depth': 3, 'random_state': 42}

    def test_a_later_train_does_not_drop_earlier_predictors(self, pipeline, sample_data, sp_v):
        """Registration is an upsert, not a replace — dropping 'a' here would
        leave its trained artifacts behind with no definition to read them."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors(_dt('a'), _dt('b')))
        trainer.train(_predictors(_dt('b')))
        assert [p.name for p in trainer.predictor_defs.list_predictors()] == ['a', 'b']

    def test_provenance_round_trips(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trial = Trial('dt', DT, DT_EDGES, params={'max_depth': 3}, tag=['x'])
        trainer.train([Predictor.from_trial(trial, experimenter='exp1')])

        stored = trainer.predictor_defs.get_by_name('dt')
        assert stored.src_trial == 'dt'
        assert stored.src_experimenter == 'exp1'
        assert stored.tag == ['x']

    def test_from_trial_copies_the_definition(self):
        trial = Trial('dt', DT, DT_EDGES, method='predict_proba',
                      params={'max_depth': 3}, desc='best so far')
        predictor = Predictor.from_trial(trial, name='final', experimenter='exp1')
        assert predictor.name == 'final'
        assert predictor.src_trial == 'dt'          # recorded even when renamed
        assert predictor.get_spec().edges == trial.get_spec().edges
        assert predictor.get_spec().params == trial.get_spec().params
        assert predictor.method == 'predict_proba'
        assert predictor.desc == 'best so far'


class TestProcess:
    def test_process_yields_per_split(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        results = list(trainer.process(sample_data))
        assert len(results) == trainer.get_n_splits()

    def test_process_output_shape(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        for output in trainer.process(sample_data):
            assert output.get_shape()[0] == len(sample_data)

    def test_process_after_reload(self, pipeline, sample_data, sp_v):
        """Regression: a reloaded flow has no edges for a Predictor (neither
        the artifact nor this flow's history carries them), so process() used
        to resolve every Predictor to None and yield nothing at all."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())

        loaded = Trainer.load_trainer(trainer.path, wrap(sample_data), cache=DataCache())
        results = list(loaded.process(sample_data))
        assert len(results) == loaded.get_n_splits()
        for output in results:
            assert output.get_shape()[0] == len(sample_data)


class TestResetNodes:
    def test_reset_clears_node_objs(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        trainer.reset_nodes(['scaler'])
        assert trainer.get_status('scaler') is None
        assert trainer.get_status('dt') is None

    def test_reset_allows_retrain(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        trainer.reset_nodes(['dt'])
        assert trainer.get_status('dt') is None
        trainer.train(_predictors())
        assert trainer.get_status('dt') == 'built'

    def test_reset_predictor_leaves_nodes_built(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(pipeline.build())
        trainer.train(_predictors())
        trainer.reset_nodes(['dt'])
        assert trainer.get_status('scaler') == 'built'


class TestSaveLoad:
    def _trained(self, pipeline, sample_data, sp_v):
        built = pipeline.build()
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        trainer.set_pipeline(built)
        trainer.train(_predictors())
        return trainer, built

    def test_save_creates_the_store(self, pipeline, sample_data, sp_v):
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        assert (trainer.path / '__trainer.db').exists()

    def test_splits_live_in_the_store(self, pipeline, sample_data, sp_v):
        """Splitter and the resolved indices are one blob — a Trainer may have
        no splitter at all, and reopening must land on the trained folds."""
        trainer = _make_trainer(pipeline, sample_data, sp_v)
        splits = trainer._store.load_splits()
        assert set(splits) == {'splitter', 'splitter_params', 'split_indices'}
        assert len(splits['split_indices']) == trainer.get_n_splits()

    def test_load_unknown_path_raises(self, tmp_path, sample_data):
        with pytest.raises(KeyError, match='No trainer'):
            Trainer.load_trainer(tmp_path / 'nope', wrap(sample_data))
        assert not (tmp_path / 'nope' / '__trainer.db').exists()

    def test_load_accepts_native_data(self, pipeline, sample_data):
        trainer = Trainer(name='native_load', data=sample_data,
                          path=_trainer_path(pipeline, 'native_load'),
                          splitter=None, cache=DataCache())
        trainer.set_pipeline(pipeline.build())
        reopened = Trainer.load_trainer(trainer.path, sample_data)
        assert reopened.get_n_splits() == 1
        assert reopened.data.get_shape()[0] == len(sample_data)

    def test_save_load_roundtrip(self, pipeline, sample_data, sp_v):
        trainer, _ = self._trained(pipeline, sample_data, sp_v)
        loaded = Trainer.load_trainer(trainer.path, wrap(sample_data), cache=DataCache())
        assert loaded.name == 't1'
        assert loaded.get_status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'

    def test_load_restores_predictors(self, pipeline, sample_data, sp_v):
        """Predictors come back from PredictorStore, so nothing needs
        re-supplying and nothing is reset by reopening."""
        trainer, _ = self._trained(pipeline, sample_data, sp_v)
        loaded = Trainer.load_trainer(trainer.path, wrap(sample_data), cache=DataCache())
        assert loaded.predictor_names() == ['dt']
        assert loaded.predictors[0].params == {'max_depth': 3, 'random_state': 42}
        assert loaded.get_status('dt') == 'built'

    def test_load_restores_pipeline_and_splits(self, pipeline, sample_data, sp_v):
        trainer, _ = self._trained(pipeline, sample_data, sp_v)
        loaded = Trainer.load_trainer(trainer.path, wrap(sample_data), cache=DataCache())
        assert loaded.pipeline is not None
        assert 'scaler' in loaded.pipeline.nodes
        assert loaded.selected_nodes == ['scaler']
        assert loaded.get_n_splits() == trainer.get_n_splits()
        loaded.reset_nodes(['scaler'])  # uses the restored pipeline
        assert loaded.get_status('scaler') is None
