import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs.sampler import Sampler, ImbLearnSampler
from mllabs._data_wrapper import PandasWrapper, wrap
from mllabs._node_processor import TransformProcessor, PredictProcessor
from mllabs import Project
from mllabs._experimenter import Experimenter
from mllabs._trainer import Trainer
from mllabs._cache import DataCache
from mllabs._pipeline import PipelineBuilder


# ── helpers ───────────────────────────────────────────────────────────────────

def pw(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return PandasWrapper(data)
    return PandasWrapper(pd.DataFrame(np.array(data), columns=columns))


def make_data(X=None, y=None):
    """Build {key: DataWrapper} dict for fit/fit_process/process."""
    d = {}
    if X is not None:
        d['X'] = pw(X) if not isinstance(X, PandasWrapper) else X
    if y is not None:
        d['y'] = pw(y) if not isinstance(y, PandasWrapper) else y
    return d


SPLITTERS = dict(
    sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
    sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
)


def make_exp(path, data, aug_data=None, name='e1'):
    return Experimenter(path, name, data, aug_data=aug_data, **SPLITTERS)


def make_project_exp(root, data, aug_data=None, name='e1'):
    """An Experimenter created through a Project.

    aug_data is not persisted with the Experimenter, and Project keeps no
    project-level default of its own any more — register it under a name in
    project.ext_data and pass '@ext:name' to add_experimenter instead.
    """
    project = Project(root, data=data)
    if aug_data is not None:
        project.ext_data.register('aug', aug_data)
        project.add_experimenter(name, aug_data='@ext:aug', **SPLITTERS)
    else:
        project.add_experimenter(name, **SPLITTERS)
    return project


class MockResampler:
    """imblearn API compatible mock: appends extra rows."""
    def __init__(self, extra_X, extra_y=None):
        self.extra_X = np.array(extra_X)
        self.extra_y = np.array(extra_y) if extra_y is not None else None

    def fit_resample(self, X, y):
        X_res = np.vstack([X, self.extra_X])
        y_res = np.concatenate([y, self.extra_y]) if y is not None and self.extra_y is not None else y
        return X_res, y_res


class AddRowsSampler(Sampler):
    """Test sampler: appends known extra rows to fit_params X and y."""
    def __init__(self, extra_X, extra_y=None):
        self.extra_X = np.array(extra_X)
        self.extra_y = np.array(extra_y) if extra_y is not None else None

    def sample(self, fit_params):
        result = dict(fit_params)
        result['X'] = np.vstack([fit_params['X'], self.extra_X])
        if 'y' in fit_params and self.extra_y is not None:
            result['y'] = np.concatenate([fit_params['y'], self.extra_y])
        return result


class SimpleFitPredict:
    __name__ = 'SimpleFitPredict'
    classes_ = [0, 1]

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


# ── ImbLearnSampler ───────────────────────────────────────────────────────────

class TestImbLearnSampler:
    def test_sample_augments_X_y(self):
        sampler = ImbLearnSampler(MockResampler([[10.0, 10.0]], [1]))
        fit_params = {'X': np.array([[1.0, 2.0], [3.0, 4.0]]), 'y': np.array([0, 1])}
        result = sampler.sample(fit_params)
        assert result['X'].shape[0] == 3
        assert result['y'].shape[0] == 3
        np.testing.assert_array_equal(result['X'][-1], [10.0, 10.0])
        assert result['y'][-1] == 1

    def test_sample_passes_through_other_keys(self):
        sampler = ImbLearnSampler(MockResampler([[10.0]], [1]))
        extra = [('dummy',)]
        fit_params = {'X': np.array([[1.0], [2.0]]), 'y': np.array([0, 1]), 'eval_set': extra}
        result = sampler.sample(fit_params)
        assert result['eval_set'] is extra

    def test_sample_no_y_in_fit_params(self):
        sampler = ImbLearnSampler(MockResampler([[10.0]]))
        fit_params = {'X': np.array([[1.0], [2.0]])}
        result = sampler.sample(fit_params)
        assert result['X'].shape[0] == 3
        assert 'y' not in result

    def test_does_not_mutate_original(self):
        sampler = ImbLearnSampler(MockResampler([[10.0, 10.0]], [1]))
        X_orig = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_orig = np.array([0, 1])
        fit_params = {'X': X_orig, 'y': y_orig}
        sampler.sample(fit_params)
        assert fit_params['X'].shape[0] == 2
        assert fit_params['y'].shape[0] == 2


# ── mllab_sampler in TransformProcessor ──────────────────────────────────────

class TestTransformProcessorSampler:
    TRAIN = [[1.0, 2.0], [3.0, 4.0]]
    TRAIN_V = [[1.5, 2.5]]
    EXTRA = [[10.0, 10.0]]
    COLS = ['a', 'b']

    def _train_data(self):
        return make_data(X=pd.DataFrame(self.TRAIN, columns=self.COLS))

    def _valid_data(self):
        return make_data(X=pd.DataFrame(self.TRAIN_V, columns=self.COLS))

    def test_fit_uses_augmented_data(self):
        sampler = AddRowsSampler(self.EXTRA)
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit(self._train_data(), self._valid_data())
        expected_mean = np.mean(self.TRAIN + self.EXTRA, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_without_sampler_unaffected(self):
        proc = TransformProcessor('s', StandardScaler)
        proc.fit(self._train_data(), self._valid_data())
        expected_mean = np.mean(self.TRAIN, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_sampler_not_passed_to_transformer(self):
        sampler = AddRowsSampler(self.EXTRA)
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit(self._train_data(), self._valid_data())  # must not raise

    def test_fit_process_fitted_on_augmented(self):
        sampler = AddRowsSampler(self.EXTRA)
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit_process(self._train_data(), self._valid_data())
        expected_mean = np.mean(self.TRAIN + self.EXTRA, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_process_output_shape_matches_original(self):
        sampler = AddRowsSampler(self.EXTRA)
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        result = proc.fit_process(self._train_data(), self._valid_data())
        assert result.get_shape()[0] == len(self.TRAIN)


# ── mllab_sampler in PredictProcessor ────────────────────────────────────────

class TestPredictProcessorSampler:
    N = 20
    N_V = 5
    N_EXTRA = 5

    @pytest.fixture(autouse=True)
    def data(self):
        rng = np.random.default_rng(0)
        cols = ['a', 'b']
        train_X = rng.standard_normal((self.N, 2))
        train_v_X = rng.standard_normal((self.N_V, 2))
        train_y = rng.integers(0, 2, self.N)
        train_v_y = rng.integers(0, 2, self.N_V)
        self.extra_X = rng.standard_normal((self.N_EXTRA, 2))
        self.extra_y = rng.integers(0, 2, self.N_EXTRA)
        self.train_data = make_data(
            X=pd.DataFrame(train_X, columns=cols),
            y=pd.DataFrame({'t': train_y}),
        )
        self.valid_data = make_data(
            X=pd.DataFrame(train_v_X, columns=cols),
            y=pd.DataFrame({'t': train_v_y}),
        )

    def test_fit_sampler_not_passed_to_estimator(self):
        sampler = AddRowsSampler(self.extra_X, self.extra_y)
        proc = PredictProcessor('dt', DecisionTreeClassifier,
                                params={'mllab_sampler': sampler, 'max_depth': 3, 'random_state': 0})
        proc.fit(self.train_data, self.valid_data)  # must not raise

    def test_fit_process_output_shape_matches_original(self):
        sampler = AddRowsSampler(self.extra_X, self.extra_y)
        proc = PredictProcessor('dt', SimpleFitPredict, method='fit_predict',
                                params={'mllab_sampler': sampler})
        result = proc.fit_process(self.train_data, self.valid_data)
        assert result.get_shape()[0] == self.N

    def test_fit_process_without_sampler_output_shape(self):
        proc = PredictProcessor('dt', SimpleFitPredict, method='fit_predict')
        result = proc.fit_process(self.train_data, self.valid_data)
        assert result.get_shape()[0] == self.N


# ── Experimenter.aug_data ─────────────────────────────────────────────────────

@pytest.fixture
def base_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'f1': rng.standard_normal(100),
        'f2': rng.standard_normal(100),
        'target': rng.integers(0, 2, 100),
    })


@pytest.fixture
def aug_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'f1': rng.standard_normal(20),
        'f2': rng.standard_normal(20),
        'target': rng.integers(0, 2, 20),
    })


class TestExperimenterAugData:
    def test_aug_data_stored(self, tmp_path, base_data, aug_df):
        exp = make_exp(tmp_path / 'exp', base_data, aug_data=aug_df)
        assert exp.aug_data is not None

    def test_no_aug_data_is_none(self, tmp_path, base_data):
        exp = make_exp(tmp_path / 'exp', base_data)
        assert exp.aug_data is None

    def test_inner_train_size_increased(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        n_with = exp_with.outer_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        n_without = exp_without.outer_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        assert n_with == n_without + len(aug_df)

    def test_inner_valid_unchanged(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        for i in range(len(exp_with.outer_folds[0].train_data_flows)):
            v_with = exp_with.outer_folds[0].train_data_flows[i].data_source.get_valid()
            v_without = exp_without.outer_folds[0].train_data_flows[i].data_source.get_valid()
            if v_with is not None and v_without is not None:
                assert v_with.get_shape()[0] == v_without.get_shape()[0]

    def test_outer_test_unchanged(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        assert len(exp_with.outer_folds[0].test_idx) == len(exp_without.outer_folds[0].test_idx)

    def test_column_filter_applied_to_aug(self, tmp_path, base_data, aug_df):
        exp = make_exp(tmp_path / 'exp', base_data, aug_data=aug_df)
        train = exp.outer_folds[0].train_data_flows[0].data_source.get_train()
        assert 'f1' in train.get_columns()
        assert 'f2' in train.get_columns()

    def test_load_passes_aug_data(self, tmp_path, base_data, aug_df):
        """aug_data is never persisted, and Project no longer keeps a
        project-level default either — reopening through the lazy
        `experimenters` property (see test_load_without_aug_data) therefore
        comes back with none. Carrying it across a reload is the caller's
        job: resupply the same '@ext:name' reference to
        Experimenter.load_experimenter directly."""
        make_project_exp(tmp_path / 'proj', base_data, aug_data=aug_df)
        reopened = Project(tmp_path / 'proj')
        exp2 = Experimenter.load_experimenter(
            reopened.exp_path('e1'), base_data, aug_data='@ext:aug',
            resolver=reopened.resolver,
        )
        assert exp2.aug_data is not None
        n = exp2.outer_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        assert n > 0

    def test_load_without_aug_data(self, tmp_path, base_data):
        make_project_exp(tmp_path / 'proj', base_data)
        exp2 = Project(tmp_path / 'proj').experimenters['e1']
        assert exp2.aug_data is None


# ── Trainer.aug_data ──────────────────────────────────────────────────────────

class TestTrainerAugData:
    @pytest.fixture
    def pipeline(self, tmp_path):
        p = PipelineBuilder(path=tmp_path / 'pipeline')
        p.set_datasource({'f1': 'numerical', 'f2': 'numerical', 'target': 'binary'})
        p.set_grp('model', processor='sklearn.tree.DecisionTreeClassifier',
                  method='predict',
                  edges={'X': '{f1, f2}', 'y': '{target}'},
                  params={'max_depth': 3, 'random_state': 42})
        p.set_node('dt', grp='model')
        return p

    def _make_trainer(self, pipeline, name, data, path, splitter=None, aug_data=None):
        return Trainer(name=name, data=wrap(data), path=path,
                       splitter=splitter, splitter_params={}, cache=DataCache(),
                       aug_data=aug_data)

    def test_add_trainer_stores_aug_data(self, tmp_path, pipeline, base_data, aug_df):
        trainer = self._make_trainer(pipeline, 't1', base_data, tmp_path / 'tr_t1', aug_data=aug_df)
        assert trainer.aug_data is not None

    def test_add_trainer_no_aug_data(self, tmp_path, pipeline, base_data):
        trainer = self._make_trainer(pipeline, 't1', base_data, tmp_path / 'tr_t1')
        assert trainer.aug_data is None

    def test_trainer_inner_train_size_increased(self, tmp_path, pipeline, base_data, aug_df):
        trainer_with = self._make_trainer(pipeline, 't_with', base_data, tmp_path / 'tr_with', aug_data=aug_df)
        trainer_without = self._make_trainer(pipeline, 't_without', base_data, tmp_path / 'tr_without')
        n_with = trainer_with.train_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        n_without = trainer_without.train_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        assert n_with == n_without + len(aug_df)

    def test_trainer_valid_unchanged(self, tmp_path, pipeline, base_data, aug_df):
        trainer_with = self._make_trainer(pipeline, 't_with', base_data, tmp_path / 'tr_with', aug_data=aug_df)
        trainer_without = self._make_trainer(pipeline, 't_without', base_data, tmp_path / 'tr_without')
        for fold_with, fold_without in zip(trainer_with.train_folds, trainer_without.train_folds):
            v_with = fold_with.train_data_flows[0].data_source.get_valid()
            v_without = fold_without.train_data_flows[0].data_source.get_valid()
            if v_with is not None and v_without is not None:
                assert v_with.get_shape()[0] == v_without.get_shape()[0]

    def test_trainer_no_split_aug_data(self, tmp_path, pipeline, base_data, aug_df):
        trainer_with = self._make_trainer(pipeline, 't_with', base_data, tmp_path / 'tr_with', splitter=None, aug_data=aug_df)
        trainer_without = self._make_trainer(pipeline, 't_without', base_data, tmp_path / 'tr_without', splitter=None)
        n_with = trainer_with.train_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        n_without = trainer_without.train_folds[0].train_data_flows[0].data_source.get_train().get_shape()[0]
        assert n_with == n_without + len(aug_df)

    def test_trainer_column_filter_applied_to_aug(self, tmp_path, pipeline, base_data, aug_df):
        trainer = self._make_trainer(pipeline, 't1', base_data, tmp_path / 'tr_t1', aug_data=aug_df)
        train = trainer.train_folds[0].train_data_flows[0].data_source.get_train()
        selected = train.select_columns(['f1', 'f2'])
        assert selected.get_columns() == ['f1', 'f2']
