"""
CatBoost adapter
"""

import tempfile
import json
import pandas as pd
from ._base import ModelAdapter, GPU_NO, GPU_YES, stack_evals_result


def _catboost_supports_polars():
    from packaging.version import Version
    import catboost
    return Version(catboost.__version__) >= Version('1.3.0')


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models (CatBoostClassifier, CatBoostRegressor)

    CatBoost도 eval_set을 지원합니다.
    """

    def get_gpu_usage(self, params):
        gpu = (params or {}).get('gpu', 'auto')
        if gpu is None:
            return GPU_NO
        if gpu == 'auto':
            return GPU_YES if (params or {}).get('task_type') == 'GPU' else GPU_NO
        return GPU_YES  # 'yes'

    def inject_gpu_id(self, params, gpu_id):
        params = params.copy()
        params['task_type'] = 'GPU'
        params['devices'] = str(gpu_id)
        return params

    def get_params(self, params, gpu_id_list=None, monitor=None, single_worker=False):
        if params is None:
            return {}
        gpu = params.get('gpu', 'auto')
        params = params.copy()
        params.pop('gpu', None)
        if gpu is not None and gpu_id_list:
            params['task_type'] = 'GPU'
            params['devices'] = str(gpu_id_list[0])
        """
        if not single_worker:
            params['thread_count'] = 1
        """
        return params

    def get_process_data(self, data):
        from .._data_wrapper import unwrap
        x = unwrap(data)
        if not _catboost_supports_polars() and x is not None and 'polars' in type(x).__module__:
            return x.to_pandas()
        return x

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None, single_worker=False):
        """CatBoost의 fit 파라미터 구성"""
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(train_data, valid_data, params, monitor)

        def _maybe_to_pandas(x):
            if not _catboost_supports_polars() and x is not None and 'polars' in type(x).__module__:
                return x.to_pandas()
            return x

        if 'X' in fit_params:
            fit_params['X'] = _maybe_to_pandas(fit_params['X'])
        if 'y' in fit_params:
            fit_params['y'] = _maybe_to_pandas(fit_params['y'])
        if 'sample_weight' in fit_params:
            fit_params['sample_weight'] = _maybe_to_pandas(fit_params['sample_weight'])

        train_v_X = valid_data.get('X') if valid_data else None
        train_v_y = valid_data.get('y') if valid_data else None

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            v_X = _maybe_to_pandas(unwrap(train_v_X))
            v_y = _maybe_to_pandas(unwrap(train_v_y))
            v_w = valid_data.get('sample_weight') if valid_data else None
            v_w = _maybe_to_pandas(unwrap(v_w.squeeze())) if v_w is not None else None
            t_w = fit_params.get('sample_weight')

            if self.eval_mode == 'valid':
                entries = [(v_X, v_y, v_w)]
            else:
                entries = [(fit_params['X'], fit_params['y'], t_w), (v_X, v_y, v_w)]

            if t_w is not None or v_w is not None:
                # CatBoost has no eval-side weight parameter of its own — a
                # Pool is where a weight column attaches, so every entry
                # becomes one (weight=None on a Pool is just uniform, same
                # as the plain-tuple form below).
                from catboost import Pool
                fit_params['eval_set'] = [Pool(x, y, weight=w) for x, y, w in entries]
            else:
                fit_params['eval_set'] = [(x, y) for x, y, _ in entries]

        if self.verbose > 0:
            if self.verbose < 1:
                fit_params['verbose'] = False
            else:
                fit_params['verbose'] = int(self.verbose)
        else:
            fit_params['verbose'] = False

        return fit_params

    @staticmethod
    def _get_feature_importances_pvc(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        return pd.Series(
            obj.get_feature_importance(type='PredictionValuesChange'),
            index=input_vars, name = 'PredictionValuesChange'
        )

    @staticmethod
    def _get_feature_importances_interaction(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        interaction = obj.get_feature_importance(type='Interaction')
        return pd.DataFrame(
            interaction, columns=['feat1', 'feat2', 'importance']
        ).assign(
            feat1=lambda x: x['feat1'].astype('int').apply(lambda y: input_vars[y]),
            feat2=lambda x: x['feat2'].astype('int').apply(lambda y: input_vars[y]),
        ).set_index(['feat1', 'feat2'])['importance']

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.get_evals_result() if hasattr(obj, 'get_evals_result') else {}
        return stack_evals_result(evals_result)

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            obj.save_model(f.name, format="json")
            trees = json.load(f).get('oblivious_trees', [])
        return trees

CatBoostAdapter.result_objs = {
    'feature_importances_pvc': (CatBoostAdapter._get_feature_importances_pvc, True),
    'feature_importances_interaction': (CatBoostAdapter._get_feature_importances_interaction, True),
    'evals_result': (CatBoostAdapter._get_evals_result, True), 
    'trees': (CatBoostAdapter._get_trees, False)
}