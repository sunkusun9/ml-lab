"""
XGBoost adapter
"""

import pandas as pd
from ._base import ModelAdapter, GPU_NO, GPU_YES, stack_evals_result

from xgboost.callback import TrainingCallback

class ProgressCallback(TrainingCallback):
    def __init__(self, n_estimators, period_pct, monitor):
        self.n_estimators = n_estimators
        self.period_pct = period_pct
        self.last_printed = -1
        self.monitor = monitor

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        percentage = (current / self.n_estimators) * 100
        if int(percentage / (self.period_pct * 100)) > self.last_printed:
            self.last_printed = int(percentage / (self.period_pct * 100))
            metrics_str = ""
            if evals_log:
                last_metrics = []
                for dataset, metrics in evals_log.items():
                    for metric_name, values in metrics.items():
                        last_metrics.append(f"{dataset}-{metric_name}: {values[-1]:.4f}")
                metrics_str = ", ".join(last_metrics)
            if self.monitor is not None:
                self.monitor.report(current, self.n_estimators, metrics_str if metrics_str else None)
        return False

    def after_training(self, model):
        self.monitor = None
        return model

class XGBoostAdapter(ModelAdapter):
    result_objs = [
        'feature_importances_weight', 'feature_importances_gain', 'feature_importances_cover',
        'feature_importances_total_gain', 'feature_importances_total_cover',
        'evals_result', 'trees'
    ]

    def get_gpu_usage(self, params):
        gpu = (params or {}).get('gpu', 'auto')
        if gpu is None:
            return GPU_NO
        if gpu == 'auto':
            device = (params or {}).get('device', '')
            return GPU_YES if isinstance(device, str) and 'cuda' in device else GPU_NO
        return GPU_YES

    def inject_gpu_id(self, params, gpu_id):
        params = params.copy()
        params['device'] = f'cuda:{gpu_id}'
        return params

    def get_params(self, params, gpu_id_list=None, monitor=None, single_worker=False):
        if params is None:
            params = {}
        else:
            params = params.copy()

        gpu = params.pop('gpu', 'auto')
        if gpu is not None and gpu_id_list:
            params['device'] = f'cuda:{gpu_id_list[0]}'

        if self.verbose > 0 and self.verbose < 1:
            n_estimators = params.get('n_estimators', 100)
            callbacks = params.get('callbacks', [])
            if monitor is not None:
                callbacks.append(ProgressCallback(n_estimators, self.verbose, monitor))
            params['callbacks'] = callbacks
        """
        if not single_worker:
            params['n_jobs'] = 1
        """
        return params

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None, single_worker=False):
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(train_data, valid_data, params, monitor)

        if params is not None and params.get('verbosity', 0) > 0:
            fit_params['verbose'] = True
        else:
            fit_params['verbose'] = False

        train_v_X = valid_data.get('X') if valid_data else None
        train_v_y = valid_data.get('y') if valid_data else None

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            if self.eval_mode == 'valid':
                fit_params['eval_set'] = [(unwrap(train_v_X), unwrap(train_v_y))]
            elif self.eval_mode == 'both':
                fit_params['eval_set'] = [(fit_params['X'], fit_params['y']), (unwrap(train_v_X), unwrap(train_v_y))]

            train_v_w = valid_data.get('sample_weight') if valid_data else None
            train_v_w_native = unwrap(train_v_w.squeeze()) if train_v_w is not None else None
            eval_weight = self._eval_weight_list(
                fit_params.get('sample_weight'), train_v_w_native, self.eval_mode)
            if eval_weight is not None:
                fit_params['sample_weight_eval_set'] = eval_weight

        return fit_params

    def _get_feature_importances(processor, importance_type):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))
        booster = obj.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        return pd.Series([scores.get(f, 0) for f in input_vars], index=input_vars, name='importance')

    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.evals_result() if hasattr(obj, 'evals_result') else {}
        return stack_evals_result(evals_result)

    def _get_trees(processor):
        obj = processor.obj
        booster = obj.get_booster()
        return booster.trees_to_dataframe()


XGBoostAdapter.result_objs = {
    'feature_importances': (XGBoostAdapter._get_feature_importances, True),
    'evals_result': (XGBoostAdapter._get_evals_result, True),
    'trees': (XGBoostAdapter._get_trees, False)
}
