"""
LightGBM adapter
"""

import pandas as pd
from ._base import ModelAdapter, GPU_NO, GPU_YES, stack_evals_result
from lightgbm import early_stopping

def create_progress_callback(n_estimators, period_pct, monitor):
    last_printed = [-1]
    def callback(env):
        current = env.iteration + 1
        percentage = (current / n_estimators) * 100
        if int(percentage / (period_pct * 100)) > last_printed[0]:
            last_printed[0] = int(percentage / (period_pct * 100))
            metrics_str = ""
            if env.evaluation_result_list:
                last_metrics = []
                for item in env.evaluation_result_list:
                    dataset_name, metric_name, value, _ = item
                    last_metrics.append(f"{dataset_name}-{metric_name}: {value:.4f}")
                metrics_str = ", ".join(last_metrics)
            monitor.report(current, n_estimators, metrics_str if metrics_str else None)
    return callback

class LightGBMAdapter(ModelAdapter):
    def get_gpu_usage(self, params):
        gpu = (params or {}).get('gpu', 'auto')
        if gpu is None:
            return GPU_NO
        if gpu == 'auto':
            return GPU_YES if (params or {}).get('device') == 'gpu' else GPU_NO
        return GPU_YES

    def inject_gpu_id(self, params, gpu_id):
        params = params.copy()
        params['device'] = 'gpu'
        params['gpu_device_id'] = gpu_id
        return params

    def get_params(self, params, gpu_id_list=None, monitor=None, single_worker=True):
        gpu = (params or {}).get('gpu', 'auto')
        params = {k: v for k, v in params.items() if k not in ['early_stopping', 'eval_metric', 'gpu']}
        if gpu is not None and gpu_id_list:
            params['device'] = 'gpu'
            params['gpu_device_id'] = gpu_id_list[0]
        if not single_worker:
            params['n_jobs'] = 1
        return params

    def get_process_data(self, data):
        from .._data_wrapper import unwrap
        x = unwrap(data)
        if x is not None and 'polars' in type(x).__module__:
            return x.to_pandas()
        return x

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None, single_worker=True):
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(train_data, valid_data, params, monitor)

        def _not_polars(x):
            if x is not None and 'polars' in type(x).__module__:
                return x.to_pandas()
            return x

        if 'X' in fit_params:
            fit_params['X'] = _not_polars(fit_params['X'])
        if 'y' in fit_params:
            fit_params['y'] = _not_polars(fit_params['y'])
        if 'sample_weight' in fit_params:
            fit_params['sample_weight'] = _not_polars(fit_params['sample_weight'])

        train_v_X = valid_data.get('X') if valid_data else None
        train_v_y = valid_data.get('y') if valid_data else None

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            train_v_X_native = _not_polars(unwrap(train_v_X))
            train_v_y_native = _not_polars(unwrap(train_v_y.squeeze()))
            if self.eval_mode == 'valid':
                fit_params['eval_set'] = [(train_v_X_native, train_v_y_native)]
            elif self.eval_mode == 'both':
                fit_params['eval_set'] = [(fit_params['X'], fit_params['y']), (train_v_X_native, train_v_y_native)]

            train_v_w = valid_data.get('sample_weight') if valid_data else None
            train_v_w_native = _not_polars(unwrap(train_v_w.squeeze())) if train_v_w is not None else None
            eval_weight = self._eval_weight_list(
                fit_params.get('sample_weight'), train_v_w_native, self.eval_mode)
            if eval_weight is not None:
                fit_params['eval_sample_weight'] = eval_weight

        if self.verbose > 0:
            if self.verbose < 1:
                n_estimators = params.get('n_estimators', 100) if params else 100
                callbacks = fit_params.get('callbacks', [])
                if monitor is not None:
                    callbacks.append(create_progress_callback(n_estimators, self.verbose, monitor))
                fit_params['callbacks'] = callbacks
            else:
                fit_params['verbose'] = int(self.verbose)

        if params and 'early_stopping' in params:
            if 'callbacks' not in fit_params:
                fit_params['callbacks'] = list()
            es = params['early_stopping']
            if isinstance(es, dict):
                es = early_stopping(**es)
            fit_params['callbacks'].append(es)
        if params and 'eval_metric' in params:
            fit_params['eval_metric'] = params['eval_metric']
        return fit_params

    @staticmethod
    def _get_feature_importances(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))
        return pd.Series(obj.feature_importances_, index=input_vars, name='importance')

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.evals_result_ if hasattr(obj, 'evals_result_') else {}
        return stack_evals_result(evals_result)

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        dump = obj.booster_.dump_model()
        return dump.get('tree_info', [])

LightGBMAdapter.result_objs = {
    'feature_importances': (LightGBMAdapter._get_feature_importances, True),
    'evals_result': (LightGBMAdapter._get_evals_result, True),
    'trees': (LightGBMAdapter._get_trees, False)
}
