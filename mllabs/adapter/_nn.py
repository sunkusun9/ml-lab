from ._base import ModelAdapter, GPU_NO, GPU_POSSIBLE, GPU_YES, stack_evals_result

try:
    import tensorflow as tf
    _keras_cb = tf.keras.callbacks.Callback
except ImportError:
    tf = None
    _keras_cb = object


class _ProgressCallback(_keras_cb):

    def __init__(self, n_epochs, verbose, monitor):
        if _keras_cb is not object:
            super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.monitor = monitor
        self.last_printed = -1

    def on_epoch_end(self, epoch, logs=None):
        current = epoch + 1
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()) if logs else ""

        if self.verbose < 1:
            bucket = int((current / self.n_epochs) / self.verbose)
            if bucket > self.last_printed:
                self.last_printed = bucket
                self.monitor.report(current, self.n_epochs, metrics_str or None)
        else:
            interval = int(self.verbose)
            if current % interval == 0 or current == self.n_epochs:
                self.monitor.report(current, self.n_epochs, metrics_str or None)


class NNAdapter(ModelAdapter):

    def get_gpu_usage(self, params):
        device = (params or {}).get('device', None)
        if device:
            return GPU_YES
        return GPU_POSSIBLE

    def inject_gpu_id(self, params, gpu_id):
        params['device'] = f'/GPU:{gpu_id}'

    def get_params(self, params, gpu_id_list=None, monitor=None, single_worker=False):
        if params is None:
            return {}
        gpu = params.get('gpu', 'auto')
        params = params.copy()
        params.pop('gpu', None)
        if gpu is not None and gpu_id_list:
            params['device'] = f'/GPU:{gpu_id_list[0]}'
        return params

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None, single_worker=False):
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(train_data, valid_data, params, monitor)

        train_v_X = valid_data.get('X') if valid_data else None
        train_v_y = valid_data.get('y') if valid_data else None

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            fit_params['eval_set'] = [(unwrap(train_v_X), unwrap(train_v_y))]

        if self.verbose > 0 and monitor is not None and tf is not None:
            n_epochs = (params or {}).get('epochs', 100)
            fit_params['callbacks'] = [_ProgressCallback(n_epochs, self.verbose, monitor)]

        return fit_params

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = getattr(obj, 'evals_result_', {})
        return stack_evals_result(evals_result)


NNAdapter.result_objs = {
    'evals_result': (NNAdapter._get_evals_result, True),
}
