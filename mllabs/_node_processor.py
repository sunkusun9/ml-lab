from .adapter import resolve_node_adapter
from collections.abc import Iterable


class ProgressMonitor:
    def report(self, current, total, metrics=None):
        pass

    def message(self, msg, typ='info'):
        pass


def _resolve_col_selectors(params, data):
    if not params or data is None:
        return params
    from ._pipeline import ColSelector
    from ._edge_dsl import parse, eval_expr
    resolved = {}
    for k, v in params.items():
        resolved[k] = eval_expr(parse(v.dsl_string), data) if isinstance(v, ColSelector) else v
    return resolved


class TransformProcessor():
    def __init__(self, name, transformer, adapter = None, params = {}):
        """*transformer* and *params* arrive already resolved — a live class
        and plain-data-or-live-object params — the caller's job now
        (``_executor._process`` via ``Resolver``, e.g.), not this
        constructor's. *adapter* is the one exception: ``resolve_node_adapter``
        stays here because its ``adapter=None`` default-lookup is genuine
        work (picks an adapter from *transformer*'s class name), not a
        passthrough resolve() would make idempotent."""
        self.name = name
        self.params = params
        self.adapter = resolve_node_adapter(transformer, adapter)
        self.transformer = transformer
        self.output_vars = None

    def fit(self, train_data, valid_data=None, gpu_id_list=None, monitor=None, single_worker=False):
        sampler = self.params.get('mllab_sampler') if self.params else None
        _params = {k: v for k, v in self.params.items() if k != 'mllab_sampler'} if self.params else {}

        if 'X' in train_data:
            train_X = train_data['X']
            self.X_ = train_X.get_columns()
        else:
            train_X = None
            self.X_ = []

        if 'y' in train_data:
            train_y = train_data['y']
            self.y_columns = train_y.get_columns()
        else:
            train_y = None
            self.y_columns = None

        _ref_data = train_X if train_X is not None else train_y
        resolved_params = _resolve_col_selectors(_params, _ref_data)
        self.obj = self.transformer(**self.adapter.get_params(resolved_params, gpu_id_list=gpu_id_list, monitor=monitor, single_worker=single_worker))

        fit_params = self.adapter.get_fit_params(train_data, valid_data, params=resolved_params, monitor=monitor, single_worker=single_worker)
        if sampler is not None:
            fit_params = sampler.sample(fit_params)
        self.obj.fit(**fit_params)

        if hasattr(self.obj, 'get_feature_names_out'):
            column_names = list(self.obj.get_feature_names_out())
            column_names = [f"{self.name}__{col}" for col in column_names]
        else:
            column_names = None

        if column_names is not None:
            self.output_vars = column_names
        elif not self.X_ and self.y_columns is not None:
            self.output_vars = list(self.y_columns)
        return self

    def fit_process(self, train_data, valid_data=None, gpu_id_list=None, monitor=None, single_worker=True):
        sampler = self.params.get('mllab_sampler') if self.params else None
        _params = {k: v for k, v in self.params.items() if k != 'mllab_sampler'} if self.params else {}

        if 'X' in train_data:
            train_X = train_data['X']
            self.X_ = train_X.get_columns()
            train_index = train_X.get_index()
            train_wrapper_class = type(train_X)
        else:
            train_X = None
            self.X_ = []
            train_index = None
            train_wrapper_class = None

        if 'y' in train_data:
            train_y = train_data['y']
            self.y_columns = train_y.get_columns()
            if train_X is None:
                train_index = train_y.get_index()
                train_wrapper_class = type(train_y)
        else:
            train_y = None
            self.y_columns = None

        _ref_data = train_X if train_X is not None else train_y
        resolved_params = _resolve_col_selectors(_params, _ref_data)
        self.obj = self.transformer(**self.adapter.get_params(resolved_params, gpu_id_list=gpu_id_list, monitor=monitor, single_worker=single_worker))

        fit_params = self.adapter.get_fit_params(train_data, valid_data, params=resolved_params, monitor=monitor, single_worker=single_worker)
        if sampler is not None:
            self.obj.fit(**sampler.sample(fit_params))
            result = self.obj.transform(fit_params['X'])
        else:
            result = self.obj.fit_transform(**fit_params)

        if hasattr(self.obj, 'get_feature_names_out'):
            column_names = list(self.obj.get_feature_names_out())
            column_names = [f"{self.name}__{col}" for col in column_names]
        else:
            column_names = None

        if column_names is None and hasattr(result, 'columns'):
            if type(result.columns) is str:
                cols = [result.columns]
            else:
                cols = result.columns
            column_names = [f"{self.name}__{col}" for col in cols]

        if column_names is not None:
            self.output_vars = column_names
        elif not self.X_ and self.y_columns is not None:
            if isinstance(self.y_columns, Iterable) and not isinstance(self.y_columns, (str, bytes)):
                self.output_vars = self.y_columns
            else:
                self.output_vars = [self.y_columns]
                
        return train_wrapper_class.from_output(result, self.output_vars, train_index)

    def process(self, data):
        if 'X' in data:
            data = data['X']
        else:
            data = data['y']
        data_index = data.get_index()
        wrapper_class = type(data)
        data = self.adapter.get_process_data(data)

        data_input = data if self.X_ else data.squeeze()
        data_native = self.adapter.get_process_data(data_input)

        result = self.obj.transform(data_native)
        output_vars = self.output_vars
        if output_vars is None and hasattr(result, 'columns'):
            if isinstance(result.columns, Iterable) and not isinstance(result.columns, (str, bytes)):
                cols = result.columns
            else:
                cols = [result.columns]
                
            output_vars = [f"{self.name}__{col}" for col in cols]
        return wrapper_class.from_output(result, output_vars, data_index)

class PredictProcessor():
    def __init__(self, name, estimator, method='predict', adapter = None, params = {}):
        """*estimator* and *params* arrive already resolved — see
        ``TransformProcessor.__init__`` for why, and why *adapter* is the
        exception that stays resolved here."""
        self.name = name
        self.params = params
        self.method = method
        self.output_vars = None
        self.adapter = resolve_node_adapter(estimator, adapter)
        self.estimator = estimator
        self.y_columns = None

    def fit(self, train_data, valid_data=None, gpu_id_list=None, monitor=None, single_worker=True):
        sampler = self.params.get('mllab_sampler') if self.params else None
        _params = {k: v for k, v in self.params.items() if k != 'mllab_sampler'} if self.params else {}

        train_X = train_data['X']
        self.X_ = train_X.get_columns()

        if 'y' in train_data:
            train_y = train_data['y']
            self.y_columns = train_y.get_columns()
        else:
            self.y_columns = None

        resolved_params = _resolve_col_selectors(_params, train_X)
        self.obj = self.estimator(**self.adapter.get_params(resolved_params, gpu_id_list=gpu_id_list, monitor=monitor, single_worker=single_worker))
        fit_params = self.adapter.get_fit_params(train_data, valid_data, params=resolved_params, monitor=monitor, single_worker=single_worker)
        if sampler is not None:
            fit_params = sampler.sample(fit_params)
        self.obj.fit(**fit_params)

        if isinstance(self.y_columns, Iterable) and not isinstance(self.y_columns, (str, bytes)):
            y_name = '_'.join(self.y_columns) if self.y_columns else 'prediction'
        else:
            y_name = self.y_columns if self.y_columns else 'prediction'
        
        if self.method == 'predict':
            self.output_vars = [f"{self.name}__{y_name}"]
        elif self.method == 'predict_proba':
            self.output_vars = [f"{self.name}__{y_name}_{i}" for i in self.obj.classes_]
        return self

    def fit_process(self, train_data, valid_data=None, gpu_id_list=None, monitor=None, single_worker=False):
        sampler = self.params.get('mllab_sampler') if self.params else None
        _params = {k: v for k, v in self.params.items() if k != 'mllab_sampler'} if self.params else {}

        train_X = train_data['X']
        self.X_ = train_X.get_columns()
        train_index = train_X.get_index()

        if 'y' in train_data:
            train_y = train_data['y']
            self.y_columns = train_y.get_columns()
        else:
            self.y_columns = None

        resolved_params = _resolve_col_selectors(_params, train_X)
        self.obj = self.estimator(**self.adapter.get_params(resolved_params, gpu_id_list=gpu_id_list, monitor=monitor, single_worker=single_worker))

        fit_params = self.adapter.get_fit_params(train_data, valid_data, params=resolved_params, monitor=monitor, single_worker=single_worker)
        if sampler is not None:
            self.obj.fit(**sampler.sample(fit_params))
            predictions = self.obj.predict(fit_params['X'])
        else:
            predictions = self.obj.fit_predict(**fit_params)

        y_name = '_'.join(self.y_columns) if self.y_columns else 'prediction'
        col_name = f"{self.name}__{y_name}"
        column_names = [col_name]
        self.output_vars = column_names

        train_wrapper_class = type(train_X)
        return train_wrapper_class.from_output(predictions, column_names, train_index)

    def process(self, data):
        if 'X' in data:
            data = data['X']
        else:
            data = data['y']
        wrapper_class = type(data)
        data_index = data.get_index()
        data = self.adapter.get_process_data(data)

        if self.method == 'predict':
            predictions = self.obj.predict(data)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        elif self.method == 'predict_proba':
            if not hasattr(self.obj, 'predict_proba'):
                raise Exception(f"Model {self.estimator.__name__} does not support predict_proba")

            predictions = self.obj.predict_proba(data)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'predict' or 'predict_proba'")

        # data의 Wrapper 타입으로 변환
        return wrapper_class.from_output(predictions, column_names, data_index)
