import re
from ._data_wrapper import unwrap

def resolve_columns(data, X, y=None, processor=None):
    """X와 y를 실제 컬럼 리스트로 변환"""
    columns = data.get_columns()

    # y 처리 (y가 있으면 X에서 제외할 컬럼)
    y_cols = []
    if y is not None:
        if isinstance(y, slice):
            y_cols = columns[y]
        elif isinstance(y, list):
            y_cols = y
        else:
            y_cols = [y]

    # X 처리
    if X is None:
        # y가 있으면 y를 제외한 모든 컬럼
        if y is not None:
            return [col for col in columns if col not in y_cols]
        else:
            return columns
    elif isinstance(X, str):
        # 정규 표현식 패턴이면 매칭되는 컬럼만 선택
        return [col for col in columns if re.match(X, col)]
    elif callable(X):
        mask = X(columns, processor=processor)
        return [col for col, keep in zip(columns, mask) if keep]
    elif isinstance(X, slice):
        # slice 객체면 컬럼을 슬라이싱
        return columns[X]
    elif isinstance(X, tuple):
        if len(X) == 0:
            return []
        head = X[0]
        if callable(head):
            mask = head(columns, *X[1:], processor=processor)
            return [col for col, keep in zip(columns, mask) if keep]
        elif head == 'cup':
            seen = set()
            ret = list()
            for x in X[1:]:
                for col in resolve_columns(data, x, processor=processor):
                    if col not in seen:
                        seen.add(col)
                        ret.append(col)
            return ret
        elif head == 'cap':
            sets = [set(resolve_columns(data, x, processor=processor)) for x in X[1:]]
            if not sets:
                return []
            common = sets[0]
            for s in sets[1:]:
                common &= s
            first = resolve_columns(data, X[1], processor=processor)
            return [col for col in first if col in common]
        else:
            raise ValueError(f"tuple의 첫번째 요소가 callable이 아닌 경우 'cup' 또는 'cap'이어야 함: {head}")
    elif isinstance(X, list):
        for x in X:
            if x is None or isinstance(x, list):
                raise ValueError(f"list 요소에 list나 None은 허용되지 않음: {x}")
        seen = set()
        ret = list()
        for x in X:
            for col in resolve_columns(data, x, processor=processor):
                if col not in seen:
                    seen.add(col)
                    ret.append(col)
        return ret
    else:
        # 단일 값이면 리스트로 변환
        ret = list()
        for col in columns:
            if re.match(X, col):
                ret.append(col)
                break
        return ret

class TransformProcessor():
    def __init__(self, name, transformer, adapter = None, params = {}, logger = None):
        self.name = name
        self.transformer = transformer
        self.params = params
        self.adapter = adapter
        self.output_vars = None
        self.logger = logger

    def fit(self, data_dict):
        # X key로 데이터 가져오기
        train_X, train_v_X = data_dict['X']
        self.X_ = train_X.get_columns()

        # y key로 데이터 가져오기 (있으면)
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
            self.y_columns = train_y.get_columns()
        else:
            train_y, train_v_y = None, None
            self.y_columns = None

        params = self.adapter.get_params(self.params, logger = self.logger) if self.adapter is not None else self.params
        self.obj = self.transformer(**params)

        # adapter에서 fit_params 생성
        if self.adapter is not None:
            fit_params = self.adapter.get_fit_params(
                data_dict=data_dict, X=self.X_, y=self.y_columns, params=self.params,logger=self.logger
            )
        else:
            fit_params = {}

        # DataWrapper에서 native로 변환
        train_X_native = unwrap(train_X)

        if train_y is None:
            self.obj.fit(train_X_native, **fit_params)
        else:
            train_y_native = unwrap(train_y.squeeze())
            self.obj.fit(train_X_native, train_y_native, **fit_params)

        # 컬럼명 결정 (get_feature_names_out이 있으면 사용)
        if hasattr(self.obj, 'get_feature_names_out'):
            column_names = self.obj.get_feature_names_out().tolist()
            column_names = [f"{self.name}__{col}" for col in column_names]
        else:
            column_names = None

        if column_names is not None:
            self.output_vars = column_names
        return self

    def fit_process(self, data_dict):
        # X key로 데이터 가져오기
        train_X, train_v_X = data_dict['X']
        self.X_ = train_X.get_columns()
        train_index = train_X.get_index()

        # y key로 데이터 가져오기 (있으면)
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
            self.y_columns = train_y.get_columns()
        else:
            train_y, train_v_y = None, None
            self.y_columns = None

        params = self.adapter.get_params(self.params, logger = self.logger) if self.adapter is not None else self.params
        self.obj = self.transformer(**params)

        # adapter에서 fit_params 생성
        if self.adapter is not None:
            fit_params = self.adapter.get_fit_params(
                data_dict=data_dict, X=self.X_, y=self.y_columns, params=self.params, logger=self.logger
            )
        else:
            fit_params = {}

        # DataWrapper에서 native로 변환
        train_X_native = unwrap(train_X)

        if train_y is None:
            result = self.obj.fit_transform(train_X_native, **fit_params)
        else:
            train_y_native = unwrap(train_y.squeeze())
            result = self.obj.fit_transform(train_X_native, train_y_native, **fit_params)

        # train의 Wrapper 타입으로 변환
        train_wrapper_class = type(train_X)
        # 컬럼명 결정 (get_feature_names_out이 있으면 사용)
        if hasattr(self.obj, 'get_feature_names_out'):
            column_names = self.obj.get_feature_names_out().tolist()
            column_names = [f"{self.name}__{col}" for col in column_names]
        else:
            column_names = None

        if column_names is not None:
            self.output_vars = column_names
        return train_wrapper_class.from_output(result, self.output_vars, train_index)

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data)
        data_index = data.get_index()

        result = self.obj.transform(data_X)

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(result, self.output_vars, data_index)

class PredictProcessor():
    def __init__(self, name, estimator, method='predict', adapter = None, params = {}, logger = None):
        self.name = name
        self.estimator = estimator
        self.params = params
        self.method = method
        self.output_vars = None
        self.adapter = adapter
        self.y_columns = None
        self.logger = logger

    def fit(self, data_dict):
        # X key로 데이터 가져오기
        train_X, train_v_X = data_dict['X']
        self.X_ = train_X.get_columns()

        # y key로 데이터 가져오기 (있으면)
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
            self.y_columns = train_y.get_columns()
        else:
            train_y, train_v_y = None, None
            self.y_columns = None

        # adapter가 있으면 params 조정 (callbacks 등 설정)
        params = self.adapter.get_params(self.params, logger = self.logger) if self.adapter is not None else self.params
        self.obj = self.estimator(**params)

        # adapter에서 fit_params 생성
        if self.adapter is not None:
            fit_params = self.adapter.get_fit_params(
                data_dict=data_dict, X=self.X_, y=self.y_columns, params=self.params, logger=self.logger
            )
        else:
            fit_params = {}

        # DataWrapper에서 native로 변환
        train_X_native = unwrap(train_X)

        if train_y is None:
            # 비지도학습
            self.obj.fit(train_X_native, **fit_params)
        else:
            train_y_native = unwrap(train_y.squeeze())
            # 지도학습
            self.obj.fit(train_X_native, train_y_native, **fit_params)

        if self.method == 'predict':
            # y 변수명 결정
            if self.y_columns is None:
                y_name = 'prediction'
            else:
                y_name = '_'.join(self.y_columns)

            col_name = f"{self.name}__{y_name}"
            self.output_vars = [col_name]
        elif self.method == 'predict_proba':
            # y 변수명 결정
            if self.y_columns is None:
                y_name = 'prediction'
            else:
                y_name = '_'.join(self.y_columns)

            columns = [f"{self.name}__{y_name}_{i}" for i in self.obj.classes_]
            self.output_vars = columns
        return self

    def fit_process(self, data_dict):
        # X key로 데이터 가져오기
        train_X, train_v_X = data_dict['X']
        self.X_ = train_X.get_columns()
        train_index = train_X.get_index()

        # y key로 데이터 가져오기 (있으면)
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
            self.y_columns = train_y.get_columns()
        else:
            train_y, train_v_y = None, None
            self.y_columns = None

        # adapter가 있으면 params 조정 (callbacks 등 설정)
        params = self.adapter.get_params(self.params, logger = self.logger) if self.adapter is not None else self.params
        self.obj = self.estimator(**params)

        # adapter에서 fit_params 생성
        if self.adapter is not None:
            fit_params = self.adapter.get_fit_params(
                data_dict=data_dict, X=self.X_, y=self.y_columns, params=self.params,
                logger=self.logger
            )
        else:
            fit_params = {}

        # DataWrapper에서 native로 변환
        train_X_native = unwrap(train_X)

        if train_y is None:
            # 비지도학습
            predictions = self.obj.fit_predict(train_X_native, **fit_params)
        else:
            # 지도학습
            train_y_native = unwrap(train_y.squeeze())
            predictions = self.obj.fit_predict(train_X_native, train_y_native, **fit_params)

        # 컬럼명 결정
        if self.y_columns is None:
            y_name = 'prediction'
        else:
            y_name = '_'.join(self.y_columns)

        col_name = f"{self.name}__{y_name}"
        column_names = [col_name]
        self.output_vars = column_names

        # train의 Wrapper 타입으로 변환
        train_wrapper_class = type(train_X)
        return train_wrapper_class.from_output(predictions, column_names, train_index)

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data)
        data_index = data.get_index()

        if self.method == 'predict':
            predictions = self.obj.predict(data_X)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        elif self.method == 'predict_proba':
            if not hasattr(self.obj, 'predict_proba'):
                raise Exception(f"Model {self.estimator.__name__} does not support predict_proba")

            predictions = self.obj.predict_proba(data_X)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'predict' or 'predict_proba'")

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(predictions, column_names, data_index)
