"""
Base adapter class for ML model frameworks
"""

from abc import ABC, abstractmethod

import pandas as pd

GPU_NO = 'no'
GPU_POSSIBLE = 'possible'
GPU_YES = 'yes'


def stack_evals_result(evals_result):
    if not evals_result:
        return pd.Series(dtype='float64')
    return pd.concat(
        [pd.DataFrame({metric: pd.Series(curve) for metric, curve in split.items()})
         .stack().rename(name)
         for name, split in evals_result.items()],
        axis=1,
    ).stack().dropna()


class ModelAdapter(ABC):
    """Abstract base class for ML framework adapters.

    Adapters translate ml-labs' unified data format into the framework-specific
    ``fit()`` parameters of each model. Registered by model class name via
    :func:`~mllabs.adapter.register_adapter`.

    Class Attributes:
        result_objs (dict): ``{key: (callable, mergeable_bool)}`` mapping of
            extractable model attributes for use with
            :class:`~mllabs.collector.ModelAttrCollector`.
    """

    result_objs = {}
    """Abstract base class for model adapters

    각 머신러닝 프레임워크별로 eval_set 처리 방식이 다르므로,
    이를 통일된 인터페이스로 추상화합니다.
    """

    def __init__(self, eval_mode='both', verbose=0.1):
        """Adapter 초기화

        Args:
            eval_mode (str): Evaluation mode
                - 'none' or None: eval_set 없이
                - 'valid': validation set만 전달
                - 'both': train + validation set 전달
            verbose: Verbose 설정
                - 0: 출력 안함
                - 0 < verbose < 1: 전체 진행률을 % 단위로 표시 (주기: verbose * 100%)
                  예: 0.1이면 10%마다, 0.05면 5%마다
                - verbose >= 1: iteration 단위로 표시 (매 verbose번째 iteration)
                  예: 1이면 매 iteration마다, 10이면 매 10 iteration마다
        """
        self.eval_mode = eval_mode
        self.verbose = verbose

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None, single_worker=False):
        """모델의 fit()에 전달할 파라미터를 구성

        Args:
            train_data: {key: data} 형태의 train 데이터 딕셔너리
            valid_data: {key: data} 형태의 valid 데이터 딕셔너리 (Optional)
            params (dict): Processor에서 전달된 추가 파라미터 (Optional, default=None)
            monitor: ProgressMonitor 인스턴스

        Returns:
            dict: fit()에 unpacking으로 전달할 파라미터
                  예: model.fit(**fit_params)
        """
        from .._data_wrapper import unwrap
        fit_params = {}
        if 'X' in train_data:
            fit_params['X'] = unwrap(train_data['X'])
        if 'y' in train_data:
            fit_params['y'] = unwrap(train_data['y'].squeeze())
        if 'sample_weight' in train_data:
            fit_params['sample_weight'] = unwrap(train_data['sample_weight'].squeeze())
        return fit_params

    @staticmethod
    def _eval_weight_list(train_weight, valid_weight, eval_mode):
        """Weights positioned to match the ``eval_set`` list ``eval_mode``
        builds — ``[valid]`` for ``'valid'``, ``[train, valid]`` for
        ``'both'`` — for adapters (LightGBM, XGBoost) that take eval weights
        as a list parallel to ``eval_set`` rather than folded into a Pool.

        ``None`` for a side with no ``sample_weight`` edge is left in place
        rather than dropped — LightGBM/XGBoost both treat a ``None`` entry as
        uniform weight for just that eval set, not an error. Returns
        ``None`` (skip the param entirely) only when neither side has one,
        so the unweighted call shape is unchanged when no one asked for this.
        """
        if train_weight is None and valid_weight is None:
            return None
        if eval_mode == 'valid':
            return [valid_weight]
        if eval_mode == 'both':
            return [train_weight, valid_weight]
        return None

    def get_process_data(self, data):
        from .._data_wrapper import unwrap
        return unwrap(data)

    def get_gpu_usage(self, params):
        """Returns whether the current params will use GPU.

        Returns:
            'no'      — GPU not used
            'possible' — GPU may be used (framework auto-selects based on hardware)
            'yes'     — GPU will definitely be used
        """
        gpu = (params or {}).get('gpu', 'auto')
        if gpu is None:
            return GPU_NO
        if gpu == 'auto':
            return GPU_NO
        return GPU_YES

    def get_params(self, params, gpu_id_list=None, monitor=None, single_worker=False):
        """모델 생성자에 전달할 파라미터를 조정

        Args:
            params (dict): 원본 파라미터

        Returns:
            dict: 조정된 파라미터. gpu 키는 제거됨
        """
        if params is None:
            return params
        params = params.copy()
        params.pop('gpu', None)
        return params

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return id(self)
