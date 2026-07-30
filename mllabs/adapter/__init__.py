"""
Model adapters for handling eval_set in different ML frameworks
"""

import importlib

from ._base import ModelAdapter, GPU_NO, GPU_POSSIBLE, GPU_YES
from ._gpu import get_gpus, get_idle_gpu
from ._catboost import CatBoostAdapter
from ._keras import KerasAdapter
from ._default import DefaultAdapter
from ._sklearn import LMAdapter, PCAAdapter, LDAAdapter, DecisionTreeAdapter

# NNAdapter (._nn) imports TensorFlow at module load. Keep it out of eager
# import so ``import mllabs`` does not pull in TF; it is loaded lazily on first
# use (see _LAZY_ADAPTERS / get_adapter / __getattr__).

try:
    from ._xgboost import XGBoostAdapter
except ImportError:
    XGBoostAdapter = None

try:
    from ._lightgbm import LightGBMAdapter
except ImportError:
    LightGBMAdapter = None

# Model adapter registry (인스턴스 저장)
# 모델 클래스명을 키로, 해당 어댑터 인스턴스를 값으로 매핑
# 기본 설정: eval_mode='both', verbose=0.1
MODEL_ADAPTERS = {
    'CatBoostClassifier': CatBoostAdapter(),
    'CatBoostRegressor': CatBoostAdapter(),
    'CatBoostRanker': CatBoostAdapter(),

    'KerasClassifier': KerasAdapter(),
    'KerasRegressor': KerasAdapter(),

    'LinearRegression': LMAdapter(),
    'LogisticRegression': LMAdapter(),

    'PCA': PCAAdapter(),

    'LinearDiscriminantAnalysis': LDAAdapter(),

    'DecisionTreeClassifier': DecisionTreeAdapter(),
    'DecisionTreeRegressor': DecisionTreeAdapter(),
}

if XGBoostAdapter is not None:
    MODEL_ADAPTERS.update({
        'XGBClassifier': XGBoostAdapter(),
        'XGBRegressor': XGBoostAdapter(),
        'XGBRFClassifier': XGBoostAdapter(),
        'XGBRFRegressor': XGBoostAdapter(),
    })

if LightGBMAdapter is not None:
    MODEL_ADAPTERS.update({
        'LGBMClassifier': LightGBMAdapter(),
        'LGBMRegressor': LightGBMAdapter(),
        'LGBMRanker': LightGBMAdapter(),
    })


# model_name -> (submodule, class attr) for adapters whose import loads a heavy
# optional dependency (TensorFlow). Instantiated and cached into MODEL_ADAPTERS
# only on first use so ``import mllabs`` stays TF-free.
_LAZY_ADAPTERS = {
    'NNClassifier': ('_nn', 'NNAdapter'),
    'NNRegressor': ('_nn', 'NNAdapter'),
}


def _load_lazy_adapter(model_name):
    submod, cls_name = _LAZY_ADAPTERS[model_name]
    cls = getattr(importlib.import_module(f'{__name__}.{submod}'), cls_name)
    adapter = cls()
    MODEL_ADAPTERS[model_name] = adapter
    return adapter


def get_adapter(model_or_name):
    """모델 또는 모델명에 해당하는 어댑터 인스턴스를 반환

    Args:
        model_or_name: Model instance or model class name (str)

    Returns:
        ModelAdapter: Corresponding adapter instance, or DefaultAdapter instance if not found

    Example:
        >>> from xgboost import XGBClassifier
        >>> adapter = get_adapter(XGBClassifier)
        >>> # or
        >>> adapter = get_adapter('XGBClassifier')
    """
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        # model instance or class
        if hasattr(model_or_name, '__name__'):
            model_name = model_or_name.__name__
        else:
            model_name = model_or_name.__class__.__name__

    if model_name in MODEL_ADAPTERS:
        return MODEL_ADAPTERS[model_name]
    if model_name in _LAZY_ADAPTERS:
        return _load_lazy_adapter(model_name)
    return DefaultAdapter()


def __getattr__(name):
    # PEP 562: expose TF-backed adapter classes without importing TF at package
    # load. ``from mllabs.adapter import NNAdapter`` triggers TF only here.
    if name == 'NNAdapter':
        from ._nn import NNAdapter
        return NNAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_adapter(model_name, adapter):
    """새로운 어댑터를 레지스트리에 등록

    Args:
        model_name (str): Model class name
        adapter (ModelAdapter): Adapter instance

    Example:
        >>> class MyCustomAdapter(ModelAdapter):
        ...     def get_fit_params(self, X_train, y_train, X_eval=None, y_eval=None, params=None):
        ...         # custom implementation
        ...         return {...}
        >>>
        >>> register_adapter('MyCustomModel', MyCustomAdapter())
    """
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"adapter must be an instance of ModelAdapter, got {type(adapter)}")

    MODEL_ADAPTERS[model_name] = adapter


__all__ = [
    'ModelAdapter',
    'GPU_NO',
    'GPU_POSSIBLE',
    'GPU_YES',
    'get_gpus',
    'get_idle_gpu',
    'XGBoostAdapter',
    'LightGBMAdapter',
    'CatBoostAdapter',
    'KerasAdapter',
    'NNAdapter',
    'DefaultAdapter',
    'LMAdapter',
    'PCAAdapter',
    'LDAAdapter',
    'DecisionTreeAdapter',
    'MODEL_ADAPTERS',
    'get_adapter',
    'register_adapter',
]
