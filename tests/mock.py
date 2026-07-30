"""Shared dummy processor classes for tests.

Referenced by tests as ``processor='mock.ClassName'`` string refs (Pipeline
stores/passes processor as a string only — see mllabs/_pipeline.py), so they
need a real importable module path. Centralized here instead of duplicated
per test file.
"""
import numpy as np


class DummyStage:
    """Structural-only placeholder (Pipeline graph tests never build/fit it)."""
    __name__ = 'DummyStage'


class DummyHead:
    """Structural-only placeholder (Pipeline graph tests never build/fit it)."""
    __name__ = 'DummyHead'


class AnotherProcessor:
    """Structural-only placeholder (Pipeline graph tests never build/fit it)."""
    __name__ = 'AnotherProcessor'


class _DummyProc:
    def fit_transform(self, X, y=None):
        return X


class BadProcessor:
    __name__ = 'BadProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise ValueError("intentional error")
    def transform(self, X):
        pass


class BadPredictor:
    __name__ = 'BadPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise RuntimeError("predict error")
    def predict(self, X):
        pass


class ErrorProcessor:
    __name__ = 'ErrorProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise TypeError("test error msg")
    def transform(self, X):
        pass


class FailPredictor:
    __name__ = 'FailPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise RuntimeError("fail")
    def predict(self, X):
        pass


class NativeChatterStage:
    """Writes to OS-level fd 1/2 (like a native lib), bypassing Python stdout."""
    __name__ = 'NativeChatterStage'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        import os
        os.write(1, b'NATIVE_STDOUT_XYZ\n')
        os.write(2, b'NATIVE_STDERR_XYZ\n')
        return self
    def transform(self, X):
        return X


class WarnStage:
    """Emits a Python warning during fit (captured into info['warnings'])."""
    __name__ = 'WarnStage'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        import warnings
        warnings.warn("WORKER_WARN_ABC")
        return self
    def transform(self, X):
        return X


class WarnPredictor:
    """Emits a Python warning during predict (i.e. at collector/process time)."""
    __name__ = 'WarnPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        import warnings
        warnings.warn("PREDICT_WARN_XYZ")
        return np.zeros(len(X), dtype=int)
