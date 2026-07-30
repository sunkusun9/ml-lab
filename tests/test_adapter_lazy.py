import subprocess
import sys

import pytest


def _run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)


class TestLazyTensorFlow:
    def test_import_mllabs_does_not_load_tensorflow(self):
        # A fresh interpreter: importing mllabs must not pull in TensorFlow.
        r = _run(
            "import sys, mllabs; "
            "assert 'tensorflow' not in sys.modules, 'tensorflow eagerly imported'; "
            "print('ok')"
        )
        assert r.returncode == 0, r.stderr
        assert 'ok' in r.stdout

    def test_get_adapter_lazy_loads_nn_adapter(self):
        from mllabs.adapter import get_adapter
        adapter = get_adapter('NNClassifier')
        assert type(adapter).__name__ == 'NNAdapter'

    def test_get_adapter_caches_lazy_adapter(self):
        from mllabs.adapter import get_adapter, MODEL_ADAPTERS
        get_adapter('NNRegressor')
        assert 'NNRegressor' in MODEL_ADAPTERS

    def test_nn_adapter_importable_via_getattr(self):
        import mllabs.adapter as A
        from mllabs.adapter._nn import NNAdapter as Direct
        assert A.NNAdapter is Direct

    def test_unknown_attr_raises(self):
        import mllabs.adapter as A
        with pytest.raises(AttributeError):
            A.NoSuchAdapter
