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

    def test_set_node_with_nn_named_processor_does_not_load_tensorflow(self):
        # PipelineBuilder.set_grp/set_node must not resolve the by-processor-class
        # default adapter eagerly — that used to import TensorFlow just from
        # defining a node, before any build/exp ever ran.
        code = (
            "import sys\n"
            "from mllabs import PipelineBuilder\n"
            "class NNClassifier:\n"
            "    __name__ = 'NNClassifier'\n"
            "    def __init__(self, **kwargs): pass\n"
            "p = PipelineBuilder()\n"
            "p.set_datasource({'x1': 'numerical', 'target': 'numerical'})\n"
            "p.set_grp('g1', processor='mllabs.nn.NNClassifier', method='predict',\n"
            "          edges={'X': '{x1}', 'y': '{target}'})\n"
            "p.set_node('n1', grp='g1')\n"
            "p.get_node_attrs('n1')\n"
            "assert 'tensorflow' not in sys.modules, 'tensorflow eagerly imported'\n"
            "print('ok')\n"
        )
        r = _run(code)
        assert r.returncode == 0, r.stderr
        assert 'ok' in r.stdout
