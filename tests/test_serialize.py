import pytest
import json

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mllabs._serialize import (
    serialize_value, deserialize_value,
    serialize_to_json, deserialize_from_json,
    resolve_processor, resolve_adapter, resolve_ref_values,
    _obj_to_ref, _ref_to_obj,
)


# ---------------------------------------------------------------------------
# _obj_to_ref / _ref_to_obj
# ---------------------------------------------------------------------------

class TestObjRef:
    def test_class_to_ref(self):
        ref = _obj_to_ref(StandardScaler)
        assert 'StandardScaler' in ref
        assert '.' in ref

    def test_ref_roundtrip_class(self):
        ref = _obj_to_ref(StandardScaler)
        assert _ref_to_obj(ref) is StandardScaler

    def test_ref_roundtrip_function(self):
        from sklearn.metrics import accuracy_score
        ref = _obj_to_ref(accuracy_score)
        assert _ref_to_obj(ref) is accuracy_score

    def test_ref_invalid_raises(self):
        with pytest.raises(Exception):
            _ref_to_obj("not.a.real.module.Thing")


# ---------------------------------------------------------------------------
# resolve_processor / resolve_adapter
# ---------------------------------------------------------------------------

class TestResolveProcessor:
    def test_string_ref(self):
        ref = _obj_to_ref(StandardScaler)
        assert resolve_processor(ref) is StandardScaler

    def test_class_passthrough(self):
        assert resolve_processor(StandardScaler) is StandardScaler

    def test_none_passthrough(self):
        assert resolve_processor(None) is None


class TestResolveAdapter:
    def test_none_passthrough(self):
        assert resolve_adapter(None) is None

    def test_instance_passthrough(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter import LightGBMAdapter
        adapter = LightGBMAdapter(eval_mode='valid')
        assert resolve_adapter(adapter) is adapter

    def test_string_ref_instantiates_with_defaults(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter._lightgbm import LightGBMAdapter
        ref = _obj_to_ref(LightGBMAdapter)
        result = resolve_adapter(ref)
        assert type(result) is LightGBMAdapter
        assert result.eval_mode == 'both'

    def test_ref_dict_with_params(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter._lightgbm import LightGBMAdapter
        ref = _obj_to_ref(LightGBMAdapter)
        result = resolve_adapter({"__ref__": ref, "__params__": {"eval_mode": "valid", "verbose": 0.25}})
        assert type(result) is LightGBMAdapter
        assert result.eval_mode == 'valid'
        assert result.verbose == pytest.approx(0.25)

    def test_ref_dict_without_params(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter._lightgbm import LightGBMAdapter
        ref = _obj_to_ref(LightGBMAdapter)
        result = resolve_adapter({"__ref__": ref})
        assert type(result) is LightGBMAdapter
        assert result.eval_mode == 'both'


class TestResolveRefValues:
    def test_scalars_and_strings_untouched(self):
        params = {'n': 50, 'metric': 'AUC', 'flag': True, 'rate': 0.1, 'none': None}
        assert resolve_ref_values(params) == params

    def test_ref_dict_instantiated(self):
        from mllabs import ColSelector
        params = {'cat_features': {'__ref__': 'mllabs.ColSelector',
                                   '__params__': {'dsl_string': '*@categorical'}}}
        out = resolve_ref_values(params)
        assert isinstance(out['cat_features'], ColSelector)
        assert out['cat_features'].dsl_string == '*@categorical'

    def test_ref_dict_in_list(self):
        from mllabs import ColSelector
        out = resolve_ref_values({'sel': [{'__ref__': 'mllabs.ColSelector',
                                           '__params__': {'dsl_string': '^cat_'}}]})
        assert isinstance(out['sel'][0], ColSelector)
        assert out['sel'][0].dsl_string == '^cat_'

    def test_plain_dict_recursed_not_instantiated(self):
        params = {'nested': {'a': 1, 'b': 'x'}}
        assert resolve_ref_values(params) == params

    def test_bad_params_type_raises(self):
        with pytest.raises(ValueError, match='kwargs dict'):
            resolve_ref_values({'sel': {'__ref__': 'mllabs.ColSelector',
                                        '__params__': '*@categorical'}})

    def test_callable_ref_not_invoked(self):
        from sklearn.metrics import balanced_accuracy_score
        out = resolve_ref_values(
            {'metric_func': {'__callable__': 'sklearn.metrics.balanced_accuracy_score'}})
        assert out['metric_func'] is balanced_accuracy_score

    def test_callable_ref_to_class_not_instantiated(self):
        out = resolve_ref_values({'cls': {'__callable__': _obj_to_ref(StandardScaler)}})
        assert out['cls'] is StandardScaler

    def test_callable_nested_in_plain_dict_and_list(self):
        from sklearn.metrics import accuracy_score
        out = resolve_ref_values({
            'metrics': [{'__callable__': 'sklearn.metrics.accuracy_score'}],
            'cfg': {'fn': {'__callable__': 'sklearn.metrics.accuracy_score'}},
        })
        assert out['metrics'][0] is accuracy_score
        assert out['cfg']['fn'] is accuracy_score


# ---------------------------------------------------------------------------
# serialize_value primitives
# ---------------------------------------------------------------------------

class TestSerializePrimitives:
    def test_none(self):
        assert serialize_value(None) is None

    def test_bool(self):
        assert serialize_value(True) is True
        assert serialize_value(False) is False

    def test_int(self):
        assert serialize_value(42) == 42

    def test_float(self):
        assert serialize_value(3.14) == pytest.approx(3.14)

    def test_str(self):
        assert serialize_value("hello") == "hello"

    def test_list(self):
        assert serialize_value([1, 2, 3]) == [1, 2, 3]

    def test_nested_list(self):
        result = serialize_value([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_dict(self):
        result = serialize_value({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    def test_tuple(self):
        result = serialize_value((1, "a", None))
        assert result == {"__type__": "tuple", "__items__": [1, "a", None]}

    def test_numpy_int(self):
        import numpy as np
        result = serialize_value(np.int64(7))
        assert result == 7
        assert isinstance(result, int)

    def test_numpy_float(self):
        import numpy as np
        result = serialize_value(np.float32(1.5))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# serialize_value classes / callables / instances
# ---------------------------------------------------------------------------

class TestSerializeObjects:
    def test_class(self):
        result = serialize_value(StandardScaler)
        assert result["__type__"] == "class"
        assert "StandardScaler" in result["__ref__"]

    def test_callable_function(self):
        from sklearn.metrics import accuracy_score
        result = serialize_value(accuracy_score)
        assert result["__type__"] == "callable"
        assert "accuracy_score" in result["__ref__"]

    def test_instance_with_get_params(self):
        scaler = StandardScaler(with_std=False)
        result = serialize_value(scaler)
        assert result["__type__"] == "instance"
        assert "StandardScaler" in result["__ref__"]
        assert result["__params__"]["with_std"] is False

    def test_instance_default_params(self):
        clf = DecisionTreeClassifier(max_depth=3)
        result = serialize_value(clf)
        assert result["__type__"] == "instance"
        assert result["__params__"]["max_depth"] == 3

    def test_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda"):
            serialize_value(lambda x: x)

    def test_local_function_raises(self):
        def local():
            pass
        with pytest.raises(ValueError):
            serialize_value(local)

    def test_instance_in_params(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter import LightGBMAdapter
        adapter = LightGBMAdapter()
        result = serialize_value(adapter)
        assert result["__type__"] == "instance"
        assert "LightGBMAdapter" in result["__ref__"]


# ---------------------------------------------------------------------------
# deserialize_value
# ---------------------------------------------------------------------------

class TestDeserializeValue:
    def test_primitives(self):
        for v in [None, True, False, 1, 3.14, "hello"]:
            assert deserialize_value(v) == v

    def test_list(self):
        assert deserialize_value([1, 2, 3]) == [1, 2, 3]

    def test_plain_dict(self):
        assert deserialize_value({"a": 1}) == {"a": 1}

    def test_tuple(self):
        d = {"__type__": "tuple", "__items__": [1, "a", None]}
        result = deserialize_value(d)
        assert result == (1, "a", None)
        assert isinstance(result, tuple)

    def test_class(self):
        d = {"__type__": "class", "__ref__": _obj_to_ref(StandardScaler)}
        assert deserialize_value(d) is StandardScaler

    def test_callable(self):
        from sklearn.metrics import accuracy_score
        d = {"__type__": "callable", "__ref__": _obj_to_ref(accuracy_score)}
        assert deserialize_value(d) is accuracy_score

    def test_instance(self):
        scaler = StandardScaler(with_std=False)
        d = serialize_value(scaler)
        restored = deserialize_value(d)
        assert isinstance(restored, StandardScaler)
        assert restored.with_std is False

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            deserialize_value({"__type__": "bogus", "__ref__": "x"})


# ---------------------------------------------------------------------------
# roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def _rt(self, v):
        return deserialize_value(serialize_value(v))

    def test_none(self):
        assert self._rt(None) is None

    def test_primitives(self):
        for v in [True, 42, 3.14, "hello"]:
            assert self._rt(v) == v

    def test_list(self):
        assert self._rt([1, "a", None]) == [1, "a", None]

    def test_tuple(self):
        result = self._rt((1, "x"))
        assert result == (1, "x")
        assert isinstance(result, tuple)

    def test_nested(self):
        v = {"key": [1, (2, 3), None]}
        result = self._rt(v)
        assert result["key"][0] == 1
        assert result["key"][1] == (2, 3)
        assert result["key"][2] is None

    def test_class_roundtrip(self):
        assert self._rt(StandardScaler) is StandardScaler

    def test_instance_roundtrip(self):
        scaler = StandardScaler(with_mean=False, with_std=True)
        restored = self._rt(scaler)
        assert isinstance(restored, StandardScaler)
        assert restored.with_mean is False
        assert restored.with_std is True

    def test_adapter_roundtrip(self):
        pytest.importorskip("lightgbm")
        from mllabs.adapter import LightGBMAdapter
        adapter = LightGBMAdapter(eval_mode='train', verbose=0.5)
        restored = self._rt(adapter)
        assert type(restored) is LightGBMAdapter
        assert restored.eval_mode == 'train'
        assert restored.verbose == pytest.approx(0.5)

    def test_edges_roundtrip(self):
        edges = {'X': [(None, ['f1', 'f2']), ('scaler', None)], 'y': [(None, 'target')]}
        result = self._rt(edges)
        assert result['X'][0] == (None, ['f1', 'f2'])
        assert result['X'][1] == ('scaler', None)
        assert result['y'][0] == (None, 'target')

    def test_params_with_class_roundtrip(self):
        params = {'max_depth': 3, 'criterion': 'gini', 'random_state': 42}
        assert self._rt(params) == params


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_serialize_to_json_returns_string(self):
        result = serialize_to_json({"a": 1, "b": [2, 3]})
        assert isinstance(result, str)
        assert json.loads(result) == {"a": 1, "b": [2, 3]}

    def test_deserialize_from_json_none(self):
        assert deserialize_from_json(None) is None

    def test_json_roundtrip_instance(self):
        scaler = StandardScaler(with_std=False)
        s = serialize_to_json(scaler)
        restored = deserialize_from_json(s)
        assert isinstance(restored, StandardScaler)
        assert restored.with_std is False

    def test_json_roundtrip_tuple_in_edges(self):
        edges = {'X': [(None, ['f1', 'f2'])]}
        s = serialize_to_json(edges)
        restored = deserialize_from_json(s)
        assert restored['X'][0] == (None, ['f1', 'f2'])
        assert isinstance(restored['X'][0], tuple)
