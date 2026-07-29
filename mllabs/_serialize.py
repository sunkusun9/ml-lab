import importlib
import json

try:
    import numpy as _np
    _NP_INT = (_np.integer,)
    _NP_FLOAT = (_np.floating,)
except ImportError:
    _NP_INT = ()
    _NP_FLOAT = ()


def _obj_to_ref(obj):
    return f"{obj.__module__}.{obj.__qualname__}"


def _ref_to_obj(ref):
    module_path, _, attr = ref.rpartition('.')
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def resolve_processor(processor):
    """Resolve a ``"module.ClassName"`` string reference to the actual class."""
    if isinstance(processor, str):
        return _ref_to_obj(processor)
    return processor


def serialize_value(v):
    """Reduce any Python value to a JSON-serializable form.

    Supported:
      - None, bool, int, float, str  → as-is
      - numpy integer/float          → int/float
      - list                         → list (elements recursed)
      - tuple                        → {"__type__": "tuple", "__items__": [...]}
      - dict                         → dict (values recursed)
      - type (class)                 → {"__type__": "class",    "__ref__": "mod.Cls"}
      - importable callable          → {"__type__": "callable", "__ref__": "mod.fn"}
      - instance w/ get_params       → {"__type__": "instance", "__ref__": "...", "__params__": {...}}
      - instance w/ __dict__         → same, using __dict__
      - lambda / local function      → ValueError
    """
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, _NP_INT):
        return int(v)
    if isinstance(v, _NP_FLOAT):
        return float(v)
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, tuple):
        return {"__type__": "tuple", "__items__": [serialize_value(x) for x in v]}
    if isinstance(v, list):
        return [serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: serialize_value(val) for k, val in v.items()}
    if isinstance(v, type):
        return {"__type__": "class", "__ref__": _obj_to_ref(v)}
    if callable(v):
        qualname = getattr(v, '__qualname__', '')
        if '<lambda>' in qualname or '<locals>' in qualname:
            raise ValueError(
                f"Cannot serialize lambda or local function: {v!r}. "
                "Use an importable function or class instead."
            )
        return {"__type__": "callable", "__ref__": _obj_to_ref(v)}
    if hasattr(v, '__class__'):
        ref = _obj_to_ref(type(v))
        if hasattr(v, 'get_params'):
            try:
                raw = v.get_params()
            except TypeError:
                raw = v.__dict__ if hasattr(v, '__dict__') else {}
        elif hasattr(v, '__dict__'):
            raw = v.__dict__
        else:
            raise ValueError(f"Cannot serialize {type(v)!r}: {v!r}")
        return {
            "__type__": "instance",
            "__ref__": ref,
            "__params__": {k: serialize_value(val) for k, val in raw.items()},
        }
    raise ValueError(f"Cannot serialize {type(v)!r}: {v!r}")


def deserialize_value(d):
    """Restore a value produced by serialize_value back to a Python object."""
    if d is None or isinstance(d, (bool, int, float, str)):
        return d
    if isinstance(d, list):
        return [deserialize_value(x) for x in d]
    if not isinstance(d, dict):
        return d
    typ = d.get("__type__")
    if typ is None:
        return {k: deserialize_value(v) for k, v in d.items()}
    if typ == "tuple":
        return tuple(deserialize_value(x) for x in d["__items__"])
    if typ == "class":
        return _ref_to_obj(d["__ref__"])
    if typ == "callable":
        return _ref_to_obj(d["__ref__"])
    if typ == "instance":
        cls = _ref_to_obj(d["__ref__"])
        params = {k: deserialize_value(v) for k, v in d.get("__params__", {}).items()}
        return cls(**params)
    raise ValueError(f"Unknown __type__ marker: {typ!r}")


def serialize_to_json(v):
    """serialize_value → compact JSON string."""
    return json.dumps(serialize_value(v), ensure_ascii=False, separators=(',', ':'))


def deserialize_from_json(s):
    """Compact JSON string → deserialize_value."""
    if s is None:
        return None
    return deserialize_value(json.loads(s))
