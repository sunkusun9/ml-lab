"""
Named column-selector functions referenced via '@name' in _edge_dsl patterns.
"""
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

_REGISTRY = {}


def col_selector(*processor_classes, name=None):
    """Register a column-selector function for use as ``'@name'`` in edges/output_var.

    Every registered function has the same signature — ``(data,
    processor=None) -> mask`` — ``data`` is a ``DataWrapper`` already narrowed
    to whatever candidate columns the caller passed in (e.g. via a preceding
    regex pattern); ``data.get_columns()`` gives the candidate names in order.
    ``processor_classes`` restricts which ``processor.obj`` types the selector
    is valid for; ``resolve_selector`` raises ``ValueError`` if it doesn't
    match. ``name`` overrides the registry key (default: ``func.__name__``) —
    needed for selectors named after Python builtins (``float``/``int``/``string``).
    """
    def deco(func):
        _REGISTRY[name or func.__name__] = (func, processor_classes)
        return func
    return deco


def parse_selector_ref(ref):
    """Parse ``'@name'`` or ``'@name()'`` into the bare selector name."""
    body = ref[1:]
    if body.endswith('()'):
        body = body[:-2]
    if not body.isidentifier():
        raise ValueError(f"Invalid column selector reference: {ref!r}")
    return body


def resolve_selector(ref, data, processor):
    """Resolve ``'@name'``/``'@name()'`` against ``data``/``processor`` into a column list.

    ``data`` is already narrowed to the pattern-matched candidate columns.
    """
    name = parse_selector_ref(ref)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown column selector {name!r}. Known: {sorted(_REGISTRY)}")
    func, allowed = _REGISTRY[name]
    if allowed:
        if processor is None:
            raise ValueError(f"Column selector {name!r} requires a processor")
        if not isinstance(processor.obj, allowed):
            raise ValueError(
                f"Column selector {name!r} requires processor "
                f"{tuple(c.__name__ for c in allowed)}, got {type(processor.obj).__name__}"
            )
    columns = data.get_columns()
    mask = func(data, processor=processor)
    return [col for col, keep in zip(columns, mask) if keep]


@col_selector(OneHotEncoder)
def ohe_drop_first(data, processor):
    columns = data.get_columns()
    # 각 원래 변수에 대해 첫 번째 컬럼을 만났는지 추적
    org_X = processor.X_
    first_seen = {var: False for var in org_X}

    mask = []
    for col in columns:
        # '__' 뒤의 부분 추출
        if '__' not in col:
            mask.append(False)
            continue

        suffix = col.split('__', 1)[1]

        # 어떤 org_X 변수에 속하는지 확인
        matched = False
        for org_var in org_X:
            # suffix가 org_var로 시작하는지 확인 (예: color_red는 color로 시작)
            if suffix.startswith(f"{org_var}_"):
                matched = True
                if not first_seen[org_var]:
                    # 첫 번째 컬럼은 제외 (False)
                    mask.append(False)
                    first_seen[org_var] = True
                else:
                    # 나머지는 포함 (True)
                    mask.append(True)
                break

        # org_X의 어떤 변수에도 속하지 않으면 제외
        if not matched:
            mask.append(False)

    return mask

def _polynomial_feature_names(input_features, degree=2, interaction_only=False, include_bias=True):
    from itertools import combinations, combinations_with_replacement
    n_features = len(input_features)
    feature_names = []
    if include_bias:
        feature_names.append("1")
    comb_func = combinations if interaction_only else combinations_with_replacement
    for d in range(1, degree + 1):
        for comb in comb_func(range(n_features), d):
            counts = {}
            for idx in comb:
                counts[idx] = counts.get(idx, 0) + 1
            terms = []
            for idx, power in counts.items():
                name = input_features[idx]
                if power > 1:
                    name = f"{name}^{power}"
                terms.append(name)
            feature_names.append(" ".join(terms))
    return feature_names

def _vars_in_suffix(suffix, org_X):
    """Extract the origin var name(s) mentioned in a polynomial feature suffix
    (e.g. 'v1' -> {'v1'}, 'v1^2' -> {'v1'}, 'v1 v2' -> {'v1', 'v2'})."""
    names = set()
    for term in suffix.split(' '):
        name = term.split('^', 1)[0]
        if name in org_X:
            names.add(name)
    return names


@col_selector(PolynomialFeatures)
def subset_poly(data, processor=None):
    """Snap `data`'s columns (already narrowed by a preceding pattern) to the
    full, degree/interaction/bias-consistent polynomial feature set for
    whichever origin variables those columns mention."""
    columns = data.get_columns()
    obj = processor.obj
    degree = obj.degree if hasattr(obj, 'degree') else 2
    interaction_only = obj.interaction_only if hasattr(obj, 'interaction_only') else False
    include_bias = obj.include_bias if hasattr(obj, 'include_bias') else True

    org_X = processor.X_
    node_name = processor.name
    prefix = f"{node_name}__"

    mentioned = set()
    for col in columns:
        if col.startswith(prefix):
            mentioned |= _vars_in_suffix(col[len(prefix):], org_X)
    vars_ = [x for x in org_X if x in mentioned]

    subset_names = set(_polynomial_feature_names(
        vars_, degree=degree, interaction_only=interaction_only, include_bias=include_bias
    ))

    mask = []
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            mask.append(suffix in subset_names)
        else:
            mask.append(False)
    return mask

_DTYPE_SELECTOR_KINDS = {
    'numeric': 'numeric',
    'categorical': 'category',
    'binary': 'bool',
    'float': 'float',
    'int': 'int',
    'string': 'str',
}


def _make_dtype_selector(kind):
    def selector(data, processor=None):
        matched = set(data.select_by_dtype(kind))
        return [c in matched for c in data.get_columns()]
    return selector


for _name, _kind in _DTYPE_SELECTOR_KINDS.items():
    col_selector(name=_name)(_make_dtype_selector(_kind))
del _name, _kind


def get_origin_var(columns, org_X):
    l = list()
    for col in columns:
        suffix = col.split('__', 1)[-1]
        org = "Unknown"
        for org_var in org_X:
            if suffix.startswith(f'{org_var}_') or suffix == org_var:
                org = org_var
                break
        l.append(org)
    return l
