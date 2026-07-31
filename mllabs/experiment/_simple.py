from itertools import product

from ._base import BaseExperiment
from ._trial import Trial
from .._pipeline import _validate_processor, _validate_adapter, _validate_params


class SimpleExperiment(BaseExperiment):
    """Fixed processor/adapter/edges, hyperparameters swept as a grid.

    Every Trial shares the same processor, method, adapter and edges; only
    *param_grid* varies. ``params`` holds the values that stay constant across
    the sweep::

        exp = SimpleExperiment(
            'lgbm',
            processor='lightgbm.LGBMClassifier',
            edges={'X': 'scaler:(*)', 'y': '{target}'},
            params={'random_state': 42},
            param_grid={'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        )
        exp.get_trial_nums()   # 4

    Trial order is deterministic (grid keys sorted, values in the order given),
    so the same definition always produces the same trial names.

    Args:
        name (str): Experiment name. Also the Trial ``label`` and name prefix.
        processor (str): ``"module.ClassName"`` reference.
        edges (dict): ``{key: dsl_string}`` shared by every trial.
        method (str): Processor method. Default ``'predict'``.
        adapter: ``None`` / string ref / ``{"__ref__": ...}`` spec.
        params (dict, optional): Params fixed across the whole sweep.
        param_grid (dict, optional): ``{param: [values]}`` — cartesian product.
            Empty/omitted yields a single trial from ``params`` alone.
        collectors (list[str], optional): Names of Collectors to report into.
        tags (list[str], optional): Tags applied to every produced Trial.
    """

    def __init__(self, name, processor, edges, method='predict', adapter=None,
                 params=None, param_grid=None, collectors=None, tags=None):
        super().__init__(name, collectors=collectors, tags=tags)

        where = f"SimpleExperiment({name!r})"
        _validate_processor(processor, where)
        _validate_adapter(adapter, where)
        _validate_params(params, where)
        _validate_params(param_grid, where)

        if not isinstance(edges, dict) or not edges:
            raise ValueError(f"{where}: edges must be a non-empty {{key: dsl_string}} dict")
        for key, dsl_string in edges.items():
            if not isinstance(dsl_string, str):
                raise TypeError(
                    f"{where}: edges[{key!r}] must be a DSL string, got "
                    f"{type(dsl_string).__name__}"
                )

        param_grid = dict(param_grid or {})
        for key, values in param_grid.items():
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    f"{where}: param_grid[{key!r}] must be a list of values, got "
                    f"{type(values).__name__}"
                )
            if not values:
                raise ValueError(f"{where}: param_grid[{key!r}] is empty")

        self.processor = processor
        self.edges = dict(edges)
        self.method = method
        self.adapter = adapter
        self.params = dict(params or {})
        self.param_grid = param_grid

        self._grid_keys = sorted(param_grid)
        self._combos = [
            dict(zip(self._grid_keys, values))
            for values in product(*(param_grid[k] for k in self._grid_keys))
        ]
        self._cursor = 0

    def get_trial_nums(self):
        return len(self._combos)

    def reset(self):
        self._cursor = 0

    def get_next_trial(self):
        if self._cursor >= len(self._combos):
            raise StopIteration(
                f"SimpleExperiment({self.name!r}) has no trial left "
                f"({len(self._combos)} total) — call reset() to rewind"
            )
        idx = self._cursor
        self._cursor += 1
        return self._build_trial(idx)

    def get_trial(self, idx):
        """Trial at *idx* without moving the cursor."""
        if not 0 <= idx < len(self._combos):
            raise IndexError(f"trial index {idx} out of range (0..{len(self._combos) - 1})")
        return self._build_trial(idx)

    def _build_trial(self, idx):
        params = dict(self.params)
        params.update(self._combos[idx])
        return Trial(
            name=self._trial_name(idx),
            processor=self.processor,
            edges=dict(self.edges),
            method=self.method,
            adapter=self.adapter,
            params=params,
            label=self.name,
            tag=list(self.tags),
        )

    def _trial_name(self, idx):
        if len(self._combos) == 1:
            return self.name
        width = len(str(len(self._combos) - 1))
        return f"{self.name}_{idx:0{width}d}"
