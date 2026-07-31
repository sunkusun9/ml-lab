import json
from pathlib import Path

from .._serialize import resolve_processor, resolve_instance, resolve_ref_values, _obj_to_ref, _ref_to_obj


class Collectors:
    """Registry that owns Collector instances and their storage.

    An :class:`~mllabs.BaseExperiment` records only the *names* of the
    Collectors it reports into; the instances live here. That split means one
    registry can serve several Experiments — so their metrics land in the same
    place and stay comparable — and it keeps an Experiment pure definition
    (trial space + collector names), with nothing live inside it.

    Args:
        path (str | Path, optional): Base directory. A Collector registered
            without an explicit ``path`` gets ``{path}/{name}``.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.collectors = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def set_collector(self, name, collector, connector, path=None, params=None, exist='skip'):
        """Build and register a Collector from its parts.

        Args:
            name (str): Collector name — what an Experiment refers to it by.
            collector: Collector class, or ``"module.ClassName"`` string reference.
            connector: :class:`~mllabs.Connector` instance, or
                ``{"__ref__": ..., "__params__": {...}}`` spec.
            path (str | Path, optional): Storage directory. Defaults to
                ``{registry path}/{name}``; required if the registry has no path.
            params (dict): Remaining constructor parameters (everything after
                ``name``/``connector``). ``{"__callable__": "mod.fn"}`` resolves
                to the referenced object (not called); ``{"__ref__": "mod.Cls",
                "__params__": {...}}`` is instantiated.
            exist (str): ``'skip'`` (default) returns the existing collector;
                ``'error'`` raises; ``'replace'`` drops and rebuilds it.

        Returns:
            Collector: The registered collector.
        """
        if name in self.collectors:
            if exist == 'skip':
                return self.collectors[name]
            if exist == 'error':
                raise RuntimeError(f"Collector '{name}' already registered")
            if exist != 'replace':
                raise ValueError(f"Unknown exist mode: {exist!r}")
            self.remove_collector(name)

        if path is None:
            if self.path is None:
                raise ValueError(
                    f"Collector '{name}': no path given and the registry has no base path"
                )
            path = self.path / name

        cls = resolve_processor(collector)
        connector = resolve_instance(connector)
        obj = cls(name, connector, **resolve_ref_values(params or {}))
        obj.path = Path(path)
        self.collectors[name] = obj
        return obj

    def get_collector(self, name):
        return self.collectors.get(name)

    def remove_collector(self, name):
        return self.collectors.pop(name, None)

    def names(self):
        return list(self.collectors)

    def __contains__(self, name):
        return name in self.collectors

    def __len__(self):
        return len(self.collectors)

    def __iter__(self):
        return iter(self.collectors.values())

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def resolve(self, names):
        """Collector instances for *names* (``None`` → every registered one).

        Raises:
            KeyError: If a name is not registered — a silent miss would look
                exactly like "this collector produced nothing".
        """
        if names is None:
            return list(self.collectors.values())
        missing = [n for n in names if n not in self.collectors]
        if missing:
            raise KeyError(f"Collector(s) not registered: {sorted(missing)}")
        return [self.collectors[n] for n in names]

    def match(self, node_attrs, names=None):
        """Collectors among *names* whose Connector matches *node_attrs*."""
        return [c for c in self.resolve(names) if c.connector.match(node_attrs)]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist each Collector plus an index of name → class ref and path."""
        if self.path is None:
            raise ValueError("Collectors has no path to save into")
        self.path.mkdir(parents=True, exist_ok=True)
        index = {}
        for name, collector in self.collectors.items():
            collector.save()
            index[name] = {'cls': _obj_to_ref(type(collector)), 'path': str(collector.path)}
        with open(self.path / '__collectors.json', 'w') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Restore a registry saved by :meth:`save`."""
        path = Path(path)
        registry = cls(path)
        index_path = path / '__collectors.json'
        if not index_path.exists():
            return registry
        with open(index_path) as f:
            index = json.load(f)
        for name, entry in index.items():
            collector_cls = _ref_to_obj(entry['cls'])
            registry.collectors[name] = collector_cls.load(Path(entry['path']))
        return registry

    def __repr__(self):
        return f"<Collectors {sorted(self.collectors)} path={self.path}>"
