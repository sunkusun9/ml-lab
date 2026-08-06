from pathlib import Path

from ._collect_hist import CollectHist
from ._store import CollectorEntity, CollectorStore, build_collector


class Collectors:
    """Registry that owns Collector instances and their storage.

    One registry belongs to one run — an Experimenter builds its own over
    ``{exp path}/collectors`` and hands it out as ``Experimenter.collectors``.
    Everything a Collector writes is keyed by node name and nothing more
    (``MetricCollector``'s PK is ``(node, idx, inner_idx, split)``; the
    file-based ones use ``{path}/{node}...``), so the path is what keeps two
    runs apart. Sharing one registry between runs would have them overwrite
    each other's results for every Trial whose name they have in common —
    silently, and precisely when the results were worth comparing.

    Registration persists immediately when the registry has a path: a
    :class:`CollectorEntity` row goes into ``{path}/collectors.db`` and the
    constructor params into ``{path}/__params/{name}.pkl``. Constructing a
    registry over that path rebuilds every Collector registered before from
    those two halves — the instance itself is never stored.

    ``hist`` is the :class:`~mllabs.CollectHist` over the same path — one row
    per (collector, node, fold) describing what that collect call did. It has
    no ``experimenter`` column because the registry holding it already is one
    run's.

    Args:
        path (str | Path, optional): Base directory. A Collector registered
            without an explicit ``path`` gets ``{path}/{name}``. Without a path
            the registry is memory-only — nothing is persisted, history included.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._store = CollectorStore(self.path) if self.path is not None else None
        self.collectors = {c.name: c for c in self._store.load_all()} if self._store else {}
        self.hist = CollectHist(self.path) if self.path is not None else None

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

        entity = CollectorEntity.of(name, collector, connector, path)
        obj = build_collector(entity, params)
        self.collectors[name] = obj
        if self._store is not None:
            self._store.register(entity, params)
        return obj

    def get_collector(self, name):
        return self.collectors.get(name)

    def remove_collector(self, name):
        if self._store is not None:
            self._store.remove(name)
        return self.collectors.pop(name, None)

    def names(self):
        return list(self.collectors)

    def remove_results(self, node_name):
        """Drop everything collected for *node_name* — data and history both.

        The registry is the only thing that sees both halves: ``hist`` records
        what each collect call did, while the result itself is inside whichever
        Collector produced it. Removing one without the other leaves a history
        row pointing at data that is gone, or data no history accounts for.

        Definitions are untouched — this removes what a node produced, not the
        Collectors that were watching for it.
        """
        if self.hist is not None:
            self.hist.remove_hist(node_name=node_name)
        for collector in self.collectors.values():
            collector.reset_nodes([node_name])

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

    def __repr__(self):
        return f"<Collectors {sorted(self.collectors)} path={self.path}>"
