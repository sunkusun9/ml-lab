from pathlib import Path

from ._collect_hist import CollectHist
from ._store import CollectorEntity, CollectorStore, _validate_collector, _validate_connector
from .._resolver import Resolver
from .._pipeline import _validate_params


class Collectors:
    """Registry that owns Collector instances and their storage.

    One registry belongs to one Experimenter — it builds its own over
    ``{exp path}/collectors`` and hands it out as ``Experimenter.collectors``.
    Everything a Collector writes is keyed by node name and nothing more
    (``MetricCollector``'s PK is ``(node, idx, inner_idx, split)``; the
    file-based ones use ``{path}/{node}...``), so the path is what keeps two
    Experimenters apart. Sharing one registry between them would have them overwrite
    each other's results for every Trial whose name they have in common —
    silently, and precisely when the results were worth comparing.

    Registration persists immediately when the registry has a path: a
    :class:`CollectorEntity` row goes into ``{path}/collectors.db`` and the
    constructor params into ``{path}/__params/{name}.pkl``. Constructing a
    registry over that path rebuilds every Collector registered before from
    those two halves — the instance itself is never stored.

    ``hist`` is the :class:`~mllabs.CollectHist` over the same path — one row
    per (collector, node, fold) describing what that collect call did. It has
    no ``experimenter`` column because the registry holding it already belongs
    to one.

    Turning a stored ``CollectorEntity``/params pair into a live instance is
    this class's job, not the store's — ``_build`` is the one place that
    happens, using an injected :class:`~mllabs.Resolver`. ``CollectorStore``
    only ever hands back entities and params, never an instance.

    Args:
        path (str | Path, optional): Base directory. A Collector registered
            without an explicit ``path`` gets ``{path}/{name}``. Without a path
            the registry is memory-only — nothing is persisted, history included.
        resolver (Resolver, optional): Turns a stored spec into live objects.
            Defaults to a bare ``Resolver()`` (no ``ExtDataProvider``, so an
            ``'@ext:name'`` param would raise) — pass one with ``ext_data``
            set to resolve those.
    """

    def __init__(self, path=None, resolver=None):
        self.path = Path(path) if path is not None else None
        self._store = CollectorStore(self.path) if self.path is not None else None
        self.resolver = resolver if resolver is not None else Resolver()
        self.collectors = (
            {e.name: self._build(e, self._store.get_params(e.name))
             for e in self._store.list_entities()}
            if self._store is not None else {}
        )
        self.hist = CollectHist(self.path) if self.path is not None else None

    def _build(self, entity, params):
        """A live Collector from *entity*/*params*, via :attr:`resolver`."""
        cls = self.resolver.processor(entity.collector)
        obj = cls(entity.name, self.resolver.instance(entity.connector),
                  **self.resolver.params(params or {}))
        obj.path = Path(entity.path) if entity.path is not None else None
        return obj

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def set_collector(self, name, collector, connector, path=None, params=None, exist='skip'):
        """Build and register a Collector from its parts.

        Args:
            name (str): Collector name — what an Experiment refers to it by.
            collector (str): ``"module.ClassName"`` reference — never a class
                or instance, same rule ``PipelineBuilder`` enforces for a
                node's processor.
            connector: ``None`` / ``"module.ClassName"`` string / ``{"__ref__":
                ..., "__params__": {...}}`` spec — never a live
                :class:`~mllabs.Connector` instance, same rule for a node's
                adapter.
            path (str | Path, optional): Storage directory. Defaults to
                ``{registry path}/{name}``; required if the registry has no path.
            params (dict): Remaining constructor parameters (everything after
                ``name``/``connector``). ``{"__callable__": "mod.fn"}`` resolves
                to the referenced object (not called); ``{"__ref__": "mod.Cls",
                "__params__": {...}}`` is instantiated; ``"@ext:name"`` resolves
                against the registry's ``Resolver.ext_data``. Live objects are
                rejected — same rule ``PipelineBuilder.set_grp``/``set_node``
                enforce for a node's params, checked the same way.
            exist (str): ``'skip'`` (default) returns the existing collector;
                ``'error'`` raises; ``'replace'`` drops and rebuilds it.

        Returns:
            Collector: The registered collector.

        Raises:
            TypeError: If *collector*/*connector*/*params* hold a live object
                rather than plain data or a ref spec.
        """
        where = f"set_collector({name!r})"
        _validate_collector(collector, where)
        _validate_connector(connector, where)
        _validate_params(params, where)
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
        obj = self._build(entity, params)
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
