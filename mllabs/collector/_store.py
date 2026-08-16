import json
import sqlite3
from pathlib import Path

from .._serialize import serialize_to_json, deserialize_from_json
from .._pipeline import _ref_hint

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS collectors (
        name      TEXT PRIMARY KEY,
        collector TEXT NOT NULL,
        connector TEXT,
        params    TEXT,
        path      TEXT
    );
"""


def _validate_collector(collector, where):
    """collector must be a ``"module.ClassName"`` string, never a class/instance.

    Same rule as ``_pipeline._validate_processor`` and the same reason:
    nothing downstream resolves a live class back to a string, so one stored
    here would silently fail ``Connector.match``, which compares the stored
    value as a string.
    """
    if isinstance(collector, str):
        return
    hint = _ref_hint(collector)
    suffix = f" Use {hint!r} instead." if hint else ""
    raise TypeError(
        f"{where}: collector must be a \"module.ClassName\" string, got "
        f"{type(collector).__name__}.{suffix}"
    )


def _validate_connector(connector, where):
    """connector must be a string ref or a ``{'__ref__': ..., '__params__': ...}`` spec.

    Same rule as ``_pipeline._validate_adapter`` — a live ``Connector``
    instance is rejected so the definition stays plain data.
    """
    if connector is None or isinstance(connector, str):
        return
    if isinstance(connector, dict) and '__ref__' in connector:
        return
    hint = _ref_hint(connector)
    suffix = f" Use {hint!r} or {{'__ref__': {hint!r}, '__params__': {{...}}}}." if hint else ""
    raise TypeError(
        f"{where}: connector must be a \"module.ClassName\" string or a "
        f"{{'__ref__': ...}} spec, got {type(connector).__name__}.{suffix}"
    )


class CollectorEntity:
    __slots__ = ('name', 'collector', 'connector', 'path')

    def __init__(self, name, collector, connector=None, path=None):
        self.name = name
        self.collector = collector
        self.connector = connector
        self.path = path

    @classmethod
    def of(cls, name, collector, connector, path):
        """*collector*/*connector* are already validated spec values by the
        time this is called (see ``Collectors.set_collector``) — this just
        assembles them, no conversion left to do."""
        return cls(
            name=name,
            collector=collector,
            connector=connector,
            path=str(path) if path is not None else None,
        )

    def __eq__(self, other):
        return (isinstance(other, CollectorEntity)
                and all(getattr(self, s) == getattr(other, s) for s in self.__slots__))

    def __repr__(self):
        return (f"<CollectorEntity {self.name!r} collector={self.collector!r} "
                f"connector={self.connector!r} path={self.path!r}>")


class CollectorStore:
    """Persistence for Collector definitions — storage only.

    Reads and writes ``CollectorEntity``/params, nothing more. Turning those
    back into a live Collector instance is ``Collectors``' job (see
    ``_registry.py``'s ``_build``, using an injected ``Resolver``) — this
    class never imports or instantiates anything itself.

    ``params`` is a ``TEXT`` column (``serialize_to_json``/``deserialize_from_json``),
    same shape as ``connector`` and the same mechanism ``PipelineBuilder``
    uses for node/group params — not the separate pickle file it used to be.
    That was only ever needed because a live object (``ProcessCollector``'s
    ``ext_data``, say) could slip through unvalidated; ``Collectors.set_collector``
    now rejects one with ``_validate_params`` the same way ``PipelineBuilder``
    does, and an ``'@ext:name'`` string (see ``Resolver``) is how that data
    gets in instead.
    """

    def __init__(self, path, name='collectors'):
        self.path = Path(path)
        self.db_path = self.path / f'{name}.db'
        self.path.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def register(self, entity, params=None):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collectors (name, collector, connector, params, path) "
                "VALUES (?, ?, ?, ?, ?)",
                (entity.name, entity.collector, json.dumps(entity.connector),
                 serialize_to_json(params or {}), entity.path),
            )

    def remove(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM collectors WHERE name = ?", (name,))

    def get_params(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT params FROM collectors WHERE name = ?", (name,)
            ).fetchone()
        if row is None or row['params'] is None:
            return None
        return deserialize_from_json(row['params'])

    def get_entity(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM collectors WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def list_entities(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM collectors ORDER BY rowid").fetchall()
        return [self._row_to_entity(r) for r in rows]

    def names(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            return [r[0] for r in conn.execute(
                "SELECT name FROM collectors ORDER BY rowid").fetchall()]

    @staticmethod
    def _row_to_entity(row):
        return CollectorEntity(
            name=row['name'],
            collector=row['collector'],
            connector=json.loads(row['connector']) if row['connector'] else None,
            path=row['path'],
        )

    def __repr__(self):
        return f"<CollectorStore {self.db_path}>"
