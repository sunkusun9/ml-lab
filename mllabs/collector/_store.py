import json
import pickle
import sqlite3
from pathlib import Path

from .._connector import Connector
from .._serialize import (resolve_processor, resolve_instance, resolve_ref_values,
                          _obj_to_ref)

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS collectors (
        name      TEXT PRIMARY KEY,
        collector TEXT NOT NULL,
        connector TEXT,
        path      TEXT
    );
"""


class CollectorEntity:
    __slots__ = ('name', 'collector', 'connector', 'path')

    def __init__(self, name, collector, connector=None, path=None):
        self.name = name
        self.collector = collector
        self.connector = connector
        self.path = path

    @classmethod
    def of(cls, name, collector, connector, path):
        return cls(
            name=name,
            collector=collector if isinstance(collector, str) else _obj_to_ref(collector),
            connector=connector_spec(connector),
            path=str(path) if path is not None else None,
        )

    def __eq__(self, other):
        return (isinstance(other, CollectorEntity)
                and all(getattr(self, s) == getattr(other, s) for s in self.__slots__))

    def __repr__(self):
        return (f"<CollectorEntity {self.name!r} collector={self.collector!r} "
                f"connector={self.connector!r} path={self.path!r}>")


def connector_spec(connector):
    if connector is None or isinstance(connector, dict):
        return connector
    if isinstance(connector, str):
        return {'__ref__': connector}
    if isinstance(connector, Connector):
        return {'__ref__': _obj_to_ref(type(connector)), '__params__': dict(vars(connector))}
    raise TypeError(f"Cannot describe connector of type {type(connector).__name__}")


def build_collector(entity, params=None):
    cls = resolve_processor(entity.collector)
    obj = cls(entity.name, resolve_instance(entity.connector),
              **resolve_ref_values(params or {}))
    obj.path = Path(entity.path) if entity.path is not None else None
    return obj


class CollectorStore:
    def __init__(self, path, name='collectors'):
        self.path = Path(path)
        self.db_path = self.path / f'{name}.db'
        self.params_path = self.path / '__params'
        self.path.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def _params_file(self, name):
        return self.params_path / f'{name}.pkl'

    def register(self, entity, params=None):
        self.params_path.mkdir(parents=True, exist_ok=True)
        with open(self._params_file(entity.name), 'wb') as f:
            pickle.dump(params or {}, f)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collectors (name, collector, connector, path) "
                "VALUES (?, ?, ?, ?)",
                (entity.name, entity.collector, json.dumps(entity.connector), entity.path),
            )

    def remove(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM collectors WHERE name = ?", (name,))
        f = self._params_file(name)
        if f.exists():
            f.unlink()

    def get_params(self, name):
        f = self._params_file(name)
        if not f.exists():
            return None
        with open(f, 'rb') as fp:
            return pickle.load(fp)

    def build(self, name):
        entity = self.get_entity(name)
        return build_collector(entity, self.get_params(name)) if entity else None

    def load_all(self):
        return [build_collector(e, self.get_params(e.name)) for e in self.list_entities()]

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
