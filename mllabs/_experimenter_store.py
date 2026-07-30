"""SQLite persistence for an Experimenter's meta and collector registry.

All table/schema knowledge lives here. Experimenter deals only with plain
Python values (meta) and Collector instances; it does not know column names
or table structure.

- ``meta``: key → JSON-serialized value. Holds only simple, ref-serializable
  values (data_key, title, cache_maxsize, exp_id, tags, status). Arbitrary
  runtime objects (splitters, splitter_params) stay in ``__splitters.pkl``.
- ``collectors``: name → ``module.QualName`` class reference, so restoration
  needs no hardcoded type map.
"""
import sqlite3
from pathlib import Path

from ._serialize import (
    serialize_to_json, deserialize_from_json, _obj_to_ref, _ref_to_obj,
)

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    CREATE TABLE IF NOT EXISTS collectors (
        name TEXT PRIMARY KEY,
        cls TEXT NOT NULL
    );
"""


class ExperimenterStore:
    """SQLite-backed persistence for a single Experimenter's ``__exp.db``."""

    def __init__(self, path, name='__exp'):
        self.db_path = Path(path) / f'{name}.db'

    def exists(self):
        return self.db_path.exists()

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def save_meta(self, meta):
        """Write ``{key: python_value}`` into the meta table (values serialized)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [(k, serialize_to_json(v)) for k, v in meta.items()],
            )

    def set_meta(self, key, value):
        """Write a single meta key (value serialized) without touching others."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, serialize_to_json(value)),
            )

    def fetch_meta(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return {
                row['key']: deserialize_from_json(row['value'])
                for row in conn.execute("SELECT key, value FROM meta").fetchall()
            }

    def write_collector(self, collector):
        """Register a collector by name + ``module.QualName`` class reference."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collectors (name, cls) VALUES (?, ?)",
                (collector.name, _obj_to_ref(type(collector))),
            )

    def remove_collector(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM collectors WHERE name = ?", (name,))

    def fetch_collectors(self):
        """Return ``{name: collector_class}`` — classes resolved from refs."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return {
                row['name']: _ref_to_obj(row['cls'])
                for row in conn.execute("SELECT name, cls FROM collectors").fetchall()
            }
