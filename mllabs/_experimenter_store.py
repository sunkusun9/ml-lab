"""Project-level registry of Experimenter metadata, keyed by name.

One table for the whole project rather than an ``__exp.db`` per run directory:
an Experimenter's name is its identity now, so listing or comparing runs should
be a query, not a directory scan.

The columns are typed rather than the key/value JSON the per-run store used —
that shape existed because the value set was open-ended, and it no longer is.

Two things deliberately stay outside this table:

- the splitters, which are not ref-serializable and remain in each run's
  ``__splitters.pkl``
- the Pipeline, which lives once under the Project as a version. Only the
  ``(pipeline_name, pipeline_version)`` pointer is stored here.
"""
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS experimenters (
        name             TEXT PRIMARY KEY,
        data_key         TEXT,
        title            TEXT,
        status           TEXT,
        pipeline_name    TEXT,
        pipeline_version INTEGER
    );
"""

_COLUMNS = ('name', 'data_key', 'title', 'status',
            'pipeline_name', 'pipeline_version')


class ExperimenterStore:
    """SQLite-backed registry of every Experimenter in one project."""

    def __init__(self, path, name='experimenters'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def save(self, meta):
        """Insert or replace one Experimenter's row from a ``{column: value}`` dict."""
        unknown = set(meta) - set(_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown experimenter meta column(s): {sorted(unknown)}")
        if 'name' not in meta:
            raise ValueError("experimenter meta requires 'name'")
        columns = [c for c in _COLUMNS if c in meta]
        placeholders = ', '.join('?' for _ in columns)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO experimenters ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [meta[c] for c in columns],
            )

    def set(self, name, column, value):
        """Update a single column without touching the rest of the row."""
        if column not in _COLUMNS or column == 'name':
            raise ValueError(f"Cannot set experimenter column {column!r}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"UPDATE experimenters SET {column} = ? WHERE name = ?", (value, name)
            )

    def fetch(self, name):
        """One Experimenter's row as a dict, or ``None``."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experimenters WHERE name = ?", (name,)
            ).fetchone()
        return dict(row) if row else None

    def exists(self, name):
        return self.fetch(name) is not None

    def list_names(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            return [r[0] for r in conn.execute(
                "SELECT name FROM experimenters ORDER BY name").fetchall()]

    def list_all(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                "SELECT * FROM experimenters ORDER BY name").fetchall()]

    def remove(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM experimenters WHERE name = ?", (name,))

    def __repr__(self):
        return f"<ExperimenterStore {self.db_path}>"
