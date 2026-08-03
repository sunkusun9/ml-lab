"""One Experimenter's own persisted state, living in its own directory.

An Experimenter owns this — it builds one from its ``path`` rather than being
handed one, so everything needed to reopen a run is inside that directory and
nothing has to be resolved from a Project. The project-wide question of *which*
runs exist is a different one, answered by
:class:`~mllabs._project_store.ProjectStore`.

Three kinds of state live here:

- the meta row (data_key, title, and the ``(pipeline_name, pipeline_version)``
  pointer, which is provenance for the Pipeline copy — see below)
- the splitters, as a pickle blob rather than columns: they are arbitrary
  sklearn objects and not ref-serializable
- the Pipeline, written beside the db as ``pipeline.pkl`` rather than into it,
  via :meth:`save_pipeline`
"""
import pickle as pkl
import sqlite3
from pathlib import Path

from ._run_common import load_pipeline, save_pipeline

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS experimenter (
        name             TEXT PRIMARY KEY,
        data_key         TEXT,
        title            TEXT,
        pipeline_name    TEXT,
        pipeline_version INTEGER,
        splitters        BLOB
    );
"""

_COLUMNS = ('name', 'data_key', 'title', 'pipeline_name', 'pipeline_version')

#: Basename of the per-run db, without the ``.db`` suffix.
DB_NAME = '__exp'


class ExperimenterStore:
    """SQLite-backed state for the single Experimenter rooted at *path*."""

    def __init__(self, path, name=DB_NAME):
        self.path = Path(path)
        self.db_path = self.path / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def save(self, meta):
        """Insert or replace the meta row from a ``{column: value}`` dict.

        Leaves ``splitters`` alone — it is written by :meth:`save_splitters`,
        and an ``INSERT OR REPLACE`` naming only the meta columns would blank
        it out.
        """
        unknown = set(meta) - set(_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown experimenter meta column(s): {sorted(unknown)}")
        if 'name' not in meta:
            raise ValueError("experimenter meta requires 'name'")
        columns = [c for c in _COLUMNS if c in meta]
        assignments = ', '.join(f"{c} = ?" for c in columns if c != 'name')
        values = [meta[c] for c in columns if c != 'name']
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO experimenter (name) VALUES (?)", (meta['name'],)
            )
            if assignments:
                conn.execute(
                    f"UPDATE experimenter SET {assignments} WHERE name = ?",
                    values + [meta['name']],
                )

    def set(self, name, column, value):
        """Update a single column without touching the rest of the row."""
        if column not in _COLUMNS or column == 'name':
            raise ValueError(f"Cannot set experimenter column {column!r}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"UPDATE experimenter SET {column} = ? WHERE name = ?", (value, name)
            )

    def fetch(self, name=None):
        """The meta row as a dict (without ``splitters``), or ``None``.

        *name* is optional: there is only ever one row, so omitting it reads
        whichever run this directory holds.
        """
        query = "SELECT * FROM experimenter"
        params = ()
        if name is not None:
            query += " WHERE name = ?"
            params = (name,)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in _COLUMNS}

    def exists(self, name=None):
        return self.fetch(name) is not None

    def save_splitters(self, name, splitters):
        """Store ``{'sp', 'sp_v', 'splitter_params'}`` as a pickle blob."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO experimenter (name) VALUES (?)", (name,))
            conn.execute(
                "UPDATE experimenter SET splitters = ? WHERE name = ?",
                (pkl.dumps(splitters), name),
            )

    def load_splitters(self, name=None):
        """The stored splitters dict, or ``None`` if this run has none."""
        query = "SELECT splitters FROM experimenter"
        params = ()
        if name is not None:
            query += " WHERE name = ?"
            params = (name,)
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(query, params).fetchone()
        if row is None or row[0] is None:
            return None
        return pkl.loads(row[0])

    def save_pipeline(self, pipeline):
        """Write the Pipeline this Experimenter adopted, beside the db."""
        save_pipeline(self.path, pipeline)

    def load_pipeline(self):
        """The Pipeline saved beside the db, or ``None`` if there is none."""
        return load_pipeline(self.path)

    def remove(self, name=None):
        query = "DELETE FROM experimenter"
        params = ()
        if name is not None:
            query += " WHERE name = ?"
            params = (name,)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(query, params)

    def __repr__(self):
        return f"<ExperimenterStore {self.db_path}>"
