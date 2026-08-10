"""One Trainer's own persisted state, living in its own directory.

The Experimenter-side counterpart is
:class:`~mllabs._experimenter_store.ExperimenterStore`, and this holds the same
kinds of thing for the same reason — a Trainer builds one from its ``path``, so
its directory is all that reopening needs.

Two differences from the Experimenter's, both from Trainer's splitting being
simpler and its result being artifacts rather than a comparison:

- the split blob carries the resolved ``split_indices`` alongside the splitter,
  because a Trainer may have no splitter at all (a single full-data fold), and
  reopening must land on exactly the folds that were trained
- there is no ``data_key``/``title``: nothing here is compared against another
  Trainer, so there is nothing to label or guard a mismatch against

What a Trainer trains is *not* here — that is
:class:`~mllabs.PredictorStore`'s, kept separate so the definitions stay
readable as their own thing.
"""
import pickle as pkl
import sqlite3
from pathlib import Path

from ._common import load_pipeline, save_pipeline

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS trainer (
        name             TEXT PRIMARY KEY,
        pipeline_name    TEXT,
        pipeline_version INTEGER,
        splits           BLOB
    );
"""

_COLUMNS = ('name', 'pipeline_name', 'pipeline_version')

#: Basename of the Trainer's own db, without the ``.db`` suffix.
DB_NAME = '__trainer'


class TrainerStore:
    """SQLite-backed state for the single Trainer rooted at *path*."""

    def __init__(self, path, name=DB_NAME):
        self.path = Path(path)
        self.db_path = self.path / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    @staticmethod
    def stored_at(path, name=DB_NAME):
        """Whether *path* holds a saved Trainer.

        Static for the same reason as
        :meth:`ExperimenterStore.stored_at` — constructing the store would
        create the very thing being asked about.
        """
        return (Path(path) / f'{name}.db').exists()

    def save(self, meta):
        """Insert or replace the meta row from a ``{column: value}`` dict.

        Leaves ``splits`` alone — written by :meth:`save_splits`, and an
        ``INSERT OR REPLACE`` naming only the meta columns would blank it.
        """
        unknown = set(meta) - set(_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown trainer meta column(s): {sorted(unknown)}")
        if 'name' not in meta:
            raise ValueError("trainer meta requires 'name'")
        columns = [c for c in _COLUMNS if c in meta and c != 'name']
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO trainer (name) VALUES (?)", (meta['name'],))
            if columns:
                conn.execute(
                    f"UPDATE trainer SET {', '.join(f'{c} = ?' for c in columns)} "
                    f"WHERE name = ?",
                    [meta[c] for c in columns] + [meta['name']],
                )

    def set(self, name, column, value):
        """Update a single column without touching the rest of the row."""
        if column not in _COLUMNS or column == 'name':
            raise ValueError(f"Cannot set trainer column {column!r}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(f"UPDATE trainer SET {column} = ? WHERE name = ?", (value, name))

    def fetch(self, name=None):
        """The meta row as a dict (without ``splits``), or ``None``.

        *name* is optional: there is only ever one row, so omitting it reads
        whichever Trainer this directory holds.
        """
        query = "SELECT * FROM trainer"
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

    def save_splits(self, name, splits):
        """Store ``{'splitter', 'splitter_params', 'split_indices'}`` as a blob."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO trainer (name) VALUES (?)", (name,))
            conn.execute("UPDATE trainer SET splits = ? WHERE name = ?",
                         (pkl.dumps(splits), name))

    def load_splits(self, name=None):
        """The stored splits dict, or ``None`` if this Trainer has none."""
        query = "SELECT splits FROM trainer"
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
        """Write the Pipeline this Trainer adopted, beside the db."""
        save_pipeline(self.path, pipeline)

    def load_pipeline(self):
        """The Pipeline saved beside the db, or ``None`` if there is none."""
        return load_pipeline(self.path)

    def remove(self, name=None):
        query = "DELETE FROM trainer"
        params = ()
        if name is not None:
            query += " WHERE name = ?"
            params = (name,)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(query, params)

    def __repr__(self):
        return f"<TrainerStore {self.db_path}>"
