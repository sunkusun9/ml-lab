"""Project-level index of which Experimenters and Trainers exist.

Names only. Everything *about* one — its splitters, its data key, which
Pipeline it adopted — belongs to it and is stored in its own directory,
so this table stays a list and never becomes a second source of truth to keep
in sync. Answering "what is in this project?" without opening every
directory is the whole job.

Registration is :class:`~mllabs.Project`'s doing: an Experimenter or Trainer
constructed directly, without going through a Project, is simply not in
anyone's index.
"""
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS experimenters (name TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS trainers (name TEXT PRIMARY KEY);
"""


class ProjectStore:
    """SQLite-backed list of the Experimenters and Trainers in one project."""

    def __init__(self, path, name='project'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def _register(self, table, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(f"INSERT OR IGNORE INTO {table} (name) VALUES (?)", (name,))

    def _list(self, table):
        with sqlite3.connect(str(self.db_path)) as conn:
            return [r[0] for r in conn.execute(
                f"SELECT name FROM {table} ORDER BY name").fetchall()]

    def _remove(self, table, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(f"DELETE FROM {table} WHERE name = ?", (name,))

    def register_experimenter(self, name):
        self._register('experimenters', name)

    def register_trainer(self, name):
        self._register('trainers', name)

    def list_experimenters(self):
        return self._list('experimenters')

    def list_trainers(self):
        return self._list('trainers')

    def remove_experimenter(self, name):
        self._remove('experimenters', name)

    def remove_trainer(self, name):
        self._remove('trainers', name)

    def __repr__(self):
        return f"<ProjectStore {self.db_path}>"
