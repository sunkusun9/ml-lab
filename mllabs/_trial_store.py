"""SQLite persistence for Trial definitions and their run history.

Two tables:

``trials``
    One row per distinct Trial *definition*, keyed by ``content_key`` — a hash
    of the definition itself. A natural key, not a minted one.

``experiment_hist``
    One row per (trial name, experimenter name, outer fold, inner fold) — what
    ran, against which Pipeline, and whether it succeeded.

Both parts of the key are names, matching what is on disk: a trial's artifacts
live at ``{exp}/__folds/{outer}/{inner}/{trial_name}/`` and an Experimenter's
at ``{project}/exp/{name}``. Keeping the keys aligned with the layout means a
hist row is readable on its own, and that redefining a name overwrites the row
exactly as it overwrites the artifact. ``content_key`` rides along as a plain
column, so which definition a given run used stays recoverable.

``content_key`` covers the definition only. Whether an artifact is still valid
also depends on the preprocessing it read, but that is judged per fold against
the fold itself — it is not part of a Trial's identity here.
"""
import json
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS trials (
        content_key TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        label       TEXT,
        processor   TEXT NOT NULL,
        method      TEXT,
        adapter     TEXT,
        params      TEXT,
        edges       TEXT,
        tag         TEXT
    );
    CREATE TABLE IF NOT EXISTS experiment_hist (
        trial_name       TEXT NOT NULL,
        experimenter  TEXT NOT NULL,
        outer_idx        INTEGER NOT NULL,
        inner_idx        INTEGER NOT NULL,
        content_key      TEXT,
        pipeline_version TEXT,
        status           TEXT,
        PRIMARY KEY (trial_name, experimenter, outer_idx, inner_idx)
    );
    CREATE INDEX IF NOT EXISTS idx_hist_experimenter
        ON experiment_hist (experimenter);
"""


class TrialStore:
    """Registry of Trial definitions plus a per-fold run history."""

    def __init__(self, path, name='trials'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # trials
    # ------------------------------------------------------------------

    def register(self, trial):
        """Store *trial*'s definition if new; return its ``content_key``.

        Args:
            trial (Trial): Trial to register.

        Returns:
            str: Content key identifying this definition.
        """
        content_key = trial.content_key()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trials "
                "(content_key, name, label, processor, method, adapter, params, edges, tag) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (content_key, trial.name, trial.label, trial.processor,
                 trial.method, json.dumps(trial.adapter), json.dumps(trial.params),
                 json.dumps(trial.edges), json.dumps(trial.tag)),
            )
        return content_key

    def register_all(self, trials):
        """``{trial.name: content_key}`` for *trials*."""
        return {t.name: self.register(t) for t in trials}

    def has(self, trial):
        """Whether *trial*'s definition is already stored."""
        return self.get_definition(trial.content_key()) is not None

    def get_definition(self, content_key):
        """Stored definition as a dict, or ``None``."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trials WHERE content_key = ?", (content_key,)
            ).fetchone()
        return self._row_to_trial(row) if row else None

    def list_trials(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trials ORDER BY rowid").fetchall()
        return [self._row_to_trial(r) for r in rows]

    @staticmethod
    def _row_to_trial(row):
        return {
            'content_key': row['content_key'],
            'name': row['name'],
            'label': row['label'],
            'processor': row['processor'],
            'method': row['method'],
            'adapter': json.loads(row['adapter']) if row['adapter'] else None,
            'params': json.loads(row['params']) if row['params'] else {},
            'edges': json.loads(row['edges']) if row['edges'] else {},
            'tag': json.loads(row['tag']) if row['tag'] else [],
        }

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    def record(self, trial_name, experimenter, outer_idx, inner_idx,
               content_key=None, pipeline_version=None, status=None):
        """Upsert one fold's outcome.

        Args:
            trial_name (str): Trial name — also its artifact directory.
            experimenter (str): The Experimenter's name.
            outer_idx (int), inner_idx (int): Fold coordinates.
            content_key (str, optional): Which definition this name held at the
                time, so a later redefinition stays distinguishable.
            pipeline_version (str, optional): ``Pipeline.content_key()`` — the
                Pipeline this ran against. A Project maps it to a readable
                version number; on its own it is still a stable identity.
            status (str, optional): ``'built'``, ``'finalized'``, ``'error'``, ...
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiment_hist "
                "(trial_name, experimenter, outer_idx, inner_idx, content_key, "
                "pipeline_version, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (trial_name, experimenter, outer_idx, inner_idx, content_key,
                 pipeline_version, status),
            )

    def get_hist(self, trial_name=None, experimenter=None, pipeline_version=None):
        """History rows matching whichever filters are given."""
        where, params = [], []
        for column, value in (('trial_name', trial_name),
                              ('experimenter', experimenter),
                              ('pipeline_version', pipeline_version)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM experiment_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY trial_name, experimenter, outer_idx, inner_idx"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_status(self, trial_name, experimenter):
        """``{(outer_idx, inner_idx): status}`` for one trial in one experimenter."""
        return {
            (r['outer_idx'], r['inner_idx']): r['status']
            for r in self.get_hist(trial_name=trial_name, experimenter=experimenter)
        }

    def remove_hist(self, trial_name=None, experimenter=None):
        where, params = [], []
        for column, value in (('trial_name', trial_name), ('experimenter', experimenter)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "DELETE FROM experiment_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(sql, params)

    def __repr__(self):
        return f"<TrialStore {self.db_path}>"
