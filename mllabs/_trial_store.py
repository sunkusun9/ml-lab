"""SQLite persistence for Trial definitions and their run history.

Two tables:

``trials``
    One row per Trial *name* — the same identity its on-disk artifacts use.
    Redefining a name overwrites its row exactly as it overwrites the
    artifact.

``experiment_hist``
    One row per (trial name, experimenter name, outer fold, inner fold) — what
    ran, against which Pipeline, and whether it succeeded.

Both tables key on names, matching what is on disk: a trial's artifacts live
at ``{exp}/__folds/{outer}/{inner}/{trial_name}/`` and an Experimenter's at
``{project}/exp/{name}``. Keeping the keys aligned with the layout means a
hist row is readable on its own, and that redefining a name overwrites its
row exactly as it overwrites the artifact — true in both tables.

Neither table stores a content hash. Whether a stored definition still
matches a given Trial is a plain value comparison (``has``), and whether an
artifact needs rebuilding is decided the same way, against the artifact's own
``info['definition']`` on disk (see ``Experimenter._make_jobs``) — a hash
would only restate what these already compare directly. ``experiment_hist``
is a run log, not the source of truth for a definition: it does not attempt
to recover what a name's definition used to be before it was redefined.
"""
import json
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS trials (
        name        TEXT PRIMARY KEY,
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
        pipeline_version INTEGER,
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
        """Store *trial*'s definition under its name.

        ``name`` is the row's identity, so redefining a name overwrites its
        row exactly as it overwrites the artifact on disk.

        Args:
            trial (Trial): Trial to register.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trials "
                "(name, label, processor, method, adapter, params, edges, tag) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trial.name, trial.label, trial.processor,
                 trial.method, json.dumps(trial.adapter), json.dumps(trial.params),
                 json.dumps(trial.edges), json.dumps(trial.tag)),
            )

    def register_all(self, trials):
        """Register every Trial in *trials*."""
        for t in trials:
            self.register(t)

    def has(self, trial):
        """Whether *trial*'s exact definition is the one stored under its name."""
        row = self.get_by_name(trial.name)
        return (row is not None
                and row['processor'] == trial.processor
                and row['method'] == trial.method
                and row['adapter'] == trial.adapter
                and row['params'] == trial.params
                and row['edges'] == trial.edges)

    def get_by_name(self, name):
        """Stored definition for *name*, or ``None``."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trials WHERE name = ?", (name,)
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
               pipeline_version=None, status=None):
        """Upsert one fold's outcome.

        Args:
            trial_name (str): Trial name — also its artifact directory.
            experimenter (str): The Experimenter's name.
            outer_idx (int), inner_idx (int): Fold coordinates.
            pipeline_version (int, optional): The Pipeline version this ran
                against (the Experimenter's ``pipeline_version``).
            status (str, optional): ``'built'``, ``'finalized'``, ``'error'``, ...
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiment_hist "
                "(trial_name, experimenter, outer_idx, inner_idx, "
                "pipeline_version, status) VALUES (?, ?, ?, ?, ?, ?)",
                (trial_name, experimenter, outer_idx, inner_idx,
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
