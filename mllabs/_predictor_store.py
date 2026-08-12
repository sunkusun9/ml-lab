"""SQLite persistence for a Trainer's :class:`~mllabs.Predictor` definitions.

One table, ``predictors``, keyed by name — the same identity the artifacts
use, so redefining a name overwrites its row exactly as it overwrites the
artifact directory.

Definitions, plus one lifecycle field. A Predictor's per-split outcome
(fit_time, error, which fold, ...) is execution history and lives in the
Trainer's predictor ``NodeStore`` alongside the artifacts it describes,
recorded by ``NodeInfoTracker`` the same way Pipeline nodes are. That split
is why this store is much smaller than :class:`~mllabs.TrialStore`, which
carries both halves: a TrialStore is project-wide and has to answer "which
Experimenter ran this, and how did it go", whereas each Trainer's history is
already scoped by being its own store.

``status`` is the exception, and it earns the place. Three of its four values
are recoverable from disk, but :data:`RETIRED` is not: retiring drops the
artifacts and leaves the history rows standing, so what remains on disk is
indistinguishable from a Predictor that was reset and is waiting to be
trained. Retiring is an event that happened, not a fact still legible in what
survives it, and the difference decides whether ``train()`` builds a Job.
Keeping the other three here too costs a column and means one place answers
"where does this Predictor stand" instead of a cross-reference only someone
holding the convention can perform.

Scoped per Trainer rather than per project for the same reason: what this
answers is "what is *this* Trainer training", which is that Trainer's own
state, not a project-wide fact. Restoring it is what lets
:meth:`Trainer.load` come back with its selection intact.
"""
import json
import sqlite3
from pathlib import Path

from ._predictor import Predictor

#: Registered, with at least one split still to train.
INIT = 'init'
#: Every split built — this is what :meth:`Trainer.to_inferencer` can ship.
TRAINED = 'trained'
#: Ended by a Pipeline version switch its inputs did not survive. Terminal:
#: no Job is ever built for it again. Wanting the same specification back
#: means registering it afresh, under the version that can serve it.
RETIRED = 'retired'
#: At least one split failed. One bad fold is a bad Predictor.
ERROR = 'error'

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS predictors (
        name             TEXT PRIMARY KEY,
        desc             TEXT,
        processor        TEXT NOT NULL,
        method           TEXT,
        adapter          TEXT,
        params           TEXT,
        edges            TEXT,
        tag              TEXT,
        src_trial        TEXT,
        src_experimenter TEXT,
        pipeline_version INTEGER,
        status           TEXT NOT NULL DEFAULT 'init'
    );
"""


class PredictorStore:
    """Registry of one Trainer's Predictor definitions.

    Args:
        path: Directory holding the store — the Trainer's predictor
            directory, shared with the ``NodeStore`` keeping the artifacts
            (different file names, so they coexist).
    """

    def __init__(self, path, name='__predictors'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def register(self, predictor):
        """Store *predictor*'s definition under its name.

        An upsert that writes every definition field and leaves ``status``
        alone, rather than ``INSERT OR REPLACE``, which would silently return
        a Predictor to :data:`INIT` every time its row is rewritten —
        including on ``Trainer.train``'s re-registration of what it was just
        handed, and on the restamp that follows adopting a version. Status
        changes only through :meth:`set_status`, where something decided it.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO predictors "
                "(name, desc, processor, method, adapter, params, edges, tag, "
                "src_trial, src_experimenter, pipeline_version, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "desc=excluded.desc, processor=excluded.processor, "
                "method=excluded.method, adapter=excluded.adapter, "
                "params=excluded.params, edges=excluded.edges, "
                "tag=excluded.tag, src_trial=excluded.src_trial, "
                "src_experimenter=excluded.src_experimenter, "
                "pipeline_version=excluded.pipeline_version",
                (predictor.name, predictor.desc, predictor.processor,
                 predictor.method, json.dumps(predictor.adapter),
                 json.dumps(predictor.params), json.dumps(predictor.edges),
                 json.dumps(predictor.tag), predictor.src_trial,
                 predictor.src_experimenter, predictor.pipeline_version,
                 INIT),
            )

    def register_all(self, predictors):
        """Register every Predictor in *predictors*."""
        for p in predictors:
            self.register(p)

    def replace_all(self, predictors):
        """Make *predictors* the entire stored selection.

        Names no longer selected are dropped, so what comes back from
        :meth:`list_predictors` is always the current selection rather than
        everything a Trainer ever trained. Insertion order is preserved.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM predictors")
        self.register_all(predictors)

    def remove(self, name):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM predictors WHERE name = ?", (name,))

    def has(self, predictor):
        """Whether *predictor*'s exact definition is the one stored under its name.

        ``pipeline_version`` counts, for the same reason it does in
        ``TrialStore.has``: it decides which Trainer state the definition may
        be trained under, so the same fields against another version are
        another definition.
        """
        stored = self.get_by_name(predictor.name)
        return (stored is not None
                and stored.processor == predictor.processor
                and stored.method == predictor.method
                and stored.adapter == predictor.adapter
                and stored.params == predictor.params
                and stored.edges == predictor.edges
                and stored.pipeline_version == predictor.pipeline_version)

    def set_status(self, name, status):
        """Move *name* to *status*. No-op if the name is not stored."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("UPDATE predictors SET status = ? WHERE name = ?",
                         (status, name))

    def get_status(self, name):
        """Stored status for *name*, or ``None`` if it is not registered."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM predictors WHERE name = ?", (name,)
            ).fetchone()
        return row[0] if row else None

    def list_status(self):
        """``{name: status}`` for every stored Predictor.

        One query, so a caller deciding across the whole registry — which to
        build Jobs for, which to restamp — does not ask per name.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT name, status FROM predictors ORDER BY rowid").fetchall()
        return {name: status for name, status in rows}

    def list_names(self, status=None):
        """Stored names, optionally only those in *status*."""
        return [n for n, s in self.list_status().items()
                if status is None or s == status]

    def get_by_name(self, name):
        """Stored :class:`~mllabs.Predictor` for *name*, or ``None``."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM predictors WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_predictor(row) if row else None

    def list_predictors(self):
        """Every stored Predictor, in the order they were registered.

        Returns :class:`~mllabs.Predictor` objects rather than dicts — this
        is what :meth:`Trainer.load` hands straight back to
        :meth:`Trainer.set_predictors`.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM predictors ORDER BY rowid").fetchall()
        return [self._row_to_predictor(r) for r in rows]

    @staticmethod
    def _row_to_predictor(row):
        return Predictor(
            name=row['name'],
            processor=row['processor'],
            edges=json.loads(row['edges']) if row['edges'] else {},
            method=row['method'],
            adapter=json.loads(row['adapter']) if row['adapter'] else None,
            params=json.loads(row['params']) if row['params'] else {},
            desc=row['desc'],
            tag=json.loads(row['tag']) if row['tag'] else [],
            src_trial=row['src_trial'],
            src_experimenter=row['src_experimenter'],
            pipeline_version=row['pipeline_version'],
        )

    def __repr__(self):
        return f"<PredictorStore {self.db_path}>"
