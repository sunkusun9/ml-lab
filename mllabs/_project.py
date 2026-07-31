"""Project — owns the directory layout and ties the pieces together.

The components (PipelineBuilder, Experimenter, Trainer, Collectors, TrialStore)
each take a path and know nothing about each other. A Project hands out those
paths from one root and keeps the registries that span them: Pipeline versions
and the Trial store.

Components still work standalone — a Project is a convenience over them, not a
requirement.
"""
import sqlite3
import pickle as pkl
from pathlib import Path

from ._cache import DataCache
from ._trial_store import TrialStore
from ._experimenter_store import ExperimenterStore
from .collector import Collectors

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS pipeline_versions (
        name        TEXT NOT NULL,
        version     INTEGER NOT NULL,
        content_key TEXT NOT NULL,
        pipeline_id TEXT,
        path        TEXT NOT NULL,
        PRIMARY KEY (name, version)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_content
        ON pipeline_versions (name, content_key);
"""


class Project:
    """Root of a project directory.

    Layout::

        {path}/
          project.db          pipeline version index
          experimenters.db    Experimenter registry, keyed by name
          trials.db           TrialStore (definitions + run history)
          pipelines/{name}/   PipelineBuilder db, and v{n}.pkl per built version
          collectors/         Collectors registry
          exp/{name}/         Experimenter artifacts, keyed by its name
          trainers/{name}/    Trainer artifacts
          inferencers/{name}/ saved Inferencers

    Args:
        path (str | Path): Project root. Created if missing.
        cache_maxsize (int): Stage-output cache size in bytes, shared by every
            Experimenter in the project. Default 4 GB.
    """

    def __init__(self, path, cache_maxsize=4 * 1024 ** 3):
        self.path = Path(path)
        self.cache_maxsize = cache_maxsize
        self.cache = DataCache(maxsize=cache_maxsize)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / 'project.db'
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)
        self.trials = TrialStore(self.path)
        self.experimenters = ExperimenterStore(self.path)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def pipeline_path(self, name='pipeline'):
        return self._sub('pipelines', name)

    def exp_path(self, name):
        return self._sub('exp', name)

    def trainer_path(self, name):
        return self._sub('trainers', name)

    def inferencer_path(self, name):
        return self._sub('inferencers', name)

    def collectors_path(self):
        p = self.path / 'collectors'
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sub(self, kind, name):
        p = self.path / kind / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # components
    # ------------------------------------------------------------------

    def pipeline_builder(self, name='pipeline'):
        """A :class:`~mllabs.PipelineBuilder` stored under this project."""
        from ._pipeline import PipelineBuilder
        return PipelineBuilder(path=self.pipeline_path(name), name=name)

    def collectors(self):
        """The project's :class:`~mllabs.Collectors` registry, restored if saved."""
        return Collectors.load(self.collectors_path())

    def experimenter(self, name, data, **kwargs):
        """Create an Experimenter named *name* under ``{project}/exp/{name}``.

        Its name is its identity: it is both the directory and the key used in
        :class:`~mllabs.TrialStore` history.
        """
        from ._experimenter import Experimenter
        return Experimenter(self, name, data, **kwargs)

    def load_experimenter(self, name, data, **kwargs):
        """Reopen a previously created Experimenter by name."""
        from ._experimenter import Experimenter
        return Experimenter.load(self, name, data, **kwargs)

    def trainer(self, name, data, **kwargs):
        """Create a Trainer named *name* under ``{project}/trainers/{name}``."""
        from ._trainer import Trainer
        return Trainer(self, name, data, **kwargs)

    def load_trainer(self, name, data, **kwargs):
        """Reopen a previously created Trainer by name."""
        from ._trainer import Trainer
        return Trainer.load(self, name, data, **kwargs)

    def list_experimenters(self):
        """Names of every Experimenter registered in this project."""
        return self.experimenters.list_names()

    # ------------------------------------------------------------------
    # pipeline versions
    # ------------------------------------------------------------------

    def save_pipeline(self, pipeline, name='pipeline'):
        """Record *pipeline* as a version of *name* and return the version number.

        Versions are keyed by ``Pipeline.content_key()``, so re-saving an
        unchanged Pipeline returns the existing number instead of minting a new
        one — rebuilding is not a version bump.

        Stored as a pickle for now; the format is deliberately behind this
        method so it can change without touching callers.
        """
        content_key = pipeline.content_key()
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT version FROM pipeline_versions WHERE name = ? AND content_key = ?",
                (name, content_key),
            ).fetchone()
            if row:
                return row[0]
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM pipeline_versions WHERE name = ?",
                (name,),
            ).fetchone()
            version = row[0] + 1
            version_path = self.pipeline_path(name) / f'v{version}.pkl'
            with open(version_path, 'wb') as f:
                pkl.dump(pipeline, f)
            conn.execute(
                "INSERT INTO pipeline_versions (name, version, content_key, pipeline_id, path) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, version, content_key, pipeline.pipeline_id, str(version_path)),
            )
            return version

    def load_pipeline(self, name='pipeline', version=None):
        """Load a saved Pipeline version (latest if *version* is omitted)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if version is None:
                row = conn.execute(
                    "SELECT path FROM pipeline_versions WHERE name = ? "
                    "ORDER BY version DESC LIMIT 1", (name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT path FROM pipeline_versions WHERE name = ? AND version = ?",
                    (name, version),
                ).fetchone()
        if not row:
            raise KeyError(f"No saved pipeline {name!r} version {version!r}")
        with open(row[0], 'rb') as f:
            return pkl.load(f)

    def get_pipeline_version(self, pipeline, name='pipeline'):
        """Version number recorded for *pipeline*'s content, or ``None``."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT version FROM pipeline_versions WHERE name = ? AND content_key = ?",
                (name, pipeline.content_key()),
            ).fetchone()
        return row[0] if row else None

    def list_pipeline_versions(self, name='pipeline'):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM pipeline_versions WHERE name = ? ORDER BY version", (name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_version(self, content_key, name='pipeline'):
        """Version number for a ``content_key`` as stored in run history."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT version FROM pipeline_versions WHERE name = ? AND content_key = ?",
                (name, content_key),
            ).fetchone()
        return row[0] if row else None

    def __repr__(self):
        return f"<Project {self.path}>"
