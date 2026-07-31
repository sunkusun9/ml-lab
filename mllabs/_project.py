"""Project — owns the directory layout and ties the pieces together.

The components (PipelineBuilder, Experimenter, Trainer, Collectors, TrialStore)
each take a path and know nothing about each other. A Project hands out those
paths from one root and keeps the registry that spans them: the Trial store.
Pipeline versions are *not* a project-wide registry — each pipeline tracks its
own version counter in its own db (see :meth:`build_pipeline`).

Components still work standalone — a Project is a convenience over them, not a
requirement.
"""
from pathlib import Path

from ._cache import DataCache
from ._trial_store import TrialStore
from ._experimenter_store import ExperimenterStore
from .collector import Collectors


class Project:
    """Root of a project directory.

    Layout::

        {path}/
          experimenters.db    Experimenter registry, keyed by name
          trials.db           TrialStore (definitions + run history)
          pipelines/{name}/   PipelineBuilder db (incl. its own version counter),
                               and v{n}.pkl per built version
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

    def build_pipeline(self, builder):
        """Build *builder* and persist the result as its next version.

        Every call mints a new version — there is no content dedup, so
        rebuilding an unchanged builder still bumps the version. The counter
        lives in the builder's own db (wherever it was created — see
        :meth:`pipeline_builder`), not in a project-wide registry.

        Returns:
            Pipeline: The built Pipeline, with ``.version`` set.

        Raises:
            ValueError: If *builder* has no db (created without a path).
        """
        if builder._store is None:
            raise ValueError(
                "builder has no db path; create it via project.pipeline_builder(name)"
            )
        pipeline = builder.build()
        builder._store.save_version(pipeline)
        return pipeline

    def load_pipeline(self, name='pipeline', version=None):
        """Load a saved Pipeline version (latest if *version* is omitted)."""
        from ._pipeline_store import PipelineStore
        return PipelineStore(self.pipeline_path(name), name).load_version(version)

    def list_pipeline_versions(self, name='pipeline'):
        from ._pipeline_store import PipelineStore
        return PipelineStore(self.pipeline_path(name), name).list_versions()

    def __repr__(self):
        return f"<Project {self.path}>"
