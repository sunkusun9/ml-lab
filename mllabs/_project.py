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
from ._project_store import ProjectStore
from .collector import Collectors


class Project:
    """Root of a project directory.

    Layout::

        {path}/
          project.db          ProjectStore — which Experimenters and Trainers exist
          trials.db           TrialStore (definitions + run history)
          pipelines/{name}/   PipelineBuilder db (incl. its own version counter),
                               and v{n}.pkl per built version
          collectors/         Collectors registry
          exp/{name}/         Experimenter — its own store, Pipeline copy and
                               NodeStore, all self-contained
          trainers/{name}/    Trainer (same, own NodeStore)
          inferencers/{name}/ saved Inferencers

    Project holds only what is genuinely project-wide: the pipelines, the
    Collectors, the TrialStore, the shared cache, and the index of run names.
    Everything about an individual run — its splitters, its data key, the
    Pipeline it adopted, its node artifacts and history — belongs to that
    run's own directory, so it can be reopened without a Project at all.

    Args:
        path (str | Path): Project root. Created if missing.
        cache_maxsize (int): Stage-output cache size in bytes, shared by every
            Experimenter/Trainer in the project. Default 4 GB.
    """

    def __init__(self, path, cache_maxsize=4 * 1024 ** 3):
        self.path = Path(path)
        self.cache_maxsize = cache_maxsize
        self.cache = DataCache(maxsize=cache_maxsize)
        self.path.mkdir(parents=True, exist_ok=True)
        self.trials = TrialStore(self.path)
        self.store = ProjectStore(self.path)

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
        return Collectors(self.collectors_path())

    def experimenter(self, name, data, pipeline_name='pipeline', pipeline_version=None, **kwargs):
        """Create an Experimenter named *name* under ``{project}/exp/{name}``.

        Its name is its identity: it is both the directory and the key used in
        :class:`~mllabs.TrialStore` history. Creating one *starts* a run — use
        :meth:`load_experimenter` to reopen an existing one.

        All this factory adds to a bare ``Experimenter(...)`` is the path, the
        shared cache, a name in this project's index, and — when
        *pipeline_version* is given — resolving that version into a Pipeline
        for the new run to adopt.
        """
        from ._experimenter import Experimenter
        exp = Experimenter(self.exp_path(name), name, data, cache=self.cache, **kwargs)
        self.store.register_experimenter(name)
        if pipeline_version is not None:
            exp.set_pipeline(self.load_pipeline(pipeline_name, pipeline_version),
                             pipeline_name)
        return exp

    def load_experimenter(self, name, data, data_key=None, aug_data=None):
        """Reopen a previously created Experimenter by name.

        Delegates to :meth:`Experimenter.load_experimenter`, which reads
        everything out of that run's own directory — no Pipeline version is
        resolved here.

        Args:
            name (str): Experimenter name — its directory under ``exp/``.
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): If the saved experiment has a
                ``data_key``, this must match.

        Raises:
            KeyError: If no Experimenter named *name* exists in this project.
            ValueError: If *data_key* does not match the saved value.
        """
        from ._experimenter import Experimenter
        return Experimenter.load_experimenter(
            self.exp_path(name), data, data_key=data_key,
            aug_data=aug_data, cache=self.cache,
        )

    def trainer(self, name, data, pipeline_name='pipeline', pipeline_version=None, **kwargs):
        """Create a Trainer named *name* under ``{project}/trainers/{name}``.

        Same shape as :meth:`experimenter`: the path, the shared cache, a name
        in this project's index, and version resolution.
        """
        from ._trainer import Trainer
        trainer = Trainer(self.trainer_path(name), name, data, cache=self.cache, **kwargs)
        self.store.register_trainer(name)
        if pipeline_version is not None:
            trainer.set_pipeline(self.load_pipeline(pipeline_name, pipeline_version),
                                 pipeline_name)
        return trainer

    def load_trainer(self, name, data, aug_data=None):
        """Reopen a previously created Trainer by name.

        Delegates to :meth:`Trainer.load_trainer`, which reads everything out
        of that Trainer's own directory — no Pipeline version is resolved here.
        """
        from ._trainer import Trainer
        return Trainer.load_trainer(self.trainer_path(name), data,
                                    aug_data=aug_data, cache=self.cache)

    def list_experimenters(self):
        """Names of every Experimenter created through this project."""
        return self.store.list_experimenters()

    def list_trainers(self):
        """Names of every Trainer created through this project."""
        return self.store.list_trainers()

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
