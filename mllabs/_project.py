"""Project — owns the directory layout and ties the pieces together.

The components (PipelineBuilder, Experimenter, Trainer, Collectors, TrialStore)
each take a path and know nothing about each other. A Project hands out those
paths from one root, holds what is project-wide — the dataset, the pipeline,
the Trial store — and keeps what it manages so the same name gives back the
same object.

Components still work standalone — a Project is a convenience over them, not a
requirement. The dataset is the one thing they cannot restore alone, which is
why it lives here.
"""
import pickle
import shutil
from pathlib import Path

from ._cache import DataCache
from ._trial_store import TrialStore
from ._project_store import ProjectStore
from ._common import format_errors

DATA_FILE = 'data.pkl'
AUG_DATA_FILE = 'aug_data.pkl'


class Project:
    """Root of a project directory.

    Layout::

        {path}/
          project.db          ProjectStore — what this project manages
          trials.db           TrialStore (definitions + execution history)
          data.pkl            the dataset everything here is built against
          aug_data.pkl        data appended to inner train splits, if any
          pipeline/           pipeline.db — the working copy, i.e. the open
                               version — plus v{n}.pkl and v{n}_builder.pkl
                               per published version
          exp/{name}/         Experimenter — its own store, Pipeline copy,
                               NodeStore and Collectors, all self-contained
          trainers/{name}/    Trainer (same, own NodeStore)
          inferencers/{name}/ saved Inferencers

    Project holds what is project-wide — the dataset, the pipelines, the
    TrialStore, the shared cache — and keeps the components it hands out, so
    asking for the same name twice gives the same object rather than a second
    one over the same directory. Everything about an individual Experimenter or
    Trainer — its splitters, the Pipeline it adopted, its node artifacts and
    history, its Collectors — still belongs to its own directory, so it can be
    reopened without a Project at all.

    The dataset is the exception that makes the rest work: it is the one thing
    they cannot restore from their own directories, so without it here every
    question about the project needs the caller to bring a dataframe.

    There is no ``save()``. Each component writes through as it changes, and a
    second place to persist from would be a second source of truth.

    Args:
        path (str | Path): Project root. Created if missing.
        data: Dataset for this project. Stored under ``data.pkl`` and handed to
            every Experimenter/Trainer that does not override it.
        aug_data: Data appended at DataSource level to inner train splits.
        cache_maxsize (int): Stage-output cache size in bytes, shared by every
            Experimenter/Trainer in the project. Default 4 GB.
    """

    def __init__(self, path, data=None, aug_data=None, cache_maxsize=4 * 1024 ** 3):
        self.path = Path(path)
        self.cache_maxsize = cache_maxsize
        self.cache = DataCache(maxsize=cache_maxsize)
        self.path.mkdir(parents=True, exist_ok=True)
        self.trials = TrialStore(self.path)
        self.store = ProjectStore(self.path)
        self._data = None
        self._aug_data = None
        self._pipeline = None
        self._experimenters = {}
        self._trainers = {}
        self._seed_base_version()
        if data is not None:
            self.set_data(data)
        if aug_data is not None:
            self.set_aug_data(aug_data)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    @property
    def data(self):
        """The project dataset, read from disk on first use (``None`` if unset)."""
        if self._data is None:
            self._data = self._read_data(DATA_FILE)
        return self._data

    @property
    def aug_data(self):
        """Augmentation data, read from disk on first use (``None`` if unset)."""
        if self._aug_data is None:
            self._aug_data = self._read_data(AUG_DATA_FILE)
        return self._aug_data

    def set_data(self, data):
        """Store *data* as this project's dataset."""
        self._data = data
        self._write_data(DATA_FILE, data)

    def set_aug_data(self, aug_data):
        """Store *aug_data* as this project's augmentation data."""
        self._aug_data = aug_data
        self._write_data(AUG_DATA_FILE, aug_data)

    def _read_data(self, fname):
        p = self.path / fname
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)

    def _write_data(self, fname, data):
        with open(self.path / fname, 'wb') as f:
            pickle.dump(data, f)

    def _resolve_data(self, data):
        if data is not None:
            return data
        if self.data is None:
            raise ValueError(
                'This project has no data. Pass data= to this call, or set it '
                'once with Project(path, data=...) / project.set_data(...).'
            )
        return self.data

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    @property
    def pipeline_path(self):
        p = self.path / 'pipeline'
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exp_path(self, name):
        return self._sub('exp', name)

    def trainer_path(self, name):
        return self._sub('trainers', name)

    def _sub(self, kind, name):
        p = self.path / kind / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # components
    # ------------------------------------------------------------------

    @property
    def pipeline(self):
        """This project's :class:`~mllabs.PipelineBuilder` — the working copy.

        One per project. The builder *is* the ``open`` version: it is the only
        editable state there is, and building it publishes whatever it currently
        says (see :meth:`PipelineBuilder.build`). Asking twice gives the same
        object rather than a second builder over the same db, which would let
        two in-memory copies of the definitions drift apart.
        """
        if self._pipeline is None:
            from ._pipeline import PipelineBuilder
            self._pipeline = PipelineBuilder(path=self.pipeline_path, name='pipeline')
        return self._pipeline

    @property
    def experimenters(self):
        """This project's Experimenters, by name.

        The objects, opened on demand — which needs the project's data, that
        being the one thing an Experimenter cannot read back from its own
        directory. :meth:`list_experimenters` answers the cheaper question of
        which names this project manages.

        Only names not already held are opened, so one added through
        :meth:`add_experimenter` comes back as the very object that call
        returned. Two live Experimenters over one directory would each hold
        their own Collectors and node caches, and a change through one would be
        invisible to the other.
        """
        return self._hold(self._experimenters, self.list_experimenters(),
                          self._open_experimenter)

    @property
    def trainers(self):
        """This project's Trainers, by name. Opened like :attr:`experimenters`."""
        return self._hold(self._trainers, self.list_trainers(), self._open_trainer)

    def _hold(self, held, names, open_one):
        for name in names:
            if name not in held:
                held[name] = open_one(name)
        return held

    def add_experimenter(self, name, data=None, pipeline_version=None,
                         aug_data=None, **kwargs):
        """Add a new Experimenter named *name*, under ``{project}/exp/{name}``.

        An addition, so a taken name raises rather than being written over.
        Constructing an Experimenter splits the data afresh and resets its
        provenance, which over an existing one is damage rather than a reopen;
        reach for that one through :attr:`experimenters`.

        Its name is its identity: it is both the directory and the key used in
        :class:`~mllabs.TrialStore` history.

        Args:
            name (str): Experimenter name.
            data: Dataset for it. Defaults to the project's.
            pipeline_version (int, optional): Version for it to adopt.
                Omitted, it adopts the published one — version 0, the empty
                Pipeline, until something has been built. Adopting an
                unpublished edit is a separate act: build it first.
            aug_data: Augmentation data. Defaults to the project's.
            **kwargs: Passed to :class:`~mllabs.Experimenter` (``sp``, ``sp_v``,
                ``splitter_params``, ``title``, ``data_key``, ...).

        Raises:
            ValueError: If this project already manages *name*, or something
                else already occupies its directory.
        """
        from ._experimenter import Experimenter
        from ._experimenter_store import ExperimenterStore
        self._require_free('Experimenter', name, self.list_experimenters(),
                           'exp', ExperimenterStore.stored_at, 'experimenters')
        exp = Experimenter(
            self.exp_path(name), name, self._resolve_data(data),
            aug_data=aug_data if aug_data is not None else self.aug_data,
            cache=self.cache, trial_store=self.trials, **kwargs,
        )
        exp.set_pipeline(self.load_pipeline(pipeline_version))
        self.store.register_experimenter(name)
        self._experimenters[name] = exp
        return exp

    def add_trainer(self, name, data=None, pipeline_version=None, aug_data=None, **kwargs):
        """Add a new Trainer named *name*, under ``{project}/trainers/{name}``.

        Same shape as :meth:`add_experimenter`, and the same rule: a taken name
        raises. *pipeline_version* defaults the same way too — the published
        version, which is frozen and so is exactly what a Trainer is allowed to
        adopt. What it refuses is the working copy, and this never hands it one.
        """
        from ._trainer import Trainer
        from ._trainer_store import TrainerStore
        self._require_free('Trainer', name, self.list_trainers(),
                           'trainers', TrainerStore.stored_at, 'trainers')
        trainer = Trainer(
            self.trainer_path(name), name, self._resolve_data(data),
            aug_data=aug_data if aug_data is not None else self.aug_data,
            cache=self.cache, **kwargs,
        )
        trainer.set_pipeline(self.load_pipeline(pipeline_version))
        self.store.register_trainer(name)
        self._trainers[name] = trainer
        return trainer

    def remove_experimenter(self, name):
        """Delete the Experimenter named *name* — its directory, and its Trial history.

        The history goes because the name is what keys it. Left behind, it would
        attach to whatever is added under that name next, and ``exp()`` would
        skip folds it had never actually run — the silent kind of wrong. Trial
        *definitions* are untouched: those belong to the project.

        Raises:
            KeyError: If this project manages no Experimenter named *name*.
        """
        self._drop('Experimenter', name, self.list_experimenters(), 'exp')
        self.store.remove_experimenter(name)
        self.trials.remove_hist(experimenter=name)
        self._experimenters.pop(name, None)

    def remove_trainer(self, name):
        """Delete the Trainer named *name* — its directory, Predictors included.

        Nothing outside it to clean up: a Trainer's history lives in its own
        stores rather than in a project-wide one.

        Raises:
            KeyError: If this project manages no Trainer named *name*.
        """
        self._drop('Trainer', name, self.list_trainers(), 'trainers')
        self.store.remove_trainer(name)
        self._trainers.pop(name, None)

    def list_experimenters(self):
        """Names of every Experimenter this project manages."""
        return self.store.list_experimenters()

    def list_trainers(self):
        """Names of every Trainer this project manages."""
        return self.store.list_trainers()

    def _require_free(self, kind, name, managed, sub, stored_at, attr):
        # Two different questions. The index says what this project manages;
        # the directory says whether the spot is physically taken — by something
        # built outside this project, say. Either is a refusal, but they are not
        # the same fact and should not read as if they were.
        if name in managed:
            raise ValueError(
                f'{kind} {name!r} already exists in this project. Reach for it '
                f'through project.{attr}[{name!r}], or remove it first.'
            )
        path = self.path / sub / name
        if stored_at(path):
            raise ValueError(
                f'{kind} {name!r} is not in this project, but {path} already '
                f'holds one. Constructing over it would restart it.'
            )

    def _drop(self, kind, name, managed, sub):
        if name not in managed:
            raise KeyError(f'No {kind} named {name!r} in this project')
        path = self.path / sub / name
        if path.is_dir():
            shutil.rmtree(path)

    def _open_experimenter(self, name):
        from ._experimenter import Experimenter
        return Experimenter.load_experimenter(
            self.exp_path(name), self._resolve_data(None), aug_data=self.aug_data,
            cache=self.cache, trial_store=self.trials,
        )

    def _open_trainer(self, name):
        from ._trainer import Trainer
        return Trainer.load_trainer(
            self.trainer_path(name), self._resolve_data(None),
            aug_data=self.aug_data, cache=self.cache,
        )

    def set_trial(self, trial):
        """Add or update one Trial definition. Returns its name if anything changed.

        Authoring a Trial is a project-level act, separate from running one.
        Registration used to be a side effect of ``Experimenter.exp()``, which
        meant a Trial could not enter the project without being executed, and
        that one Experimenter could redefine a name another had already used.

        A name that has already succeeded somewhere is frozen. The
        history is keyed by name, and a Trial leaves no artifact, so a
        redefinition would silently attach the old results to a definition
        that never produced them — the rows would look like one Trial's record
        and be two. Changing such a Trial means either a new name, or
        :meth:`remove_trial` to give up the results with it.

        Args:
            trial (Trial): Definition to store.

        Returns:
            str | None: *trial*'s name if it was added or changed, ``None`` if
            the stored definition already matched — so the return value is the
            work list for the next ``exp()``.

        Raises:
            ValueError: If the name has a ``'built'`` fold in
                ``experiment_hist`` and the definition differs from the
                stored one.
        """
        changed = self.set_trials([trial])
        return changed[0] if changed else None

    def set_trials(self, trials):
        """:meth:`set_trial` for many. Returns the names that were added or changed.

        Nothing is written until every Trial has been checked, so a batch
        holding one frozen name changes nothing at all rather than leaving
        half of itself registered.
        """
        changed = [t for t in trials if not self.trials.has(t)]
        for trial in changed:
            built = self.trials.get_hist(trial_name=trial.name, status='built')
            if built:
                where = sorted({r['experimenter'] for r in built})
                raise ValueError(
                    f"Trial '{trial.name}' already ran successfully in {where} and "
                    f"cannot be redefined — its history would then describe a "
                    f"definition that never produced it. Use a new name, or "
                    f"project.remove_trial({trial.name!r}) to drop the results too."
                )
        self.trials.register_all(changed)
        return [t.name for t in changed]

    def show_error_trials(self, experimenter=None, traceback=False):
        """Errors from Trial execution, one line per failed fold.

        The Trial counterpart of ``Experimenter.show_error_nodes``, and it
        lives here for the same reason that one lives there: a failure is
        reported by whoever owns the history it is recorded in. Node history is
        the Experimenter's; Trial history is the project's, keyed by experimenter.

        Args:
            experimenter (str, optional): Report only this Experimenter's
                failures. ``None`` reports every one's.
            traceback (bool): Include full traceback in output.

        Returns:
            list[str] | None: One line per failed fold, or ``None`` if clean.
        """
        rows = [
            (r['trial_name'], r['outer_idx'], r['inner_idx'], r['info'])
            for r in self.trials.get_hist(experimenter=experimenter, status='error')
        ]
        return format_errors(rows, traceback)

    def pending_trials(self, experimenter=None):
        """Registered Trials that errored or have not run.

        The work list: a Trial with an ``'error'`` fold, and one with no
        history at all, both still have to be executed and are indistinguishable
        in what they need. Written by hand this filter tends to catch only the
        second — the notebook's version did, so an errored Trial silently
        dropped out of the list of things left to do.

        Deliberately coarse: a Trial interrupted partway through
        the folds is *not* reported, because deciding that means comparing
        history against a fold grid this store does not know. Running the
        returned names is safe either way — ``exp()`` skips the folds that
        are done.

        Args:
            experimenter (str, optional): Judge against this Experimenter's history.
                ``None`` asks whether a Trial ran anywhere in the project.

        Returns:
            list[str]: Trial names, in registration order.
        """
        pending = []
        for trial in self.trials.list_trials():
            rows = self.trials.get_hist(trial_name=trial.name,
                                        experimenter=experimenter)
            if not rows or any(r['status'] == 'error' for r in rows):
                pending.append(trial.name)
        return pending

    def remove_trial(self, name):
        """Drop *name* from the project — definition, history and collected data.

        A Trial leaves no artifact, so everything it produced is spread across
        stores that deliberately don't know about each other: its definition
        and per-fold history in the project's ``TrialStore``, and — inside
        every Experimenter that ran it — the collected data and the
        ``CollectHist`` rows describing it. Project is the only thing that
        sees all of them, so removing a Trial belongs here.

        Every Experimenter, with no way to ask for a subset. Leaving one out
        would leave results behind for a Trial the project no longer defines,
        and nothing afterwards would say which one still had them.

        The definition and the whole of its history go in one statement each;
        the rest is a pass over :attr:`experimenters`, delegating to
        :meth:`Experimenter.remove_trial_result`. That opens the ones not
        already held, which is what makes the removal reach a Collector
        answering from an in-memory cache (``ModelAttrCollector``/
        ``SHAPCollector``) — a registry reopened from disk would not be the
        instance holding that cache.

        Args:
            name (str): Trial name. Removing one that was never registered is
                a no-op, not an error.

        Raises:
            ValueError: If this project has no dataset, which opening an
                Experimenter needs.
        """
        self.trials.remove(name)
        self.trials.remove_hist(trial_name=name)
        for exp in self.experimenters.values():
            exp.remove_trial_result(name)

    # ------------------------------------------------------------------
    # pipeline versions
    # ------------------------------------------------------------------

    def load_pipeline(self, version=None):
        """A frozen version by number; without one, the published (current) one.

        A read, and only a read. It used to build :attr:`pipeline` when given
        no version, which publishes — so a call that reads like "load" minted a
        version, and adding an Experimenter did too. Minting now happens in one
        place, :meth:`PipelineBuilder.build`, where it is what you asked for.

        There is always something to return: a project publishes the empty
        Pipeline as version 0 when it is created.
        """
        return self._pipeline_store().load_version(version)

    def _seed_base_version(self):
        """Publish the empty Pipeline as version 0, once, at creation.

        Version 0 is what "this project has no pipeline yet" *is*, rather than
        an absence every caller has to handle. It is a real published row, so
        :meth:`load_pipeline` never comes up empty, a Trainer can adopt it
        (published is frozen — an empty definition has nothing to change under
        it), and "exactly one published version" is true from the start.

        Building an untouched working copy returns version 0 rather than
        minting: nothing was defined, so nothing was changed. Version 1 is the
        first real definition.
        """
        from ._pipeline import Pipeline, PipelineBuilder
        store = self._pipeline_store()
        if store.list_versions():
            return
        store.publish(Pipeline.empty(self.pipeline.pipeline_id),
                      PipelineBuilder(), version=0)

    def list_pipeline_versions(self):
        """Every frozen version: ``{version, status, path, builder_path}`` rows."""
        return self._pipeline_store().list_versions()

    def remove_pipeline_version(self, version):
        """Delete an archived version.

        Archived only — the published version is the current definition and the
        working copy is not a version at all. Nothing that ran against it
        breaks: every Experimenter and Trainer holds its own Pipeline copy. What
        is lost is what their provenance points at.
        """
        self._pipeline_store().remove_version(version)

    def _pipeline_store(self):
        return self.pipeline._store

    def __repr__(self):
        return f"<Project {self.path}>"
