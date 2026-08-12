from pathlib import Path
import numpy as np

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._predictor_store import PredictorStore, INIT, TRAINED, RETIRED, ERROR
from ._trainer_store import TrainerStore
from ._trial import Trial
from ._edge_dsl import parse, eval_expr, referenced_nodes
from ._logger import resolve_logger
from ._pipeline import Pipeline
from ._serialize import _obj_to_ref, serialize_value
from ._common import (resolve_common_status, require_built_pipeline,
                      require_published_pipeline)


class TrainFold:
    """Single split data flow for Trainer.

    Analogous to OuterFold for Experimenter — every split shares the same
    NodeStore (the Trainer's own, at its own base path, so it never collides
    with an Experimenter's even while both use natural split_idx/outer_idx
    values), told apart only by ``split_idx``.

    get_test_data() returns None (Trainer has no separate test set).
    """

    def __init__(self, split_idx, store, data, train_idx, valid_idx=None, cache=None, aug_data=None):
        self.split_idx = split_idx
        provider = DataWrapperProvider(data, train_idx, valid_idx=valid_idx, aug_data=aug_data)
        self.train_data_flows = [
            TrainDataFlow(
                store=store,
                data_source=provider,
                cache=cache,
                outer_idx=split_idx,
                inner_idx=0,
            )
        ]

    def set_data(self, data, cache=None, aug_data=None):
        self.train_data_flows[0].data_source.set_data(data, aug_data)
        if cache is not None:
            self.train_data_flows[0].cache = cache

    def get_test_data(self, edges, inner_idx=0):
        return None


class Trainer:
    """Runs cross-validation training on a subset of Pipeline nodes.

    No ``Project`` dependency, and nothing to inject but a ``cache``: like
    :class:`~mllabs.Experimenter`, a Trainer owns its own store — here a
    :class:`~mllabs._trainer_store.TrainerStore`, built from ``path`` — and
    everything needed to reopen it lives in that directory. ``Project``
    supplies the path and records the name in its index.

    Constructing is *creating*: it splits the data and writes fresh state, so
    pointing it at an existing directory starts that Trainer over rather than
    resuming it. :meth:`load_trainer` is how an existing one comes back, on
    exactly the splits it was trained with.

    Adopt a Pipeline with :meth:`set_pipeline` (never a constructor argument),
    and pass the Predictors to :meth:`train`. There is no separate "select"
    step: what you train is what you asked to train, and the Pipeline nodes
    those Predictors read are pulled in with them.

    Two artifact stores, not one. Pipeline nodes keep the Trainer's own base
    path; Predictors get their own directory underneath it. They are the same
    :class:`~mllabs._store.NodeStore` class — a Predictor's fitted model and
    per-split history are stored exactly like a node's — but separating them
    means a Predictor is told apart structurally rather than by which history
    table happened to record it, and it keeps the two ``__node_hist.db`` files
    from colliding. ``__predictors/`` also holds the
    :class:`~mllabs.PredictorStore` with the definitions themselves, so a
    reloaded Trainer comes back with its selection intact.

    Attributes:
        name (str): Trainer name.
        pipeline (Pipeline): The adopted Pipeline, kept as this Trainer's own
            ``pipeline.pkl``. ``pipeline_version`` is persisted too, but only
            as provenance — it names the published version this copy was
            taken from.
        predictors (list[Predictor]): Everything :meth:`train` has been given,
            read from ``predictor_defs``.
        selected_nodes (list[str]): Pipeline nodes those Predictors read,
            topologically ordered. Derived, not stored.
        node_store (NodeStore): Artifacts + history for Pipeline nodes.
        predictor_store (NodeStore): Artifacts + history for Predictors.
        predictor_defs (PredictorStore): The Predictor definitions themselves.
        train_folds (list[TrainFold]): Per-split data flows and artifact stores.
    """

    def __init__(self, path, name, data, splitter=None, splitter_params=None,
                 aug_data=None, cache=None):
        self.name = name
        self.data = wrap(data)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.splitter = splitter
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.cache = cache
        self._store = TrainerStore(self.path)
        self.node_store = NodeStore(self.path)
        self.predictor_path = self.path / '__predictors'
        self.predictor_store = NodeStore(self.predictor_path)
        self.predictor_defs = PredictorStore(self.predictor_path)
        self.aug_data = wrap(aug_data) if aug_data is not None else None

        self.pipeline = Pipeline.empty()

        split_indices = self._make_splits()
        self.train_folds = self._make_train_folds(split_indices)
        self.save()

    @staticmethod
    def load_trainer(path, data, aug_data=None, cache=None):
        """Reopen the Trainer rooted at *path*.

        Everything comes out of that directory — meta and splits from its
        ``__trainer.db``, the Pipeline from its ``pipeline.pkl``, the
        Predictors from ``__predictors/``. The splits come back as the stored
        indices rather than being recomputed, so the folds are exactly the
        ones that were trained.

        Args:
            path: The Trainer's base directory.
            data: Dataset to attach.
            aug_data (optional): External data appended to train splits.
            cache (DataCache, optional): Shared LRU cache.

        Returns:
            Trainer: The reopened Trainer.

        Raises:
            KeyError: If *path* holds no saved Trainer.
        """
        path = Path(path)
        if not TrainerStore.stored_at(path):
            raise KeyError(f"No trainer saved at {path}")
        store = TrainerStore(path)
        meta = store.fetch()
        if meta is None:
            raise KeyError(f"No trainer saved at {path}")
        splits = store.load_splits() or {}

        trainer = object.__new__(Trainer)
        trainer.name = meta['name']
        trainer.data = wrap(data)
        trainer.path = path
        trainer.splitter = splits.get('splitter')
        trainer.splitter_params = splits.get('splitter_params') or {}
        trainer.cache = cache
        trainer._store = store
        trainer.node_store = NodeStore(path)
        trainer.predictor_path = path / '__predictors'
        trainer.predictor_store = NodeStore(trainer.predictor_path)
        trainer.predictor_defs = PredictorStore(trainer.predictor_path)
        trainer.aug_data = wrap(aug_data) if aug_data is not None else None
        trainer.pipeline = Pipeline.empty()
        trainer.train_folds = trainer._make_train_folds(splits.get('split_indices'))

        pipeline = store.load_pipeline()
        if pipeline is not None:
            trainer.set_pipeline(pipeline)
        return trainer

    @property
    def pipeline_version(self):
        """The adopted Pipeline's version — always a frozen one, or ``None`` if unset.

        Read off :attr:`pipeline` rather than kept beside it, so there is no
        second copy to fall out of step with it.
        """
        return self.pipeline.version

    # ------------------------------------------------------------------
    # split / fold setup
    # ------------------------------------------------------------------

    def _make_splits(self):
        if self.splitter is None:
            return None
        data_native = unwrap(self.data)
        split_params = {'X': data_native}
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))
        return [
            (train_idx, valid_idx)
            for train_idx, valid_idx in self.splitter.split(**split_params)
        ]

    def _make_train_folds(self, split_indices):
        if split_indices is None:
            n_rows = self.data.get_shape()[0]
            full_idx = np.arange(n_rows)
            return [
                TrainFold(0, self.node_store, self.data, full_idx, valid_idx=None,
                          cache=self.cache, aug_data=self.aug_data)
            ]
        return [
            TrainFold(i, self.node_store, self.data, train_idx, valid_idx=valid_idx,
                      cache=self.cache, aug_data=self.aug_data)
            for i, (train_idx, valid_idx) in enumerate(split_indices)
        ]

    def get_n_splits(self):
        return len(self.train_folds)

    # ------------------------------------------------------------------
    # node selection
    # ------------------------------------------------------------------

    def set_pipeline(self, pipeline):
        """Adopt an already-loaded Pipeline. It must be a published version.

        Takes the Pipeline object directly rather than a version number —
        this class has no way to load one by version itself (see the class
        docstring); ``Project.add_trainer()`` resolves that before calling
        this. ``self.pipeline_version`` is read straight off *pipeline* (its
        ``.version``), never tracked separately.

        A draft is refused, and only a draft. What a Trainer produces is what
        gets deployed, so it has to be able to say what it was trained against,
        and a draft carries no number to say it with. *Which* published version
        is not this call's business: an older one answers that question as well
        as the newest, and training a Predictor evaluated against an older
        definition is exactly what adopting that version is for.

        The Pipeline is written to this Trainer's own ``pipeline.pkl``, which
        :meth:`load` reads back — reopening needs only this directory, never
        a Project.

        Which nodes are actually selected depends on the Predictors — pass
        them to :meth:`train`.

        Diffs the two Pipelines (:meth:`Pipeline.diff_from`) and resets the
        node artifacts the change invalidated via :meth:`reset_nodes`, which
        also cascades into any selected Predictor that reads a reset node —
        unlike ``Experimenter.set_pipeline``, a Trainer has no notion of a
        past execution to preserve, so a Predictor trained against a
        since-changed node is simply stale.

        The Predictors already registered here move onto the adopted version
        with it (see ``_restamp_predictors``): adopting is what says which
        definition this Trainer trains against, and stranding them behind would
        leave their artifacts deleted and unrebuildable.

        Args:
            pipeline (Pipeline): Already-built, already-published Pipeline.

        Raises:
            ValueError: If *pipeline* is a draft.
        """
        require_built_pipeline(pipeline)
        require_published_pipeline(pipeline)
        pipeline.check_data_compatibility(self.data)
        # See Experimenter.set_pipeline: the empty Pipeline means nothing has
        # been adopted, so there is nothing to invalidate against.
        retiring = self.retiring_predictors(pipeline)
        if not self.pipeline.is_empty:
            stale = pipeline.diff_from(self.pipeline)
            if stale:
                affected, predictor_names = self._reach_of(sorted(stale))
                self._drop_artifacts(affected, predictor_names)
        self.pipeline = pipeline
        self._store.save_pipeline(pipeline)
        self.save()
        for name in retiring:
            self.predictor_defs.set_status(name, RETIRED)
        self._restamp_predictors(exclude=retiring)
        return pipeline

    def stale_nodes(self, pipeline):
        """Pipeline nodes adopting *pipeline* would reset, sorted.

        The node half of what a version switch costs;
        :meth:`retiring_predictors` is the other. They are different kinds of
        loss and are worth asking for separately — a node comes back on the
        next :meth:`train`, a retired Predictor does not come back at all.

        Same traversal :meth:`set_pipeline` acts on, so this is a preview
        rather than a second opinion about it. The counterpart to
        ``Experimenter.stale_nodes``, which a Trainer had no equivalent of.

        Args:
            pipeline (Pipeline): The version being considered.

        Returns:
            list[str]: Node names, sorted.
        """
        require_built_pipeline(pipeline)
        if self.pipeline.is_empty:
            return []
        stale = pipeline.diff_from(self.pipeline)
        if not stale:
            return []
        affected, predictor_names = self._reach_of(sorted(stale))
        return sorted(affected - predictor_names)

    def retiring_predictors(self, pipeline):
        """Which Predictors adopting *pipeline* would retire, before adopting.

        The question has to be asked beforehand, because afterwards there is
        nothing left to ask: the artifacts are gone and the only record of why
        is the status this call decides. Same shape as
        ``Experimenter.stale_nodes``, and for the same reason — this is the
        code :meth:`set_pipeline` acts on, not a second opinion about it.

        A Predictor is retired when the change reaches a node it reads, which
        means the inputs that produced its artifact no longer exist under the
        adopted definition. One that reads only nodes defined identically in
        both versions keeps its artifact and is restamped onto the new version
        instead: training it there would produce the same model, so claiming
        the new version is true.

        Nothing is retired twice, and adopting onto an empty Pipeline retires
        nothing — there is no previous definition to have diverged from.

        Args:
            pipeline (Pipeline): The version being considered.

        Returns:
            list[str]: Predictor names, sorted.
        """
        require_built_pipeline(pipeline)
        if self.pipeline.is_empty:
            return []
        stale = pipeline.diff_from(self.pipeline)
        if not stale:
            return []
        affected, predictor_names = self._reach_of(sorted(stale))
        statuses = self.predictor_defs.list_status()
        return sorted(name for name in affected & predictor_names
                      if statuses.get(name) != RETIRED)

    def _restamp_predictors(self, exclude=()):
        """Move the surviving Predictors onto the version just adopted.

        Only the survivors. A Predictor whose artifact came through the switch
        reads nodes defined identically in both versions, so it would train to
        the same model under the new one and naming it is accurate. A retired
        one is the opposite case — its inputs really did change — and it goes
        on naming the version it was actually trained against, which is the
        only true thing left to say about it.

        The gate in :meth:`train` is not weakened by this, because it guards a
        different motion — a Predictor arriving from elsewhere, promoted from a
        Trial evaluated against some other version. That one is still refused,
        which is the case where training would ship something unmeasured.

        Args:
            exclude (iterable[str]): Names not to move — the ones being
                retired by this same adoption.
        """
        exclude = set(exclude)
        statuses = self.predictor_defs.list_status()
        for predictor in self.predictor_defs.list_predictors():
            if predictor.name in exclude:
                continue
            if statuses.get(predictor.name) == RETIRED:
                continue
            if predictor.pipeline_version != self.pipeline_version:
                predictor.pipeline_version = self.pipeline_version
                self.predictor_defs.register(predictor)

    @property
    def predictors(self):
        """Every Predictor this Trainer has been asked to train.

        Read from ``predictor_defs`` rather than held as state: :meth:`train`
        registers what it is given, so this is what has actually been trained
        (or attempted) here, and it survives a reload for free.
        """
        return self.predictor_defs.list_predictors()

    @property
    def selected_nodes(self):
        """Pipeline nodes the registered Predictors read, topologically ordered."""
        return self._nodes_for(self.predictors)

    def predictor_names(self):
        return [p.name for p in self.predictors]

    def predictor_specs(self):
        """``{name: ProcessorSpec}`` for the registered Predictors."""
        return {p.name: p.get_spec() for p in self.predictors}

    def predictor_status(self, name):
        """Where *name* stands: ``init`` / ``trained`` / ``retired`` /
        ``error``, or ``None`` if it is not registered here."""
        return self.predictor_defs.get_status(name)

    def predictor_statuses(self):
        """``{name: status}`` for every registered Predictor."""
        return self.predictor_defs.list_status()

    def _observed_status(self, name):
        """The status implied by what is on disk right now.

        Retirement is never among the answers, and cannot be: retiring drops
        the artifacts and leaves the history rows, which is exactly what a
        reset leaves behind too. That is the whole reason the status is
        stored rather than derived — this method covers the other three.
        """
        if any(s == 'error' for s in self.predictor_store.get_status(name).values()):
            return ERROR
        n = len(self.train_folds)
        built = sum(1 for i in range(n)
                    if self.predictor_store.status(name, i, 0) == 'built')
        return TRAINED if n and built == n else INIT

    def _sync_predictor_status(self, names):
        """Write back what training left on disk, retirement excepted."""
        statuses = self.predictor_defs.list_status()
        for name in names:
            if statuses.get(name) == RETIRED:
                continue
            self.predictor_defs.set_status(name, self._observed_status(name))

    def _has_pending(self, name):
        """Whether any split of *name* still needs training."""
        return any(self.predictor_store.status(name, i, 0) != 'built'
                   for i in range(len(self.train_folds)))

    def _nodes_for(self, predictors):
        """Pipeline nodes *predictors* need, topologically ordered.

        With no Predictors at all, every node is selected — training the whole
        preprocessing graph is still meaningful on its own.
        """
        if not predictors:
            return list(self.pipeline.topo_order())
        needed = set()
        for predictor in predictors:
            for dsl_string in predictor.edges.values():
                for name in referenced_nodes(dsl_string):
                    if name is not None and name not in needed:
                        needed.add(name)
                        self._collect_upstream(self.pipeline, name, needed)
        return [n for n in self.pipeline.topo_order() if n in needed]

    def _collect_upstream(self, pipeline, node_name, selected):
        spec = pipeline.get_node_spec(node_name)
        for dsl_string in spec.edges.values():
            for source_node in referenced_nodes(dsl_string):
                if source_node is not None and source_node not in selected:
                    selected.add(source_node)
                    self._collect_upstream(pipeline, source_node, selected)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _store_for(self, name):
        """Which of the two stores holds *name*'s artifacts."""
        return self.predictor_store if name in set(self.predictor_names()) else self.node_store

    def get_status(self, node_name):
        """Return the disk status of a node or Predictor across all splits.

        Returns ``'built'``, ``None`` (init, or errored — ``NodeStore`` only
        knows whether ``obj.pkl`` exists; see :meth:`get_node_error` for the
        error detail), or ``'inconsistent'`` if splits differ.
        """
        store = self._store_for(node_name)
        return resolve_common_status(
            store.status(node_name, fold.split_idx, 0)
            for fold in self.train_folds
        )

    def get_node_error(self, node_name):
        """Return the error dict for a node or Predictor in error state, or ``None``.

        Predictors are covered as well as Pipeline nodes: each store records
        its own history, so a failed Predictor leaves an ``'error'`` row in
        ``predictor_store`` the same way a failed node does in ``node_store``.
        """
        for r in self._store_for(node_name).get_hist(node_name=node_name):
            if r['status'] == 'error':
                return (r['info'] or {}).get('error')
        return None

    def _reach_of(self, nodes):
        """Everything dropping *nodes* would take with it.

        Returns ``(affected, predictor_names)``: the node and Predictor names
        the drop reaches, and the set of names among them that are Predictors
        rather than Pipeline nodes.

        Split out from the dropping so that asking and doing run the same
        traversal — :meth:`retiring_predictors` previews with it and
        :meth:`set_pipeline` acts on it, which is what keeps a preview from
        being a second opinion about the real thing.
        """
        pipeline = self.pipeline
        predictor_names = set(self.predictor_names())
        selected_set = set(self.selected_nodes) | predictor_names
        affected = set(n for n in nodes if n in selected_set)

        # Only Pipeline nodes have downstream nodes to cascade into; a
        # Predictor is a leaf and is not part of the pipeline graph at all.
        queue = [n for n in affected if n in pipeline.nodes]
        while queue:
            n = queue.pop(0)
            for downstream in pipeline.nodes[n].output_edges:
                if downstream in selected_set and downstream not in affected:
                    affected.add(downstream)
                    if downstream in pipeline.nodes:
                        queue.append(downstream)

        # A Predictor reading a dropped node cannot keep its own artifact:
        # the inputs that produced it are gone.
        node_reset = affected & set(self.selected_nodes)
        if node_reset:
            for predictor in self.predictors:
                if predictor.name in affected:
                    continue
                if predictor.node_names() & node_reset:
                    affected.add(predictor.name)

        return affected, predictor_names

    def _drop_artifacts(self, affected, predictor_names):
        """Delete the artifacts of *affected*, and forget them in every flow.

        Says nothing about what it means — :meth:`reset_nodes` calls it to
        return a Predictor to ``init`` and :meth:`set_pipeline` calls it to
        retire one. The files removed are the same either way; the difference
        is the status written afterwards, and whether a Job is ever built
        again.
        """
        for name in affected:
            if name in predictor_names:
                for fold in self.train_folds:
                    self.predictor_store.reset_node(name, fold.split_idx, 0)
                    fold.train_data_flows[0].node_objs.pop(name, None)
                    fold.train_data_flows[0]._node_edges.pop(name, None)
            else:
                for fold in self.train_folds:
                    fold.train_data_flows[0].reset_node(name)

        if self.cache is not None:
            self.cache.clear_nodes(affected)

    def reset_nodes(self, nodes):
        """Drop *nodes*' artifacts so the next :meth:`train` rebuilds them.

        This is the initialising sense of the word, and it is deliberately
        not the retiring one. Asking for a rebuild leaves the Pipeline
        definition untouched, so a Predictor caught by the cascade will be fed
        exactly the inputs it had before and would train to the same model —
        there is nothing to end, and it goes back to ``init``.

        A version switch is the other case, and takes the other path: there
        the inputs really did change, the same model cannot be produced again,
        and the Predictor is retired rather than reset (:meth:`set_pipeline`).
        A Predictor already retired stays that way — a rebuild request is not
        a resurrection.
        """
        affected, predictor_names = self._reach_of(nodes)
        self._drop_artifacts(affected, predictor_names)
        for name in affected & predictor_names:
            if self.predictor_defs.get_status(name) != RETIRED:
                self.predictor_defs.set_status(name, INIT)
        self.save()

    def remove_predictor(self, name):
        """Erase *name* from this Trainer entirely — definition, artifacts,
        history.

        Retiring deliberately leaves all three standing: the definition is the
        inscription on the tombstone, and the history says what ran before it.
        They accumulate, so clearing them is a separate act with its own call
        rather than a side effect of the switch that ended the Predictor —
        the shape ``Project.remove_trial`` has, for the same reason.

        Works on any status, not only retired ones.
        """
        for fold in self.train_folds:
            self.predictor_store.reset_node(name, fold.split_idx, 0)
            fold.train_data_flows[0].node_objs.pop(name, None)
            fold.train_data_flows[0]._node_edges.pop(name, None)
        self.predictor_store.remove_hist(node_name=name)
        self.predictor_defs.remove(name)
        if self.cache is not None:
            self.cache.clear_nodes([name])
        self.save()

    def purge_retired(self):
        """Remove every retired Predictor. Returns the names removed."""
        names = self.predictor_defs.list_names(status=RETIRED)
        for name in names:
            self.remove_predictor(name)
        return names

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def _make_predictor_jobs(self, predictors):
        """One Job per (Predictor, split) still needing training.

        A Trainer has a single flow per split, so the fold coordinate is
        ``(split_idx, 0)`` — the same shape the Experimenter uses. The flow
        passed to the Job is still the node flow: that is what feeds the
        Predictor its inputs. Only the artifacts land elsewhere, via the
        store the executor is handed.

        A split is skipped only if its artifact is already built — the same
        disk-based test :meth:`_make_node_jobs` uses, so a redefined
        Predictor does not by itself force a rerun of splits already built.

        A retired Predictor is skipped whole. Its artifacts are gone, so the
        disk test alone would hand it a Job for every split and train it back
        into existence, which is precisely what retirement means not to
        happen. Disk cannot tell that case from a reset one; the status can.
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        statuses = self.predictor_defs.list_status()
        gpu_cache = {}
        jobs = []
        for predictor in predictors:
            if statuses.get(predictor.name) == RETIRED:
                continue
            spec = predictor.get_spec()
            if predictor.name not in gpu_cache:
                adapter = resolve_node_adapter(spec.processor, spec.adapter)
                gpu_cache[predictor.name] = adapter.get_gpu_usage(spec.params) != GPU_NO
            for split_idx, fold in enumerate(self.train_folds):
                if self.predictor_store.status(predictor.name, split_idx, 0) == 'built':
                    continue
                jobs.append(Job(predictor.name, spec, split_idx, 0,
                                fold.train_data_flows[0],
                                need_gpu=gpu_cache[predictor.name]))
        return jobs

    def _make_node_jobs(self, pipeline, node_names, gpu_id_list):
        """Expand Pipeline node names into per-split Jobs.

        Same role ``_make_predictor_jobs`` plays for Predictors: skip
        decisions (a split already built for a given node) live here, not in
        the executor — it only orders dispatch among what's left.
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        jobs = []
        for name in node_names:
            node = pipeline.get_node(name)
            spec = node.get_spec()
            if gpu_id_list and name not in gpu_cache:
                adapter = resolve_node_adapter(node.processor, node.adapter)
                gpu_cache[name] = adapter.get_gpu_usage(node.params) != GPU_NO
            need_gpu = gpu_cache.get(name, False)

            for split_idx, fold in enumerate(self.train_folds):
                flow = fold.train_data_flows[0]
                if flow.status(name) == 'built':
                    continue
                jobs.append(Job(name, spec, split_idx, 0, flow, need_gpu=need_gpu))
        return jobs

    def train(self, predictors=None, n_jobs=1, gpu_id_list=None, logger=None):
        """Train *predictors* and the Pipeline nodes they read, across all splits.

        Nodes go first (topological order), then the Predictors. Only what is
        not already built on disk is run, so calling this again after adding a
        Predictor trains just that one — nothing already trained is discarded.
        Node staleness is settled when a Pipeline is adopted
        (:meth:`set_pipeline`), which does reset what it invalidates.

        The Predictors are registered in ``predictor_defs``, which is what
        :attr:`predictors`, :meth:`process` and :meth:`to_inferencer` read, and
        what a reopened Trainer comes back with. Registering is an upsert: a
        Predictor trained by an earlier call keeps its definition and its
        artifacts.

        **Each one has to be defined against the version this Trainer
        adopted.** One promoted from a Trial carries the version its candidate
        was evaluated on, so training it elsewhere would ship something that
        was never measured; adopt that version instead, since every published
        version is adoptable. One authored here with no version gets this
        Trainer's — the version it is being defined against is the one in
        front of it, and this Trainer knows that without a registry to ask.

        Args:
            predictors (list[Predictor], optional): What to train. A
                :class:`~mllabs.Trial` is rejected — promote it explicitly with
                ``Predictor.from_trial(trial, experimenter=...)`` so where it
                came from is recorded rather than guessed. Omit to train the
                Pipeline nodes alone (or to resume the already-registered
                Predictors, which are used when this is ``None``).
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.

        Raises:
            TypeError: If *predictors* holds a :class:`~mllabs.Trial`.
            ValueError: If a Predictor is defined against a different Pipeline
                version than the adopted one.
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, NodeInfoTracker

        logger = resolve_logger(logger)
        pipeline = self.pipeline
        pipeline.check_data_compatibility(self.data)

        statuses = self.predictor_defs.list_status()
        if predictors is None:
            # Resuming: retired names are not work left over, they are over.
            predictors = [p for p in self.predictors
                          if statuses.get(p.name) != RETIRED]
        else:
            predictors = list(predictors)
            for p in predictors:
                if isinstance(p, Trial):
                    raise TypeError(
                        f"train() got a Trial ({p.name!r}); promote it with "
                        f"Predictor.from_trial(trial, experimenter=...) first"
                    )
                if statuses.get(p.name) == RETIRED:
                    raise ValueError(
                        f"Predictor '{p.name}' is retired: a Pipeline version "
                        f"switch changed the nodes it reads, so the model it "
                        f"held cannot be produced again. Register the same "
                        f"specification under another name to train it against "
                        f"version {self.pipeline_version}."
                    )
                if p.pipeline_version is None:
                    p.pipeline_version = self.pipeline_version
            self.predictor_defs.register_all(predictors)
        for p in predictors:
            # Only what would actually be trained. A Predictor whose splits
            # are all built files no Job, so its version is a record of what
            # produced it rather than a claim about what is about to run —
            # and after a version switch that record is deliberately left
            # behind (see _restamp_predictors).
            if p.pipeline_version != self.pipeline_version and self._has_pending(p.name):
                raise ValueError(
                    f"Predictor '{p.name}' is defined against pipeline version "
                    f"{p.pipeline_version}, and this Trainer has adopted "
                    f"version {self.pipeline_version}. Training it here would "
                    f"ship a definition that was never evaluated — adopt that "
                    f"version with set_pipeline(project.load_pipeline("
                    f"{p.pipeline_version!r}))."
                )

        # Node staleness is settled when a Pipeline is adopted
        # (set_pipeline); Predictor staleness is checked per job below.
        node_jobs = self._make_node_jobs(pipeline, self._nodes_for(predictors), gpu_id_list)
        predictor_jobs = self._make_predictor_jobs(predictors)

        if not node_jobs and not predictor_jobs:
            logger.info("No nodes to train")
            return

        total = len(node_jobs) + len(predictor_jobs)
        n_jobs = min(n_jobs, total)
        base_tracker = LoggerExecuteTracker(total, n_jobs, logger)
        # Each store records its own history, so the two kinds of job get a
        # tracker apiece — same class, different store.
        node_tracker = NodeInfoTracker(base_tracker, self.node_store, self.pipeline_version)
        predictor_tracker = NodeInfoTracker(base_tracker, self.predictor_store, self.pipeline_version)
        error_nodes = set()
        try:
            if node_jobs:
                if n_jobs > 1:
                    node_errors = _execute_multi(
                        node_jobs, n_jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=node_tracker,
                        log_dir=self.path / '__worker_logs', chained=True)
                else:
                    node_errors = _execute_single(
                        node_jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=node_tracker,
                        chained=True)
                error_nodes.update(n for _, _, n in node_errors)

            if predictor_jobs:
                # A Predictor is a leaf: it reads the node flow but nothing
                # reads it, so these run unchained — no dependency gate, since
                # a Predictor whose node never built must surface as a prep
                # error rather than vanish. Its artifact goes to
                # predictor_store, which the node flow's lazy load never
                # reaches. No Collectors either, a Trainer isn't an
                # Experimenter.
                if n_jobs > 1:
                    predictor_errors = _execute_multi(
                        predictor_jobs, n_jobs, self.predictor_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=predictor_tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    predictor_errors = _execute_single(
                        predictor_jobs, self.predictor_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=predictor_tracker)
                error_nodes.update(n for _, _, n in predictor_errors)
        finally:
            base_tracker.close()

        target_all = sorted({j.name for j in node_jobs}) + sorted({j.name for j in predictor_jobs})
        n_ok = len(target_all) - len(error_nodes)
        if error_nodes:
            logger.info(
                f"Train complete: {n_ok}/{len(target_all)} node(s), "
                f"{len(error_nodes)} error(s): {sorted(error_nodes)}"
            )
        else:
            logger.info(f"Train complete: {len(target_all)} node(s)")

        self._sync_predictor_status([p.name for p in predictors])
        self.save()

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------

    def process(self, data, v=None):
        """Apply trained processors to new data, yielding one result per split.

        Args:
            data: Input dataset.
            v: Output column filter applied to Predictor outputs.

        Yields:
            DataFrame: Concatenated Predictor outputs for each split.
        """
        data = wrap(data)
        predictor_edges = {p.name: p.edges for p in self.predictors}
        for fold in self.train_folds:
            flow = fold.train_data_flows[0]
            outputs = []
            for name in self.predictor_names():
                if name not in flow.node_objs:
                    # A Predictor's model is in a store of its own, so the
                    # flow's own lazy load would never find it — put it in
                    # node_objs directly, which is consulted before the store.
                    if self.predictor_store.status(name, fold.split_idx, 0) != 'built':
                        continue
                    flow.node_objs[name] = self.predictor_store.get_obj(
                        name, fold.split_idx, 0)
                # _resolve needs the edges too, and neither the artifact nor
                # this flow's history carries a Predictor's — the definition
                # is the source of truth for them.
                flow._node_edges[name] = predictor_edges[name]
                output = flow._resolve(data, name)
                if output is None:
                    continue
                if v is not None:
                    cols = eval_expr(parse(v), output, processor=flow.node_objs[name])
                    output = output.select_columns(cols)
                outputs.append(output)
            if not outputs:
                continue
            if len(outputs) == 1:
                yield outputs[0]
            else:
                yield type(outputs[0]).concat(outputs, axis=1)

    # ------------------------------------------------------------------
    # to_inferencer
    # ------------------------------------------------------------------

    def _trainer_spec(self):
        return {
            'name': self.name,
            'pipeline_version': self.pipeline_version,
            'n_splits': self.get_n_splits(),
            'splitter': None if self.splitter is None else _obj_to_ref(type(self.splitter)),
            'splitter_params': serialize_value(self.splitter_params),
        }

    def to_inferencer(self, v=None):
        """Export trained processors to a standalone :class:`~mllabs.Inferencer`.

        All selected nodes must be in ``built`` state. The Inferencer is stamped
        with :meth:`_trainer_spec` so the deployed pickle can say where it came
        from without holding a Trainer.

        Args:
            v: Output column filter passed to the Inferencer.

        Returns:
            Inferencer: Independent inferencer ready for deployment.

        Raises:
            RuntimeError: If any selected node is not built.
        """
        from ._inferencer import Inferencer
        pipeline = self.pipeline

        all_selected = self.selected_nodes + self.predictor_names()
        for name in all_selected:
            if self.get_status(name) != 'built':
                raise RuntimeError(f"Node '{name}' is not built. Run train() first.")

        node_objs = {}
        for name in all_selected:
            store = self._store_for(name)
            node_objs[name] = [
                store.get_obj(name, fold.split_idx, 0) for fold in self.train_folds
            ]

        node_specs = {n: pipeline.get_node_spec(n) for n in self.selected_nodes}
        node_specs.update(self.predictor_specs())
        return Inferencer(node_specs, list(self.selected_nodes), self.predictor_names(),
                          self.get_n_splits(), node_objs, v=v,
                          trainer_spec=self._trainer_spec())

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self):
        """Persist meta and splits. Artifacts and Predictors save themselves."""
        if self.splitter is None:
            split_indices = None
        else:
            split_indices = [
                (fold.train_data_flows[0].data_source.train_idx,
                 fold.train_data_flows[0].data_source.valid_idx)
                for fold in self.train_folds
            ]
        self._store.save({
            'name': self.name,
            'pipeline_version': self.pipeline_version,
        })
        self._store.save_splits(self.name, {
            'splitter': self.splitter,
            'splitter_params': self.splitter_params,
            'split_indices': split_indices,
        })
