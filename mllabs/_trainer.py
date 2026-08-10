from pathlib import Path
import numpy as np

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._predictor_store import PredictorStore
from ._trainer_store import TrainerStore
from ._trial import Trial
from ._edge_dsl import parse, eval_expr, referenced_nodes
from ._logger import resolve_logger
from ._common import (resolve_common_status, require_built_pipeline,
                      require_frozen_pipeline)


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

        self.pipeline = None

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
        trainer.pipeline = None
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
        return self.pipeline.version if self.pipeline is not None else None

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
        """Adopt an already-loaded Pipeline. It must be a frozen version.

        Takes the Pipeline object directly rather than a version number —
        this class has no way to load one by version itself (see the class
        docstring); ``Project.add_trainer()`` resolves that before calling
        this. ``self.pipeline_version`` is read straight off *pipeline* (its
        ``.version``), never tracked separately.

        The working copy is refused. What a Trainer produces is what gets
        deployed, so it has to be able to say what it was trained against, and
        a draft changes under it. An *archived* version is fine — it is frozen,
        so it answers that question as well as the published one does. What is
        ruled out is training against something still being edited, not
        training against something old.

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

        Args:
            pipeline (Pipeline): Already-built, already-published Pipeline.

        Raises:
            ValueError: If *pipeline* is the unpublished working copy.
        """
        require_built_pipeline(pipeline)
        require_frozen_pipeline(pipeline)
        pipeline.check_data_compatibility(self.data)
        if self.pipeline is not None:
            stale = pipeline.diff_from(self.pipeline)
            if stale:
                self.reset_nodes(sorted(stale))
        self.pipeline = pipeline
        self._store.save_pipeline(pipeline)
        self.save()
        return pipeline

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

    def _nodes_for(self, predictors):
        """Pipeline nodes *predictors* need, topologically ordered.

        With no Predictors at all, every node is selected — training the whole
        preprocessing graph is still meaningful on its own.
        """
        if self.pipeline is None:
            return []
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

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline(pipeline) first.")
        return self.pipeline

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

    def reset_nodes(self, nodes):
        pipeline = self._require_pipeline()
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

        # A Predictor reading a reset node must be retrained too.
        node_reset = affected & set(self.selected_nodes)
        if node_reset:
            for predictor in self.predictors:
                if predictor.name in affected:
                    continue
                if predictor.node_names() & node_reset:
                    affected.add(predictor.name)

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

        self.save()

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
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        jobs = []
        for predictor in predictors:
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
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, NodeInfoTracker

        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)

        if predictors is None:
            predictors = self.predictors
        else:
            predictors = list(predictors)
            for p in predictors:
                if isinstance(p, Trial):
                    raise TypeError(
                        f"train() got a Trial ({p.name!r}); promote it with "
                        f"Predictor.from_trial(trial, experimenter=...) first"
                    )
            self.predictor_defs.register_all(predictors)

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

    def to_inferencer(self, v=None):
        """Export trained processors to a standalone :class:`~mllabs.Inferencer`.

        All selected nodes must be in ``built`` state.

        Args:
            v: Output column filter passed to the Inferencer.

        Returns:
            Inferencer: Independent inferencer ready for deployment.

        Raises:
            RuntimeError: If any selected node is not built.
        """
        from ._inferencer import Inferencer
        pipeline = self._require_pipeline()

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
                          self.get_n_splits(), node_objs, v=v)

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
