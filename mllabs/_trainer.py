import pickle as pkl
from pathlib import Path
import numpy as np

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._predictor_store import PredictorStore
from ._trial import Trial
from ._edge_dsl import parse, eval_expr, referenced_nodes
from ._logger import resolve_logger
from ._run_common import resolve_common_status, require_built_pipeline


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

    No ``Project`` dependency — like :class:`~mllabs.Experimenter`, this
    class only sees the narrow pieces it needs (``path``, ``cache``, an
    already-loaded ``pipeline``), each handed in explicitly.
    ``Project.trainer()``/``load_trainer()`` is the usual caller: it
    resolves ``(pipeline_name, pipeline_version)`` into a loaded Pipeline via
    its own ``load_pipeline`` and supplies its own ``cache`` — but nothing
    stops constructing this directly, standalone.

    Uses ``self.pipeline`` — set via the constructor or :meth:`set_pipeline`.
    Trains the Predictors supplied via :meth:`set_predictors` plus the
    Pipeline nodes they depend on; the node selection is recomputed whenever
    either is set.

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
        pipeline (Pipeline): The adopted Pipeline. Only the
            ``(pipeline_name, pipeline_version)`` pointer is persisted — the
            Pipeline itself lives once, wherever the caller keeps its versions.
        selected_nodes (list[str]): Pipeline nodes included in training.
        predictors (list[Predictor]): Predictors to train (set via
            :meth:`set_predictors`).
        node_store (NodeStore): Artifacts + history for Pipeline nodes.
        predictor_store (NodeStore): Artifacts + history for Predictors.
        predictor_defs (PredictorStore): The Predictor definitions themselves.
        train_folds (list[TrainFold]): Per-split data flows and artifact stores.
    """

    def __init__(self, path, name, data, splitter=None, splitter_params=None,
                 aug_data=None, cache=None, pipeline=None, pipeline_name='pipeline'):
        self.name = name
        self.data = data
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.splitter = splitter
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.cache = cache
        self.node_store = NodeStore(self.path)
        self.predictor_path = self.path / '__predictors'
        self.predictor_store = NodeStore(self.predictor_path)
        self.predictor_defs = PredictorStore(self.predictor_path)
        self.aug_data = wrap(aug_data) if aug_data is not None else None

        self.selected_nodes = []
        self.predictors = []
        self.pipeline = None
        self.pipeline_name = pipeline_name
        self.pipeline_version = None

        split_indices = self._make_splits()
        self.train_folds = self._make_train_folds(split_indices)
        if pipeline is not None:
            self.set_pipeline(pipeline, pipeline_name)
        else:
            self.save()

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

    def set_pipeline(self, pipeline, pipeline_name=None):
        """Adopt an already-loaded Pipeline.

        Takes the Pipeline object directly rather than a version number —
        this class has no way to load one by name/version itself (see the
        class docstring); ``Project.trainer()``/``load_trainer()`` resolve
        that before calling this. ``self.pipeline_version`` is read straight
        off *pipeline* (its ``.version``), never tracked separately.

        Which nodes are actually selected depends on the Predictors — call
        :meth:`set_predictors` to supply them.

        Diffs the two Pipelines (:meth:`Pipeline.diff_from`) and resets the
        node artifacts the change invalidated via :meth:`reset_nodes`, which
        also cascades into any selected Predictor that reads a reset node —
        unlike ``Experimenter.set_pipeline``, a Trainer has no notion of a
        "historical" run to preserve, so a Predictor trained against a
        since-changed node is simply stale.

        Args:
            pipeline (Pipeline): Already-built, already-loaded Pipeline.
            pipeline_name (str, optional): Name to record this Pipeline
                under in this Trainer's own persisted meta.
        """
        require_built_pipeline(pipeline)
        pipeline.check_data_compatibility(self.data)
        if pipeline_name is not None:
            self.pipeline_name = pipeline_name
        if self.pipeline is not None:
            stale = pipeline.diff_from(self.pipeline)
            if stale:
                self.reset_nodes(sorted(stale))
        self.pipeline = pipeline
        self.pipeline_version = pipeline.version
        self._recompute_selection()
        self.save()
        return pipeline

    def set_predictors(self, predictors):
        """Select the Predictors to train, plus the nodes they depend on.

        The selection is persisted (``predictor_defs``), so a reloaded
        Trainer keeps it without this having to be called again.

        Args:
            predictors (list[Predictor]): Predictors to train. A
                :class:`~mllabs.Trial` is rejected — promote it explicitly
                with ``Predictor.from_trial(trial)`` so the provenance it
                came from is recorded rather than guessed.
        """
        predictors = list(predictors)
        for p in predictors:
            if isinstance(p, Trial):
                raise TypeError(
                    f"set_predictors() got a Trial ({p.name!r}); promote it with "
                    f"Predictor.from_trial(trial, experimenter=...) first"
                )
        self.predictors = predictors
        self.predictor_defs.replace_all(self.predictors)
        self._recompute_selection()
        self.reset_nodes(self.selected_nodes + self.predictor_names())
        self.save()

    def predictor_names(self):
        return [p.name for p in self.predictors]

    def predictor_specs(self):
        """``{name: ProcessorSpec}`` for the selected Predictors."""
        return {p.name: p.get_spec() for p in self.predictors}

    def _recompute_selection(self):
        """Pipeline nodes needed by the selected Predictors, topologically ordered.

        With no Predictors set yet, every node is selected — training the
        whole preprocessing graph is still meaningful on its own.
        """
        if self.pipeline is None:
            self.selected_nodes = []
            return
        if not self.predictors:
            self.selected_nodes = list(self.pipeline.topo_order())
            return
        needed = set()
        for predictor in self.predictors:
            for dsl_string in predictor.edges.values():
                for name in referenced_nodes(dsl_string):
                    if name is not None and name not in needed:
                        needed.add(name)
                        self._collect_upstream(self.pipeline, name, needed)
        self.selected_nodes = [n for n in self.pipeline.topo_order() if n in needed]

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

    def _make_predictor_jobs(self, spec_map):
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
        for predictor in self.predictors:
            spec = spec_map[predictor.name]
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

    def train(self, n_jobs=1, gpu_id_list=None, logger=None):
        """Train all unbuilt selected nodes across all splits.

        Pipeline nodes are trained first (topological order), then
        Predictors. Node staleness is settled when a Pipeline is adopted
        (:meth:`set_pipeline`); Predictor staleness is checked per job.

        Args:
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, NodeInfoTracker

        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)
        spec_map = self.predictor_specs()
        # Node staleness is settled when a Pipeline is adopted
        # (set_pipeline); Predictor staleness is checked per job below.
        node_jobs = self._make_node_jobs(pipeline, self.selected_nodes, gpu_id_list)
        predictor_jobs = self._make_predictor_jobs(spec_map)

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
                        log_dir=self.path / '__worker_logs')
                else:
                    node_errors = _execute_single(
                        node_jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=node_tracker)
                error_nodes.update(n for _, _, n in node_errors)

            if predictor_jobs:
                # No Collectors here (a Trainer isn't an Experimenter) — pass
                # [] rather than the default None so the returned error keys
                # match the (outer_idx, name) shape expected below.
                if n_jobs > 1:
                    predictor_errors = _execute_multi(
                        predictor_jobs, n_jobs, self.predictor_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=predictor_tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    predictor_errors = _execute_single(
                        predictor_jobs, self.predictor_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=predictor_tracker)
                error_nodes.update(n for _, n in predictor_errors)
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
            flow.load()
            outputs = []
            for name in self.predictor_names():
                if name not in flow.node_objs:
                    # Predictor models live in their own store, so flow.load()
                    # never pulls them in (which is what keeps them out of
                    # memory) — fetch on demand.
                    if self.predictor_store.status(name, fold.split_idx, 0) != 'built':
                        continue
                    flow.node_objs[name] = self.predictor_store.get_objs(
                        name, fold.split_idx, 0)
                # _resolve needs the edges too, and neither the artifact nor
                # this flow's history carries a Predictor's — the definition
                # is the source of truth for them.
                flow._node_edges[name] = predictor_edges[name]
                output = flow._resolve(data, name)
                if output is None:
                    continue
                if v is not None:
                    obj = flow.node_objs[name][0]
                    cols = eval_expr(parse(v), output, processor=obj)
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
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        if self.splitter is None:
            split_indices = None
        else:
            split_indices = [
                (fold.train_data_flows[0].data_source.train_idx,
                 fold.train_data_flows[0].data_source.valid_idx)
                for fold in self.train_folds
            ]
        save_data = {
            'name': self.name,
            'splitter': self.splitter,
            'splitter_params': self.splitter_params,
            'split_indices': split_indices,
            'pipeline_name': self.pipeline_name,
            'pipeline_version': self.pipeline_version,
        }
        with open(self.path / '__trainer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

    @staticmethod
    def _read_save_data(path):
        """The raw ``__trainer.pkl`` dict for the Trainer at *path*.

        Split out from :meth:`load` so a caller (``Project.load_trainer``)
        can read ``pipeline_name``/``pipeline_version`` to resolve a Pipeline
        object *before* calling :meth:`load`, without reading the file twice.
        """
        with open(Path(path) / '__trainer.pkl', 'rb') as f:
            return pkl.load(f)

    @classmethod
    def load(cls, path, data, save_data=None, cache=None, pipeline=None, aug_data=None):
        """Reopen a saved Trainer from *path*.

        Args:
            path: This Trainer's base directory.
            data: Dataset to attach.
            save_data (dict, optional): Already-read ``__trainer.pkl``
                contents — pass this if the caller already read it (e.g. to
                resolve *pipeline*) to avoid reading the file twice. ``None``
                reads it here via :meth:`_read_save_data`.
            cache (DataCache, optional): Shared LRU cache.
            pipeline (Pipeline, optional): Already-loaded Pipeline matching
                this Trainer's saved ``(pipeline_name, pipeline_version)`` —
                this class has no way to load one itself (see the class
                docstring). ``None`` leaves the Trainer without a pipeline
                (matching a Trainer that was saved before one was ever set).
        """
        path = Path(path)
        if save_data is None:
            save_data = cls._read_save_data(path)

        trainer = object.__new__(cls)
        trainer.name = save_data['name']
        trainer.data = data
        trainer.path = path
        trainer.splitter = save_data['splitter']
        trainer.splitter_params = save_data['splitter_params']
        trainer.cache = cache
        trainer.node_store = NodeStore(path)
        trainer.predictor_path = path / '__predictors'
        trainer.predictor_store = NodeStore(trainer.predictor_path)
        trainer.predictor_defs = PredictorStore(trainer.predictor_path)
        trainer.aug_data = wrap(aug_data) if aug_data is not None else None
        trainer.pipeline = None
        trainer.pipeline_name = save_data.get('pipeline_name', 'pipeline')
        trainer.pipeline_version = None
        trainer.selected_nodes = []
        trainer.predictors = trainer.predictor_defs.list_predictors()

        split_indices = save_data['split_indices']
        trainer.train_folds = trainer._make_train_folds(split_indices)

        if pipeline is not None:
            # Recomputes the node selection off the restored Predictors —
            # set_predictors() does not need calling again.
            trainer.set_pipeline(pipeline)

        return trainer
