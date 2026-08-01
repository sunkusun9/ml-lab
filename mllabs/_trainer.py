import pickle as pkl
from pathlib import Path
import numpy as np

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
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
    Trains the Trials supplied via :meth:`set_trials` plus the Stages they
    depend on; the Stage selection is recomputed whenever either is set.

    Attributes:
        name (str): Trainer name.
        pipeline (Pipeline): The adopted Pipeline. Only the
            ``(pipeline_name, pipeline_version)`` pointer is persisted — the
            Pipeline itself lives once, wherever the caller keeps its versions.
        selected_stages (list[str]): Stage nodes included in training.
        trials (list[Trial]): Trials to train (set via :meth:`set_trials`).
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
        self.aug_data = wrap(aug_data) if aug_data is not None else None

        self.selected_stages = []
        self.trials = []
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

        Which Stages are actually selected depends on the Trials — call
        :meth:`set_trials` to supply them.

        Diffs the two Pipelines (:meth:`Pipeline.diff_from`) and resets the
        Stage artifacts the change invalidated via :meth:`reset_nodes`, which
        also cascades into any selected Trial that reads a reset Stage —
        unlike ``Experimenter.set_pipeline``, a Trainer has no notion of a
        "historical" run to preserve, so a Trial trained against a
        since-changed Stage is simply stale.

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

    def set_trials(self, trials):
        """Select the Trials to train, plus the Stages they depend on.

        Args:
            trials (list[Trial]): Trials to train.
        """
        self.trials = list(trials)
        self._recompute_selection()
        self.reset_nodes(self.selected_stages + self.trial_names())
        self.save()

    def trial_names(self):
        return [t.name for t in self.trials]

    def trial_specs(self):
        """``{name: ProcessorSpec}`` for the selected Trials."""
        return {t.name: t.get_spec() for t in self.trials}

    def _recompute_selection(self):
        """Stages needed by the selected Trials, in topological order.

        With no Trials set yet, every Stage is selected — training the whole
        preprocessing graph is still meaningful on its own.
        """
        if self.pipeline is None:
            self.selected_stages = []
            return
        if not self.trials:
            self.selected_stages = list(self.pipeline.topo_order())
            return
        needed = set()
        for trial in self.trials:
            for dsl_string in trial.edges.values():
                for name in referenced_nodes(dsl_string):
                    if name is not None and name not in needed:
                        needed.add(name)
                        self._collect_upstream(self.pipeline, name, needed)
        self.selected_stages = [n for n in self.pipeline.topo_order() if n in needed]

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

    def get_status(self, node_name):
        """Return the disk status of a node across all folds.

        Returns ``'built'``, ``None`` (init, or errored — ``NodeStore`` only
        knows whether ``obj.pkl`` exists; see :meth:`get_node_error` for
        Stage error detail), or ``'inconsistent'`` if folds differ.
        """
        return resolve_common_status(
            fold.train_data_flows[0].status(node_name)
            for fold in self.train_folds
        )

    def get_node_error(self, node_name):
        """Return error dict for a Stage node in error state, or ``None``.

        Trial errors aren't covered — a Trainer has no ``experiment_hist``
        (see ``set_pipeline``) to record them in; only Stage errors are, in
        this run's own ``NodeStore`` history.
        """
        for r in self.node_store.get_hist(node_name=node_name):
            if r['status'] == 'error':
                return (r['info'] or {}).get('error')
        return None

    def reset_nodes(self, nodes):
        pipeline = self._require_pipeline()
        selected_set = set(self.selected_stages + self.trial_names())
        affected = set(n for n in nodes if n in selected_set)

        # Only Stages have downstream nodes to cascade into; a Trial is a leaf
        # and is not part of the pipeline graph at all.
        queue = [n for n in affected if n in pipeline.nodes]
        while queue:
            n = queue.pop(0)
            for downstream in pipeline.nodes[n].output_edges:
                if downstream in selected_set and downstream not in affected:
                    affected.add(downstream)
                    if downstream in pipeline.nodes:
                        queue.append(downstream)

        # A Trial reading a reset Stage must be retrained too.
        stage_reset = affected & set(self.selected_stages)
        if stage_reset:
            for trial in self.trials:
                if trial.name in affected:
                    continue
                if trial.stage_names() & stage_reset:
                    affected.add(trial.name)

        for name in affected:
            for fold in self.train_folds:
                fold.train_data_flows[0].reset_node(name)

        if self.cache is not None:
            self.cache.clear_nodes(affected)

        self.save()

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def _make_trial_jobs(self, spec_map):
        """One Job per (Trial, split) still needing training.

        A Trainer has a single flow per split, so the fold coordinate is
        ``(split_idx, 0)`` — the same shape the Experimenter uses.

        A split is skipped only if its artifact is already built — like
        ``Experimenter._make_jobs``, redefining a Trial no longer forces a
        rerun of splits already built (``NodeStore`` no longer carries a
        definition to compare against).
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        jobs = []
        for trial in self.trials:
            spec = spec_map[trial.name]
            if trial.name not in gpu_cache:
                adapter = resolve_node_adapter(spec.processor, spec.adapter)
                gpu_cache[trial.name] = adapter.get_gpu_usage(spec.params) != GPU_NO
            for split_idx, fold in enumerate(self.train_folds):
                flow = fold.train_data_flows[0]
                if flow.status(trial.name) == 'built':
                    continue
                jobs.append(Job(trial.name, spec, split_idx, 0, flow,
                                need_gpu=gpu_cache[trial.name]))
        return jobs

    def _make_stage_jobs(self, pipeline, node_names, gpu_id_list):
        """Expand Stage node names into per-split Jobs.

        Same role ``_make_trial_jobs`` plays for Trials: skip decisions
        (a split already built for a given node) live here, not in the
        executor — it only orders dispatch among what's left.
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

        Stages are trained first (topological order), then Trials. Stage
        staleness is settled when a Pipeline is adopted (:meth:`set_pipeline`);
        Trial staleness is checked per job.

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
        spec_map = self.trial_specs()
        # Stage staleness is settled when a Pipeline is adopted
        # (set_pipeline); Trial staleness is checked per job below.
        stage_jobs = self._make_stage_jobs(pipeline, self.selected_stages, gpu_id_list)
        trial_jobs = self._make_trial_jobs(spec_map)

        if not stage_jobs and not trial_jobs:
            logger.info("No nodes to train")
            return

        total = len(stage_jobs) + len(trial_jobs)
        n_jobs = min(n_jobs, total)
        base_tracker = LoggerExecuteTracker(total, n_jobs, logger)
        # Trials have no experiment_hist here (a Trainer isn't an Experimenter),
        # so only Stages get history recording — trial_jobs use base_tracker
        # directly.
        stage_tracker = NodeInfoTracker(base_tracker, self.node_store, self.pipeline_version)
        error_nodes = set()
        try:
            if stage_jobs:
                if n_jobs > 1:
                    stage_errors = _execute_multi(
                        stage_jobs, n_jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=stage_tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    stage_errors = _execute_single(
                        stage_jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=stage_tracker)
                error_nodes.update(n for _, _, n in stage_errors)

            if trial_jobs:
                # No Collectors here (a Trainer isn't an Experimenter) — pass
                # [] rather than the default None so the returned error keys
                # match the (outer_idx, name) shape expected below.
                if n_jobs > 1:
                    head_errors = _execute_multi(
                        trial_jobs, n_jobs, self.node_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=base_tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    head_errors = _execute_single(
                        trial_jobs, self.node_store, gpu_id_list=gpu_id_list,
                        collectors=[], tracker=base_tracker)
                error_nodes.update(n for _, n in head_errors)
        finally:
            base_tracker.close()

        target_all = sorted({j.name for j in stage_jobs}) + sorted({j.name for j in trial_jobs})
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
            v: Output column filter applied to Head outputs.

        Yields:
            DataFrame: Concatenated Head outputs for each split.
        """
        data = wrap(data)
        for fold in self.train_folds:
            flow = fold.train_data_flows[0]
            flow.load()
            head_outputs = []
            for name in self.trial_names():
                if name not in flow.node_objs:
                    # Trial models are not loaded with the flow (they are leaves
                    # and would only bloat it) — pull this one in on demand.
                    if flow.status(name) != 'built':
                        continue
                    flow.load_objs(name)
                output = flow._resolve(data, name)
                if output is None:
                    continue
                if v is not None:
                    obj = flow.node_objs[name][0]
                    cols = eval_expr(parse(v), output, processor=obj)
                    output = output.select_columns(cols)
                head_outputs.append(output)
            if not head_outputs:
                continue
            if len(head_outputs) == 1:
                yield head_outputs[0]
            else:
                yield type(head_outputs[0]).concat(head_outputs, axis=1)

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

        all_selected = self.selected_stages + self.trial_names()
        for name in all_selected:
            if self.get_status(name) != 'built':
                raise RuntimeError(f"Node '{name}' is not built. Run train() first.")

        node_objs = {}
        for name in all_selected:
            objs = []
            for fold in self.train_folds:
                objs.append(fold.train_data_flows[0].get_obj(name))
            node_objs[name] = objs

        node_specs = {n: pipeline.get_node_spec(n) for n in self.selected_stages}
        node_specs.update(self.trial_specs())
        return Inferencer(node_specs, list(self.selected_stages), self.trial_names(),
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
        trainer.aug_data = wrap(aug_data) if aug_data is not None else None
        trainer.pipeline = None
        trainer.pipeline_name = save_data.get('pipeline_name', 'pipeline')
        trainer.pipeline_version = None
        trainer.selected_stages = []
        trainer.trials = []

        split_indices = save_data['split_indices']
        trainer.train_folds = trainer._make_train_folds(split_indices)

        if pipeline is not None:
            # Trials are not persisted — re-supply them with set_trials().
            trainer.set_pipeline(pipeline)

        return trainer
