import pickle as pkl
import numpy as np
from pathlib import Path

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._edge_dsl import parse, eval_expr, referenced_nodes
from ._logger import resolve_logger
from ._run_common import resolve_common_status, require_built_pipeline


class TrainFold:
    """Single split data flow and artifact store for Trainer.

    Analogous to OuterFold for Experimenter. Holds exactly one TrainDataFlow
    and one NodeStore at the same fold path.

    get_test_data() returns None (Trainer has no separate test set).
    """

    def __init__(self, split_idx, base_path, data, train_idx, valid_idx=None, cache=None, aug_data=None):
        self.split_idx = split_idx
        fold_path = Path(base_path) / str(split_idx)
        provider = DataWrapperProvider(data, train_idx, valid_idx=valid_idx, aug_data=aug_data)
        self.train_data_flows = [
            TrainDataFlow(
                path=fold_path,
                data_source=provider,
                cache=cache,
                outer_idx=-(split_idx + 1),  # 음수로 Experimenter cache와 충돌 방지
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

    Uses ``self.pipeline`` — set via the constructor or :meth:`set_pipeline`.
    Trains the Trials supplied via :meth:`set_trials` plus the Stages they
    depend on; the Stage selection is recomputed whenever either is set.

    Attributes:
        name (str): Trainer name.
        pipeline (Pipeline): The loaded Pipeline. Only the
            ``(pipeline_name, pipeline_version)`` pointer is persisted — the
            Pipeline itself lives once, under the Project.
        selected_stages (list[str]): Stage nodes included in training.
        trials (list[Trial]): Trials to train (set via :meth:`set_trials`).
        train_folds (list[TrainFold]): Per-split data flows and artifact stores.
    """

    def __init__(self, project, name, data, splitter=None, splitter_params=None,
                 aug_data=None, pipeline_name='pipeline', pipeline_version=None):
        self.project = project
        self.name = name
        self.data = data
        self.path = project.trainer_path(name)
        self.splitter = splitter
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.cache = project.cache
        self.aug_data = wrap(aug_data) if aug_data is not None else None

        self.selected_stages = []
        self.trials = []
        self.pipeline = None
        self.pipeline_name = pipeline_name
        self.pipeline_version = None

        split_indices = self._make_splits()
        self.train_folds = self._make_train_folds(split_indices)
        if pipeline_version is not None:
            self.set_pipeline_version(pipeline_version)
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
                TrainFold(0, self.path, self.data, full_idx, valid_idx=None,
                          cache=self.cache, aug_data=self.aug_data)
            ]
        return [
            TrainFold(i, self.path, self.data, train_idx, valid_idx=valid_idx,
                      cache=self.cache, aug_data=self.aug_data)
            for i, (train_idx, valid_idx) in enumerate(split_indices)
        ]

    def get_n_splits(self):
        return len(self.train_folds)

    # ------------------------------------------------------------------
    # node selection
    # ------------------------------------------------------------------

    def set_pipeline_version(self, version, pipeline_name=None):
        """Point this Trainer at a Pipeline version from its Project.

        Which Stages are actually selected depends on the Trials — call
        :meth:`set_trials` to supply them.

        Args:
            version (int): Pipeline version number.
            pipeline_name (str, optional): Pipeline name within the Project.
        """
        if pipeline_name is not None:
            self.pipeline_name = pipeline_name
        pipeline = self.project.load_pipeline(self.pipeline_name, version)
        require_built_pipeline(pipeline)
        pipeline.check_data_compatibility(self.data)
        if self.pipeline is not None:
            self._drop_stale(pipeline.diff_from(self.pipeline))
        self.pipeline = pipeline
        self.pipeline_version = version
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
        self._reset_stale(self.selected_stages + self.trial_names())
        self.save()

    def trial_names(self):
        return [t.name for t in self.trials]

    def trial_attrs(self):
        """``{name: resolved attrs}`` for the selected Trials."""
        return {t.name: t.get_attrs() for t in self.trials}

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

    def _drop_stale(self, stale_stages):
        """Remove the artifacts a set of stale Stages invalidates.

        Same rule as :meth:`Experimenter._drop_stale`: Trials are not in the
        Pipeline, so they are found through the ``edges`` each artifact recorded.
        """
        if not stale_stages:
            return
        doomed = set(stale_stages)
        for fold in self.train_folds:
            flow = fold.train_data_flows[0]
            for name in flow.list_nodes():
                if name in doomed:
                    continue
                info = flow.get_info(name)
                if info is None or info.get('role') != 'head':
                    continue
                for dsl_string in (info.get('edges') or {}).values():
                    if doomed & referenced_nodes(dsl_string):
                        doomed.add(name)
                        break
        self.reset_nodes(sorted(doomed))

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline_version(version) first.")
        return self.pipeline

    def _collect_upstream(self, pipeline, node_name, selected):
        node_attrs = pipeline.get_node_attrs(node_name)
        for dsl_string in node_attrs['edges'].values():
            for source_node in referenced_nodes(dsl_string):
                if source_node is not None and source_node not in selected:
                    selected.add(source_node)
                    self._collect_upstream(pipeline, source_node, selected)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def get_status(self, node_name):
        """Return the disk status of a node across all folds.

        Returns ``'built'``, ``'finalized'``, ``'error'``, ``None`` (init),
        or ``'inconsistent'`` if folds differ.
        """
        return resolve_common_status(
            fold.train_data_flows[0].status(node_name)
            for fold in self.train_folds
        )

    def get_node_error(self, node_name):
        """Return error dict for a node in error state, or None."""
        for fold in self.train_folds:
            info = fold.train_data_flows[0].get_info(node_name)
            if info is not None and info.get('status') == 'error':
                return info.get('error')
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

    def _make_trial_jobs(self, attrs_map):
        """One TrialJob per (Trial, split) still needing training.

        A Trainer has a single flow per split, so the fold coordinate is
        ``(split_idx, 0)`` — the same shape the Experimenter uses.
        """
        from ._executor import TrialJob
        from ._pipeline import _definition_of
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        jobs = []
        for trial in self.trials:
            attrs = attrs_map[trial.name]
            if trial.name not in gpu_cache:
                adapter = resolve_node_adapter(attrs.get('processor'), attrs.get('adapter'))
                gpu_cache[trial.name] = adapter.get_gpu_usage(attrs.get('params')) != GPU_NO
            for split_idx, fold in enumerate(self.train_folds):
                flow = fold.train_data_flows[0]
                info = flow.get_info(trial.name)
                if info is not None and info.get('definition') != _definition_of(attrs):
                    flow.reset_node(trial.name)
                    info = None
                if info is not None and info.get('status') in ('built', 'finalized'):
                    continue
                jobs.append(TrialJob(trial, attrs, (split_idx, 0), flow,
                                     need_gpu=gpu_cache[trial.name]))
        return jobs

    def train(self, n_jobs=1, gpu_id_list=None, logger=None):
        """Train all unbuilt selected nodes across all splits.

        Stages are trained first (topological order), then Trials. Stage
        staleness is settled when a Pipeline version is adopted
        (:meth:`set_pipeline_version`); Trial staleness is checked per job.

        Args:
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _build_flow_single, _build_flow_multi, _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker

        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)
        attrs_map = self.trial_attrs()
        # Stage staleness is settled when a version is adopted
        # (set_pipeline_version); Trial staleness is checked per job below.
        target_stages = [
            n for n in self.selected_stages
            if self.get_status(n) not in ['built', 'finalized']
        ]
        trial_jobs = self._make_trial_jobs(attrs_map)

        if not target_stages and not trial_jobs:
            logger.info("No nodes to train")
            return

        total = len(self.train_folds) * len(target_stages) + len(trial_jobs)
        n_jobs = min(n_jobs, total)
        tracker = LoggerExecuteTracker(total, n_jobs, logger)
        error_nodes = set()
        try:
            if target_stages:
                if n_jobs > 1:
                    stage_errors = _build_flow_multi(
                        self.train_folds, pipeline, target_stages, n_jobs,
                        gpu_id_list=gpu_id_list, tracker=tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    stage_errors = _build_flow_single(
                        self.train_folds, pipeline, target_stages,
                        gpu_id_list=gpu_id_list, tracker=tracker)
                error_nodes.update(n for _, _, n in stage_errors)

            if trial_jobs:
                if n_jobs > 1:
                    head_errors = _experiment_multi(
                        trial_jobs, n_jobs, gpu_id_list=gpu_id_list, tracker=tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    head_errors = _experiment_single(
                        trial_jobs, gpu_id_list=gpu_id_list, tracker=tracker)
                error_nodes.update(n for _, n in head_errors)
        finally:
            tracker.close()

        target_all = target_stages + sorted({j.name for j in trial_jobs})
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

        node_attrs = {n: pipeline.get_node_attrs(n) for n in self.selected_stages}
        node_attrs.update(self.trial_attrs())
        return Inferencer(node_attrs, list(self.selected_stages), self.trial_names(),
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

    @classmethod
    def load(cls, project, name, data, aug_data=None):
        """Reopen a saved Trainer by name, restoring its Pipeline version."""
        path = project.trainer_path(name)
        with open(path / '__trainer.pkl', 'rb') as f:
            save_data = pkl.load(f)

        trainer = object.__new__(cls)
        trainer.project = project
        trainer.name = save_data['name']
        trainer.data = data
        trainer.path = path
        trainer.splitter = save_data['splitter']
        trainer.splitter_params = save_data['splitter_params']
        trainer.cache = project.cache
        trainer.aug_data = wrap(aug_data) if aug_data is not None else None
        trainer.pipeline = None
        trainer.pipeline_name = save_data.get('pipeline_name', 'pipeline')
        trainer.pipeline_version = None
        trainer.selected_stages = []
        trainer.trials = []

        split_indices = save_data['split_indices']
        trainer.train_folds = trainer._make_train_folds(split_indices)

        version = save_data.get('pipeline_version')
        if version is not None:
            # Trials are not persisted — re-supply them with set_trials().
            trainer.set_pipeline_version(version)

        return trainer
