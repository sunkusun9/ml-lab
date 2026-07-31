import os
import pickle as pkl
import numpy as np
from pathlib import Path

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._edge_dsl import parse, eval_expr, referenced_nodes
from ._logger import resolve_logger
from ._run_common import resolve_common_status, find_stale_nodes, require_built_pipeline


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
        self.artifact_stores = [NodeStore(path=fold_path)]

    def set_data(self, data, cache=None, aug_data=None):
        self.train_data_flows[0].data_source.set_data(data, aug_data)
        if cache is not None:
            self.train_data_flows[0].cache = cache

    def get_test_data(self, edges, inner_idx=0):
        return None


class Trainer:
    """Runs cross-validation training on a subset of Pipeline nodes.

    Uses ``self.pipeline`` — set via the constructor or :meth:`set_pipeline`.
    Selected nodes (Head nodes whose ``tag`` intersects ``self.tags``, all
    heads if ``tags`` is empty, plus their upstream Stage dependencies) are
    recomputed whenever the pipeline is set.

    Attributes:
        name (str): Trainer name.
        tags (list[str]): Restricts training to Head nodes whose ``tag``
            intersects it (all heads if empty).
        pipeline (Pipeline): Pipeline set via the constructor or
            :meth:`set_pipeline`, persisted to ``{path}/pipeline.pkl``.
        selected_stages (list[str]): Stage nodes included in training.
        trials (list[Trial]): Trials to train (set via :meth:`set_experiment`).
        train_folds (list[TrainFold]): Per-split data flows and artifact stores.
    """

    def __init__(self, name, data, path, splitter, splitter_params, cache, tags=None, aug_data=None, pipeline=None):
        self.name = name
        self.data = data
        self.path = Path(path)
        self.splitter = splitter
        self.splitter_params = splitter_params
        self.cache = cache
        self.tags = list(tags) if tags is not None else []
        self.aug_data = wrap(aug_data) if aug_data is not None else None

        self.selected_stages = []
        self.trials = []
        self.pipeline = None

        split_indices = self._make_splits()
        self.train_folds = self._make_train_folds(split_indices)
        if pipeline is not None:
            self.set_pipeline(pipeline)
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

    def set_pipeline(self, pipeline):
        """Set (or replace) the Stage graph this trainer trains against.

        Takes a built :class:`Pipeline`, not a :class:`PipelineBuilder` — see
        :meth:`Experimenter.set_pipeline`. Which Stages are actually selected
        depends on the Trials: call :meth:`set_experiment` to supply them.

        Args:
            pipeline (Pipeline): Built pipeline defining the Stage graph.
        """
        require_built_pipeline(pipeline)
        pipeline.check_data_compatibility(self.data)
        was_set = self.pipeline is not None
        self.pipeline = pipeline
        self._recompute_selection()
        if was_set:
            self._reset_stale(self.selected_stages + self.trial_names())
        self._save_pipeline()
        self.save()

    def set_experiment(self, experiment, trials=None):
        """Select the Trials to train, plus the Stages they depend on.

        Args:
            experiment (BaseExperiment): Source of Trials.
            trials: Trial-name filter — ``None`` (all), ``list``, or regex ``str``.
        """
        from ._experimenter import _name_matches
        self.trials = [t for t in experiment.get_trials() if _name_matches(t.name, trials)]
        self._recompute_selection()
        self._reset_stale(self.selected_stages + self.trial_names())
        self.save()

    def trial_names(self):
        return [t.name for t in self.trials]

    def trial_attrs(self):
        """``{name: attrs}`` for the selected Trials, with ``serial`` set to
        the Trial id so staleness works the same way it does for Stages."""
        pipeline = self._require_pipeline()
        return {
            t.name: {**t.get_attrs(), 'serial': t.trial_id(pipeline)}
            for t in self.trials
        }

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

    def _reset_stale(self, names):
        current = {n: self.pipeline.nodes[n].serial for n in self.selected_stages}
        current.update({n: a['serial'] for n, a in self.trial_attrs().items()})
        stale = find_stale_nodes(
            current, [n for n in names if n in current],
            lambda name: (fold.artifact_stores[0] for fold in self.train_folds)
        )
        if stale:
            self.reset_nodes(stale)

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline(pipeline) first.")
        return self.pipeline

    def _save_pipeline(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / 'pipeline.pkl', 'wb') as f:
            pkl.dump(self.pipeline, f)

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
            fold.artifact_stores[0].status(node_name)
            for fold in self.train_folds
        )

    def get_node_error(self, node_name):
        """Return error dict for a node in error state, or None."""
        for fold in self.train_folds:
            info = fold.artifact_stores[0].get_info(node_name)
            if info is not None and info.get('status') == 'error':
                return info.get('error')
        return None

    def reset_nodes(self, nodes):
        pipeline = self._require_pipeline()
        selected_set = set(self.selected_stages + self.trial_names())
        affected = set(n for n in nodes if n in selected_set)

        queue = list(affected)
        while queue:
            n = queue.pop(0)
            node = pipeline.get_node(n)
            for downstream in node.output_edges:
                if downstream in selected_set and downstream not in affected:
                    affected.add(downstream)
                    queue.append(downstream)

        for name in affected:
            for fold in self.train_folds:
                fold.train_data_flows[0].reset_node(name)
                fold.artifact_stores[0].reset_node(name)

        if self.cache is not None:
            self.cache.clear_nodes(affected)

        self.save()

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def _reset_serial_stale_nodes(self, current_serials, logger):
        stale = find_stale_nodes(
            current_serials, list(current_serials),
            lambda name: (fold.artifact_stores[0] for fold in self.train_folds)
        )
        if stale:
            self.reset_nodes(stale)
            logger.info(f"Serial mismatch: reset {len(stale)} node(s): {sorted(stale)}")

    def train(self, n_jobs=1, gpu_id_list=None, logger=None):
        """Train all unbuilt selected nodes across all splits.

        Stages are trained first (topological order), then Head nodes. Uses
        ``self.pipeline`` — compared against the artifacts already on disk,
        nodes whose ``serial`` no longer matches are reset and retrained
        automatically.

        Args:
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _build_flow_single, _build_flow_multi, _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker

        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        self._save_pipeline()

        pipeline.check_data_compatibility(self.data)
        attrs_map = self.trial_attrs()
        self._reset_serial_stale_nodes(
            {**{n: pipeline.nodes[n].serial for n in self.selected_stages},
             **{n: a['serial'] for n, a in attrs_map.items()}}, logger)

        target_stages = [
            n for n in self.selected_stages
            if self.get_status(n) not in ['built', 'finalized']
        ]
        target_heads = [
            n for n in attrs_map
            if self.get_status(n) not in ['built', 'finalized']
        ]

        if not target_stages and not target_heads:
            logger.info("No nodes to train")
            return

        total = len(self.train_folds) * (len(target_stages) + len(target_heads))
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

            if target_heads:
                if n_jobs > 1:
                    head_errors = _experiment_multi(
                        self.train_folds, attrs_map, target_heads, n_jobs,
                        gpu_id_list=gpu_id_list, tracker=tracker,
                        log_dir=self.path / '__worker_logs')
                else:
                    head_errors = _experiment_single(
                        self.train_folds, attrs_map, target_heads,
                        gpu_id_list=gpu_id_list, tracker=tracker)
                error_nodes.update(n for _, n in head_errors)
        finally:
            tracker.close()

        target_all = target_stages + target_heads
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
                    continue
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
                objs.append(fold.artifact_stores[0].get_obj(name))
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
            'tags': self.tags,
            'split_indices': split_indices,
        }
        with open(self.path / '__trainer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

    @classmethod
    def _load(cls, path, data, cache, aug_data=None):
        path = Path(path)
        with open(path / '__trainer.pkl', 'rb') as f:
            save_data = pkl.load(f)

        trainer = object.__new__(cls)
        trainer.name = save_data['name']
        trainer.data = data
        trainer.path = path
        trainer.splitter = save_data['splitter']
        trainer.splitter_params = save_data['splitter_params']
        trainer.cache = cache
        trainer.tags = save_data.get('tags', [])
        trainer.aug_data = wrap(aug_data) if aug_data is not None else None
        trainer.pipeline = None
        trainer.selected_stages = []
        trainer.trials = []

        split_indices = save_data['split_indices']
        trainer.train_folds = trainer._make_train_folds(split_indices)

        pipeline_path = path / 'pipeline.pkl'
        if pipeline_path.exists():
            with open(pipeline_path, 'rb') as f:
                trainer.pipeline = pkl.load(f)
            # Trials are not persisted — re-supply them with set_experiment().
            trainer._recompute_selection()

        return trainer
