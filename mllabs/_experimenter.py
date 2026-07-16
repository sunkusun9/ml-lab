import os
import uuid
import pickle as pkl
import shutil
import traceback
import warnings
from pathlib import Path

import pandas as pd

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._describer import desc_spec
from ._logger import resolve_logger
from ._cache import DataCache

from ._pipeline import Pipeline
from ._node_processor import resolve_columns
from ._connector import Connector
from ._run_common import resolve_common_status, find_stale_nodes, filter_node_names_by_tags
from .collector import Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector, ProcessCollector


class OuterFold:
    """One outer fold: test indices, base path, and per-inner-fold TrainDataFlows.

    Serializes test_idx, path, and TrainDataFlow list.
    DataWrapperProvider inside each TrainDataFlow persists only indices — DataWrapper is transient.

    Call set_data(data) to re-inject DataWrapper and cache after load.
    """

    def __init__(self, outer_idx, path, data, test_idx, train_idx_list, cache=None, aug_data=None):
        self.outer_idx = outer_idx
        self.path = Path(path)
        self.test_idx = test_idx
        self.data = data
        self.train_data_flows = [
            TrainDataFlow(
                path=self.path / str(j),
                data_source=DataWrapperProvider(data, train_idx, valid_idx=valid_idx, aug_data=aug_data),
                cache=cache,
                outer_idx=outer_idx,
                inner_idx=j,
            )
            for j, (train_idx, valid_idx) in enumerate(train_idx_list)
        ]
        self.artifact_stores = [
            NodeStore(path=self.path / str(j))
            for j in range(len(train_idx_list))
        ]

    def set_data(self, data, cache=None, aug_data=None):
        self.data = data
        for flow in self.train_data_flows:
            flow.data_source.set_data(data, aug_data)
            if cache is not None:
                flow.cache = cache

    def get_data(self, data, edges, inner_idx=0):
        return self.train_data_flows[inner_idx].get_data(data, edges)

    def get_test_data(self, edges, inner_idx=0):
        test_source = self.data.iloc(self.test_idx)
        return self.get_data(test_source, edges, inner_idx)



class Experimenter():
    """Executes and manages a Pipeline experiment on a single dataset.

    Splits data using *sp* (outer) and optionally *sp_v* (inner), then runs
    Stage builds and Head experiments fold-by-fold.

    Args:
        data: Input dataset (pandas DataFrame, polars DataFrame, or numpy array).
        path (str | Path): Directory for persisting experiment artifacts.
        data_names (list[str], optional): Column names override.
        sp: Outer splitter (sklearn splitter API). Default
            ``ShuffleSplit(n_splits=1, random_state=1)``.
        sp_v: Inner splitter for nested cross-validation. ``None`` disables.
        splitter_params (dict, optional): Maps splitter keyword args to column
            names in *data*, e.g. ``{'y': 'target'}``.
        title (str, optional): Human-readable experiment title.
        data_key (str, optional): Identifier verified on :meth:`load` to prevent
            data mismatch.
        cache_maxsize (int): Stage output cache size in bytes. Default 4 GB.
        tags (list[str], optional): When set, ``exp(pipeline)`` without an
            explicit ``nodes`` targets only Head nodes whose ``tag`` intersects it.

    Attributes:
        cache (DataCache): Shared LRU cache.
        collectors (dict): Registered :class:`~mllabs.collector.Collector` instances.
        status (str): ``'open'`` or ``'closed'``.

    Note:
        No Pipeline reference is held between calls. Pass ``pipeline``
        explicitly to :meth:`build`, :meth:`exp`, and other node-graph-aware
        methods each time; artifacts are reconciled against it via node
        ``serial`` comparison on every call.
    """

    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            aug_data=None, tags=None, _save=True
        ):
        self.cache_maxsize = cache_maxsize
        self.tags = list(tags) if tags is not None else []
        self.path = Path(path)
        if not os.path.exists(path):
            self.path.mkdir(parents=True, exist_ok=True)
        data_native = data
        self.data = wrap(data)
        self.aug_data = wrap(aug_data) if aug_data is not None else None
        self.title = title
        self.data_key = data_key
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}
        if data_names is None:
            data_names = self.data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))

        raw_splits = []
        for outer_train_idx, test_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = self.data.iloc(outer_train_idx)
                train_data_native = unwrap(train_data)
                inner_split_params = {'X': train_data_native}
                for k, v in self.splitter_params.items():
                    inner_split_params[k] = unwrap(train_data.select_columns(v))
                inner_folds = [
                    (outer_train_idx[train_idx], outer_train_idx[valid_idx])
                    for train_idx, valid_idx in sp_v.split(**inner_split_params)
                ]
            else:
                inner_folds = [(outer_train_idx, None)]
            raw_splits.append((test_idx, inner_folds))

        self.cache = DataCache(maxsize=cache_maxsize)

        self.outer_folds = [
            OuterFold(
                outer_idx=i,
                path=self.path / '__folds' / str(i),
                data=self.data,
                test_idx=test_idx,
                train_idx_list=inner_folds,
                cache=self.cache,
                aug_data=self.aug_data,
            )
            for i, (test_idx, inner_folds) in enumerate(raw_splits)
        ]
        self.collectors = {}
        self.status = "open"
        if _save:
            self._save()

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def open(self):
        """Experimenter를 open 상태로 변경"""
        self.status = "open"
        self._save()

    def close(self):
        """Experimenter를 close 상태로 변경"""
        self.status = "close"
        self._save()

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3, aug_data=None):

        if os.path.exists(path):
            raise RuntimeError(f"Exists: {path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params,
            title=title, data_key=data_key, cache_maxsize=cache_maxsize,aug_data=aug_data
        )

    def get_n_splits(self):
        return len(self.outer_folds)

    def get_n_splits_inner(self):
        return len(self.outer_folds[0].train_data_flows)

    def get_collector(self, name):
        return self.collectors.get(name)

    def remove_collector(self, name):
        if name in self.collectors:
            collector_path = self.path / '__collector' / name
            if collector_path.exists():
                shutil.rmtree(collector_path)
            del self.collectors[name]
            self._save()

    def add_collector(self, collector, exist = 'skip'):
        """Register a Collector.

        Does not collect from already-built Head nodes — call
        :meth:`collect` explicitly (with a ``pipeline``) afterward if needed.

        Args:
            collector (Collector): Collector instance to register.
            exist (str): ``'skip'`` (default) returns existing if already registered;
                ``'error'`` raises; ``'replace'`` removes the existing collector and
                registers the new one from scratch.

        Returns:
            Collector: The registered collector.
        """
        if collector.name in self.collectors:
            if exist == 'skip':
                return self.collectors[collector.name]
            elif exist == 'error':
                raise RuntimeError("")
            elif exist == 'replace':
                self.remove_collector(collector.name)

        self._check_open()
        collector.path = self.path / '__collector' / collector.name
        collector.on_attach(self)
        collector._setup(
            len(self.outer_folds), len(self.outer_folds[0].train_data_flows)
        )
        collector.save()
        self.collectors[collector.name] = collector
        self._save()
        return collector

    def get_collect_status(self, pipeline, collector, nodes=None):
        if isinstance(collector, str):
            collector = self.collectors[collector]
        all_node_names = pipeline.get_node_names(nodes)
        head_nodes = [
            n for n in all_node_names
            if n is not None and pipeline.get_node_attrs(n).get('role') == 'head'
            and collector.connector.match(pipeline.get_node_attrs(n))
        ]
        result = {}
        for node in head_nodes:
            if collector.has_node(node):
                result[node] = 'collected'
            else:
                node_status = self.get_status(node)
                if node_status == 'finalized':
                    result[node] = 'finalized'
                elif node_status == 'error':
                    result[node] = 'error'
                else:
                    result[node] = 'not_collected'
        return result


    def _all_stores(self):
        """Flatten every store (both Stage and Head, all folds) into one list."""
        return [
            store
            for outer_fold in self.outer_folds
            for store in outer_fold.train_data_flows + outer_fold.artifact_stores
        ]

    def get_status(self, node_name):
        """Return the disk status of a node across all folds.

        Checks both Stage (``train_data_flows``) and Head (``artifact_stores``)
        storage — a node only ever has a real status in the store type that
        matches its role, so ``None`` entries from the other store type are
        ignored. Returns the common status if all folds agree, or
        ``'inconsistent'`` if they differ.

        Returns:
            ``'built'``, ``'finalized'``, ``'error'``, ``None`` (init),
            or ``'inconsistent'``.
        """
        return resolve_common_status(
            status
            for store in self._all_stores()
            if (status := store.status(node_name)) is not None
        )

    def finalize(self, nodes):
        """Release memory for built Head nodes (``built`` → ``finalized``).

        Disk artifacts are preserved so nodes can be reloaded.

        Args:
            nodes (list[str]): Head node names to finalize.
        """
        self._check_open()
        finalized_list = list()
        for i in nodes:
            if i is None:
                continue
            finalized_list.append(i)
            for outer_fold in self.outer_folds:
                for store in outer_fold.artifact_stores:
                    if store.status(i) == 'built':
                        store.finalize(i)
        return finalized_list

    def reinitialize(self, nodes):
        self._check_open()
        reinitialized_list = list()
        for i in nodes:
            if i is None:
                continue
            reinitialized = False
            for store in self._all_stores():
                if store.status(i) == 'finalized':
                    store.reset_node(i)
                    reinitialized = True
            if reinitialized:
                reinitialized_list.append(i)
        return reinitialized_list

    def close_exp(self):
        """Finalize all built nodes and mark the experiment as closed.

        Collector data is preserved. After this call, :attr:`status` is
        ``'closed'`` and no further builds or experiments are permitted until
        :meth:`reopen_exp` is called.
        """
        finalized_list = list()
        if self.status != "open":
            raise RuntimeError("")
        for outer_fold in self.outer_folds:
            for train_flow, artifact_store in zip(outer_fold.train_data_flows, outer_fold.artifact_stores):
                # train_flow and artifact_store share the same on-disk directory
                names = train_flow.list_nodes()
                for store in (train_flow, artifact_store):
                    for name in names:
                        if store.status(name) == 'built':
                            finalized_list.append(name)
                            store.finalize(name)
        self.status = "closed"
        self._save()
        return finalized_list

    def reopen_exp(self, pipeline):
        """Reopen a closed experiment and rebuild Stage nodes.

        Clears all Stage node objects, sets status back to ``'open'``, then
        calls :meth:`build`.

        Args:
            pipeline (Pipeline): Pipeline defining the node graph, forwarded
                to :meth:`build` to rebuild the reset Stage nodes.
        """
        if self.status != "closed":
            raise RuntimeError("")
        for outer_fold in self.outer_folds:
            for store in outer_fold.train_data_flows:
                for name in store.list_nodes():
                    if store.status(name) == 'finalized':
                        store.reset_node(name)
        self.status = "open"
        self.build(pipeline)
        self._save()

    def reset_nodes(self, nodes):
        """Reset nodes to ``init`` state.

        Removes node objects, clears cache entries, and resets Collector and
        Trainer data for the affected nodes.

        Args:
            nodes (list[str]): Node names to reset.
        """
        for name in nodes:
            for store in self._all_stores():
                store.reset_node(name)

        self.cache.clear_nodes(nodes)

        for v in self.collectors.values():
            v.reset_nodes(nodes)

    def _reset_serial_stale_nodes(self, pipeline, node_names):
        def stores_for_name(name):
            grp = pipeline.get_grp(pipeline.get_node(name).grp)
            for outer_fold in self.outer_folds:
                yield from (
                    outer_fold.train_data_flows if grp.role == 'stage'
                    else outer_fold.artifact_stores
                )

        stale = find_stale_nodes(pipeline, node_names, stores_for_name)
        if stale:
            self.reset_nodes(stale)

    def show_error_nodes(self, nodes=None, traceback=False):
        """Print nodes in ``error`` state.

        Args:
            nodes (list[str], optional): Node names to check. ``None`` checks
                every node found on disk.
            traceback (bool): Include full traceback in output.
        """
        stores = self._all_stores()
        if nodes is None:
            # train_data_flows and artifact_stores share their fold directory,
            # so listing one side per fold covers every known node name.
            node_names = {
                name
                for outer_fold in self.outer_folds
                for flow in outer_fold.train_data_flows
                for name in flow.list_nodes()
            }
        else:
            node_names = nodes

        errors = list()
        for n in node_names:
            if n is None:
                continue
            info = next((s.get_info(n) for s in stores if s.status(n) == 'error'), None)
            if info is None:
                continue
            err = info['error']
            if traceback:
                errors.append(f"[{n}] {err['type']}: {err['message']}\n{err['traceback']}")
            else:
                errors.append(f"[{n}] {err['type']}: {err['message']}")
        return errors if errors else None

    def build(self, pipeline, nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None):
        """Build Stage nodes.

        Args:
            pipeline (Pipeline): Pipeline defining the node graph. Compared
                against the artifacts already on disk — nodes whose ``serial``
                no longer matches are reset and rebuilt automatically.
            nodes: Node query — ``None`` (all stages), ``list``, or regex ``str``.
            rebuild (bool): If ``True``, rebuild already-built nodes.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _build_flow_single, _build_flow_multi
        from ._tracker import LoggerExecuteTracker
        logger = resolve_logger(logger)
        self._check_open()
        pipeline.check_data_compatibility(self.data)
        node_names = set(pipeline.get_node_names(nodes))
        target_nodes = [
            i for i in pipeline._get_affected_nodes([None])
            if i is not None
            and i in node_names
            and pipeline.grps[pipeline.nodes[i].grp].role == 'stage'
        ]
        if rebuild:
            self.reset_nodes(target_nodes)
        else:
            self._reset_serial_stale_nodes(pipeline, target_nodes)
            target_nodes = [
                i for i in target_nodes
                if self.get_status(i) not in ['built', 'finalized']
            ]
        if not target_nodes:
            logger.info("No stage nodes to build")
            return

        logger.info(f"Building {len(target_nodes)} node(s)")
        collectors = list(self.collectors.values())
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        tracker = LoggerExecuteTracker(total, n_jobs, logger)

        try:
            if n_jobs > 1:
                errors = _build_flow_multi(self.outer_folds, pipeline, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker)
            else:
                errors = _build_flow_single(self.outer_folds, pipeline, target_nodes,
                                            gpu_id_list=gpu_id_list, collectors=collectors,
                                            tracker=tracker)
        finally:
            tracker.close()

        error_nodes = list({n for _, _, n in errors})
        n_ok = len(target_nodes) - len(error_nodes)
        if error_nodes:
            logger.info(f"Build complete: {n_ok}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            logger.info(f"Build complete: {len(target_nodes)} node(s)")

    def exp(self, pipeline, nodes=None, finalize=False, n_jobs=1, gpu_id_list=None, logger=None):
        """Run Head nodes and invoke all matching Collectors.

        Args:
            pipeline (Pipeline): Pipeline defining the node graph. Compared
                against the artifacts already on disk — nodes whose ``serial``
                no longer matches are reset and rerun automatically.
            nodes: Node query — ``None`` (all heads matching ``self.tags`` if
                set, else all heads), ``list``, or regex ``str``.
            finalize (bool): If ``True``, finalize after all folds complete.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker
        logger = resolve_logger(logger)
        self._check_open()
        pipeline.check_data_compatibility(self.data)
        if nodes is None and self.tags:
            node_names = filter_node_names_by_tags(pipeline, self.tags)
        else:
            node_names = set(pipeline.get_node_names(nodes))
        candidate_nodes = [
            i for i in pipeline._get_affected_nodes([None])
            if i is not None
            and i in node_names
            and pipeline.grps[pipeline.nodes[i].grp].role == 'head'
        ]
        self._reset_serial_stale_nodes(pipeline, candidate_nodes)
        target_nodes = [
            i for i in candidate_nodes
            if self.get_status(i) not in ['built', 'finalized']
        ]
        if not target_nodes:
            logger.info("No head nodes to experiment")
            return

        logger.info(f"Experimenting {len(target_nodes)} node(s)")
        collectors = list(self.collectors.values())
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        tracker = LoggerExecuteTracker(total, n_jobs, logger)

        try:
            if n_jobs > 1:
                errors = _experiment_multi(self.outer_folds, pipeline, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker, finalize=finalize)
            else:
                errors = _experiment_single(self.outer_folds, pipeline, target_nodes,
                                            gpu_id_list=gpu_id_list, collectors=collectors,
                                            tracker=tracker, finalize=finalize)
        finally:
            tracker.close()

        error_nodes = list({n for _, n in errors})
        n_ok = len(target_nodes) - len(error_nodes)
        if error_nodes:
            logger.info(f"Exp complete: {n_ok}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            logger.info(f"Exp complete: {len(target_nodes)} node(s)")

    def collect(self, pipeline, collector, nodes=None, exist='skip', logger=None):
        """Run a Collector ad-hoc over already-built Head nodes.

        Args:
            pipeline (Pipeline): Pipeline defining the node graph.
            collector (Collector): Collector instance to run.
            nodes: Node query — ``None`` (all heads), ``list``, or regex ``str``.
            exist (str): ``'skip'`` (default) skips nodes already collected.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.

        Returns:
            Collector: The same collector after collection.
        """
        from ._executor import _run_collectors
        from ._node_processor import ProgressMonitor
        logger = resolve_logger(logger)

        node_names = set(pipeline.get_node_names(nodes))
        target_nodes = [
            name for name in pipeline._get_affected_nodes([None])
            if name is not None
            and name in node_names
            and not (exist == 'skip' and collector.has(name))
            and self.get_status(name) == 'built'
            and collector.connector.match(pipeline.get_node_attrs(name))
        ]

        if not target_nodes:
            return collector

        collector.on_attach(self)
        collector._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        monitor = ProgressMonitor()
        n_total = self.get_n_splits() * len(target_nodes)
        try:
            logger.create_session(0)
            logger.create_session(1)
            logger.start_progress(0, 'Collect', n_total)
            n_done = 0
            for name in target_nodes:
                node_attrs = pipeline.get_node_attrs(name)
                edges = node_attrs['edges']
                logger.start_progress(1, name)
                for outer_idx, outer_fold in enumerate(self.outer_folds):
                    for inner_idx, (train_flow, artifact_store) in enumerate(
                        zip(outer_fold.train_data_flows, outer_fold.artifact_stores)
                    ):
                        if artifact_store.status(name) != 'built':
                            continue
                        obj, result, info = artifact_store.get_objs(name)
                        train_data = train_flow.get_train(edges)
                        valid_data = train_flow.get_valid(edges)
                        test_data = outer_fold.get_test_data(edges)
                        ext_data = {}
                        if collector.get_properties().get('need_process_data', False):
                            ext_data[collector.name] = train_flow.get_data(collector.get_ext_data(), node_attrs['edges'])
                        _run_collectors(
                            [collector], node_attrs, obj, result, info,
                            train_data, valid_data, test_data, ext_data,
                            outer_idx, inner_idx, monitor
                        )
                    n_done += 1
                    logger.update_progress(0, n_done)
                logger.end_progress(1)
            logger.end_progress(0, n_total)
        finally:
            logger.remove_session(1)
            logger.remove_session(0)
        return collector

    def get_train_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_train(edges)

    def get_valid_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_valid(edges)

    def get_test_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].get_test_data(edges, i_idx)

    def get_node_info(self, pipeline):
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **DataSource**\n")

        for name in pipeline.nodes.keys():
            if name is None:
                continue
            node = pipeline.get_node(name)
            node_attrs = node.get_attrs(pipeline.grps)
            processor_name = node_attrs['processor'].__name__ if node_attrs['processor'] else 'None'
            edges_info_parts = []
            for key, edge_list in node_attrs['edges'].items():
                edge_strs = [f"{n or 'DataSource'}{f'[{v}]' if v else ''}" for n, v in edge_list]
                edges_info_parts.append(f"{key}: [{', '.join(edge_strs)}]")
            edges_info = ", ".join(edges_info_parts)
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node_attrs['method']}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = pipeline._find_descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def get_objs(self, node_name, outer_idx = 0, inner_idx = 0):
        fold = self.outer_folds[outer_idx]
        artifact_store = fold.artifact_stores[inner_idx]
        if artifact_store.status(node_name) is not None:
            return artifact_store.get_objs(node_name)
        return fold.train_data_flows[inner_idx].get_objs(node_name)

    def _save(self, filepath=None):
        if filepath is None:
            filepath = self.path / '__exp.pkl'

        save_data = {
            'data_key': self.data_key,
            'title': self.title,
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
            'cache_maxsize': self.cache_maxsize,
            'exp_id': self.exp_id,
            'tags': self.tags,
            'collector_keys': {name: type(c).__name__ for name, c in self.collectors.items()},
            'status': self.status
        }

        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)

    @staticmethod
    def load(filepath, data, data_key=None, aug_data=None):
        """Load a saved Experimenter from disk.

        Args:
            filepath (str | Path): Path to the experiment directory
                (contains ``__exp.pkl``).
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): If the saved experiment has a ``data_key``,
                this must match.

        Returns:
            Experimenter: Restored experimenter with all nodes, collectors, and
            trainers reloaded.

        Raises:
            ValueError: If ``data_key`` does not match the saved value.
        """
        COLLECTOR_TYPES = {
            'MetricCollector': MetricCollector,
            'StackingCollector': StackingCollector,
            'ModelAttrCollector': ModelAttrCollector,
            'SHAPCollector': SHAPCollector,
            'OutputCollector': OutputCollector,
            'ProcessCollector': ProcessCollector,
        }

        filepath = Path(filepath)
        with open(filepath / '__exp.pkl', 'rb') as f:
            save_data = pkl.load(f)

        saved_data_key = save_data.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        exp = Experimenter(
            data=data,
            path=filepath,
            sp=save_data['sp'],
            sp_v=save_data['sp_v'],
            splitter_params=save_data['splitter_params'],
            title=save_data['title'],
            data_key=saved_data_key,
            cache_maxsize=save_data.get('cache_maxsize', 4 * 1024 ** 3),
            aug_data=aug_data,
            _save=False
        )
        exp.exp_id = save_data['exp_id']
        exp.tags = save_data.get('tags', [])
        exp.status = save_data['status']

        # Collector 복원
        collector_keys = save_data.get('collector_keys', {})
        for coll_name, type_name in collector_keys.items():
            cls = COLLECTOR_TYPES.get(type_name)
            if cls is None:
                continue
            coll_path = filepath / '__collector' / coll_name
            if (coll_path / '__config.pkl').exists():
                collector = cls.load(coll_path)
                exp.collectors[coll_name] = collector

        return exp

    def desc_spec(self):
        return desc_spec(self)
