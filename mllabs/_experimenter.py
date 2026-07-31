import os
import re
import sys
import uuid
import pickle as pkl
import traceback
import warnings
import contextlib
from pathlib import Path

import pandas as pd

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._describer import desc_spec
from ._logger import resolve_logger
from ._cache import DataCache


def _start_native_redirect(log_path):
    """Redirect the calling process's OS-level stdout/stderr (fd 1/2) to
    *log_path*, capturing native library chatter (TensorFlow, LightGBM,
    CatBoost, cuDNN/XLA) written directly to the fd. Split into start/stop so
    ``Experimenter.open_os_log``/``close_os_log`` can hold the state open
    across multiple separate ``build``/``exp`` calls, rather than scoping a
    redirect to a single call.

    The calling process may also write progress/log output the user is meant
    to see over real stdout (e.g. ``DefaultLogger``'s progress bars via
    ``sys.stdout.write``/``print``). So before fd 1/2 are redirected to the
    log file, ``sys.stdout``/``sys.stderr`` are rebound to a duplicate of the
    original fd — Python-level output keeps reaching the real console, while
    only native C-level writes (which bypass those Python objects and hit fd
    1/2 directly) land in the log file.

    Returns:
        dict: Opaque state to pass to :func:`_stop_native_redirect`.
    """
    log_path = str(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = os.fdopen(saved_out_fd, 'w')
    sys.stderr = os.fdopen(saved_err_fd, 'w')
    with  open(log_path, 'w') as log_f:
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)
    return {
        'saved_out_fd': saved_out_fd, 'saved_err_fd': saved_err_fd,
        'orig_stdout': orig_stdout, 'orig_stderr': orig_stderr, 'log_f': log_f,
    }


def _stop_native_redirect(state):
    """Undo :func:`_start_native_redirect` — restore fd 1/2 and sys.stdout/stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(state['saved_out_fd'], 1)
    os.dup2(state['saved_err_fd'], 2)
    sys.stdout.close()  # closes saved_out_fd
    sys.stderr.close()  # closes saved_err_fd
    sys.stdout = state['orig_stdout']
    sys.stderr = state['orig_stderr']
    state['log_f'].close()
from ._run_common import resolve_common_status, find_stale_nodes, require_built_pipeline, name_matches
from ._experimenter_store import ExperimenterStore


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
        pipeline (Pipeline, optional): Pipeline this experimenter targets.
            Equivalent to calling :meth:`set_pipeline` after construction.

    Attributes:
        cache (DataCache): Shared LRU cache.
        status (str): ``'open'`` or ``'closed'``.
        pipeline (Pipeline): Pipeline set via the constructor or
            :meth:`set_pipeline`, persisted to ``{path}/pipeline.pkl``.

    Note:
        ``build``, ``exp``, ``collect``, and other node-graph-aware methods
        use ``self.pipeline`` — call :meth:`set_pipeline` first (or pass
        ``pipeline`` to the constructor).
    """

    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            aug_data=None, pipeline=None, _save=True
        ):
        self.cache_maxsize = cache_maxsize
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
        self.status = "open"
        self.pipeline = None
        self._os_log_state = None
        self._store = ExperimenterStore(self.path)
        self._store.initialize()
        if pipeline is not None:
            self.set_pipeline(pipeline)
        if _save:
            self._save()

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def set_status(self, status):
        """Set status and persist only the status meta row."""
        self.status = status
        self._store.set_meta('status', status)

    def open(self):
        """Experimenter를 open 상태로 변경"""
        self.set_status("open")

    def close(self):
        """Experimenter를 close 상태로 변경"""
        self.set_status("close")

    def open_os_log(self, log_path=None):
        """Start capturing this process's OS-level stdout/stderr — native
        library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) written
        directly to fd 1/2, which would otherwise pollute the console.

        Unrelated to the ``open``/``close`` experiment status — this only
        toggles OS-level log capture. While open:

        - ``build``/``exp`` with ``n_jobs=1`` run in this same process, so
          their native chatter is captured directly by this redirect — no
          separate handling needed per call.
        - ``build``/``exp`` with ``n_jobs>1`` additionally redirect each
          worker's own native chatter to ``{path}/__worker_logs/worker_{i}.log``
          for the duration of that call. When log capture is *not* open,
          workers do not redirect at all (matches pre-existing behavior).

        Call :meth:`close_os_log` to stop and restore the console. See also
        :meth:`os_log` for a context-manager form.

        Args:
            log_path: File to write to. Default:
                ``{path}/__worker_logs/master.log``.
        """
        if self._os_log_state is not None:
            raise RuntimeError("OS log capture is already open")
        if log_path is None:
            log_path = self.path / '__worker_logs' / 'master.log'
        self._os_log_state = _start_native_redirect(log_path)

    def close_os_log(self):
        """Stop OS-level stdout/stderr capture started by :meth:`open_os_log`."""
        if self._os_log_state is None:
            return
        _stop_native_redirect(self._os_log_state)
        self._os_log_state = None

    @contextlib.contextmanager
    def os_log(self, log_path=None):
        """Context manager form of :meth:`open_os_log`/:meth:`close_os_log`.

        Usage::

            with e.os_log():
                e.build(n_jobs=1)
                e.exp(n_jobs=4)
        """
        self.open_os_log(log_path)
        try:
            yield
        finally:
            self.close_os_log()

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3, aug_data=None):

        if os.path.exists(path):
            raise RuntimeError(f"Exists: {path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params,
            title=title, data_key=data_key, cache_maxsize=cache_maxsize,aug_data=aug_data
        )

    def set_pipeline(self, pipeline):
        """Set (or replace) the Pipeline this experimenter targets.

        If a Pipeline was already set, resets any node whose ``serial`` no
        longer matches the artifacts already on disk before adopting the
        new pipeline.

        Takes a built :class:`Pipeline`, not a :class:`PipelineBuilder` — the
        snapshot is what makes later builder edits unable to change a run that
        is already under way. Re-publish definitions with
        ``e.set_pipeline(p.build())``.

        Args:
            pipeline (Pipeline): Built pipeline defining the node graph.
        """
        require_built_pipeline(pipeline)
        if self.pipeline is not None:
            all_node_names = [n for n in pipeline.nodes if n is not None]
            self._reset_serial_stale_nodes(pipeline, all_node_names)
        self.pipeline = pipeline
        self._save_pipeline()

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline(pipeline) first.")
        return self.pipeline

    def _save_pipeline(self):
        with open(self.path / 'pipeline.pkl', 'wb') as f:
            pkl.dump(self.pipeline, f)

    def get_n_splits(self):
        return len(self.outer_folds)

    def get_n_splits_inner(self):
        return len(self.outer_folds[0].train_data_flows)

    def _all_stores(self):
        """Flatten every store (both Stage and Head, all folds) into one list."""
        return [
            store
            for outer_fold in self.outer_folds
            for store in outer_fold.train_data_flows
        ]

    def get_status(self, node_name):
        """Return the disk status of a node across all folds.

        One :class:`NodeStore` per fold now covers both Stages and Trials —
        they always shared the directory. Returns the common status if all
        folds agree, or ``'inconsistent'`` if they differ.

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
                for store in outer_fold.train_data_flows:
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
            for store in outer_fold.train_data_flows:
                for name in store.list_nodes():
                    if store.status(name) == 'built':
                        finalized_list.append(name)
                        store.finalize(name)
        self.set_status("closed")
        return finalized_list

    def reopen_exp(self):
        """Reopen a closed experiment and rebuild Stage nodes.

        Clears all Stage node objects, sets status back to ``'open'``, then
        calls :meth:`build`.
        """
        self._require_pipeline()
        if self.status != "closed":
            raise RuntimeError("")
        for outer_fold in self.outer_folds:
            for store in outer_fold.train_data_flows:
                for name in store.list_nodes():
                    if store.status(name) == 'finalized':
                        store.reset_node(name)
        self.set_status("open")
        self.build()

    def reset_nodes(self, nodes):
        """Reset nodes to ``init`` state.

        Removes node objects and clears cache entries for the affected nodes.

        Args:
            nodes (list[str]): Node names to reset.
        """
        for name in nodes:
            for store in self._all_stores():
                store.reset_node(name)

        self.cache.clear_nodes(nodes)


    def _reset_serial_stale_nodes(self, pipeline, node_names):
        """Reset Stage nodes whose stored serial no longer matches the pipeline."""
        def stores_for_name(name):
            for outer_fold in self.outer_folds:
                yield from outer_fold.train_data_flows

        current = {n: pipeline.nodes[n].serial for n in node_names}
        stale = find_stale_nodes(current, node_names, stores_for_name)
        if stale:
            self.reset_nodes(stale)

    def _reset_stale_trials(self, attrs_map):
        """Reset Trials whose stored id no longer matches their current definition.

        A Trial's id folds in the serials of every Stage it reads, so editing a
        Stage lands here as a mismatch — the replacement for the ``_bump_serials``
        cascade that used to reach Head nodes through ``output_edges``.
        """
        def stores_for_name(name):
            for outer_fold in self.outer_folds:
                yield from outer_fold.train_data_flows

        current = {name: attrs['serial'] for name, attrs in attrs_map.items()}
        stale = find_stale_nodes(current, list(attrs_map), stores_for_name)
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

    def build(self, nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None):
        """Build Stage nodes.

        Uses ``self.pipeline`` — compared against the artifacts already on
        disk, nodes whose ``serial`` no longer matches are reset and
        rebuilt automatically.

        Args:
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
        pipeline = self._require_pipeline()
        self._save_pipeline()
        pipeline.check_data_compatibility(self.data)
        node_names = set(pipeline.get_node_names(nodes))
        target_nodes = [i for i in pipeline.topo_order() if i in node_names]
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
        collectors = []   # Collectors belong to an Experiment and run against Trials
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        n_jobs = min(n_jobs, total)
        tracker = LoggerExecuteTracker(total, n_jobs, logger)

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _build_flow_multi(self.outer_folds, pipeline, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker, log_dir=log_dir)
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

    def exp(self, experiment, collectors=None, trials=None, finalize=False,
            n_jobs=1, gpu_id_list=None, logger=None):
        """Run an :class:`~mllabs.BaseExperiment`'s Trials and invoke its Collectors.

        The Trial sequence is drained up front (``get_trial_nums()`` then that
        many ``get_next_trial()`` calls) so the dispatcher still knows its full
        target list before it starts.

        Each Trial's id folds in the serials of the Stages it reads, so a Trial
        whose definition — or whose upstream preprocessing — changed is reset
        and rerun automatically.

        Args:
            experiment (BaseExperiment): Source of Trials.
            collectors (Collectors, optional): Registry the Experiment's
                collector names are resolved against.
            trials: Trial-name filter — ``None`` (all), ``list``, or regex ``str``.
            finalize (bool): If ``True``, finalize after all folds complete.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker
        logger = resolve_logger(logger)
        self._check_open()
        pipeline = self._require_pipeline()
        self._save_pipeline()
        pipeline.check_data_compatibility(self.data)

        attrs_map = {
            t.name: {**t.get_attrs(), 'serial': t.trial_id(pipeline)}
            for t in experiment.get_trials()
            if name_matches(t.name, trials)
        }
        self._reset_stale_trials(attrs_map)
        target_nodes = [
            name for name in attrs_map
            if self.get_status(name) not in ['built', 'finalized']
        ]
        if not target_nodes:
            logger.info("No trials to run")
            return

        logger.info(f"Experimenting {len(target_nodes)} trial(s)")
        collectors = (
            collectors.resolve(experiment.collector_names) if collectors is not None else []
        )
        for c in collectors:
            c.on_attach(self)
            c._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        n_jobs = min(n_jobs, total)
        tracker = LoggerExecuteTracker(total, n_jobs, logger)

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _experiment_multi(self.outer_folds, attrs_map, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker, finalize=finalize,
                                           log_dir=log_dir)
            else:
                errors = _experiment_single(self.outer_folds, attrs_map, target_nodes,
                                            gpu_id_list=gpu_id_list, collectors=collectors,
                                            tracker=tracker, finalize=finalize)
        finally:
            tracker.close()

        error_nodes = list({n for _, n in errors})
        n_ok = len(target_nodes) - len(error_nodes)
        if error_nodes:
            logger.info(f"Exp complete: {n_ok}/{len(target_nodes)} trial(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            logger.info(f"Exp complete: {len(target_nodes)} trial(s)")

    def get_train_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_train(edges)

    def get_valid_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_valid(edges)

    def get_test_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].get_test_data(edges, i_idx)

    def get_node_info(self):
        pipeline = self._require_pipeline()
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **DataSource**\n")

        for name in pipeline.nodes.keys():
            if name is None:
                continue
            node_attrs = pipeline.get_node_attrs(name)
            processor = node_attrs['processor']
            processor_name = getattr(processor, '__name__', processor) if processor else 'None'
            edges_info = ", ".join(
                f"{key}: {dsl_string}" for key, dsl_string in node_attrs['edges'].items()
            )
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node_attrs['method']}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = pipeline.descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def get_objs(self, node_name, outer_idx = 0, inner_idx = 0):
        return self.outer_folds[outer_idx].train_data_flows[inner_idx].get_objs(node_name)

    def get_worker_logs(self, worker=None):
        """Native (OS-level stdout/stderr) output captured while OS log
        capture was open (see :meth:`open_os_log`/:meth:`os_log`).

        Multi-worker ``build``/``exp`` (``n_jobs > 1``) redirect each worker's
        stdout/stderr to ``{path}/__worker_logs/worker_{i}.log``, capturing
        native library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) that
        would otherwise pollute the console. This process's own (master)
        output goes to ``worker_logs/master.log``. Each run overwrites the
        previous.

        Args:
            worker: ``int`` for a per-worker log, ``'master'`` for this
                process's own log, or ``None`` for all.

        Returns:
            dict mapping worker index (``int``) / ``'master'`` to captured
            text, or a single string if *worker* is given. Empty if nothing
            was captured.
        """
        log_dir = self.path / '__worker_logs'
        if worker is not None:
            fname = 'master.log' if worker == 'master' else f'worker_{worker}.log'
            f = log_dir / fname
            return f.read_text() if f.exists() else ''
        if not log_dir.exists():
            return {}
        logs = {
            int(f.stem.split('_')[1]): f.read_text()
            for f in sorted(log_dir.glob('worker_*.log'))
        }
        master_f = log_dir / 'master.log'
        if master_f.exists():
            logs['master'] = master_f.read_text()
        return logs

    def _save(self):
        self._store.save_meta({
            'data_key': self.data_key,
            'title': self.title,
            'cache_maxsize': self.cache_maxsize,
            'exp_id': self.exp_id,
            'status': self.status,
        })
        self._save_splitters()

    def _save_splitters(self):
        with open(self.path / '__splitters.pkl', 'wb') as f:
            pkl.dump({
                'sp': self.sp,
                'sp_v': self.sp_v,
                'splitter_params': self.splitter_params,
            }, f)

    @staticmethod
    def load(filepath, data, data_key=None, aug_data=None):
        """Load a saved Experimenter from disk.

        Args:
            filepath (str | Path): Path to the experiment directory
                (contains ``__exp.db``).
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): If the saved experiment has a ``data_key``,
                this must match.

        Returns:
            Experimenter: Restored experimenter with all nodes, collectors, and
            trainers reloaded.

        Raises:
            ValueError: If ``data_key`` does not match the saved value.
        """
        filepath = Path(filepath)
        store = ExperimenterStore(filepath)
        meta = store.fetch_meta()

        saved_data_key = meta.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        with open(filepath / '__splitters.pkl', 'rb') as f:
            splitters = pkl.load(f)

        exp = Experimenter(
            data=data,
            path=filepath,
            sp=splitters['sp'],
            sp_v=splitters['sp_v'],
            splitter_params=splitters['splitter_params'],
            title=meta['title'],
            data_key=saved_data_key,
            cache_maxsize=meta.get('cache_maxsize', 4 * 1024 ** 3),
            aug_data=aug_data,
            _save=False
        )
        exp.exp_id = meta['exp_id']
        exp.status = meta['status']

        pipeline_path = filepath / 'pipeline.pkl'
        if pipeline_path.exists():
            with open(pipeline_path, 'rb') as f:
                exp.pipeline = pkl.load(f)

        return exp

    def desc_spec(self):
        return desc_spec(self)
