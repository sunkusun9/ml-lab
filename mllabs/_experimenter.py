import os
import sys
import pickle as pkl
import traceback
import contextlib
from pathlib import Path


from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._describer import desc_spec
from ._logger import resolve_logger


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
from ._edge_dsl import referenced_nodes
from ._pipeline import _definition_of
from ._run_common import resolve_common_status, require_built_pipeline


def _resolve_collectors(collectors):
    """Accept a Collectors registry, a list of instances, or None."""
    if collectors is None:
        return []
    if hasattr(collectors, 'resolve'):
        return collectors.resolve(None)
    return list(collectors)


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
                data_source=DataWrapperProvider(data, train_idx, valid_idx=valid_idx,
                                                test_idx=test_idx, aug_data=aug_data),
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
        project (Project): Owning project. Supplies the path and the Pipeline
            versions this Experimenter runs against.
        name (str): Experimenter name. This is its identity — the directory
            ``{project}/exp/{name}`` and the key used in TrialStore history.
        data: Input dataset (pandas DataFrame, polars DataFrame, or numpy array).
        data_names (list[str], optional): Column names override.
        sp: Outer splitter (sklearn splitter API). Default
            ``ShuffleSplit(n_splits=1, random_state=1)``.
        sp_v: Inner splitter for nested cross-validation. ``None`` disables.
        splitter_params (dict, optional): Maps splitter keyword args to column
            names in *data*, e.g. ``{'y': 'target'}``.
        title (str, optional): Human-readable experiment title.
        data_key (str, optional): Identifier verified on :meth:`load` to prevent
            data mismatch.
        pipeline_name (str): Which Pipeline in the project to run against.
            Default ``'pipeline'``.
        pipeline_version (int, optional): Version to adopt immediately;
            equivalent to :meth:`set_pipeline_version` after construction.

    Attributes:
        cache (DataCache): Shared LRU cache.
        status (str): ``'open'`` or ``'closed'``.
        pipeline (Pipeline): The loaded Pipeline. Not stored here — only the
            ``(pipeline_name, pipeline_version)`` pointer is, so the Pipeline
            itself lives once, under the Project.

    Note:
        ``build``, ``exp`` and other node-graph-aware methods use
        ``self.pipeline`` — call :meth:`set_pipeline_version` first (or pass
        ``pipeline_version`` to the constructor).
    """

    def __init__(
            self, project, name, data, data_names = None,
            sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None,
            aug_data=None, pipeline_name='pipeline', pipeline_version=None, _save=True
        ):
        self.project = project
        self.name = name
        self.path = project.exp_path(name)
        data_native = data
        self.data = wrap(data)
        self.aug_data = wrap(aug_data) if aug_data is not None else None
        self.title = title
        self.data_key = data_key
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}

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

        self.cache = project.cache

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
        self.pipeline_name = pipeline_name
        self.pipeline_version = None
        self._os_log_state = None
        self._store = project.experimenters
        if pipeline_version is not None:
            self.set_pipeline_version(pipeline_version)
        if _save:
            self._save()

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def set_status(self, status):
        """Set status and persist only the status meta row."""
        self.status = status
        self._store.set(self.name, 'status', status)

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

    def set_pipeline_version(self, version, pipeline_name=None):
        """Point this Experimenter at a Pipeline version from its Project.

        The Pipeline is loaded from the Project rather than handed in, so an
        Experimenter records *which version* it ran against instead of keeping
        its own copy.

        Moving between versions diffs the two Pipelines (:meth:`Pipeline.diff_from`)
        and drops exactly the artifacts the change invalidates: the Stages whose
        definition or inputs differ, everything downstream of them, Stages that
        no longer exist, and any Trial that read one of those Stages.

        Args:
            version (int): Pipeline version number (see ``Project.build_pipeline``).
            pipeline_name (str, optional): Pipeline name within the Project.
                Defaults to the one this Experimenter was created with.

        Returns:
            Pipeline: The loaded, built Pipeline.
        """
        if pipeline_name is not None:
            self.pipeline_name = pipeline_name
        pipeline = self.project.load_pipeline(self.pipeline_name, version)
        require_built_pipeline(pipeline)
        if self.pipeline is not None:
            self._drop_stale(pipeline.diff_from(self.pipeline))
        self.pipeline = pipeline
        self.pipeline_version = version
        self._store.set(self.name, 'pipeline_name', self.pipeline_name)
        self._store.set(self.name, 'pipeline_version', version)
        return pipeline

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline_version(version) first.")
        return self.pipeline

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


    def _drop_stale(self, stale_stages):
        """Remove the artifacts a set of stale Stages invalidates.

        Trials live outside the Pipeline, so the diff cannot name them. It does
        not have to: each artifact records the ``edges`` it was built from, so a
        Trial that read a stale Stage is found by looking at what is on disk.
        """
        if not stale_stages:
            return
        doomed = set(stale_stages)
        for outer_fold in self.outer_folds:
            for flow in outer_fold.train_data_flows:
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
        return doomed

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

        Staleness is settled when a Pipeline version is adopted
        (:meth:`set_pipeline_version`), so anything still on disk when this
        runs is current — it only builds what is missing.

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
        pipeline.check_data_compatibility(self.data)
        node_names = set(pipeline.get_node_names(nodes))
        target_nodes = [i for i in pipeline.topo_order() if i in node_names]
        if rebuild:
            self.reset_nodes(target_nodes)
        else:
            # Staleness is settled when a version is adopted
            # (set_pipeline_version), so anything still on disk here is current.
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

    def exp(self, trials, collectors=None, n_jobs=1, gpu_id_list=None, logger=None):
        """Run *trials* against the Stage graph and invoke matching Collectors.

        Args:
            trials: ``[(Trial, outer_idx, inner_idx), ...]`` — each entry names
                exactly which fold a Trial runs on. Expanding folds here rather
                than in the executor keeps fold policy in one place and lets the
                dispatcher take its target list literally.
            collectors: :class:`~mllabs.Collectors` registry, a list of
                Collector instances, or ``None`` to collect nothing.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled trials.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.

        Trial definitions are registered in the project's
        :class:`~mllabs.TrialStore`, and every fold that actually runs records
        its outcome there as it finishes (see
        :class:`~mllabs._tracker.TrialHistTracker`). Folds skipped as already
        built produce no new row — their result was recorded when they ran.
        """
        from ._executor import _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker, TrialHistTracker
        logger = resolve_logger(logger)
        self._check_open()
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)

        collectors = _resolve_collectors(collectors)
        jobs = self._make_jobs(trials)
        if not jobs:
            logger.info("No trials to run")
            return

        logger.info(f"Experimenting {len(jobs)} job(s)")
        for c in collectors:
            c.on_attach(self)
            c._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        n_jobs = min(n_jobs, len(jobs))
        self.project.trials.register_all(t for t, _, _ in trials)
        tracker = TrialHistTracker(
            LoggerExecuteTracker(len(jobs), n_jobs, logger),
            self.project.trials, self.name, self.pipeline_version,
        )

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _experiment_multi(jobs, n_jobs, gpu_id_list=gpu_id_list,
                                           collectors=collectors, tracker=tracker,
                                           log_dir=log_dir)
            else:
                errors = _experiment_single(jobs, gpu_id_list=gpu_id_list,
                                            collectors=collectors, tracker=tracker)
        finally:
            tracker.close()

        error_names = list({n for _, n in errors})
        n_ok = len(jobs) - len(error_names)
        if error_names:
            logger.info(f"Exp complete: {n_ok}/{len(jobs)} job(s), "
                        f"{len(error_names)} error(s): {error_names}")
        else:
            logger.info(f"Exp complete: {len(jobs)} job(s)")

    def _make_jobs(self, trials):
        """Expand ``(Trial, outer, inner)`` entries into runnable TrialJobs.

        A Trial reruns when its own definition differs from the one its artifact
        was built with — compared value by value, since params are plain data by
        construction. Stage changes are already handled when a Pipeline version
        is adopted, which drops the Trials that read them.
        """
        from ._executor import TrialJob
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        jobs = []
        for trial, outer_idx, inner_idx in trials:
            flow = self.outer_folds[outer_idx].train_data_flows[inner_idx]
            attrs = trial.get_attrs()

            info = flow.get_info(trial.name)
            if info is not None and info.get('definition') != _definition_of(attrs):
                flow.reset_node(trial.name)
                self.cache.clear_nodes([trial.name])
                info = None
            if info is not None and info.get('status') in ('built', 'finalized'):
                continue

            if trial.name not in gpu_cache:
                adapter = resolve_node_adapter(attrs.get('processor'), attrs.get('adapter'))
                gpu_cache[trial.name] = adapter.get_gpu_usage(attrs.get('params')) != GPU_NO

            jobs.append(TrialJob(trial, attrs, (outer_idx, inner_idx), flow,
                                 need_gpu=gpu_cache[trial.name]))
        return jobs

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
        self._store.save({
            'name': self.name,
            'data_key': self.data_key,
            'title': self.title,
            'status': self.status,
            'pipeline_name': self.pipeline_name,
            'pipeline_version': self.pipeline_version,
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
    def load(project, name, data, data_key=None, aug_data=None):
        """Reopen a saved Experimenter by name.

        Args:
            project (Project): Owning project.
            name (str): Experimenter name — its directory under ``exp/``.
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): If the saved experiment has a ``data_key``,
                this must match.

        Returns:
            Experimenter: Restored experimenter, with its Pipeline version
            reloaded from the project.

        Raises:
            ValueError: If ``data_key`` does not match the saved value.
        """
        path = project.exp_path(name)
        meta = project.experimenters.fetch(name)
        if meta is None:
            raise KeyError(f"No experimenter named {name!r} in this project")

        saved_data_key = meta.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        with open(path / '__splitters.pkl', 'rb') as f:
            splitters = pkl.load(f)

        exp = Experimenter(
            project=project,
            name=name,
            data=data,
            sp=splitters['sp'],
            sp_v=splitters['sp_v'],
            splitter_params=splitters['splitter_params'],
            title=meta.get('title'),
            data_key=saved_data_key,
            aug_data=aug_data,
            pipeline_name=meta.get('pipeline_name', 'pipeline'),
            _save=False
        )
        exp.status = meta['status']
        version = meta.get('pipeline_version')
        if version is not None:
            exp.set_pipeline_version(version)
        return exp

    def desc_spec(self):
        return desc_spec(self)
