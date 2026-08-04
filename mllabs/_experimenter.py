import os
import sys
import pickle as pkl
import traceback
import contextlib
from pathlib import Path

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._experimenter_store import ExperimenterStore, DB_NAME as _EXP_DB
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
from ._run_common import resolve_common_status, require_built_pipeline


def _resolve_collectors(collectors):
    """``(instances, CollectHist | None)`` from a registry, a list, or None.

    A registry brings its own history — it owns the project-global path both
    the Collectors and their history live under. A bare list has no such
    place, so its outcomes go unrecorded unless ``exp(collect_hist=...)``
    names one.
    """
    if collectors is None:
        return [], None
    if hasattr(collectors, 'resolve'):
        return collectors.resolve(None), getattr(collectors, 'hist', None)
    return list(collectors), None


class OuterFold:
    """One outer fold: test indices and per-inner-fold TrainDataFlows.

    Every TrainDataFlow of every OuterFold shares the same NodeStore (this
    run's own — see ``Experimenter.node_store``); only ``outer_idx``/``j``
    (inner_idx) tell them apart.

    DataWrapperProvider inside each TrainDataFlow persists only indices — DataWrapper is transient.
    Call set_data(data) to re-inject DataWrapper and cache after load.
    """

    def __init__(self, outer_idx, store, data, test_idx, train_idx_list, cache=None, aug_data=None):
        self.outer_idx = outer_idx
        self.test_idx = test_idx
        self.data = data
        self.train_data_flows = [
            TrainDataFlow(
                store=store,
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
    node builds and Trial experiments fold-by-fold.

    No ``Project`` dependency, and nothing to inject but a ``cache``: the run
    owns its own :class:`~mllabs.ExperimenterStore`, built from ``path``, and
    everything needed to reopen it lives in that directory. ``Project`` only
    supplies the path and records the name in its index.

    Constructing is *creating*. It splits the data and writes a fresh state,
    so pointing it at an existing directory starts that run over rather than
    resuming it — :meth:`load_experimenter` is how an existing one comes back.
    A Pipeline is never a constructor argument either way; adopt one with
    :meth:`set_pipeline`, which saves it as this run's ``pipeline.pkl``.

    Args:
        path: This run's own base directory (``{project}/exp/{name}`` when
            created via a Project), created if it does not exist.
        name (str): Experimenter name. This is its identity — the directory
            above and the key used in TrialStore history.
        data: Input dataset (pandas DataFrame, polars DataFrame, or numpy array).
        data_names (list[str], optional): Column names override.
        sp: Outer splitter (sklearn splitter API). Default
            ``ShuffleSplit(n_splits=1, random_state=1)``.
        sp_v: Inner splitter for nested cross-validation. ``None`` disables.
        splitter_params (dict, optional): Maps splitter keyword args to column
            names in *data*, e.g. ``{'y': 'target'}``.
        title (str, optional): Human-readable experiment title.
        data_key (str, optional): Identifier verified on reload to prevent
            data mismatch.
        cache (DataCache, optional): Shared LRU cache.

    Attributes:
        cache (DataCache): Shared LRU cache, or ``None``.
        pipeline (Pipeline): The adopted Pipeline, kept as this run's own
            ``pipeline.pkl``. The ``(pipeline_name, pipeline_version)``
            pointer is recorded too, but only as provenance — it names the
            project version this copy was taken from.

    Note:
        ``build``, ``exp`` and other node-graph-aware methods use
        ``self.pipeline`` — call :meth:`set_pipeline` first.
    """

    def __init__(
            self, path, name, data, data_names = None,
            sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None,
            aug_data=None, cache=None
        ):
        self.name = name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._store = ExperimenterStore(self.path)
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

        self.cache = cache
        self.node_store = NodeStore(self.path / '__folds')

        self.outer_folds = [
            OuterFold(
                outer_idx=i,
                store=self.node_store,
                data=self.data,
                test_idx=test_idx,
                train_idx_list=inner_folds,
                cache=self.cache,
                aug_data=self.aug_data,
            )
            for i, (test_idx, inner_folds) in enumerate(raw_splits)
        ]
        self.pipeline = None
        self.pipeline_name = 'pipeline'
        self.pipeline_version = None
        self._os_log_state = None
        self._save()

    @staticmethod
    def load_experimenter(path, data, data_key=None, aug_data=None, cache=None):
        """Reopen the Experimenter rooted at *path*.

        Everything comes out of that directory — meta and splitters from its
        ``__exp.db``, the Pipeline from its ``pipeline.pkl`` — so no Project
        is involved and no ``(pipeline_name, pipeline_version)`` is resolved.
        The name is the directory's own.

        Args:
            path: The Experimenter's base directory.
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): Must match the saved value, if one was
                given when the Experimenter was created.
            aug_data (optional): External data appended to inner train splits.
            cache (DataCache, optional): Shared LRU cache.

        Returns:
            Experimenter: The reopened run.

        Raises:
            KeyError: If *path* holds no saved Experimenter.
            ValueError: If *data_key* does not match the saved value.
        """
        path = Path(path)
        # Checked before building the store, which would otherwise create the
        # directory and an empty db for a run that does not exist.
        if not (path / f'{_EXP_DB}.db').exists():
            raise KeyError(f"No experimenter saved at {path}")
        store = ExperimenterStore(path)
        meta = store.fetch()
        if meta is None:
            raise KeyError(f"No experimenter saved at {path}")

        saved_data_key = meta.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        splitters = store.load_splitters() or {}
        # 'sp' has a non-None default; passing the stored None through would
        # break the split rather than fall back to it.
        split_kwargs = {k: v for k, v in splitters.items() if v is not None}
        exp = Experimenter(
            path, meta['name'], data,
            title=meta.get('title'), data_key=saved_data_key,
            aug_data=aug_data, cache=cache, **split_kwargs,
        )
        pipeline = store.load_pipeline()
        if pipeline is not None:
            exp.set_pipeline(pipeline, meta.get('pipeline_name') or 'pipeline')
        return exp

    def open_os_log(self, log_path=None):
        """Start capturing this process's OS-level stdout/stderr — native
        library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) written
        directly to fd 1/2, which would otherwise pollute the console.

        This only toggles OS-level log capture, independent of anything else
        on the Experimenter. While open:

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

    def set_pipeline(self, pipeline, pipeline_name=None):
        """Adopt an already-loaded Pipeline.

        Takes the Pipeline object directly rather than a version number —
        this class has no way to load one by name/version itself (see the
        class docstring); ``Project.experimenter()`` resolves that before
        calling this. ``self.pipeline_version`` is read straight off
        *pipeline* (its ``.version``), so it's never tracked as a separate,
        possibly-diverging value.

        The Pipeline is written to this Experimenter's own ``pipeline.pkl``,
        which is what the constructor reads back — reopening needs only this
        directory, never a Project. Persisting requires an
        ``experimenter_store``; without one this Experimenter keeps the
        Pipeline in memory only, the same way it skips saving its meta.

        Moving between versions diffs the two Pipelines (:meth:`Pipeline.diff_from`)
        and drops exactly the Stage artifacts the change invalidates: Stages
        whose definition or inputs differ, everything downstream of them, and
        Stages that no longer exist. Trials are left as-is — a Trial's
        artifact and its ``TrialStore.experiment_hist`` row document the
        pipeline version it actually ran against, which stays valid even
        after a newer version is adopted; rerunning it is a separate,
        explicit action.

        Args:
            pipeline (Pipeline): Already-built, already-loaded Pipeline.
            pipeline_name (str, optional): Name to record this Pipeline
                under in this Experimenter's own meta. Defaults to the one
                this Experimenter was created with.

        Returns:
            Pipeline: *pipeline*, unchanged.
        """
        require_built_pipeline(pipeline)
        if pipeline_name is not None:
            self.pipeline_name = pipeline_name
        if self.pipeline is not None:
            stale = pipeline.diff_from(self.pipeline)
            if stale:
                self.reset_nodes(sorted(stale))
        self.pipeline = pipeline
        self.pipeline_version = pipeline.version
        self._store.save_pipeline(pipeline)
        self._store.set(self.name, 'pipeline_name', self.pipeline_name)
        self._store.set(self.name, 'pipeline_version', self.pipeline_version)
        return pipeline

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No pipeline set. Call set_pipeline(pipeline) first.")
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

        One :class:`NodeStore` covers this whole run — Stages and Trials
        always shared its directory, and every fold now shares the same
        store instance too (told apart by ``outer_idx``/``inner_idx``).
        Returns the common status if all folds agree, or ``'inconsistent'``
        if they differ.

        Returns:
            ``'built'``, ``'error'``, ``None`` (init), or ``'inconsistent'``.
        """
        return resolve_common_status(
            status
            for store in self._all_stores()
            if (status := store.status(node_name)) is not None
        )

    def reset_nodes(self, nodes):
        """Reset nodes to ``init`` state.

        Removes node objects and clears cache entries for the affected nodes.

        Args:
            nodes (list[str]): Node names to reset.
        """
        for name in nodes:
            for store in self._all_stores():
                store.reset_node(name)

        if self.cache is not None:
            self.cache.clear_nodes(nodes)

    def show_error_nodes(self, nodes=None, traceback=False, trial_store=None):
        """Print nodes in ``error`` state.

        Error detail no longer lives on the artifact itself (see
        ``NodeStore``) — it's recorded in ``TrialStore.experiment_hist``
        (Trials) or this run's own ``NodeStore`` history (Stages), so this
        queries both instead of walking fold directories.

        Args:
            nodes (list[str], optional): Node names to check. ``None`` checks
                every node recorded for this run.
            traceback (bool): Include full traceback in output.
            trial_store (TrialStore, optional): Where Trial history for this
                run lives. ``None`` skips the Trial half, reporting only
                Stage errors from this run's own ``NodeStore``.
        """
        rows = [
            (r['node_name'], r['outer_idx'], r['inner_idx'], r['info'])
            for r in self.node_store.get_hist()
            if r['status'] == 'error'
        ]
        if trial_store is not None:
            rows += [
                (r['trial_name'], r['outer_idx'], r['inner_idx'], r['info'])
                for r in trial_store.get_hist(experimenter=self.name)
                if r['status'] == 'error'
            ]
        if nodes is not None:
            node_set = set(nodes)
            rows = [r for r in rows if r[0] in node_set]

        errors = list()
        for name, outer_idx, inner_idx, info in rows:
            err = (info or {}).get('error', {})
            label = f"[{name}] fold {outer_idx}_{inner_idx}"
            if traceback:
                errors.append(f"{label} {err.get('type')}: {err.get('message')}\n{err.get('traceback')}")
            else:
                errors.append(f"{label} {err.get('type')}: {err.get('message')}")
        return errors if errors else None

    def build(self, nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None):
        """Build Stage nodes.

        Staleness is settled when a Pipeline is adopted (:meth:`set_pipeline`),
        so anything still on disk when this runs is current — it only builds
        what is missing.

        Args:
            nodes: Node query — ``None`` (all stages), ``list``, or regex ``str``.
            rebuild (bool): If ``True``, rebuild already-built nodes.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, NodeInfoTracker
        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)
        node_names = set(pipeline.get_node_names(nodes))
        target_nodes = [i for i in pipeline.topo_order() if i in node_names]
        if rebuild:
            self.reset_nodes(target_nodes)

        jobs = self._make_node_jobs(pipeline, target_nodes, gpu_id_list)
        if not jobs:
            logger.info("No stage nodes to build")
            return

        logger.info(f"Building {len(jobs)} job(s)")
        n_jobs = min(n_jobs, len(jobs))
        tracker = NodeInfoTracker(
            LoggerExecuteTracker(len(jobs), n_jobs, logger),
            self.node_store, self.pipeline_version,
        )

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _execute_multi(jobs, n_jobs, self.node_store, gpu_id_list=gpu_id_list,
                                        tracker=tracker, log_dir=log_dir)
            else:
                errors = _execute_single(jobs, self.node_store, gpu_id_list=gpu_id_list, tracker=tracker)
        finally:
            tracker.close()

        error_names = sorted({n for _, _, n in errors})
        n_ok = len(jobs) - len(errors)
        if errors:
            logger.info(f"Build complete: {n_ok}/{len(jobs)} job(s), {len(errors)} error(s): {error_names}")
        else:
            logger.info(f"Build complete: {len(jobs)} job(s)")

    def _make_node_jobs(self, pipeline, node_names, gpu_id_list):
        """Expand Stage node names into per-fold Jobs.

        Skip decisions live here, not in the executor (mirrors ``_make_jobs``
        for Trials) — a fold already built for a given node is left out; the
        executor only orders dispatch among what's left, via
        ``flow.get_missing_nodes``.
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

            for outer_idx, outer_fold in enumerate(self.outer_folds):
                for inner_idx, flow in enumerate(outer_fold.train_data_flows):
                    if flow.status(name) == 'built':
                        continue
                    jobs.append(Job(name, spec, outer_idx, inner_idx, flow, need_gpu=need_gpu))
        return jobs

    def exp(self, trials, trial_store, collectors=None, collect_hist=None,
            n_jobs=1, gpu_id_list=None, logger=None):
        """Run *trials* against the Stage graph and invoke matching Collectors.

        Args:
            trials: ``[(Trial, outer_idx, inner_idx), ...]`` — each entry names
                exactly which fold a Trial runs on. Expanding folds here rather
                than in the executor keeps fold policy in one place and lets the
                dispatcher take its target list literally.
            trial_store (TrialStore): Where Trial definitions are registered
                and fold outcomes are recorded — also what decides which
                folds are skipped as already built (see :meth:`_make_jobs`).
            collectors: :class:`~mllabs.Collectors` registry, a list of
                Collector instances, or ``None`` to collect nothing.
            collect_hist (CollectHist, optional): Where each Collector's
                per-fold outcome is recorded. Defaults to the registry's own
                ``hist`` when *collectors* is a registry; a bare list has none
                unless one is named here.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled trials.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.

        Trial definitions are registered in *trial_store*, and every fold
        that actually runs records its outcome there as it finishes (see
        :class:`~mllabs._tracker.TrialHistTracker`). Folds skipped as already
        built produce no new row — their result was recorded when they ran.
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, TrialHistTracker
        logger = resolve_logger(logger)
        pipeline = self._require_pipeline()
        pipeline.check_data_compatibility(self.data)

        collectors, registry_hist = _resolve_collectors(collectors)
        collect_hist = collect_hist if collect_hist is not None else registry_hist
        jobs = self._make_jobs(trials, trial_store)
        if not jobs:
            logger.info("No trials to run")
            return

        logger.info(f"Experimenting {len(jobs)} job(s)")
        for c in collectors:
            c.on_attach(self)
            c._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        n_jobs = min(n_jobs, len(jobs))
        trial_store.register_all(t for t, _, _ in trials)
        tracker = TrialHistTracker(
            LoggerExecuteTracker(len(jobs), n_jobs, logger),
            trial_store, self.name, self.pipeline_version,
            collect_hist=collect_hist,
        )

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _execute_multi(jobs, n_jobs, self.node_store, gpu_id_list=gpu_id_list,
                                        collectors=collectors, tracker=tracker,
                                        log_dir=log_dir)
            else:
                errors = _execute_single(jobs, self.node_store, gpu_id_list=gpu_id_list,
                                         collectors=collectors, tracker=tracker)
        finally:
            tracker.close()

        error_names = sorted({n for _, _, n in errors})
        n_ok = len(jobs) - len(errors)
        if errors:
            logger.info(f"Exp complete: {n_ok}/{len(jobs)} job(s), "
                        f"{len(errors)} error(s): {error_names}")
        else:
            logger.info(f"Exp complete: {len(jobs)} job(s)")

    def _make_jobs(self, trials, trial_store):
        """Expand ``(Trial, outer, inner)`` entries into runnable Jobs.

        A fold is skipped only if ``TrialStore.experiment_hist`` already has
        it recorded as ``'built'`` — a fold recorded as ``'error'``, or with
        no history row at all, gets a job. Whether the Trial's definition
        changed since that history was recorded is not checked here; history
        is the sole source of truth for what still needs to run.

        A fold that does get a job has its NodeStore entry reset first — the
        write on rerun would overwrite the on-disk artifact regardless, but
        without this, a flow's in-memory info cache (populated by an earlier
        ``get_info``/``get_status`` call in this same process) would keep
        returning the stale pre-rerun info even after the new write lands.
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        gpu_cache = {}
        hist_cache = {}
        jobs = []
        for trial, outer_idx, inner_idx in trials:
            flow = self.outer_folds[outer_idx].train_data_flows[inner_idx]
            spec = trial.get_spec()

            if trial.name not in hist_cache:
                hist_cache[trial.name] = trial_store.get_status(trial.name, self.name)
            if hist_cache[trial.name].get((outer_idx, inner_idx)) == 'built':
                continue
            flow.reset_node(trial.name)

            if trial.name not in gpu_cache:
                adapter = resolve_node_adapter(spec.processor, spec.adapter)
                gpu_cache[trial.name] = adapter.get_gpu_usage(spec.params) != GPU_NO

            jobs.append(Job(trial.name, spec, outer_idx, inner_idx, flow,
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
            spec = pipeline.get_node_spec(name)
            processor = spec.processor
            processor_name = getattr(processor, '__name__', processor) if processor else 'None'
            edges_info = ", ".join(
                f"{key}: {dsl_string}" for key, dsl_string in spec.edges.items()
            )
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {spec.method}")
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
            'pipeline_name': self.pipeline_name,
            'pipeline_version': self.pipeline_version,
        })
        self._store.save_splitters(self.name, {
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
        })

    def desc_spec(self):
        return desc_spec(self)
