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
from ._experimenter_store import ExperimenterStore
from ._describer import desc_spec
from ._logger import resolve_logger
from .collector import Collectors


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
from ._common import resolve_common_status, require_built_pipeline, error_payload
from ._pipeline import Pipeline




class OuterFold:
    """One outer fold: test indices and per-inner-fold TrainDataFlows.

    Every TrainDataFlow of every OuterFold shares the same NodeStore (the
    Experimenter's own — see ``Experimenter.node_store``); only ``outer_idx``/``j``
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

    No ``Project`` dependency, and nothing to inject but a ``cache``: an
    Experimenter owns its own :class:`~mllabs.ExperimenterStore`, built from
    ``path``, and everything needed to reopen it lives in that directory.
    ``Project`` only supplies the path and records the name in its index.

    Constructing is *creating*. It splits the data and writes a fresh state,
    so pointing it at an existing directory starts over rather than
    resuming — :meth:`load_experimenter` is how an existing one comes back.
    A Pipeline is never a constructor argument either way; adopt one with
    :meth:`set_pipeline`, which saves it as this Experimenter's ``pipeline.pkl``.

    Args:
        path: Its own base directory (``{project}/exp/{name}`` when
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
        trial_store (TrialStore, optional): Where this Experimenter reads the
            Trials it is asked to run and records what they did. Injected once,
            like *cache*, rather than re-supplied per call — a project has
            exactly one and it does not change over an Experimenter's life. It
            is not persisted: a ``TrialStore`` is project-level, and an
            Experimenter reopened from its own directory has no way to find one
            that would not amount to guessing at the layout above it.

    Attributes:
        cache (DataCache): Shared LRU cache, or ``None``.
        collectors (Collectors): This Experimenter's own Collector registry, over
            ``{path}/collectors`` — definitions, collected data and
            :class:`~mllabs.CollectHist` alike. It belongs here rather
            than to the project because what a Collector writes is keyed by node
            name and nothing else, so two Experimenters sharing a registry would
            overwrite each other on any Trial name they have in common.
            Constructing the Experimenter restores it, so reopening one
            brings back its Collectors with it.
        pipeline (Pipeline): The adopted Pipeline, kept as this Experimenter's own
            ``pipeline.pkl``. ``pipeline_version`` is recorded too, but only
            as provenance — it names the published version this copy was
            taken from, and is ``None`` for an unpublished draft.

    Note:
        ``build``, ``exp`` and other node-graph-aware methods use
        ``self.pipeline`` — call :meth:`set_pipeline` first.
    """

    def __init__(
            self, path, name, data, data_names = None,
            sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None,
            aug_data=None, cache=None, trial_store=None
        ):
        self.name = name
        self.trial_store = trial_store
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._store = ExperimenterStore(self.path)
        data_native = unwrap(data)
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
        self.collectors = Collectors(self.path / 'collectors')

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
        self.pipeline = Pipeline.empty()
        self._os_log_state = None
        self._save()

    @property
    def pipeline_version(self):
        """The adopted Pipeline's version — ``None`` only for an in-memory build.

        Read off :attr:`pipeline` rather than kept beside it, so there is no
        second copy to fall out of step with it.
        """
        return self.pipeline.version

    @staticmethod
    def load_experimenter(path, data, data_key=None, aug_data=None, cache=None,
                          trial_store=None):
        """Reopen the Experimenter rooted at *path*.

        Everything comes out of that directory — meta and splitters from its
        ``__exp.db``, the Pipeline from its ``pipeline.pkl`` — so no Project
        is involved and no ``pipeline_version`` is resolved.
        The name is the directory's own.

        Args:
            path: The Experimenter's base directory.
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): Must match the saved value, if one was
                given when the Experimenter was created.
            aug_data (optional): External data appended to inner train splits.
            cache (DataCache, optional): Shared LRU cache.
            trial_store (TrialStore, optional): Not saved on disk, so a
                reopened Experimenter has one only if it is given one here.

        Returns:
            Experimenter: The reopened Experimenter.

        Raises:
            KeyError: If *path* holds no saved Experimenter.
            ValueError: If *data_key* does not match the saved value.
        """
        path = Path(path)
        if not ExperimenterStore.stored_at(path):
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
            aug_data=aug_data, cache=cache, trial_store=trial_store,
            **split_kwargs,
        )
        pipeline = store.load_pipeline()
        if pipeline is not None:
            exp.set_pipeline(pipeline)
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

    def set_pipeline(self, pipeline):
        """Adopt an already-loaded Pipeline, published or still a draft.

        Takes the Pipeline object directly rather than a version number —
        this class has no way to load one by version itself (see the class
        docstring); ``Project.add_experimenter()`` resolves that before
        calling this. ``self.pipeline_version`` is read straight off
        *pipeline* (its ``.version``), so it's never tracked as a separate,
        possibly-diverging value.

        Any status is accepted, unlike ``Trainer.set_pipeline``. Experimenting
        against a definition still being edited is the point of experimenting;
        it is training that has to be able to say what it trained on.

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

        Returns:
            Pipeline: *pipeline*, unchanged.
        """
        stale = self.stale_nodes(pipeline)
        if stale:
            self.reset_nodes(stale)
        self.pipeline = pipeline
        self._store.save_pipeline(pipeline)
        self._store.set(self.name, 'pipeline_version', self.pipeline_version)
        return pipeline

    def stale_nodes(self, pipeline):
        """Nodes whose artifacts adopting *pipeline* would drop.

        The answer :meth:`set_pipeline` acts on, available before acting — that
        call resets exactly this list, because it is this method's caller. One
        implementation, so a preview cannot drift from the act it previews.

        Asked before rather than after because adoption *spends* the answer:
        staleness is derived by comparing two Pipelines
        (:meth:`Pipeline.diff_from`) and immediately turned into deletions, so
        afterwards there is no stale artifact left to find. Pass the published
        Pipeline instead of a draft and the same call answers whether this
        Experimenter is behind, and what catching up would cost.

        Nothing is stale while nothing is adopted: the empty Pipeline is the
        absence of a prior definition, not a claim that nothing was built, and
        diffing against it would name every node here.

        Args:
            pipeline (Pipeline): The candidate, already built. Building is what
                mints a version, so build first and then ask — this never
                builds, and so never publishes.

        Returns:
            list[str]: Node names, sorted. Names, not a claim that each has an
            artifact here — ``get_status(name)`` says which ones actually cost
            something to drop.
        """
        require_built_pipeline(pipeline)
        if self.pipeline.is_empty:
            return []
        return sorted(pipeline.diff_from(self.pipeline))

    def _require_trial_store(self):
        if self.trial_store is None:
            raise RuntimeError(
                "No trial_store. An Experimenter made by Project.experimenter() / "
                "load_experimenter() gets the project's; a standalone one takes it "
                "as Experimenter(..., trial_store=...)."
            )
        return self.trial_store

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
        """Return the disk status of a Pipeline node across all folds.

        One :class:`NodeStore` covers this whole Experimenter, every fold sharing the
        same instance (told apart by ``outer_idx``/``inner_idx``). Returns
        the common status if all folds agree, or ``'inconsistent'`` if they
        differ.

        Nodes only. A Trial persists nothing, so it is always ``None`` here
        no matter how many times it has run — its per-fold status lives in
        ``TrialStore.get_status(trial_name, experimenter)``.

        Returns:
            ``'built'``, ``'error'``, ``None`` (init), or ``'inconsistent'``.
        """
        return resolve_common_status(
            status
            for store in self._all_stores()
            if (status := store.status(node_name)) is not None
        )

    def reset_nodes(self, nodes):
        """Reset Pipeline nodes to ``init`` state.

        Removes node objects and clears cache entries for the affected nodes.

        Pipeline nodes only — the artifacts a build produced. A Trial has
        none to remove; :meth:`remove_trial_result` is its counterpart.

        Args:
            nodes (list[str]): Node names to reset.
        """
        for name in nodes:
            for store in self._all_stores():
                store.reset_node(name)

        if self.cache is not None:
            self.cache.clear_nodes(nodes)

    def remove_trial_result(self, name):
        """Drop what this Experimenter has of Trial *name* — its results and its history.

        A Trial leaves no artifact, so its result is whatever its Collectors
        kept, held in this Experimenter's own registry;
        :meth:`Collectors.remove_results` clears that along with the matching
        ``CollectHist`` rows. Its ``experiment_hist`` rows go too, which is what
        makes the next ``exp()`` run the Trial again: folds are skipped on the
        strength of a recorded ``'built'`` status and nothing else, so the
        history is the only thing holding one back.

        Only this Experimenter's rows go — another Experimenter's record of the
        same Trial is its own — and the definition stays, since it belongs to
        the project. Removing that too is ``Project.remove_trial``.

        The history half is skipped when there is no :attr:`trial_store`,
        leaving only the collected results to remove.

        Args:
            name (str): Trial name.
        """
        self.collectors.remove_results(name)
        if self.trial_store is not None:
            self.trial_store.remove_hist(trial_name=name, experimenter=self.name)

    def collect_errors(self, collectors=None):
        """Collection failures recorded in this Experimenter's registry.

        The read half of what :meth:`remove_trial_result` writes: a Collector's
        history is this Experimenter's, so asking the project about it means
        coming through here (``Project.collect_errors``).

        Args:
            collectors (str | list[str], optional): Registered Collector names.
                ``None`` reads every row in the history, including any left by a
                Collector that has since been removed from the registry — the
                failure still happened.

        Returns:
            list[dict]: ``collector_name``, ``node_name``, ``outer_idx``,
            ``inner_idx``, ``pipeline_version``, and the failure merged in flat
            — ``phase`` (``'output'``/``'ext'``/``'collect'``/``'push'``, which
            of the four points broke), ``type``, ``message``, ``traceback``.
            Empty when nothing failed. ``collect_date``/``elapsed`` are not
            here — that is ``collectors.hist.get_errors()``.
        """
        hist = self.collectors.hist
        if collectors is None:
            rows = hist.get_errors()
        else:
            rows = [row for c in self._collectors_named(collectors)
                    for row in hist.get_errors(collector_name=c.name)]
        return [{'collector_name': r['collector_name'],
                 'node_name': r['node_name'],
                 'outer_idx': r['outer_idx'],
                 'inner_idx': r['inner_idx'],
                 'pipeline_version': r['pipeline_version'],
                 **(r['info'] or {})}
                for r in rows]

    def _collectors_named(self, collectors):
        """Registered Collectors for a name or a list of them.

        A bare string is wrapped rather than iterated — ``resolve`` takes a
        list, so passing one name straight through would ask for its letters.
        """
        if isinstance(collectors, str):
            collectors = [collectors]
        return self.collectors.resolve(collectors)

    def uncollected_trials(self, collectors=None):
        """Trials a Collector matches but has no result for, per Collector.

        Answers the one question about this Experimenter that no single store
        can: whether a Trial that *ran* here left the Collector anything.
        `experiment_hist` knows what ran (project-wide, so this Experimenter's
        rows are keyed by its name), `collect_hist` knows what was collected
        (registry-local), and which Collectors even apply comes from matching
        their Connectors against the Trial spec. All three meet here because
        ``trial_store`` is injected and the registry is ours.

        Written by hand the join tends to lose two things. A Trial that has no
        history row at all reads as "collected" unless absence is checked for
        separately, and a Trial that never ran reads as a collection failure —
        which then invites ``remove_trial_result`` on history that was fine.
        So only folds recorded ``'built'`` here are considered, and a Trial is
        reported when any of them lacks a ``'collected'`` row: an ``'error'``, an
        ``'empty'`` (usually a misconfigured ``output_var``) and a missing row
        all fall on the same side, since all three mean nothing was kept.

        A Trial that has not run is not a collection problem — that is
        ``Project.pending_trials``.

        Args:
            collectors (str | list[str], optional): Registered Collector names.
                ``None`` asks about every one of them.

        Returns:
            dict: ``{collector_name: [trial_name, ...]}``, one key per asked-for
            Collector, empty list when it has everything.
        """
        trial_store = self._require_trial_store()
        hist = self.collectors.hist

        collected = {(r['collector_name'], r['node_name'], r['outer_idx'], r['inner_idx'])
                     for r in hist.get_hist(status='collected')}
        built = {}
        for r in trial_store.get_hist(experimenter=self.name, status='built'):
            built.setdefault(r['trial_name'], []).append((r['outer_idx'], r['inner_idx']))

        targets = {c.name for c in self._collectors_named(collectors)}
        result = {name: [] for name in targets}
        for trial in trial_store.list_trials():
            folds = built.get(trial.name)
            if not folds:
                continue
            for c in self.collectors.match(trial.get_spec()):
                if c.name in targets and any(
                        (c.name, trial.name, o, i) not in collected for o, i in folds):
                    result[c.name].append(trial.name)
        return result

    def error_nodes(self, nodes=None):
        """Failed folds of this Experimenter's Pipeline nodes, one dict each.

        Pipeline nodes only, as the name says. Error detail does not live on
        the artifact (see ``NodeStore``) — it is in this Experimenter's own
        ``node_hist`` — so this reads history rather than walking fold
        directories. Trial errors are the project's to report, since that is
        where their history lives: ``Project.error_trials(experimenter=)``.

        Args:
            nodes (list[str], optional): Node names to check. ``None`` checks
                every node recorded here.

        Returns:
            list[dict]: ``node_name``, ``outer_idx``, ``inner_idx``,
            ``pipeline_version`` and the flattened failure (``type``,
            ``message``, ``traceback``). Empty when nothing failed. The rest of
            a node's ``info`` (definition, edges, fit_time…) is not here — that
            is ``node_store.get_hist()``.
        """
        node_set = None if nodes is None else set(nodes)
        return [
            {'node_name': r['node_name'],
             'outer_idx': r['outer_idx'],
             'inner_idx': r['inner_idx'],
             'pipeline_version': r['pipeline_version'],
             **error_payload(r['info'])}
            for r in self.node_store.get_hist()
            if r['status'] == 'error' and (node_set is None or r['node_name'] in node_set)
        ]

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
        pipeline = self.pipeline
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
                                        tracker=tracker, log_dir=log_dir, chained=True)
            else:
                errors = _execute_single(jobs, self.node_store, gpu_id_list=gpu_id_list,
                                         tracker=tracker, chained=True)
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

    def exp(self, trials, collectors=None, n_jobs=1, gpu_id_list=None, logger=None):
        """Run *trials* against the Stage graph and invoke matching Collectors.

        Args:
            trials (list[str]): Trial names, out of :attr:`trial_store`. Each
                runs on every fold here; folds already recorded
                ``'built'`` are dropped, so passing the same names again
                continues an interrupted pass instead of repeating it.
            collectors (list[str], optional): Names to collect with, out of
                this Experimenter's own :attr:`collectors` registry. ``None``
                (default) uses every Collector registered there; ``[]`` collects
                nothing. Outcomes go to that registry's ``hist`` — the history
                belongs to the Experimenter, not to the selection made for one call.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled trials.
            logger: Logger instance. Default: shared ``DefaultLogger.get_instance()``.

        **Names, not Trials.** A Trial belongs to the project, and this reads
        it out of :attr:`trial_store` rather than taking a definition in.
        Running was previously also how a Trial got registered, which put
        authoring inside execution — you could not add one to the project
        without executing it, and an ``exp()`` call could silently redefine a
        name another Experimenter had already used. Authoring is ``Project.set_trial`` now,
        and this only executes what is already there.

        Every fold that actually runs records its outcome in the store as it
        finishes (see :class:`~mllabs._tracker.TrialHistTracker`). Folds
        skipped as already built produce no new row — their result was
        recorded when they ran.

        Raises:
            RuntimeError: If this Experimenter has no ``trial_store``.
            KeyError: If a name is not registered in it.
        """
        from ._executor import _execute_single, _execute_multi
        from ._tracker import LoggerExecuteTracker, TrialHistTracker
        logger = resolve_logger(logger)
        pipeline = self.pipeline
        pipeline.check_data_compatibility(self.data)
        trial_store = self._require_trial_store()

        collectors = self._resolve_collectors(collectors)
        jobs = self._make_jobs(trials)
        if not jobs:
            logger.info("No trials to run")
            return

        logger.info(f"Experimenting {len(jobs)} job(s)")
        for c in collectors:
            c.on_attach(self)
            c._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        n_jobs = min(n_jobs, len(jobs))
        tracker = TrialHistTracker(
            LoggerExecuteTracker(len(jobs), n_jobs, logger),
            trial_store, self.name, self.pipeline_version,
            collect_hist=self.collectors.hist,
        )

        try:
            if n_jobs > 1:
                log_dir = self.path / '__worker_logs' if self._os_log_state is not None else None
                errors = _execute_multi(jobs, n_jobs, trial_store, gpu_id_list=gpu_id_list,
                                        collectors=collectors, tracker=tracker,
                                        log_dir=log_dir)
            else:
                errors = _execute_single(jobs, trial_store, gpu_id_list=gpu_id_list,
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

    def _resolve_collectors(self, names):
        """Collector instances for *names*, out of this Experimenter's own registry.

        Names rather than instances, for the same reason ``processor`` and
        ``adapter`` are string refs: a Collector this registry does not know
        has no place here to write to, and would quietly deposit its
        results outside this Experimenter — which is exactly what giving each
        one its own registry prevents. ``Collectors.resolve`` raises on an unknown name,
        so a typo is not a silent "collected nothing" either.
        """
        if names is not None:
            bad = [n for n in names if not isinstance(n, str)]
            if bad:
                raise TypeError(
                    f"exp(collectors=): expected Collector names registered on "
                    f"this Experimenter, got {[type(b).__name__ for b in bad]}. "
                    f"Register with e.collectors.set_collector(name, ...) and "
                    f"pass the name."
                )
        return self.collectors.resolve(names)

    def _make_jobs(self, trials):
        """Expand Trial names into one Job per fold that still needs running.

        A name means the whole grid: every ``(outer_idx, inner_idx)`` here.
        Fold selection is not something a caller has to spell out —
        ``'built'`` folds are dropped here, so handing the same names back
        after an interrupted pass continues it rather than repeating it.

        A fold is skipped only if ``TrialStore.experiment_hist`` records it as
        ``'built'``; one recorded ``'error'``, or with no row at all, gets a
        job. Whether the definition changed since is not checked, and can no
        longer differ silently: ``Project.set_trial`` refuses to redefine a
        name that has succeeded before. Rerunning one is
        :meth:`remove_trial_result` and nothing else.

        The definition, its spec and its GPU verdict are resolved once per
        name, not once per fold.
        """
        from ._executor import Job
        from .adapter import resolve_node_adapter
        from .adapter._base import GPU_NO

        trial_store = self._require_trial_store()
        folds = [(o, i)
                 for o in range(len(self.outer_folds))
                 for i in range(len(self.outer_folds[o].train_data_flows))]

        jobs = []
        for name in trials:
            if not isinstance(name, str):
                raise TypeError(
                    f"exp(trials=): expected Trial names, got "
                    f"{type(name).__name__}. Register the Trial with "
                    f"project.set_trial(trial) and pass its name."
                )
            trial = trial_store.get_by_name(name)
            if trial is None:
                raise KeyError(
                    f"Trial '{name}' is not registered. Add it with "
                    f"project.set_trial(trial) before running it."
                )
            spec = trial.get_spec()
            adapter = resolve_node_adapter(spec.processor, spec.adapter)
            need_gpu = adapter.get_gpu_usage(spec.params) != GPU_NO
            status = trial_store.get_status(name, self.name)

            for outer_idx, inner_idx in folds:
                if status.get((outer_idx, inner_idx)) == 'built':
                    continue
                flow = self.outer_folds[outer_idx].train_data_flows[inner_idx]
                jobs.append(Job(name, spec, outer_idx, inner_idx, flow,
                                need_gpu=need_gpu))
        return jobs

    def get_train_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_train(edges)

    def get_valid_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_valid(edges)

    def get_test_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].get_test_data(edges, i_idx)

    def get_node_info(self):
        pipeline = self.pipeline
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
        """``(obj, result)`` for one built Pipeline node in one fold.

        Nodes only — a Trial's fitted model is never written anywhere, so
        naming one here raises ``FileNotFoundError``. What a Trial produced
        is whatever its Collectors kept.
        """
        return self.outer_folds[outer_idx].train_data_flows[inner_idx].get_objs(node_name)

    def get_worker_logs(self, worker=None):
        """Native (OS-level stdout/stderr) output captured while OS log
        capture was open (see :meth:`open_os_log`/:meth:`os_log`).

        Multi-worker ``build``/``exp`` (``n_jobs > 1``) redirect each worker's
        stdout/stderr to ``{path}/__worker_logs/worker_{i}.log``, capturing
        native library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) that
        would otherwise pollute the console. This process's own (master)
        output goes to ``worker_logs/master.log``. Each execution overwrites
        the previous.

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
            'pipeline_version': self.pipeline_version,
        })
        self._store.save_splitters(self.name, {
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
        })

    def desc_spec(self):
        return desc_spec(self)
