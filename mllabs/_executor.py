import uuid
import os
import gc
import time
import traceback
import warnings
import multiprocessing
from multiprocessing.connection import wait

_mp_ctx = multiprocessing.get_context('spawn')

from ._node_processor import ProgressMonitor
from ._pipeline import _definition_of


def _prep_error_info(edges, exc):
    """Data-prep (get_train/get_valid/get_test_data) failed before dispatch to a worker.

    Builds the same 'error' info shape a fit-time failure produces (see
    ``_process``), so callers can hand it to a tracker's ``error(...)``
    uniformly regardless of where the failure happened. Not persisted here —
    recording belongs to the tracker (``NodeInfoTracker``/
    ``TrialHistTracker``), not ``NodeStore``.
    """
    return {
        'build_id': str(uuid.uuid4()),
        'fit_time': 0.0,
        'train_shape': None,
        'edges': edges,
        'status': 'error',
        'error': {
            'type': type(exc).__name__,
            'message': str(exc),
            'traceback': traceback.format_exc(),
        },
    }


def _process(spec, train_data, valid_data, fit_process, monitor, gpu_id_list=None, single_worker = True):
    from ._node_processor import TransformProcessor, PredictProcessor
    method = spec.method
    if method in ['transform', 'fit_transform']:
        obj = TransformProcessor(spec.name, spec.processor, spec.adapter, spec.params)
    else:
        obj = PredictProcessor(spec.name, spec.processor, method, spec.adapter, spec.params)

    start_time = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if fit_process:
                result = obj.fit_process(train_data, valid_data, gpu_id_list=gpu_id_list, monitor=monitor, single_worker = single_worker)
            else:
                result = None
                obj.fit(train_data, valid_data, gpu_id_list=gpu_id_list, monitor=monitor, single_worker = single_worker)
        except Exception as e:
            warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
            info = {
                'build_id': str(uuid.uuid4()),
                'definition': _definition_of(spec),
                'fit_time': time.time() - start_time,
                'train_shape': None,
                'edges': spec.edges,
                'status': 'error',
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc(),
                },
            }
            if warn_msgs:
                info['warnings'] = warn_msgs
            return None, 'error', info

    elapsed_time = time.time() - start_time
    _ref_key = 'X' if 'X' in train_data else 'y'
    ref_data = train_data[_ref_key]
    info = {
        'build_id': str(uuid.uuid4()),
        'definition': _definition_of(spec),
        'fit_time': elapsed_time,
        'train_shape': ref_data.get_shape() if ref_data is not None else None,
        'edges': spec.edges,
    }
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    if warn_msgs:
        info['warnings'] = warn_msgs
    return obj, result, info


COLLECT_OK = 'collected'
COLLECT_EMPTY = 'empty'
COLLECT_ERROR = 'error'


def _collect_error(collector, phase, exc, node_name, outer_idx, inner_idx, monitor, elapsed=None):
    tb = traceback.format_exc()
    monitor.message(
        f"[Collector:{collector.name}] {phase} failed on {node_name} "
        f"fold {outer_idx}_{inner_idx}: {type(exc).__name__}: {exc}\n{tb}", typ='warning')
    return {
        'collector': collector.name,
        'status': COLLECT_ERROR,
        'elapsed': elapsed,
        'info': {'phase': phase, 'type': type(exc).__name__,
                 'message': str(exc), 'traceback': tb},
    }


def _safe_collect(collector, context, on_collect, node_name, outer_idx, inner_idx,
                  obj, ext_data, monitor):
    if ext_data and collector.name in ext_data:
        try:
            context['output_ext'] = obj.process(ext_data[collector.name])
        except Exception as e:
            return _collect_error(collector, 'ext', e, node_name, outer_idx, inner_idx, monitor)

    started = time.perf_counter()
    try:
        result = collector.collect(context)
    except Exception as e:
        return _collect_error(collector, 'collect', e, node_name, outer_idx, inner_idx,
                              monitor, time.perf_counter() - started)
    elapsed = time.perf_counter() - started

    try:
        on_collect(collector, node_name, outer_idx, inner_idx, result)
    except Exception as e:
        return _collect_error(collector, 'push', e, node_name, outer_idx, inner_idx,
                              monitor, elapsed)

    return {
        'collector': collector.name,
        'status': COLLECT_OK if result is not None else COLLECT_EMPTY,
        'elapsed': elapsed,
        'info': None,
    }


def _default_on_collect(collector, node_name, outer_idx, inner_idx, result):
    collector.push(node_name, outer_idx, inner_idx, result)


def _run_collectors(collectors, spec, obj, result, info, train_data, valid_data, test_data, ext_data,
                    outer_idx, inner_idx, monitor, on_collect=_default_on_collect):
    matched = [c for c in collectors if c.connector.match(spec)]
    if not matched:
        return [], []
    outcomes = []
    # Capture predict/collect-time warnings (e.g. XGBoost device-mismatch on
    # process()) so they flow through the logger channel like fit warnings,
    # instead of leaking raw to stderr.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proc_test = False
        proc_train = False
        for c in matched:
            prop = c.get_properties()
            if prop['need_output_train']:
                proc_train = True
            if prop['need_output_test']:
                proc_test = True
        context = {
            'node_spec': spec,
            'processor': obj,
            'info': info,
            'input': (train_data, valid_data, test_data),
            'outer_idx': outer_idx,
            'inner_idx': inner_idx,
        }
        try:
            if proc_test:
                context['output_test'] = obj.process(test_data) if test_data else None
            if proc_train:
                context['output_valid'] = obj.process(valid_data) if valid_data else None
                context['output_train'] = (result if result is not None else obj.process(train_data))
        except Exception as e:
            outcomes = [_collect_error(c, 'output', e, spec.name, outer_idx, inner_idx, monitor)
                        for c in matched]
        else:
            for c in matched:
                outcomes.append(_safe_collect(c, context, on_collect, spec.name,
                                              outer_idx, inner_idx, obj, ext_data, monitor))
    return [f"{w.category.__name__}: {w.message}" for w in caught], outcomes



class _PipeLogger:
    def __init__(self, conn):
        self._conn = conn

    def adhoc_progress(self, current, total, metrics=None):
        self._conn.send(('progress', current, total, metrics))

    def warning(self, msg):
        self._conn.send(('warning', msg))

    def info(self, msg):
        self._conn.send(('info', msg))

class _ProgressRouter(ProgressMonitor):
    def __init__(self, conn):
        self._conn = conn
    
    def report(self, current, total, metrics=None):
        self._conn.send(('progress', current, total, metrics))

    def message(self, msg, typ='info'):
        pass


class ProcessWorker(_mp_ctx.Process):
    """Process-based worker. Receives jobs via Pipe, reports results/progress back.

    Job tuple: ``(spec, outer_idx, inner_idx, train_data, valid_data,
    test_data, ext_data)``. Sentinel ``None`` stops the worker.

    Messages sent to main via conn:
        ('progress', current, total, metrics)
        ('warning', msg)
        ('info', msg)
        ('done', info)
        ('error', error_info)

    Args:
        store (NodeStore): This run's store, constructed in the parent and
            handed to the worker at spawn — ``NodeStore`` holds nothing but
            a path (no open connections on ``self``), so it pickles across
            the process boundary fine. Used to write the fitted obj/result
            once a job finishes, the same call a single-process run makes.
    """

    def __init__(self, conn, collectors, store, gpu_id=None, log_path=None):
        super().__init__(daemon=True)
        self.conn = conn
        self.collectors = collectors
        self.store = store
        self.gpu_id = gpu_id
        self.log_path = log_path

    def run(self):
        if self.log_path is not None:
            # Redirect this worker's OS-level stdout/stderr (fd 1/2) so native
            # library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) is
            # captured here instead of polluting the parent console. Framework
            # results/progress travel over the Pipe and are unaffected. Done
            # before any fit / lazy TF import so their banners are captured too.
            with open(self.log_path, 'w') as _f:
                os.dup2(_f.fileno(), 1)
                os.dup2(_f.fileno(), 2)
        logger = _PipeLogger(self.conn)
        monitor = _ProgressRouter(self.conn)
        gpu_id_list = [self.gpu_id] if self.gpu_id is not None else []
        self.conn.send(('ready',))
        while True:
            job = self.conn.recv()
            if job is None:
                break
            spec, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data = job
            obj = result = info = None
            try:
                node_name = spec.name
                method = spec.method
                fit_process = method in ['fit_transform', 'fit_predict']
                obj, result, info = _process(spec, train_data, valid_data, fit_process, monitor, gpu_id_list, single_worker = False)
                for w in info.get('warnings', []):
                    logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
                if obj is None:
                    self.conn.send(('error', {**info, 'fold': (outer_idx, inner_idx)}))
                    continue

                self.store.write_objs(node_name, outer_idx, inner_idx, obj, result)

                def _send_collect(collector, node_name, outer_idx, inner_idx, res):
                    self.conn.send(('collect', collector.name, node_name, outer_idx, inner_idx, res))

                collect_warns, outcomes = _run_collectors(
                    self.collectors, spec, obj, result, info, train_data, valid_data, test_data, ext_data,
                    outer_idx, inner_idx, monitor, on_collect=_send_collect,
                )
                for w in collect_warns:
                    logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
                if outcomes:
                    self.conn.send(('collect_hist', node_name, outer_idx, inner_idx, outcomes))
                self.conn.send(('done', info))
            finally:
                # Release this job's inputs and fitted model before blocking on the
                # next recv(). Otherwise the previous fold's data and estimator stay
                # bound (job also holds a second reference) while the next job's data
                # is received, so worker peak = old data + model + new data. Estimators
                # (Keras models, LightGBM callbacks, evals_result_) hold reference
                # cycles, so refcounting alone will not free them here.
                job = spec = train_data = valid_data = test_data = ext_data = None
                obj = result = info = None
                gc.collect()

# ---------------------------------------------------------------------------
# Flow build
# ---------------------------------------------------------------------------

class _TrackerRouter(ProgressMonitor):
    def __init__(self, worker_idx, tracker):
        self.worker_idx = worker_idx
        self._tracker = tracker
    
    def report(self, current, total, metrics=None):
        if self._tracker is not None:
            self._tracker.progress(self.worker_idx, current, total, metrics)

    def message(self, msg, typ='info'):
        if self._tracker is not None:
            self._tracker.message(self.worker_idx, msg, typ)

class Job:
    """One unit of build/experiment work, fully resolved before dispatch.

    A node build and a Trial fit dispatch the same way once name/spec/fold/
    flow/need_gpu are known, so both are this one class. The caller builds
    the list and settles what's already done (``_make_node_jobs`` for nodes,
    ``_make_jobs``/``_make_trial_jobs`` for Trials); the executor never walks
    ``outer_folds``/``pipeline`` or decides what to skip. Nodes still need the
    executor to order dispatch (via ``flow.get_missing_nodes``, since nodes
    can feed each other) — Trials are leaves and don't.

    Attributes:
        name (str): Node/Trial name — also its artifact directory.
        spec (ProcessorSpec): What to build and what to feed it — the same
            shape whether a node or a Trial produced it.
        outer_idx, inner_idx (int): Fold coordinates — also the
            NodeStore/DataCache key.
        flow (TrainDataFlow): Supplies train/valid/test inputs and owns the
            artifact directory for this fold.
        need_gpu (bool): Whether this job's adapter wants a GPU.
    """

    __slots__ = ('name', 'spec', 'outer_idx', 'inner_idx', 'flow', 'need_gpu')

    def __init__(self, name, spec, outer_idx, inner_idx, flow, need_gpu=False):
        self.name = name
        self.spec = spec
        self.outer_idx = outer_idx
        self.inner_idx = inner_idx
        self.flow = flow
        self.need_gpu = need_gpu

    def node_path(self):
        return self.flow.node_path(self.name)

    def __repr__(self):
        return f"<Job {self.name!r} fold=({self.outer_idx}, {self.inner_idx}) gpu={self.need_gpu}>"


def _job_data(job):
    """``(train, valid, test)`` for *job*, raising like any prep failure.

    No ``ext_data``/collectors here — Collectors run against Trials, never
    nodes (see ``Experimenter.build``).
    """
    edges = job.spec.edges
    train_data = job.flow.get_train(edges)
    valid_data = job.flow.get_valid(edges)
    test_data = job.flow.get_test(edges)
    return train_data, valid_data, test_data


def _execute_single(jobs, store, gpu_id_list=None, collectors=None, tracker=None):
    """Run *jobs* to completion, single-process.

    Nodes and Trials dispatch identically — both go through ``_process`` and
    persist the same way — so *collectors* is what tells the two apart:

    - ``None`` is the node/build case. Input prep skips ``ext_data``, nothing
      is matched or run, and the dependency gate below applies.
    - A list (empty is fine — a Trainer has no Collectors) is the Trial case.
      ``[]`` and ``None`` differ only in those three things, so a Trial path
      with no Collectors must still pass ``[]``.

    Jobs are dispatched in dependency order by a ``while True: ready = [...]``
    loop, needed because nodes can feed each other. The ``get_missing_nodes``
    gate inside it is node-only: a Trial's edges only ever reference
    already-built nodes, and gating Trials on it would make one whose node
    never built vanish silently — never dispatched, never an error — instead
    of raising a ``KeyError`` from ``TrainDataFlow._resolve_typ`` that gets
    caught and recorded as a prep error. ``job.flow.set_objs`` runs for every
    completed job regardless of kind; nothing else marks a job done, so
    without it the job would stay in ``ready`` and be redispatched forever.

    *store* is the run's ``NodeStore``, shared by every job's ``flow`` in a
    single call, and is where fitted obj/result are written.

    Returns ``{(outer_idx, inner_idx, name): error_info}`` for every kind of
    job — the key is a job identity, which the ready-loop needs to keep a
    failed job from being redispatched forever.
    """
    gpu_id_list = gpu_id_list or []
    errors = {}  # {(outer_idx, inner_idx, name): error_info}
    router = _TrackerRouter(0, tracker)

    while True:
        ready = [
            job for job in jobs
            if job.name not in job.flow.node_objs
            and (job.outer_idx, job.inner_idx, job.name) not in errors
            and (collectors is not None or not job.flow.get_missing_nodes(job.spec.edges))
        ]
        if not ready:
            break

        for job in ready:
            outer_idx, inner_idx, node_name = job.outer_idx, job.inner_idx, job.name
            matched = [c for c in collectors if c.connector.match(job.spec)] if collectors is not None else []

            try:
                if collectors is not None:
                    train_data, valid_data, test_data, ext_data = _job_inputs(job, collectors)
                else:
                    train_data, valid_data, test_data = _job_data(job)
                    ext_data = {}
            except Exception as e:
                info = _prep_error_info(job.spec.edges, e)
                errors[(outer_idx, inner_idx, node_name)] = info
                if tracker:
                    tracker.error(0, node_name, outer_idx, inner_idx, info)
                for c in matched:
                    c.abort_node(node_name)
                continue

            fit_process = job.spec.method in ['fit_transform', 'fit_predict']
            if tracker:
                tracker.start(0, node_name, outer_idx, inner_idx)
            obj, result, info = _process(job.spec, train_data, valid_data, fit_process,
                                         router, gpu_id_list, single_worker=True)
            for w in info.get('warnings', []):
                router.message(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')
            if obj is None:
                errors[(outer_idx, inner_idx, node_name)] = info
                if tracker:
                    tracker.error(0, node_name, outer_idx, inner_idx, info)
                for c in matched:
                    c.abort_node(node_name)
                continue

            store.write_objs(node_name, outer_idx, inner_idx, obj, result)
            job.flow.set_objs(node_name, obj, result, info)
            if tracker:
                tracker.done(0, node_name, outer_idx, inner_idx, info)

            if matched:
                collect_warns, outcomes = _run_collectors(
                    matched, job.spec, obj, result, info,
                    train_data, valid_data, test_data, ext_data,
                    outer_idx, inner_idx, router,
                )
                for w in collect_warns:
                    router.message(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')
                if tracker:
                    tracker.collect(node_name, outer_idx, inner_idx, outcomes)

    return errors


def _execute_multi(jobs, n_jobs, store, gpu_id_list=None, collectors=None, tracker=None,
                   gpu_fallback_cpu=True, cpu_fallback_gpu=True, log_dir=None):
    """Run *jobs* to completion with a worker pool.

    *collectors* splits node from Trial exactly as in ``_execute_single``:
    ``None`` is the node/build case (workers get an empty collectors list,
    input prep skips ``ext_data``, nothing is matched, and the dependency
    gate applies); a list — empty is fine, a Trainer has no Collectors — is
    the Trial case.

    Readiness is recomputed from scratch every dispatch cycle
    (``_collect_ready``) rather than tracked in mutable lists, because a
    node's readiness changes as the nodes it reads complete. The dependency
    check itself (``flow.get_missing_nodes``) is node-only for the same
    reason as in ``_execute_single``: gating Trials on it would make one
    whose node never built vanish silently instead of being recorded as a
    prep error.

    Worker assignment prefers matching type: a free worker of the "wrong"
    type is handed to a ready job only when nothing of its own type is
    waiting. Because readiness is a fresh snapshot per cycle, a job
    dispatched by the GPU pass isn't visible to the CPU pass's "is anything
    of my own type still waiting" check within that same ``_try_dispatch``
    call — at worst an opportunistic cross-type dispatch is delayed one
    cycle, corrected on the next 'done'/'error' event.

    Returns ``{(outer_idx, inner_idx, name): error_info}``, same shape and
    same reason as ``_execute_single``.
    """
    gpu_id_list = gpu_id_list or []
    n_gpu = min(len(gpu_id_list), n_jobs)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    workers = []  # [(process, parent_conn)]
    for i in range(n_jobs):
        parent_conn, child_conn = _mp_ctx.Pipe()
        # Node jobs (collectors=None) never have Collectors to run, so the
        # worker always gets an empty list in that case.
        w = ProcessWorker(child_conn, collectors or [], store,
                          gpu_id=gpu_id_list[i] if i < n_gpu else None,
                          log_path=os.path.join(log_dir, f'worker_{i}.log') if log_dir else None)
        w.start()
        child_conn.close()
        workers.append((w, parent_conn))

    for _, conn in workers:
        conn.recv()  # wait for 'ready'

    free_gpu = list(range(n_gpu))
    free_cpu = list(range(n_gpu, n_jobs))
    busy = {}         # conn -> Job
    errors = {}       # {(outer_idx, inner_idx, name): error_info}
    push_errors = {}  # {(collector, node, outer_idx, inner_idx): outcome}
    router = _TrackerRouter(0, tracker)
    all_conns = [conn for _, conn in workers]

    def _collect_ready():
        in_flight = {(j.outer_idx, j.inner_idx, j.name) for j in busy.values()}
        gpu_ready, cpu_ready = [], []
        for job in jobs:
            key = (job.outer_idx, job.inner_idx, job.name)
            if job.name in job.flow.node_objs or key in errors or key in in_flight:
                continue
            if collectors is None and job.flow.get_missing_nodes(job.spec.edges):
                continue
            (gpu_ready if job.need_gpu else cpu_ready).append(job)
        return gpu_ready, cpu_ready

    def _dispatch(job, worker_idx):
        matched = [c for c in collectors if c.connector.match(job.spec)] if collectors is not None else []
        try:
            if collectors is not None:
                train_data, valid_data, test_data, ext_data = _job_inputs(job, collectors)
            else:
                train_data, valid_data, test_data = _job_data(job)
                ext_data = {}
        except Exception as e:
            info = _prep_error_info(job.spec.edges, e)
            errors[(job.outer_idx, job.inner_idx, job.name)] = info
            if tracker:
                tracker.error(worker_idx, job.name, job.outer_idx, job.inner_idx, info)
            for c in matched:
                c.abort_node(job.name)
            return

        _, conn = workers[worker_idx]
        conn.send((job.spec, job.outer_idx, job.inner_idx, train_data, valid_data, test_data, ext_data))
        busy[conn] = job
        (free_gpu if worker_idx < n_gpu else free_cpu).remove(worker_idx)
        if tracker:
            tracker.start(worker_idx, job.name, job.outer_idx, job.inner_idx)

    def _try_dispatch():
        gpu_ready, cpu_ready = _collect_ready()
        for job in gpu_ready:
            if free_gpu:
                _dispatch(job, free_gpu[0])
            elif free_cpu and not cpu_ready and gpu_fallback_cpu:
                _dispatch(job, free_cpu[0])
            else:
                break
        for job in cpu_ready:
            if free_cpu:
                _dispatch(job, free_cpu[0])
            elif free_gpu and not gpu_ready and cpu_fallback_gpu:
                _dispatch(job, free_gpu[0])
            else:
                break

    _try_dispatch()

    while busy:
        for conn in wait(all_conns):
            msg_type, *data = conn.recv()
            worker_idx = next(i for i, (_, c) in enumerate(workers) if c is conn)
            job = busy[conn]
            outer_idx, inner_idx, node_name = job.outer_idx, job.inner_idx, job.name

            if msg_type == 'done':
                info = data[0]
                # Read back through *store*, not job.flow's own: the worker
                # wrote where this call was told to write, and a caller can
                # aim the two elsewhere (a Trainer feeds Predictors from the
                # node flow while storing them separately).
                obj, result = store.get_objs(node_name, outer_idx, inner_idx)
                job.flow.set_objs(node_name, obj, result, {'edges': job.spec.edges})
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.done(worker_idx, node_name, outer_idx, inner_idx, info)
                _try_dispatch()

            elif msg_type == 'error':
                info = data[0]
                errors[(outer_idx, inner_idx, node_name)] = info
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.error(worker_idx, node_name, outer_idx, inner_idx, info)
                if collectors is not None:
                    for c in collectors:
                        if c.connector.match(job.spec):
                            c.abort_node(node_name)
                _try_dispatch()

            elif msg_type == 'collect':
                coll_name, n, o, i, res = data
                for c in (collectors or []):
                    if c.name == coll_name:
                        try:
                            c.push(n, o, i, res)
                        except Exception as e:
                            push_errors[(coll_name, n, o, i)] = _collect_error(
                                c, 'push', e, n, o, i, router)
                        break

            elif msg_type == 'collect_hist':
                n, o, i, outcomes = data
                if tracker:
                    tracker.collect(n, o, i, [
                        push_errors.pop((oc['collector'], n, o, i), oc) for oc in outcomes
                    ])

            elif msg_type == 'progress':
                if tracker:
                    tracker.progress(worker_idx, *data)
            elif msg_type == 'warning':
                if tracker:
                    tracker.message(worker_idx, data[0], typ = 'warning')
            elif msg_type == 'info':
                if tracker:
                    tracker.message(worker_idx, data[0])

    for _, conn in workers:
        conn.send(None)
    for w, _ in workers:
        w.join()
    for _, conn in workers:
        conn.close()

    return errors


# ---------------------------------------------------------------------------
# Experiment flow
# ---------------------------------------------------------------------------

def _job_inputs(job, collectors):
    """``(train, valid, test, ext)`` for *job*, raising like any prep failure."""
    edges = job.spec.edges
    train_data = job.flow.get_train(edges)
    valid_data = job.flow.get_valid(edges)
    test_data = job.flow.get_test(edges)
    ext_data = {}
    for c in collectors:
        if c.connector.match(job.spec) and c.get_properties().get('need_process_data', False):
            ext_data[c.name] = job.flow.get_data(c.get_ext_data(), edges)
    return train_data, valid_data, test_data, ext_data


