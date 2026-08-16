import uuid
import os
import gc
import time
import traceback
import warnings
import multiprocessing
from multiprocessing.connection import wait

_mp_ctx = multiprocessing.get_context('spawn')

_JOIN_TIMEOUT = 10

from ._node_processor import ProgressMonitor
from ._pipeline import _definition_of
from ._resolver import Resolver


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


def _worker_lost_info(edges, message):
    """A worker died with a job in flight, or none was left to take one.

    Same shape as :func:`_prep_error_info` — there is no exception to format
    here, because the failure is the *absence* of the process that would have
    raised one.
    """
    return {
        'build_id': str(uuid.uuid4()),
        'fit_time': 0.0,
        'train_shape': None,
        'edges': edges,
        'status': 'error',
        'error': {
            'type': 'WorkerLost',
            'message': message,
            'traceback': None,
        },
    }


def _process(spec, train_data, valid_data, fit_process, monitor, gpu_id_list=None,
             single_worker = True, resolver=None):
    from ._node_processor import TransformProcessor, PredictProcessor
    method = spec.method

    start_time = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # Resolving processor/adapter/params (an '@ext:name' reference,
            # say) can fail the same way fit can — caught here too, not just
            # construction — so it lands as this job's 'error' info instead
            # of an uncaught exception out of build()/exp()/train(). adapter
            # goes through Resolver.instance() same as a spec'd one would —
            # resolve_instance(None) is a passthrough, so TransformProcessor/
            # PredictProcessor's own resolve_node_adapter(transformer, None)
            # still does the default-lookup-by-class-name it always did;
            # when a spec was given, that same call becomes a harmless
            # no-op re-resolve of an already-live instance.
            _resolver = resolver or Resolver()
            processor = _resolver.processor(spec.processor)
            adapter = _resolver.instance(spec.adapter)
            params = _resolver.params(spec.params)
            if method in ['transform', 'fit_transform']:
                obj = TransformProcessor(spec.name, processor, adapter, params)
            else:
                obj = PredictProcessor(spec.name, processor, method, adapter, params)

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
        store (NodeStore): The store owning this job kind's records,
            constructed in the parent and handed to the worker at spawn —
            ``NodeStore`` holds nothing but a path (no open connections on
            ``self``), so it pickles across the process boundary fine. Used to
            write the fitted obj/result once a job finishes, the same call
            single-process execution makes.
        resolver (Resolver, optional): Resolves a job's spec params at
            Processor-construction time — see ``_process``. Holds at most an
            ``ExtDataProvider`` (just a path), so it pickles across the
            process boundary the same way ``store`` does. ``None`` falls back
            to a bare ``Resolver()`` inside ``_process`` — no ``ext_data``,
            so an ``'@ext:name'`` param would raise there.
    """

    def __init__(self, conn, collectors, store, gpu_id=None, log_path=None, resolver=None):
        super().__init__(daemon=True)
        self.conn = conn
        self.collectors = collectors
        self.store = store
        self.gpu_id = gpu_id
        self.resolver = resolver
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
                obj, result, info = _process(spec, train_data, valid_data, fit_process, monitor, gpu_id_list, single_worker = False, resolver=self.resolver)
                for w in info.get('warnings', []):
                    logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
                if obj is None:
                    self.conn.send(('error', {**info, 'fold': (outer_idx, inner_idx)}))
                    continue

                if self.store.stores_artifacts:
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

    A node build, a Trial fit and a Predictor fit dispatch the same way once
    name/spec/fold/flow/need_gpu are known, so all three are this one class.
    The caller builds the list and settles what's already done
    (``_make_node_jobs`` for nodes, ``Experimenter._make_jobs`` for Trials,
    ``Trainer._make_predictor_jobs`` for Predictors); the executor never
    walks ``outer_folds``/``pipeline`` or decides what to skip. Nodes still
    need the executor to order dispatch (via ``flow.get_missing_nodes``,
    since nodes can feed each other) — Trials and Predictors are leaves and
    don't, which is what the executor's ``chained`` argument says.

    Attributes:
        name (str): Node/Trial/Predictor name.
        spec (ProcessorSpec): What to build and what to feed it — the same
            shape whichever of the three produced it.
        outer_idx, inner_idx (int): Fold coordinates — also the key under
            which a persisted artifact is stored.
        flow (TrainDataFlow): Supplies train/valid/test inputs. Where the
            outcome goes, if anywhere, is the executor's *store* — not
            necessarily this flow's own.
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


def _execute_single(jobs, store, gpu_id_list=None, collectors=None, tracker=None,
                    chained=False, resolver=None):
    """Run *jobs* to completion, single-process.

    Every kind of job dispatches identically through ``_process``; what the
    caller varies is where the outcome goes.

    *store* is whichever store owns this kind of job's record — a
    ``NodeStore`` for nodes and Predictors, the ``TrialStore`` for Trials.
    Fitted obj/result are written only if that store keeps artifacts at all
    (``ArtifactStore.stores_artifacts``); a Trial leaves none, so its store
    says so and nothing is persisted.

    *chained* says these jobs feed each other through ``job.flow`` — true
    only for Pipeline nodes. A job then waits until everything its edges
    reference is built (``get_missing_nodes``). Leaf jobs (Trials,
    Predictors) don't: a leaf whose node never built must raise a
    ``KeyError`` out of the flow, caught and recorded as a prep error — under
    the gate it would instead vanish silently, never dispatched and never an
    error.

    Nothing is handed to the flow after a job finishes. The artifact on disk
    is what the next job reads, and the flow loads it when asked. That makes
    ``store.stores_artifacts`` load-bearing wherever *chained* is set — which
    holds, since chained means Pipeline nodes and those go to a ``NodeStore``.

    *collectors* is separate, and only about Collectors: ``None`` skips
    ``ext_data`` prep and matching entirely (nodes never have Collectors), a
    list runs them (``[]`` is a list with nothing to match).

    *resolver* (``Resolver``, optional) resolves a job's spec params at
    Processor-construction time — passed straight to ``_process``. ``None``
    falls back to a bare ``Resolver()`` there.

    Returns ``{(outer_idx, inner_idx, name): error_info}`` for every kind of
    job — the key is a job identity, which the ready-loop needs to keep a
    failed job from being redispatched forever.
    """
    gpu_id_list = gpu_id_list or []
    errors = {}  # {(outer_idx, inner_idx, name): error_info}
    done = set()  # {(outer_idx, inner_idx, name)} — completed, don't redispatch
    router = _TrackerRouter(0, tracker)

    while True:
        ready = [
            job for job in jobs
            if (job.outer_idx, job.inner_idx, job.name) not in done
            and (job.outer_idx, job.inner_idx, job.name) not in errors
            and (not chained or not job.flow.get_missing_nodes(job.spec.edges))
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
                                         router, gpu_id_list, single_worker=True, resolver=resolver)
            for w in info.get('warnings', []):
                router.message(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')
            if obj is None:
                errors[(outer_idx, inner_idx, node_name)] = info
                if tracker:
                    tracker.error(0, node_name, outer_idx, inner_idx, info)
                for c in matched:
                    c.abort_node(node_name)
                continue

            if store.stores_artifacts:
                store.write_objs(node_name, outer_idx, inner_idx, obj, result)
            done.add((outer_idx, inner_idx, node_name))
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
                   gpu_fallback_cpu=True, cpu_fallback_gpu=True, log_dir=None,
                   chained=False, resolver=None):
    """Run *jobs* to completion with a worker pool.

    *store*, *chained*, *collectors* and *resolver* mean exactly what they do
    in ``_execute_single``: the store that owns this kind of job's record
    (written to only if it keeps artifacts), whether these jobs feed each
    other through ``job.flow``, whether Collectors run, and what resolves a
    job's spec params at Processor-construction time. *resolver* is handed to
    each ``ProcessWorker`` at spawn, same as *store*.

    Readiness is recomputed from scratch every dispatch cycle
    (``_collect_ready``) rather than tracked in mutable lists, because a
    node's readiness changes as the nodes it reads complete. The dependency
    check itself (``flow.get_missing_nodes``) applies only when *chained*,
    for the same reason as in ``_execute_single``: gating a leaf job on it
    would make one whose node never built vanish silently instead of being
    recorded as a prep error.

    A finished job's artifact is never read back here. The parent only
    orchestrates; reading each worker's obj/result to hand to the flow made it
    accumulate every fitted model and every intermediate output of the whole
    execution, for the sake of a load the next job can do itself.

    Worker assignment prefers matching type: a free worker of the "wrong"
    type is handed to a ready job only when nothing of its own type is
    waiting. Because readiness is a fresh snapshot per cycle, a job
    dispatched by the GPU pass isn't visible to the CPU pass's "is anything
    of my own type still waiting" check within that same ``_try_dispatch``
    call — at worst an opportunistic cross-type dispatch is delayed one
    cycle, corrected on the next 'done'/'error' event.

    A worker dying is a normal outcome here, not an exception. ``wait()``
    reports a closed pipe as readable, so a worker killed by the OOM killer or
    a segfault in a native library surfaces as ``EOFError`` out of ``recv()``.
    That used to escape the dispatch loop entirely, which meant the shutdown
    below never ran and every *other* worker stayed blocked in its own
    ``recv()`` forever — daemon processes outlive nothing when the parent is a
    notebook kernel that never exits, so they sat there holding their memory
    and, for GPU workers, their CUDA context. The job in flight is recorded as
    a ``WorkerLost`` error instead, the connection leaves ``all_conns`` (EOF is
    a persistent state — polling it again would spin), and execution continues on
    whatever workers are left. If none are, the jobs that never got to run are
    recorded too, so the caller's ``len(jobs) - len(errors)`` still counts.

    Shutdown is in a ``finally`` for the same reason: the loop has several ways
    out that are not the normal one (a history write, a Collector push, a
    tracker call), and any of them reaching the end of the function decided
    whether the pool got cleaned up.

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
                          log_path=os.path.join(log_dir, f'worker_{i}.log') if log_dir else None,
                          resolver=resolver)
        w.start()
        child_conn.close()
        workers.append((w, parent_conn))

    free_gpu = list(range(n_gpu))
    free_cpu = list(range(n_gpu, n_jobs))
    busy = {}         # conn -> Job
    errors = {}       # {(outer_idx, inner_idx, name): error_info}
    done = set()      # {(outer_idx, inner_idx, name)} — completed, don't redispatch
    push_errors = {}  # {(collector, node, outer_idx, inner_idx): outcome}
    router = _TrackerRouter(0, tracker)
    all_conns = [conn for _, conn in workers]

    def _teardown():
        for _, conn in workers:
            try:
                conn.send(None)
            except Exception:
                pass
        for w, _ in workers:
            w.join(timeout=_JOIN_TIMEOUT)
            if w.is_alive():
                w.terminate()
                w.join(timeout=_JOIN_TIMEOUT)
        for _, conn in workers:
            try:
                conn.close()
            except Exception:
                pass

    def _worker_died(worker_idx, conn, reason):
        if conn in all_conns:
            all_conns.remove(conn)
        for pool in (free_gpu, free_cpu):
            if worker_idx in pool:
                pool.remove(worker_idx)
        job = busy.pop(conn, None)
        if job is None:
            return
        info = _worker_lost_info(job.spec.edges, f"worker {worker_idx} {reason}")
        errors[(job.outer_idx, job.inner_idx, job.name)] = info
        if tracker:
            tracker.error(worker_idx, job.name, job.outer_idx, job.inner_idx, info)
        if collectors is not None:
            for c in collectors:
                if c.connector.match(job.spec):
                    c.abort_node(job.name)

    def _collect_ready():
        in_flight = {(j.outer_idx, j.inner_idx, j.name) for j in busy.values()}
        gpu_ready, cpu_ready = [], []
        for job in jobs:
            key = (job.outer_idx, job.inner_idx, job.name)
            if key in done or key in errors or key in in_flight:
                continue
            if chained and job.flow.get_missing_nodes(job.spec.edges):
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
        busy[conn] = job
        try:
            conn.send((job.spec, job.outer_idx, job.inner_idx,
                       train_data, valid_data, test_data, ext_data))
        except (EOFError, OSError):
            _worker_died(worker_idx, conn, 'died before its job could be sent')
            return
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

    try:
        for worker_idx, (_, conn) in enumerate(workers):
            try:
                conn.recv()  # wait for 'ready'
            except (EOFError, OSError):
                _worker_died(worker_idx, conn, 'died before it was ready')

        _try_dispatch()

        while busy and all_conns:
            for conn in wait(all_conns):
                worker_idx = next(i for i, (_, c) in enumerate(workers) if c is conn)
                try:
                    msg_type, *data = conn.recv()
                except (EOFError, OSError):
                    _worker_died(worker_idx, conn, 'died with a job in flight')
                    _try_dispatch()
                    continue
                job = busy.get(conn)

                if msg_type in ('done', 'error') and job is None:
                    continue
                if job is not None:
                    outer_idx, inner_idx, node_name = job.outer_idx, job.inner_idx, job.name

                if msg_type == 'done':
                    info = data[0]
                    done.add((outer_idx, inner_idx, node_name))
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

        if not all_conns:
            for job in jobs:
                key = (job.outer_idx, job.inner_idx, job.name)
                if key in errors or key in done:
                    continue
                errors[key] = _worker_lost_info(job.spec.edges, 'no worker left to run it')
                if tracker:
                    tracker.error(0, job.name, job.outer_idx, job.inner_idx, errors[key])
    finally:
        _teardown()

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


