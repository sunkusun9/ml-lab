import uuid
import os
import time
import traceback
import warnings
import multiprocessing
from multiprocessing.connection import wait

_mp_ctx = multiprocessing.get_context('spawn')

from ._node_processor import ProgressMonitor
from ._store import NodeStore


def _write_prep_error(node_path, node_attrs, edges, exc):
    """Data-prep (get_train/get_valid/get_test_data) failed before dispatch to a worker.
    Persist an 'error' info like a fit-time failure so the node's state stays consistent
    and the run can move on to the remaining nodes/folds instead of crashing."""
    error = {
        'type': type(exc).__name__,
        'message': str(exc),
        'traceback': traceback.format_exc(),
    }
    info = {
        'build_id': str(uuid.uuid4()),
        'node_serial': node_attrs.get('serial'),
        'fit_time': 0.0,
        'train_shape': None,
        'edges': edges,
        'status': 'error',
        'error': error,
    }
    NodeStore.write_info(node_path, info)
    return error


def _process(node_attrs, train_data, valid_data, fit_process, monitor, gpu_id_list=None, single_worker = True):
    from ._node_processor import TransformProcessor, PredictProcessor
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        obj = TransformProcessor(node_attrs['name'], node_attrs['processor'], node_attrs['adapter'], node_attrs['params'])
    else:
        obj = PredictProcessor(node_attrs['name'], node_attrs['processor'], method, node_attrs['adapter'], node_attrs['params'])

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
                'node_serial': node_attrs.get('serial'),
                'fit_time': time.time() - start_time,
                'train_shape': None,
                'edges': node_attrs.get('edges'),
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
        'node_serial': node_attrs.get('serial'),
        'fit_time': elapsed_time,
        'train_shape': ref_data.get_shape() if ref_data is not None else None,
        'edges': node_attrs.get('edges'),
    }
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    if warn_msgs:
        info['warnings'] = warn_msgs
    return obj, result, info


def _safe_collect_call(collector, context, monitor):
    try:
        return collector.collect(context)
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"[Collector:{collector.name}] collect failed: {type(e).__name__}: {e}\n{tb}"
        monitor.message(msg, typ = 'warning')
        collector.warnings.append({
            'method': 'collect', 'type': type(e).__name__, 'message': str(e), 'traceback': tb,
        })
        return None


def _default_on_collect(collector, node_name, outer_idx, inner_idx, result):
    collector.push(node_name, outer_idx, inner_idx, result)


def _run_collectors(collectors, node_attrs, obj, result, info, train_data, valid_data, test_data, ext_data,
                    outer_idx, inner_idx, monitor, on_collect=_default_on_collect):
    matched = [c for c in collectors if c.connector.match(node_attrs)]
    if not matched:
        return []
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
            'node_attrs': node_attrs,
            'processor': obj,
            'spec': info,
            'input': (train_data, valid_data, test_data),
            'outer_idx': outer_idx,
            'inner_idx': inner_idx,
        }
        if proc_test:
            context['output_test'] = obj.process(test_data) if test_data else None
        if proc_train:
            context['output_valid'] = obj.process(valid_data) if valid_data else None
            context['output_train'] = (result if result is not None else obj.process(train_data))
        for c in matched:
            if ext_data is not None and c.name in ext_data:
                context['output_ext'] = obj.process(ext_data[c.name])
            on_collect(c, node_attrs['name'], outer_idx, inner_idx, _safe_collect_call(c, context, monitor))
    return [f"{w.category.__name__}: {w.message}" for w in caught]



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

    Job tuple: ``(node_path, file, node_attrs, idx, no, train_data, valid_data)``
    Sentinel ``None`` stops the worker.

    Messages sent to main via conn:
        ('progress', current, total, metrics)
        ('warning', msg)
        ('info', msg)
        ('done', info)
        ('error', error_info)
    """

    def __init__(self, conn, collectors, finalize=False, gpu_id=None, log_path=None):
        super().__init__(daemon=True)
        self.conn = conn
        self.collectors = collectors
        self.finalize = finalize
        self.gpu_id = gpu_id
        self.log_path = log_path

    def run(self):
        if self.log_path is not None:
            # Redirect this worker's OS-level stdout/stderr (fd 1/2) so native
            # library chatter (TensorFlow, LightGBM, CatBoost, cuDNN/XLA) is
            # captured here instead of polluting the parent console. Framework
            # results/progress travel over the Pipe and are unaffected. Done
            # before any fit / lazy TF import so their banners are captured too.
            _f = open(self.log_path, 'w')
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
            node_path, node_attrs, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data = job
            node_name = node_attrs['name']
            method = node_attrs['method']
            fit_process = method in ['fit_transform', 'fit_predict']
            obj, result, info = _process(node_attrs, train_data, valid_data, fit_process, monitor, gpu_id_list, single_worker = False)
            for w in info.get('warnings', []):
                logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
            if obj is None:
                NodeStore.write_info(node_path, info)
                self.conn.send(('error', {**info['error'], 'fold': (outer_idx, inner_idx)}))
                continue

            if self.finalize:
                NodeStore.write_info(node_path, {**info, 'status': 'finalized'})
            else:
                NodeStore.write_objs(node_path, obj, result, info)

            def _send_collect(collector, node_name, outer_idx, inner_idx, res):
                self.conn.send(('collect', collector.name, node_name, outer_idx, inner_idx, res))

            collect_warns = _run_collectors(
                self.collectors, node_attrs, obj, result, info, train_data, valid_data, test_data, ext_data,
                outer_idx, inner_idx, monitor, on_collect=_send_collect,
            )
            for w in collect_warns:
                logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
            self.conn.send(('done', info))

# ---------------------------------------------------------------------------
# Flow build
# ---------------------------------------------------------------------------

def _is_stage_ready(flow, pipeline, node_name):
    from ._edge_dsl import referenced_nodes
    for dsl_string in pipeline.get_node_attrs(node_name)['edges'].values():
        for src_name in referenced_nodes(dsl_string):
            if src_name is None:
                continue
            if pipeline.grps[pipeline.nodes[src_name].grp].role == 'stage':
                if src_name not in flow.node_objs:
                    return False
    return True


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

def _build_flow_single(outer_folds, pipeline, nodes, gpu_id_list=None, collectors=None, tracker=None):
    gpu_id_list = gpu_id_list or []
    collectors = collectors or []

    errors = {}
    router = _TrackerRouter(1, tracker)

    if tracker:
        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, flow in enumerate(outer_fold.train_data_flows):
                for n in nodes:
                    if n in flow.node_objs:
                        tracker.done(0, n, outer_idx, inner_idx, None)

    while True:
        ready = [
            (outer_idx, inner_idx, flow, n)
            for outer_idx, outer_fold in enumerate(outer_folds)
            for inner_idx, flow in enumerate(outer_fold.train_data_flows)
            for n in nodes
            if n not in flow.node_objs and (outer_idx, inner_idx, n) not in errors
            and _is_stage_ready(flow, pipeline, n)
        ]
        if not ready:
            break

        for outer_idx, inner_idx, flow, node_name in ready:
            node_attrs = pipeline.get_node_attrs(node_name)
            try:
                train_data = flow.get_train(node_attrs['edges'])
                valid_data = flow.get_valid(node_attrs['edges'])
                test_data = outer_folds[outer_idx].get_test_data(node_attrs['edges'])
                ext_data = {}
                for i in [c for c in collectors if c.connector.match(node_attrs)]:
                    if i.get_properties().get('need_process_data', False):
                        ext_data[i.name] = flow.get_data(i.get_ext_data(), node_attrs['edges'])
            except Exception as e:
                error = _write_prep_error(flow._node_path(node_name), node_attrs, node_attrs['edges'], e)
                errors[(outer_idx, inner_idx, node_name)] = error
                if tracker:
                    tracker.error(0, node_name, outer_idx, inner_idx, error)
                continue
            fit_process = node_attrs['method'] in ['fit_transform', 'fit_predict']

            if tracker:
                tracker.start(0, node_name, outer_idx, inner_idx)
            obj, result, info = _process(node_attrs, train_data, valid_data, fit_process, router, gpu_id_list, single_worker = True)
            for w in info.get('warnings', []):
                if tracker:
                    tracker.message(0, f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')
            if obj is None:
                errors[(outer_idx, inner_idx, node_name)] = info['error']
                NodeStore.write_info(flow._node_path(node_name), info)
                if tracker:
                    tracker.error(0, node_name, outer_idx, inner_idx, info['error'])
                continue

            NodeStore.write_objs(flow._node_path(node_name), obj, result, info)
            flow.set_objs(node_name, obj, result, info)
            if tracker:
                tracker.done(0, node_name, outer_idx, inner_idx, info)

            if collectors:
                collect_warns = _run_collectors(collectors, node_attrs, obj, result, info, train_data, valid_data, test_data, ext_data,
                                                outer_idx, inner_idx, router)
                if tracker:
                    for w in collect_warns:
                        tracker.message(0, f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')

    return errors


def _build_flow_multi(outer_folds, pipeline, nodes, n_jobs, gpu_id_list=None, collectors=None, tracker=None,
                      gpu_fallback_cpu=True, cpu_fallback_gpu=True, log_dir=None):
    from .adapter._base import GPU_NO

    gpu_id_list = gpu_id_list or []
    collectors = collectors or []
    n_gpu = min(len(gpu_id_list), n_jobs)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    flow_map = {
        (outer_idx, inner_idx): flow
        for outer_idx, outer_fold in enumerate(outer_folds)
        for inner_idx, flow in enumerate(outer_fold.train_data_flows)
    }

    def _needs_gpu(node_attrs):
        if not gpu_id_list:
            return False
        adapter = node_attrs.get('adapter')
        return adapter is not None and adapter.get_gpu_usage(node_attrs.get('params')) != GPU_NO

    workers = []  # [(process, parent_conn)]
    for i in range(n_jobs):
        parent_conn, child_conn = _mp_ctx.Pipe()
        w = ProcessWorker(child_conn, collectors, gpu_id=gpu_id_list[i] if i < n_gpu else None,
                          log_path=os.path.join(log_dir, f'worker_{i}.log') if log_dir else None)
        w.start()
        child_conn.close()
        workers.append((w, parent_conn))

    for _, conn in workers:
        conn.recv()  # wait for 'ready'

    if tracker:
        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, flow in enumerate(outer_fold.train_data_flows):
                for n in nodes:
                    if n in flow.node_objs:
                        tracker.done(0, n, outer_idx, inner_idx, None)

    free_gpu = list(range(n_gpu))
    free_cpu = list(range(n_gpu, n_jobs))
    busy = {}   # parent_conn -> (outer_idx, inner_idx, node_name)
    errors = {}
    all_conns = [conn for _, conn in workers]

    def _collect_ready():
        in_flight = set(busy.values())
        gpu_ready, cpu_ready = [], []
        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, flow in enumerate(outer_fold.train_data_flows):
                for n in nodes:
                    key = (outer_idx, inner_idx, n)
                    if n in flow.node_objs or key in errors or key in in_flight:
                        continue
                    if not _is_stage_ready(flow, pipeline, n):
                        continue
                    node_attrs = pipeline.get_node_attrs(n)
                    (gpu_ready if _needs_gpu(node_attrs) else cpu_ready).append(key)
        return gpu_ready, cpu_ready

    def _dispatch(outer_idx, inner_idx, node_name, worker_idx):
        flow = flow_map[(outer_idx, inner_idx)]
        node_attrs = pipeline.get_node_attrs(node_name)
        try:
            train_data = flow.get_train(node_attrs['edges'])
            valid_data = flow.get_valid(node_attrs['edges'])
            test_data = outer_folds[outer_idx].get_test_data(node_attrs['edges'], inner_idx)
            ext_data = {}
            for i in [c for c in collectors if c.connector.match(node_attrs)]:
                if i.get_properties().get('need_process_data', False):
                    ext_data[i.name] = flow.get_data(i.get_ext_data(), node_attrs['edges'])
        except Exception as e:
            error = _write_prep_error(flow._node_path(node_name), node_attrs, node_attrs['edges'], e)
            errors[(outer_idx, inner_idx, node_name)] = error
            if tracker:
                tracker.error(worker_idx, node_name, outer_idx, inner_idx, error)
            return

        _, conn = workers[worker_idx]
        conn.send((flow._node_path(node_name), node_attrs, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data))
        busy[conn] = (outer_idx, inner_idx, node_name)
        (free_gpu if worker_idx < n_gpu else free_cpu).remove(worker_idx)
        if tracker:
            tracker.start(worker_idx, node_name, outer_idx, inner_idx)

    def _try_dispatch():
        gpu_ready, cpu_ready = _collect_ready()
        for outer_idx, inner_idx, node_name in gpu_ready:
            if free_gpu:
                _dispatch(outer_idx, inner_idx, node_name, free_gpu[0])
            elif free_cpu and gpu_fallback_cpu:
                _dispatch(outer_idx, inner_idx, node_name, free_cpu[0])
        for outer_idx, inner_idx, node_name in cpu_ready:
            if free_cpu:
                _dispatch(outer_idx, inner_idx, node_name, free_cpu[0])
            elif free_gpu and cpu_fallback_gpu:
                _dispatch(outer_idx, inner_idx, node_name, free_gpu[0])

    _try_dispatch()

    while busy:
        for conn in wait(all_conns):
            msg_type, *data = conn.recv()
            worker_idx = next(i for i, (_, c) in enumerate(workers) if c is conn)
            outer_idx, inner_idx, node_name = busy[conn]

            if msg_type == 'done':
                info = data[0]
                flow_map[(outer_idx, inner_idx)].load_objs(node_name)
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.done(worker_idx, node_name, outer_idx, inner_idx, info)
                _try_dispatch()

            elif msg_type == 'error':
                error_info = data[0]
                errors[(outer_idx, inner_idx, node_name)] = error_info
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.error(worker_idx, node_name, outer_idx, inner_idx, error_info)
                _try_dispatch()

            elif msg_type == 'collect':
                coll_name, node_name, fi, no, res = data
                for c in collectors:
                    if c.name == coll_name:
                        c.push(node_name, fi, no, res)
                        break

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

def _experiment_single(outer_folds, pipeline, nodes,
                        gpu_id_list=None, collectors=None, tracker=None,
                        finalize=False, include_train=True, include_input=True):
    gpu_id_list = gpu_id_list or []
    collectors = collectors or []
    errors = {}  # {(outer_idx, node_name): error_info}
    monitor = _TrackerRouter(0, tracker)

    for node_name in nodes:
        node_attrs = pipeline.get_node_attrs(node_name)
        fit_process = node_attrs['method'] in ['fit_transform', 'fit_predict']
        matched = [c for c in collectors if c.connector.match(node_attrs)]
        edges = node_attrs['edges']

        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, (train_flow, artifact_store) in enumerate(
                zip(outer_fold.train_data_flows, outer_fold.artifact_stores)
            ):
                if (outer_idx, node_name) in errors:
                    continue
                status = artifact_store.status(node_name)
                if status == 'finalized':
                    if tracker:
                        tracker.done(0, node_name, outer_idx, inner_idx, None)
                    continue
                try:
                    train_data = train_flow.get_train(edges)
                    valid_data = train_flow.get_valid(edges)
                    test_data = outer_fold.get_test_data(edges)
                except Exception as e:
                    error = _write_prep_error(artifact_store._node_path(node_name), node_attrs, edges, e)
                    errors[(outer_idx, node_name)] = {**error, 'fold': (outer_idx, inner_idx)}
                    if tracker:
                        tracker.error(0, node_name, outer_idx, inner_idx, error)
                    for c in matched:
                        c.abort_node(node_name)
                    continue

                if status == 'built':
                    obj, result, info = artifact_store.get_objs(node_name)
                    if tracker:
                        tracker.done(0, node_name, outer_idx, inner_idx, info)
                else:
                    if tracker:
                        tracker.start(0, node_name, outer_idx, inner_idx)
                    obj, result, info = _process(node_attrs, train_data, valid_data, fit_process, monitor, gpu_id_list, single_worker = True)
                    for w in info.get('warnings', []):
                        monitor.message(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')
                    if obj is None:
                        errors[(outer_idx, node_name)] = {**info['error'], 'fold': (outer_idx, inner_idx)}
                        NodeStore.write_info(artifact_store._node_path(node_name), info)
                        if tracker:
                            tracker.error(0, node_name, outer_idx, inner_idx, info['error'])
                        for c in matched:
                            c.abort_node(node_name)
                        continue
                    if finalize:
                        NodeStore.write_info(artifact_store._node_path(node_name), {**info, 'status': 'finalized'})
                    else:
                        NodeStore.write_objs(artifact_store._node_path(node_name), obj, result, info)
                    if tracker:
                        tracker.done(0, node_name, outer_idx, inner_idx, info)

                if matched:
                    ext_data = {}
                    for i in [c for c in collectors if c.connector.match(node_attrs)]:
                        if i.get_properties().get('need_process_data', False):
                            ext_data[i.name] = train_flow.get_data(i.get_ext_data(), node_attrs['edges'])
                    collect_warns = _run_collectors(matched, node_attrs, obj, result, info,
                                    train_data, valid_data, test_data, ext_data,
                                    outer_idx, inner_idx, monitor)
                    if tracker:
                        for w in collect_warns:
                            tracker.message(0, f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}", typ='warning')

    return errors


def _experiment_multi(outer_folds, pipeline, nodes, n_jobs,
                       gpu_id_list=None, collectors=None, tracker=None,
                       finalize=False, gpu_fallback_cpu=True, cpu_fallback_gpu=True, log_dir=None):
    from .adapter._base import GPU_NO

    gpu_id_list = gpu_id_list or []
    collectors = collectors or []
    n_gpu = min(len(gpu_id_list), n_jobs)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    def _needs_gpu(node_attrs):
        if not gpu_id_list:
            return False
        adapter = node_attrs.get('adapter')
        return adapter is not None and adapter.get_gpu_usage(node_attrs.get('params')) != GPU_NO

    workers = []
    for i in range(n_jobs):
        parent_conn, child_conn = _mp_ctx.Pipe()
        w = ProcessWorker(child_conn, collectors,
                          finalize=finalize,
                          gpu_id=gpu_id_list[i] if i < n_gpu else None,
                          log_path=os.path.join(log_dir, f'worker_{i}.log') if log_dir else None)
        w.start()
        child_conn.close()
        workers.append((w, parent_conn))

    for _, conn in workers:
        conn.recv()  # wait for 'ready'

    free_gpu = list(range(n_gpu))
    free_cpu = list(range(n_gpu, n_jobs))
    busy = {}   # conn -> (outer_idx, inner_idx, node_name, outer_fold, train_flow, artifact_store)
    errors = {}  # {(outer_idx, node_name): error_info}
    all_conns = [conn for _, conn in workers]

    def _make_jobs():
        gpu_jobs, cpu_jobs = [], []
        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, (train_flow, artifact_store) in enumerate(
                zip(outer_fold.train_data_flows, outer_fold.artifact_stores)
            ):
                for node_name in nodes:
                    if (outer_idx, node_name) in errors:
                        continue
                    if artifact_store.status(node_name) in ('built', 'finalized'):
                        continue
                    node_attrs = pipeline.get_node_attrs(node_name)
                    job = (outer_idx, inner_idx, node_name, outer_fold, train_flow, artifact_store)
                    (gpu_jobs if _needs_gpu(node_attrs) else cpu_jobs).append(job)
        return gpu_jobs, cpu_jobs

    gpu_jobs, cpu_jobs = _make_jobs()

    if tracker:
        for outer_idx, outer_fold in enumerate(outer_folds):
            for inner_idx, (_, artifact_store) in enumerate(
                zip(outer_fold.train_data_flows, outer_fold.artifact_stores)
            ):
                for node_name in nodes:
                    if artifact_store.status(node_name) in ('built', 'finalized'):
                        tracker.done(0, node_name, outer_idx, inner_idx, None)

    def _dispatch(job, worker_idx):
        outer_idx, inner_idx, node_name, outer_fold, train_flow, artifact_store = job
        node_attrs = pipeline.get_node_attrs(node_name)
        edges = node_attrs['edges']
        try:
            train_data = train_flow.get_train(edges)
            valid_data = train_flow.get_valid(edges)
            test_data = outer_fold.get_test_data(edges)
            ext_data = {}
            for i in [c for c in collectors if c.connector.match(node_attrs)]:
                if i.get_properties().get('need_process_data', False):
                    ext_data[i.name] = train_flow.get_data(i.get_ext_data(), node_attrs['edges'])
        except Exception as e:
            error = _write_prep_error(artifact_store._node_path(node_name), node_attrs, edges, e)
            errors[(outer_idx, node_name)] = error
            if tracker:
                tracker.error(worker_idx, node_name, outer_idx, inner_idx, error)
            for c in collectors:
                if c.connector.match(node_attrs):
                    c.abort_node(node_name)
            gpu_jobs[:] = [j for j in gpu_jobs if not (j[0] == outer_idx and j[2] == node_name)]
            cpu_jobs[:] = [j for j in cpu_jobs if not (j[0] == outer_idx and j[2] == node_name)]
            return False

        node_path = artifact_store._node_path(node_name)
        _, conn = workers[worker_idx]
        conn.send((node_path, node_attrs, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data))
        busy[conn] = job
        (free_gpu if worker_idx < n_gpu else free_cpu).remove(worker_idx)
        if tracker:
            tracker.start(worker_idx, node_name, outer_idx, inner_idx)
        return True

    def _try_dispatch():
        for job in list(gpu_jobs):
            if job not in gpu_jobs:
                continue
            if free_gpu:
                _dispatch(job, free_gpu[0])
            elif free_cpu and len(cpu_jobs) == 0 and gpu_fallback_cpu:
                _dispatch(job, free_cpu[0])
            else:
                break
            if job in gpu_jobs:
                gpu_jobs.remove(job)
        for job in list(cpu_jobs):
            if job not in cpu_jobs:
                continue
            if free_cpu:
                _dispatch(job, free_cpu[0])
            elif free_gpu and len(gpu_jobs) == 0 and cpu_fallback_gpu:
                _dispatch(job, free_gpu[0])
            else:
                break
            if job in cpu_jobs:
                cpu_jobs.remove(job)

    _try_dispatch()

    while busy:
        for conn in wait(all_conns):
            msg_type, *data = conn.recv()
            worker_idx = next(i for i, (_, c) in enumerate(workers) if c is conn)
            outer_idx, inner_idx, node_name, *_ = busy[conn]

            if msg_type == 'done':
                info = data[0]
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.done(worker_idx, node_name, outer_idx, inner_idx, info)
                _try_dispatch()

            elif msg_type == 'error':
                error_info = data[0]
                errors[(outer_idx, node_name)] = error_info
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if tracker:
                    tracker.error(worker_idx, node_name, outer_idx, inner_idx, error_info)
                node_attrs = pipeline.get_node_attrs(node_name)
                for c in collectors:
                    if c.connector.match(node_attrs):
                        c.abort_node(node_name)
                gpu_jobs[:] = [j for j in gpu_jobs if not (j[0] == outer_idx and j[2] == node_name)]
                cpu_jobs[:] = [j for j in cpu_jobs if not (j[0] == outer_idx and j[2] == node_name)]
                _try_dispatch()

            elif msg_type == 'collect':
                coll_name, n, o, i, res = data
                for c in collectors:
                    if c.name == coll_name:
                        c.push(n, o, i, res)
                        break

            elif msg_type == 'progress':
                if tracker:
                    tracker.progress(worker_idx, *data)
            elif msg_type == 'warning':
                if tracker:
                    tracker.message(worker_idx, data[0], typ='warning')
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

