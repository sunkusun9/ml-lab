class ExecuteTracker:
    def __init__(self, total, n_workers=1):
        self.total = total
        self.records = {}
        self.workers = {i: None for i in range(n_workers)}

    def start(self, worker_idx, node_name, outer_idx, inner_idx):
        self.workers[worker_idx] = {
            'job': (node_name, outer_idx, inner_idx),
            'progress': None,
        }
        self._on_update('start', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx)

    def progress(self, worker_idx, current, total, metrics=None):
        self.workers[worker_idx]['progress'] = (current, total, metrics)
        self._on_update('progress', worker_idx=worker_idx,
                        current=current, total=total, metrics=metrics)

    def done(self, worker_idx, node_name, outer_idx, inner_idx, info):
        self.workers[worker_idx] = None
        self.records[(node_name, outer_idx, inner_idx)] = {
            'status': 'done',
            'fit_time': info.get('fit_time') if info is not None else None,
        }
        self._on_update('done', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx, info=info)

    def error(self, worker_idx, node_name, outer_idx, inner_idx, error_info):
        self.workers[worker_idx] = None
        self.records[(node_name, outer_idx, inner_idx)] = {
            'status': 'error',
            'error': error_info,
        }
        self._on_update('error', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx, error_info=error_info)

    def message(self, worker_idx, msg, typ='info'):
        self._on_update('message', worker_idx=worker_idx, msg=msg, typ=typ)

    def block(self, node_name, outer_idx, inner_idx):
        self.records[(node_name, outer_idx, inner_idx)] = {'status': 'blocked'}
        self._on_update('block', node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx)

    def _on_update(self, event, **kwargs):
        pass

    def close(self):
        pass

    @property
    def n_done(self):
        return sum(1 for r in self.records.values() if r['status'] == 'done')

    @property
    def n_error(self):
        return sum(1 for r in self.records.values() if r['status'] == 'error')

    @property
    def n_blocked(self):
        return sum(1 for r in self.records.values() if r['status'] == 'blocked')

    def get_errors(self):
        return {k: r['error'] for k, r in self.records.items() if r['status'] == 'error'}

    def node_summary(self, node_name):
        counts = {'done': 0, 'error': 0, 'blocked': 0}
        for (n, *_), r in self.records.items():
            if n == node_name:
                counts[r['status']] += 1
        return counts

    def summary(self):
        return [
            {'node': k[0], 'outer_idx': k[1], 'inner_idx': k[2], **r}
            for k, r in self.records.items()
        ]


class LoggerExecuteTracker(ExecuteTracker):
    def __init__(self, total, n_workers, logger):
        super().__init__(total, n_workers)
        self.logger = logger
        self.logger.create_session(0)
        self.logger.start_progress(0, 'tasks', total=total)
        for i in range(n_workers):
            logger.create_session(i + 1) 
        self.n_workers = n_workers

    def _on_update(self, event, **kwargs):
        if event == 'message':
            text = f"[worker {kwargs['worker_idx']}] {kwargs['msg']}"
            if kwargs.get('typ') == 'warning':
                self.logger.warning(text)
            else:
                self.logger.info(text)
            return

        if event == 'start':
            wi = kwargs['worker_idx']
            label = f"[{wi}] {kwargs['node_name']} {kwargs['outer_idx']}_{kwargs['inner_idx']}"
            self.logger.start_progress(wi + 1, label)

        elif event == 'progress':
            wi = kwargs['worker_idx']
            self.logger.adhoc_progress(
                wi + 1, kwargs['current'], kwargs['total'], kwargs.get('metrics')
            )

        elif event in ('done', 'error'):
            self.logger.end_progress(kwargs['worker_idx'] + 1)
            self.logger.update_progress(0, self.n_done + self.n_error + self.n_blocked)

        elif event == 'block':
            self.logger.update_progress(0, self.n_done + self.n_error + self.n_blocked)

    def close(self):
        self.logger.remove_session(0)
        for session in range(self.n_workers):
            self.logger.remove_session(session + 1)
            


class TrialHistTracker(ExecuteTracker):
    """Records Trial run history while delegating display to another tracker.

    Wrapping the tracker puts the recording where the outcome actually is:
    ``done``/``error`` fire once per (trial, fold) with the real result, in the
    parent process, so multi-worker runs are covered without the executor
    knowing anything about history. The alternative — re-reading status off
    disk after the run — cannot tell a fold that just ran from one that was
    already built, and duplicates work the tracker has done anyway.

    Args:
        tracker (ExecuteTracker): The display/logging tracker to delegate to.
        store (TrialStore): Where history is written.
        experimenter (str): Experimenter name — half of the history key.
        pipeline_version (int): The Experimenter's ``pipeline_version`` for the run.
    """

    def __init__(self, tracker, store, experimenter, pipeline_version):
        super().__init__(tracker.total, len(tracker.workers))
        self._tracker = tracker
        self._store = store
        self._experimenter = experimenter
        self._pipeline_version = pipeline_version

    def _record(self, node_name, outer_idx, inner_idx, status):
        self._store.record(
            node_name, self._experimenter, outer_idx, inner_idx,
            pipeline_version=self._pipeline_version,
            status=status,
        )

    def start(self, worker_idx, node_name, outer_idx, inner_idx):
        super().start(worker_idx, node_name, outer_idx, inner_idx)
        self._tracker.start(worker_idx, node_name, outer_idx, inner_idx)

    def progress(self, worker_idx, current, total, metrics=None):
        super().progress(worker_idx, current, total, metrics)
        self._tracker.progress(worker_idx, current, total, metrics)

    def done(self, worker_idx, node_name, outer_idx, inner_idx, info):
        super().done(worker_idx, node_name, outer_idx, inner_idx, info)
        self._record(node_name, outer_idx, inner_idx, 'built')
        self._tracker.done(worker_idx, node_name, outer_idx, inner_idx, info)

    def error(self, worker_idx, node_name, outer_idx, inner_idx, error_info):
        super().error(worker_idx, node_name, outer_idx, inner_idx, error_info)
        self._record(node_name, outer_idx, inner_idx, 'error')
        self._tracker.error(worker_idx, node_name, outer_idx, inner_idx, error_info)

    def message(self, worker_idx, msg, typ='info'):
        super().message(worker_idx, msg, typ)
        self._tracker.message(worker_idx, msg, typ)

    def block(self, node_name, outer_idx, inner_idx):
        super().block(node_name, outer_idx, inner_idx)
        self._tracker.block(node_name, outer_idx, inner_idx)

    def close(self):
        self._tracker.close()
