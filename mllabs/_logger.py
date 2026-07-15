
import sys
import threading
from abc import ABC, abstractmethod


def resolve_logger(logger):
    """Return *logger* if given, else the shared process-wide DefaultLogger."""
    if logger is None:
        return DefaultLogger.get_instance(level=['info', 'progress'])
    return logger


class BaseLogger(ABC):
    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000):
        self._progress = list()
        self.show_info = 'info' in level
        self.show_warning = 'warning' in level
        self.show_progress = 'progress' in level
        self.warning_list = list()
        self.n_store_warning = n_store_warning

    @abstractmethod
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def start_progress(self, session_id, title, total=None):
        pass

    def update_progress(self, session_id, current):
        pass

    def end_progress(self, session_id, current=None):
        pass

    def adhoc_progress(self, session_id, current, total, msg=None):
        pass

    def rename_progress(self, session_id, title):
        pass

    def clear_progress(self, session_id):
        pass

    def create_session(self, position, session_cls=None, **session_kwargs):
        return None

    def remove_session(self, position):
        pass

class BaseProgressSession:
    """Progress display for one position. Standalone — no Logger inheritance."""
    _ansi_managed = True
    def __init__(self, position):
        self._position = position
    @property
    def is_empty(self):
        return True
    def start_progress(self, title, total=None):
        pass
    def update_progress(self, current):
        pass
    def end_progress(self, current=None):
        pass
    def adhoc_progress(self, current, total, msg=None):
        pass
    def rename_progress(self, title):
        pass
    def clear(self):
        pass

class DefaultProgressSession(BaseProgressSession):
    _ansi_managed = True

    def __init__(self, position):
        super().__init__(position)
        self._progress = []
        self._adhoc = None


class TqdmProgressSession(BaseProgressSession):
    _ansi_managed = False

    def __init__(self, position):
        super().__init__(position)
        from tqdm.auto import tqdm as _tqdm_cls
        self._tqdm_cls = _tqdm_cls
        self._bar= self._tqdm_cls(position=self._position, leave=False, total=100)

    def _format_line(self):
        return ''

    def start_progress(self, title, total=100):
        self._bar.n = 0
        self._bar.set_description(title)
        self._bar.reset(total)
        self._bar.refresh()

    def update_progress(self, current):
        self._bar.n = current
        self._bar.refresh()

    def end_progress(self, current=None):
        if current is not None:
            self._bar.n = current
        else:
            self._bar.n = self._bar.total
        self._bar.refresh()

    def adhoc_progress(self, current, total, msg=None):
        if self._bar.total != total:
            self._bar.reset(total)
        self._bar.n = current
        if msg is not None:
            self._bar.set_description(msg)
        self._bar.refresh()

    def rename_progress(self, title):
        self._bar.set_description(title)

    def clear(self):
        self._bar.clear()
        self._bar.close()


class DefaultLogger(BaseLogger):
    """Logger with built-in ANSI multi-session progress display.

    Each session occupies one terminal line. Progress lines are redrawn
    in-place using ANSI cursor movement so info/warning messages scroll
    normally above them.

    Falls back to plain printing when stdout is not a TTY (e.g. Jupyter,
    pipes) — in that case progress is not rendered.

    Since progress rendering owns the terminal cursor (a resource shared
    across the whole process), independent instances writing at the same
    time will corrupt each other's output. Use :meth:`get_instance` to
    share one process-wide instance instead of constructing directly.
    """

    _instance = None

    @classmethod
    def get_instance(cls, level=['info', 'warning', 'progress'], n_store_warning=1000):
        """Return the shared process-wide DefaultLogger, creating it on first use.

        ``level``/``n_store_warning`` only apply to the first call that
        creates the instance — later calls return the same object as-is.
        """
        if cls._instance is None:
            cls._instance = cls(level=level, n_store_warning=n_store_warning)
        return cls._instance

    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000):
        super().__init__(level, n_store_warning)
        self._sessions = {}   # {session_id: state_dict}
        self._n_lines = 0     # lines currently reserved in terminal
        self._lock = threading.Lock()
        self._use_ansi = sys.stdout.isatty()

    def _n_progress_lines(self):
        if not self._sessions or not self.show_progress:
            return 0
        return max(self._sessions.keys()) + 1

    def _format_session(self, sid):
        s = self._sessions.get(sid)
        if s is None:
            return ''
        if s['adhoc']:
            return s['adhoc']
        title = s['title'] or ''
        current = s['current']
        total = s['total']
        if total:
            pct = min(current / total, 1.0)
            bar_w = 20
            filled = int(bar_w * pct)
            bar = '\u2588' * filled + '\u2591' * (bar_w - filled)
            return f"[{bar}] {current}/{total} {title}"
        return f"{title} ({current})" if current else title

    def _redraw(self):
        """Redraw all progress lines. Must be called with _lock held."""
        if not self._use_ansi:
            return
        n_old = self._n_lines
        n_new = self._n_progress_lines()

        if n_old > 0:
            sys.stdout.write(f"\033[{n_old}A")

        for i in range(n_new):
            sys.stdout.write(f"\r\033[2K{self._format_session(i)}\n")

        surplus = n_old - n_new
        for _ in range(surplus):
            sys.stdout.write(f"\r\033[2K\n")
        if surplus > 0:
            sys.stdout.write(f"\033[{surplus}A")

        sys.stdout.flush()
        self._n_lines = n_new

    def _write(self, msg):
        with self._lock:
            n = self._n_lines
            if self._use_ansi and n > 0 and self.show_progress:
                sys.stdout.write(f"\033[{n}A\r\033[2K{msg}\n")
                for i in range(n):
                    sys.stdout.write(f"\r\033[2K{self._format_session(i)}\n")
                sys.stdout.flush()
            else:
                print(msg)

    def info(self, msg):
        if self.show_info:
            self._write(msg)

    def warning(self, msg):
        if self.show_warning:
            self._write(msg)
        self.warning_list.append(msg)
        self.warning_list = self.warning_list[-self.n_store_warning:]

    def create_session(self, session_id, **kwargs):
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {'title': '', 'current': 0, 'total': None, 'adhoc': None}
                self._redraw()

    def remove_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._redraw()

    def start_progress(self, session_id, title, total=100):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update({'title': title, 'total': total, 'current': 0, 'adhoc': None})
                self._redraw()

    def update_progress(self, session_id, current):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['current'] = current
                self._redraw()

    def end_progress(self, session_id, current=None):
        with self._lock:
            if session_id in self._sessions:
                s = self._sessions[session_id]
                s['current'] = current if current is not None else (s['total'] or s['current'])
                s['adhoc'] = None
                self._redraw()

    def adhoc_progress(self, session_id, current, total, msg=None):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update({'current': current, 'total': total, 'adhoc': msg})
                self._redraw()

    def rename_progress(self, session_id, title):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['title'] = title
                self._redraw()

    def clear_progress(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['current'] = 0
                self._redraw()


class ProgressSessionLogger(BaseLogger):
    """Logger that delegates progress display to injected session objects.

    Pass ``session_cls`` to control how each progress slot is rendered
    (e.g. ``TqdmProgressSession``).
    """

    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000, session_cls=None):
        super().__init__(level, n_store_warning)
        self.session_cls = session_cls if session_cls is not None else BaseProgressSession
        self._sessions = {}

    def _write(self, msg):
        print(msg)

    def info(self, msg):
        if self.show_info:
            self._write(msg)

    def warning(self, msg):
        if self.show_warning:
            self._write(msg)
        self.warning_list.append(msg)
        self.warning_list = self.warning_list[-self.n_store_warning:]

    def start_progress(self, session_id, title, total=100):
        if session_id in self._sessions:
            self._sessions[session_id].start_progress(title, total)

    def update_progress(self, session_id, current):
        if session_id in self._sessions:
            self._sessions[session_id].update_progress(current)

    def end_progress(self, session_id, current=None):
        if session_id in self._sessions:
            self._sessions[session_id].end_progress(current)

    def adhoc_progress(self, session_id, current, total, msg=None):
        if session_id in self._sessions:
            self._sessions[session_id].adhoc_progress(current, total, msg)

    def rename_progress(self, session_id, title):
        if session_id in self._sessions:
            self._sessions[session_id].rename_progress(title)

    def clear_progress(self, session_id):
        if session_id in self._sessions:
            self._sessions[session_id].update_progress(0)

    def create_session(self, session_id, **session_kwargs):
        if session_id in self._sessions:
            return self._sessions[session_id]
        session = self.session_cls(session_id, **session_kwargs)
        self._sessions[session_id] = session
        return session

    def remove_session(self, session_id):
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
