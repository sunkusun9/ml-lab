import json
import sqlite3
from pathlib import Path

from .._common import utc_now

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS collect_hist (
        collector_name   TEXT NOT NULL,
        node_name        TEXT NOT NULL,
        outer_idx        INTEGER NOT NULL,
        inner_idx        INTEGER NOT NULL,
        pipeline_version INTEGER,
        status           TEXT,
        collect_date     TEXT,
        elapsed          REAL,
        info             TEXT,
        PRIMARY KEY (collector_name, node_name, outer_idx, inner_idx)
    );
"""

_KEY_COLUMNS = ('collector_name', 'node_name', 'outer_idx', 'inner_idx')


class CollectHist:
    """Per-fold outcome of every collect call, for one Experimenter's registry.

    Lives beside the :class:`~mllabs.Collectors` registry that owns it, which
    is one Experimenter's — so a row is identified by
    ``(collector_name, node_name, outer_idx, inner_idx)`` alone and nothing
    here needs an ``experimenter`` column, the same way ``node_hist`` needs no
    ``run_name``.

    It is a log, not a gate: nothing is skipped on the strength of a row here.
    What it does make visible is a fold recorded 'built' before a Collector
    was attached — that fold is skipped without dispatch, so it collects
    nothing, and the absence of a row is the only trace of it.
    """

    def __init__(self, path, name='collect_hist'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def record(self, collector_name, node_name, outer_idx, inner_idx,
               pipeline_version=None, status=None, collect_date=None,
               elapsed=None, info=None):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collect_hist "
                "(collector_name, node_name, outer_idx, inner_idx, "
                "pipeline_version, status, collect_date, elapsed, info) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (collector_name, node_name, outer_idx, inner_idx,
                 pipeline_version, status, collect_date or utc_now(), elapsed,
                 json.dumps(info) if info is not None else None),
            )

    def record_all(self, rows):
        for row in rows:
            self.record(**row)

    def get_hist(self, collector_name=None, node_name=None,
                 status=None, pipeline_version=None):
        where, params = [], []
        for column, value in (('collector_name', collector_name),
                              ('node_name', node_name),
                              ('status', status),
                              ('pipeline_version', pipeline_version)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM collect_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY collector_name, node_name, outer_idx, inner_idx"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            row['info'] = json.loads(row['info']) if row['info'] else None
            result.append(row)
        return result

    def get_status(self, collector_name, node_name=None):
        return {
            (r['node_name'], r['outer_idx'], r['inner_idx']): r['status']
            for r in self.get_hist(collector_name=collector_name, node_name=node_name)
        }

    def get_info(self, collector_name, node_name=None):
        return {
            (r['node_name'], r['outer_idx'], r['inner_idx']): r['info']
            for r in self.get_hist(collector_name=collector_name, node_name=node_name)
        }

    def get_errors(self, collector_name=None):
        return self.get_hist(collector_name=collector_name, status='error')

    def remove_hist(self, collector_name=None, node_name=None):
        where, params = [], []
        for column, value in (('collector_name', collector_name),
                              ('node_name', node_name)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "DELETE FROM collect_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(sql, params)

    def __repr__(self):
        return f"<CollectHist {self.db_path}>"
