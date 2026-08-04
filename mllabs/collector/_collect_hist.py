import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS collect_hist (
        collector_name   TEXT NOT NULL,
        experimenter     TEXT NOT NULL,
        node_name        TEXT NOT NULL,
        outer_idx        INTEGER NOT NULL,
        inner_idx        INTEGER NOT NULL,
        pipeline_version INTEGER,
        status           TEXT,
        collect_date     TEXT,
        elapsed          REAL,
        info             TEXT,
        PRIMARY KEY (collector_name, experimenter, node_name, outer_idx, inner_idx)
    );
    CREATE INDEX IF NOT EXISTS idx_collect_hist_experimenter
        ON collect_hist (experimenter);
"""

_KEY_COLUMNS = ('collector_name', 'experimenter', 'node_name', 'outer_idx', 'inner_idx')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class CollectHist:
    def __init__(self, path, name='collect_hist'):
        self.db_path = Path(path) / f'{name}.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def record(self, collector_name, experimenter, node_name, outer_idx, inner_idx,
               pipeline_version=None, status=None, collect_date=None,
               elapsed=None, info=None):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collect_hist "
                "(collector_name, experimenter, node_name, outer_idx, inner_idx, "
                "pipeline_version, status, collect_date, elapsed, info) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (collector_name, experimenter, node_name, outer_idx, inner_idx,
                 pipeline_version, status, collect_date or utc_now(), elapsed,
                 json.dumps(info) if info is not None else None),
            )

    def record_all(self, rows):
        for row in rows:
            self.record(**row)

    def get_hist(self, collector_name=None, experimenter=None, node_name=None,
                 status=None, pipeline_version=None):
        where, params = [], []
        for column, value in (('collector_name', collector_name),
                              ('experimenter', experimenter),
                              ('node_name', node_name),
                              ('status', status),
                              ('pipeline_version', pipeline_version)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM collect_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY collector_name, experimenter, node_name, outer_idx, inner_idx"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            row['info'] = json.loads(row['info']) if row['info'] else None
            result.append(row)
        return result

    def get_status(self, collector_name, experimenter, node_name=None):
        return {
            (r['node_name'], r['outer_idx'], r['inner_idx']): r['status']
            for r in self.get_hist(collector_name=collector_name,
                                   experimenter=experimenter, node_name=node_name)
        }

    def get_info(self, collector_name, experimenter, node_name=None):
        return {
            (r['node_name'], r['outer_idx'], r['inner_idx']): r['info']
            for r in self.get_hist(collector_name=collector_name,
                                   experimenter=experimenter, node_name=node_name)
        }

    def get_errors(self, experimenter=None, collector_name=None):
        return self.get_hist(collector_name=collector_name, experimenter=experimenter,
                             status='error')

    def remove_hist(self, collector_name=None, experimenter=None, node_name=None):
        where, params = [], []
        for column, value in (('collector_name', collector_name),
                              ('experimenter', experimenter),
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
