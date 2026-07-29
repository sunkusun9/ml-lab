import re
import sqlite3

import numpy as np
import pandas as pd

from ._base import Collector
from .._edge_dsl import parse, eval_expr


class ProbToLabel:
    def __init__(self, metric_func, var, thresholds=None):
        self.metric_func = metric_func
        self.var = var
        self.thresholds = thresholds
        self._classes = None

    def _normalize_var(self):
        v = self.var
        if isinstance(v, str):
            return [(None, v)]
        if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], tuple):
            return [v]
        return v  # already list

    def on_attach(self, experimenter):
        edges = {'_y': self._normalize_var()}
        data_dict = experimenter.get_test_data(edges, o_idx=0, i_idx=0)
        y_arr = data_dict['_y'].to_array().ravel()
        # np.unique returns sorted order — matches predict_proba column order
        self._classes = np.unique(y_arr)

    def _convert(self, y_prob):
        y_prob = np.asarray(y_prob)
        n_classes = len(self._classes)

        if n_classes == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            threshold = self.thresholds if isinstance(self.thresholds, (int, float)) else 0.5
            indices = (y_prob >= threshold).astype(int)
        else:
            if self.thresholds is not None:
                thresholds = np.asarray(self.thresholds)  # shape (n_classes,)
                above = y_prob >= thresholds
                masked = np.where(above, y_prob, -np.inf)
                indices = np.where(
                    above.any(axis=1),
                    np.argmax(masked, axis=1),
                    np.argmax(y_prob, axis=1),
                )
            else:
                indices = np.argmax(y_prob, axis=1)

        return self._classes[indices]

    def __call__(self, y_true, y_prob):
        return self.metric_func(y_true, self._convert(y_prob))


class MetricCollector(Collector):
    def __init__(self, name, connector, output_var, metric_func, include_train=False):
        super().__init__(name, connector)
        self.output_var = output_var
        self.metric_func = metric_func
        self.include_train = include_train

    def _on_attach(self, experimenter):
        if hasattr(self.metric_func, 'on_attach'):
            self.metric_func.on_attach(experimenter)

    def collect(self, context):
        output_test = context['output_test']
        if self.output_var is None:
            cols = output_test.get_columns()
        else:
            cols = eval_expr(parse(self.output_var), output_test, processor=context['processor'])
        if len(cols) == 0:
            return None

        prd_test = output_test.select_columns(cols)
        result = {'test': self.metric_func(context['input'][2]['y'].data, prd_test.data)}

        if self.include_train and context.get('output_train') is not None:
            result['train'] = self.metric_func(
                context['input'][0]['y'].data, context['output_train'].select_columns(cols).data
            )
            if context['output_valid'] is not None:
                result['valid'] = self.metric_func(
                    context['input'][1]['y'].data, context['output_valid'].select_columns(cols).data
                )

        return result

    @property
    def _db_path(self):
        return self.path / 'metrics.db'

    def _get_conn(self):
        self.path.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                node TEXT, idx INTEGER, inner_idx INTEGER,
                split TEXT, value REAL,
                PRIMARY KEY (node, idx, inner_idx, split)
            )
        """)
        return conn

    def push(self, node, outer_idx, inner_idx, result):
        if result is not None and self.path is not None:
            rows = [(node, outer_idx, inner_idx, split, float(value))
                    for split, value in result.items()]
            with self._get_conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO metrics "
                    "(node, idx, inner_idx, split, value) VALUES (?,?,?,?,?)",
                    rows
                )

    def _load_results(self, node):
        if self.path is None or not self._db_path.exists():
            return None
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT idx, inner_idx, split, value FROM metrics WHERE node = ?", (node,)
            ).fetchall()
        if not rows:
            return None
        result = {}
        for idx, inner_idx, split, value in rows:
            result.setdefault((idx, inner_idx), {})[split] = value
        return result

    def has_node(self, node):
        if self.path is None or not self._db_path.exists():
            return False
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM metrics WHERE node = ? LIMIT 1", (node,)
            ).fetchone()
        return row is not None

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)
        if self.path is not None and self._db_path.exists():
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    f"DELETE FROM metrics WHERE node IN ({','.join('?' * len(nodes))})",
                    list(nodes)
                )

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _get_saved_nodes(self):
        if self.path is None or not self._db_path.exists():
            return []
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT DISTINCT node FROM metrics").fetchall()
        return [r[0] for r in rows]

    def get_metric(self, node):
        data = self._load_results(node)
        if data is None:
            return None
        outer_idxs = sorted(set(k[0] for k in data.keys()))
        l = []
        for idx in outer_idxs:
            sub = [data[(idx, inner_idx)] for inner_idx in sorted(
                k[1] for k in data if k[0] == idx
            )]
            l.append(
                pd.concat([pd.Series(j, name=str(no)) for no, j in enumerate(sub)], axis=1).unstack()
            )
        return pd.concat(l, axis=1).unstack(level=[0, 1]).rename(node)

    def get_metrics(self, nodes=None):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        results = [self.get_metric(node) for node in node_names]
        results = [r for r in results if r is not None]
        if not results:
            return None
        return pd.concat(results, axis=1).T

    def get_metrics_agg(self, nodes=None, inner_fold=True, outer_fold=True, include_std=False):
        if outer_fold and not inner_fold:
            raise ValueError("")
        df = self.get_metrics(nodes)
        if inner_fold:
            df_agg_mean = df.stack(level=1, future_stack=True).groupby(level=0).mean()
            if include_std:
                df_agg_std = df.stack(level=1, future_stack=True).groupby(level=0).std()
            else:
                df_agg_std = None
            if outer_fold:
                if include_std:
                    df_agg_std = df_agg_mean.stack(level=0, future_stack=True).groupby(level=0).std()
                df_agg_mean = df_agg_mean.stack(level=0, future_stack=True).groupby(level=0).mean()
            return df_agg_mean, df_agg_std
        return df

    def get_properties(self):
        return {
            'need_output_train': self.include_train,
            'need_output_test': True,
            'need_process_data': False,
        }