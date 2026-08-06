import os
import json
import shutil
import sqlite3
import pickle as pkl
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS node_hist (
        node_name        TEXT NOT NULL,
        outer_idx        INTEGER NOT NULL,
        inner_idx        INTEGER NOT NULL,
        pipeline_version INTEGER,
        status           TEXT,
        info             TEXT,
        PRIMARY KEY (node_name, outer_idx, inner_idx)
    );
"""


class ArtifactStore:
    """Shape shared by ``NodeStore`` and ``TrialStore``.

    Not every method here means something to both:

    - **Artifact** methods (``write_objs``/``write_obj``/``write_result``/
      ``get_objs``/``get_obj``/``get_result``/``list_nodes``/``status``/
      ``reset_node``) — ``NodeStore`` is the only one that overrides these,
      because it's the only one that actually persists obj/result files.
      ``TrialStore`` inherits them unoverridden: a Trial leaves no artifact
      anywhere, so there is no obj/result for it to serve. Calling one of
      these on a ``TrialStore`` raises ``NotImplementedError`` rather than
      the ``AttributeError`` it would raise without this base class.
    - **History** methods (``record``/``get_hist``/``get_status``/
      ``get_info``/``remove_hist``) — both override these for real, each
      against its own table. Declared here only so the shape is documented
      in one place; there's no shared body because ``TrialStore`` keys on
      an extra experimenter name that ``NodeStore`` (already scoped to one
      run) doesn't need, so the signatures aren't identical across
      overrides.

    ``stores_artifacts`` states which half a given store actually implements,
    so a caller can hand the executor whichever store owns that kind of job's
    record and let it work out whether there is anything to persist. It is
    the same split the two groups above describe, in a form code can read.
    """

    stores_artifacts = False

    # -- artifact --------------------------------------------------------

    def write_objs(self, node_name, outer_idx, inner_idx, obj, result):
        raise NotImplementedError

    def write_obj(self, node_name, outer_idx, inner_idx, obj):
        raise NotImplementedError

    def write_result(self, node_name, outer_idx, inner_idx, result):
        raise NotImplementedError

    def get_objs(self, node_name, outer_idx, inner_idx):
        raise NotImplementedError

    def get_obj(self, node_name, outer_idx, inner_idx):
        raise NotImplementedError

    def get_result(self, node_name, outer_idx, inner_idx):
        raise NotImplementedError

    def list_nodes(self, outer_idx, inner_idx):
        raise NotImplementedError

    def status(self, node_name, outer_idx, inner_idx):
        raise NotImplementedError

    def reset_node(self, node_name, outer_idx, inner_idx):
        raise NotImplementedError

    # -- history ----------------------------------------------------------

    def record(self, *args, **kwargs):
        raise NotImplementedError

    def get_hist(self, *args, **kwargs):
        raise NotImplementedError

    def get_status(self, *args, **kwargs):
        raise NotImplementedError

    def get_info(self, *args, **kwargs):
        raise NotImplementedError

    def remove_hist(self, *args, **kwargs):
        raise NotImplementedError


class NodeStore(ArtifactStore):
    """Artifact + run-history store for one run (an Experimenter or Trainer).

    Constructed once per run, at that run's own base path (e.g.
    ``{project}/exp/{name}`` or ``{project}/trainers/{name}``) — a Trainer
    and an Experimenter never share a base path, so nothing here needs a
    ``run_name`` column to disambiguate; a fold is identified purely by
    ``(node_name, outer_idx, inner_idx)`` within that base path.

    Node artifacts live at ``{path}/{outer_idx}/{inner_idx}/{node_name}/``:
      obj.pkl    — processor object
      result.pkl — fit_transform/fit_predict output

    No info.pkl (2026-08-01) — build/error metadata (edges, definition,
    fit_time, ...) lives in this same store's ``node_hist`` table instead
    (formerly the separate NodeInfoStore, now merged in here), recorded by
    NodeInfoTracker/TrialHistTracker rather than written alongside the
    artifact. ``status(...)`` (the artifact query) is derived purely from
    whether ``obj.pkl`` exists; ``get_status``/``get_info`` (the history
    query) read ``node_hist`` and can see ``'error'``, which the artifact
    alone cannot.

    write_* / get_* / status / reset_node all take ``(node_name, outer_idx,
    inner_idx)`` and resolve the path themselves via ``node_path()`` — a
    subprocess worker gets its own ``store`` instance (picklable — no open
    connections held on ``self``), so it calls these the same way the main
    process does. record/get_hist/etc. are the history side.

    Args:
        path: This run's base path.
    """

    stores_artifacts = True

    def __init__(self, path):
        self.path = Path(path)
        self.db_path = self.path / '__node_hist.db'
        self.path.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)

    def node_path(self, node_name, outer_idx, inner_idx):
        return self.path / str(outer_idx) / str(inner_idx) / node_name

    # -------------------------------------------------------------------------
    # Write (resolve node_path via self.node_path(), like get_*/status/reset_node)
    # -------------------------------------------------------------------------

    def write_objs(self, node_name, outer_idx, inner_idx, obj, result):
        node_path = self.node_path(node_name, outer_idx, inner_idx)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'obj.pkl', 'wb') as f:
            pkl.dump(obj, f)
        with open(node_path / 'result.pkl', 'wb') as f:
            pkl.dump(result, f)

    def write_obj(self, node_name, outer_idx, inner_idx, obj):
        node_path = self.node_path(node_name, outer_idx, inner_idx)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'obj.pkl', 'wb') as f:
            pkl.dump(obj, f)

    def write_result(self, node_name, outer_idx, inner_idx, result):
        node_path = self.node_path(node_name, outer_idx, inner_idx)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'result.pkl', 'wb') as f:
            pkl.dump(result, f)

    # -------------------------------------------------------------------------
    # Get / status / reset (instance — resolve their own path)
    # -------------------------------------------------------------------------

    def get_objs(self, node_name, outer_idx, inner_idx):
        node_path = self.node_path(node_name, outer_idx, inner_idx)
        with open(node_path / 'obj.pkl', 'rb') as f:
            obj = pkl.load(f)
        with open(node_path / 'result.pkl', 'rb') as f:
            result = pkl.load(f)
        return obj, result

    def get_obj(self, node_name, outer_idx, inner_idx):
        with open(self.node_path(node_name, outer_idx, inner_idx) / 'obj.pkl', 'rb') as f:
            return pkl.load(f)

    def get_result(self, node_name, outer_idx, inner_idx):
        with open(self.node_path(node_name, outer_idx, inner_idx) / 'result.pkl', 'rb') as f:
            return pkl.load(f)

    def list_nodes(self, outer_idx, inner_idx):
        fold_path = self.path / str(outer_idx) / str(inner_idx)
        if not fold_path.is_dir():
            return []
        return [p.name for p in fold_path.iterdir() if p.is_dir()]

    def status(self, node_name, outer_idx, inner_idx):
        return 'built' if (self.node_path(node_name, outer_idx, inner_idx) / 'obj.pkl').exists() else None

    def reset_node(self, node_name, outer_idx, inner_idx):
        node_path = self.node_path(node_name, outer_idx, inner_idx)
        if node_path.is_dir():
            shutil.rmtree(node_path)

    # -------------------------------------------------------------------------
    # History (was NodeInfoStore) — instance, SQLite-backed
    # -------------------------------------------------------------------------

    def record(self, node_name, outer_idx, inner_idx, pipeline_version=None, status=None, info=None):
        """Upsert one fold's outcome.

        Args:
            node_name (str): Node name — also its artifact directory.
            outer_idx (int), inner_idx (int): Fold coordinates.
            pipeline_version (int, optional): The Pipeline version this ran against.
            status (str, optional): ``'built'`` or ``'error'``.
            info (dict, optional): Everything ``_process()``/``_prep_error_info``
                produced besides ``status`` — ``build_id``,
                ``definition``, ``fit_time``, ``edges``, ``train_shape``,
                ``warnings``, and, on failure, ``error``. JSON-encoded.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO node_hist "
                "(node_name, outer_idx, inner_idx, pipeline_version, status, info) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_name, outer_idx, inner_idx, pipeline_version, status,
                 json.dumps(info) if info is not None else None),
            )

    def get_hist(self, node_name=None, outer_idx=None, inner_idx=None, pipeline_version=None):
        """History rows matching whichever filters are given.

        Each row's ``info`` is decoded back into a dict (``None`` if it was
        never recorded).
        """
        where, params = [], []
        for column, value in (('node_name', node_name),
                              ('outer_idx', outer_idx),
                              ('inner_idx', inner_idx),
                              ('pipeline_version', pipeline_version)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM node_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY node_name, outer_idx, inner_idx"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            row['info'] = json.loads(row['info']) if row['info'] else None
            result.append(row)
        return result

    def get_status(self, node_name):
        """``{(outer_idx, inner_idx): status}`` for one node, from history."""
        return {
            (r['outer_idx'], r['inner_idx']): r['status']
            for r in self.get_hist(node_name=node_name)
        }

    def get_info(self, node_name):
        """``{(outer_idx, inner_idx): info}`` for one node, from history."""
        return {
            (r['outer_idx'], r['inner_idx']): r['info']
            for r in self.get_hist(node_name=node_name)
        }

    def get_fold_info(self, outer_idx, inner_idx):
        """``{node_name: info}`` for every node recorded in one specific fold.

        Used by ``DataFlow`` to recover the edges of an artifact built in an
        earlier process, one query per fold instead of per node.
        """
        return {
            r['node_name']: r['info']
            for r in self.get_hist(outer_idx=outer_idx, inner_idx=inner_idx)
        }

    def remove_hist(self, node_name=None):
        where, params = [], []
        if node_name is not None:
            where.append("node_name = ?")
            params.append(node_name)
        sql = "DELETE FROM node_hist"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(sql, params)

    def __repr__(self):
        return f"<NodeStore {self.path}>"
