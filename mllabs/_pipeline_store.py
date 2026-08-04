"""SQLite persistence for Pipeline groups, nodes, and datasource.

All table/schema knowledge lives here. Pipeline deals only with in-memory
_PipelineGroup/_PipelineNode/_DataSourceNode objects and plain dicts returned
by fetch_all(); it does not know column names or table structure.
"""
import json
import pickle as pkl
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    CREATE TABLE IF NOT EXISTS grps (
        name TEXT PRIMARY KEY,
        processor TEXT,
        edges TEXT,
        method TEXT,
        parent TEXT,
        adapter TEXT,
        params TEXT,
        desc TEXT
    );
    CREATE TABLE IF NOT EXISTS nodes (
        name TEXT PRIMARY KEY,
        grp TEXT NOT NULL,
        processor TEXT,
        edges TEXT,
        method TEXT,
        adapter TEXT,
        params TEXT,
        desc TEXT
    );
    CREATE TABLE IF NOT EXISTS datasource (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        schema TEXT NOT NULL,
        targets TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS versions (
        version INTEGER PRIMARY KEY,
        path TEXT NOT NULL
    );
"""


class PipelineStore:
    """SQLite-backed persistence for a single Pipeline's ``{name}.db`` file."""

    def __init__(self, path, name='pipeline'):
        self.db_path = Path(path) / f'{name}.db'

    def exists(self):
        return self.db_path.exists()

    def initialize(self, datasource_node, pipeline_id):
        """Create the schema and seed it with the initial DataSource + meta rows."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA_SQL)
            self.write_datasource(conn, datasource_node)
            conn.execute("INSERT INTO meta (key, value) VALUES ('version', '1')")
            conn.execute("INSERT INTO meta (key, value) VALUES ('pipeline_id', ?)", (pipeline_id,))

    def fetch_all(self):
        """Read the full DB state.

        Returns:
            dict: ``{'pipeline_id', 'datasource', 'grps', 'nodes'}`` — ``grps``/
            ``nodes`` are ``{name: {...}}`` dicts (values already JSON-deserialized),
            in ``rowid`` (insertion) order. ``datasource`` is ``None`` if absent.
        """
        from ._serialize import deserialize_from_json

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            pipeline_id = None
            row = conn.execute("SELECT value FROM meta WHERE key = 'pipeline_id'").fetchone()
            if row:
                pipeline_id = row['value']

            datasource = None
            row = conn.execute("SELECT * FROM datasource WHERE id = 1").fetchone()
            if row:
                datasource = {
                    'schema': json.loads(row['schema']),
                    'targets': json.loads(row['targets']),
                }

            grps = {}
            for row in conn.execute("SELECT * FROM grps ORDER BY rowid").fetchall():
                grps[row['name']] = {
                    'processor': deserialize_from_json(row['processor']),
                    'edges': deserialize_from_json(row['edges']) or {},
                    'method': row['method'],
                    'parent': row['parent'],
                    'adapter': deserialize_from_json(row['adapter']),
                    'params': deserialize_from_json(row['params']) or {},
                    'desc': row['desc'],
                }

            nodes = {}
            for row in conn.execute("SELECT * FROM nodes ORDER BY rowid").fetchall():
                nodes[row['name']] = {
                    'grp': row['grp'],
                    'processor': deserialize_from_json(row['processor']),
                    'edges': deserialize_from_json(row['edges']) or {},
                    'method': row['method'],
                    'adapter': deserialize_from_json(row['adapter']),
                    'params': deserialize_from_json(row['params']) or {},
                    'desc': row['desc'],
                }

        return {'pipeline_id': pipeline_id, 'datasource': datasource, 'grps': grps, 'nodes': nodes}

    def execute(self, fn):
        """Run *fn(conn)* in a single connection/transaction. *fn* may issue
        arbitrary statements (including several of the ``write_*``/raw SQL
        below) to share one transaction."""
        with sqlite3.connect(str(self.db_path)) as conn:
            fn(conn)

    def write_grp(self, conn, grp):
        from ._serialize import serialize_to_json
        conn.execute(
            "INSERT OR REPLACE INTO grps "
            "(name, processor, edges, method, parent, adapter, params, desc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (grp.name,
             serialize_to_json(grp.processor) if grp.processor is not None else None,
             serialize_to_json(grp.edges),
             grp.method, grp.parent,
             serialize_to_json(grp.adapter) if grp.adapter is not None else None,
             serialize_to_json(grp.params),
             grp.desc)
        )

    def write_node(self, conn, node):
        from ._serialize import serialize_to_json
        conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(name, grp, processor, edges, method, adapter, params, desc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node.name, node.grp,
             serialize_to_json(node.processor) if node.processor is not None else None,
             serialize_to_json(node.edges),
             node.method,
             serialize_to_json(node.adapter) if node.adapter is not None else None,
             serialize_to_json(node.params),
             node.desc)
        )

    def write_datasource(self, conn, ds):
        conn.execute(
            "INSERT OR REPLACE INTO datasource (id, schema, targets) VALUES (1, ?, ?)",
            (json.dumps(ds.schema), json.dumps(ds.targets))
        )

    # ------------------------------------------------------------------
    # built Pipeline versions
    # ------------------------------------------------------------------

    def save_version(self, pipeline):
        """Pickle *pipeline* as the next version; return the version number.

        Every call mints a new version — there is no content dedup. The
        counter lives here, in the same db as the definitions that produced
        it, rather than in a project-wide registry.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM versions").fetchone()
            version = row[0] + 1
            version_path = self.db_path.parent / f'v{version}.pkl'
            pipeline.version = version
            with open(version_path, 'wb') as f:
                pkl.dump(pipeline, f)
            conn.execute(
                "INSERT INTO versions (version, path) VALUES (?, ?)",
                (version, str(version_path)),
            )
        return version

    def load_version(self, version=None):
        """Load a saved Pipeline version (latest if *version* is omitted)."""
        if not self.exists():
            raise KeyError(f"No pipeline db at {self.db_path}")
        with sqlite3.connect(str(self.db_path)) as conn:
            if version is None:
                row = conn.execute(
                    "SELECT path FROM versions ORDER BY version DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT path FROM versions WHERE version = ?", (version,)
                ).fetchone()
        if not row:
            raise KeyError(f"No saved pipeline version {version!r} in {self.db_path}")
        with open(row[0], 'rb') as f:
            return pkl.load(f)

    def list_versions(self):
        if not self.exists():
            return []
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM versions ORDER BY version").fetchall()
        return [dict(r) for r in rows]
