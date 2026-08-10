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
        status TEXT NOT NULL,
        path TEXT NOT NULL,
        builder_path TEXT NOT NULL
    );
"""

#: A version that has been frozen and is the current one. Always exactly one.
PUBLISHED = 'published'
#: A version that was published and has since been succeeded by a newer one.
#: Frozen and still referenceable, and the only status that can be deleted.
ARCHIVED = 'archived'
#: The builder's working copy. Editable, unnumbered, never a row in ``versions``.
OPEN = 'open'


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

    def publish(self, pipeline, builder, version=None):
        """Freeze *pipeline* as the next version; return the version number.

        Two artifacts, because they answer different needs. ``v{n}.pkl`` is the
        built Pipeline, which is what an Experimenter or Trainer adopts.
        ``v{n}_builder.pkl`` is *builder* itself, kept because ``build()``
        resolves group inheritance away — a built snapshot has no groups left to
        edit, so it could never come back as a working definition.

        Whatever was published before becomes :data:`ARCHIVED`: only the newest
        is the current one.

        Args:
            pipeline (Pipeline): The built Pipeline to freeze.
            builder (PipelineBuilder): The definition it was built from.
            version (int, optional): Number to publish under. Defaults to the
                next one. ``Project`` passes 0 to seed the empty baseline, so
                that "there is always a published version" holds from the
                moment a project exists.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            if version is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM versions"
                ).fetchone()
                version = row[0] + 1
            version_path = self.db_path.parent / f'v{version}.pkl'
            builder_path = self.db_path.parent / f'v{version}_builder.pkl'
            pipeline.version = version
            pipeline.status = PUBLISHED
            with open(version_path, 'wb') as f:
                pkl.dump(pipeline, f)
            with open(builder_path, 'wb') as f:
                pkl.dump(builder, f)
            conn.execute(
                "UPDATE versions SET status = ? WHERE status = ?", (ARCHIVED, PUBLISHED)
            )
            conn.execute(
                "INSERT INTO versions (version, status, path, builder_path) VALUES (?, ?, ?, ?)",
                (version, PUBLISHED, str(version_path), str(builder_path)),
            )
        return version

    def load_version(self, version=None):
        """A frozen Pipeline by number, or the published one if *version* is omitted."""
        row = self._version_row(version)
        with open(row['path'], 'rb') as f:
            return pkl.load(f)

    def load_builder(self, version=None):
        """The PipelineBuilder a version was published from."""
        row = self._version_row(version)
        with open(row['builder_path'], 'rb') as f:
            return pkl.load(f)

    def get_status(self, version):
        """:data:`PUBLISHED` or :data:`ARCHIVED`, or ``None`` if there is no such version."""
        if not self.exists():
            return None
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM versions WHERE version = ?", (version,)
            ).fetchone()
        return row[0] if row else None

    def remove_version(self, version):
        """Delete an archived version and both its pickles.

        Archived only. ``published`` is the current definition and ``open`` is
        the working copy — neither is disposable. Removing an archived version
        breaks nothing that ran against it, since every Experimenter and Trainer
        keeps its own Pipeline copy; what goes is what a provenance pointer
        refers to.

        Raises:
            KeyError: If there is no such version.
            ValueError: If it is not archived.
        """
        row = self._version_row(version)
        if row['status'] != ARCHIVED:
            raise ValueError(
                f"Pipeline version {version} is {row['status']}, and only "
                f"{ARCHIVED} versions can be removed."
            )
        for key in ('path', 'builder_path'):
            Path(row[key]).unlink(missing_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM versions WHERE version = ?", (version,))

    def list_versions(self):
        if not self.exists():
            return []
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM versions ORDER BY version").fetchall()
        return [dict(r) for r in rows]

    def _version_row(self, version=None):
        if not self.exists():
            raise KeyError(f"No pipeline db at {self.db_path}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if version is None:
                row = conn.execute(
                    "SELECT * FROM versions WHERE status = ?", (PUBLISHED,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM versions WHERE version = ?", (version,)
                ).fetchone()
        if not row:
            raise KeyError(
                f"No published pipeline in {self.db_path}" if version is None
                else f"No pipeline version {version!r} in {self.db_path}"
            )
        return row
