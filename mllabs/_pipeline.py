import re
import uuid
import json
from pathlib import Path
from ._describer import desc_pipeline, desc_node, compare_nodes
from .adapter  import get_adapter
from ._serialize import _ref_to_obj
from ._pipeline_store import PipelineStore


VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})


class ColSelector:
    def __init__(self, col_type=None, pattern=None):
        self.col_type = col_type
        self.pattern = pattern


def _resolve_processor(processor):
    """Resolve a ``"module.ClassName"`` string reference to the actual class."""
    if isinstance(processor, str):
        return _ref_to_obj(processor)
    return processor


def _params_equal(a, b):
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_params_equal(a[k], b[k]) for k in a)
    a_dict = getattr(a, '__dict__', None)
    b_dict = getattr(b, '__dict__', None)
    if a_dict is None and b_dict is None:
        try:
            return bool(a == b)
        except Exception:
            return True
    elif a_dict is None or b_dict is None:
        return False
    return _params_equal(a_dict, b_dict)
class PipelineGroup:
    """A named group that shares configuration across its member nodes.

    Groups form a hierarchy via ``parent``. Child groups and their nodes
    inherit ``processor``, ``method``, ``adapter``, ``edges``, and ``params``
    from ancestors, with child values taking precedence.

    Attributes:
        name (str): Group name.
        role (str): ``'stage'`` or ``'head'``.
        processor: Processor class (optional, may be inherited).
        edges (dict): Edge definitions (optional, merged with parent).
        method (str): Processor method name (optional, may be inherited).
        parent (str): Parent group name, or ``None``.
        adapter: ModelAdapter instance (optional, may be inherited).
        params (dict): Constructor parameters (optional, merged with parent).
        children (list[str]): Child group names.
        nodes (list[str]): Node names belonging to this group.
    """

    def __init__(
        self, name, role, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, desc=None
    ):
        self.name = name
        self.role = role  # 'stage' or 'head'
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.parent = parent  # parent group name (str)
        self.adapter = adapter
        self.params = params if params is not None else {}
        self.desc = desc
        self.children = []  # child group names
        self.nodes = []  # node names in this group
        self.attrs = None

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        if self.parent is None:
            parent_attrs = {
                'edges': {},
                'params': {},
                'processor': None,
                'method': None,
                'adapter': None
            }
        else:
            parent_attrs = grps[self.parent].get_attrs(grps)
        edges = self.edges.copy()
        if parent_attrs['edges'] is not None:
            for k, v in parent_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if parent_attrs['params'] is not None:
            for k, v in parent_attrs['params'].items():
                if k not in params:
                    params[k] = v
        processor = parent_attrs['processor'] if self.processor is None else self.processor
        if self.adapter is None:
            if parent_attrs['adapter'] is not None:
                adapter = parent_attrs['adapter']
            else:
                adapter = None
        else:
            adapter = self.adapter
        self.attrs = {
            'name': self.name,
            'edges': edges,
            'parent': self.parent,
            'adapter': adapter,
            'params': params,
            'children': self.children,
        }
        for i in ['role', 'processor', 'method']:
            self.attrs[i] = parent_attrs.get(i) if getattr(self, i) is None else getattr(self, i)

        return self.attrs

    def update_attrs(self):
        self.attrs = None

    def diff(self, processor=None, edges=None, method=None, parent=None, adapter=None, params=None):
        changed = []
        if processor != self.processor:
            changed.append('processor')
        if edges != self.edges:
            changed.append('edges')
        if method != self.method:
            changed.append('method')
        if parent != self.parent:
            changed.append('parent')
        if adapter != self.adapter:
            changed.append('adapter')
        if not _params_equal(params if params is not None else {}, self.params):
            changed.append('params')
        return changed

    def copy(self):
        ret = PipelineGroup(
            self.name, self.role, self.processor, self.edges.copy(),
            self.method, self.parent, self.adapter, self.params.copy(), self.desc
        )
        ret.children = self.children.copy()
        ret.nodes = self.nodes.copy()
        return ret

class PipelineNode:
    """An individual executable unit in the pipeline.

    Node-level attributes override group attributes. Final resolved values
    are obtained via :meth:`get_attrs`.

    Attributes:
        name (str): Node name.
        grp (str): Parent group name.
        processor: Processor class override (``None`` → inherit from group).
        edges (dict): Additional or overriding edge definitions.
        method (str): Processor method name override.
        adapter: ModelAdapter instance override.
        params (dict): Constructor parameter overrides.
        output_edges (list[str]): Names of nodes that consume this node's output.
    """

    def __init__(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None, desc=None, tag=None
    ):
        self.name = name
        self.grp = grp  # group name (str)
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.adapter = adapter
        self.params = params if params is not None else {}
        self.desc = desc
        self.tag = tag if tag is not None else []
        self.serial = str(uuid.uuid4())

        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름
        self.attrs = None

    def copy(self):
        ret = PipelineNode(
            self.name, self.grp, self.processor, self.edges.copy(),
            self.method, self.adapter, self.params.copy(), self.desc, list(self.tag)
        )
        ret.serial = self.serial
        ret.output_edges = self.output_edges.copy()
        return ret

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        grp_attrs = grps[self.grp].get_attrs(grps)
        edges = self.edges.copy()
        if grp_attrs['edges'] is not None:
            for k, v in grp_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if grp_attrs['params'] is not None:
            for k, v in grp_attrs['params'].items():
                if k not in params:
                    params[k] = v
        processor = grp_attrs['processor'] if self.processor is None else self.processor
        if self.adapter is None:
            if grp_attrs['adapter'] is None:
                adapter = get_adapter(processor)
            else:
                adapter = grp_attrs['adapter']
        else:
            adapter = self.adapter
        self.attrs = {
            'name': self.name,
            'grp': self.grp,
            'edges': edges,
            'processor': processor,
            'adapter': adapter,
            'params': params,
            'method': grp_attrs.get('method') if self.method is None else self.method,
            'role': grp_attrs['role'],
            'serial': self.serial,
            'tag': list(self.tag),
        }

        return self.attrs

    def update_attrs(self):
        self.attrs = None

    def diff(self, grp, processor=None, edges=None, method=None, adapter=None, params=None):
        changed = []
        if grp != self.grp:
            changed.append('grp')
        if processor != self.processor:
            changed.append('processor')
        if edges != self.edges:
            changed.append('edges')
        if method != self.method:
            changed.append('method')
        if adapter != self.adapter:
            changed.append('adapter')
        if not _params_equal(params if params is not None else {}, self.params):
            changed.append('params')
        return changed


class DataSourceNode(PipelineNode):
    """DataSource node that defines input schema and target columns.

    Attributes:
        schema (dict[str, str]): {col_name: var_type} where var_type is one of VAR_TYPES.
        targets (list[str]): Column names designated as targets.
    """

    def __init__(self):
        super().__init__("Data_Source", '__datasource__', None, None, None, None)
        self.schema = {}
        self.targets = []

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        self.attrs = {
            'name': self.name,
            'grp': self.grp,
            'role': 'datasource',
            'serial': self.serial,
            'schema': self.schema.copy(),
            'targets': list(self.targets),
        }
        return self.attrs

    def copy(self):
        ret = DataSourceNode()
        ret.serial = self.serial
        ret.schema = self.schema.copy()
        ret.targets = list(self.targets)
        ret.output_edges = self.output_edges.copy()
        return ret


class Pipeline:
    """Node graph that describes an ML workflow.

    Holds groups (:class:`PipelineGroup`) and nodes (:class:`PipelineNode`).
    The implicit DataSource node is stored as ``nodes[None]``.

    Attributes:
        grps (dict[str, PipelineGroup]): All registered groups.
        nodes (dict[str | None, PipelineNode]): All nodes, keyed by name.
            ``None`` is the DataSource.
    """

    def __init__(self, path=None, name='pipeline'):
        self.grps = {'__datasource__': PipelineGroup('__datasource__', role='datasource')}
        self.nodes = {None: DataSourceNode()}
        self._store = None
        self.pipeline_id = str(uuid.uuid4())

        if path is not None:
            self._store = PipelineStore(path, name)
            if self._store.exists():
                self._load_db()
            else:
                self._store.initialize(self.nodes[None], self.pipeline_id)

    def _load_db(self):
        data = self._store.fetch_all()

        if data['pipeline_id']:
            self.pipeline_id = data['pipeline_id']

        if data['datasource']:
            ds = DataSourceNode()
            ds.schema = data['datasource']['schema']
            ds.targets = data['datasource']['targets']
            ds.serial = data['datasource']['serial']
            self.nodes[None] = ds

        self.grps = {'__datasource__': PipelineGroup('__datasource__', role='datasource')}
        for name, d in data['grps'].items():
            self.grps[name] = PipelineGroup(
                name=name, role=d['role'], processor=d['processor'],
                edges=d['edges'], method=d['method'], parent=d['parent'],
                adapter=d['adapter'], params=d['params'], desc=d['desc'],
            )

        self.nodes = {None: self.nodes[None]}
        for name, d in data['nodes'].items():
            node = PipelineNode(
                name=name, grp=d['grp'], processor=d['processor'],
                edges=d['edges'], method=d['method'], adapter=d['adapter'],
                params=d['params'], desc=d['desc'], tag=d['tag'],
            )
            node.serial = d['serial']
            self.nodes[name] = node

        self._rebuild_derived_state()

    def _rebuild_derived_state(self):
        """Rebuild grp.children, grp.nodes, and node.output_edges from self.grps/self.nodes.

        Called after self.grps/self.nodes have been (re)populated from the DB,
        whether by a full load or a partial sync.
        """
        for name, grp in self.grps.items():
            if name != '__datasource__':
                grp.children = []
                grp.nodes = []

        for name, grp in self.grps.items():
            if name == '__datasource__':
                continue
            if grp.parent and grp.parent in self.grps:
                if name not in self.grps[grp.parent].children:
                    self.grps[grp.parent].children.append(name)

        for name, node in self.nodes.items():
            if name is None:
                continue
            node.output_edges = []
            if node.grp in self.grps and name not in self.grps[node.grp].nodes:
                self.grps[node.grp].nodes.append(name)

        for name, node in list(self.nodes.items()):
            if name is None or node.grp not in self.grps:
                continue
            attrs = node.get_attrs(self.grps)
            for key, edge_list in attrs.get('edges', {}).items():
                for src_name, _ in edge_list:
                    if src_name in self.nodes:
                        src_node = self.nodes[src_name]
                        if name not in src_node.output_edges:
                            src_node.output_edges.append(name)

    def _db_write(self, fn):
        if self._store is not None:
            self._store.execute(fn)

    def sync(self):
        """Update in-memory Pipeline state to match the SQLite DB.

        DB is always the source of truth. Each element is compared and
        overwritten if different. After applying all changes, children,
        grp.nodes, and output_edges are fully rebuilt.

        Returns:
            dict: {
                'datasource': 'updated' | 'skip',
                'grps': {'added': [...], 'removed': [...], 'updated': [...]},
                'nodes': {'added': [...], 'removed': [...], 'updated': [...]},
            }

        Raises:
            ValueError: If Pipeline has no DB path.
        """
        if self._store is None:
            raise ValueError("Pipeline has no DB path; cannot sync")

        changes = {
            'datasource': 'skip',
            'grps': {'added': [], 'removed': [], 'updated': []},
            'nodes': {'added': [], 'removed': [], 'updated': []},
        }

        data = self._store.fetch_all()

        # datasource
        db_ds = data['datasource']
        if db_ds and db_ds['serial'] != self.nodes[None].serial:
            ds = self.nodes[None]
            ds.schema = db_ds['schema']
            ds.targets = db_ds['targets']
            ds.serial = db_ds['serial']
            ds.update_attrs()
            changes['datasource'] = 'updated'

        # grps
        db_grps = data['grps']
        mem_grp_names = set(self.grps.keys()) - {'__datasource__'}
        db_grp_names = set(db_grps.keys())

        for name in mem_grp_names - db_grp_names:
            del self.grps[name]
            changes['grps']['removed'].append(name)

        for name in db_grp_names - mem_grp_names:
            d = db_grps[name]
            self.grps[name] = PipelineGroup(
                name=name, role=d['role'], processor=d['processor'],
                edges=d['edges'], method=d['method'], parent=d['parent'],
                adapter=d['adapter'], params=d['params'], desc=d['desc'],
            )
            changes['grps']['added'].append(name)

        for name in mem_grp_names & db_grp_names:
            d = db_grps[name]
            grp = self.grps[name]
            changed = grp.diff(d['processor'], d['edges'], d['method'],
                               d['parent'], d['adapter'], d['params'])
            if changed or grp.role != d['role'] or grp.desc != d['desc']:
                grp.role = d['role']
                grp.processor = d['processor']
                grp.edges = d['edges']
                grp.method = d['method']
                grp.parent = d['parent']
                grp.adapter = d['adapter']
                grp.params = d['params']
                grp.desc = d['desc']
                grp.update_attrs()
                changes['grps']['updated'].append(name)

        # nodes
        db_nodes = data['nodes']
        mem_node_names = set(self.nodes.keys()) - {None}
        db_node_names = set(db_nodes.keys())

        for name in mem_node_names - db_node_names:
            del self.nodes[name]
            changes['nodes']['removed'].append(name)

        for name in db_node_names - mem_node_names:
            d = db_nodes[name]
            node = PipelineNode(
                name=name, grp=d['grp'], processor=d['processor'],
                edges=d['edges'], method=d['method'], adapter=d['adapter'],
                params=d['params'], desc=d['desc'], tag=d['tag'],
            )
            node.serial = d['serial']
            self.nodes[name] = node
            changes['nodes']['added'].append(name)

        for name in mem_node_names & db_node_names:
            d = db_nodes[name]
            node = self.nodes[name]
            if node.serial != d['serial']:
                node.grp = d['grp']
                node.processor = d['processor']
                node.edges = d['edges']
                node.method = d['method']
                node.adapter = d['adapter']
                node.params = d['params']
                node.desc = d['desc']
                node.tag = d['tag']
                node.serial = d['serial']
                node.update_attrs()
                changes['nodes']['updated'].append(name)

        self._rebuild_derived_state()

        return changes

    @property
    def datasource(self):
        return self.nodes[None]

    def set_datasource(self, schema, targets=None):
        """Define the input data schema and target columns.

        Args:
            schema (dict[str, str]): {col_name: var_type}. var_type must be one of
                'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'.
            targets (list[str], optional): Target column names. Must all exist in schema.

        Returns:
            str: ``'update'`` if schema/targets changed, ``'skip'`` if unchanged.

        Raises:
            ValueError: If any type is invalid or any target column is not in schema.
        """
        if targets is None:
            targets = []
        targets = list(targets)

        for col, typ in schema.items():
            if typ not in VAR_TYPES:
                raise ValueError(
                    f"Invalid type '{typ}' for column '{col}'. Must be one of {sorted(VAR_TYPES)}"
                )
        for col in targets:
            if col not in schema:
                raise ValueError(f"Target column '{col}' not in schema")

        ds = self.nodes[None]
        if ds.schema == schema and ds.targets == targets:
            return 'skip'

        ds.schema = dict(schema)
        ds.targets = targets
        ds.serial = str(uuid.uuid4())
        ds.update_attrs()

        self._bump_serials(self._get_affected_nodes([None]))
        self._db_write(lambda conn: self._store.write_datasource(conn, self.nodes[None]))
        return 'update'

    def check_data_compatibility(self, data):
        """Verify *data* contains every column declared in the DataSource schema.

        Args:
            data (DataWrapper): Wrapped dataset to check.

        Raises:
            ValueError: If any schema column is missing from *data*.
        """
        schema_cols = set(self.datasource.schema.keys())
        if not schema_cols:
            return
        missing = schema_cols - set(data.get_columns())
        if missing:
            raise ValueError(
                f"Data is missing columns defined in datasource schema: {sorted(missing)}"
            )

    def copy(self):
        """Return a deep copy of the entire pipeline.

        Returns:
            Pipeline: New pipeline with all groups and nodes copied.
        """
        ret = Pipeline()
        ret.grps = {k: v.copy() for k, v in self.grps.items()}
        ret.nodes = {k: v.copy() for k, v in self.nodes.items()}
        return ret

    def copy_stage(self):
        """Return a copy containing only Stage groups and nodes.

        Returns:
            Pipeline: Pipeline with only ``role='stage'`` groups and nodes.
        """
        ret = Pipeline()

        stage_grp_names = {name for name, grp in self.grps.items() if grp.role == 'stage'}

        for name in stage_grp_names:
            grp = self.grps[name].copy()
            grp.children = [c for c in grp.children if c in stage_grp_names]
            if grp.parent not in stage_grp_names:
                grp.parent = None
            ret.grps[name] = grp

        stage_node_names = set()
        for name, node in self.nodes.items():
            if name is None:
                continue
            if node.grp in stage_grp_names:
                stage_node_names.add(name)

        for name in stage_node_names:
            node = self.nodes[name].copy()
            node.output_edges = [e for e in node.output_edges if e in stage_node_names]
            ret.nodes[name] = node

        data_source = self.nodes[None].copy()
        data_source.output_edges = [e for e in data_source.output_edges if e in stage_node_names]
        ret.nodes[None] = data_source

        return ret

    def copy_nodes(self, node_names):
        """Return a copy containing the specified nodes and all their ancestors.

        Args:
            node_names (list[str]): Target node names. Their upstream Stage
                dependencies are included automatically.

        Returns:
            Pipeline: Minimal pipeline needed to run *node_names*.
        """
        needed_nodes = set()
        queue = list(node_names)

        while queue:
            name = queue.pop(0)
            if name is None or name in needed_nodes:
                continue
            if name not in self.nodes:
                continue
            needed_nodes.add(name)
            attrs = self.nodes[name].get_attrs(self.grps)
            for edge_list in attrs.get('edges', {}).values():
                for edge_name, _ in edge_list:
                    if edge_name is not None and edge_name not in needed_nodes:
                        queue.append(edge_name)

        needed_grps = set()
        for name in needed_nodes:
            grp_name = self.nodes[name].grp
            while grp_name is not None and grp_name not in needed_grps:
                needed_grps.add(grp_name)
                grp_name = self.grps[grp_name].parent

        ret = Pipeline()

        for name in needed_grps:
            grp = self.grps[name].copy()
            grp.children = [c for c in grp.children if c in needed_grps]
            grp.nodes = [n for n in grp.nodes if n in needed_nodes]
            if grp.parent not in needed_grps:
                grp.parent = None
            ret.grps[name] = grp

        for name in needed_nodes:
            node = self.nodes[name].copy()
            node.output_edges = [e for e in node.output_edges if e in needed_nodes]
            ret.nodes[name] = node

        data_source = self.nodes[None].copy()
        data_source.output_edges = [e for e in data_source.output_edges if e in needed_nodes]
        ret.nodes[None] = data_source

        return ret

    def _validate_name(self, name):
        if name is None:
            return

        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def _find_descendants(self, node_name):
        descendants = set()
        queue = [node_name]

        while queue:
            current = queue.pop(0)

            if current not in self.nodes:
                continue

            for child_name in self.nodes[current].output_edges:
                if child_name not in descendants:
                    descendants.add(child_name)
                    queue.append(child_name)

        return descendants

    def _check_cycle(self, node_name, new_edges):
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for key, edge_list in new_edges.items():
            for edge_name, _ in edge_list:
                if edge_name is None:
                    continue

                if edge_name not in self.nodes:
                    continue

                if edge_name in descendants:
                    cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []

    def _check_edges(self, edges):
        if edges is None or len(edges) == 0:
            return False
        schema_cols = set(self.datasource.schema.keys())
        for key, edge_list in edges.items():
            for name, var_spec in edge_list:
                if name is None:
                    if not isinstance(var_spec, list) or not all(isinstance(v, str) for v in var_spec):
                        raise ValueError(
                            f"DataSource edge (key='{key}') must specify var_spec as an explicit "
                            f"list of column names, got {var_spec!r}"
                        )
                    if schema_cols:
                        missing = [v for v in var_spec if v not in schema_cols]
                        if missing:
                            raise ValueError(
                                f"DataSource edge (key='{key}') references column(s) not in "
                                f"datasource schema: {missing}"
                            )
                    continue
                if name not in self.nodes:
                    raise ValueError(f"Edge node '{name}' not found")
                node_grp = self.nodes[name].grp
                if self.grps[node_grp].role != 'stage':
                    raise ValueError(f"Edge node '{name}' must be a stage node, got '{self.grps[node_grp].role}'")
        return True

    def _get_all_nodes_in_grp(self, grp):
        result = list(grp.nodes)
        for child_name in grp.children:
            child_grp = self.grps[child_name]
            result.extend(self._get_all_nodes_in_grp(child_grp))
        return result

    def _cascade_clear_attrs(self, grp_name):
        for child_name in self.grps[grp_name].children:
            self.grps[child_name].update_attrs()
            self._cascade_clear_attrs(child_name)

    def _bump_serials(self, node_names):
        for name in node_names:
            if name is not None and name in self.nodes:
                self.nodes[name].serial = str(uuid.uuid4())
                self.nodes[name].update_attrs()

        def _do(conn):
            for name in node_names:
                if name is not None and name in self.nodes:
                    conn.execute(
                        "UPDATE nodes SET serial = ? WHERE name = ?",
                        (self.nodes[name].serial, name)
                    )
        self._db_write(_do)

    def _get_affected_nodes(self, nodes):
        priorities = {}
        queue = []

        for node_name in nodes:
            priorities[node_name] = 1
            queue.append((node_name, 1))

        while queue:
            current_node, current_priority = queue.pop(0)

            descendants = self._find_descendants(current_node)

            for desc_node in descendants:
                new_priority = current_priority + 1
                if desc_node not in priorities or priorities[desc_node] < new_priority:
                    priorities[desc_node] = new_priority
                    queue.append((desc_node, new_priority))

        sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])
        return [i[0] for i in sorted_nodes if i[0] is not None]

    def set_grp(
            self, name, role=None, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, desc=None, exist='diff'
        ):
        """Create or update a group.

        Args:
            name (str): Group name. Cannot contain ``__`` or path-invalid chars.
            role (str): ``'stage'`` or ``'head'``. Inherited from parent if omitted.
            processor: Processor class, or ``"module.ClassName"`` string reference.
            edges (dict): Edge definitions ``{key: [(node_name, var_spec), ...]}``.
            method (str): Processor method name (e.g. ``'fit_transform'``).
            parent (str): Parent group name, or ``None``.
            adapter: ModelAdapter instance.
            params (dict): Constructor parameters for the processor.
            exist (str): Conflict resolution — ``'diff'`` (default, skip if unchanged),
                ``'skip'``, ``'error'``, or ``'replace'``.

        Returns:
            dict: ``{result, grp, affected_nodes, [old_grp]}`` where *result* is
            ``'new'``, ``'skip'``, or ``'update'``.

        Raises:
            ValueError: If name is invalid, role conflicts, or edges form a cycle.
        """
        self._validate_name(name)
        if name in self.nodes:
            raise ValueError(f"Name '{name}' already exists as a node")
        processor = _resolve_processor(processor)
        if edges is None:
            edges = {}
        if params is None:
            params = {}

        if parent is not None:
            if parent not in self.grps:
                raise ValueError(f"Parent group '{parent}' not found")
            if role is None:
                role = self.grps[parent].role
        if role is None and name in self.grps:
            role = self.grps[name].role
        if role not in ['stage', 'head']:
            raise ValueError(f"Role must be 'stage' or 'head', got '{role}'")

        if name not in self.grps:
            self._check_edges(edges)
            grp = PipelineGroup(
                name, role, processor=processor, edges=edges, method=method, parent=parent, adapter=adapter, params=params, desc=desc
            )

            if parent is not None:
                self.grps[parent].children.append(name)

            self.grps[name] = grp
            self._db_write(lambda conn: self._store.write_grp(conn, grp))
            return {
                "result": "new", "grp": grp, "affected_nodes": list()
            }
        elif exist == 'skip':
            grp = self.grps[name]
            return {"result": "skip", "grp": grp, "affected_nodes": list()}
        elif exist == 'error':
            raise ValueError(f"Group '{name}' already exists.")
        elif exist == 'diff':
            old_grp = self.grps[name]
            if not old_grp.diff(processor, edges, method, parent, adapter, params):
                old_grp.desc = desc
                self._db_write(lambda conn: conn.execute(
                    "UPDATE grps SET desc = ? WHERE name = ?", (desc, name)
                ))
                return {"result": "skip", "grp": old_grp, "affected_nodes": list()}

        self._check_edges(edges)
        old_grp = self.grps[name]
        if old_grp.role != role:
            raise ValueError(f"Cannot change role of group '{name}': existing '{old_grp.role}', requested '{role}'")
        grp = old_grp.copy()

        parent_changed = False
        old_parent = old_grp.parent

        if old_parent != parent:
            parent_changed = True
            if old_parent is not None:
                self.grps[old_parent].children.remove(name)
            grp.parent = parent
            if parent is not None:
                self.grps[parent].children.append(name)

        grp.processor = processor
        grp.edges = edges
        grp.method = method
        grp.adapter = adapter
        grp.params = params
        grp.desc = desc

        grp.update_attrs()
        attrs = grp.get_attrs(self.grps)
        new_edges = attrs['edges']
        affected_nodes = self._get_all_nodes_in_grp(grp)
        self.grps[name] = grp
        self._cascade_clear_attrs(name)
        if len(new_edges) > 0 or len(affected_nodes) > 0:
            for node_name in affected_nodes:
                if node_name not in self.nodes:
                    continue

                node = self.nodes[node_name]
                node.update_attrs()
                node_attrs = node.get_attrs(self.grps)
                node_own_edges = node_attrs.get('edges', {})

                final_edges = {k: list(v) for k, v in new_edges.items()}
                for k, v in node_own_edges.items():
                    if k in final_edges:
                        final_edges[k].extend(v)
                    else:
                        final_edges[k] = list(v)

                has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
                if has_cycle:
                    cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                    self.grps[name] = old_grp
                    raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")

            # Clear node attrs cache after cycle check to prevent stale data
            # from being used when nodes are next built.
            for node_name in affected_nodes:
                if node_name in self.nodes:
                    self.nodes[node_name].update_attrs()

        self._bump_serials(self._get_affected_nodes(affected_nodes))
        self._db_write(lambda conn: self._store.write_grp(conn, grp))

        return {
            "result": "update", "affected_nodes": affected_nodes, "old_grp": old_grp, "grp": grp
        }
    
    def get_grp(self, name):
        return self.grps.get(name, None)
    
    def rename_grp(self, name_from, name_to):
        self._validate_name(name_to)

        if name_from not in self.grps:
            raise ValueError(f"Group '{name_from}' not found")
        if name_to in self.grps:
            raise ValueError(f"Group '{name_to}' already exists")

        old_grp = self.grps[name_from]
        grp = old_grp.copy()
        grp.name = name_to
        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name_from)
            self.grps[grp.parent].children.append(name_to)

        for node_name in grp.nodes:
            self.nodes[node_name].grp = name_to
            self.nodes[node_name].update_attrs()

        for child_name in grp.children:
            self.grps[child_name].parent = name_to
            self.grps[child_name].update_attrs()

        del self.grps[name_from]
        self.grps[name_to] = grp

        def _do_rename(conn):
            conn.execute("DELETE FROM grps WHERE name = ?", (name_from,))
            self._store.write_grp(conn, grp)
            conn.execute("UPDATE nodes SET grp = ? WHERE grp = ?", (name_to, name_from))
            conn.execute("UPDATE grps SET parent = ? WHERE parent = ?", (name_to, name_from))
        self._db_write(_do_rename)

    def remove_grp(self, name):
        if name not in self.grps:
            raise ValueError(f"Group '{name}' not found")

        grp = self.grps[name]

        if len(grp.children) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.children)} child group(s)")

        if len(grp.nodes) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.nodes)} node(s)")

        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name)

        del self.grps[name]
        self._db_write(lambda conn: conn.execute("DELETE FROM grps WHERE name = ?", (name,)))

    def get_parents(self, node_name):
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        if node.grp is None or node.grp == '__datasource__':
            return []

        result = []
        current_grp = self.grps.get(node.grp)

        while current_grp is not None:
            result.append(current_grp.name)
            current_grp = self.grps.get(current_grp.parent) if current_grp.parent else None

        return result

    def get_node_names(self, query):
        """Resolve a node query to a list of node names.

        Args:
            query: ``None`` (all nodes), ``list`` (exact names), or
                ``str`` (regex pattern matched against node names).

        Returns:
            list[str]: Matching node names (DataSource ``None`` excluded for
            str/list queries).
        """
        if query is None:
            node_names = list(self.nodes.keys())
        elif isinstance(query, list):
            node_names = [n for n in query if n in self.nodes]
        elif isinstance(query, str):
            pat = re.compile(query)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"query must be None, list, or str, got {type(query)}")
        return node_names

    def remove_node(self, name):
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")

        if name is None:
            raise ValueError("Cannot remove DataSource node")

        descendants = self._find_descendants(name)
        if descendants:
            descendants_list = sorted(descendants)
            raise ValueError(f"Cannot remove node '{name}': has {len(descendants)} dependent node(s): {descendants_list}")

        node = self.nodes[name]
        node_attr = node.get_attrs(self.grps)
        self._update_output_edges(name, node_attr['edges'], None)

        grp_name = node.grp
        if grp_name is not None and grp_name in self.grps:
            grp = self.grps[grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)

        del self.nodes[name]
        self._db_write(lambda conn: conn.execute("DELETE FROM nodes WHERE name = ?", (name,)))

    def _update_output_edges(self, node_name, old_edges, new_edges):
        if old_edges is not None:
            for key, edge_list in old_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name in parent_node.output_edges:
                            parent_node.output_edges.remove(node_name)

        if new_edges is not None:
            for key, edge_list in new_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name not in parent_node.output_edges:
                            parent_node.output_edges.append(node_name)

    def set_node(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None, desc=None, tag=None, exist='diff'
    ):
        """Create or update a node.

        Args:
            name (str): Node name.
            grp (str): Group the node belongs to.
            processor: Processor class override, or ``"module.ClassName"`` string reference.
            edges (dict): Additional edge definitions merged on top of the group.
            method (str): Method name override.
            adapter: ModelAdapter instance override.
            params (dict): Constructor parameter overrides.
            exist (str): Conflict resolution — ``'diff'`` (default), ``'skip'``,
                ``'error'``, or ``'replace'``.

        Returns:
            dict: ``{result, obj, old_obj, affected_nodes}``.

        Raises:
            ValueError: If the resolved processor or method is missing, edges are
                invalid, or a cycle would be created.
        """
        self._validate_name(name)

        if name in self.grps:
            raise ValueError(f"Name '{name}' already exists as a group")

        if grp not in self.grps:
            raise ValueError(f"Group '{grp}' not found")

        processor = _resolve_processor(processor)
        if edges is None:
            edges = {}
        if params is None:
            params = {}

        self._check_edges(edges)

        is_update = name in self.nodes
        if is_update:
            if exist == 'skip':
                return {'result': 'skip', 'affected_nodes': [], 'old_obj': self.nodes[name], 'obj': self.nodes[name]}
            elif exist == 'error':
                raise ValueError(f"Node '{name}' already exists.")
            elif exist == 'diff':
                old_node = self.nodes[name]
                if not old_node.diff(grp, processor, edges, method, adapter, params):
                    old_node.desc = desc
                    old_node.tag = tag if tag is not None else []
                    self._db_write(lambda conn: conn.execute(
                        "UPDATE nodes SET desc = ?, tag = ? WHERE name = ?",
                        (desc, json.dumps(old_node.tag), name)
                    ))
                    return {'result': 'skip', 'affected_nodes': [], 'old_obj': old_node, 'obj': old_node}

        old_edges = None
        old_output_edges = None
        old_node = None
        if is_update:
            old_node = self.nodes[name]
            old_edges = old_node.get_attrs(self.grps)['edges']
            old_output_edges = old_node.output_edges

        node = PipelineNode(
            name, grp, processor, edges, method=method, adapter=adapter, params=params, desc=desc, tag=tag
        )

        grp_obj = self.grps[grp]
        attrs = node.get_attrs(self.grps)

        if attrs.get('processor') is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        if attrs.get('method') is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        if len(attrs.get('edges', {})) == 0:
            raise ValueError(f"Cannot create node '{name}': edges is required")

        has_cycle, cycle_edges = self._check_cycle(name, attrs['edges'])
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        self._update_output_edges(name, old_edges, attrs['edges'])

        if old_output_edges is not None:
            node.output_edges = old_output_edges

        if name not in grp_obj.nodes:
            grp_obj.nodes.append(name)

        if is_update:
            affected_nodes = list(self._find_descendants(name))
            old_grp_name = old_node.grp
            if old_grp_name != grp and old_grp_name in self.grps:
                old_grp = self.grps[old_grp_name]
                if name in old_grp.nodes:
                    old_grp.nodes.remove(name)
        else:
            affected_nodes = list()

        self.nodes[name] = node

        if is_update:
            self._bump_serials(affected_nodes)

        self._db_write(lambda conn: self._store.write_node(conn, node))

        return {
            'result': 'update' if is_update else 'new',
            'affected_nodes': affected_nodes,
            'old_obj': old_node,
            'obj': node
        }

    def get_node(self, name):
        return self.nodes.get(name, None)

    def add_tag(self, name, *tags):
        if name not in self.nodes or name is None:
            raise ValueError(f"Node '{name}' not found")
        node = self.nodes[name]
        changed = False
        for tag in tags:
            if tag not in node.tag:
                node.tag.append(tag)
                changed = True
        if changed:
            node.update_attrs()
            self._db_write(lambda conn: conn.execute(
                "UPDATE nodes SET tag = ? WHERE name = ?", (json.dumps(node.tag), name)
            ))

    def remove_tag(self, name, *tags):
        if name not in self.nodes or name is None:
            raise ValueError(f"Node '{name}' not found")
        node = self.nodes[name]
        changed = False
        for tag in tags:
            if tag in node.tag:
                node.tag.remove(tag)
                changed = True
        if changed:
            node.update_attrs()
            self._db_write(lambda conn: conn.execute(
                "UPDATE nodes SET tag = ? WHERE name = ?", (json.dumps(node.tag), name)
            ))

    def get_node_attrs(self, name):
        """Return fully resolved attributes for a node (group hierarchy merged).

        Args:
            name (str): Node name.

        Returns:
            dict: Keys — ``name``, ``grp``, ``processor``, ``method``,
            ``adapter``, ``edges``, ``params``.
        """
        node = self.get_node(name)
        return node.get_attrs(self.grps)

    def desc_pipeline(self, max_depth=None, direction='TD'):
        """파이프라인 구조를 Mermaid Markdown으로 반환

        Args:
            max_depth: 최대 표시 깊이 (None이면 무제한)
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        """
        return desc_pipeline(self, max_depth, direction)

    def compare_nodes(self, nodes):
        """Compare params and X-edges across nodes that share the same processor.

        Args:
            nodes (list[str]): Node names to compare.

        Returns:
            dict[str, pd.DataFrame]: ``{processor_name: DataFrame}`` where the
            DataFrame index is node names and columns are a MultiIndex of
            ``('params', param_key)`` and ``('X', stage_label)``.
        """
        return compare_nodes(self, nodes)

    def desc_node(self, node_name, direction='TD', show_params=False):
        """특정 노드까지의 연결 구조를 Mermaid Markdown으로 반환

        Args:
            node_name: 대상 노드 이름
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
            show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
        """
        return desc_node(self, node_name, direction, show_params)