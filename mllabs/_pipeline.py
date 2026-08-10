import re
import uuid
from pathlib import Path
from ._describer import desc_pipeline, desc_node, compare_nodes
from ._pipeline_store import PipelineStore, OPEN, PUBLISHED, ARCHIVED
from ._edge_dsl import referenced_nodes, validate_edges, iter_segments, eval_expr


VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})


class ColSelector:
    """Deferred column selector for processor params (e.g. ``cat_features``).

    Holds a DSL string only (see ``_edge_dsl``) — resolved against real data
    at fit time by ``_node_processor._resolve_col_selectors`` via
    ``eval_expr(parse(dsl_string), data)``, the same lazy-resolution
    principle as ``edges[key]``.
    """
    def __init__(self, dsl_string='*'):
        self.dsl_string = dsl_string


def _combine_edges_value(own, parent_val, key):
    """Combine a grp/node's own DSL string for *key* with the (already
    resolved) parent's value for that key.

    A leading ``'+'``/``'-'`` means "continue from the parent"; anything else
    fully replaces the parent (no inheritance).
    """
    if own is None:
        return parent_val
    stripped = own.strip()
    if stripped[:1] in ('+', '-'):
        if parent_val is None:
            raise ValueError(
                f"Edge (key='{key}') uses '+'/'-' continuation but there is no parent value to continue from"
            )
        return f"{parent_val} {stripped}"
    return own


def _combine_edges(own_edges, parent_edges):
    """Combine a full ``{key: str}`` edges dict with its resolved parent's."""
    parent_edges = parent_edges or {}
    return {
        k: _combine_edges_value(own_edges.get(k), parent_edges.get(k), k)
        for k in set(own_edges) | set(parent_edges)
    }


def _params_equal(a, b):
    """Compare two params dicts.

    ``_validate_params`` guarantees params hold plain data or ref specs, so
    plain ``==`` is exact here — no identity or ``__dict__`` fallbacks needed.
    """
    return a == b


def _ref_hint(value):
    """Best-guess ``"module.ClassName"`` string for *value*, for error messages."""
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, '__module__', None)
    qualname = getattr(target, '__qualname__', getattr(target, '__name__', None))
    if not module or not qualname:
        return None
    return f"{module}.{qualname}"


def _validate_processor(processor, where):
    """processor must be a ``"module.ClassName"`` string, never a class/instance.

    Nothing downstream resolves it back to a string, so a class stored here
    silently fails :meth:`Connector.match`, which compares the stored value as
    a string. Rejecting at definition time keeps that from ever happening.
    """
    if processor is None or isinstance(processor, str):
        return
    hint = _ref_hint(processor)
    suffix = f" Use {hint!r} instead." if hint else ""
    raise TypeError(
        f"{where}: processor must be a \"module.ClassName\" string, got "
        f"{type(processor).__name__}.{suffix}"
    )


def _validate_adapter(adapter, where):
    """adapter must be a string ref or a ``{'__ref__': ..., '__params__': ...}`` spec."""
    if adapter is None or isinstance(adapter, str):
        return
    if isinstance(adapter, dict) and '__ref__' in adapter:
        return
    hint = _ref_hint(adapter)
    suffix = f" Use {hint!r} or {{'__ref__': {hint!r}, '__params__': {{...}}}}." if hint else ""
    raise TypeError(
        f"{where}: adapter must be a \"module.ClassName\" string or a "
        f"{{'__ref__': ...}} spec, got {type(adapter).__name__}.{suffix}"
    )


def _validate_param_value(value, where, path):
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_param_value(item, where, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        if '__ref__' in value or '__callable__' in value:
            return  # lazy spec — instantiated in _node_processor at point of use
        for key, item in value.items():
            _validate_param_value(item, where, f"{path}[{key!r}]")
        return
    if hasattr(value, 'item') and hasattr(value, 'dtype') and getattr(value, 'shape', None) == ():
        return  # numpy scalar
    hint = _ref_hint(value)
    if isinstance(value, type):
        form = f"{{'__ref__': {hint!r}}}" if hint else "{'__ref__': ...}"
    elif callable(value):
        form = f"{{'__callable__': {hint!r}}}" if hint else "{'__callable__': ...}"
    else:
        form = f"{{'__ref__': {hint!r}, '__params__': {{...}}}}" if hint else "{'__ref__': ..., '__params__': {...}}"
    raise TypeError(
        f"{where}: params{path} must be plain data or a ref spec, got "
        f"{type(value).__name__}. Use {form} — it is instantiated lazily at fit time."
    )


def _validate_params(params, where):
    """params must hold plain data or ref specs — never live objects.

    Live objects (estimator instances, ColSelector, samplers, callbacks) make
    the pipeline unserializable and force identity-based comparison when
    deciding whether a definition changed. Ref specs are resolved lazily by
    ``_node_processor`` instead.
    """
    if not params:
        return
    for key, value in params.items():
        _validate_param_value(value, where, f"[{key!r}]")


class ProcessorSpec:
    """Everything needed to build one Processor and feed it — nothing else.

    The single shape both executable node kinds resolve to: a Stage
    (``_BuiltNode``/``_PipelineNode``) and a :class:`~mllabs.Trial`. Whoever
    consumes one — the executor, ``Connector``, the describers, ``Inferencer``
    — sees the same six fields regardless of which produced it.

    Five of them are the Processor constructor's own arguments (``name``,
    ``processor``, ``method``, ``adapter``, ``params`` — see
    ``_node_processor.py``). ``edges`` is not: it is the input wiring the
    flow uses to decide *what to feed* the processor, resolved lazily
    against real data at execution time. "Spec" rather than "attrs" because
    nothing here is resolved yet — ``processor``/``adapter``/``params`` stay
    raw declarations until a Processor is actually constructed.

    Display-only fields deliberately do not appear: a Stage's ``label`` (its
    originating group) and a Trial's ``desc``/``tag`` are reachable on the
    source object itself and were never read through this shape.

    Treat instances as immutable.
    """

    __slots__ = ('name', 'processor', 'edges', 'method', 'adapter', 'params')

    def __init__(self, name, processor, edges, method=None, adapter=None, params=None):
        self.name = name
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.adapter = adapter
        self.params = params if params is not None else {}

    def __eq__(self, other):
        if not isinstance(other, ProcessorSpec):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in self.__slots__)

    def __repr__(self):
        return (f"<ProcessorSpec {self.name!r} processor={self.processor!r} "
                f"method={self.method!r}>")


_DEFINITION_KEYS = ('processor', 'method', 'adapter', 'params', 'edges')


def _definition_of(spec):
    """The part of a :class:`ProcessorSpec` that determines its output.

    ``name`` is excluded — renaming a node does not change what it computes.
    """
    return {k: getattr(spec, k) for k in _DEFINITION_KEYS}


def _same_definition(pipeline, other):
    """True if *pipeline* and *other* say exactly the same thing.

    Not the same question as :meth:`Pipeline.diff_from`, which answers "what
    artifacts must be reset" and so returns *node* names only. A DataSource
    change that no node reads stales nothing — deliberately — but it is still
    a different definition, and a version has to appear for it. Without the
    DataSource comparison here, a Pipeline of a schema and no nodes could
    never leave the version it was first published as.
    """
    return (not pipeline.diff_from(other)
            and pipeline.datasource.schema == other.datasource.schema
            and pipeline.datasource.targets == other.datasource.targets)


class _SchemaColumns:
    """Stand-in for ``eval_expr``'s ``data`` argument, backed by a DataSource
    ``schema`` mapping instead of real data — enough to resolve a bare
    (DataSource-origin) edge segment's column *names*, since ``diff_from``
    never has actual data to work with."""
    __slots__ = ('_columns',)

    def __init__(self, schema):
        self._columns = list(schema.keys())

    def get_columns(self):
        return self._columns


def _ds_columns_unchanged(edges, old_schema, new_schema):
    """True if every bare (DataSource-origin) segment across *edges*
    resolves to the same column list under *old_schema* and *new_schema*.

    Segments that can't be resolved from schema alone (dtype/processor
    ``@selector``s, or a column no longer present) are treated as changed —
    there's no data here to prove they still resolve the same way.
    """
    for dsl_string in edges.values():
        for name, expr in iter_segments(dsl_string):
            if name is not None:
                continue
            try:
                old_cols = eval_expr(expr, _SchemaColumns(old_schema))
                new_cols = eval_expr(expr, _SchemaColumns(new_schema))
            except Exception:
                return False
            if old_cols != new_cols:
                return False
    return True


def _find_descendants(nodes, node_name):
    """Names of every node reachable downstream of *node_name*.

    Shared by :class:`PipelineBuilder` and the built :class:`Pipeline` — both
    hold ``output_edges`` on their node objects, which is all this needs.
    """
    descendants = set()
    queue = [node_name]

    while queue:
        current = queue.pop(0)

        if current not in nodes:
            continue

        for child_name in nodes[current].output_edges:
            if child_name not in descendants:
                descendants.add(child_name)
                queue.append(child_name)

    return descendants


def _affected_nodes(nodes, roots):
    """Node names downstream of *roots*, ordered by depth (DataSource dropped)."""
    priorities = {}
    queue = []

    for node_name in roots:
        priorities[node_name] = 1
        queue.append((node_name, 1))

    while queue:
        current_node, current_priority = queue.pop(0)

        for desc_node_name in _find_descendants(nodes, current_node):
            new_priority = current_priority + 1
            if desc_node_name not in priorities or priorities[desc_node_name] < new_priority:
                priorities[desc_node_name] = new_priority
                queue.append((desc_node_name, new_priority))

    sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])
    return [i[0] for i in sorted_nodes if i[0] is not None]


def _select_node_names(nodes, query):
    """Resolve a node query (``None`` / list / regex str) against *nodes*."""
    if query is None:
        return list(nodes.keys())
    if isinstance(query, list):
        return [n for n in query if n in nodes]
    if isinstance(query, str):
        pat = re.compile(query)
        return [k for k in nodes.keys() if k is not None and pat.search(k)]
    raise ValueError(f"query must be None, list, or str, got {type(query)}")


def _check_data_compatibility(schema, data):
    """Raise if *data* is missing any column declared in *schema*."""
    schema_cols = set(schema.keys())
    if not schema_cols:
        return
    missing = schema_cols - set(data.get_columns())
    if missing:
        raise ValueError(
            f"Data is missing columns defined in datasource schema: {sorted(missing)}"
        )


class _PipelineGroup:
    """A named group that shares configuration across its member nodes.

    Groups form a hierarchy via ``parent``. Child groups and their nodes
    inherit ``processor``, ``method``, ``adapter``, ``edges``, and ``params``
    from ancestors, with child values taking precedence.

    Attributes:
        name (str): Group name.
        processor: ``"module.ClassName"`` string (optional, may be inherited)
            — stored as-is, resolved to the actual class only at point of use
            (``_node_processor.py``).
        edges (dict): Edge definitions (optional, merged with parent).
        method (str): Processor method name (optional, may be inherited).
        parent (str): Parent group name, or ``None``.
        adapter: ``None`` / ``"module.ClassName"`` string / ``{"__ref__":...}``
            dict / instance (optional, may be inherited) — stored as-is,
            resolved only at point of use (``resolve_node_adapter``).
        params (dict): Constructor parameters (optional, merged with parent).
        children (list[str]): Child group names.
        nodes (list[str]): Node names belonging to this group.
    """

    def __init__(
        self, name, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, desc=None
    ):
        self.name = name
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
        edges = _combine_edges(self.edges, parent_attrs['edges'])
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
        for i in ['processor', 'method']:
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
        ret = _PipelineGroup(
            self.name, self.processor, self.edges.copy(),
            self.method, self.parent, self.adapter, self.params.copy(), self.desc
        )
        ret.children = self.children.copy()
        ret.nodes = self.nodes.copy()
        return ret

class _PipelineNode:
    """An individual executable unit in the pipeline.

    Node-level attributes override group attributes. Final resolved values
    are obtained via :meth:`get_attrs`.

    Attributes:
        name (str): Node name.
        grp (str): Parent group name.
        processor: ``"module.ClassName"`` string override (``None`` → inherit
            from group) — stored as-is, resolved only at point of use.
        edges (dict): Additional or overriding edge definitions.
        method (str): Processor method name override.
        adapter: ``None`` / ``"module.ClassName"`` string / ``{"__ref__":...}``
            dict / instance override — stored as-is, resolved only at point
            of use.
        params (dict): Constructor parameter overrides.
        output_edges (list[str]): Names of nodes that consume this node's output.
    """

    def __init__(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None, desc=None
    ):
        self.name = name
        self.grp = grp  # group name (str)
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.adapter = adapter
        self.params = params if params is not None else {}
        self.desc = desc

        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름
        self.spec = None

    def copy(self):
        ret = _PipelineNode(
            self.name, self.grp, self.processor, self.edges.copy(),
            self.method, self.adapter, self.params.copy(), self.desc
        )
        ret.output_edges = self.output_edges.copy()
        return ret

    def get_spec(self, grps):
        if self.spec is not None:
            return self.spec
        grp_attrs = grps[self.grp].get_attrs(grps)
        edges = _combine_edges(self.edges, grp_attrs['edges'])
        params = self.params.copy()
        if grp_attrs['params'] is not None:
            for k, v in grp_attrs['params'].items():
                if k not in params:
                    params[k] = v
        processor = grp_attrs['processor'] if self.processor is None else self.processor
        # adapter is left as whatever spec was stored (str / {'__ref__':...} /
        # instance / None) — never resolved to an instance here. The
        # by-processor-class default (when nothing was specified anywhere) is
        # also deferred, via resolve_node_adapter, to the point of use
        # (_node_processor.py / _executor.py._needs_gpu), not pipeline-definition
        # time.
        adapter = grp_attrs['adapter'] if self.adapter is None else self.adapter
        self.spec = ProcessorSpec(
            name=self.name,
            processor=processor,
            edges=edges,
            method=grp_attrs.get('method') if self.method is None else self.method,
            adapter=adapter,
            params=params,
        )
        return self.spec

    def update_spec(self):
        self.spec = None

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


class _DataSourceNode(_PipelineNode):
    """DataSource node that defines input schema and target columns.

    Attributes:
        schema (dict[str, str]): {col_name: var_type} where var_type is one of VAR_TYPES.
        targets (list[str]): Column names designated as targets.
    """

    def __init__(self):
        super().__init__("Data_Source", '__datasource__', None, None, None, None)
        self.schema = {}
        self.targets = []

    def get_attrs(self, grps=None):
        """DataSource attrs — a plain dict, deliberately *not* a ProcessorSpec.

        A DataSource has no processor/method/adapter/params/edges to declare;
        it is never executed, so it never becomes a job. It therefore does not
        override :meth:`_PipelineNode.get_spec` — the two are different shapes
        with different names rather than one polymorphic method returning
        either.
        """
        if self.spec is not None:
            return self.spec
        self.spec = {
            'name': self.name,
            'grp': self.grp,
            'schema': self.schema.copy(),
            'targets': list(self.targets),
        }
        return self.spec

    def copy(self):
        ret = _DataSourceNode()
        ret.schema = self.schema.copy()
        ret.targets = list(self.targets)
        ret.output_edges = self.output_edges.copy()
        return ret


class _BuiltNode:
    """A node in a built :class:`Pipeline`.

    Group inheritance is already resolved — every attribute here is the final
    value. ``label`` is the group name the node came from, kept for display and
    error messages only; it carries no structural meaning — every node in a
    Pipeline is a Stage.

    Treat instances as immutable. ``params``/``edges`` are shallow-copied at
    build time, so nested values are shared with the builder.
    """

    __slots__ = ('name', 'label', 'processor', 'edges', 'method',
                 'adapter', 'params', 'desc', 'output_edges')

    def __init__(self, name, label, processor, edges, method, adapter,
                 params, desc, output_edges):
        self.name = name
        self.label = label
        self.processor = processor
        self.edges = edges
        self.method = method
        self.adapter = adapter
        self.params = params
        self.desc = desc
        self.output_edges = output_edges

    def get_spec(self):
        return ProcessorSpec(
            name=self.name,
            processor=self.processor,
            edges=self.edges,
            method=self.method,
            adapter=self.adapter,
            params=self.params,
        )

    def copy(self, output_edges=None):
        return _BuiltNode(
            self.name, self.label, self.processor, dict(self.edges),
            self.method, self.adapter, dict(self.params),
            self.desc,
            list(self.output_edges if output_edges is None else output_edges),
        )

    def __repr__(self):
        return f"<_BuiltNode {self.name!r} label={self.label!r}>"


class _BuiltDataSource:
    """DataSource entry of a built :class:`Pipeline` (``nodes[None]``)."""

    __slots__ = ('name', 'schema', 'targets', 'output_edges')

    def __init__(self, name, schema, targets, output_edges):
        self.name = name
        self.schema = schema
        self.targets = targets
        self.output_edges = output_edges

    def get_attrs(self):
        return {
            'name': self.name,
            'schema': self.schema,
            'targets': self.targets,
        }

    def copy(self, output_edges=None):
        return _BuiltDataSource(
            self.name, dict(self.schema), list(self.targets),
            list(self.output_edges if output_edges is None else output_edges),
        )

    def __repr__(self):
        return f"<_BuiltDataSource {self.name!r} targets={self.targets!r}>"


class Pipeline:
    """Immutable node graph produced by :meth:`PipelineBuilder.build`.

    Holds only what running the graph needs: nodes with every group-inherited
    value already resolved, plus the DataSource schema. Groups do not survive
    the build, and a node's originating group name is kept as ``label``.
    A Pipeline holds Stages only; Heads are now :class:`~mllabs.Trial` objects
    handed to ``Experimenter.exp``.

    Consumers (Experimenter, Trainer, Inferencer) hold one of these rather than
    a builder, so later edits to the builder cannot silently change work that
    is already under way.

    Attributes:
        nodes (dict[str | None, _BuiltNode]): Nodes keyed by name. ``None`` is
            the DataSource (a :class:`_BuiltDataSource`).
        pipeline_id (str): Identity of the builder this was built from.
        build_id (str): Identity of this particular build.
        version (int | None): Set by :meth:`Project.publish_pipeline` when this
            Pipeline is frozen as a version. ``None`` while it is an
            unpublished build of the working copy, which is deliberate: a draft
            changes under anyone who adopted it, so a number here would be a
            claim the record could not keep. ``build_id`` says which draft.
        status (str): ``'open'`` until published, then ``'published'``. A
            version demoted by a later publish is ``'archived'`` — read from
            the store, since the copies already handed out cannot be updated.
    """

    def __init__(self, nodes, datasource, pipeline_id, build_id=None):
        self.nodes = {None: datasource}
        self.nodes.update(nodes)
        self.pipeline_id = pipeline_id
        self.build_id = build_id if build_id is not None else str(uuid.uuid4())
        self.version = None
        self.status = OPEN
        self._topo_order = _affected_nodes(self.nodes, [None])
        self._specs = {
            name: node.get_spec()
            for name, node in self.nodes.items() if name is not None
        }

    @classmethod
    def empty(cls, pipeline_id=None):
        """A Pipeline with no nodes — "there is nothing to build", as an object.

        Having no pipeline is a legitimate state, not a missing one: Trials
        read the DataSource directly, so an Experimenter with no nodes at all
        still runs. Saying that with an empty Pipeline rather than ``None``
        means every consumer stays branch-free — ``build()`` finds no jobs,
        ``topo_order()`` is empty, ``check_data_compatibility`` passes
        vacuously — instead of each one guarding for absence.

        Inside a Project this is version 0, published at creation, so the empty
        state is a real row rather than a fabricated object. This constructor
        is for the standalone case, where there is no store to mint from — the
        same reason a db-less builder's build stays ``open`` and unnumbered.
        """
        return cls({}, _BuiltDataSource('Data_Source', {}, [], []), pipeline_id)

    @property
    def is_empty(self):
        """No nodes at all — only the DataSource, which ``nodes`` always holds.

        What it means depends on which side of an adoption you are on: as the
        Pipeline being adopted, nothing to build; as the one being replaced,
        nothing that was ever adopted, so nothing a switch could invalidate.
        """
        return len(self.nodes) == 1

    @property
    def datasource(self):
        return self.nodes[None]

    def get_node(self, name):
        return self.nodes[name]

    def get_node_spec(self, name):
        """This node's :class:`ProcessorSpec` — see :meth:`_BuiltNode.get_spec`.

        Stage nodes only. The DataSource (``name=None``) has no spec — reach
        its schema/targets via ``pipeline.datasource``.
        """
        return self._specs[name]

    def get_node_names(self, query=None):
        """Resolve a node query to a list of node names.

        Args:
            query: ``None`` (all nodes), ``list`` (exact names), or ``str``
                (regex pattern matched against node names).

        Returns:
            list[str]: Matching node names (DataSource ``None`` excluded for
            str/list queries).
        """
        return _select_node_names(self.nodes, query)

    def topo_order(self):
        """Node names ordered from the DataSource downwards (DataSource excluded).

        Computed once at build time — the graph cannot change afterwards.
        """
        return list(self._topo_order)

    def descendants(self, name):
        """Names of every node downstream of *name*."""
        return _find_descendants(self.nodes, name)

    def diff_from(self, old):
        """Names whose output would differ from *old* — i.e. what is now stale.

        Walks this Pipeline from the DataSource downwards. A node is unchanged
        only if it exists in *old* under the same name, its definition matches,
        and every node it reads is itself unchanged; otherwise it is stale, and
        so is everything downstream of it (which falls out of the walk, since
        the walk is in topological order).

        Names that existed in *old* but are gone here are reported too, so their
        artifacts can be cleaned up rather than left orphaned.

        A DataSource whose schema or targets changed only stales a node if the
        DataSource-origin variables its own edges actually pull differ between
        the two schemas — a node that never reads the changed column(s) stays
        put.

        Args:
            old (Pipeline): The previously adopted Pipeline.

        Returns:
            set[str]: Node names to reset.
        """
        stale = set(old.nodes) - set(self.nodes) - {None}

        old_schema = old.datasource.schema
        new_schema = self.datasource.schema
        ds_changed = old_schema != new_schema or old.datasource.targets != self.datasource.targets

        for name in self.topo_order():
            if name not in old.nodes:
                stale.add(name)
                continue
            if ds_changed and not _ds_columns_unchanged(self.nodes[name].edges, old_schema, new_schema):
                stale.add(name)
                continue
            if _definition_of(old.get_node_spec(name)) != _definition_of(self.get_node_spec(name)):
                stale.add(name)
                continue
            for dsl_string in self.nodes[name].edges.values():
                if stale & referenced_nodes(dsl_string):
                    stale.add(name)
                    break
        return stale

    def check_data_compatibility(self, data):
        """Verify *data* contains every column declared in the DataSource schema.

        Args:
            data (DataWrapper): Wrapped dataset to check.

        Raises:
            ValueError: If any schema column is missing from *data*.
        """
        _check_data_compatibility(self.datasource.schema, data)

    def subset(self, node_names):
        """Return a new Pipeline with *node_names* and all their ancestors.

        Args:
            node_names (list[str]): Target node names. Upstream dependencies
                referenced through ``edges`` are pulled in automatically.

        Returns:
            Pipeline: Minimal pipeline needed to run *node_names*.
        """
        needed = set()
        queue = list(node_names)

        while queue:
            name = queue.pop(0)
            if name is None or name in needed or name not in self.nodes:
                continue
            needed.add(name)
            for dsl_string in self.nodes[name].edges.values():
                for edge_name in referenced_nodes(dsl_string):
                    if edge_name is not None and edge_name not in needed:
                        queue.append(edge_name)

        nodes = {
            name: self.nodes[name].copy(
                output_edges=[e for e in self.nodes[name].output_edges if e in needed]
            )
            for name in needed
        }
        datasource = self.datasource.copy(
            output_edges=[e for e in self.datasource.output_edges if e in needed]
        )
        return Pipeline(nodes, datasource, self.pipeline_id, self.build_id)

    def __repr__(self):
        return (f"<Pipeline nodes={len(self.nodes) - 1} "
                f"build_id={self.build_id[:8]}>")


class PipelineBuilder:
    """Node graph that describes an ML workflow.

    Holds groups (:class:`_PipelineGroup`) and nodes (:class:`_PipelineNode`).
    The implicit DataSource node is stored as ``nodes[None]``.

    Attributes:
        grps (dict[str, _PipelineGroup]): All registered groups.
        nodes (dict[str | None, _PipelineNode]): All nodes, keyed by name.
            ``None`` is the DataSource.
    """

    def __init__(self, path=None, name='pipeline'):
        self.grps = {'__datasource__': _PipelineGroup('__datasource__')}
        self.nodes = {None: _DataSourceNode()}
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
            ds = _DataSourceNode()
            ds.schema = data['datasource']['schema']
            ds.targets = data['datasource']['targets']
            self.nodes[None] = ds

        self.grps = {'__datasource__': _PipelineGroup('__datasource__')}
        for name, d in data['grps'].items():
            self.grps[name] = _PipelineGroup(
                name=name, processor=d['processor'],
                edges=d['edges'], method=d['method'], parent=d['parent'],
                adapter=d['adapter'], params=d['params'], desc=d['desc'],
            )

        self.nodes = {None: self.nodes[None]}
        for name, d in data['nodes'].items():
            node = _PipelineNode(
                name=name, grp=d['grp'], processor=d['processor'],
                edges=d['edges'], method=d['method'], adapter=d['adapter'],
                params=d['params'], desc=d['desc'],
            )
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
            spec = node.get_spec(self.grps)
            for dsl_string in spec.edges.values():
                for src_name in referenced_nodes(dsl_string):
                    if src_name in self.nodes:
                        src_node = self.nodes[src_name]
                        if name not in src_node.output_edges:
                            src_node.output_edges.append(name)

    def _db_write(self, fn):
        if self._store is not None:
            self._store.execute(fn)

    def sync(self):
        """Update in-memory PipelineBuilder state to match the SQLite DB.

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
            ValueError: If PipelineBuilder has no DB path.
        """
        if self._store is None:
            raise ValueError("PipelineBuilder has no DB path; cannot sync")

        changes = {
            'datasource': 'skip',
            'grps': {'added': [], 'removed': [], 'updated': []},
            'nodes': {'added': [], 'removed': [], 'updated': []},
        }

        data = self._store.fetch_all()

        # datasource
        db_ds = data['datasource']
        ds = self.nodes[None]
        if db_ds and (db_ds['schema'] != ds.schema or db_ds['targets'] != ds.targets):
            ds.schema = db_ds['schema']
            ds.targets = db_ds['targets']
            ds.update_spec()
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
            self.grps[name] = _PipelineGroup(
                name=name, processor=d['processor'],
                edges=d['edges'], method=d['method'], parent=d['parent'],
                adapter=d['adapter'], params=d['params'], desc=d['desc'],
            )
            changes['grps']['added'].append(name)

        for name in mem_grp_names & db_grp_names:
            d = db_grps[name]
            grp = self.grps[name]
            changed = grp.diff(d['processor'], d['edges'], d['method'],
                               d['parent'], d['adapter'], d['params'])
            if changed or grp.desc != d['desc']:
                grp.processor = d['processor']
                grp.edges = d['edges']
                grp.method = d['method']
                grp.parent = d['parent']
                grp.adapter = d['adapter']
                grp.params = d['params']
                grp.desc = d['desc']
                grp.update_attrs()
                changes['grps']['updated'].append(name)

        # A grp's own field change doesn't touch its member nodes' db rows, so
        # their attrs cache (which embeds the grp's inherited values) needs
        # invalidating here explicitly — the node loop below only catches
        # changes to a node's *own* row.
        affected_by_grp = set()
        for name in changes['grps']['updated']:
            affected_by_grp.update(self._get_all_nodes_in_grp(self.grps[name]))

        # nodes
        db_nodes = data['nodes']
        mem_node_names = set(self.nodes.keys()) - {None}
        db_node_names = set(db_nodes.keys())

        for name in mem_node_names - db_node_names:
            del self.nodes[name]
            changes['nodes']['removed'].append(name)

        for name in db_node_names - mem_node_names:
            d = db_nodes[name]
            node = _PipelineNode(
                name=name, grp=d['grp'], processor=d['processor'],
                edges=d['edges'], method=d['method'], adapter=d['adapter'],
                params=d['params'], desc=d['desc'],
            )
            self.nodes[name] = node
            changes['nodes']['added'].append(name)

        for name in mem_node_names & db_node_names:
            d = db_nodes[name]
            node = self.nodes[name]
            changed = node.diff(d['grp'], d['processor'], d['edges'],
                                d['method'], d['adapter'], d['params'])
            if changed or node.desc != d['desc'] or name in affected_by_grp:
                node.grp = d['grp']
                node.processor = d['processor']
                node.edges = d['edges']
                node.method = d['method']
                node.adapter = d['adapter']
                node.params = d['params']
                node.desc = d['desc']
                node.update_spec()
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
        ds.update_spec()

        self._db_write(lambda conn: self._store.write_datasource(conn, self.nodes[None]))
        return 'update'

    def check_data_compatibility(self, data):
        """Verify *data* contains every column declared in the DataSource schema.

        Args:
            data (DataWrapper): Wrapped dataset to check.

        Raises:
            ValueError: If any schema column is missing from *data*.
        """
        _check_data_compatibility(self.datasource.schema, data)

    def build(self):
        """Resolve group inheritance and return an immutable, published :class:`Pipeline`.

        This is the hand-off point to Experimenter/Trainer/Inferencer: they take
        the built Pipeline, never the builder, so later ``set_grp``/``set_node``
        calls do not reach into work already in progress.

        **Building publishes.** A builder with a db mints a version here, which
        makes a version appear exactly when the definition changes — an
        unchanged builder returns the version it already has rather than a
        duplicate of it. There is no separate publish step and no such thing as
        an unnumbered snapshot to run against, so every history row can name the
        definition it ran on.

        A builder with no db (constructed without a path) has nowhere to mint
        from, so its build stays ``open`` and unnumbered. That is the one
        in-memory case ``Trainer.set_pipeline`` refuses.

        Returns:
            Pipeline: Snapshot with every node's inherited values resolved,
            carrying ``version`` and ``status`` unless this builder has no db.
        """
        nodes = {}
        for name, node in self.nodes.items():
            if name is None:
                continue
            spec = node.get_spec(self.grps)
            nodes[name] = _BuiltNode(
                name=name,
                label=node.grp,
                processor=spec.processor,
                edges=dict(spec.edges),
                method=spec.method,
                adapter=spec.adapter,
                params=dict(spec.params),
                desc=node.desc,
                output_edges=list(node.output_edges),
            )

        ds = self.nodes[None]
        datasource = _BuiltDataSource(
            name=ds.name,
            schema=dict(ds.schema),
            targets=list(ds.targets),
            output_edges=list(ds.output_edges),
        )
        pipeline = Pipeline(nodes, datasource, self.pipeline_id)
        if self._store is not None:
            self._version(pipeline)
        return pipeline

    def _version(self, pipeline):
        """Give *pipeline* the version its definition belongs to, minting one if new."""
        try:
            current = self._store.load_version()
        except KeyError:
            current = None
        if current is not None and _same_definition(pipeline, current):
            # Same definition, so the same version — a second row would be a
            # duplicate of it, and every build would make one.
            pipeline.version = current.version
            pipeline.status = PUBLISHED
            return
        self._store.publish(pipeline, self)

    def copy(self):
        """Return a deep copy of the entire pipeline.

        Returns:
            PipelineBuilder: New pipeline with all groups and nodes copied.
        """
        ret = PipelineBuilder()
        ret.grps = {k: v.copy() for k, v in self.grps.items()}
        ret.nodes = {k: v.copy() for k, v in self.nodes.items()}
        return ret

    def copy_nodes(self, node_names):
        """Return a copy containing the specified nodes and all their ancestors.

        Args:
            node_names (list[str]): Target node names. Their upstream Stage
                dependencies are included automatically.

        Returns:
            PipelineBuilder: Minimal pipeline needed to run *node_names*.
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
            spec = self.nodes[name].get_spec(self.grps)
            for dsl_string in spec.edges.values():
                for edge_name in referenced_nodes(dsl_string):
                    if edge_name is not None and edge_name not in needed_nodes:
                        queue.append(edge_name)

        needed_grps = set()
        for name in needed_nodes:
            grp_name = self.nodes[name].grp
            while grp_name is not None and grp_name not in needed_grps:
                needed_grps.add(grp_name)
                grp_name = self.grps[grp_name].parent

        ret = PipelineBuilder()

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
        return _find_descendants(self.nodes, node_name)

    def _check_cycle(self, node_name, new_edges):
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for dsl_string in new_edges.values():
            for edge_name in referenced_nodes(dsl_string):
                if edge_name is None or edge_name not in self.nodes:
                    continue
                if edge_name in descendants:
                    cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []

    def _check_grp_update_cycles(self, name, old_grp, affected_nodes):
        """Raise ValueError if a group edge update would create a cycle in any
        affected node's merged edges, rolling ``self.grps[name]`` back to
        *old_grp* first."""
        for node_name in affected_nodes:
            if node_name not in self.nodes:
                continue

            # self.grps[name] already holds the proposed new grp (assigned by the
            # caller before this check runs), so get_spec() here already reflects
            # new_edges merged with the node's own edges — no further combination needed.
            node = self.nodes[node_name]
            node.update_spec()
            final_edges = node.get_spec(self.grps).edges

            has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
            if has_cycle:
                cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                self.grps[name] = old_grp
                raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")

    def _check_edges(self, edges, parent_edges=None):
        """Validate a raw (not-yet-merged) edges dict.

        Each value must be a DSL string. A leading ``'+'``/``'-'`` continuation
        is combined with *parent_edges* (the grp's own parent, or the node's
        grp) before structural validation via ``_edge_dsl.validate_edges`` —
        syntax and namespace references only, never columns/variables (those
        are resolved lazily, at process time, against real data).
        """
        if edges is None or len(edges) == 0:
            return False
        for key, dsl_string in edges.items():
            if not isinstance(dsl_string, str):
                raise ValueError(f"Edge (key='{key}') must be a DSL string, got {type(dsl_string).__name__}")
            combined = _combine_edges_value(dsl_string, (parent_edges or {}).get(key), key)
            try:
                validate_edges(combined, self)
            except ValueError as e:
                raise ValueError(f"Edge (key='{key}') is invalid: {e}") from e
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

    def set_grp(
            self, name, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, desc=None, exist='diff'
        ):
        """Create or update a group.

        Args:
            name (str): Group name. Cannot contain ``__`` or path-invalid chars.
            processor: ``"module.ClassName"`` string reference — not a class.
                Stored as-is; resolved to the actual class only at point of
                use (``_node_processor.py``), never here.
            edges (dict): Edge definitions ``{key: dsl_string}`` (see ``_edge_dsl``).
            method (str): Processor method name (e.g. ``'fit_transform'``).
            parent (str): Parent group name, or ``None``.
            adapter: ModelAdapter instance, a ``"module.ClassName"`` string
                (instantiated with defaults), or ``{"__ref__": ..., "__params__": {...}}``.
            params (dict): Constructor parameters for the processor. A value of
                the form ``{"__ref__": "mod.Cls", "__params__": {...}}`` is
                instantiated (e.g. a ``ColSelector``); ``{"__callable__": "mod.fn"}``
                resolves to the object itself (not called); plain strings/scalars pass through.
            exist (str): Conflict resolution — ``'diff'`` (default, skip if unchanged),
                ``'skip'``, ``'error'``, or ``'replace'``.

        Returns:
            dict: ``{result, grp, affected_nodes, [old_grp]}`` where *result* is
            ``'new'``, ``'skip'``, or ``'update'``.

        Raises:
            ValueError: If name is invalid or edges form a cycle.
        """
        self._validate_name(name)
        if name in self.nodes:
            raise ValueError(f"Name '{name}' already exists as a node")
        # processor/adapter/params are validated as *specs* here but never
        # resolved — resolution happens only at point of use
        # (_node_processor.py / resolve_node_adapter).
        _validate_processor(processor, f"set_grp({name!r})")
        _validate_adapter(adapter, f"set_grp({name!r})")
        _validate_params(params, f"set_grp({name!r})")
        if edges is None:
            edges = {}
        if params is None:
            params = {}

        if parent is not None and parent not in self.grps:
            raise ValueError(f"Parent group '{parent}' not found")

        parent_edges = self.grps[parent].get_attrs(self.grps)['edges'] if parent is not None else None

        if name not in self.grps:
            self._check_edges(edges, parent_edges)
            grp = _PipelineGroup(
                name, processor=processor, edges=edges, method=method, parent=parent, adapter=adapter, params=params, desc=desc
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

        self._check_edges(edges, parent_edges)
        old_grp = self.grps[name]
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
            self._check_grp_update_cycles(name, old_grp, affected_nodes)

            # Clear node attrs cache after cycle check to prevent stale data
            # from being used when nodes are next built.
            for node_name in affected_nodes:
                if node_name in self.nodes:
                    self.nodes[node_name].update_spec()

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
            self.nodes[node_name].update_spec()

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
        return _select_node_names(self.nodes, query)

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
        spec = node.get_spec(self.grps)
        self._update_output_edges(name, spec.edges, None)

        grp_name = node.grp
        if grp_name is not None and grp_name in self.grps:
            grp = self.grps[grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)

        del self.nodes[name]
        self._db_write(lambda conn: conn.execute("DELETE FROM nodes WHERE name = ?", (name,)))

    def _update_output_edges(self, node_name, old_edges, new_edges):
        if old_edges is not None:
            for dsl_string in old_edges.values():
                for edge_name in referenced_nodes(dsl_string):
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name in parent_node.output_edges:
                            parent_node.output_edges.remove(node_name)

        if new_edges is not None:
            for dsl_string in new_edges.values():
                for edge_name in referenced_nodes(dsl_string):
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name not in parent_node.output_edges:
                            parent_node.output_edges.append(node_name)

    def set_node(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None, desc=None, exist='diff'
    ):
        """Create or update a node.

        Args:
            name (str): Node name.
            grp (str): Group the node belongs to.
            processor: ``"module.ClassName"`` string reference override — not
                a class. Stored as-is; resolved to the actual class only at
                point of use (``_node_processor.py``), never here.
            edges (dict): Edge definitions ``{key: dsl_string}`` (see ``_edge_dsl``),
                merged on top of the group.
            method (str): Method name override.
            adapter: ``"module.ClassName"`` string (instantiated with defaults)
                or ``{"__ref__": ..., "__params__": {...}}`` — not an instance.
            params (dict): Constructor parameter overrides. Plain data only; a
                value of the form ``{"__ref__": "mod.Cls", "__params__": {...}}``
                is instantiated (e.g. a ``ColSelector``) and
                ``{"__callable__": "mod.fn"}`` resolves to the object itself
                (not called) — both lazily, at point of use.
            exist (str): Conflict resolution — ``'diff'`` (default), ``'skip'``,
                ``'error'``, or ``'replace'``.

        Returns:
            dict: ``{result, obj, old_obj, affected_nodes}``.

        Raises:
            TypeError: If processor/adapter/params hold live objects instead of
                string refs or ref specs.
            ValueError: If the resolved processor or method is missing, edges are
                invalid, or a cycle would be created.
        """
        _validate_processor(processor, f"set_node({name!r})")
        _validate_adapter(adapter, f"set_node({name!r})")
        _validate_params(params, f"set_node({name!r})")
        self._validate_name(name)

        if name in self.grps:
            raise ValueError(f"Name '{name}' already exists as a group")

        if grp not in self.grps:
            raise ValueError(f"Group '{grp}' not found")

        # processor and adapter are stored as-is (str / {'__ref__':...} dict /
        # class-or-instance / None) — never eagerly resolved here. Resolution
        # happens only at point of use (_node_processor.py / resolve_node_adapter).
        if edges is None:
            edges = {}
        if params is None:
            params = {}
        # params is stored as-is too — {'__ref__':...}/{'__callable__':...}
        # entries inside it are resolved lazily in _node_processor.py.

        grp_edges = self.grps[grp].get_attrs(self.grps)['edges']
        self._check_edges(edges, grp_edges)

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
                    self._db_write(lambda conn: conn.execute(
                        "UPDATE nodes SET desc = ? WHERE name = ?",
                        (desc, name)
                    ))
                    return {'result': 'skip', 'affected_nodes': [], 'old_obj': old_node, 'obj': old_node}

        old_edges = None
        old_output_edges = None
        old_node = None
        if is_update:
            old_node = self.nodes[name]
            old_edges = old_node.get_spec(self.grps).edges
            old_output_edges = old_node.output_edges

        node = _PipelineNode(
            name, grp, processor, edges, method=method, adapter=adapter, params=params, desc=desc
        )

        grp_obj = self.grps[grp]
        spec = node.get_spec(self.grps)

        if spec.processor is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        if spec.method is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        if len(spec.edges) == 0:
            raise ValueError(f"Cannot create node '{name}': edges is required")

        has_cycle, cycle_edges = self._check_cycle(name, spec.edges)
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        self._update_output_edges(name, old_edges, spec.edges)

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

        self._db_write(lambda conn: self._store.write_node(conn, node))

        return {
            'result': 'update' if is_update else 'new',
            'affected_nodes': affected_nodes,
            'old_obj': old_node,
            'obj': node
        }

    def get_node(self, name):
        return self.nodes.get(name, None)

    def get_node_spec(self, name):
        """Fully resolved :class:`ProcessorSpec` for a node (group hierarchy merged).

        Stage nodes only — the DataSource (``name=None``) has no spec; reach
        its schema/targets via ``builder.datasource.get_attrs()``.

        Args:
            name (str): Node name.

        Returns:
            ProcessorSpec
        """
        node = self.get_node(name)
        return node.get_spec(self.grps)

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