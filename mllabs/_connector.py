import re

from ._serialize import resolve_processor as _resolve_processor


class Connector:
    """Selects nodes by matching against name, processor, and/or edges.

    All provided criteria are combined with AND logic. Omitted criteria
    always match.

    Args:
        node_query: Node name filter. A ``str`` is treated as a regex pattern;
            a ``list`` requires exact membership.
        edges: Edge filter. ``{key: dsl_string}`` — for each key, the node's
            resolved ``edges[key]`` must equal this DSL string exactly.
        processor: Processor class filter, or ``"module.ClassName"`` string
            reference (same convention as ``Pipeline.set_grp``/``set_node``).
            The node's resolved processor must be exactly this class.
    """

    def __init__(self, node_query=None, edges=None, processor=None, role=None):
        self.node_query = node_query
        self.edges = edges
        self.processor = _resolve_processor(processor)
        self.role = role

    def match(self, node_attrs):
        """Return True if the node satisfies all configured criteria.

        Args:
            node_attrs (dict): Resolved node attributes from
                ``Pipeline.get_node_attrs()``.

        Returns:
            bool: True if all criteria match.
        """
        node_name = node_attrs['name']
        if self.node_query is not None:
            if isinstance(self.node_query, str):
                if not re.search(self.node_query, node_name):
                    return False
            elif isinstance(self.node_query, list):
                if node_name not in self.node_query:
                    return False

        if self.processor is not None:
            if node_attrs.get('processor') != self.processor:
                return False

        if self.role is not None:
            if node_attrs.get('role') != self.role:
                return False

        if self.edges is not None:
            node_edges = node_attrs.get('edges', {})
            for key, required in self.edges.items():
                if node_edges.get(key) != required:
                    return False

        return True
