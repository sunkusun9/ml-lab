import re


class Connector:
    """Selects nodes by matching against name, processor, and/or edges.

    All provided criteria are combined with AND logic. Omitted criteria
    always match.

    Args:
        node_query: Node name filter. A ``str`` is treated as a regex pattern;
            a ``list`` requires exact membership.
        edges: Edge filter. ``{key: dsl_string}`` — for each key, the node's
            resolved ``edges[key]`` must equal this DSL string exactly.
        processor: Processor filter as a ``"module.ClassName"`` string — not
            a class. Compared directly (string equality) against the node's
            stored ``processor`` value, which Pipeline also keeps unresolved
            (whatever form ``set_grp``/``set_node`` was given). Use the same
            string form there for the match to line up.
    """

    def __init__(self, node_query=None, edges=None, processor=None):
        self.node_query = node_query
        self.edges = edges
        self.processor = processor

    def match(self, spec):
        """Return True if the node satisfies all configured criteria.

        Args:
            spec (ProcessorSpec): The node's resolved spec — from
                ``Pipeline.get_node_spec()`` or ``Trial.get_spec()``.

        Returns:
            bool: True if all criteria match.
        """
        if self.node_query is not None:
            if isinstance(self.node_query, str):
                if not re.search(self.node_query, spec.name):
                    return False
            elif isinstance(self.node_query, list):
                if spec.name not in self.node_query:
                    return False

        if self.processor is not None:
            if spec.processor != self.processor:
                return False

        if self.edges is not None:
            for key, required in self.edges.items():
                if spec.edges.get(key) != required:
                    return False

        return True
