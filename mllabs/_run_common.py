"""Shared helpers between Experimenter and Trainer (serial tracking, tag filtering, status)."""


def resolve_common_status(statuses):
    """Collapse an iterable of per-fold node statuses into one value.

    Returns the common status if all agree, ``'inconsistent'`` if they
    differ, or ``None`` if the iterable is empty.
    """
    statuses = set(statuses)
    if not statuses:
        return None
    return statuses.pop() if len(statuses) == 1 else 'inconsistent'


def find_stale_nodes(pipeline, node_names, stores_for_name):
    """Return node names whose stored ``node_serial`` no longer matches the pipeline.

    Args:
        pipeline (Pipeline): Source of truth for current node serials.
        node_names (iterable[str]): Candidate node names to check.
        stores_for_name (callable): ``name -> iterable[NodeStore]`` — the
            stores to check for that node (callers differ in which stores
            are relevant, e.g. by node role or fold layout).
    """
    stale = []
    for name in node_names:
        current_serial = pipeline.nodes[name].serial
        for store in stores_for_name(name):
            info = store.get_info(name)
            if info is not None and info.get('node_serial') != current_serial:
                stale.append(name)
                break
    return stale


def filter_node_names_by_tags(pipeline, tags):
    """Return DataSource-excluded node names whose ``tag`` intersects *tags*."""
    tag_set = set(tags)
    return {
        n for n in pipeline.get_node_names(None)
        if n is not None and set(pipeline.nodes[n].tag) & tag_set
    }
