"""Shared helpers between Experimenter and Trainer (serial tracking, status)."""


def require_built_pipeline(pipeline):
    """Raise unless *pipeline* is a built :class:`Pipeline`.

    Experimenter/Trainer deliberately take the snapshot rather than the
    builder, so that editing the builder afterwards cannot reach into a run
    already in progress.
    """
    from ._pipeline import Pipeline, PipelineBuilder

    if isinstance(pipeline, PipelineBuilder):
        raise TypeError(
            "Expected a built Pipeline but got a PipelineBuilder. "
            "Build it first, e.g. set_pipeline(p.build())."
        )
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected a Pipeline, got {type(pipeline).__name__}")


def resolve_common_status(statuses):
    """Collapse an iterable of per-fold node statuses into one value.

    Returns the common status if all agree, ``'inconsistent'`` if they
    differ, or ``None`` if the iterable is empty.
    """
    statuses = set(statuses)
    if not statuses:
        return None
    return statuses.pop() if len(statuses) == 1 else 'inconsistent'


def find_stale_nodes(current_serials, node_names, stores_for_name):
    """Return names whose stored ``node_serial`` no longer matches the current one.

    Works for both Pipeline nodes (serial) and Trials (trial_id) — the caller
    supplies whichever identity applies.

    Args:
        current_serials (dict): ``{name: serial}`` — the source of truth.
        node_names (iterable[str]): Candidate names to check.
        stores_for_name (callable): ``name -> iterable[NodeStore]`` — the
            stores to check for that name (callers differ in which stores
            are relevant, e.g. by fold layout).
    """
    stale = []
    for name in node_names:
        current_serial = current_serials[name]
        for store in stores_for_name(name):
            info = store.get_info(name)
            if info is not None and info.get('node_serial') != current_serial:
                stale.append(name)
                break
    return stale
