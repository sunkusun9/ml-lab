"""Shared helpers between Experimenter and Trainer."""
import pickle as pkl
from pathlib import Path

PIPELINE_FILE = 'pipeline.pkl'


def save_pipeline(path, pipeline):
    """Write *pipeline* to ``{path}/pipeline.pkl``.

    A run keeps its own copy of the Pipeline it is working against, so that
    reopening it needs nothing but its directory — no Project to resolve a
    ``pipeline_version`` pointer into an object. The pointer
    is still recorded alongside, as provenance for which project version this
    copy came from.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / PIPELINE_FILE, 'wb') as f:
        pkl.dump(pipeline, f)


def load_pipeline(path):
    """The Pipeline saved at ``{path}/pipeline.pkl``, or ``None`` if absent."""
    file = Path(path) / PIPELINE_FILE
    if not file.exists():
        return None
    with open(file, 'rb') as f:
        return pkl.load(f)


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


def require_frozen_pipeline(pipeline):
    """Raise unless *pipeline* is a published or archived version.

    The Trainer-side gate. Both frozen statuses pass: what matters is that the
    definition cannot change under what was trained against it, and an archived
    version is as fixed as the published one. Only the working copy is refused.
    """
    from ._pipeline_store import OPEN

    if pipeline.status == OPEN:
        raise ValueError(
            "This Pipeline is the working copy, which is still editable, so "
            "nothing could later name what was trained against it. It has no "
            "version because its builder has no db to mint one from — build it "
            "from a builder that has one (project.pipeline)."
        )


def resolve_common_status(statuses):
    """Collapse an iterable of per-fold node statuses into one value.

    Returns the common status if all agree, ``'inconsistent'`` if they
    differ, or ``None`` if the iterable is empty.
    """
    statuses = set(statuses)
    if not statuses:
        return None
    return statuses.pop() if len(statuses) == 1 else 'inconsistent'


def format_errors(rows, traceback=False):
    """``(name, outer_idx, inner_idx, info)`` rows as printable error lines.

    Shared by the node-side and Trial-side reporters, which read different
    stores but describe a failure the same way.

    Returns:
        list[str] | None: One line per row, or ``None`` if there were none —
        so an empty result reads as "nothing failed" at a glance.
    """
    errors = []
    for name, outer_idx, inner_idx, info in rows:
        err = (info or {}).get('error', {})
        label = f"[{name}] fold {outer_idx}_{inner_idx}"
        line = f"{label} {err.get('type')}: {err.get('message')}"
        if traceback:
            line += f"\n{err.get('traceback')}"
        errors.append(line)
    return errors or None
