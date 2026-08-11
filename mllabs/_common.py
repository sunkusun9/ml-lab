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


def require_published_pipeline(pipeline):
    """Raise unless *pipeline* is a stored version.

    The adoption gate, shared by Experimenter and Trainer. Any published
    version passes — the newest and the oldest answer "what did this run
    against" equally well, and nothing demotes one.

    What is refused is a draft. It carries no number, so a run that adopted it
    could not say what it ran against, and a Trial or Predictor could not be
    checked against it at all. A draft is for looking at a definition, not for
    running one.
    """
    from ._pipeline_store import PUBLISHED

    if pipeline.status != PUBLISHED:
        raise ValueError(
            "This Pipeline is a draft, so it carries no version and nothing "
            "that ran against it could name what that was. Publish it with "
            "build() on a builder that has a db (project.pipeline), or adopt "
            "a stored version with project.load_pipeline(version)."
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


def error_payload(info):
    """A history row's failure, flattened to ``type``/``message``/``traceback``.

    Shared by the node-side and Trial-side readers, which read different stores
    but record a failure the same way — under ``info['error']``. A row with no
    payload still answers, with ``None`` in every field, so the caller never has
    to check whether the key was there.

    Collection history is not one of these: ``CollectHist`` writes ``phase``
    beside the same three fields and writes them at the top level, so it is
    merged rather than dug out.
    """
    err = (info or {}).get('error', {})
    return {'type': err.get('type'),
            'message': err.get('message'),
            'traceback': err.get('traceback')}
