"""Resolver — turns a stored spec into the live value it names.

_serialize.py's resolve_processor/resolve_instance/resolve_ref_values are
pure functions: given only the spec string/dict, they can answer on their
own, because the one thing they ever reach for is an import. That stops
being true for an ``'@ext:name'`` param value — resolving it means asking a
specific project's ExtDataProvider, which no pure function can hold.
Resolver wraps the pure functions and adds that one stateful case, so it is
the single thing a Collector (and, in time, a Processor) needs at
construction time to turn its stored spec into live objects.
"""
import re

from ._serialize import resolve_processor, resolve_instance, resolve_ref_values

_EXT_RE = re.compile(r'^@ext:(.+)$')


class Resolver:
    """Args:
        ext_data (ExtDataProvider, optional): Backs ``'@ext:name'`` param
            values. Left unset, encountering one raises rather than silently
            passing the literal string through.
    """

    def __init__(self, ext_data=None):
        self.ext_data = ext_data

    def processor(self, processor):
        """``"module.ClassName"`` string -> class."""
        return resolve_processor(processor)

    def instance(self, spec):
        """String ref / ``{'__ref__': ...}`` spec -> instance (adapter, connector)."""
        return resolve_instance(spec)

    def params(self, params):
        """Every value in *params*, resolved.

        ``{'__ref__': ...}``/``{'__callable__': ...}`` specs resolve via
        ``resolve_ref_values`` exactly as they do for a Processor.
        ``'@ext:name'`` strings resolve against :attr:`ext_data`. Plain
        dicts/lists are recursed into so an ``@ext:`` reference nested
        inside one is still found; a value already inside a ``__ref__``/
        ``__callable__`` spec is handed to ``resolve_ref_values`` whole; it
        does not by itself understand ``@ext:``.
        """
        if not params:
            return params
        return {k: self._resolve_value(v) for k, v in params.items()}

    def _resolve_value(self, value):
        if isinstance(value, str):
            m = _EXT_RE.match(value)
            if m:
                name = m.group(1)
                if self.ext_data is None:
                    raise ValueError(
                        f"{value!r} references ext_data, but this Resolver "
                        f"has no ExtDataProvider"
                    )
                return self.ext_data.get(name)
            return value
        if isinstance(value, dict):
            if '__callable__' in value or '__ref__' in value:
                return resolve_ref_values(value)
            return {k: self._resolve_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(v) for v in value]
        return value
