"""ExtDataProvider — named external data, beyond the project's data/aug_data.

A Collector like ProcessCollector needs a live dataset (e.g. a held-out test
set) that cannot be expressed as a processor/adapter/params ref spec the way
everything else in a definition is — pickling it straight into params, as
CollectorStore.register does today, buries a live dataframe inside what is
otherwise a plain-data definition. This gives that dataset a name instead:
the definition holds the name (a string, document-friendly); the actual data
lives here.
"""
import pickle
from pathlib import Path


class ExtDataProvider:
    """Named registry of external data, persisted one pickle per name.

    Nothing is held in memory between calls — ``get`` reads straight off
    disk every time, ``register`` writes and forgets. A registry that
    cached would trade an unbounded number of entries against a byte
    budget it would then have to police, the way ``DataCache`` does for
    computed node outputs. Reading fresh instead means nothing here is
    ever evicted — appropriate because this data is not recomputable, it
    is exactly what the caller handed in — at the cost of a disk read per
    request. That cost is cheap where it is paid: a Collector reads this
    once per fold, not in a hot loop.

    Args:
        path (str | Path): Base directory — one file per registered name,
            created if missing.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, name):
        return self.path / f'{name}.pkl'

    def register(self, name, data):
        """Persist *data* under *name*, overwriting whatever was there."""
        with open(self._file(name), 'wb') as f:
            pickle.dump(data, f)

    def get(self, name):
        """*name*'s data, read fresh from disk.

        Raises:
            KeyError: If nothing is registered under *name*.
        """
        f = self._file(name)
        if not f.exists():
            raise KeyError(f"No data registered under {name!r}")
        with open(f, 'rb') as fp:
            return pickle.load(fp)

    def remove(self, name):
        """Delete *name*'s file. No-op if nothing was registered under it."""
        f = self._file(name)
        if f.exists():
            f.unlink()

    def names(self):
        """Every registered name, sorted."""
        return sorted(p.stem for p in self.path.glob('*.pkl'))

    def __contains__(self, name):
        return self._file(name).exists()

    def size(self, name):
        """*name*'s file size in bytes, or ``None`` if unregistered."""
        f = self._file(name)
        return f.stat().st_size if f.exists() else None

    def sizes(self):
        """``{name: bytes}`` for every registered name."""
        return {n: self.size(n) for n in self.names()}

    def __repr__(self):
        return f"<ExtDataProvider {self.path}>"
