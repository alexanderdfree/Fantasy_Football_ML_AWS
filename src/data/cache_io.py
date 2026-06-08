"""Atomic parquet cache writes.

The data-layer loaders cache nflverse pulls to ``data/raw/*.parquet`` (and the
splits to ``data/splits/``) with the pattern
``if os.path.exists(path): read; else: fetch + write``. A plain
``df.to_parquet(path)`` is **not atomic**: it creates ``path`` and fills it
incrementally, so a *concurrent* reader — e.g. a second pytest-xdist worker that
hit the same cache miss — sees ``os.path.exists(path) == True`` and reads a
half-written file -> ``pyarrow.lib.ArrowInvalid``. (This is the CI "node down" /
``ArrowInvalid`` flake family, #1056/#1057: in CI ``data/raw`` starts empty
except the pre-fetched schedule, so the first test to need ``team_stats`` /
``redzone`` / a weekly pull races N workers on the same cache build.)

Writing to a unique temp file in the *same directory* and ``os.replace``-ing it
into place makes the swap atomic (a same-filesystem rename), so a reader only
ever sees a complete file — the old one or the new one, never a partial.
Concurrent writers each use their own temp (``mkstemp``) and the last
``os.replace`` wins; every observable state is a complete parquet.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write ``df`` to ``path`` atomically.

    Drop-in for ``df.to_parquet(path, **kwargs)`` at cache-write sites: writes to
    a unique temp file in ``path``'s directory, then ``os.replace`` swaps it in
    (atomic on the same filesystem). ``**kwargs`` is forwarded verbatim (e.g.
    ``index=False``). A stray temp is cleaned up if the write/replace fails.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
