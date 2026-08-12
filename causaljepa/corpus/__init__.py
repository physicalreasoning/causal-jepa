"""Kalshi corpus construction, vendored so this repository stands alone.

The predecessor repository (`pm-jepa`) built the corpus and is private. Rather
than require a checkout of it, the four modules that actually construct the data
are vendored here unmodified. They reach `api.elections.kalshi.com`, which is
Kalshi's UNAUTHENTICATED host: no API key, no account, no credential. Anyone can
regenerate the corpus with `scripts/build_corpus.py`.

    kalshi.py    HTTP client for the public endpoints, with caching and backoff
    slate.py     reconstructs a strike ladder from per-strike candlesticks
    dataset.py   window construction and probe targets
    cache.py     on-disk memoisation so a rebuild is incremental

The probe targets are documented in `dataset.py`. Read that docstring before
interpreting any number in `results/`: two of the four targets are derivable
from the input and are sanity checks rather than measures of learned structure,
which is the subject of finding 10 in FINDINGS.md.
"""
from . import cache, dataset, kalshi, slate  # noqa: F401

__all__ = ["cache", "dataset", "kalshi", "slate"]
