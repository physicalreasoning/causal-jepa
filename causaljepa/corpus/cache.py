"""Content-addressed stage cache.

Deliberately NOT an orchestration framework. The pipeline here is linear with a
single fan-out over events:

    list events -> [per event: fetch rows + candles -> ladders -> arrays] -> corpus

That is a map, not a DAG, and the real needs are narrow: never refetch what we
already have, resume after a crash, and invalidate when the code that produced a
value changes. A cache keyed on (stage, version, params) covers all three in
about a hundred lines, with no scheduler, no database, and no daemon.

Revisit a real orchestrator when there are several series x sectors x backfill
windows needing scheduling, retries, and alerting. That is after the model
works, not before.

Invalidation is explicit: bump `version` on the stage when its output format or
computation changes. Silent staleness is the failure mode that makes caches
worse than useless, so the version is part of the key rather than a convention.
"""

# Vendored from the pm-jepa corpus builder so this repository stands alone.
# Unmodified except for the import rewrite noted below and this header. The
# corpus is reconstructed from Kalshi's UNAUTHENTICATED public API; no key,
# account or credential is involved at any point.

import hashlib
import json
import pathlib
from typing import Any, Callable, Dict, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_cache"


def _key(stage: str, version: int, params: Dict[str, Any]) -> str:
    blob = json.dumps({"stage": stage, "version": version, "params": params},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _path(stage: str, key: str, suffix: str) -> pathlib.Path:
    d = ROOT / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / "{}{}".format(key, suffix)


def json_stage(stage: str, version: int, params: Dict[str, Any],
               compute: Callable[[], Any], refresh: bool = False) -> Any:
    """Cache a JSON-serialisable result."""
    p = _path(stage, _key(stage, version, params), ".json")
    if p.exists() and not refresh:
        try:
            return json.loads(p.read_text())
        except (ValueError, OSError):
            p.unlink(missing_ok=True)       # corrupt entry, recompute
    value = compute()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(p)                          # atomic, so a crash cannot half-write
    return value


def array_stage(stage: str, version: int, params: Dict[str, Any],
                compute: Callable[[], Optional[Dict[str, np.ndarray]]],
                refresh: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """Cache a dict of numpy arrays. `compute` may return None to cache a miss.

    Caching a legitimate None matters: most skipped events are skipped for a
    stable reason (too few usable ladder minutes), and re-deriving that on every
    run costs two API calls per event forever.
    """
    key = _key(stage, version, params)
    p = _path(stage, key, ".npz")
    miss = _path(stage, key, ".miss")

    if miss.exists() and not refresh:
        return None
    if p.exists() and not refresh:
        try:
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        except (ValueError, OSError):
            p.unlink(missing_ok=True)

    value = compute()
    if value is None:
        miss.write_text("")
        return None
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **value)
    tmp.replace(p)
    return value


def stats() -> Dict[str, int]:
    """What is on disk, per stage. For reporting, not for control flow."""
    out: Dict[str, int] = {}
    if not ROOT.exists():
        return out
    for d in sorted(ROOT.iterdir()):
        if d.is_dir():
            out[d.name] = len([f for f in d.iterdir() if f.suffix in (".json", ".npz")])
    return out


def clear(stage: Optional[str] = None) -> int:
    """Remove cached entries. Returns how many files were deleted."""
    target = ROOT / stage if stage else ROOT
    if not target.exists():
        return 0
    n = 0
    for f in target.rglob("*"):
        if f.is_file():
            f.unlink()
            n += 1
    return n
