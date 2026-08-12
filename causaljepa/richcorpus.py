"""Rebuild the corpus keeping the candle fields the original builder discards.

WHY. Finding 18: Kalshi's candlestick endpoint returns TEN fields per strike per
minute, open/high/low/close for both `yes_bid` and `yes_ask` plus volume and
open interest. `data/kalshi.py::candle_quote` reads TWO of them, both closes. The
intra-minute range of each side, which is the cheapest good volatility estimator
below the minute grid, is dropped at ingest, and E7's headline targets were
volatility targets. Separately, events carry up to 188 strikes and are
interpolated onto a 24-point grid.

So the programme's conclusion, that these ladders carry too little structure for
a representation learner, was never controlled for how much of the data the
corpus keeps. This module builds both feature sets from ONE fetch so that
question can be answered.

THE LEAN ARM MUST BE BIT-IDENTICAL, and the first attempt at this failed exactly
there. That attempt recomputed spot and width with a crude quantile instead of
calling slate's own functions, rejected 566 minutes the real builder keeps, and
its lean arm returned -0.5459 where the corpus gets +0.1487. A control arm that
misses its own target by 0.7 invalidates everything measured beside it.

This module therefore does not reimplement any of it. `slate.ladders_from_candles`
builds the ladder, `slate.validate_ladder` accepts or rejects the minute,
`slate.implied_spot` and `slate.implied_width` set the anchor, and
`slate.resample_ladder` produces the lean channels. The rich channels are
interpolated onto THE SAME GRID, from the same accepted minutes, using the same
spot. The only difference between the arms is how many channels survive.

RICH CHANNELS, 10 in total: the original 4, then
    bid range        high - low of yes_bid within the minute
    ask range        high - low of yes_ask
    mid range        the average of the two, an intra-minute realised range
    mid drift        close - open of the mid, signed intra-minute movement
    bid open->close  signed, separates quote-side movement from mid movement
    ask open->close

WRITES INTO THIS REPOSITORY, NEVER INTO pm-jepa. docs/RESEARCH_PLAN.md rule 7
makes the predecessor repos read-only sources; the rich cache lives under this
repo's own `data_cache/event_arrays_rich/` and is gitignored like the rest.
"""
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

RICH_VERSION = 1
N_LEAN, N_RICH_EXTRA = 4, 6
GRID_N, STEP_FRAC = 24, 0.002


def _pm_root() -> str:
    from . import data as cj_data
    root = os.environ.get("PM_JEPA_ROOT")
    return str(pathlib.Path(root).expanduser().resolve()) if root else cj_data._resolve_root(None)


def _slate():
    """Import pm-jepa's data package. Read-only; we never call anything that writes."""
    root = _pm_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from data import kalshi, slate  # noqa
    finally:
        sys.dont_write_bytecode = was
    return kalshi, slate


def cache_dir() -> pathlib.Path:
    d = pathlib.Path(__file__).resolve().parents[1] / "data_cache" / "event_arrays_rich"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _f(kalshi, x, default=0.0):
    v = kalshi.to_float(x)
    return default if v is None else v


def _extra_by_ts(kalshi, rows, candles) -> Dict[int, Dict[float, Tuple]]:
    """{ts: {strike: (bid_rng, ask_rng, mid_rng, mid_drift, bid_oc, ask_oc)}}.

    Keyed by the same (ts, strike) that `ladders_from_candles` keys on, so the
    two views line up without any re-derivation of which minutes exist.
    """
    strike_by_ticker = {}
    for r in rows:
        k = kalshi.strike_of(r)
        if k is not None:
            strike_by_ticker[r["ticker"]] = k

    out: Dict[int, Dict[float, Tuple]] = {}
    for tkr, cs in candles.items():
        k = strike_by_ticker.get(tkr)
        if k is None:
            continue
        for c in cs:
            t = c.get("end_period_ts")
            if t is None:
                continue
            yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
            bh, bl = _f(kalshi, yb.get("high_dollars")), _f(kalshi, yb.get("low_dollars"))
            ah, al = _f(kalshi, ya.get("high_dollars")), _f(kalshi, ya.get("low_dollars"))
            bo, bc = _f(kalshi, yb.get("open_dollars")), _f(kalshi, yb.get("close_dollars"))
            ao, ac = _f(kalshi, ya.get("open_dollars")), _f(kalshi, ya.get("close_dollars"))
            out.setdefault(int(t), {})[k] = (
                bh - bl, ah - al, ((bh + ah) - (bl + al)) / 2.0,
                ((bc + ac) - (bo + ao)) / 2.0, bc - bo, ac - ao)
    return out


def build_event(series: str, event_ticker: str, grid_n: int = GRID_N,
                step_frac: float = STEP_FRAC) -> Optional[dict]:
    """One event -> lean (T,K,4), rich (T,K,10), ts, state, settle.

    The lean path is `data/dataset.py::build_event` verbatim in its choices: same
    ladder builder, same validator, same spot and width, same resampler. Only the
    extra channels are new, and they ride the SAME accepted minutes and the SAME
    moneyness grid.
    """
    kalshi, slate = _slate()
    rows = kalshi._get("markets", event_ticker=event_ticker,
                       limit=1000).get("markets", [])
    if not rows:
        return None
    settle = kalshi.event_settlement(rows)
    if settle is None:
        return None

    open_ts = kalshi.ts(rows[0]["open_time"])
    close_ts = kalshi.ts(rows[0]["close_time"])
    open_ts = max(open_ts, close_ts - 360 * 60)

    cbt = kalshi.batch_candles([r["ticker"] for r in rows], open_ts, close_ts, 1)
    lads = slate.ladders_from_candles(rows, cbt)
    if len(lads) < 8:
        return None
    extra = _extra_by_ts(kalshi, rows, cbt)

    lean, rich, tss, state = [], [], [], []
    for t in sorted(lads):
        lad = lads[t]
        if not slate.validate_ladder(lad, len(rows), "threshold").ok:
            continue
        spot = slate.implied_spot(lad)
        width = slate.implied_width(lad)
        if spot is None or width is None or spot <= 0:
            continue

        base = slate.resample_ladder(lad, spot, n=grid_n, step_frac=step_frac)
        ks = np.array([r.strike for r in lad], dtype=np.float64)
        offs = (np.arange(grid_n) - (grid_n - 1) / 2.0) * step_frac
        grid = spot * np.exp(offs)

        ex = extra.get(t, {})
        cols = []
        for j in range(N_RICH_EXTRA):
            v = np.array([ex.get(r.strike, (0.0,) * N_RICH_EXTRA)[j] for r in lad],
                         dtype=np.float64)
            cols.append(np.interp(grid, ks, v, left=v[0], right=v[-1]))
        lean.append(base)
        rich.append(np.concatenate([base, np.stack(cols, axis=-1)], axis=-1))
        tss.append(t)
        state.append((spot, width, float(close_ts - t)))

    if len(lean) < 8:
        return None
    return {"lean": np.stack(lean).astype(np.float32),
            "rich": np.stack(rich).astype(np.float32),
            "ts": np.array(tss, dtype=np.int64),
            "state": np.array(state, dtype=np.float64),
            "settle": np.array([settle], dtype=np.float64)}


def cached_build(series: str, event_ticker: str, refresh: bool = False) -> Optional[dict]:
    """Per-event cache so a 10,000-call rebuild is resumable after any interruption."""
    import hashlib
    key = hashlib.sha256(json.dumps(
        {"stage": "event_arrays_rich", "version": RICH_VERSION,
         "params": {"series": series, "event": event_ticker,
                    "grid_n": GRID_N, "step_frac": STEP_FRAC}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    p = cache_dir() / (key + ".npz")
    miss = cache_dir() / (key + ".miss")
    if miss.exists() and not refresh:
        return None
    if p.exists() and not refresh:
        try:
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        except (ValueError, OSError):
            p.unlink(missing_ok=True)
    got = build_event(series, event_ticker)
    if got is None:
        miss.write_text("")
        return None
    np.savez_compressed(p, **got)
    return got


def windows(root=None, window: int = 24, stride: int = 4, horizon: int = 10,
            max_gap_factor: float = 2.0, arm: str = "rich"):
    """Assemble cached rich events into windows plus E7's volatility targets.

    `arm` selects which channel set is returned; both come from identical
    minutes, so switching it is the only difference between the two conditions.
    """
    X, Y, owner = [], [], []
    files = sorted(cache_dir().glob("*.npz"))
    for i, f in enumerate(files):
        try:
            with np.load(f) as z:
                obs = z[arm]
                ts, state = z["ts"], z["state"]
        except (ValueError, OSError, KeyError):
            continue
        T = obs.shape[0]
        if T < window:
            continue
        for start in range(0, T - window + 1, stride):
            end = start + window - 1
            fut = end + horizon
            if fut > T - 1:
                continue
            spot, width, t_rem = (float(v) for v in state[end])
            if spot <= 0 or width <= 0 or t_rem <= 0:
                continue
            dt = float(ts[fut] - ts[end])
            if dt <= 0 or dt > max_gap_factor * horizon * 60.0:
                continue
            path = np.asarray(state[end:fut + 1, 0], dtype=np.float64)
            if (path <= 0).any():
                continue
            r = np.diff(np.log(path))
            rv = float(np.sqrt((r ** 2).sum()))
            iv = float(width / spot * np.sqrt(dt / t_rem))
            fwd = float(np.log(path[-1] / path[0]))
            X.append(obs[start:start + window])
            Y.append((rv, np.log((rv + 1e-6) / (iv + 1e-6)), abs(fwd), fwd))
            owner.append(i)
    if not X:
        raise RuntimeError("no windows; run scripts/build_rich_corpus.py first")
    return (np.stack(X), np.array(Y, dtype=np.float32), np.array(owner),
            ["fwd_realised_vol", "vol_forecast_error", "fwd_abs_return",
             "fwd_signed_return"])
