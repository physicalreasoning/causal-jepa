"""Build training windows and probe targets from settled Kalshi events.

One hourly event gives up to 60 minutes of reconstructed ladder. A window is a
contiguous run of minutes; the token grid is (minute x moneyness-grid-point),
matching slate-jepa's (time x market) layout so the ported model code applies
unchanged.

Probe targets, and which are real:

    log_return_to_settle   log(settlement / implied_spot at window end)
                           GENUINELY FUTURE. The headline target.
    time_to_expiry         seconds from window end to settlement.
                           Not in the input, provided windows are sampled at
                           random offsets (see `random_offset`).
    implied_width          scale of the implied distribution. DERIVABLE from
                           the input, so it is a sanity check: a healthy
                           encoder should nail it. Low R^2 means broken, not
                           interesting.
    implied_spot_norm      the 0.5 crossing, normalised. Also derivable, also
                           a sanity check.

The clock is never a feature. Absolute timestamps and close_time are dropped
before anything reaches the model, exactly as in slate-jepa, because a leaked
clock manufactures probe R^2 out of nothing.
"""

# Vendored from the pm-jepa corpus builder so this repository stands alone.
# Unmodified except for the import rewrite noted below and this header. The
# corpus is reconstructed from Kalshi's UNAUTHENTICATED public API; no key,
# account or credential is involved at any point.

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import kalshi, slate

TARGET_NAMES = ("log_return_to_settle", "time_to_expiry",
                "implied_width", "window_log_return")
N_CHANNELS = 4

# The candlesticks endpoint caps candles per response. At 1-minute resolution a
# multi-day window blows through it and returns HTTP 400 with no explanation.
MAX_LOOKBACK_MINUTES = 360


def build_event(series: str, rows: Sequence[Dict], grid_n: int = 24,
                step_frac: float = 0.002
                ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One settled event -> (obs, minute_ts, per-minute state).

    obs:   (T, grid_n, N_CHANNELS)
    ts:    (T,) unix seconds
    state: (T, 3) implied_spot, implied_width, seconds_to_settlement
    """
    settle = kalshi.event_settlement(rows)
    if settle is None:
        return None
    open_ts = kalshi.ts(rows[0]["open_time"])
    close_ts = kalshi.ts(rows[0]["close_time"])

    # Clamp the candle window. Some events open days before expiry, and at
    # 1-minute resolution that exceeds the API's per-response candle cap, which
    # surfaces as an opaque HTTP 400. We only want the run-up to settlement
    # anyway, so cap the lookback rather than asking for the full life.
    open_ts = max(open_ts, close_ts - MAX_LOOKBACK_MINUTES * 60)

    cbt = kalshi.batch_candles([r["ticker"] for r in rows], open_ts, close_ts, 1)
    lads = slate.ladders_from_candles(rows, cbt)
    if len(lads) < 8:
        return None

    obs, tss, state = [], [], []
    for t in sorted(lads):
        lad = lads[t]
        rep = slate.validate_ladder(lad, len(rows), "threshold")
        if not rep.ok:
            continue
        spot = slate.implied_spot(lad)
        width = slate.implied_width(lad)
        if spot is None or width is None or spot <= 0:
            continue
        obs.append(slate.resample_ladder(lad, spot, n=grid_n, step_frac=step_frac))
        tss.append(t)
        state.append((spot, width, float(close_ts - t)))

    if len(obs) < 8:
        return None
    return (np.stack(obs).astype(np.float32),
            np.array(tss, dtype=np.int64),
            np.array(state, dtype=np.float64))


def windows_from_event(obs: np.ndarray, ts: np.ndarray, state: np.ndarray,
                       settle: float, window: int, stride: int = 4,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Slice an event into windows with targets taken at each window's END.

    Windows are strided rather than always ending at expiry, so window position
    carries no information about time-to-expiry. Without this, the time
    embedding alone would predict the clock and the probe would be vacuous.
    """
    T = obs.shape[0]
    if T < window:
        return np.empty((0,)), np.empty((0,))

    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        end = start + window - 1
        spot, width, t_rem = state[end]
        spot_0 = state[start][0]
        xs.append(obs[start:start + window])
        ys.append((
            np.log(settle / spot),      # genuinely future
            t_rem,                      # not in the input
            width / spot,               # derivable: sanity check, scale-free
            np.log(spot / spot_0),      # derivable: sanity check, scale-free
        ))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def build_dataset(series: str = "KXBTCD", max_events: int = 40, window: int = 24,
                  grid_n: int = 24, stride: int = 4, max_pages: int = 12,
                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """-> obs (N, window, grid_n, N_CHANNELS), targets (N, 4), event names.

    Events are kept whole so a by-event split is possible downstream; mixing
    windows from one event across train and test would leak.
    """
    events = kalshi.settled_events(series, max_pages=max_pages)
    names = sorted(events)[::-1][:max_events]

    X, Y, owner = [], [], []
    for i, name in enumerate(names):
        rows = events[name]
        settle = kalshi.event_settlement(rows)
        built = build_event(series, rows, grid_n=grid_n)
        if built is None or settle is None:
            if verbose:
                print("  skip {} (insufficient usable ladder)".format(name))
            continue
        obs, ts, state = built
        x, y = windows_from_event(obs, ts, state, settle, window, stride)
        if len(x) == 0:
            continue
        X.append(x)
        Y.append(y)
        owner.extend([name] * len(x))
        if verbose:
            print("  [{:>3}/{}] {}  minutes={:>2}  windows={:>2}  settle={:.0f}".format(
                i + 1, len(names), name, obs.shape[0], len(x), settle))

    if not X:
        raise RuntimeError("no usable events; check Gate 0a")
    return np.concatenate(X), np.concatenate(Y), owner


def normalise_targets(y: np.ndarray) -> np.ndarray:
    """Standardise each target so probe R^2 is comparable across them."""
    mu = y.mean(axis=0, keepdims=True)
    sd = y.std(axis=0, keepdims=True) + 1e-8
    return (y - mu) / sd
