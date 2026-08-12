"""Two correlated assets, one shared latent: the simulator's winning condition, on real data.

WHY THIS EXISTS. The simulator beats raw features (+0.9014 against +0.8142) and beats them by more
as noise rises. The Kalshi corpus loses everywhere. Findings 11 and 12 ruled out the obvious
explanations, and finding 14 ruled out the target set. What is left is a structural difference
nobody has been able to remove:

  * The simulator observes a 3-dimensional latent through EIGHT HETEROGENEOUS MARKETS, a moneyline,
    four spread lines and three totals. No single market identifies the latent. Combining them is
    the only way to recover it, which is exactly the job a representation learner exists to do.

  * A Kalshi BTC strike ladder is 24 strikes of ONE asset, a smooth near-deterministic function of
    TWO numbers, spot and width. Raw features already carry both. There is no latent left to infer,
    so a compressed embedding can only lose information relative to the raw input.

That is not a defect of the objective, it is an absence of the thing the objective consumes.

THE OBSERVATION THAT MAKES THIS RUNNABLE. The cached corpus is not one asset. It holds 1,675
KXBTCD events and 1,661 KXETHD events over the same ten weeks, and 99.1% of the BTC events have a
time-overlapping ETH event. Bitcoin and Ethereum share a latent (crypto beta, risk sentiment) and
carry idiosyncratic components on top of it, and NEITHER LADDER ALONE IDENTIFIES THE SHARED FACTOR.
That is the simulator's geometry, reproduced on real market data.

It is also, word for word, the last untested item in `docs/RESEARCH_PLAN.md` and in the published
post: "the settlement of a different correlated market".

RECOVERING THE SERIES LABEL COSTS NOTHING. `build_corpus.py` keys each cached event by
`sha256({stage, version, params})[:20]` where params includes the series and the event ticker, so
re-deriving which npz belongs to which asset is a pure hash recomputation against the cached event
list. No API calls, no network, no re-download.

WHAT IS GENUINELY HIDDEN HERE, and it is the point. A strike ladder is a MARGINAL distribution: it
prices where one asset lands. The DEPENDENCE between two assets is in neither ladder at any strike.
Realised correlation, and the error in the implied volatility RATIO, are joint properties that no
single cross-section identifies and that raw features therefore cannot simply read off. Whether
they are recoverable at all is an empirical question, which is why `scripts/horizon_headroom.py`
runs against these targets before anything is trained.
"""
import bisect
import hashlib
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# Must match build_corpus.py::ARRAY_VERSION and its default grid params, or no
# key resolves and the index comes back empty.
ARRAY_VERSION = 1
GRID_N = 24
STEP_FRAC = 0.002

CROSS_TARGET_NAMES = (
    "vol_ratio_forecast_error",   # headline: both ladders' forecasts divided out
    "realised_corr",              # joint dependence, in neither marginal
    "beta_residual_return",       # ETH idiosyncratic future move
    "eth_vol_forecast_error",     # single-asset comparison point
    "fwd_abs_spread_return",      # magnitude of the relative move
    "fwd_signed_spread_return",   # negative control: should be unpredictable
)

_EPS = 1e-6


def _key(stage: str, version: int, params: dict) -> str:
    blob = json.dumps({"stage": stage, "version": version, "params": params},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def series_index(root, grid_n: int = GRID_N, step_frac: float = STEP_FRAC
                 ) -> Dict[str, Tuple[str, str]]:
    """-> {cache_key: (series, event_ticker)} for every resolvable cached event.

    Pure recomputation of `build_corpus.py`'s cache keys against the cached
    event list. Any npz whose key does not resolve is left out rather than
    guessed at, and the caller is told how many that was.
    """
    root = pathlib.Path(root).expanduser().resolve()
    listing = root / "data_cache" / "event_list"
    arrays = root / "data_cache" / "event_arrays"
    if not listing.is_dir() or not arrays.is_dir():
        raise FileNotFoundError(
            "need both data_cache/event_list and data_cache/event_arrays under "
            "{}; the series label is recovered from the event list".format(root))

    have = {p.stem for p in arrays.glob("*.npz")}
    tickers: Dict[str, set] = {}
    for f in sorted(listing.glob("*.json")):
        try:
            rows = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(rows, list):
            rows = rows.get("markets") or rows.get("events") or []
        for r in rows:
            t = r.get("event_ticker") or r.get("ticker") or ""
            m = re.match(r"^(KX[A-Z]+)", t)
            if m:
                tickers.setdefault(m.group(1), set()).add(t)

    out = {}
    for ser, ts in tickers.items():
        for t in ts:
            k = _key("event_arrays", ARRAY_VERSION,
                     {"series": ser, "event": t, "grid_n": grid_n,
                      "step_frac": step_frac})
            if k in have:
                out[k] = (ser, t)
    return out


def _load(root, key):
    p = pathlib.Path(root) / "data_cache" / "event_arrays" / (key + ".npz")
    try:
        with np.load(p) as z:
            return z["obs"], z["ts"], z["state"], float(z["settle"][0])
    except (ValueError, OSError, KeyError):
        return None


def pair_events(root, series_a: str = "KXBTCD", series_b: str = "KXETHD",
                min_shared: int = 34) -> List[dict]:
    """Pair each `series_a` event with the `series_b` event it shares most minutes with.

    Pairing is on the INTERSECTION OF TIMESTAMPS, not on nominal event windows.
    Minutes that failed `validate_ladder` were dropped when each event was
    cached, independently per asset, so two events that nominally cover the same
    hour can disagree about which minutes exist. Aligning on anything other than
    the actual shared timestamps would silently pair a BTC minute with a
    different ETH minute, which manufactures cross-asset structure that is not
    there.

    One-to-one, and matched GLOBALLY BY OVERLAP rather than greedily in
    iteration order. Every candidate pair is scored first, then accepted
    strongest-first. Matching in iteration order lets an event that overlaps by
    a handful of minutes claim a partner and block the correct hour-aligned
    pair, which cost 39% of the pairs in the first version of this function.
    Sorting by shared-minute count makes the result independent of the order the
    cache happens to enumerate keys in.
    """
    idx = series_index(root)
    a_keys = [k for k, (s, _t) in idx.items() if s == series_a]
    b_keys = [k for k, (s, _t) in idx.items() if s == series_b]

    b_meta = []
    for k in b_keys:
        got = _load(root, k)
        if got is None:
            continue
        b_meta.append((int(got[1][0]), int(got[1][-1]), k, got[1]))
    b_meta.sort()
    b_starts = [m[0] for m in b_meta]
    # Longest event observed in this corpus is 350 minutes; bound the backward
    # scan generously rather than assuming ends are sorted, which they are not
    # when the list is sorted by start.
    MAX_EVENT_S = 400 * 60

    # Score every candidate pair whose time ranges touch at all.
    cands = []
    for k in sorted(a_keys):
        got = _load(root, k)
        if got is None:
            continue
        ts_a = got[1]
        lo, hi = int(ts_a[0]), int(ts_a[-1])
        j_hi = bisect.bisect_right(b_starts, hi)
        j_lo = bisect.bisect_left(b_starts, lo - MAX_EVENT_S)
        for j in range(j_lo, j_hi):
            if b_meta[j][1] < lo:
                continue
            n = int(len(np.intersect1d(ts_a, b_meta[j][3], assume_unique=False)))
            if n >= min_shared:
                cands.append((n, k, b_meta[j][2]))

    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_a, used_b, pairs = set(), set(), []
    for n, ka, kb in cands:
        if ka in used_a or kb in used_b:
            continue
        used_a.add(ka)
        used_b.add(kb)
        pairs.append({"a_key": ka, "b_key": kb, "n_shared": n,
                      "a_ticker": idx[ka][1], "b_ticker": idx[kb][1]})
    return pairs


def _aligned(root, pair):
    """-> (obs_a, obs_b, state_a, state_b, ts) restricted to shared timestamps."""
    ga, gb = _load(root, pair["a_key"]), _load(root, pair["b_key"])
    if ga is None or gb is None:
        return None
    obs_a, ts_a, st_a, _ = ga
    obs_b, ts_b, st_b, _ = gb
    shared, ia, ib = np.intersect1d(ts_a, ts_b, return_indices=True)
    order = np.argsort(shared)
    return (obs_a[ia[order]], obs_b[ib[order]],
            st_a[ia[order]], st_b[ib[order]], shared[order])


def windows_from_pair(obs_a, obs_b, st_a, st_b, ts, window: int, stride: int,
                      horizon: int, max_gap_factor: float = 2.0):
    """-> (X, Y) where X is (n, window, 2*K, C): both ladders side by side.

    The two ladders are concatenated along the STRIKE axis rather than stacked on
    a new one, so the existing factorised encoder needs no change and its strike
    attention, which is already bidirectional because a ladder is a cross-section
    rather than a sequence, now spans both assets. That is precisely the
    cross-market attention the simulator's heterogeneous menu forces and a single
    BTC ladder cannot.
    """
    T = obs_a.shape[0]
    if T < window:
        return None
    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        end = start + window - 1
        fut = end + horizon
        if fut > T - 1:
            continue
        sa, wa, ra = (float(v) for v in st_a[end])
        sb, wb, rb = (float(v) for v in st_b[end])
        if min(sa, wa, ra, sb, wb, rb) <= 0.0:
            continue
        dt = float(ts[fut] - ts[end])
        if dt <= 0.0 or dt > max_gap_factor * horizon * 60.0:
            continue

        pa = np.asarray(st_a[end:fut + 1, 0], dtype=np.float64)
        pb = np.asarray(st_b[end:fut + 1, 0], dtype=np.float64)
        if (pa <= 0).any() or (pb <= 0).any():
            continue
        ra_r, rb_r = np.diff(np.log(pa)), np.diff(np.log(pb))
        rv_a = float(np.sqrt((ra_r ** 2).sum()))
        rv_b = float(np.sqrt((rb_r ** 2).sum()))
        iv_a = float(wa / sa * np.sqrt(dt / ra))
        iv_b = float(wb / sb * np.sqrt(dt / rb))

        # Realised correlation over the forward window. Undefined when either
        # path is flat, which does happen on quiet minutes; those windows are
        # dropped rather than imputed to zero, because a fabricated zero would
        # be indistinguishable from a real one to the probe.
        sda, sdb = ra_r.std(), rb_r.std()
        if sda <= 1e-12 or sdb <= 1e-12:
            continue
        corr = float(np.clip(((ra_r - ra_r.mean()) * (rb_r - rb_r.mean())).mean()
                             / (sda * sdb), -0.999, 0.999))

        fwd_a = float(np.log(pa[-1] / pa[0]))
        fwd_b = float(np.log(pb[-1] / pb[0]))
        # Beta from the WITHIN-WINDOW path only, so nothing after window end
        # enters the predictor. The residual is then a genuinely future
        # idiosyncratic move rather than a fitted artefact.
        wa_r = np.diff(np.log(np.asarray(st_a[start:end + 1, 0], dtype=np.float64)))
        wb_r = np.diff(np.log(np.asarray(st_b[start:end + 1, 0], dtype=np.float64)))
        var_a = float((wa_r ** 2).sum())
        beta = float((wa_r * wb_r).sum() / var_a) if var_a > 1e-18 else 1.0
        beta = float(np.clip(beta, -5.0, 5.0))

        xs.append(np.concatenate([obs_a[start:start + window],
                                  obs_b[start:start + window]], axis=1))
        ys.append((
            np.log((rv_b + _EPS) / (iv_b + _EPS)) - np.log((rv_a + _EPS) / (iv_a + _EPS)),
            corr,
            fwd_b - beta * fwd_a,
            np.log((rv_b + _EPS) / (iv_b + _EPS)),
            abs(fwd_b - fwd_a),
            fwd_b - fwd_a,
        ))
    if not xs:
        return None
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def load_corpus(root=None, window: int = 24, stride: int = 4, horizon: int = 10,
                max_gap_factor: float = 2.0):
    """-> (X, Y, owner, target_names, stats). X is (N, window, 48, C)."""
    from . import data as cj_data
    root = cj_data._resolve_root(root)
    pairs = pair_events(root, min_shared=window + horizon)

    X, Y, owner = [], [], []
    n_used = 0
    for i, pr in enumerate(pairs):
        al = _aligned(root, pr)
        if al is None:
            continue
        built = windows_from_pair(al[0], al[1], al[2], al[3], al[4],
                                  window, stride, horizon, max_gap_factor)
        if built is None:
            continue
        x, y = built
        X.append(x)
        Y.append(y)
        owner.extend([i] * len(x))
        n_used += 1
    if not X:
        raise RuntimeError("no cross-asset windows survived")

    stats = {"pairs_found": len(pairs), "pairs_contributing": n_used,
             "window": window, "stride": stride, "horizon_steps": horizon,
             "max_gap_factor": max_gap_factor}
    return (np.concatenate(X), np.concatenate(Y), np.array(owner),
            list(CROSS_TARGET_NAMES), stats)
