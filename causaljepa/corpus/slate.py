"""Strike ladders, no-arbitrage validation, and slate tensors.

A Kalshi threshold event is a ladder of digital calls on one underlying:

    market at strike K  pays $1 iff  S_T > K

so the vector of prices across K *is* the risk-neutral survival function
P(S_T > K). That gives three constraints which no matching engine enforces,
because each strike is a separate order book:

    1. monotone      P(S_T > K) is non-increasing in K
    2. mass          the drop across the quoted range is <= 1
    3. density >= 0  the discrete second difference is non-negative

Violations are real, tradeable, and measurable. They are also a research
output in their own right, so `validate_ladder` reports rates rather than
silently repairing anything.

IMPORTANT: threshold ladders and range brackets need DIFFERENT tests. A
bracket ladder is a density, so it should be unimodal and sum to 1, not
monotone. Applying the monotone test to brackets produces spurious
"violations" at the mode. `kind` selects the right test.
"""

# Vendored from the pm-jepa corpus builder so this repository stands alone.
# Unmodified except for the import rewrite noted below and this header. The
# corpus is reconstructed from Kalshi's UNAUTHENTICATED public API; no key,
# account or credential is involved at any point.

from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from . import kalshi


class Rung(NamedTuple):
    strike: float
    bid: float
    ask: float
    mid: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    volume: float = 0.0          # candles carry volume/OI but no depth
    open_interest: float = 0.0


class LadderReport(NamedTuple):
    n_rungs: int
    n_quoted: int
    monotone_violations: int
    monotone_pairs: int
    implied_mass: float
    negative_density: int
    density_points: int
    median_depth: float

    @property
    def ok(self) -> bool:
        """Pass criteria for Gate 0a, stated numerically."""
        if self.n_quoted < 8:
            return False
        if self.monotone_pairs and self.monotone_violations / self.monotone_pairs > 0.05:
            return False
        if not (0.80 <= self.implied_mass <= 1.20):
            return False
        return True


def _informative(bid: Optional[float], ask: Optional[float]) -> bool:
    """Reject empty books quoting the full 0..1 range, which carry no information."""
    if bid is None or ask is None:
        return False
    if ask <= 0:
        return False
    if bid <= 0.0 and ask >= 1.0:
        return False
    return True


def ladder_from_rows(rows: Sequence[Dict]) -> List[Rung]:
    """Build a strike-sorted ladder from raw market rows."""
    out = []
    for r in rows:
        k = kalshi.strike_of(r)
        b, a = kalshi.quote(r)
        if k is None or not _informative(b, a):
            continue
        out.append(Rung(
            strike=k, bid=b, ask=a, mid=(b + a) / 2.0,
            bid_size=kalshi.to_float(r.get("yes_bid_size_fp")) or 0.0,
            ask_size=kalshi.to_float(r.get("yes_ask_size_fp")) or 0.0,
        ))
    return sorted(out, key=lambda x: x.strike)


def validate_ladder(ladder: Sequence[Rung], n_rungs: int,
                    kind: str = "threshold") -> LadderReport:
    """Measure the three no-arbitrage constraints. Reports, never repairs."""
    p = [r.mid for r in ladder]
    n = len(p)
    if n < 3:
        return LadderReport(n_rungs, n, 0, 0, float("nan"), 0, 0, 0.0)

    if kind == "threshold":
        viol = sum(1 for i in range(n - 1) if p[i + 1] > p[i] + 1e-9)
        pairs = n - 1
        mass = sum(max(p[i] - p[i + 1], 0.0) for i in range(n - 1))
        neg = sum(1 for i in range(1, n - 1)
                  if (p[i - 1] - p[i]) - (p[i] - p[i + 1]) < -1e-6)
        pts = max(n - 2, 0)
    elif kind == "bracket":
        # A density: no monotonicity requirement. Mass is a direct sum, and
        # "negative density" cannot occur since prices are non-negative, so we
        # count multi-modality instead as the shape diagnostic.
        viol, pairs = 0, 0
        mass = sum(p)
        turns = sum(1 for i in range(1, n - 1)
                    if (p[i] - p[i - 1]) * (p[i + 1] - p[i]) < 0)
        neg, pts = max(turns - 1, 0), max(n - 2, 0)
    else:
        raise ValueError("kind must be 'threshold' or 'bracket'")

    depths = sorted(r.bid_size for r in ladder)
    median_depth = depths[len(depths) // 2] if depths else 0.0
    return LadderReport(n_rungs, n, viol, pairs, mass, neg, pts, median_depth)


def ladders_from_candles(rows: Sequence[Dict], candles_by_ticker: Dict[str, List[Dict]]
                         ) -> Dict[int, List[Rung]]:
    """Reconstruct the ladder at every minute of a settled event.

    Necessary because a SETTLED market's live quote fields are zeroed out: only
    `expiration_value` survives on the row. The book history lives in the
    candles, so historical training data has to be rebuilt from them.

    -> {unix_minute_ts: [Rung, ...]} with strikes sorted.
    """
    strike_by_ticker = {}
    for r in rows:
        k = kalshi.strike_of(r)
        if k is not None:
            strike_by_ticker[r["ticker"]] = k

    by_ts: Dict[int, List[Rung]] = {}
    for tkr, cs in candles_by_ticker.items():
        k = strike_by_ticker.get(tkr)
        if k is None:
            continue
        for c in cs:
            b, a = kalshi.candle_quote(c)
            if not _informative(b, a):
                continue
            t = c.get("end_period_ts")
            if t is None:
                continue
            by_ts.setdefault(int(t), []).append(Rung(
                strike=k, bid=b, ask=a, mid=(b + a) / 2.0,
                volume=kalshi.to_float(c.get("volume_fp")) or 0.0,
                open_interest=kalshi.to_float(c.get("open_interest_fp")) or 0.0,
            ))

    return {t: sorted(v, key=lambda x: x.strike) for t, v in by_ts.items()}


def implied_spot(ladder: Sequence[Rung]) -> Optional[float]:
    """Strike where P(S_T > K) crosses 0.5, linearly interpolated.

    This is a sanity-check probe target: it is recoverable from the input, so a
    healthy encoder should predict it near-perfectly. Treat a low R^2 here as
    evidence the encoder is broken, not as a finding.
    """
    for i in range(len(ladder) - 1):
        hi, lo = ladder[i], ladder[i + 1]
        if hi.mid >= 0.5 >= lo.mid:
            span = hi.mid - lo.mid
            if span <= 1e-9:
                return (hi.strike + lo.strike) / 2.0
            w = (hi.mid - 0.5) / span
            return hi.strike + w * (lo.strike - hi.strike)
    return None


def implied_width(ladder: Sequence[Rung]) -> Optional[float]:
    """Interquartile width of the implied distribution, a scale-free vol proxy.

    Distance between the strikes where the survival function crosses 0.75 and
    0.25. Wider means more uncertainty, which scales with sigma * sqrt(t).
    """
    def crossing(level):
        for i in range(len(ladder) - 1):
            hi, lo = ladder[i], ladder[i + 1]
            if hi.mid >= level >= lo.mid:
                span = hi.mid - lo.mid
                if span <= 1e-9:
                    return (hi.strike + lo.strike) / 2.0
                w = (hi.mid - level) / span
                return hi.strike + w * (lo.strike - hi.strike)
        return None

    q75, q25 = crossing(0.75), crossing(0.25)
    if q75 is None or q25 is None:
        return None
    return abs(q25 - q75)


def resample_ladder(ladder: Sequence[Rung], centre: float, n: int = 24,
                    step_frac: float = 0.002) -> np.ndarray:
    """Interpolate the ladder onto a fixed grid of RELATIVE moneyness.

    The model needs a rectangular (time x strike) token grid, but Kalshi's
    absolute strikes move between events as the underlying drifts. Anchoring on
    log-moneyness around `centre` makes events comparable and keeps the encoder
    from memorising price levels.

    Returns (n, 4): survival probability, quoted spread, log volume, log OI.
    """
    if not ladder or centre is None or centre <= 0:
        return np.zeros((n, 4), dtype=np.float32)

    ks = np.array([r.strike for r in ladder], dtype=np.float64)
    ps = np.array([r.mid for r in ladder], dtype=np.float64)
    sp = np.array([r.ask - r.bid for r in ladder], dtype=np.float64)
    vol = np.log1p(np.array([r.volume for r in ladder], dtype=np.float64))
    oi = np.log1p(np.array([r.open_interest for r in ladder], dtype=np.float64))

    offsets = (np.arange(n) - (n - 1) / 2.0) * step_frac
    grid = centre * np.exp(offsets)

    # np.interp needs increasing x; strikes already are. Survival is decreasing,
    # which is fine: interp works on the values, not their monotonicity.
    out = np.stack([
        np.interp(grid, ks, ps, left=ps[0], right=ps[-1]),
        np.interp(grid, ks, sp, left=0.0, right=0.0),
        np.interp(grid, ks, vol, left=0.0, right=0.0),
        np.interp(grid, ks, oi, left=0.0, right=0.0),
    ], axis=-1)
    return out.astype(np.float32)
