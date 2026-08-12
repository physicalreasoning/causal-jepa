"""Kalshi public market-data client.

Everything here uses the UNAUTHENTICATED host `api.elections.kalshi.com`, which
serves market data, candlesticks, and settlement values for free. The
authenticated host (`trading-api.kalshi.com`) returns 401 and is not needed:
we never place orders.

Verified against the live API on 2026-08-04:
  - `KXBTCD` hourly events carry ~188 strikes, all two-sided quoted
  - 1-minute candlesticks are retained AFTER a market settles
  - settled markets expose `expiration_value`, the exact settlement price,
    which is our probe ground truth and costs nothing

Rate limits are token buckets per account tier. We are anonymous, so we stay
deliberately polite: a fixed minimum interval between calls plus bounded
exponential backoff on 429.
"""

# Vendored from the pm-jepa corpus builder so this repository stands alone.
# Unmodified except for the import rewrite noted below and this header. The
# corpus is reconstructed from Kalshi's UNAUTHENTICATED public API; no key,
# account or credential is involved at any point.

import calendar
import json
import time
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Optional

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MIN_INTERVAL = 0.12          # seconds between requests
MAX_RETRIES = 6
MAX_CANDLES_PER_RESPONSE = 9000   # documented ~10k; leave headroom

# Threshold ladders (strike_type="greater"), NOT the range brackets.
# Measured 2026-08-04: threshold books carry a median 6,941 contracts at the
# best bid; the matching range brackets (KXBTC/KXETH) had a median of 0.
SERIES = {
    "btc_hourly": "KXBTCD",
    "eth_hourly": "KXETHD",
}

_last_call = [0.0]


def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def _get(path: str, **params) -> Dict:
    """GET with throttling and bounded exponential backoff on 429/5xx."""
    q = "&".join("{}={}".format(k, v) for k, v in params.items() if v is not None)
    url = "{}/{}{}".format(BASE, path.lstrip("/"), ("?" + q) if q else "")
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                # strict=False: Kalshi rule text contains raw control characters
                return json.loads(r.read().decode(), strict=False)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def ts(iso: str) -> int:
    """ISO8601 -> unix seconds (Kalshi timestamps are UTC)."""
    return calendar.timegm(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S"))


def to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def iter_markets(series_ticker: str, status: str, max_pages: int = 50) -> Iterator[Dict]:
    """Paginate the markets endpoint. ~1000 markets/page, ~5 hourly events."""
    cursor = None
    for _ in range(max_pages):
        d = _get("markets", series_ticker=series_ticker, status=status,
                 limit=1000, cursor=cursor)
        rows = d.get("markets", [])
        for r in rows:
            yield r
        cursor = d.get("cursor")
        if not cursor or not rows:
            return


def settled_events(series_ticker: str, max_pages: int = 50) -> Dict[str, List[Dict]]:
    """Settled events -> their strike ladders, newest first.

    Settled markets are what we train on: their candles are retained and their
    `expiration_value` gives ground truth.
    """
    events: Dict[str, List[Dict]] = {}
    for r in iter_markets(series_ticker, "settled", max_pages=max_pages):
        events.setdefault(r["event_ticker"], []).append(r)
    return events


def event_settlement(rows: List[Dict]) -> Optional[float]:
    """The settled underlying price for an event, from `expiration_value`.

    Every market in an event shares one settlement value, so any row with a
    parseable value wins. Returns None if the event has not settled.
    """
    for r in rows:
        v = to_float(r.get("expiration_value"))
        if v is not None and v > 0:
            return v
    return None


def candles(series_ticker: str, market_ticker: str, start_ts: int, end_ts: int,
            interval: int = 1) -> List[Dict]:
    """1-minute OHLC of yes_bid / yes_ask, plus volume and open interest.

    The window must bracket the market's own open/close; a window far wider
    than the market's life returns HTTP 400.
    """
    d = _get("series/{}/markets/{}/candlesticks".format(series_ticker, market_ticker),
             start_ts=start_ts, end_ts=end_ts, period_interval=interval)
    return d.get("candlesticks", [])


def batch_candles(market_tickers: List[str], start_ts: int, end_ts: int,
                  interval: int = 1) -> Dict[str, List[Dict]]:
    """Candles for many markets at once. -> {ticker: [candle, ...]}.

    Undocumented in the public reference but live: `markets/candlesticks` takes
    `market_tickers` as a comma list. Capped at 100 tickers per request, so a
    188-strike event costs 2 calls instead of 188.

    The response order does NOT match the request order. Each entry carries its
    own `market_ticker` and that is what we key on. Zipping against the request
    list silently scrambles strikes against prices, which produces a ladder that
    violates monotonicity everywhere; `tests/test_kalshi.py::test_batch_candles_
    are_keyed_by_returned_ticker` locks this down.
    """
    # The cap is on TOTAL candles in the response (~10k), not on ticker count,
    # so the safe chunk size depends on how many periods the window spans.
    # 100 tickers x 60 minutes = 6k is fine; 100 x 360 = 36k returns an opaque
    # HTTP 400. This is what made every daily-cadence event (25h span, mixed
    # into the hourly series) fail while hourly events succeeded.
    n_periods = max(1, (end_ts - start_ts) // (60 * max(interval, 1)))
    chunk_size = max(1, min(100, int(MAX_CANDLES_PER_RESPONSE // n_periods)))

    out: Dict[str, List[Dict]] = {}
    for i in range(0, len(market_tickers), chunk_size):
        chunk = market_tickers[i:i + chunk_size]
        d = _get("markets/candlesticks", market_tickers=",".join(chunk),
                 start_ts=start_ts, end_ts=end_ts, period_interval=interval)
        for entry in d.get("markets", []):
            tkr = (entry or {}).get("market_ticker")
            if tkr:
                out[tkr] = entry.get("candlesticks", []) or []
    return out


def strike_of(row: Dict) -> Optional[float]:
    return to_float(row.get("floor_strike")) or to_float(row.get("cap_strike"))


def quote(row: Dict):
    """(bid, ask) in dollars, or (None, None) when the book is uninformative."""
    b, a = to_float(row.get("yes_bid_dollars")), to_float(row.get("yes_ask_dollars"))
    if b is None or a is None:
        return None, None
    return b, a


def candle_quote(c: Dict):
    """(bid, ask) at the close of a candle."""
    b = to_float((c.get("yes_bid") or {}).get("close_dollars"))
    a = to_float((c.get("yes_ask") or {}).get("close_dollars"))
    return b, a
