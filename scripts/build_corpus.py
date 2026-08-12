#!/usr/bin/env python3
"""Snapshot the free Kalshi corpus to local cache, before it is purged.

Measured 2026-08-04 (results/row_cutoff.json): /events lists 11,076 settled BTC
events back to 2024-06, but /markets returns rows for only the most recent
**1,677** of them, cutoff 2026-05-25. Strike values live on market rows, so the
usable corpus is those 1,677 per series and it is a ROLLING window: roughly 168
events fall off the back every week.

That makes this a snapshot job, not a query. Once an event is cached it is ours
permanently; every week we delay costs a week of history that cannot be
recovered at any price, since no vendor sells Kalshi backfill either.

    python3 build_corpus.py --series KXBTCD KXETHD


Vendored from pm-jepa so this repository can rebuild its own corpus. Reaches
Kalshi's UNAUTHENTICATED public API; no key or account is required. Writes
to  at the repository root, which  excludes.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa.corpus import cache, dataset, kalshi  # noqa: E402

ARRAY_VERSION = 1        # bump when build_event's output changes


def event_arrays(series, event_ticker, grid_n, step_frac, refresh=False):
    """Cached (obs, ts, state, settle) for one event. None if unusable."""
    params = {"series": series, "event": event_ticker,
              "grid_n": grid_n, "step_frac": step_frac}

    def compute():
        rows = kalshi._get("markets", event_ticker=event_ticker,
                           limit=1000).get("markets", [])
        if not rows:
            return None
        settle = kalshi.event_settlement(rows)
        built = dataset.build_event(series, rows, grid_n=grid_n, step_frac=step_frac)
        if built is None or settle is None:
            return None
        obs, ts, state = built
        return {"obs": obs, "ts": ts, "state": state,
                "settle": np.array([settle], dtype=np.float64)}

    return cache.array_stage("event_arrays", ARRAY_VERSION, params, compute, refresh)


def usable_events(series):
    """Events that still have market rows, newest first."""
    def compute():
        cur, evs = None, []
        while True:
            d = kalshi._get("events", series_ticker=series, status="settled",
                            limit=200, cursor=cur)
            got = d.get("events", [])
            evs.extend(got)
            cur = d.get("cursor")
            if not cur or not got:
                break
        evs = [e for e in evs if e.get("strike_date")]
        evs.sort(key=lambda e: e["strike_date"], reverse=True)
        return [{"event_ticker": e["event_ticker"], "strike_date": e["strike_date"]}
                for e in evs]

    # Short version so the listing refreshes as new events settle.
    return cache.json_stage("event_list", 1, {"series": series, "day": time.strftime("%Y-%m-%d")},
                            compute)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=["KXBTCD"])
    ap.add_argument("--max-events", type=int, default=2000)
    ap.add_argument("--grid-n", type=int, default=24)
    ap.add_argument("--step-frac", type=float, default=0.002)
    ap.add_argument("--stop-after-misses", type=int, default=40,
                    help="consecutive row-less events before assuming the purge cutoff")
    args = ap.parse_args()

    summary = {}
    for series in args.series:
        evs = usable_events(series)[:args.max_events]
        print("{}: {} settled events listed".format(series, len(evs)), flush=True)

        kept, skipped, misses, errors, t0 = 0, 0, 0, 0, time.time()
        for i, e in enumerate(evs):
            try:
                got = event_arrays(series, e["event_ticker"], args.grid_n, args.step_frac)
            except Exception as exc:
                # One malformed event must not kill an unattended snapshot that
                # is racing a rolling purge. Log it and keep going.
                errors += 1
                if errors <= 5:
                    print("  error on {}: {}".format(e["event_ticker"], exc), flush=True)
                got = None
            if got is None:
                skipped += 1
                misses += 1
                if misses >= args.stop_after_misses:
                    print("  {} consecutive unusable events at {}; assuming purge cutoff"
                          .format(misses, e["strike_date"][:10]), flush=True)
                    break
            else:
                kept += 1
                misses = 0
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print("  [{:>5}/{}] kept={:<5} skipped={:<5} {:.0f}s ({:.2f}s/event)"
                      .format(i + 1, len(evs), kept, skipped, el, el / (i + 1)), flush=True)

        summary[series] = {"listed": len(evs), "kept": kept, "skipped": skipped,
                           "errors": errors, "seconds": round(time.time() - t0, 1)}
        print("{}: kept {} events, skipped {}, {:.0f}s".format(
            series, kept, skipped, time.time() - t0), flush=True)

    p = pathlib.Path(__file__).resolve().parent / "results" / "corpus.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"summary": summary, "cache": cache.stats()}, indent=2))
    print("cache: {}".format(cache.stats()), flush=True)


if __name__ == "__main__":
    main()
