#!/usr/bin/env python3
"""Rebuild the corpus keeping the candle fields the original builder discards.

Roughly three unauthenticated API calls per event, so a full two-series rebuild
is on the order of 10,000 requests. Every event is cached individually, so an
interruption costs only the event in flight and a re-run resumes.

    python3 scripts/build_rich_corpus.py --series KXBTCD,KXETHD --max-events 2000

Writes into THIS repository's `data_cache/event_arrays_rich/`, never into
pm-jepa, per docs/RESEARCH_PLAN.md rule 7.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa import richcorpus as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTCD,KXETHD")
    ap.add_argument("--max-events", type=int, default=2000)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="pause between events; the endpoint is public and "
                         "unauthenticated, so do not hammer it")
    args = ap.parse_args()

    kalshi, _slate = rc._slate()
    total_ok = total_miss = total_cached = 0
    t0 = time.time()

    for series in [s for s in args.series.split(",") if s]:
        events = kalshi.settled_events(series, max_pages=args.max_pages)
        names = sorted(events)[::-1][:args.max_events]
        print("\n{}: {} settled events".format(series, len(names)), flush=True)
        ok = miss = 0
        for i, name in enumerate(names):
            before = rc.cache_dir() / "_"
            try:
                got = rc.cached_build(series, name)
            except Exception as e:
                print("  [{:>4}/{}] {} FAILED {}: {}".format(
                    i + 1, len(names), name, type(e).__name__, e), flush=True)
                time.sleep(1.0)
                continue
            if got is None:
                miss += 1
            else:
                ok += 1
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                rate = (i + 1) / max(el, 1e-9)
                print("  [{:>4}/{}] usable {}  skipped {}  {:.1f} ev/s  "
                      "elapsed {:.0f}s".format(i + 1, len(names), ok, miss,
                                               rate, el), flush=True)
            time.sleep(args.sleep)
        print("  {} done: {} usable, {} skipped".format(series, ok, miss), flush=True)
        total_ok += ok
        total_miss += miss

    n = len(list(rc.cache_dir().glob("*.npz")))
    print("\ncache now holds {} events at {}".format(n, rc.cache_dir()), flush=True)
    print("usable {}  skipped {}  in {:.0f}s".format(
        total_ok, total_miss, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
