#!/usr/bin/env python3
"""Assemble the cached per-event arrays into training windows.

Reads only from `data_cache/`, so this runs offline and instantly once
`build_corpus.py` has snapshotted. Splits are BY EVENT: windows from one event
must never straddle train and test, since consecutive windows within an event
share most of their minutes and would leak.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa.corpus import dataset  # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parents[1] / "data_cache" / "event_arrays"


def load_events():
    """-> list of (obs, ts, state, settle) from cache, in arbitrary order."""
    out = []
    for f in sorted(CACHE.glob("*.npz")):
        try:
            with np.load(f) as z:
                out.append((z["obs"], z["ts"], z["state"], float(z["settle"][0])))
        except (ValueError, OSError, KeyError):
            continue
    return out


def build(window=24, stride=4, min_minutes=None):
    min_minutes = min_minutes or window
    X, Y, owner = [], [], []
    n_short = 0
    for i, (obs, ts, state, settle) in enumerate(load_events()):
        if obs.shape[0] < min_minutes:
            n_short += 1
            continue
        x, y = dataset.windows_from_event(obs, ts, state, settle, window, stride)
        if len(x) == 0:
            continue
        X.append(x)
        Y.append(y)
        owner.extend([i] * len(x))
    if not X:
        raise RuntimeError("no windows; run build_corpus.py first")
    return np.concatenate(X), np.concatenate(Y), np.array(owner), n_short


def split_by_event(owner, frac=0.2, seed=0):
    """Hold out whole events. Returns boolean masks (train, test)."""
    ids = np.unique(owner)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * frac))
    test_ids = set(ids[:n_test].tolist())
    is_test = np.array([o in test_ids for o in owner])
    return ~is_test, is_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    X, Y, owner, n_short = build(args.window, args.stride)
    tr, te = split_by_event(owner)

    print("corpus assembled from cache")
    print("  windows      {:>8,}   obs {}".format(len(X), X.shape))
    print("  events       {:>8,}   ({} too short for window={})".format(
        len(np.unique(owner)), n_short, args.window))
    print("  train/test   {:>8,} / {:,}  (split by event)".format(tr.sum(), te.sum()))
    print("  memory       {:>8.1f} MB".format(X.nbytes / 1e6))
    print()
    for i, n in enumerate(dataset.TARGET_NAMES):
        print("  {:<22} mean {:>12.5f}  std {:>11.5f}".format(n, Y[:, i].mean(), Y[:, i].std()))

    p = pathlib.Path(__file__).resolve().parent / "results" / "corpus_shape.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({
        "windows": int(len(X)), "obs_shape": list(X.shape),
        "events": int(len(np.unique(owner))), "events_too_short": n_short,
        "train": int(tr.sum()), "test": int(te.sum()),
        "window": args.window, "stride": args.stride,
    }, indent=2))
    return X, Y, owner


if __name__ == "__main__":
    main()
