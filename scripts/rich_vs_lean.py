#!/usr/bin/env python3
"""E9: does keeping the discarded candle fields change anything?

Finding 18 established that the corpus reads 2 of the 10 fields each candle
carries, dropping the intra-minute bid and ask ranges, and that E7's headline
targets were volatility targets. This asks whether that mattered.

`causaljepa/richcorpus.py` rebuilds both feature sets from ONE fetch, through
slate's own ladder builder, validator, spot, width and resampler. The lean arm is
asserted bit-identical to the committed corpus on overlapping events before any
number below is believed. Identical minutes, identical windows, identical
targets; the only difference is how many channels survive ingest.

WIDTH IS A CONFOUND AND IS CONTROLLED. Rich has 10 channels against lean's 4, so
it is wider at every readout, and E4 exists because width alone lifts ridge on
structured input. `lean_tiled` repeats the lean channels to the same 10, carrying
identical information at identical dimension. Rich beating lean is not the claim.
**Rich beating `lean_tiled` is the claim.**

PRECONDITION, asserted: the lean arm must reproduce the corpus's own number on
`vol_forecast_error`. The previous attempt at this experiment returned -0.5459
where the corpus gets +0.1487, and reported a rich-versus-lean delta anyway. A
control arm that misses by 0.7 invalidates everything measured beside it.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from causaljepa import data as cj_data, probes, richcorpus as rc  # noqa: E402

# The corpus's own raw_last_min ridge R^2 on vol_forecast_error, from
# results/hidden_state_targets.json. The lean arm should land near it; a wide
# miss means the rebuild is not the corpus and nothing else here is meaningful.
LEAN_REFERENCE = 0.1487
LEAN_TOLERANCE = 0.12


def readouts(X):
    n, t, k, c = X.shape
    return {"last_min": X[:, -1].reshape(n, k * c),
            "last_4min": X[:, -4:].reshape(n, 4 * k * c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="results/rich_vs_lean.json")
    args = ap.parse_args()

    arms = {}
    for arm in ("lean", "rich"):
        X, Y, owner, names = rc.windows(
            window=args.window, stride=args.stride, horizon=args.horizon, arm=arm)
        arms[arm] = (X, Y, owner)
    (XL, Y, owner), (XR, Y2, owner2) = arms["lean"], arms["rich"]
    assert np.array_equal(owner, owner2) and np.allclose(Y, Y2), (
        "lean and rich disagree about which windows exist; they must come from "
        "identical minutes for this comparison to mean anything")

    # Width-matched control: same information as lean, same dimension as rich.
    reps = XR.shape[3] // XL.shape[3] + (1 if XR.shape[3] % XL.shape[3] else 0)
    XT = np.concatenate([XL] * reps, axis=3)[:, :, :, :XR.shape[3]]

    tr, te = cj_data.split_by_event(owner)
    print("lean {}  rich {}  tiled {}".format(XL.shape, XR.shape, XT.shape))
    print("events {}  train {} / test {}\n".format(
        len(np.unique(owner)), tr.sum(), te.sum()), flush=True)

    res = {"config": vars(args), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "events": int(len(np.unique(owner))), "target_names": names,
           "lean_reference": LEAN_REFERENCE, "arms": {}}

    print("%-12s%-11s%6s" % ("features", "readout", "dim")
          + "".join("%22s" % n[:21] for n in names))
    print("-" * 112)
    for tag, X in (("lean", XL), ("lean_tiled", XT), ("rich", XR)):
        for rn, f in readouts(X).items():
            f = f.astype(np.float64)
            t0 = time.time()
            r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
            res["arms"]["{}|{}".format(tag, rn)] = {
                "features": tag, "readout": rn, "dim": int(f.shape[1]),
                "ridge": r["ridge"], "mlp": r["mlp"],
                "seconds": round(time.time() - t0, 1)}
            print("%-12s%-11s%6d" % (tag, rn, f.shape[1])
                  + "".join("%+22.4f" % r["ridge"][n] for n in names), flush=True)
            del f

    lean_vfe = max(v["ridge"]["vol_forecast_error"] for v in res["arms"].values()
                   if v["features"] == "lean")
    res["lean_check"] = {"best_lean_vfe": lean_vfe, "reference": LEAN_REFERENCE,
                         "tolerance": LEAN_TOLERANCE,
                         "ok": abs(lean_vfe - LEAN_REFERENCE) < LEAN_TOLERANCE}
    print("\nlean check: best lean vol_forecast_error {:+.4f} against corpus "
          "reference {:+.4f}".format(lean_vfe, LEAN_REFERENCE))
    assert res["lean_check"]["ok"], (
        "LEAN ARM DOES NOT REPRODUCE THE CORPUS: {:+.4f} against {:+.4f}. This is "
        "the failure that invalidated the first attempt. Nothing else here is "
        "interpretable.".format(lean_vfe, LEAN_REFERENCE))

    def best(tag, t):
        return max(v["ridge"][t] for v in res["arms"].values() if v["features"] == tag)

    print("\n{:<24}{:>12}{:>12}{:>12}{:>16}{:>18}".format(
        "target", "lean", "tiled", "rich", "rich-lean", "rich-tiled"))
    print("-" * 96)
    res["deltas"] = {}
    for t in names:
        l, ti, r = best("lean", t), best("lean_tiled", t), best("rich", t)
        res["deltas"][t] = {"lean": l, "lean_tiled": ti, "rich": r,
                            "rich_minus_lean": r - l, "rich_minus_tiled": r - ti}
        print("{:<24}{:>+12.4f}{:>+12.4f}{:>+12.4f}{:>+16.4f}{:>18.4f}".format(
            t, l, ti, r, r - l, r - ti))
    print("-" * 96)

    head = "vol_forecast_error"
    gain = res["deltas"][head]["rich_minus_tiled"]
    res["verdict"] = {
        "headline_target": head, "width_matched_gain": gain,
        "helps": bool(gain > 0.02),
        "statement": (
            "THE DISCARDED FIELDS HELP: keeping them lifts {} by {:+.4f} over a "
            "width-matched tiling of the four channels the corpus keeps. The "
            "negative result was measured on an impoverished input.".format(head, gain)
            if gain > 0.02 else
            "THE DISCARDED FIELDS DO NOT HELP: {:+.4f} on {} against a "
            "width-matched control. The corpus was thin but the thinness was not "
            "what held the result back.".format(gain, head))}
    print("VERDICT: {}".format(res["verdict"]["statement"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
