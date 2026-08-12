#!/usr/bin/env python3
"""E7b: headroom per target, per horizon, from raw features alone.

Two jobs, both cheap, neither requiring a single training step.

FIRST, it checks that E7's horizon of 10 is not cherry-picked. A target set
chosen after seeing which horizon flattered the result would be exactly the
failure this programme spent five experiments documenting in other people's
work, so the horizon dependence is measured and published rather than asserted.

SECOND, and more usefully outside this repo, it is the PRE-FLIGHT TEST. The
whole causal-jepa programme could have been aborted in an afternoon by the
measurement below, run before any encoder was written:

    fit raw features -> target, with ridge and with an MLP, and look at what is
    left over.

A target that raw features already explain has no headroom for a representation
to add anything, so any apparent gain on it is readout width rather than
learning. A target that NOTHING explains has no headroom either, for the
opposite reason, and cannot rank representations at all. Only targets in the
middle band can discriminate, and which band a target is in is knowable in
advance, for free.

That gives three verdicts per target, printed as `regime`:

    DERIVABLE      raw ridge R^2 >= DERIVABLE_BAR. The target is largely a
                   function of the input. Gains here measure reconstruction.
    UNPREDICTABLE  best raw R^2 <= NOISE_BAR. Nothing predicts it, so it cannot
                   separate a good representation from a bad one.
    DISCRIMINATIVE anything between. This is the only regime in which a probe
                   comparison carries information.
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
sys.path.insert(0, str(_ROOT / "scripts"))

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes, targets as cj_targets  # noqa: E402

import readout_ablation as e4  # noqa: E402

DERIVABLE_BAR = 0.60      # at or above this, the target is mostly the input
NOISE_BAR = 0.02          # at or below this, nothing predicts it


def regime(best_ridge: float, best_any: float) -> str:
    if best_ridge >= DERIVABLE_BAR:
        return "DERIVABLE"
    if best_any <= NOISE_BAR:
        return "UNPREDICTABLE"
    return "DISCRIMINATIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--horizons", default="5,10,15,20")
    ap.add_argument("--out", default="results/horizon_headroom.json")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h]
    res = {"config": vars(args), "derivable_bar": DERIVABLE_BAR,
           "noise_bar": NOISE_BAR, "horizons": {}}

    for h in horizons:
        t0 = time.time()
        X, Yo, Yn, owner, on, nn_, cstats = cj_targets.load_corpus(
            args.pm_jepa_root, horizon=h)
        tr, te = cj_data.split_by_event(owner)
        print("\nhorizon {:>3}: {:>6} windows over {:>5} events  "
              "(train {} / test {})".format(
                  h, len(X), len(np.unique(owner)), tr.sum(), te.sum()), flush=True)

        per = {}
        for cname, f in e4.raw_controls(X).items():
            f = f.astype(np.float64)
            for blk, Y, names in (("orig", Yo, on), ("new", Yn, nn_)):
                r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
                for t in names:
                    d = per.setdefault(t, {"block": blk, "by_control": {}})
                    d["by_control"][cname] = {"dim": int(f.shape[1]),
                                              "ridge": r["ridge"][t],
                                              "mlp": r["mlp"][t]}
            del f

        for t, d in per.items():
            rs = [v["ridge"] for v in d["by_control"].values()]
            ms = [v["mlp"] for v in d["by_control"].values()]
            d["best_ridge"] = float(max(rs))
            d["best_mlp"] = float(max(ms))
            d["best_any"] = float(max(max(rs), max(ms)))
            d["regime"] = regime(d["best_ridge"], d["best_any"])

        res["horizons"][str(h)] = {
            "corpus": cstats, "windows": int(len(X)),
            "events": int(len(np.unique(owner))),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "targets": per, "seconds": round(time.time() - t0, 1)}

        print("  {:<22}{:>12}{:>12}{:>18}".format(
            "target", "best ridge", "best mlp", "regime"))
        for t in list(on) + list(nn_):
            d = per[t]
            print("  {:<22}{:>+12.4f}{:>+12.4f}{:>18}".format(
                t, d["best_ridge"], d["best_mlp"], d["regime"]), flush=True)
        del X, Yo, Yn

    # ---- horizon stability of the headline target ------------------------
    head = "vol_forecast_error"
    band = [res["horizons"][str(h)]["targets"][head]["best_ridge"] for h in horizons]
    res["headline_stability"] = {
        "target": head, "horizons": horizons, "best_ridge_by_horizon": band,
        "min": float(min(band)), "max": float(max(band)),
        "range": float(max(band) - min(band)),
        "all_discriminative": all(
            res["horizons"][str(h)]["targets"][head]["regime"] == "DISCRIMINATIVE"
            for h in horizons)}

    w = 92
    print("\n" + "=" * w)
    print("E7b HORIZON HEADROOM  (raw features only, no training)")
    print("=" * w)
    print("{:<22}".format("target") + "".join("{:>16}".format("h=" + str(h)) for h in horizons))
    print("-" * w)
    all_names = None
    for h in horizons:
        all_names = all_names or list(res["horizons"][str(h)]["targets"].keys())
    for t in all_names:
        cells = []
        for h in horizons:
            d = res["horizons"][str(h)]["targets"][t]
            cells.append("{:+.4f} {}".format(d["best_ridge"], d["regime"][:4]))
        print("{:<22}".format(t) + "".join("{:>16}".format(c) for c in cells))
    print("=" * w)
    s = res["headline_stability"]
    print("{} across horizons {}: {:+.4f} to {:+.4f} (range {:.4f}), "
          "discriminative everywhere: {}".format(
              head, horizons, s["min"], s["max"], s["range"], s["all_discriminative"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
