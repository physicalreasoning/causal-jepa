#!/usr/bin/env python3
"""E8a: do cross-asset targets have headroom, and do they need both ladders?

Raw features only. No training, no model, no GPU. This runs BEFORE any encoder
is trained, which is the whole lesson of finding 15: a target's regime is
knowable in advance, and three of the four targets this programme was originally
scored on were unusable in ways a one-minute check would have caught.

TWO QUESTIONS, and the second is the one that matters.

FIRST, the regime question, same as `horizon_headroom.py`. A target that raw
features already explain has no headroom for a representation to add anything. A
target nothing explains cannot rank representations either. Only the middle band
can discriminate.

SECOND, and this is the actual hypothesis: **does the second asset add anything
at all?** Every arm is run three ways on identical windows and identical targets:

    btc_only    the 24 BTC strikes                (K=24)
    eth_only    the 24 ETH strikes                (K=24)
    both        both ladders side by side         (K=48)

The claim motivating this whole experiment is that BTC and ETH share a latent
that NEITHER LADDER ALONE IDENTIFIES, which is the property that makes the
simulator work and that a single BTC ladder lacks. That claim is testable right
here, with no learning involved:

  * `both` clearly beats `max(btc_only, eth_only)`  -> the shared structure is
    real and is genuinely cross-sectional. A representation learner has
    something to find, and the experiment is worth training.
  * `both` matches the better single ladder        -> there is no cross-asset
    latent to recover, the premise is wrong, and training a JEPA here would
    repeat the last five experiments at greater cost. Stop.

The second outcome kills the idea for one minute of compute instead of an
afternoon, which is exactly what finding 15 says to do.

NOTE ON WIDTH. `both` has twice the features of either single arm, and width
alone helps ridge on structured input; E4 exists because of that confound. So
the comparison that counts is not `both` against a single ladder at half the
width. It is `both` against `eth_only`/`btc_only` AT MATCHED WIDTH, which is why
the doubled arm is reported: it duplicates ONE ladder to the same 48 strikes,
carrying the same information at the same dimension. If `both` cannot beat a
duplicated single ladder, its advantage is width and not the second asset.
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

from causaljepa import crossasset as ca  # noqa: E402
from causaljepa import data as cj_data, probes  # noqa: E402

DERIVABLE_BAR = 0.60
NOISE_BAR = 0.02


def regime(best_ridge, best_any):
    if best_ridge >= DERIVABLE_BAR:
        return "DERIVABLE"
    if best_any <= NOISE_BAR:
        return "UNPREDICTABLE"
    return "DISCRIMINATIVE"


def views(X):
    """-> {name: (N, T, K, C)}. Strikes 0..23 are BTC, 24..47 are ETH."""
    k = X.shape[2] // 2
    btc, eth = X[:, :, :k, :], X[:, :, k:, :]
    return {
        "btc_only": btc,
        "eth_only": eth,
        "both": X,
        # Width-matched control: same information as eth_only, same dimension as
        # `both`. Any advantage `both` has over THIS is the second asset rather
        # than the extra columns.
        "eth_doubled": np.concatenate([eth, eth], axis=2),
    }


def raw_features(V):
    """Two readouts per view, deliberately narrow.

    Only 2,654 training rows survive the pairing, so a 4,608-dim `raw_full`
    would overfit the ridge and report a worse number for having MORE
    information, which measures the probe rather than the data. The last-minute
    and last-4-minute readouts are the ones that stayed sane at this sample size.
    """
    n, t, k, c = V.shape
    return {"last_min": V[:, -1].reshape(n, k * c),
            "last_4min": V[:, -4:].reshape(n, 4 * k * c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--horizons", default="3,5,10")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--out", default="results/crossasset_headroom.json")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h]
    res = {"config": vars(args), "derivable_bar": DERIVABLE_BAR,
           "noise_bar": NOISE_BAR, "horizons": {}}

    for h in horizons:
        t0 = time.time()
        X, Y, owner, names, st = ca.load_corpus(
            args.pm_jepa_root, window=args.window, horizon=h)
        tr, te = cj_data.split_by_event(owner)
        print("\nhorizon {:>3}: {:>5} windows over {:>4} pairs "
              "(train {} / test {})".format(
                  h, len(X), len(np.unique(owner)), tr.sum(), te.sum()), flush=True)

        per = {}
        for vname, V in views(X).items():
            for rname, f in raw_features(V).items():
                f = f.astype(np.float64)
                r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
                for t in names:
                    d = per.setdefault(t, {})
                    d["{}|{}".format(vname, rname)] = {
                        "view": vname, "readout": rname, "dim": int(f.shape[1]),
                        "ridge": r["ridge"][t], "mlp": r["mlp"][t]}
                del f

        summary = {}
        for t, d in per.items():
            def best(view):
                vs = [v for v in d.values() if v["view"] == view]
                return max(v["ridge"] for v in vs), max(v["mlp"] for v in vs)
            b_both_r, b_both_m = best("both")
            b_btc_r, _ = best("btc_only")
            b_eth_r, _ = best("eth_only")
            b_dbl_r, _ = best("eth_doubled")
            b_single = max(b_btc_r, b_eth_r)
            all_r = max(v["ridge"] for v in d.values())
            all_m = max(v["mlp"] for v in d.values())
            summary[t] = {
                "by_arm": d,
                "best_ridge_both": b_both_r, "best_mlp_both": b_both_m,
                "best_ridge_btc_only": b_btc_r, "best_ridge_eth_only": b_eth_r,
                "best_ridge_eth_doubled": b_dbl_r,
                "best_ridge_single": b_single,
                "cross_asset_gain": b_both_r - b_single,
                "cross_asset_gain_width_matched": b_both_r - b_dbl_r,
                "regime": regime(all_r, max(all_r, all_m))}

        res["horizons"][str(h)] = {
            "corpus": st, "windows": int(len(X)),
            "pairs": int(len(np.unique(owner))),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "targets": summary, "seconds": round(time.time() - t0, 1)}

        print("  {:<26}{:>10}{:>10}{:>10}{:>12}{:>14}".format(
            "target", "btc", "eth", "both", "vs single", "vs matched"))
        for t in names:
            v = summary[t]
            print("  {:<26}{:>+10.4f}{:>+10.4f}{:>+10.4f}{:>+12.4f}{:>+14.4f}  {}".format(
                t, v["best_ridge_btc_only"], v["best_ridge_eth_only"],
                v["best_ridge_both"], v["cross_asset_gain"],
                v["cross_asset_gain_width_matched"], v["regime"][:4]), flush=True)
        del X, Y

    # ---- verdict --------------------------------------------------------
    best_h, best_t, best_gain = None, None, -9.9
    for h in horizons:
        for t, v in res["horizons"][str(h)]["targets"].items():
            if v["regime"] != "DISCRIMINATIVE":
                continue
            g = v["cross_asset_gain_width_matched"]
            if g > best_gain:
                best_h, best_t, best_gain = h, t, g
    worth_training = bool(best_gain > 0.02)
    res["verdict"] = {
        "best_target": best_t, "best_horizon": best_h,
        "best_width_matched_gain": best_gain,
        "worth_training": worth_training,
        "statement": (
            "PROCEED: '{}' at horizon {} gains {:+.4f} ridge R^2 from the second "
            "ladder over a width-matched single ladder. There is cross-sectional "
            "structure here that one asset does not carry, so a representation "
            "learner has something to find.".format(best_t, best_h, best_gain)
            if worth_training else
            "STOP: the best width-matched cross-asset gain on any discriminative "
            "target is {:+.4f}. The second ladder adds nothing a single one does "
            "not already carry, so the premise behind this experiment is wrong "
            "and training a JEPA on it would repeat the previous five results at "
            "greater cost.".format(best_gain))}

    w = 92
    print("\n" + "=" * w)
    print("E8a CROSS-ASSET HEADROOM  (raw features only, no training)")
    print("=" * w)
    print(res["verdict"]["statement"])
    print("=" * w)

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
