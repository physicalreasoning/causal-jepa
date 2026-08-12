#!/usr/bin/env python3
"""E8: train the JEPA where the cross-sectional latent actually exists.

E8a established the premise this experiment needs, without training anything.
On `realised_corr` at horizon 10, raw features score +0.1867 from the BTC ladder
alone, +0.2520 from the ETH ladder alone, and +0.3693 from both, a width-matched
gain of +0.1064 over a duplicated single ladder. The gain rises monotonically
with horizon, +0.0271 -> +0.0794 -> +0.1064. Targets that should NOT need both
assets show no gain, which is the internal control: `eth_vol_forecast_error` is
+0.1363 from ETH alone against +0.1302 from both.

So realised correlation is a genuine joint property. A strike ladder is a
MARGINAL distribution, pricing where one asset lands; the dependence between two
assets is in neither ladder at any strike. This is the first target in the whole
programme that no single cross-section identifies, which is precisely the
condition that makes the simulator work (a 3-dim latent behind eight
heterogeneous markets) and that a lone BTC ladder lacks.

Every previous negative in this repo was measured where that condition was
absent. This measures it where it is present.

WHY THIS IS A HARDER TEST FOR THE MODEL THAN IT LOOKS. The pairing leaves only
~2,650 training windows, an order of magnitude less than the 13,763 of E7, so
wide raw readouts overfit and a compressed representation has more room to win
than it has had anywhere else in this programme. If a JEPA is ever going to beat
raw features on this corpus, these are the most favourable conditions it will
get. A negative here is therefore a much stronger negative than a negative on
the single-asset corpus.

CONTROLS, unchanged in spirit from E4 and E7:
  * raw features from BTC alone, ETH alone, both, and ETH duplicated to matched
    width, so any cross-asset claim survives the width confound;
  * an UNTRAINED encoder at identical architecture and seeds, at every readout,
    because a model that cannot separate from its own initialisation has learned
    nothing;
  * every arm probed on all six targets, reported per target rather than as a
    mean, since finding 10 is what happens when a mean hides a derivable target.

PRECONDITION, and note the bar is not zero. `fwd_signed_spread_return` is the
martingale control, but raw ETH features already reach +0.0516 on it at h=10,
plausibly short-horizon microstructure drift rather than a bug. Setting the void
bar at zero would fire on the raw controls themselves. It is set at +0.15, well
above what raw achieves and well below anything that would indicate the future
leaking through the features.

Pre-registered gate (docs/RESEARCH_PLAN.md E8):
  best trained arm beats the best raw control on `realised_corr` by more than
  2 pooled SD
      -> the JEPA wins, on the first target in this corpus that has a
         cross-sectional latent. The negative result was about the DATA, and we
         can say exactly which property of it.
  best trained arm shows no matched-width lift above 2 pooled SD
      -> the objective fails even where the latent provably exists, which is a
         far stronger claim than anything in findings 1 to 15.
  lift but still below raw
      -> report the frontier, claim nothing.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from causaljepa import crossasset as ca  # noqa: E402
from causaljepa import data as cj_data, probes  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import pick_device, train_arm  # noqa: E402

import readout_ablation as e4  # noqa: E402
import crossasset_headroom as e8a  # noqa: E402

LEAK_BAR = 0.15
SD_BAR = 2.0
HEAD = "realised_corr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/crossasset_jepa.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Y, owner, names, cstats = ca.load_corpus(
        args.pm_jepa_root, window=args.window, horizon=args.horizon)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}  pairs {}".format(
        device, X.shape, tr.sum(), te.sum(), len(np.unique(owner))), flush=True)

    res = {"config": vars(args), "device": device, "corpus": cstats,
           "leak_bar": LEAK_BAR, "sd_bar": SD_BAR, "headline_target": HEAD,
           "target_names": names, "n_train": int(tr.sum()),
           "n_test": int(te.sum()), "runs": []}

    def probe(f):
        f = f.astype(np.float64)
        return probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")

    def record(model, readout, dim, seed, r, secs):
        rec = {"model": model, "readout": readout, "dim": int(dim), "seed": seed,
               "ridge": r["ridge"], "mlp": r["mlp"],
               "probe_seconds": round(secs, 1)}
        res["runs"].append(rec)
        print("  {:<11}{:<16} dim {:>5}  {} {:+.4f}   sig {:+.4f}  ({:.0f}s)".format(
            model, readout, dim, HEAD[:9], r["ridge"][HEAD],
            r["ridge"]["fwd_signed_spread_return"], secs), flush=True)
        return rec

    print("\n--- raw controls, four views (seed-independent) ---", flush=True)
    for vname, V in e8a.views(X).items():
        for rname, f in e8a.raw_features(V).items():
            t0 = time.time()
            record("raw", "{}|{}".format(vname, rname), f.shape[1], None,
                   probe(f), time.time() - t0)
            del f

    for seed in range(args.seeds):
        t0 = time.time()
        print("\n--- seed {} ---".format(seed), flush=True)
        model, _h, secs = train_arm(
            X, tr, strategy=args.strategy, causal_context=False,
            causal_target=False, device=device, steps=args.steps,
            batch=args.batch, lr=1e-3, lam=0.0, d_model=args.d_model,
            n_layers=args.layers, n_held_out=6, seed=seed,
            log_every=args.steps, verbose=False)
        torch.manual_seed(seed)
        np.random.seed(seed)
        untrained = CausalJEPA(X.shape[1], X.shape[2], X.shape[3],
                               d_model=args.d_model, n_layers=args.layers,
                               causal_context=False, causal_target=False).to(device)
        for tag, m in (("trained", model), ("untrained", untrained)):
            for rname, f in e4.extract(m, X, device).items():
                p0 = time.time()
                record(tag, rname, f.shape[1], seed, probe(f), time.time() - p0)
        del model, untrained
        print("  seed {} done in {:.0f}s (train {:.0f}s)".format(
            seed, time.time() - t0, secs), flush=True)
        pathlib.Path(args.out).parent.mkdir(exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    leaks = [r for r in res["runs"]
             if r["ridge"]["fwd_signed_spread_return"] > LEAK_BAR]
    res["leak_check"] = {
        "bar": LEAK_BAR, "n_violations": len(leaks),
        "worst": max(r["ridge"]["fwd_signed_spread_return"] for r in res["runs"])}
    assert not leaks, "FUTURE LEAKED: {} arm(s) above {}".format(len(leaks), LEAK_BAR)

    agg = {}
    for r in res["runs"]:
        agg.setdefault((r["model"], r["readout"]), []).append(r)
    summary = {}
    for (tag, ro), rs in agg.items():
        e = {"model": tag, "readout": ro, "dim": rs[0]["dim"],
             "seeds": len(rs), "per_target": {}}
        for t in names:
            a = np.array([x["ridge"][t] for x in rs])
            m = np.array([x["mlp"][t] for x in rs])
            e["per_target"][t] = {
                "ridge_mean": float(a.mean()), "ridge_std": float(a.std()),
                "mlp_mean": float(m.mean()), "mlp_std": float(m.std())}
        summary["{}|{}".format(tag, ro)] = e
    res["summary"] = summary

    lift = {}
    for v in summary.values():
        if v["model"] != "trained":
            continue
        u = summary.get("untrained|" + v["readout"])
        if not u:
            continue
        per = {}
        for t in names:
            a, b = v["per_target"][t], u["per_target"][t]
            sd = (a["ridge_std"] ** 2 + b["ridge_std"] ** 2) ** 0.5
            per[t] = {"trained": a["ridge_mean"], "untrained": b["ridge_mean"],
                      "lift": a["ridge_mean"] - b["ridge_mean"],
                      "in_sd": ((a["ridge_mean"] - b["ridge_mean"]) / sd
                                if sd > 1e-12 else float("nan"))}
        lift[v["readout"]] = {"dim": v["dim"], "per_target": per}
    res["training_lift_at_matched_readout"] = lift

    def best(model):
        vs = [v for v in summary.values() if v["model"] == model]
        return max(vs, key=lambda v: v["per_target"][HEAD]["ridge_mean"]) if vs else None

    bt, br = best("trained"), best("raw")
    btm, brm = bt["per_target"][HEAD], br["per_target"][HEAD]
    sd = (btm["ridge_std"] ** 2 + brm["ridge_std"] ** 2) ** 0.5
    margin = (btm["ridge_mean"] - brm["ridge_mean"]) / sd if sd > 1e-12 else float("nan")
    best_lift = max((lift[ro]["per_target"][HEAD]["in_sd"] for ro in lift),
                    default=float("nan"))

    decidable = args.seeds >= 2
    beats_raw = bool(decidable and np.isfinite(margin) and margin > SD_BAR)
    any_lift = bool(decidable and np.isfinite(best_lift) and best_lift > SD_BAR)
    res["gate"] = {
        "decidable": decidable, "headline_target": HEAD,
        "best_trained": {"readout": bt["readout"], "dim": bt["dim"],
                         "ridge_mean": btm["ridge_mean"], "ridge_std": btm["ridge_std"]},
        "best_raw": {"readout": br["readout"], "dim": br["dim"],
                     "ridge_mean": brm["ridge_mean"]},
        "margin_in_pooled_sd": margin,
        "best_matched_width_lift_in_sd": best_lift,
        "beats_raw": beats_raw, "any_matched_width_lift": any_lift,
        "verdict": (
            "UNDECIDABLE: {} seed(s), pooled SD needs at least 2.".format(args.seeds)
            if not decidable else
            "THE JEPA WINS: it beats the best raw control on {} by {:.2f} pooled "
            "SD, on the first target in this corpus with a cross-sectional "
            "latent.".format(HEAD, margin) if beats_raw else
            "NEGATIVE EVEN HERE: no matched-width training lift above {} pooled "
            "SD on {}, where the latent provably exists. This is a stronger "
            "negative than findings 1-15.".format(SD_BAR, HEAD) if not any_lift else
            "PARTIAL: training lifts the representation ({:.2f} sd) but raw "
            "features still win by {:.2f} sd. Report the frontier.".format(
                best_lift, -margin))}

    w = 100
    print("\n" + "=" * w)
    print("E8 CROSS-ASSET JEPA  ({} seeds, {} steps, horizon {}, {} windows)".format(
        args.seeds, args.steps, args.horizon, len(X)))
    print("=" * w)
    print("{:<11}{:<22}{:>6}{:>22}".format("model", "readout", "dim", HEAD))
    print("-" * w)
    for v in sorted(summary.values(), key=lambda v: -v["per_target"][HEAD]["ridge_mean"]):
        p = v["per_target"][HEAD]
        print("{:<11}{:<22}{:>6}{:>+16.4f}+-{:.4f}".format(
            v["model"], v["readout"], v["dim"], p["ridge_mean"], p["ridge_std"]))
    print("=" * w)
    print("\nMATCHED-WIDTH TRAINING LIFT ON {}".format(HEAD))
    print("{:<16}{:>6}{:>12}{:>12}{:>12}{:>10}".format(
        "readout", "dim", "trained", "untrained", "lift", "in sd"))
    for ro, d in sorted(lift.items(), key=lambda kv: -kv[1]["per_target"][HEAD]["lift"]):
        p = d["per_target"][HEAD]
        print("{:<16}{:>6}{:>+12.4f}{:>+12.4f}{:>+12.4f}{:>10.2f}".format(
            ro, d["dim"], p["trained"], p["untrained"], p["lift"], p["in_sd"]))
    print("=" * w)
    print("LEAK CHECK: worst {:+.4f} (bar {:.2f})".format(
        res["leak_check"]["worst"], LEAK_BAR))
    print("GATE: {}".format(res["gate"]["verdict"]))
    print("      best trained {:<20} dim {:>5} {:+.4f}".format(
        bt["readout"], bt["dim"], btm["ridge_mean"]))
    print("      best raw     {:<20} dim {:>5} {:+.4f}".format(
        br["readout"], br["dim"], brm["ridge_mean"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
