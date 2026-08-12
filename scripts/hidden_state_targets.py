#!/usr/bin/env python3
"""E7: was the negative result about the model, or about the targets?

E3c found nearly the whole apparent gap between arms sitting in
`implied_width`, a target `data/dataset.py` itself documents as derivable from
the input. On `log_return_to_settle` every arm collapsed into a band too narrow
to rank anything. So the programme's negative result has two readings that the
evidence so far cannot separate:

  (a) a JEPA learns nothing useful on this data, or
  (b) the target set had no recoverable hidden state, so the comparison was
      never able to discriminate between a good representation and a bad one.

`causaljepa/targets.py` builds four targets with genuine hidden state, headed by
`vol_forecast_error`, which divides the ladder's own volatility forecast out of
future realised volatility and therefore cannot be read off the input the way
`implied_width` could. This runs the E4 arm set against them.

WHAT MAKES THIS A FAIR TEST, and not a rerun of the mistakes this programme
already made:

  * The arms are IMPORTED from `readout_ablation.py` rather than reimplemented,
    so the raw controls, the readouts and the extraction are bit-identical to
    E4's. A difference in the table is a difference in the targets, full stop.
  * Both target blocks are probed on THE SAME WINDOWS. The horizon requirement
    drops windows near the end of each event, so the original four are
    recomputed on the surviving subset and reported beside the new four. Without
    that, a change could be the subset rather than the targets.
  * The two blocks are probed SEPARATELY, so each picks its own ridge lambda.
    Probing eight mixed columns at once would let the new targets shift the
    lambda chosen for the old ones and quietly break comparability with every
    published number in this repo.
  * Every trained arm is paired with an UNTRAINED encoder at the same readout
    width. E4 exists because that control turned a false positive into the
    correct verdict; it is not optional here either.

HARD PRECONDITION. `fwd_signed_return` is a ten-minute near-martingale and no
arm should predict it. If any arm exceeds `LEAK_BAR` ridge R^2 on it, the
future has leaked into the features and the whole run is void rather than
interesting. That check is an assertion, not a log line.

Pre-registered gate (docs/RESEARCH_PLAN.md E7):
  best trained arm beats the best raw control on `vol_forecast_error` by more
  than 2 pooled SD
      -> the negative result was a property of the TARGET SET. The programme
         reopens on this target.
  no new target shows a matched-width training lift above 2 pooled SD
      -> the negative result survives contact with targets that do contain
         hidden state, and it generalises rather than being target-specific.
  anything between
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

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes, targets as cj_targets  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import pick_device, train_arm  # noqa: E402

# The E4 arms, imported rather than copied so the two experiments cannot drift.
import readout_ablation as e4  # noqa: E402

LEAK_BAR = 0.05          # ridge R^2 on fwd_signed_return above this voids the run
SD_BAR = 2.0             # pooled-SD threshold for every claim below


def probe_blocks(f, tr, te, Yo, Yn, on, nn_):
    """Probe both target blocks separately. -> {block: run_probes output}."""
    f = f.astype(np.float64)
    return {
        "orig": probes.run_probes(f[tr], Yo[tr], f[te], Yo[te], on, device="cpu"),
        "new": probes.run_probes(f[tr], Yn[tr], f[te], Yn[te], nn_, device="cpu"),
    }


def flatten(rec, r):
    """Fold a two-block probe result into one flat record."""
    for blk in ("orig", "new"):
        rec["ridge_{}".format(blk)] = r[blk]["ridge"]
        rec["mlp_{}".format(blk)] = r[blk]["mlp"]
        rec["ridge_mean_{}".format(blk)] = r[blk]["ridge_mean"]
        rec["mlp_mean_{}".format(blk)] = r[blk]["mlp_mean"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"),
                    help="checkout of the pm-jepa corpus repo; "
                         "defaults to $PM_JEPA_ROOT then ../pm-jepa")
    ap.add_argument("--horizon", type=int, default=10,
                    help="forward steps used to build the new targets")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600,
                    help="600 on E1's evidence: the probes peak at the first "
                         "checkpoint and degrade from there, so a longer run "
                         "measures a worse model rather than a better-trained one")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/hidden_state_targets.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Yo, Yn, owner, on, nn_, cstats = cj_targets.load_corpus(
        args.pm_jepa_root, horizon=args.horizon)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}  events {}".format(
        device, X.shape, tr.sum(), te.sum(), len(np.unique(owner))), flush=True)
    print("horizon {} steps; {} of 3,336 events contribute".format(
        args.horizon, cstats["events_contributing"]), flush=True)

    res = {"config": vars(args), "device": device, "corpus": cstats,
           "leak_bar": LEAK_BAR, "sd_bar": SD_BAR,
           "orig_target_names": on, "new_target_names": nn_,
           "target_moments": {"orig": cj_targets.describe(Yo, on),
                              "new": cj_targets.describe(Yn, nn_)},
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "runs": []}

    def log(tag, name, dim, rec, secs):
        print("  {:<9} {:<15} dim {:>5}  ORIG {:+.4f}  NEW {:+.4f}   "
              "vfe {:+.4f}  sig {:+.4f}  ({:.0f}s)".format(
                  tag, name, dim, rec["ridge_mean_orig"], rec["ridge_mean_new"],
                  rec["ridge_new"]["vol_forecast_error"],
                  rec["ridge_new"]["fwd_signed_return"], secs), flush=True)

    print("\n--- raw controls (dimension-matched, seed-independent) ---", flush=True)
    for name, f in e4.raw_controls(X).items():
        t0 = time.time()
        r = probe_blocks(f, tr, te, Yo, Yn, on, nn_)
        rec = flatten({"seed": None, "model": "raw", "readout": name,
                       "dim": int(f.shape[1])}, r)
        rec["probe_seconds"] = round(time.time() - t0, 1)
        res["runs"].append(rec)
        log("raw", name, rec["dim"], rec, time.time() - t0)
        del f

    for seed in range(args.seeds):
        t0 = time.time()
        print("\n--- seed {} ---".format(seed), flush=True)
        model, _hist, secs = train_arm(
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
            feats = e4.extract(m, X, device)
            for name, f in feats.items():
                p0 = time.time()
                r = probe_blocks(f, tr, te, Yo, Yn, on, nn_)
                rec = flatten({"seed": seed, "model": tag, "readout": name,
                               "dim": int(f.shape[1])}, r)
                rec["probe_seconds"] = round(time.time() - p0, 1)
                res["runs"].append(rec)
                log(tag, name, rec["dim"], rec, time.time() - p0)
            del feats
        del model, untrained
        print("  seed {} done in {:.0f}s (train {:.0f}s)".format(
            seed, time.time() - t0, secs), flush=True)
        pathlib.Path(args.out).parent.mkdir(exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- hard precondition: nothing may predict the martingale ----------
    leaks = [r for r in res["runs"]
             if r["ridge_new"]["fwd_signed_return"] > LEAK_BAR]
    res["leak_check"] = {
        "bar": LEAK_BAR, "n_violations": len(leaks),
        "worst": max((r["ridge_new"]["fwd_signed_return"] for r in res["runs"]),
                     default=float("nan")),
        "violations": [{"model": r["model"], "readout": r["readout"],
                        "seed": r["seed"],
                        "r2": r["ridge_new"]["fwd_signed_return"]} for r in leaks]}
    assert not leaks, (
        "FUTURE LEAKED: {} arm(s) predict fwd_signed_return above {}. A "
        "ten-minute forward return is a near-martingale; anything that "
        "predicts it has seen the future through the features. The run is "
        "void. Worst: {}".format(len(leaks), LEAK_BAR, res["leak_check"]["worst"]))

    # ---- aggregate over seeds -------------------------------------------
    agg = {}
    for rec in res["runs"]:
        agg.setdefault((rec["model"], rec["readout"]), []).append(rec)
    summary = {}
    for (tag, name), rs in agg.items():
        entry = {"model": tag, "readout": name, "dim": rs[0]["dim"],
                 "seeds": len(rs), "per_target": {}}
        for blk, bn in (("orig", on), ("new", nn_)):
            rm = np.array([r["ridge_mean_{}".format(blk)] for r in rs])
            entry["ridge_mean_{}".format(blk)] = float(rm.mean())
            entry["ridge_std_{}".format(blk)] = float(rm.std())
            for t in bn:
                a = np.array([r["ridge_{}".format(blk)][t] for r in rs])
                m = np.array([r["mlp_{}".format(blk)][t] for r in rs])
                entry["per_target"][t] = {
                    "ridge_mean": float(a.mean()), "ridge_std": float(a.std()),
                    "mlp_mean": float(m.mean()), "mlp_std": float(m.std())}
        summary["{}|{}".format(tag, name)] = entry
    res["summary"] = summary

    # ---- matched-width training lift, per target ------------------------
    lift = {}
    for v in summary.values():
        if v["model"] != "trained":
            continue
        u = summary.get("untrained|" + v["readout"])
        if not u:
            continue
        per = {}
        for t in list(on) + list(nn_):
            a, b = v["per_target"][t], u["per_target"][t]
            sd = (a["ridge_std"] ** 2 + b["ridge_std"] ** 2) ** 0.5
            per[t] = {"trained": a["ridge_mean"], "untrained": b["ridge_mean"],
                      "lift": a["ridge_mean"] - b["ridge_mean"],
                      "in_sd": ((a["ridge_mean"] - b["ridge_mean"]) / sd
                                if sd > 1e-12 else float("nan"))}
        lift[v["readout"]] = {"dim": v["dim"], "per_target": per}
    res["training_lift_at_matched_readout"] = lift

    # ---- the gate --------------------------------------------------------
    def best(model, target):
        vs = [v for v in summary.values() if v["model"] == model]
        return max(vs, key=lambda v: v["per_target"][target]["ridge_mean"]) if vs else None

    HEAD = "vol_forecast_error"
    bt, br = best("trained", HEAD), best("raw", HEAD)
    bt_m = bt["per_target"][HEAD]
    br_m = br["per_target"][HEAD]
    sd = (bt_m["ridge_std"] ** 2 + br_m["ridge_std"] ** 2) ** 0.5
    margin_sd = ((bt_m["ridge_mean"] - br_m["ridge_mean"]) / sd
                 if sd > 1e-12 else float("nan"))

    best_new_lift = max(
        ((t, ro, d["per_target"][t]["in_sd"], d["per_target"][t]["lift"])
         for ro, d in lift.items() for t in nn_),
        key=lambda x: (x[2] if np.isfinite(x[2]) else -np.inf), default=None)

    # A gate stated in pooled SD is UNDECIDABLE below two seeds, because the
    # pooled SD is then exactly zero and every margin evaluates to nan. Both
    # branches below would read as "not met" and the run would print a verdict
    # it had no evidence for. E1 and E2 in this programme were each decided by
    # a gate whose wording admitted a degenerate case; this one refuses instead.
    decidable = args.seeds >= 2
    beats_raw = bool(decidable and bt_m["ridge_mean"] > br_m["ridge_mean"]
                     and np.isfinite(margin_sd) and margin_sd > SD_BAR)
    any_lift = bool(decidable and best_new_lift and np.isfinite(best_new_lift[2])
                    and best_new_lift[2] > SD_BAR)

    res["gate"] = {
        "decidable": decidable,
        "criterion": ("best trained arm beats best raw control on "
                      "vol_forecast_error by more than {} pooled SD -> the "
                      "negative result was target-specific; no new target shows "
                      "a matched-width training lift above {} pooled SD -> it "
                      "generalises".format(SD_BAR, SD_BAR)),
        "headline_target": HEAD,
        "best_trained": {"readout": bt["readout"], "dim": bt["dim"],
                         "ridge_mean": bt_m["ridge_mean"],
                         "ridge_std": bt_m["ridge_std"]},
        "best_raw": {"readout": br["readout"], "dim": br["dim"],
                     "ridge_mean": br_m["ridge_mean"]},
        "margin_in_pooled_sd": margin_sd,
        "beats_raw": beats_raw,
        "best_matched_width_lift_on_new_targets": (
            {"target": best_new_lift[0], "readout": best_new_lift[1],
             "in_sd": best_new_lift[2], "lift": best_new_lift[3]}
            if best_new_lift else None),
        "any_new_target_lift": any_lift,
        "verdict": (
            "UNDECIDABLE: {} seed(s). Every margin here is stated in pooled SD, "
            "which needs at least 2 seeds to exist. Numbers below are reported, "
            "the gate is not evaluated.".format(args.seeds)
            if not decidable else
            "TARGET-SPECIFIC: a trained arm beats raw features on {} by {:.2f} "
            "pooled SD. The negative result was a property of the old target "
            "set, and the programme reopens here.".format(HEAD, margin_sd)
            if beats_raw else
            "NEGATIVE RESULT GENERALISES: no new target shows a matched-width "
            "training lift above {} pooled SD, so it survives contact with "
            "targets that do contain hidden state.".format(SD_BAR)
            if not any_lift else
            "PARTIAL: raw features still win the headline target, but at least "
            "one new target shows a matched-width training lift. Report the "
            "frontier and claim nothing."),
    }

    # ---- print -----------------------------------------------------------
    w = 108
    print("\n" + "=" * w)
    print("E7 HIDDEN-STATE TARGETS  ({} seeds, {} steps, horizon {}, {} windows)".format(
        args.seeds, args.steps, args.horizon, len(X)))
    print("=" * w)
    hdr = "{:<11}{:<16}{:>6}" + "{:>19}" * 4
    print(hdr.format("model", "readout", "dim", *nn_))
    print("-" * w)
    for v in sorted(summary.values(),
                    key=lambda v: -v["per_target"][HEAD]["ridge_mean"]):
        cells = ["{:+.4f}+-{:.4f}".format(v["per_target"][t]["ridge_mean"],
                                          v["per_target"][t]["ridge_std"])
                 for t in nn_]
        print(("{:<11}{:<16}{:>6}" + "{:>19}" * 4).format(
            v["model"], v["readout"], v["dim"], *cells))
    print("=" * w)

    print("\nMATCHED-WIDTH TRAINING LIFT ON THE NEW TARGETS (trained - untrained)")
    print("-" * w)
    print("{:<16}{:>6}".format("readout", "dim")
          + "".join("{:>23}".format(t) for t in nn_))
    for ro, d in sorted(lift.items(),
                        key=lambda kv: -kv[1]["per_target"][HEAD]["lift"]):
        cells = ["{:+.4f} ({:>+6.2f}sd)".format(d["per_target"][t]["lift"],
                                                d["per_target"][t]["in_sd"])
                 for t in nn_]
        print("{:<16}{:>6}".format(ro, d["dim"]) + "".join("{:>23}".format(c) for c in cells))
    print("=" * w)

    print("\nORIGINAL TARGETS, same windows (the within-run control)")
    print("-" * w)
    print("{:<11}{:<16}{:>6}".format("model", "readout", "dim")
          + "".join("{:>21}".format(t[:20]) for t in on))
    for v in sorted(summary.values(), key=lambda v: -v["ridge_mean_orig"]):
        cells = ["{:+.4f}".format(v["per_target"][t]["ridge_mean"]) for t in on]
        print("{:<11}{:<16}{:>6}".format(v["model"], v["readout"], v["dim"])
              + "".join("{:>21}".format(c) for c in cells))
    print("=" * w)

    g = res["gate"]
    print("LEAK CHECK: worst fwd_signed_return ridge R^2 {:+.4f} (bar {:+.2f}) -> clean".format(
        res["leak_check"]["worst"], LEAK_BAR))
    print("GATE: {}".format(g["verdict"]))
    print("      best trained on {}: {:<15} dim {:>5}  {:+.4f}".format(
        HEAD, g["best_trained"]["readout"], g["best_trained"]["dim"],
        g["best_trained"]["ridge_mean"]))
    print("      best raw     on {}: {:<15} dim {:>5}  {:+.4f}".format(
        HEAD, g["best_raw"]["readout"], g["best_raw"]["dim"],
        g["best_raw"]["ridge_mean"]))
    print("      margin: {:+.2f} pooled SD".format(margin_sd))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
