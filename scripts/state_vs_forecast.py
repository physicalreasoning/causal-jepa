#!/usr/bin/env python3
"""E11: does a current-state target behave differently from a forecast target?

Finding 20's surviving hypothesis: the simulator wins because it INVERTS current
hidden state through a known noisy map, while every Kalshi target that mattered
was a FORECAST in a near-efficient market. It predicts that a current-but-hidden
Kalshi target should show a matched-width training lift that SURVIVES
residualisation, where every forward target's lift died (findings 14 and 17).

This tests that, with current and forward targets in the SAME run, on the same
windows, the same encoders and the same probes, so the contrast is not a
comparison across experiments.

WHY RESIDUALISATION IS THE MEASUREMENT AND NOT AN AFTERTHOUGHT. The centred
smoother at t spans [t-half, t+half], and the past half of that lies INSIDE the
input window, so these targets are partly derivable by construction. The headroom
check says so plainly: `width_denoise` reaches +0.6746 from raw features alone
and `spot_denoise` +0.2964. That does not invalidate the test, because the
quantity compared here is never the raw R^2. Every target is residualised against
raw features first, and what is compared is the lift on WHAT RAW CANNOT ALREADY
EXPLAIN. A partly derivable target is fine; an unresidualised comparison is not.

The three conditions that decide a surviving lift are the ones E7c and E8c needed
two failures to find: the SAME arm must clear zero AND beat the best untrained
arm, and the lift itself must exceed 2 pooled SD.

Pre-registered gate:
  current-state targets show surviving residual lifts and forward targets do not
      -> finding 20's hypothesis holds. The objective was a filtering tool
         pointed at a forecasting problem, and that explains findings 1 to 20
         with one mechanism.
  neither kind shows a surviving lift
      -> the hypothesis is wrong too, and the sim-to-real gap survives a seventh
         attempt.
  both kinds show surviving lifts
      -> the current/forward split is not the axis; report and stop claiming it.
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

from causaljepa import data as cj_data, probes, statetargets as stt  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import pick_device, train_arm  # noqa: E402

import readout_ablation as e4  # noqa: E402
from residual_probe import residualise  # noqa: E402

SD_BAR = 2.0
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--half", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--control", default="raw_last_4min")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/state_vs_forecast.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Y, owner, names, cstats = stt.load_corpus(args.pm_jepa_root, half=args.half)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)

    raw = e4.raw_controls(X)
    ctl = raw[args.control].astype(np.float64)

    # Residualise every target against the same raw control, fit on train only.
    resid_tr, resid_te, ctl_r2 = {}, {}, {}
    for i, n in enumerate(names):
        y = Y[:, i].astype(np.float64)
        a, b, r2 = residualise(ctl[tr], ctl[te], y[tr], y[te])
        resid_tr[n], resid_te[n], ctl_r2[n] = a, b, r2
    print("\nresidualised against {} (dim {}); raw explains:".format(
        args.control, ctl.shape[1]), flush=True)
    for n in names:
        print("  {:<20} {:<8} {:+.4f}".format(n, stt.KIND[n], ctl_r2[n]), flush=True)

    res = {"config": vars(args), "device": device, "corpus": cstats,
           "kind": stt.KIND, "control_r2_on_target": ctl_r2,
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "runs": []}

    def probe_all(f):
        f = f.astype(np.float64)
        out = {}
        for n in names:
            r = probes.run_probes(f[tr], resid_tr[n], f[te], resid_te[n],
                                  [n + "_residual"], device="cpu")
            out[n] = r["ridge"][n + "_residual"]
        return out

    print("\n--- raw controls on the residuals (sanity: the control must be ~0) ---",
          flush=True)
    for cn, f in raw.items():
        rr = probe_all(f)
        res["runs"].append({"model": "raw", "readout": cn, "seed": None,
                            "dim": int(f.shape[1]), "ridge": rr})
        print("  {:<16} dim {:>5}  ".format(cn, f.shape[1])
              + "  ".join("{} {:+.4f}".format(n[:9], rr[n]) for n in names), flush=True)
    self_r2 = next(r["ridge"] for r in res["runs"] if r["readout"] == args.control)
    assert all(v < 0.02 for v in self_r2.values()), (
        "RESIDUALISATION FAILED: the control still predicts its own residual: "
        "{}".format({k: round(v, 4) for k, v in self_r2.items()}))

    for seed in range(args.seeds):
        t0 = time.time()
        print("\n--- seed {} ---".format(seed), flush=True)
        model, _h, secs = train_arm(
            X, tr, strategy="temporal", causal_context=False, causal_target=False,
            device=device, steps=args.steps, batch=args.batch, lr=1e-3, lam=0.0,
            d_model=args.d_model, n_layers=args.layers, n_held_out=6, seed=seed,
            log_every=args.steps, verbose=False)
        torch.manual_seed(seed)
        np.random.seed(seed)
        untrained = CausalJEPA(X.shape[1], X.shape[2], X.shape[3],
                               d_model=args.d_model, n_layers=args.layers,
                               causal_context=False, causal_target=False).to(device)
        for tag, m in (("trained", model), ("untrained", untrained)):
            for rn, f in e4.extract(m, X, device).items():
                rr = probe_all(f)
                res["runs"].append({"model": tag, "readout": rn, "seed": seed,
                                    "dim": int(f.shape[1]), "ridge": rr})
                print("  {:<9} {:<15} dim {:>5}  ".format(tag, rn, f.shape[1])
                      + "  ".join("{} {:+.4f}".format(n[:9], rr[n]) for n in names),
                      flush=True)
        del model, untrained
        print("  seed {} in {:.0f}s (train {:.0f}s)".format(
            seed, time.time() - t0, secs), flush=True)
        pathlib.Path(args.out).parent.mkdir(exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- aggregate ----
    agg = {}
    for r in res["runs"]:
        if r["seed"] is None:
            continue
        agg.setdefault((r["model"], r["readout"]), []).append(r)
    summary = {}
    for (tag, rn), rs in agg.items():
        e = {"model": tag, "readout": rn, "dim": rs[0]["dim"], "seeds": len(rs),
             "per_target": {}}
        for n in names:
            a = np.array([x["ridge"][n] for x in rs])
            e["per_target"][n] = {"mean": float(a.mean()), "std": float(a.std())}
        summary["{}|{}".format(tag, rn)] = e
    res["summary"] = summary

    verdicts = {}
    for n in names:
        bt = max((v for v in summary.values() if v["model"] == "trained"),
                 key=lambda v: v["per_target"][n]["mean"])
        bu = max((v for v in summary.values() if v["model"] == "untrained"),
                 key=lambda v: v["per_target"][n]["mean"])
        u_same = summary["untrained|" + bt["readout"]]["per_target"][n]
        t_m, t_s = bt["per_target"][n]["mean"], bt["per_target"][n]["std"]
        se = t_s / (args.seeds ** 0.5) if t_s > 0 else 0.0
        t_vs0 = t_m / se if se > 1e-12 else float("nan")
        sd = (t_s ** 2 + u_same["std"] ** 2) ** 0.5
        lift = t_m - u_same["mean"]
        in_sd = lift / sd if sd > 1e-12 else float("nan")
        survives = bool(
            np.isfinite(in_sd) and in_sd > SD_BAR
            and np.isfinite(t_vs0) and t_vs0 > T_CRIT.get(args.seeds, 2.776)
            and t_m > bu["per_target"][n]["mean"])
        verdicts[n] = {
            "kind": stt.KIND[n], "readout": bt["readout"], "dim": bt["dim"],
            "trained": t_m, "trained_std": t_s, "untrained_same_readout": u_same["mean"],
            "best_untrained_any": bu["per_target"][n]["mean"],
            "lift": lift, "in_sd": in_sd, "t_vs_zero": t_vs0,
            "survives_residualisation": survives}
    res["per_target_verdict"] = verdicts

    cur = [n for n in names if stt.KIND[n] == "current"]
    fut = [n for n in names if stt.KIND[n] == "future"]
    n_cur = sum(verdicts[n]["survives_residualisation"] for n in cur)
    n_fut = sum(verdicts[n]["survives_residualisation"] for n in fut)
    res["gate"] = {
        "current_targets": cur, "future_targets": fut,
        "current_surviving": n_cur, "future_surviving": n_fut,
        "verdict": (
            "HYPOTHESIS HOLDS: {}/{} current-state targets show a residual lift "
            "that survives, against {}/{} forward targets. The objective is a "
            "filtering tool and this corpus asked it to forecast.".format(
                n_cur, len(cur), n_fut, len(fut))
            if n_cur > 0 and n_fut == 0 else
            "HYPOTHESIS REFUTED: no target of either kind shows a surviving "
            "residual lift. The current/forward split is not the axis either, and "
            "the sim-to-real gap survives a seventh attempt."
            if n_cur == 0 and n_fut == 0 else
            "SPLIT IS NOT THE AXIS: {}/{} current and {}/{} forward targets "
            "survive. Whatever separates them, it is not current versus "
            "future.".format(n_cur, len(cur), n_fut, len(fut)))}

    w = 108
    print("\n" + "=" * w)
    print("E11 STATE VERSUS FORECAST  ({} seeds, {} steps, half {})".format(
        args.seeds, args.steps, args.half))
    print("=" * w)
    print("{:<20}{:<9}{:<15}{:>10}{:>12}{:>10}{:>9}{:>11}".format(
        "target", "kind", "readout", "trained", "untrained", "lift", "in sd", "survives"))
    print("-" * w)
    for n in names:
        v = verdicts[n]
        print("{:<20}{:<9}{:<15}{:>+10.4f}{:>+12.4f}{:>+10.4f}{:>9.2f}{:>11}".format(
            n, v["kind"], v["readout"], v["trained"], v["untrained_same_readout"],
            v["lift"], v["in_sd"], "YES" if v["survives_residualisation"] else "no"))
    print("=" * w)
    print("GATE: {}".format(res["gate"]["verdict"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
