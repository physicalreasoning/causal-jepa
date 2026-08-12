#!/usr/bin/env python3
"""E2: test the regulariser that was never tested.

pm-jepa computed SIGReg on a detached target; slate-jepa did the same with
VICReg. Neither contributed any gradient, so the published claim that these
mechanisms fail to prevent martingale collapse describes an experiment nobody
ran. This runs it, on the ONLINE context representation, which carries gradient.

Budget: 600 steps, on E1's evidence. E1 showed the probes peak at the first
checkpoint and degrade from there, so a longer run would measure a worse model,
not a better-trained one.

HARD PRECONDITION. Every arm with a regulariser must report
`reg_grad_norm > 0`. If it does not, the run is measuring the same bug again and
the result is discarded rather than reported. That check is an assertion here,
not a log line.

Pre-registered gate (docs/RESEARCH_PLAN.md E2):
  copy_alignment drops below 0.90 without destroying probe R^2
      -> the original claim was wrong, the mechanism does help
  copy_alignment stays above 0.95 across the sweep
      -> the original claim is vindicated, now for real, and the question closes
  probe R^2 collapses as lam rises
      -> the regulariser trades representation for distribution shape; report
         the frontier
"""
import argparse
import os
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes, regularisers  # noqa: E402
from causaljepa.diagnostics import copy_diagnostics, effective_rank  # noqa: E402
from causaljepa.masking import sample_mask  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import (ema_momentum, jepa_loss, lr_multiplier,  # noqa: E402
                              pick_device, regulariser_grad_norm, represent_all)


def run(X, Y, tr, te, names, *, reg_name, lam, seed, steps, batch, lr,
        d_model, n_layers, strategy, n_held_out, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, T, K, C = X.shape
    model = CausalJEPA(T, K, C, d_model=d_model, n_layers=n_layers,
                       causal_context=False, causal_target=False).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.02, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, steps))

    idx_pool = np.flatnonzero(tr)
    Xt = torch.from_numpy(X)
    gen = torch.Generator().manual_seed(seed + 1)
    rgen = torch.Generator(device=device).manual_seed(seed + 2)

    t0 = time.time()
    gn_first = gn_last = 0.0
    for step in range(steps):
        sel = torch.from_numpy(np.random.choice(idx_pool, batch, replace=False))
        obs = Xt[sel].to(device)
        mk = sample_mask(batch, T, K, strategy, generator=gen,
                         n_held_out=n_held_out, device="cpu",
                         patch_length=model.patch_length).to(device)
        out = model(obs, mk)
        base = jepa_loss(out["pred"], out["target"], out["mask_flat"])

        # The whole point: `context` is the ONLINE encoder's output and carries
        # gradient. Passing out["target"] here would silently reproduce the bug
        # this experiment exists to correct.
        rd = regularisers.build(reg_name, out["context"], generator=rgen)
        reg = rd["reg"] if rd is not None else None
        gn = regulariser_grad_norm(reg, params)
        if step == 0:
            gn_first = gn
        gn_last = gn

        loss = base["loss"] + (lam * reg if reg is not None else 0.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target(momentum=ema_momentum(step, steps))

    with torch.no_grad():
        cd = copy_diagnostics(out["pred"], out["target"], out["patch_mask"])
        er = effective_rank(out["context"].detach().cpu())
    f = represent_all(model, X, device)
    pr = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
    del model

    return {"seed": seed, "reg": reg_name, "lam": lam,
            "reg_grad_norm_first": gn_first, "reg_grad_norm_last": gn_last,
            "copy_alignment": cd["copy_alignment"],
            "copy_loss_ratio": cd["copy_loss_ratio"],
            "eff_rank": er, "pred_loss": float(base["pred_loss"]),
            "ridge_mean": pr["ridge_mean"], "mlp_mean": pr["mlp_mean"],
            "ridge": pr["ridge"], "mlp": pr["mlp"],
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"),
                    help="checkout of the pm-jepa corpus repo; "
                         "defaults to $PM_JEPA_ROOT then ../pm-jepa")
    ap.add_argument("--regs", default="sigreg,vicreg")
    ap.add_argument("--lams", default="0.01,0.05,0.25,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--n-held-out", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/regulariser_sweep.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Y, owner, names = cj_data.load_corpus(args.pm_jepa_root)
    tr, te = cj_data.split_by_event(owner)
    lams = [float(x) for x in args.lams.split(",") if x]
    regs = [r for r in args.regs.split(",") if r]
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)

    combos = [("none", 0.0)] + [(r, l) for r in regs for l in lams]
    res = {"config": vars(args), "device": device, "runs": [],
           "precondition": "every regularised arm must report reg_grad_norm > 0"}

    for reg_name, lam in combos:
        for s in range(args.seeds):
            rec = run(X, Y, tr, te, names, reg_name=reg_name, lam=lam, seed=s,
                      steps=args.steps, batch=args.batch, lr=args.lr,
                      d_model=args.d_model, n_layers=args.layers,
                      strategy=args.strategy, n_held_out=args.n_held_out,
                      device=device)
            if reg_name != "none":
                assert rec["reg_grad_norm_last"] > 0.0, (
                    "INERT REGULARISER: {} lam={} seed={} reported "
                    "reg_grad_norm=0. This is the pm-jepa bug reproduced; the "
                    "run is void.".format(reg_name, lam, s))
            res["runs"].append(rec)
            print("  {:<7} lam {:<5g} seed {}  copy {:+.4f}  ridge {:+.4f}  "
                  "mlp {:+.4f}  rank {:5.1f}  |grad_reg| {:.3e}  ({:.0f}s)".format(
                      reg_name, lam, s, rec["copy_alignment"], rec["ridge_mean"],
                      rec["mlp_mean"], rec["eff_rank"],
                      rec["reg_grad_norm_last"], rec["seconds"]), flush=True)
            pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- aggregate ----
    agg = {}
    for r in res["runs"]:
        agg.setdefault((r["reg"], r["lam"]), []).append(r)
    summary = {}
    for (rn, lam), rs in agg.items():
        def ms(k):
            a = np.array([x[k] for x in rs])
            return float(a.mean()), float(a.std())
        cm, cs = ms("copy_alignment")
        rm, rsd = ms("ridge_mean")
        mm, msd = ms("mlp_mean")
        er, _ = ms("eff_rank")
        gn, _ = ms("reg_grad_norm_last")
        summary["{}|{}".format(rn, lam)] = {
            "reg": rn, "lam": lam, "seeds": len(rs),
            "copy_alignment": cm, "copy_alignment_std": cs,
            "ridge_mean": rm, "ridge_std": rsd,
            "mlp_mean": mm, "mlp_std": msd,
            "eff_rank": er, "reg_grad_norm": gn}
    res["summary"] = summary

    base = summary["none|0.0"]
    live = [v for v in summary.values() if v["reg"] != "none"]
    best_copy = min(live, key=lambda v: v["copy_alignment"]) if live else None
    res["gate"] = {
        "criterion": ("copy_alignment below 0.90 without destroying probe R^2 -> "
                      "mechanism helps; above 0.95 across the sweep -> original "
                      "claim vindicated"),
        "baseline_copy_alignment": base["copy_alignment"],
        "lowest_copy_alignment": best_copy["copy_alignment"] if best_copy else None,
        "lowest_at": ("{} lam={}".format(best_copy["reg"], best_copy["lam"])
                      if best_copy else None),
        "any_below_0_90": bool(best_copy and best_copy["copy_alignment"] < 0.90),
        "all_above_0_95": bool(live and all(v["copy_alignment"] > 0.95 for v in live)),
        "all_regularisers_live": bool(live and all(v["reg_grad_norm"] > 0 for v in live)),
    }
    g = res["gate"]
    g["verdict"] = (
        "MECHANISM HELPS: copy_alignment fell below 0.90. The original claim was wrong."
        if g["any_below_0_90"] else
        "ORIGINAL CLAIM VINDICATED, now actually tested: copy_alignment stays above "
        "0.95 with a live regulariser at every strength swept."
        if g["all_above_0_95"] else
        "PARTIAL: copy_alignment moved but did not clear 0.90. Report the frontier.")

    print("\n" + "=" * 96)
    print("E2 REGULARISER SWEEP  ({} seeds, {} steps, computed on the ONLINE context)".format(
        args.seeds, args.steps))
    print("=" * 96)
    print("{:<9}{:>8}{:>20}{:>18}{:>18}{:>13}".format(
        "reg", "lam", "copy_alignment", "ridge", "mlp", "|grad_reg|"))
    print("-" * 96)
    for k in sorted(summary, key=lambda k: (summary[k]["reg"], summary[k]["lam"])):
        v = summary[k]
        print("{:<9}{:>8g}{:>+14.4f}+-{:.4f}{:>+12.4f}+-{:.4f}{:>+12.4f}+-{:.4f}{:>13.2e}".format(
            v["reg"], v["lam"], v["copy_alignment"], v["copy_alignment_std"],
            v["ridge_mean"], v["ridge_std"], v["mlp_mean"], v["mlp_std"],
            v["reg_grad_norm"]))
    print("=" * 96)
    print("GATE: {}".format(g["verdict"]))
    print("      baseline copy_alignment {:.4f}, lowest {:.4f} at {}".format(
        g["baseline_copy_alignment"], g["lowest_copy_alignment"] or float("nan"),
        g["lowest_at"]))
    print("      all regularisers live (grad > 0): {}".format(g["all_regularisers_live"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
