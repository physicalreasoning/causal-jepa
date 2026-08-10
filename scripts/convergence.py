#!/usr/bin/env python3
"""E1: run it to convergence.

Every conclusion across slate-jepa, pm-jepa and causal-jepa is conditioned on a
600-step budget, and the load-bearing null (MLP probe indistinguishable from an
untrained encoder) has never been run further. This settles it.

DESIGN NOTE, read before comparing anything.

  The learning-rate schedule is a 100-step warmup times a cosine decay scaled to
  the TOTAL step count, so a checkpoint at step 600 of a 19,200-step run is not
  the same model as a 600-step run: the first is near peak learning rate, the
  second has fully decayed. Intermediate checkpoints here trace ONE long
  schedule. They are a trajectory, not a set of independent budgets, and the
  600-step checkpoint must NOT be compared against results/causal_ablation.json.

  The comparison that matters is the FINAL checkpoint against the untrained
  floor at the same seed, and that one is exact.

Pre-registered gate (RESEARCH_PLAN.md E1), wording corrected after the first run:
  MLP R^2 at the FINAL checkpoint exceeds the untrained floor by > 2 pooled SD,
  AND the trajectory is non-decreasing over the last two checkpoints
      -> the objective does learn; the 600-step null was a budget artefact
  no separation at the final checkpoint
      -> strong negative; proceed to E3
  separation at an early checkpoint that then reverses
      -> also a strong negative, and a more interesting one

The original criterion said "at any checkpoint", which a transient early peak
satisfies while the trajectory reverses. The first run hit exactly that case:
+3.30 pooled SD at step 600, then -23.17 by step 19,200. The transient is still
reported, it is just no longer allowed to decide.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes  # noqa: E402
from causaljepa.diagnostics import copy_diagnostics, effective_rank  # noqa: E402
from causaljepa.masking import sample_mask  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import (ema_momentum, jepa_loss, lr_multiplier,  # noqa: E402
                              pick_device, represent_all)


def probe_now(model, X, Y, tr, te, names, device):
    f = represent_all(model, X, device)
    r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
    return {"ridge_mean": r["ridge_mean"], "mlp_mean": r["mlp_mean"],
            "ridge": r["ridge"], "mlp": r["mlp"]}


def run_seed(X, Y, tr, te, names, *, seed, steps, checkpoints, batch, lr,
             d_model, n_layers, strategy, n_held_out, device):
    """One long run, probed at `checkpoints`. Mirrors train_arm exactly."""
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

    # Untrained floor at this exact seed, before a single gradient step.
    floor = probe_now(model, X, Y, tr, te, names, device)
    print("  seed {} untrained floor   ridge {:+.4f}  mlp {:+.4f}".format(
        seed, floor["ridge_mean"], floor["mlp_mean"]), flush=True)

    marks, t0 = [], time.time()
    for step in range(steps):
        sel = torch.from_numpy(np.random.choice(idx_pool, batch, replace=False))
        obs = Xt[sel].to(device)
        mk = sample_mask(batch, T, K, strategy, generator=gen,
                         n_held_out=n_held_out, device="cpu",
                         patch_length=model.patch_length).to(device)
        out = model(obs, mk)
        loss = jepa_loss(out["pred"], out["target"], out["mask_flat"])

        opt.zero_grad(set_to_none=True)
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target(momentum=ema_momentum(step, steps))

        n = step + 1
        if n in checkpoints:
            with torch.no_grad():
                cd = copy_diagnostics(out["pred"], out["target"], out["patch_mask"])
                er = effective_rank(out["context"].detach().cpu())
            p = probe_now(model, X, Y, tr, te, names, device)
            rec = {"step": n, "seed": seed,
                   "pred_loss": float(loss["pred_loss"]),
                   "copy_alignment": cd["copy_alignment"],
                   "copy_loss_ratio": cd["copy_loss_ratio"],
                   "eff_rank": er, "reg_grad_norm": 0.0,
                   "lr_mult": lr_multiplier(step, steps),
                   "elapsed_s": round(time.time() - t0, 1)}
            rec.update(p)
            marks.append(rec)
            print("  seed {} step {:>6}  ridge {:+.4f}  mlp {:+.4f}  "
                  "copy {:+.3f}  rank {:.1f}  lr {:.3f}x  ({:.0f}s)".format(
                      seed, n, p["ridge_mean"], p["mlp_mean"],
                      cd["copy_alignment"], er, rec["lr_mult"],
                      rec["elapsed_s"]), flush=True)
    del model
    return floor, marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root", default="/Users/nikita/pm-jepa")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=19200)
    ap.add_argument("--checkpoints", default="600,1200,2400,4800,9600,19200")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--n-held-out", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/convergence.json")
    args = ap.parse_args()

    cks = sorted({int(c) for c in args.checkpoints.split(",") if c})
    device = pick_device(args.device)
    X, Y, owner, names = cj_data.load_corpus(args.pm_jepa_root)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)
    print("E1: {} seeds x {} steps, checkpoints at {}".format(
        args.seeds, args.steps, cks), flush=True)

    res = {"config": vars(args), "device": device, "checkpoints": cks,
           "design_note": ("Checkpoints trace ONE long cosine schedule scaled to "
                           "--steps. The step-600 checkpoint is near peak learning "
                           "rate and is NOT comparable to results/causal_ablation.json, "
                           "whose 600-step run had a fully decayed schedule. Only the "
                           "final checkpoint versus the untrained floor is an exact "
                           "comparison."),
           "corpus": {"shape": list(X.shape), "n_train": int(tr.sum()),
                      "n_test": int(te.sum()), "target_names": list(names)},
           "floors": [], "marks": []}

    for seed in range(args.seeds):
        floor, marks = run_seed(
            X, Y, tr, te, names, seed=seed, steps=args.steps, checkpoints=set(cks),
            batch=args.batch, lr=args.lr, d_model=args.d_model,
            n_layers=args.layers, strategy=args.strategy,
            n_held_out=args.n_held_out, device=device)
        floor["seed"] = seed
        res["floors"].append(floor)
        res["marks"].extend(marks)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- aggregate and evaluate the pre-registered gate ----
    fr = np.array([f["ridge_mean"] for f in res["floors"]])
    fm = np.array([f["mlp_mean"] for f in res["floors"]])
    floor_agg = {"ridge_mean": float(fr.mean()), "ridge_std": float(fr.std()),
                 "mlp_mean": float(fm.mean()), "mlp_std": float(fm.std()),
                 "seeds": len(fr)}
    res["untrained_floor"] = floor_agg

    by_step, sep_step = {}, None
    for c in cks:
        rs = [m for m in res["marks"] if m["step"] == c]
        if not rs:
            continue
        r = np.array([m["ridge_mean"] for m in rs])
        m_ = np.array([m["mlp_mean"] for m in rs])
        pooled = float((m_.std() ** 2 + fm.std() ** 2) ** 0.5)
        sep = float((m_.mean() - fm.mean()) / pooled) if pooled > 1e-12 else float("nan")
        by_step[c] = {
            "step": c, "seeds": len(rs),
            "ridge_mean": float(r.mean()), "ridge_std": float(r.std()),
            "mlp_mean": float(m_.mean()), "mlp_std": float(m_.std()),
            "mlp_lift_over_floor": float(m_.mean() - fm.mean()),
            "mlp_lift_in_pooled_sd": sep,
            "copy_alignment": float(np.mean([m["copy_alignment"] for m in rs])),
            "eff_rank": float(np.mean([m["eff_rank"] for m in rs])),
        }
        if sep > 2.0 and sep_step is None:
            sep_step = c
    res["by_step"] = by_step
    # The gate keys on the FINAL checkpoint, not on any checkpoint. An earlier
    # version keyed on "any", which a transient early peak satisfies while the
    # trajectory reverses; that cannot answer whether more training helps. The
    # first run of this script hit exactly that case, so the transient is still
    # reported, just no longer allowed to decide.
    fin = by_step.get(cks[-1]) if cks else None
    fin_sep = fin["mlp_lift_in_pooled_sd"] if fin else float("nan")
    rising = (len(cks) >= 2 and by_step.get(cks[-1]) and by_step.get(cks[-2])
              and by_step[cks[-1]]["mlp_mean"] >= by_step[cks[-2]]["mlp_mean"])
    separated = bool(fin and fin_sep > 2.0 and rising)
    res["gate"] = {
        "criterion": ("MLP R^2 at the FINAL checkpoint exceeds the untrained floor by "
                      "> 2 pooled SD, AND the trajectory is non-decreasing over the "
                      "last two checkpoints"),
        "separated": separated,
        "final_mlp_lift_in_pooled_sd": fin_sep,
        "trajectory_rising_at_end": bool(rising),
        "transient_peak_step": sep_step,
        "verdict": ("SEPARATES at the final checkpoint: the objective does learn and "
                    "the 600-step null was a budget artefact. Re-run E4 and the causal "
                    "2x2 at this budget." if separated else
                    ("PEAKS AT STEP {} THEN REVERSES: strong negative, and the "
                     "interesting kind. The objective destroys information it briefly "
                     "had. Final MLP is {:+.2f} pooled SD from the untrained floor. "
                     "Proceed to E3.".format(sep_step, fin_sep) if sep_step else
                     "NO SEPARATION by step {}: strong negative. Proceed to E3."
                     .format(args.steps))),
    }

    print("\n" + "=" * 86)
    print("E1 CONVERGENCE  ({} seeds, {} steps)".format(args.seeds, args.steps))
    print("=" * 86)
    print("{:>8}{:>18}{:>18}{:>14}{:>12}".format(
        "step", "ridge", "mlp", "mlp-floor", "in pooled sd"))
    print("-" * 86)
    print("{:>8}{:>+12.4f}+-{:.4f}{:>+12.4f}+-{:.4f}{:>14}{:>12}".format(
        "floor", floor_agg["ridge_mean"], floor_agg["ridge_std"],
        floor_agg["mlp_mean"], floor_agg["mlp_std"], "-", "-"))
    for c in cks:
        v = by_step.get(c)
        if v:
            print("{:>8}{:>+12.4f}+-{:.4f}{:>+12.4f}+-{:.4f}{:>+14.4f}{:>12.2f}".format(
                c, v["ridge_mean"], v["ridge_std"], v["mlp_mean"], v["mlp_std"],
                v["mlp_lift_over_floor"], v["mlp_lift_in_pooled_sd"]))
    print("=" * 86)
    print("GATE: {}".format(res["gate"]["verdict"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
