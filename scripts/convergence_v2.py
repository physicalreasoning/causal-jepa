#!/usr/bin/env python3
"""E12: re-run E1's convergence sweep against a target that can detect learning.

E1 concluded that 32x more compute drives the probe 23 pooled SD BELOW an
untrained encoder, and that conclusion is leg 1 of the 3-part kill criterion that
closed this programme. It was measured before finding 10 existed, and it carries
finding 10's defect plus finding 9's:

  * WRONG TARGET. E1 probed the original four targets. Finding 10 showed those
    are mostly derivable or unpredictable, and finding 21 showed that choosing a
    target on how clean it looks is exactly what hid the real result in E7.
  * WORST READOUT. `probe_now` calls `represent_all`, which is `mean_all`, the
    pooling finding 9 identified as the one that loses the most.

So E1 may be wrong for precisely the reason E7's headline was wrong, and nobody
caught it because E1 ran first. This re-runs it with both defects removed.

WHAT CHANGES, AND ONLY THIS. Same 19,200 steps, same 3 seeds, same schedule, same
checkpoints, same architecture, same untrained-floor-at-the-same-seed protocol.
The probe target becomes `fwd_realised_vol` RESIDUALISED against the strongest
raw control, which is the one quantity in this corpus where finding 21 showed a
training lift survives every control. Every readout is probed at every
checkpoint rather than just `mean_all`.

WHY THE RESIDUAL AND NOT THE RAW TARGET. Findings 14 and 17 are two separate
cases of a large training lift that was entirely input reconstruction. Tracking
the raw R^2 across a long run would measure reconstruction improving and call it
learning. The residual has everything raw features explain already removed, so a
rising curve there cannot be reconstruction.

NOTE ON COMPARABILITY. E1 ran on the full 25,818-window corpus; this runs on the
17,449 windows that survive the 10-step horizon requirement. Absolute numbers are
therefore not comparable to E1's. The untrained floor is measured inside this run
at the same seeds, so the trained-versus-untrained comparison, which is the whole
question, is internally valid.

Pre-registered gate:
  residual lift over the untrained floor at the FINAL checkpoint exceeds 2 pooled
  SD, and the trajectory is non-decreasing over the last two checkpoints
      -> finding 8 REVERSES. The programme was closed on a measurement artefact,
         and E6 (scale), which was gated on E1 separating, opens for the first
         time.
  lift present early and gone by the end
      -> finding 8 survives in its interesting form: the objective destroys
         information it briefly had, now demonstrated on a target that can
         actually see it.
  no lift at any checkpoint
      -> finding 21 does not survive a longer budget, and finding 8 stands as
         written.
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

from causaljepa import data as cj_data, probes, targets as cj_targets  # noqa: E402
from causaljepa.diagnostics import copy_diagnostics, effective_rank  # noqa: E402
from causaljepa.masking import sample_mask  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import (ema_momentum, jepa_loss, lr_multiplier,  # noqa: E402
                              pick_device)

import readout_ablation as e4  # noqa: E402
from residual_probe import residualise  # noqa: E402

TARGET = "fwd_realised_vol"
CONTROL = "raw_last_4min"
SD_BAR = 2.0


def probe_all(model, X, device, tr, te, r_tr, r_te):
    """-> {readout: residual ridge R^2}. Every readout, not just mean_all."""
    out = {}
    for rn, f in e4.extract(model, X, device).items():
        f = f.astype(np.float64)
        r = probes.run_probes(f[tr], r_tr, f[te], r_te, [TARGET + "_residual"],
                              device="cpu")
        out[rn] = r["ridge"][TARGET + "_residual"]
        del f
    return out


def run_seed(X, device, tr, te, r_tr, r_te, *, seed, steps, checkpoints,
             batch, lr, d_model, n_layers, strategy, n_held_out, on_mark=None):
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

    floor = probe_all(model, X, device, tr, te, r_tr, r_te)
    if on_mark is not None:
        on_mark({"step": 0, "seed": seed, "residual": floor, "is_floor": True})
    print("  seed {} untrained floor  ".format(seed)
          + "  ".join("{} {:+.4f}".format(k[:6], v) for k, v in floor.items()),
          flush=True)

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
            p = probe_all(model, X, device, tr, te, r_tr, r_te)
            rec = {"step": n, "seed": seed, "residual": p,
                   "pred_loss": float(loss["pred_loss"]),
                   "copy_alignment": cd["copy_alignment"],
                   "eff_rank": er, "lr_mult": lr_multiplier(step, steps),
                   "elapsed_s": round(time.time() - t0, 1)}
            marks.append(rec)
            # Flush after EVERY checkpoint, not every seed. A 2.7-hour run that
            # only writes on seed completion loses everything to any
            # interruption, which is exactly what happened on the first attempt.
            if on_mark is not None:
                on_mark(rec)
            print("  seed {} step {:>6}  ".format(seed, n)
                  + "  ".join("{} {:+.4f}".format(k[:6], v) for k, v in p.items())
                  + "  copy {:+.3f} rank {:.1f} ({:.0f}s)".format(
                      cd["copy_alignment"], er, rec["elapsed_s"]), flush=True)
    del model
    return floor, marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--half", type=int, default=5,
                    help="forward window for fwd_realised_vol; 5 matches E11, "
                         "which is where the surviving lift was demonstrated")
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
    ap.add_argument("--out", default="results/convergence_v2.json")
    ap.add_argument("--resume", action="store_true",
                    help="reload --out and skip seeds that already reached every "
                         "checkpoint. Runs of this length are being interrupted, "
                         "so completing one seed per invocation is the only way "
                         "the sweep finishes at all.")
    ap.add_argument("--only-seed", type=int, default=None,
                    help="run exactly one seed and exit, for chunked execution")
    args = ap.parse_args()

    device = pick_device(args.device)
    cks = {int(c) for c in args.checkpoints.split(",") if c}
    # Finding 21's surviving lift was measured on `statetargets` at half=5, whose
    # `fwd_realised_vol` is realised vol over the NEXT 5 minutes. `targets.py` at
    # horizon=10 defines the same name over 10 minutes on a different window set,
    # and the untrained residual floor differs sharply between them (+0.1251
    # against +0.0195 for concat_strikes). Testing convergence of finding 21's
    # effect therefore has to use finding 21's target, not a same-named one.
    from causaljepa import statetargets as stt
    X, Yn, owner, nn_, cstats = stt.load_corpus(args.pm_jepa_root, half=args.half)
    tr, te = cj_data.split_by_event(owner)
    y = Yn[:, list(nn_).index(TARGET)].astype(np.float64)
    ctl = e4.raw_controls(X)[CONTROL].astype(np.float64)
    r_tr, r_te, ctl_r2 = residualise(ctl[tr], ctl[te], y[tr], y[te])

    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)
    print("target {} residualised against {} (explains {:+.4f})".format(
        TARGET, CONTROL, ctl_r2), flush=True)
    print("{} seeds x {} steps, checkpoints {}\n".format(
        args.seeds, args.steps, sorted(cks)), flush=True)

    res = {"config": vars(args), "device": device, "corpus": cstats,
           "target": TARGET, "control": CONTROL, "control_r2": ctl_r2,
           "checkpoints": sorted(cks), "floors": [], "marks": []}

    outp0 = pathlib.Path(args.out)
    done = set()
    if args.resume and outp0.exists():
        prev = json.loads(outp0.read_text())
        res["floors"] = prev.get("floors", [])
        res["marks"] = prev.get("marks", [])
        reached_prev = {}
        for m in res["marks"]:
            reached_prev.setdefault(m["seed"], set()).add(m["step"])
        done = {f["seed"] for f in res["floors"]
                if reached_prev.get(f["seed"], set()) >= cks}
        # A seed that started but did not finish must be discarded wholesale, or
        # it contributes some checkpoints and not others and silently changes n
        # between columns of the same table.
        partial = {f["seed"] for f in res["floors"]} - done
        if partial:
            res["floors"] = [f for f in res["floors"] if f["seed"] not in partial]
            res["marks"] = [m for m in res["marks"] if m["seed"] not in partial]
            print("resume: discarding partial seed(s) {}".format(sorted(partial)),
                  flush=True)
        print("resume: seeds already complete {}".format(sorted(done)), flush=True)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(exist_ok=True)

    def flush(rec=None):
        # The untrained floor arrives through the same hook so that a kill
        # mid-seed leaves a floor AND its checkpoints on disk, rather than a
        # trajectory with nothing to measure it against.
        if rec is not None:
            if rec.pop("is_floor", False):
                res["floors"].append({"seed": rec["seed"], "residual": rec["residual"]})
            else:
                res["marks"].append(rec)
        outp.write_text(json.dumps(res, indent=2, default=str))

    todo = [args.only_seed] if args.only_seed is not None else list(range(args.seeds))
    for s in todo:
        if s in done:
            print("seed {} already complete, skipping".format(s), flush=True)
            continue
        floor, _marks = run_seed(
            X, device, tr, te, r_tr, r_te, seed=s, steps=args.steps,
            checkpoints=cks, batch=args.batch, lr=args.lr,
            d_model=args.d_model, n_layers=args.layers,
            strategy=args.strategy, n_held_out=args.n_held_out, on_mark=flush)
        flush()

    readouts = sorted(res["floors"][0]["residual"])
    # A seed only counts if it has a floor and reached every checkpoint. An
    # interrupted final seed would otherwise contribute some checkpoints and not
    # others, silently changing n between columns of the same table.
    reached = {}
    for m in res["marks"]:
        reached.setdefault(m["seed"], set()).add(m["step"])
    complete = {f["seed"] for f in res["floors"]
                if reached.get(f["seed"], set()) >= set(sorted(cks))}
    res["seeds_complete"] = sorted(complete)
    if not complete:
        raise SystemExit("no seed completed every checkpoint; nothing to aggregate")
    by_ck = {}
    for m in res["marks"]:
        if m["seed"] in complete:
            by_ck.setdefault(m["step"], []).append(m)
    floor_by_ro = {ro: np.array([f["residual"][ro] for f in res["floors"]
                                 if f["seed"] in complete])
                   for ro in readouts}

    traj = {}
    for ro in readouts:
        fl = floor_by_ro[ro]
        row = []
        for ck in sorted(by_ck):
            a = np.array([m["residual"][ro] for m in by_ck[ck]])
            sd = (a.std() ** 2 + fl.std() ** 2) ** 0.5
            row.append({"step": ck, "mean": float(a.mean()), "std": float(a.std()),
                        "lift": float(a.mean() - fl.mean()),
                        "in_sd": float((a.mean() - fl.mean()) / sd) if sd > 1e-12
                        else float("nan")})
        traj[ro] = {"floor_mean": float(fl.mean()), "floor_std": float(fl.std()),
                    "checkpoints": row}
    res["trajectory"] = traj

    best_ro = max(readouts, key=lambda ro: traj[ro]["checkpoints"][-1]["in_sd"])
    fin = traj[best_ro]["checkpoints"][-1]
    prev = traj[best_ro]["checkpoints"][-2]
    peak = max(traj[best_ro]["checkpoints"], key=lambda c: c["lift"])
    # Every threshold here is in pooled SD, which does not exist below two seeds:
    # the floor std is then exactly 0, every `in_sd` is nan, and `nan > SD_BAR`
    # is False, so BOTH branches read as "no lift" and the run prints a verdict
    # it has no evidence for. This is the fifth criterion in this programme to
    # need this guard; it refuses instead.
    decidable = len(complete) >= 2
    reverses = bool(decidable and np.isfinite(fin["in_sd"])
                    and fin["in_sd"] > SD_BAR
                    and fin["lift"] >= prev["lift"] - 1e-9)
    any_lift = bool(decidable and any(
        np.isfinite(c["in_sd"]) and c["in_sd"] > SD_BAR
        for ro in readouts for c in traj[ro]["checkpoints"]))

    res["gate"] = {
        "decidable": decidable,
        "best_readout": best_ro, "final": fin, "previous": prev, "peak": peak,
        "reverses_finding_8": reverses, "any_checkpoint_lift": any_lift,
        "verdict": (
            "UNDECIDABLE: {} complete seed(s). Every threshold here is in pooled "
            "SD, which needs at least 2.".format(len(complete)) if not decidable else
            "FINDING 8 REVERSES: the residual lift at the final checkpoint is "
            "{:+.2f} pooled SD on {} and the trajectory is not falling. The "
            "programme was closed on a measurement artefact, and E6 opens.".format(
                fin["in_sd"], best_ro) if reverses else
            "FINDING 8 SURVIVES, in its interesting form: the lift peaks at step "
            "{} ({:+.4f}) and ends at {:+.4f}, so the objective destroys "
            "information it briefly had, now shown on a target that can see "
            "it.".format(peak["step"], peak["lift"], fin["lift"])
            if any_lift else
            "NO LIFT AT ANY BUDGET: finding 21 does not survive a longer run and "
            "finding 8 stands as written.")}

    w = 96
    print("\n" + "=" * w)
    print("E12 CONVERGENCE ON THE RESIDUAL  ({} seeds x {} steps)".format(
        args.seeds, args.steps))
    print("=" * w)
    for ro in readouts:
        t = traj[ro]
        print("\n{}  (untrained floor {:+.4f} +- {:.4f})".format(
            ro, t["floor_mean"], t["floor_std"]))
        print("  " + "".join("{:>13}".format(c["step"]) for c in t["checkpoints"]))
        print("  " + "".join("{:>+13.4f}".format(c["mean"]) for c in t["checkpoints"]))
        print("  " + "".join("{:>12.2f}s".format(c["in_sd"]) for c in t["checkpoints"]))
    print("\n" + "=" * w)
    print("GATE: {}".format(res["gate"]["verdict"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
