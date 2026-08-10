#!/usr/bin/env python3
"""H1 without the training-trajectory confound: flip the target flag on FIXED weights.

The 2x2 in `experiment.py` compares two arms that were trained separately. They
are seed-paired, so initialisation and batch order cancel, but the causal target
still feeds a different learning signal from step 1, so by step 600 the two
context encoders are not the same encoder. Any difference in
`target_autocorrelation` is therefore a mixture of two things: the intervention
itself, and 600 steps of divergent optimisation.

This script separates them. Train ONE arm, then evaluate its target encoder
twice, once with `causal=False` and once with `causal=True`, on the same weights
and the same batch. `FactorisedEncoder.causal` is read at forward time and only
selects an attention mask, so flipping it is a pure intervention: no parameter
changes, no retraining.

What that buys: an upper bound. If the weight-matched flip barely moves
`cross_boundary` over its floor, then no amount of seed pairing in the trained
2x2 can be measuring leakage, because the quantity does not respond to the
intervention even when nothing else is allowed to vary. That is the falsification
test H1 needs and the trained contrast cannot supply.

Also reports the per-patch cosine between the two target versions, which says
WHERE the intervention lands. Under `temporal` the hole is patches 4-5, at the
end of the window, where a causal token has already seen almost everything;
under `interpolation` it is patches 2-3, with three patches of future removed.
If the intervention is much smaller at the temporal hole, then `temporal` is the
weaker test of H1 by construction, and that is a fact about the mask, not the
encoder.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa.data import load_corpus, split_by_event        # noqa: E402
from causaljepa.diagnostics import target_autocorrelation      # noqa: E402
from causaljepa.masking import sample_mask                     # noqa: E402
from causaljepa.train import pick_device, train_arm            # noqa: E402


def target_both_ways(model, obs):
    """Same weights, both attention masks. Returns (bidir, causal) as (B,P,K,d)."""
    out = {}
    for name, flag in (("bidir", False), ("causal", True)):
        model.target_encoder.causal = flag
        model.target_encoder._mask_cache = {}
        with torch.no_grad():
            out[name] = model.encode(obs, None, target=True)
    return out["bidir"], out["causal"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root", default="/Users/nikita/pm-jepa")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--strategy", default="temporal",
                    help="masking strategy the ARM is trained under")
    ap.add_argument("--out", default=str(ROOT / "results" / "h1_weight_matched.json"))
    args = ap.parse_args(argv)

    device = pick_device("auto")
    X, Y, owner, names = load_corpus(args.pm_jepa_root)
    tr, te = split_by_event(owner, frac=0.2, seed=0)
    print("corpus {} train {} test {} device {}".format(
        X.shape, int(tr.sum()), int(te.sum()), device), flush=True)

    model, hist, secs = train_arm(
        X, tr, strategy=args.strategy, causal_context=False, causal_target=False,
        device=device, steps=args.steps, batch=64, lr=1e-3, lam=0.0,
        d_model=128, n_layers=4, n_held_out=6, seed=args.seed, log_every=200)
    print("trained in {:.1f}s, final pred_loss {:.4f}".format(
        secs, hist[-1]["pred_loss"]), flush=True)
    model.eval()

    rng = np.random.RandomState(0)
    pick = rng.choice(np.flatnonzero(te), size=args.batch, replace=False)
    obs = torch.from_numpy(X[pick]).to(device)

    tb, tc = target_both_ways(model, obs)
    b, p, k, d = tb.shape

    per_patch = [float(F.cosine_similarity(tb[:, i], tc[:, i], dim=-1).mean())
                 for i in range(p)]
    rel = [float(((tb[:, i] - tc[:, i]).norm(dim=-1)
                  / tb[:, i].norm(dim=-1).clamp_min(1e-9)).mean())
           for i in range(p)]

    res = {"config": vars(args), "device": device, "seconds": secs,
           "final_pred_loss": hist[-1]["pred_loss"],
           "eval_windows": int(args.batch),
           "per_patch_cos_bidir_vs_causal": per_patch,
           "per_patch_rel_l2_bidir_vs_causal": rel,
           "autocorr": {}}

    print("\nper-patch agreement between the two target encoders (same weights)")
    print("{:<8}{:>14}{:>14}".format("patch", "cos", "rel L2"))
    for i in range(p):
        print("{:<8}{:>14.6f}{:>14.6f}".format(i, per_patch[i], rel[i]))

    for strat in ("temporal", "interpolation"):
        gen = torch.Generator().manual_seed(123)
        sm = sample_mask(b, X.shape[1], k, strat, generator=gen,
                         n_held_out=6, device="cpu",
                         patch_length=model.patch_length).to(device)
        pm = model.patch_mask(sm)
        held = sorted({int(i) for i in torch.nonzero(pm[0, :, 0]).flatten()})
        row = {"masked_patches": held}
        for name, t in (("bidir", tb), ("causal", tc)):
            a = target_autocorrelation(t.reshape(b, p * k, d), pm)
            row[name] = {kk: (None if a[kk] != a[kk] else float(a[kk]))
                         for kk in ("lag_1", "lag_3", "cross_boundary",
                                    "within_side", "boundary_excess",
                                    "baseline_cos")}
        row["delta"] = {kk: (None if row["causal"][kk] is None
                             or row["bidir"][kk] is None
                             else row["causal"][kk] - row["bidir"][kk])
                        for kk in row["bidir"]}
        for kk in ("cross_boundary", "lag_3"):
            if row["delta"][kk] is not None and row["delta"]["baseline_cos"] is not None:
                row["delta"][kk + "_over_floor"] = (
                    row["delta"][kk] - row["delta"]["baseline_cos"])
        res["autocorr"][strat] = row

        print("\ntarget_autocorrelation, {} mask, masked patches {}".format(strat, held))
        print("{:<22}{:>14}{:>14}{:>14}".format("key", "bidir", "causal", "delta"))
        for kk in ("lag_1", "lag_3", "cross_boundary", "within_side",
                   "boundary_excess", "baseline_cos"):
            print("{:<22}{:>14.6f}{:>14.6f}{:>14.6f}".format(
                kk, row["bidir"][kk], row["causal"][kk], row["delta"][kk]))
        for kk in ("cross_boundary_over_floor", "lag_3_over_floor"):
            print("{:<22}{:>28}{:>14.6f}".format(kk, "", row["delta"][kk]))

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(res, indent=2))
    print("\nwrote {}".format(out))
    return res


if __name__ == "__main__":
    main()
