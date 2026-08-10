#!/usr/bin/env python3
"""Plant a known copy structure, check the diagnostics report exactly it.

Not a substitute for tests/; this is the bench that produced the numbers quoted
in the module notes. Run: python3 scripts/verify_masking_diagnostics.py
"""
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa.diagnostics import (  # noqa: E402
    copy_diagnostics, dimension_stats, directional_copy_diagnostics,
    effective_rank, nearest_visible_reference, target_autocorrelation)
from causaljepa.masking import interpolation_span, sample_mask  # noqa: E402

torch.manual_seed(0)
B, T, K, P, D = 8, 24, 24, 6, 16
PL = 4


def patchify(mask, n_patches=P, patch_length=PL):
    b, t, k = mask.shape
    return mask.reshape(b, n_patches, patch_length, k).any(dim=2)


def head(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------- masking
head("1. masking: which patches each strategy holds out")
gen = torch.Generator().manual_seed(0)
for strat in ("temporal", "slate", "contiguous", "mixed", "random", "interpolation"):
    m = sample_mask(B, T, K, strat, generator=gen, n_held_out=6, patch_length=PL)
    pm = patchify(m)
    per_patch = pm.float().mean(dim=(0, 2))
    print("{:<14} minute_frac {:.3f}   patch_frac_by_p {}".format(
        strat, m.float().mean().item(),
        " ".join("{:.2f}".format(v) for v in per_patch.tolist())))
    assert m.view(B, -1).any(1).all() and (~m.view(B, -1)).any(1).all(), strat

print("interpolation_span(24, 0.25, 4) =", interpolation_span(24, 0.25, 4))
m_int = sample_mask(B, T, K, "interpolation", patch_length=PL)
pm_int = patchify(m_int)
assert pm_int[:, [2, 3]].all() and not pm_int[:, [0, 1, 4, 5]].any()
m_tmp = sample_mask(B, T, K, "temporal")
pm_tmp = patchify(m_tmp)
assert pm_tmp[:, [4, 5]].all() and not pm_tmp[:, [0, 1, 2, 3]].any()
print("interpolation masks patches 2,3 with visible context on BOTH sides: ok")
print("temporal masks patches 4,5, no visible context after: ok")
for w, pl in ((24, 4), (24, 6), (12, 4), (8, 4), (4, 4), (3, 4)):
    print("  window {:>2} patch {:>1} -> span {}".format(w, pl, interpolation_span(w, 0.25, pl)))

# --------------------------------------------- directional copy diagnostics
head("2. directional copy oracle on planted copies (P=6, K=24, d=16)")
base = torch.randn(B, P, K, D)

# Arm FORWARD: the masked patches are literal copies of the first visible patch
# AFTER the hole, so the interpolation oracle is exact and extrapolation is not.
fwd = base.clone()
fwd[:, 2] = fwd[:, 4]
fwd[:, 3] = fwd[:, 4]
# Arm BACKWARD: masked patches copy the last visible patch BEFORE the hole.
bwd = base.clone()
bwd[:, 2] = bwd[:, 1]
bwd[:, 3] = bwd[:, 1]

pm = pm_int.clone()
zero_pred = torch.zeros(B, P, K, D)
for name, tgt in (("forward-copy", fwd), ("backward-copy", bwd), ("independent", base)):
    dd = directional_copy_diagnostics(zero_pred.reshape(B, -1, D),
                                      tgt.reshape(B, -1, D), pm)
    print("{:<15} extrap_oracle {:.4f}  interp_oracle {:.4f}  advantage {:+.4f}  "
          "extrap_frac {:.2f} interp_frac {:.2f} paired {:.2f}".format(
              name, dd["extrap_oracle_loss"], dd["interp_oracle_loss"],
              dd["interp_advantage"], dd["extrap_frac"], dd["interp_frac"],
              dd["paired_frac"]))
assert directional_copy_diagnostics(zero_pred.reshape(B, -1, D), fwd.reshape(B, -1, D), pm)["interp_advantage"] > 0
assert directional_copy_diagnostics(zero_pred.reshape(B, -1, D), bwd.reshape(B, -1, D), pm)["interp_advantage"] < 0
print("planted forward copy -> advantage > 0; planted backward copy -> advantage < 0: ok")

print("\nalignment: feed the oracle itself as the prediction")
back_pred = torch.zeros_like(base)
back_pred[:, 2] = base[:, 1]
back_pred[:, 3] = base[:, 1]
dd = directional_copy_diagnostics(back_pred.reshape(B, -1, D), base.reshape(B, -1, D), pm)
print("  pred == backward oracle: extrap_alignment {:+.4f}  extrap_loss_ratio {:.4f}".format(
    dd["extrap_alignment"], dd["extrap_loss_ratio"]))
assert abs(dd["extrap_alignment"] - 1.0) < 1e-5 and abs(dd["extrap_loss_ratio"] - 1.0) < 1e-5

head("3. nan-safety")
dd = directional_copy_diagnostics(zero_pred.reshape(B, -1, D), base.reshape(B, -1, D), pm_tmp)
print("temporal mask (nothing visible after): interp_frac {:.1f} interp_ratio {} "
      "advantage {} extrap_frac {:.2f}".format(
          dd["interp_frac"], dd["interp_loss_ratio"], dd["interp_advantage"], dd["extrap_frac"]))
assert dd["interp_frac"] == 0.0 and dd["interp_advantage"] != dd["interp_advantage"]
assert dd["extrap_frac"] == 1.0

empty = torch.zeros(B, P, K, dtype=torch.bool)
dd = directional_copy_diagnostics(zero_pred.reshape(B, -1, D), base.reshape(B, -1, D), empty)
print("empty mask:", {k: v for k, v in dd.items() if not isinstance(v, float) or v == v or True})
assert all(v != v for k, v in dd.items() if k.endswith(("alignment", "ratio", "advantage", "loss")))
cd = copy_diagnostics(zero_pred.reshape(B, -1, D), base.reshape(B, -1, D), empty)
print("copy_diagnostics on empty mask:", cd)

full = torch.ones(B, P, K, dtype=torch.bool)
dd = directional_copy_diagnostics(zero_pred.reshape(B, -1, D), base.reshape(B, -1, D), full)
print("fully masked: extrap_frac {:.1f} interp_frac {:.1f} advantage {}".format(
    dd["extrap_frac"], dd["interp_frac"], dd["interp_advantage"]))
assert dd["extrap_frac"] == 0.0 and dd["interp_frac"] == 0.0

ac = target_autocorrelation(base.reshape(B, -1, D), full)
print("target_autocorrelation fully masked: cross_boundary {} boundary_excess {}".format(
    ac["cross_boundary"], ac["boundary_excess"]))
assert ac["cross_boundary"] != ac["cross_boundary"]
ac1 = target_autocorrelation(base[:, :1].reshape(B, -1, D), pm[:, :1])
print("target_autocorrelation with P=1: lag_1 {} cross_boundary {}".format(
    ac1["lag_1"], ac1["cross_boundary"]))

# ------------------------------------------------- target autocorrelation
head("4. target_autocorrelation: planted leakage vs planted causality")
eye = torch.eye(D)[:P]                                  # orthonormal per patch
causal_t = eye.view(1, P, 1, D).expand(B, P, K, D).contiguous()
leaky_t = causal_t.clone()
blend = F.normalize(eye.mean(dim=0), dim=-1)            # mean of ALL six patches
leaky_t[:, 2] = blend
leaky_t[:, 3] = blend
uniform_t = F.normalize(causal_t + 3.0, dim=-1)         # every pair inflated equally

for name, tt in (("orthogonal (causal)", causal_t),
                 ("leaky masked=mean(all)", leaky_t),
                 ("uniformly inflated", uniform_t)):
    ac = target_autocorrelation(tt.reshape(B, -1, D), pm)
    print("{:<24} lag_1 {:+.4f} lag_2 {:+.4f}  cross_boundary {:+.4f}  "
          "within_side {:+.4f}  boundary_excess {:+.4f}  matched {}  pairs {:.0f}".format(
              name, ac["lag_1"], ac["lag_2"], ac["cross_boundary"],
              ac["within_side"], ac["boundary_excess"],
              ac["boundary_excess_matched"], ac["cross_boundary_pairs"]))
print("expected leak excess = <e_p, mean(e)>/||mean(e)|| = 1/sqrt(6) = {:.4f}".format(
    (1.0 / 6 ** 0.5)))
assert target_autocorrelation(leaky_t.reshape(B, -1, D), pm)["boundary_excess"] > 0.2
assert abs(target_autocorrelation(uniform_t.reshape(B, -1, D), pm)["boundary_excess"]) < 1e-4
ac = target_autocorrelation(causal_t.reshape(B, -1, D), pm)
assert ac["boundary_excess_matched"] != ac["boundary_excess_matched"], \
    "deterministic mask must leave the pair-matched control undefined"

head("5. the confounds, and which control removes which (random-walk targets)")
step = torch.randn(B, P, K, D) * 0.35
walk = torch.cumsum(step, dim=1)
for label, mk in (("interpolation (2,3)", pm), ("temporal (4,5)", pm_tmp)):
    ac = target_autocorrelation(walk.reshape(B, -1, D), mk)
    print("{:<22} lag_1 {:+.4f} lag_2 {:+.4f} lag_3 {:+.4f}  cross_boundary {:+.4f}  "
          "within_side {:+.4f}  boundary_excess {:+.4f}".format(
              label, ac["lag_1"], ac["lag_2"], ac["lag_3"], ac["cross_boundary"],
              ac["within_side"], ac["boundary_excess"]))
print("raw cross_boundary moves with the hole. boundary_excess removes the lag")
print("confound but NOT the patch-position one: a zero-started walk is not")
print("stationary in p, so temporal masking still reads +0.13 with no leakage.")
print("Neither is zero-referenced; both are BETWEEN-ARM statistics.")

print("\npair-matched control, mask varying per (b, p, k) so a control exists:")
rmask = torch.rand(B, P, K) < 0.4
leak = walk.clone()
pooled = F.normalize(walk.mean(dim=1, keepdim=True).expand(B, P, K, D), dim=-1)
leak = torch.where(rmask.unsqueeze(-1), pooled, F.normalize(walk, dim=-1))
for label, tt in (("no leakage planted", walk), ("masked=mean(window)", leak)):
    ac = target_autocorrelation(tt.reshape(B, -1, D), rmask)
    print("  {:<22} cross_boundary {:+.4f}  boundary_excess {:+.4f}  "
          "boundary_excess_matched {:+.4f}".format(
              label, ac["cross_boundary"], ac["boundary_excess"],
              ac["boundary_excess_matched"]))
assert abs(target_autocorrelation(walk.reshape(B, -1, D), rmask)["boundary_excess_matched"]) < 0.02
assert target_autocorrelation(leak.reshape(B, -1, D), rmask)["boundary_excess_matched"] > 0.05

head("6. ported functions still behave")
x = torch.randn(512, 32)
print("effective_rank(iid gaussian, d=32) = {:.2f}".format(effective_rank(x)))
low = torch.randn(512, 4) @ torch.randn(4, 32)
print("effective_rank(rank-4 subspace)    = {:.2f}".format(effective_rank(low)))
print("dimension_stats(constant vector)   =", dimension_stats(torch.ones(512, 32)))
ref, valid = nearest_visible_reference(base.reshape(B, -1, D), pm)
print("nearest_visible_reference prefers the past: ref for patch 2 equals patch 1 ->",
      bool(torch.equal(ref[:, 2], base[:, 1])), " valid.all() ->", bool(valid.all()))
cd = copy_diagnostics(ref.reshape(B, -1, D), base.reshape(B, -1, D), pm)
print("copy_diagnostics with pred == oracle:", {k: round(v, 6) for k, v in cd.items()})

head("7. mps device path (cummax stays on cpu)")
if torch.backends.mps.is_available():
    dev = "mps"
    t_m = base.reshape(B, -1, D).to(dev)
    p_m = zero_pred.reshape(B, -1, D).to(dev)
    m_m = pm.to(dev)
    dd = directional_copy_diagnostics(p_m, t_m, m_m)
    ac = target_autocorrelation(t_m, m_m)
    cd = copy_diagnostics(p_m, t_m, m_m)
    print("mps ok: extrap_frac {:.2f} interp_frac {:.2f} advantage {:+.4f} "
          "cross_boundary {:+.4f} copy_ratio {:.4f}".format(
              dd["extrap_frac"], dd["interp_frac"], dd["interp_advantage"],
              ac["cross_boundary"], cd["copy_loss_ratio"]))
    dd_c = directional_copy_diagnostics(zero_pred.reshape(B, -1, D), base.reshape(B, -1, D), pm)
    assert abs(dd["interp_advantage"] - dd_c["interp_advantage"]) < 1e-5
    print("mps and cpu agree on interp_advantage to 1e-5")
else:
    print("mps unavailable, skipped")

print("\nALL CHECKS PASSED")
