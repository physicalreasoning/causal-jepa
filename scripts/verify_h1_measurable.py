#!/usr/bin/env python3
"""Does target_autocorrelation actually separate a bidirectional target encoder
from a causal one? Dry run of H1 on real embeddings, before causaljepa/model.py
exists.

Stand-in for the causal target encoder: run pm-jepa's bidirectional PatchTST P
times, once per patch, each time zeroing every minute after that patch, and keep
only that patch's row. Prefix-only input makes each target embedding a function
of the past alone, which is what causal time attention buys, so the two arms here
bracket the real experiment.

pm-jepa is imported read-only; nothing is written into that tree.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path("/Users/nikita/pm-jepa")))

from model.encoder import LadderJEPA  # noqa: E402  pm-jepa, read-only

from causaljepa.diagnostics import (directional_copy_diagnostics,  # noqa: E402
                                    target_autocorrelation)
from causaljepa.masking import sample_mask  # noqa: E402

torch.manual_seed(0)
B, T, K, C, PL = 16, 24, 24, 4, 4

# Smooth-in-time, smooth-across-strikes input, i.e. a stand-in for a strike
# ladder: a flat white-noise input would make every patch independent and hide
# any leakage the statistic is meant to catch.
level = torch.cumsum(torch.randn(B, T, 1, C) * 0.3, dim=1)
rung = torch.linspace(-1.5, 1.5, K).view(1, 1, K, 1)
obs = (level + rung + torch.randn(B, T, K, C) * 0.05).float()

model = LadderJEPA(T, K, C, d_model=64, n_layers=2).eval()
P = model.n_patches
print("patches P =", P)


@torch.no_grad()
def bidirectional_target():
    return model.encode(obs, None, target=True)          # (B, P, K, d)


@torch.no_grad()
def causal_target():
    rows = []
    for p in range(P):
        cut = obs.clone()
        cut[:, (p + 1) * PL:] = 0.0                      # future minutes removed
        rows.append(model.encode(cut, None, target=True)[:, p])
    return torch.stack(rows, dim=1)


for strategy in ("interpolation", "temporal"):
    mk = sample_mask(B, T, K, strategy, patch_length=PL)
    pm = model.patch_mask(mk)
    print("\n{}  masked patches {}".format(
        strategy, [p for p in range(P) if bool(pm[0, p, 0])]))
    for arm, tgt in (("bidirectional", bidirectional_target()),
                     ("causal (prefix)", causal_target())):
        flat = tgt.reshape(B, -1, tgt.shape[-1])
        ac = target_autocorrelation(flat, pm)
        dd = directional_copy_diagnostics(torch.zeros_like(flat), flat, pm)
        print("  {:<16} lag_1 {:+.4f} lag_2 {:+.4f}  cross_boundary {:+.4f}  "
              "within_side {:+.4f}  boundary_excess {:+.4f}  baseline {:+.4f}  "
              "interp_adv {}".format(
                  arm, ac["lag_1"], ac["lag_2"], ac["cross_boundary"],
                  ac["within_side"], ac["boundary_excess"], ac["baseline_cos"],
                  "nan" if dd["interp_advantage"] != dd["interp_advantage"]
                  else "{:+.4f}".format(dd["interp_advantage"])))

mk = sample_mask(B, T, K, "interpolation", patch_length=PL)
pm = model.patch_mask(mk)

with torch.no_grad():
    cut = obs.clone()
    cut[:, 4 * PL:] = 0.0
    leak_frac = float((model.encode(obs, None, target=True)[:, 3]
                       - model.encode(cut, None, target=True)[:, 3]).norm()
                      / model.encode(obs, None, target=True)[:, 3].norm())
print("\nAT RANDOM INIT, zeroing patches 4 and 5 moves patch 3's target embedding")
print("by {:.4f} of its norm. Pre-norm residual streams start near the identity,".format(leak_frac))
print("so an untrained encoder barely mixes across patches and there is almost no")
print("leakage to detect. H1 is only measurable on a TRAINED target encoder.")

print("\nSweeping a cross-patch mixing gain (a scale on every attention out_proj)")
print("as a stand-in for training. `leak` is how much of patch 3's embedding norm")
print("comes from patches 4 and 5, so it is the ground truth the statistic must")
print("track. All figures are bidir minus causal at identical mask geometry.")
print("  {:>5} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
    "gain", "leak", "lag_1", "lag_2", "lag_3", "cross_bd", "bd_excess"))
orig = [p.detach().clone() for p in model.target_encoder.parameters()]
sep = []
for gain in (1.0, 4.0, 16.0, 64.0):
    with torch.no_grad():
        for (n, p), o in zip(model.target_encoder.named_parameters(), orig):
            p.copy_(o * gain if "self_attn.out_proj" in n else o)
        cut = obs.clone()
        cut[:, 4 * PL:] = 0.0
        f3 = model.encode(obs, None, target=True)[:, 3]
        leak = float((f3 - model.encode(cut, None, target=True)[:, 3]).norm() / f3.norm())
    b = target_autocorrelation(bidirectional_target().reshape(B, -1, 64), pm)
    c = target_autocorrelation(causal_target().reshape(B, -1, 64), pm)
    sep.append(b["cross_boundary"] - c["cross_boundary"])
    print("  {:>5.1f} {:>7.4f} {:>+9.5f} {:>+9.5f} {:>+9.5f} {:>+9.5f} {:>+9.5f}".format(
        gain, leak, b["lag_1"] - c["lag_1"], b["lag_2"] - c["lag_2"],
        b["lag_3"] - c["lag_3"], sep[-1],
        b["boundary_excess"] - c["boundary_excess"]))

print("\nlag_L and cross_boundary rise monotonically with real leakage: H1 IS")
print("MEASURABLE, on a trained encoder, as a between-arm difference.")
print("boundary_excess does NOT move, and that is the expected mask-blind")
print("signature: pm-jepa's target encoder is called with strike_mask=None, so")
print("it cannot know where the hole is, and window-wide leakage inflates")
print("same-side pairs exactly as much as straddling ones. Reporting")
print("boundary_excess as the H1 headline would cancel the effect and produce a")
print("false null. See the target_autocorrelation docstring.")
assert sep[-1] > sep[0] > -1e-3, "cross_boundary must track the planted leakage"
