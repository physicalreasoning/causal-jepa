#!/usr/bin/env python3
"""Independent integration check, written against the SPEC rather than against
any single module author's own harness.

Each of the four modules was verified by whoever wrote it, in isolation, with
fixtures of its own choosing. That catches bugs inside a module and is blind to
the class of bug that only appears when two modules meet: an axis order that
both sides agree on but that means different things, a default argument that
happens to match today, a diagnostic reading a mask the model built with
different arithmetic. So nothing here reuses a fixture from `verify_model.py`,
`verify_masking_diagnostics.py` or `verify_data_probes.py`; the numbers are
recomputed from the public API only.

Run: python3 scripts/integration_check.py
"""
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa.cache import KVCache, encode_incremental          # noqa: E402
from causaljepa.diagnostics import (                              # noqa: E402
    copy_diagnostics, directional_copy_diagnostics, effective_rank,
    target_autocorrelation)
from causaljepa.masking import interpolation_span, sample_mask    # noqa: E402
from causaljepa.model import CausalJEPA, FactorisedEncoder        # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print("{:<52} {}  {}".format(name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(name)


def section(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ---------------------------------------------------------------------------
section("1. cache exactness, independently recomputed")
# ---------------------------------------------------------------------------
torch.manual_seed(1234)
B, T, K, C = 5, 24, 24, 4
enc = FactorisedEncoder(K, C, 6, 4, d_model=64, n_layers=3, n_heads=8, causal=True).eval()
obs = torch.randn(B, T, K, C)

with torch.no_grad():
    full = enc(obs)

# One patch at a time, growing prefix, default calling convention.
cache = KVCache.empty()
pieces = []
with torch.no_grad():
    for p in range(6):
        tok, cache = encode_incremental(enc, obs[:, : (p + 1) * 4], cache)
        pieces.append(tok)
inc = torch.cat(pieces, dim=1)
dev1 = float((inc - full).abs().max())
check("stride-1 incremental == full re-encode", dev1 < 1e-4,
      "max|delta| = {:.6e}".format(dev1))
check("cache.n_seen after 6 patches", cache.n_seen == 6, "n_seen={}".format(cache.n_seen))
check("cache holds one K/V per layer", len(cache.k) == 3 == len(cache.v),
      "layers={}".format(len(cache.k)))
check("cache K layout is (B*K, heads, p_seen, head_dim)",
      tuple(cache.k[0].shape) == (B * K, 8, 6, 8), tuple(cache.k[0].shape))

# Two at a time, to prove the offset arithmetic is not accidentally right at 1.
cache2 = KVCache.empty()
pieces = []
with torch.no_grad():
    for p in range(0, 6, 2):
        tok, cache2 = encode_incremental(enc, obs[:, : (p + 2) * 4], cache2)
        pieces.append(tok)
dev2 = float((torch.cat(pieces, dim=1) - full).abs().max())
check("2-patch chunks == full re-encode", dev2 < 1e-4, "max|delta| = {:.6e}".format(dev2))

# True streaming: hand over only the new minutes.
cache3 = KVCache.empty()
pieces = []
with torch.no_grad():
    for p in range(6):
        tok, cache3 = encode_incremental(enc, obs[:, p * 4:(p + 1) * 4], cache3,
                                         start_patch=p)
        pieces.append(tok)
dev3 = float((torch.cat(pieces, dim=1) - full).abs().max())
check("true-streaming form == full re-encode", dev3 < 1e-4,
      "max|delta| = {:.6e}".format(dev3))

cache3.reset()
check("reset() empties the cache", cache3.n_seen == 0 and not cache3.k)

# Teeth. Two independent demonstrations that the bar is not trivially satisfied.
bidir = FactorisedEncoder(K, C, 6, 4, d_model=64, n_layers=3, n_heads=8, causal=False).eval()
bidir.load_state_dict(enc.state_dict())
try:
    encode_incremental(bidir, obs, None)
    check("encode_incremental refuses a bidirectional encoder", False, "no raise")
except ValueError as e:
    check("encode_incremental refuses a bidirectional encoder", True, str(e)[:40] + "...")

with torch.no_grad():
    full_bidir = bidir(obs)
bidir.causal = True                       # lift the guard, same weights
cache4 = KVCache.empty()
pieces = []
with torch.no_grad():
    for p in range(6):
        tok, cache4 = encode_incremental(bidir, obs[:, : (p + 1) * 4], cache4)
        pieces.append(tok)
dev_teeth = float((torch.cat(pieces, dim=1) - full_bidir).abs().max())
check("same cached path vs BIDIRECTIONAL full forward diverges",
      dev_teeth > 1e-2, "max|delta| = {:.6e}, {:.0f}x the bar".format(
          dev_teeth, dev_teeth / 1e-4))

# ---------------------------------------------------------------------------
section("2. causality: the past does not move")
# ---------------------------------------------------------------------------
for causal, want_zero in ((True, True), (False, False)):
    e = FactorisedEncoder(K, C, 6, 4, d_model=64, n_layers=3, n_heads=8,
                          causal=causal).eval()
    o2 = obs.clone()
    o2[:, 12:16] += 5.0                    # patch 3
    with torch.no_grad():
        a, b = e(obs), e(o2)
    past = float((a[:, :3] - b[:, :3]).abs().max())
    future = float((a[:, 3:] - b[:, 3:]).abs().max())
    check("causal={}: patches < 3 unchanged".format(causal),
          (past == 0.0) == want_zero, "past {:.3e}  future {:.3e}".format(past, future))

# ---------------------------------------------------------------------------
section("3. streaming speed (H4)")
# ---------------------------------------------------------------------------
big = torch.randn(64, T, K, C)
with torch.no_grad():
    t0 = time.time()
    for _ in range(10):
        for p in range(6):
            enc(big[:, : (p + 1) * 4])
    naive = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        cc = KVCache.empty()
        for p in range(6):
            encode_incremental(enc, big[:, : (p + 1) * 4], cc)
    cached = time.time() - t0
check("cached streaming beats full re-encode", cached < naive,
      "{:.2f}s vs {:.2f}s, {:.2f}x".format(naive, cached, naive / cached))

# ---------------------------------------------------------------------------
section("4. model <-> masking <-> diagnostics wiring")
# ---------------------------------------------------------------------------
m = CausalJEPA(T, K, C, d_model=64, n_layers=2, causal_context=False,
               causal_target=True)
check("n_patches from (T=24, len=4, stride=4)", m.n_patches == 6, m.n_patches)

for strat, want in (("temporal", {4, 5}), ("interpolation", {2, 3})):
    sm = sample_mask(4, T, K, strat, patch_length=m.patch_length)
    pm = m.patch_mask(sm)
    got = set(torch.nonzero(pm[0].any(dim=-1)).flatten().tolist())
    check("{} masks patches {}".format(strat, sorted(want)), got == want, sorted(got))
check("interpolation_span(24, 0.25, 4)", interpolation_span(24, 0.25, 4) == (8, 16),
      interpolation_span(24, 0.25, 4))

sm = sample_mask(4, T, K, "interpolation", patch_length=4)
out = m(torch.randn(4, T, K, C), sm)
shapes = {k: tuple(v.shape) for k, v in out.items()}
check("forward() keys",
      set(out) == {"pred", "target", "context", "mask_flat", "patch_mask"}, sorted(out))
check("forward() shapes",
      shapes == {"pred": (4, 144, 64), "target": (4, 144, 64), "context": (4, 144, 64),
                 "mask_flat": (4, 144), "patch_mask": (4, 6, 24)}, shapes)
check("target is detached", not out["target"].requires_grad)
check("represent() -> (B, d)", tuple(m.represent(torch.randn(4, T, K, C)).shape) == (4, 64))

cd = copy_diagnostics(out["pred"], out["target"], out["patch_mask"])
dd = directional_copy_diagnostics(out["pred"], out["target"], out["patch_mask"])
ac = target_autocorrelation(out["target"], out["patch_mask"])
check("copy_alignment in [-1, 1]", -1.0 <= cd["copy_alignment"] <= 1.0,
      "{:+.4f}".format(cd["copy_alignment"]))
check("interp_advantage finite under interpolation masking",
      dd["interp_advantage"] == dd["interp_advantage"],
      "{:+.4f} (paired_frac {:.2f})".format(dd["interp_advantage"], dd["paired_frac"]))
check("cross_boundary finite", ac["cross_boundary"] == ac["cross_boundary"],
      "{:+.4f}".format(ac["cross_boundary"]))
check("eff_rank finite and > 1", 1.0 < effective_rank(out["context"]) <= 64,
      "{:.2f}".format(effective_rank(out["context"])))

# The 2x2 must actually reach the flags.
for cc_, ct_ in ((False, False), (False, True), (True, False), (True, True)):
    mm = CausalJEPA(T, K, C, d_model=32, n_layers=1,
                    causal_context=cc_, causal_target=ct_)
    check("2x2 cell ({}, {}) reaches the encoders".format(cc_, ct_),
          mm.encoder.causal is cc_ and mm.target_encoder.causal is ct_)

# ---------------------------------------------------------------------------
section("5. nan safety in the training loop's diagnostic block")
# ---------------------------------------------------------------------------
empty = torch.zeros(2, 6, K, dtype=torch.bool)
tgt = torch.randn(2, 6 * K, 32)
for fn, nm in ((copy_diagnostics, "copy_diagnostics"),
               (directional_copy_diagnostics, "directional_copy_diagnostics")):
    r = fn(tgt, tgt, empty)
    check("{} survives an empty mask".format(nm), isinstance(r, dict))
r = target_autocorrelation(tgt, torch.ones(2, 6, K, dtype=torch.bool))
check("target_autocorrelation survives a full mask", r["cross_boundary"] != r["cross_boundary"])

print("\n" + "=" * 78)
if FAILURES:
    print("FAILED: {}".format(FAILURES))
    sys.exit(1)
print("all integration checks passed")
