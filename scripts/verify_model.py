"""Verification harness for causaljepa/model.py and causaljepa/cache.py."""
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from causaljepa.cache import KVCache, encode_incremental  # noqa: E402
from causaljepa.model import CausalJEPA, FactorisedEncoder  # noqa: E402

torch.manual_seed(0)
B, T, K, C = 3, 24, 24, 4
P, PL, PS = 6, 4, 4


def build(causal):
    torch.manual_seed(7)
    m = FactorisedEncoder(K, C, P, PL, d_model=128, n_layers=4, n_heads=8,
                          causal=causal, patch_stride=PS).double()
    return m.eval()


def build_f32(causal):
    torch.manual_seed(7)
    return FactorisedEncoder(K, C, P, PL, d_model=128, n_layers=4, n_heads=8,
                             causal=causal, patch_stride=PS).eval()


print("=" * 70)
print("1. CACHE EXACTNESS  (float32, patches fed one at a time vs full forward)")
print("=" * 70)
obs = torch.randn(B, T, K, C)
for causal in (True, False):
    enc = build_f32(causal)
    with torch.no_grad():
        full = enc(obs)
    if not causal:
        try:
            cache = KVCache.empty()
            encode_incremental(enc, obs[:, :PL], cache)
            print("  causal=False: NO ValueError raised  <-- BUG")
        except ValueError as e:
            print("  causal=False: encode_incremental refuses, as required")
            print("                ({})".format(str(e)[:70]))
        # Prove the refusal is not merely defensive: run the same cached path
        # with the guard lifted and show it does NOT reproduce the full encode.
        enc.causal = True                       # lie to the guard on purpose
        cache = KVCache.empty()
        outs = []
        with torch.no_grad():
            for p in range(P):
                tok, cache = encode_incremental(enc, obs[:, :(p + 1) * PS], cache)
                outs.append(tok)
        inc = torch.cat(outs, dim=1)
        enc.causal = False
        with torch.no_grad():
            bidir_full = build_f32(False)(obs)
        dev = (inc - bidir_full).abs().max().item()
        print("  causal=False: max|incremental - full| = {:.6e}  (must exceed 1e-4)".format(dev))
        print("               teeth: {}".format("YES" if dev > 1e-4 else "NO  <-- BUG"))
        continue

    cache = KVCache.empty()
    outs = []
    with torch.no_grad():
        for p in range(P):
            tok, cache = encode_incremental(enc, obs[:, :(p + 1) * PS], cache)
            assert tok.shape == (B, 1, K, 128), tok.shape
            outs.append(tok)
    inc = torch.cat(outs, dim=1)
    assert inc.shape == full.shape, (inc.shape, full.shape)
    dev = (inc - full).abs().max().item()
    print("  causal=True : max|incremental - full| = {:.6e}  (bar 1e-4)  {}".format(
        dev, "PASS" if dev < 1e-4 else "FAIL"))
    print("               cache.n_seen = {}, layers cached = {}, k[0].shape = {}".format(
        cache.n_seen, len(cache.k), tuple(cache.k[0].shape)))

    # Same thing in chunks of 2 and 3 patches, and via the streaming form.
    for chunk in (2, 3):
        cache = KVCache.empty()
        outs = []
        with torch.no_grad():
            for p0 in range(0, P, chunk):
                n = min(chunk, P - p0)
                tok, cache = encode_incremental(enc, obs[:, :(p0 + n) * PS], cache)
                outs.append(tok)
        dev_c = (torch.cat(outs, 1) - full).abs().max().item()
        print("  causal=True : chunk={}  max dev = {:.6e}  {}".format(
            chunk, dev_c, "PASS" if dev_c < 1e-4 else "FAIL"))

    cache = KVCache.empty()
    outs = []
    with torch.no_grad():
        for p in range(P):
            tok, cache = encode_incremental(
                enc, obs[:, p * PS:p * PS + PL], cache, start_patch=p)
            outs.append(tok)
    dev_s = (torch.cat(outs, 1) - full).abs().max().item()
    print("  causal=True : start_patch streaming form max dev = {:.6e}  {}".format(
        dev_s, "PASS" if dev_s < 1e-4 else "FAIL"))

    # float64 sanity: if the fp32 deviation is just accumulation noise, fp64
    # should be many orders of magnitude tighter.
    encd = build(True)
    with torch.no_grad():
        fulld = encd(obs.double())
        cached = KVCache.empty()
        outs = [encd(obs.double()[:, p * PS:p * PS + PL], cache=cached) for p in range(P)]
    print("  causal=True : float64 max dev = {:.3e}".format(
        (torch.cat(outs, 1) - fulld).abs().max().item()))

    cache.reset()
    print("  reset(): n_seen={}, len(k)={}".format(cache.n_seen, len(cache.k)))

print()
print("=" * 70)
print("2. CAUSALITY  (perturb input at patch p, look at tokens at patches < p)")
print("=" * 70)
obs = torch.randn(B, T, K, C)
pert = obs.clone()
p_star = 3
pert[:, p_star * PS:(p_star + 1) * PS] += 5.0
for causal in (True, False):
    enc = build_f32(causal)
    with torch.no_grad():
        a, b = enc(obs), enc(pert)
    before = (a[:, :p_star] - b[:, :p_star]).abs().max().item()
    at_after = (a[:, p_star:] - b[:, p_star:]).abs().max().item()
    print("  causal={:<5}  max|delta| at patches <{} : {:.6e}".format(causal, p_star, before))
    print("               max|delta| at patches >={} : {:.6e}".format(p_star, at_after))
    if causal:
        print("               past unchanged: {}  (bit-identical: {})".format(
            before == 0.0, bool(torch.equal(a[:, :p_star], b[:, :p_star]))))
    else:
        print("               past DOES change: {}".format(before > 1e-4))

print()
print("=" * 70)
print("3. SHAPES / KEYS / patch_mask geometry")
print("=" * 70)
model = CausalJEPA(T, K, C, causal_context=False, causal_target=True)
sm = torch.zeros(B, T, K, dtype=torch.bool)
sm[:, T - 6:, :] = True                       # temporal: last 25% of minutes
out = model(torch.randn(B, T, K, C), sm)
for key, want in (("pred", (B, P * K, 128)), ("target", (B, P * K, 128)),
                  ("context", (B, P * K, 128)), ("mask_flat", (B, P * K)),
                  ("patch_mask", (B, P, K))):
    got = tuple(out[key].shape)
    print("  {:<11} {}  expected {}  {}".format(
        key, got, want, "OK" if got == want else "MISMATCH"))
print("  keys == pm-jepa keys: {}".format(
    set(out.keys()) == {"pred", "target", "context", "mask_flat", "patch_mask"}))
print("  target requires_grad: {} (must be False)".format(out["target"].requires_grad))
print("  n_patches = {} (expected 6)".format(model.n_patches))
pm = out["patch_mask"]
print("  masked patch indices under temporal: {}".format(
    sorted(set(pm[0].any(dim=1).nonzero().flatten().tolist()))))
print("  represent(): {}".format(tuple(model.represent(torch.randn(B, T, K, C)).shape)))

print()
print("  EMA update:")
w0 = model.target_encoder.blocks[0].time_attn.q_proj.weight.clone()
model.encoder.blocks[0].time_attn.q_proj.weight.data.add_(1.0)
model.update_target(momentum=0.5)
w1 = model.target_encoder.blocks[0].time_attn.q_proj.weight
print("    target moved by mean {:.4f} at momentum 0.5 (expected 0.5)".format(
    float((w1 - w0).mean())))
_m = CausalJEPA(T, K, C, causal_context=False, causal_target=True)
print("    towers identical at init: {}".format(all(
    torch.equal(a, b) for a, b in zip(_m.encoder.parameters(),
                                      _m.target_encoder.parameters()))))

print()
print("  prefix consistency (causal encoder, no cache involved):")
enc = build_f32(True)
obs = torch.randn(B, T, K, C)
with torch.no_grad():
    dev_p = (enc(obs[:, :4 * PS]) - enc(obs)[:, :4]).abs().max().item()
print("    max|enc(first 4 patches) - enc(full)[:, :4]| = {:.6e}".format(dev_p))
enc = build_f32(False)
with torch.no_grad():
    dev_b = (enc(obs[:, :4 * PS]) - enc(obs)[:, :4]).abs().max().item()
print("    bidirectional same quantity = {:.6e}  (must be large)".format(dev_b))

print()
print("  2x2 flags -> encoder causality:")
for cc in (False, True):
    for ct in (False, True):
        m = CausalJEPA(T, K, C, causal_context=cc, causal_target=ct)
        print("    causal_context={:<5} causal_target={:<5} -> enc.causal={} tgt.causal={}".format(
            cc, ct, m.encoder.causal, m.target_encoder.causal))

print()
print("=" * 70)
print("4. STREAMING SPEED (stride-1 style: extend the window one patch at a time)")
print("=" * 70)
enc = build_f32(True)
big = torch.randn(64, T, K, C)
with torch.no_grad():
    for _ in range(2):                          # warm up
        enc(big)
    t0 = time.perf_counter()
    for _ in range(20):
        for p in range(1, P + 1):
            enc(big[:, :p * PS])
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        cache = KVCache.empty()
        for p in range(P):
            encode_incremental(enc, big[:, :(p + 1) * PS], cache)
    t_inc = time.perf_counter() - t0
print("  full re-encode per patch: {:.3f}s   cached: {:.3f}s   speedup {:.2f}x".format(
    t_full, t_inc, t_full / t_inc))

print()
print("=" * 70)
print("5. MASKED-INPUT / MPS / TRAINING SMOKE")
print("=" * 70)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = CausalJEPA(T, K, C, causal_context=True, causal_target=True).to(dev)
o = torch.randn(8, T, K, C, device=dev)
sm = torch.zeros(8, T, K, dtype=torch.bool, device=dev)
sm[:, 16:] = True
out = m(o, sm)
loss = torch.nn.functional.smooth_l1_loss(
    out["pred"] * out["mask_flat"].unsqueeze(-1),
    out["target"] * out["mask_flat"].unsqueeze(-1))
loss.backward()
gn = sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None) ** 0.5
print("  device={}  loss={:.5f}  grad_norm={:.4f}".format(dev, float(loss), gn))
print("  target tower has no grads: {}".format(
    all(p.grad is None for p in m.target_encoder.parameters())))
print("  n params: {:,}".format(sum(p.numel() for p in m.parameters())))
print("  zero-fill honoured: masked minutes contribute nothing extra ->", end=" ")
o2 = o.clone()
o2[:, 16:] += 99.0
print(bool(torch.allclose(m.encode(o, sm), m.encode(o2, sm), atol=1e-6)))
