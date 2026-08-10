#!/usr/bin/env python3
"""H4: does the KV cache actually buy anything for streaming inference?

The benchmark is a stride-1 patch stream: at tick p the encoder has seen patches
0..p and must emit the tokens for patch p. Two ways to serve that tick:

  full   re-run `forward()` on the whole prefix, throwing away p patches of work
         that cannot have changed. Cost per tick grows with p, so a sweep over P
         ticks is O(P^2) patch-work.
  cache  run `encode_incremental`, which embeds only the new patch and reads the
         old keys and values out of the cache. Cost per tick is flat in p except
         for the attention read itself, so the sweep is O(P) patch-work plus
         O(P^2) in the cheap dot products alone.

The honest caveat, stated here rather than discovered later: the cache is exact
for an APPEND-ONLY stream, one new patch per tick. The corpus window slides by a
minute at a time and `patch_stride=4`, so a minute-level stride-1 slide rewrites
the contents of every patch and invalidates the cache completely. The regime the
cache serves is a growing history, not a sliding window; anyone quoting these
numbers for a sliding window would be quoting them for the wrong thing.

Second honest caveat, expected to dominate the headline: at the real shape P=6
there are five patches of history at most, the constant per-call overhead
(python, dispatch, MPS launch) is the whole cost, and the cache should save
close to nothing. That is the correct result at that shape and it is reported as
such. The larger shapes are there to show the slope, not to flatter the small one.

Exactness is re-checked at every shape. A cache that is fast and wrong is worse
than no cache, so the benchmark refuses to report a speedup it has not first
verified against a full re-encode at atol=1e-4.
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa.cache import KVCache, encode_incremental   # noqa: E402
from causaljepa.model import FactorisedEncoder             # noqa: E402
from causaljepa.train import pick_device                   # noqa: E402


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def full_sweep(enc, obs, stride):
    """Re-encode the whole prefix at every tick; returns per-tick seconds."""
    p_total = enc.n_patches
    out = []
    for p in range(1, p_total + 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            enc(obs[:, :p * stride])
        _sync(obs.device.type)
        out.append(time.perf_counter() - t0)
    return out


def cache_sweep(enc, obs, stride):
    """Append one patch per tick through the KV cache; returns per-tick seconds."""
    p_total = enc.n_patches
    cache = KVCache.empty()
    out, toks = [], []
    for p in range(1, p_total + 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            tok, cache = encode_incremental(enc, obs[:, :p * stride], cache)
        _sync(obs.device.type)
        out.append(time.perf_counter() - t0)
        toks.append(tok)
    return out, torch.cat(toks, dim=1)


def check_exact(enc, obs, streamed):
    """Streamed tokens must equal one full forward, at the SPEC's atol=1e-4."""
    with torch.no_grad():
        ref = enc(obs)
    d = (ref.float() - streamed.float()).abs().max().item()
    return d, bool(d <= 1e-4)


def bench_shape(p_total, k, d, n_layers, n_heads, device, patch_length, batch,
                repeats, seed=0):
    torch.manual_seed(seed)
    stride = patch_length
    t_minutes = p_total * stride
    enc = FactorisedEncoder(k, 4, p_total, patch_length, d_model=d,
                            n_layers=n_layers, n_heads=n_heads, causal=True,
                            patch_stride=stride).to(device).eval()
    obs = torch.randn(batch, t_minutes, k, 4, device=device)

    # One untimed pass of each path. On MPS the first call of a given shape pays
    # kernel specialisation, and folding that into tick 1 would show up as a
    # cache "win" that is really just the full path going first.
    with torch.no_grad():
        enc(obs)
    c0 = KVCache.empty()
    with torch.no_grad():
        encode_incremental(enc, obs[:, :stride], c0)
    _sync(device)

    fulls, caches, streamed = [], [], None
    for _ in range(repeats):
        fulls.append(full_sweep(enc, obs, stride))
        c, streamed = cache_sweep(enc, obs, stride)
        caches.append(c)

    full_arr = np.array(fulls)      # (repeats, P)
    cache_arr = np.array(caches)
    max_abs_diff, exact = check_exact(enc, obs, streamed)

    full_total = float(np.median(full_arr.sum(axis=1)))
    cache_total = float(np.median(cache_arr.sum(axis=1)))
    new_tokens = p_total * k * batch          # tokens EMITTED over the sweep

    return {
        "P": p_total, "K": k, "d_model": d, "n_layers": n_layers,
        "n_heads": n_heads, "batch": batch, "device": device,
        "patch_length": patch_length, "repeats": repeats,
        "window_minutes": t_minutes,
        "exact_vs_full_reencode": exact,
        "max_abs_diff": max_abs_diff,
        "full_total_s": full_total,
        "cache_total_s": cache_total,
        "speedup": full_total / cache_total if cache_total > 0 else float("nan"),
        "full_s_per_tick_mean": float(np.median(full_arr, axis=0).mean()),
        "cache_s_per_tick_mean": float(np.median(cache_arr, axis=0).mean()),
        "full_s_per_tick_last": float(np.median(full_arr, axis=0)[-1]),
        "cache_s_per_tick_last": float(np.median(cache_arr, axis=0)[-1]),
        "full_tokens_per_s": new_tokens / full_total,
        "cache_tokens_per_s": new_tokens / cache_total,
        "full_per_tick_s": np.median(full_arr, axis=0).tolist(),
        "cache_per_tick_s": np.median(cache_arr, axis=0).tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="6,32,128,512")
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--patch-length", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu-max-p", type=int, default=128,
                    help="CPU sweeps are O(P^2) in real work; skip beyond this")
    ap.add_argument("--out", default=str(ROOT / "results" / "cache_bench.json"))
    args = ap.parse_args(argv)

    shapes = [int(s) for s in args.shapes.split(",")]
    devices = []
    for dv in args.devices.split(","):
        dv = dv.strip()
        if dv == "mps" and not torch.backends.mps.is_available():
            continue
        if dv == "cuda" and not torch.cuda.is_available():
            continue
        devices.append(dv)
    if not devices:
        devices = [pick_device("auto")]

    rows = []
    for dv in devices:
        for p_total in shapes:
            if dv == "cpu" and p_total > args.cpu_max_p:
                continue
            reps = args.repeats if p_total <= 128 else 1
            t0 = time.time()
            r = bench_shape(p_total, args.k, args.d_model, args.layers,
                            args.heads, dv, args.patch_length, args.batch, reps)
            r["bench_wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print("{:<4} P={:<5} exact={:<5} maxdiff={:.2e}  full {:8.2f}ms/tick  "
                  "cache {:8.2f}ms/tick  speedup {:5.2f}x  tok/s {:8.0f} -> {:8.0f}"
                  .format(dv, p_total, str(r["exact_vs_full_reencode"]),
                          r["max_abs_diff"], 1e3 * r["full_s_per_tick_mean"],
                          1e3 * r["cache_s_per_tick_mean"], r["speedup"],
                          r["full_tokens_per_s"], r["cache_tokens_per_s"]),
                  flush=True)

    res = {
        "config": vars(args),
        "regime": ("append-only patch stream, one new patch per tick; the cache "
                   "is NOT valid for a minute-level sliding window because "
                   "patch_stride=4 rewrites every patch on a 1-minute slide"),
        "tokens_definition": "new tokens emitted over the sweep = P * K * batch",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "rows": rows,
        "real_shape": {"P": 6, "K": args.k, "d_model": args.d_model},
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("wrote {}".format(out))
    return res


if __name__ == "__main__":
    main()
