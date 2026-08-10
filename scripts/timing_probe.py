#!/usr/bin/env python3
"""Budget probe: seconds-per-step at the REAL corpus size and the real width.

Run before committing to a protocol. The number that matters is not the mean
over a short burst but the mean AFTER the first few steps: on MPS the first
forward pays graph compilation and lazy allocator warmup, and folding that into
the estimate inflates the per-step cost by enough to talk yourself out of a
seed you could have afforded. Warmup steps are timed and reported separately
rather than discarded silently.

Also times the two fixed per-arm costs that are easy to forget when budgeting:
`represent_all` over the full corpus and `run_probes` on the resulting features.
A 2x2 at 3 seeds pays those 12 times, so if they are 30s each they are a third
of the budget on their own.
"""
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa import probes                                     # noqa: E402
from causaljepa.data import load_corpus, split_by_event           # noqa: E402
from causaljepa.masking import sample_mask                        # noqa: E402
from causaljepa.model import CausalJEPA                           # noqa: E402
from causaljepa.train import (jepa_loss, pick_device,             # noqa: E402
                              represent_all)


def probe_steps(X, tr, device, *, causal_context, causal_target, strategy,
                d_model=128, n_layers=4, batch=64, warm=5, n=25):
    torch.manual_seed(0)
    np.random.seed(0)
    N, T, K, C = X.shape
    model = CausalJEPA(T, K, C, d_model=d_model, n_layers=n_layers,
                       causal_context=causal_context,
                       causal_target=causal_target).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.02, betas=(0.9, 0.95))
    Xt = torch.from_numpy(X)
    pool = np.flatnonzero(tr)
    gen = torch.Generator().manual_seed(1)

    times = []
    for step in range(warm + n):
        t0 = time.time()
        sel = torch.from_numpy(np.random.choice(pool, batch, replace=False))
        obs = Xt[sel].to(device)
        mk = sample_mask(batch, T, K, strategy, generator=gen, n_held_out=6,
                         device="cpu", patch_length=model.patch_length).to(device)
        out = model(obs, mk)
        loss = jepa_loss(out["pred"], out["target"], out["mask_flat"])
        opt.zero_grad(set_to_none=True)
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.update_target(momentum=0.996)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.time() - t0)
    return model, np.array(times[:warm]), np.array(times[warm:])


def main():
    device = pick_device("auto")
    t0 = time.time()
    X, Y, owner, names = load_corpus(str(pathlib.Path.home() / "pm-jepa"), 24, 4)
    tr, te = split_by_event(owner, frac=0.2, seed=0)
    print("corpus load {:.1f}s  X={}  train={}  test={}".format(
        time.time() - t0, tuple(X.shape), int(tr.sum()), int(te.sum())), flush=True)

    for strategy in ("temporal", "interpolation"):
        for cc, ct in ((False, False), (False, True), (True, True)):
            model, warm, steady = probe_steps(
                X, tr, device, causal_context=cc, causal_target=ct,
                strategy=strategy)
            print("{:<14} ctx={:<6} tgt={:<6}  warm {:.3f}s/step   steady "
                  "{:.4f} +/- {:.4f} s/step".format(
                      strategy, "causal" if cc else "bidir",
                      "causal" if ct else "bidir",
                      warm.mean(), steady.mean(), steady.std()), flush=True)

    t0 = time.time()
    f = represent_all(model, X, device)
    t_rep = time.time() - t0
    t0 = time.time()
    r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
    t_prb = time.time() - t0
    print("represent_all {:.1f}s   run_probes {:.1f}s   (ridge_mean {:+.3f} on an "
          "untrained-ish net, sanity only)".format(t_rep, t_prb, r["ridge_mean"]),
          flush=True)

    per_step = steady.mean()
    fixed = t_rep + t_prb
    print("\nBUDGET MODEL: arm_seconds(steps) = {:.4f}*steps + {:.1f}".format(
        per_step, fixed))
    for steps in (400, 600, 800, 1000, 1200):
        for cells, seeds in ((4, 3), (4, 4), (4, 3)):
            pass
        a = per_step * steps + fixed
        print("  steps={:<5} per-arm {:5.0f}s   2x2x3seeds {:5.1f}min   "
              "both strategies {:5.1f}min".format(
                  steps, a, 12 * a / 60, 24 * a / 60))


if __name__ == "__main__":
    main()
