#!/usr/bin/env python3
"""End-to-end smoke run on a corpus subset: every code path, sane numbers.

Not a unit test and not a result. The unit tests pin each module's contract in
isolation; this runs the actual pipeline the experiment driver runs, on the real
corpus, and asserts that the numbers coming out are the KIND of numbers they are
supposed to be. The failures it is looking for are the ones that pass every shape
check: a loss that does not move, an effective rank that has collapsed to 1, a
copy alignment outside [-1, 1] because two modules disagree about an axis, a
probe R^2 that is nan because the frozen representations are constant.

It deliberately runs three arms rather than one, because half the wiring only
exists in the causal cells, and both masking strategies, because
`interpolation` is the only one that produces a finite `interp_advantage`.

Run: python3 scripts/smoke.py [--steps 30] [--windows 300]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa import probes                                     # noqa: E402
from causaljepa.cache import KVCache, encode_incremental          # noqa: E402
from causaljepa.data import load_corpus, split_by_event           # noqa: E402
from causaljepa.train import pick_device, represent_all, train_arm  # noqa: E402

PROBLEMS = []


def expect(name, ok, detail=""):
    print("  {:<46} {}  {}".format(name, "ok " if ok else "BAD", detail))
    if not ok:
        PROBLEMS.append("{}: {}".format(name, detail))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pm-jepa-root", default=str(pathlib.Path.home() / "pm-jepa"))
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    t_load = time.time()
    X, Y, owner, names = load_corpus(args.pm_jepa_root)
    print("corpus {} loaded in {:.1f}s, targets {}".format(
        tuple(X.shape), time.time() - t_load, names))

    # Subsample whole EVENTS, not whole windows. Taking 300 arbitrary rows would
    # hand the event split a corpus where most events contribute one window, and
    # the leak-free split this repo exists to preserve would become vacuous.
    rng = np.random.default_rng(0)
    ids = np.unique(owner)
    rng.shuffle(ids)
    keep = np.zeros(len(owner), dtype=bool)
    for e in ids:
        keep |= owner == e
        if keep.sum() >= args.windows:
            break
    sub = np.flatnonzero(keep)[: args.windows]
    Xs, Ys, os_ = X[sub], Y[sub], owner[sub]
    tr, te = split_by_event(os_, frac=0.2, seed=0)
    print("subset {} windows over {} events, train {} / test {}, device {}\n".format(
        len(Xs), len(np.unique(os_)), int(tr.sum()), int(te.sum()), device))

    summary = {}
    arms = [("temporal", False, False),
            ("temporal", False, True),
            ("interpolation", False, True),
            ("interpolation", True, True)]

    for strategy, cc, ct in arms:
        tag = "{}|ctx={}|tgt={}".format(strategy, "c" if cc else "b", "c" if ct else "b")
        print("=" * 78)
        print("ARM {}".format(tag))
        print("=" * 78)
        model, hist, secs = train_arm(
            Xs, tr, strategy=strategy, causal_context=cc, causal_target=ct,
            device=device, steps=args.steps, batch=args.batch, lr=1e-3, lam=0.0,
            d_model=args.d_model, n_layers=args.layers, n_held_out=6, seed=0,
            log_every=max(1, args.steps // 6), verbose=True)

        first, last = hist[0], hist[-1]
        print("  {} steps in {:.1f}s".format(args.steps, secs))

        expect("pred_loss falls", last["pred_loss"] < first["pred_loss"],
               "{:.5f} -> {:.5f}".format(first["pred_loss"], last["pred_loss"]))
        expect("pred_loss finite", np.isfinite(last["pred_loss"]),
               "{:.5f}".format(last["pred_loss"]))
        expect("eff_rank finite and >> 1",
               1.5 < last["eff_rank"] <= args.d_model, "{:.2f}".format(last["eff_rank"]))
        expect("copy_alignment in [-1, 1]",
               -1.0 <= last["copy_alignment"] <= 1.0,
               "{:+.4f}".format(last["copy_alignment"]))
        expect("copy_loss_ratio finite and > 0",
               np.isfinite(last["copy_loss_ratio"]) and last["copy_loss_ratio"] > 0,
               "{:.4f}".format(last["copy_loss_ratio"]))
        expect("reg_grad_norm is 0.0 (no regulariser)",
               last["reg_grad_norm"] == 0.0, repr(last["reg_grad_norm"]))
        expect("min_dim_std > 0 (no dead representation)",
               last["min_dim_std"] > 1e-6, "{:.2e}".format(last["min_dim_std"]))
        expect("target autocorr lag_1..lag_5 all finite",
               all(np.isfinite(last["lag_{}".format(L)]) for L in range(1, 6)),
               " ".join("{:+.3f}".format(last["lag_{}".format(L)]) for L in range(1, 6)))
        expect("cross_boundary finite", np.isfinite(last["cross_boundary"]),
               "{:+.4f}".format(last["cross_boundary"]))
        if strategy == "interpolation":
            expect("interp_advantage finite (both sides visible)",
                   np.isfinite(last["interp_advantage"]),
                   "{:+.5f} paired_frac {:.2f}".format(
                       last["interp_advantage"], last["paired_frac"]))
        else:
            expect("interp_advantage nan under temporal masking",
                   not np.isfinite(last["interp_advantage"]), "nan as expected")
        expect("history is JSON-serialisable",
               isinstance(json.loads(json.dumps(hist)), list))

        # Probe on frozen representations, the H3 measurement.
        f = represent_all(model, Xs, device)
        expect("representations finite", bool(np.isfinite(f).all()),
               "{} std {:.3e}".format(f.shape, float(f.std())))
        r = probes.run_probes(f[tr], Ys[tr], f[te], Ys[te], names, device="cpu")
        expect("ridge R^2 is a real number",
               all(np.isfinite(v) for v in r["ridge"].values()),
               "mean {:+.4f} lambda {}".format(r["ridge_mean"], r["ridge_lambda"]))
        expect("mlp R^2 is a real number",
               all(np.isfinite(v) for v in r["mlp"].values()),
               "mean {:+.4f}".format(r["mlp_mean"]))
        print("     per target: " + "  ".join(
            "{} {:+.3f}".format(n[:14], r["ridge"][n]) for n in names))

        # The cache only exists for the causal-context arms; exercise it here on
        # real data rather than on gaussians, because a NaN or an inf in the
        # corpus would show up as a tolerance failure and nowhere else.
        if cc:
            model.eval()
            obs = torch.from_numpy(Xs[:8]).to(device)
            with torch.no_grad():
                full = model.encode(obs, None, target=False)
                cache, pieces = KVCache.empty(), []
                for p in range(model.n_patches):
                    tok, cache = encode_incremental(model, obs[:, : (p + 1) * 4], cache)
                    pieces.append(tok)
            dev = float((torch.cat(pieces, dim=1) - full).abs().max())
            expect("cache exact on real corpus data", dev < 1e-4,
                   "max|delta| = {:.6e}".format(dev))
            model.train()

        summary[tag] = {"pred_loss_first": first["pred_loss"],
                        "pred_loss_last": last["pred_loss"],
                        "eff_rank": last["eff_rank"],
                        "copy_alignment": last["copy_alignment"],
                        "copy_loss_ratio": last["copy_loss_ratio"],
                        "lag_1": last["lag_1"],
                        "cross_boundary": last["cross_boundary"],
                        "interp_advantage": last["interp_advantage"],
                        "ridge_mean": r["ridge_mean"], "mlp_mean": r["mlp_mean"],
                        "seconds": round(secs, 1)}
        print()
        del model

    print("=" * 78)
    print("SMOKE SUMMARY  ({} windows, {} steps, d_model {}, {} layers, {})".format(
        len(Xs), args.steps, args.d_model, args.layers, device))
    print("=" * 78)
    cols = ["pred_loss_first", "pred_loss_last", "eff_rank", "copy_alignment",
            "copy_loss_ratio", "lag_1", "cross_boundary", "ridge_mean"]
    print("{:<26}".format("arm") + "".join("{:>13}".format(c[:12]) for c in cols))
    for tag, s in summary.items():
        print("{:<26}".format(tag) + "".join(
            "{:>13}".format("{:+.4f}".format(s[c])) for c in cols))
    print("=" * 78)

    out = ROOT / "results" / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": vars(args), "device": device, "summary": summary,
         "problems": PROBLEMS}, indent=2))
    print("wrote {}".format(out))

    if PROBLEMS:
        print("\nPROBLEMS:")
        for p in PROBLEMS:
            print("  " + p)
        return 1
    print("\nsmoke run clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
