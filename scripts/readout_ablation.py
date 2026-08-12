#!/usr/bin/env python3
"""E4: is the readout hiding the representation?

`represent()` crushes a (P=6 x K=24) token grid of 128-d embeddings down to one
128-vector by averaging over BOTH axes. The identity control it is measured
against keeps all 24 strikes. If that asymmetry is what loses the information,
then "no JEPA arm beats raw features" is a statement about our pooling, not
about the model, and it is a confound in every result this programme has
produced.

This trains once per seed and extracts every readout from the SAME frozen
encoder, so the only thing varying is the pooling. The untrained encoder is
probed under every readout too: a readout that lifts the trained model must
also be checked against what it does to random features, or we are just
measuring the dimension of the readout.

Pre-registered gate (docs/RESEARCH_PLAN.md E4):
  any readout puts a trained arm above identity (+0.4491 ridge)
      -> the prior conclusion was a readout artefact, stop and correct
  best readout still below identity
      -> pooling exonerated, proceed to E1
"""
import argparse
import os
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import pick_device, train_arm  # noqa: E402

IDENTITY_RIDGE = 0.4491      # pm-jepa/results/baselines.json, the bar to clear


def raw_controls(X: np.ndarray) -> dict:
    """Dimension-matched raw-feature controls. -> {name: (N, dim)}.

    Without these the experiment is rigged. A 3072-d JEPA readout measured
    against a 96-d identity control is not a fair fight: ridge on a wide random
    projection of a structured input is a strong baseline all by itself, which
    the untrained rows in this very experiment demonstrate. To claim a readout
    recovered something, it has to beat RAW FEATURES AT COMPARABLE WIDTH, not
    just the narrowest control anyone happened to run first.

    `raw_identity` reproduces pm-jepa/baselines.py::identity_features exactly,
    so the ladder is anchored to a number already in the literature of this
    programme.
    """
    n, t, k, c = X.shape
    return {
        "raw_identity":  X[:, -4:].mean(axis=1).reshape(n, k * c),   # 96, pm-jepa's control
        "raw_last_min":  X[:, -1].reshape(n, k * c),                 # 96
        "raw_last_4min": X[:, -4:].reshape(n, 4 * k * c),            # 384
        "raw_full":      X.reshape(n, t * k * c),                    # 2304
    }


def readouts(tok: torch.Tensor) -> dict:
    """tok: (B, P, K, d) frozen encoder output -> {name: (B, dim)}.

    All parameter-free. A learned pooler would stop being a frozen-representation
    probe and start being a small trained model, which is a different claim.
    """
    b, p, k, d = tok.shape
    return {
        "mean_all":       tok.mean(dim=(1, 2)),                      # d      current
        "last_patch":     tok[:, -1].mean(dim=1),                    # d      same dim
        "mean_max":       torch.cat([tok.mean(dim=(1, 2)),
                                     tok.amax(dim=(1, 2))], dim=-1),  # 2d
        "concat_patches": tok.mean(dim=2).reshape(b, p * d),          # P*d
        "concat_strikes": tok.mean(dim=1).reshape(b, k * d),          # K*d
    }


@torch.no_grad()
def extract(model, X, device, batch=256):
    """-> {readout_name: float64 (N, dim)}."""
    model.eval()
    acc = None
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        r = readouts(model.encode(xb, None, target=False))
        if acc is None:
            acc = {n: [] for n in r}
        for n, v in r.items():
            acc[n].append(v.float().cpu().numpy())
    model.train()
    return {n: np.concatenate(v).astype(np.float64) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"),
                    help="checkout of the pm-jepa corpus repo; "
                         "defaults to $PM_JEPA_ROOT then ../pm-jepa")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/readout_ablation.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Y, owner, names = cj_data.load_corpus(args.pm_jepa_root)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)

    res = {"config": vars(args), "device": device,
           "identity_ridge_bar": IDENTITY_RIDGE,
           "corpus": {"shape": list(X.shape), "n_train": int(tr.sum()),
                      "n_test": int(te.sum()), "target_names": list(names)},
           "runs": []}

    # Raw controls are deterministic, so they run once rather than per seed.
    print("\n--- raw controls (dimension-matched, seed-independent) ---", flush=True)
    for name, f in raw_controls(X).items():
        p0 = time.time()
        f = f.astype(np.float64)
        r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
        res["runs"].append({"seed": None, "model": "raw", "readout": name,
                            "dim": int(f.shape[1]), "ridge_mean": r["ridge_mean"],
                            "mlp_mean": r["mlp_mean"], "ridge": r["ridge"],
                            "mlp": r["mlp"], "probe_seconds": round(time.time() - p0, 1)})
        print("  {:<9} {:<15} dim {:>5}  ridge {:+.4f}  mlp {:+.4f}  ({:.0f}s)".format(
            "raw", name, f.shape[1], r["ridge_mean"], r["mlp_mean"],
            time.time() - p0), flush=True)
        del f

    for seed in range(args.seeds):
        t0 = time.time()
        print("\n--- seed {} ---".format(seed), flush=True)

        model, hist, secs = train_arm(
            X, tr, strategy=args.strategy, causal_context=False,
            causal_target=False, device=device, steps=args.steps,
            batch=args.batch, lr=1e-3, lam=0.0, d_model=args.d_model,
            n_layers=args.layers, n_held_out=6, seed=seed,
            log_every=args.steps, verbose=False)

        torch.manual_seed(seed)
        np.random.seed(seed)
        untrained = CausalJEPA(X.shape[1], X.shape[2], X.shape[3],
                               d_model=args.d_model, n_layers=args.layers,
                               causal_context=False, causal_target=False).to(device)

        for tag, m in (("trained", model), ("untrained", untrained)):
            feats = extract(m, X, device)
            for name, f in feats.items():
                p0 = time.time()
                r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
                rec = {"seed": seed, "model": tag, "readout": name,
                       "dim": int(f.shape[1]),
                       "ridge_mean": r["ridge_mean"], "mlp_mean": r["mlp_mean"],
                       "ridge": r["ridge"], "mlp": r["mlp"],
                       "probe_seconds": round(time.time() - p0, 1)}
                res["runs"].append(rec)
                print("  {:<9} {:<15} dim {:>5}  ridge {:+.4f}  mlp {:+.4f}  ({:.0f}s)".format(
                    tag, name, rec["dim"], rec["ridge_mean"], rec["mlp_mean"],
                    rec["probe_seconds"]), flush=True)
            del feats
        del model, untrained
        print("  seed {} done in {:.0f}s (train {:.0f}s)".format(
            seed, time.time() - t0, secs), flush=True)

    # ---- aggregate and evaluate the pre-registered gate ----
    agg = {}
    for rec in res["runs"]:
        agg.setdefault((rec["model"], rec["readout"]), []).append(rec)
    summary = {}
    for (tag, name), rs in agg.items():
        rm = np.array([r["ridge_mean"] for r in rs])
        mm = np.array([r["mlp_mean"] for r in rs])
        summary["{}|{}".format(tag, name)] = {
            "model": tag, "readout": name, "dim": rs[0]["dim"], "seeds": len(rs),
            "ridge_mean": float(rm.mean()), "ridge_std": float(rm.std()),
            "mlp_mean": float(mm.mean()), "mlp_std": float(mm.std())}
    res["summary"] = summary

    def best_of(tag):
        vs = [v for v in summary.values() if v["model"] == tag]
        return max(vs, key=lambda v: v["ridge_mean"]) if vs else None

    bt, bu, br = best_of("trained"), best_of("untrained"), best_of("raw")

    # Training lift at MATCHED readout isolates learning from readout width.
    # This is the number that actually answers "did the encoder learn anything",
    # because trained and untrained share a dimension at every readout.
    lift = {}
    for v in summary.values():
        if v["model"] != "trained":
            continue
        u = summary.get("untrained|" + v["readout"])
        if u:
            sd = (v["ridge_std"] ** 2 + u["ridge_std"] ** 2) ** 0.5
            lift[v["readout"]] = {
                "dim": v["dim"],
                "trained": v["ridge_mean"], "untrained": u["ridge_mean"],
                "ridge_lift": v["ridge_mean"] - u["ridge_mean"],
                "mlp_lift": v["mlp_mean"] - u["mlp_mean"],
                "ridge_lift_over_pooled_sd": (
                    (v["ridge_mean"] - u["ridge_mean"]) / sd if sd > 1e-12 else float("nan")),
            }
    res["training_lift_at_matched_readout"] = lift

    beats_raw = bool(bt and br and bt["ridge_mean"] > br["ridge_mean"])
    res["gate"] = {
        "criterion": ("best trained readout ridge_mean > best DIMENSION-MATCHED raw "
                      "control. The original E4 gate compared against the 96-d identity "
                      "control only; that is not a fair bar for a 3072-d readout, since "
                      "an untrained encoder at the same width nearly clears it."),
        "best_trained": {k: bt[k] for k in ("readout", "dim", "ridge_mean", "ridge_std")} if bt else None,
        "best_untrained": {k: bu[k] for k in ("readout", "dim", "ridge_mean", "ridge_std")} if bu else None,
        "best_raw": {k: br[k] for k in ("readout", "dim", "ridge_mean")} if br else None,
        "identity_bar": IDENTITY_RIDGE,
        "beats_identity_96d": bool(bt and bt["ridge_mean"] > IDENTITY_RIDGE),
        "beats_best_raw": beats_raw,
        "verdict": ("READOUT ARTEFACT: a trained readout beats the best raw control, "
                    "stop and correct prior conclusions" if beats_raw else
                    "pooling exonerated: raw features at matched width still win, "
                    "proceed to E1"),
    }

    print("\n" + "=" * 82)
    print("E4 READOUT ABLATION   ({} seeds, {} steps)".format(args.seeds, args.steps))
    print("=" * 82)
    print("{:<11}{:<16}{:>6}{:>18}{:>18}".format("model", "readout", "dim", "ridge", "mlp"))
    print("-" * 82)
    for v in sorted(summary.values(), key=lambda v: (-v["ridge_mean"])):
        print("{:<11}{:<16}{:>6}{:>+12.4f}+-{:.4f}{:>+12.4f}+-{:.4f}".format(
            v["model"], v["readout"], v["dim"],
            v["ridge_mean"], v["ridge_std"], v["mlp_mean"], v["mlp_std"]))
    print("=" * 82)
    print("\nTRAINING LIFT AT MATCHED READOUT  (trained - untrained, same width)")
    print("-" * 82)
    print("{:<16}{:>6}{:>12}{:>12}{:>12}{:>12}".format(
        "readout", "dim", "trained", "untrained", "ridge lift", "in sd"))
    for name, v in sorted(lift.items(), key=lambda kv: -kv[1]["ridge_lift"]):
        print("{:<16}{:>6}{:>+12.4f}{:>+12.4f}{:>+12.4f}{:>12.2f}".format(
            name, v["dim"], v["trained"], v["untrained"],
            v["ridge_lift"], v["ridge_lift_over_pooled_sd"]))
    print("=" * 82)
    g = res["gate"]
    print("GATE: {}".format(g["verdict"]))
    if g["best_trained"] and g["best_raw"]:
        print("      best trained : {:<16} dim {:>5}  {:+.4f}".format(
            g["best_trained"]["readout"], g["best_trained"]["dim"],
            g["best_trained"]["ridge_mean"]))
        print("      best raw     : {:<16} dim {:>5}  {:+.4f}".format(
            g["best_raw"]["readout"], g["best_raw"]["dim"], g["best_raw"]["ridge_mean"]))
        print("      identity 96d : {:+.4f}   (beaten: {})".format(
            IDENTITY_RIDGE, g["beats_identity_96d"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
