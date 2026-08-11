#!/usr/bin/env python3
"""Benchmark table: every arm against the controls that actually matter.

The control that matters most is the UNTRAINED encoder. A JEPA arm that cannot
beat its own random initialisation has learned nothing, and no amount of
favourable diagnostics changes that. slate-jepa measured random-encoder at
+0.765 against jepa-temporal +0.901 on synthetic data, so the gap was real
there; whether it survives on the real corpus is the question.

The identity and handcrafted controls come from pm-jepa/results/baselines.json
and are the raw cross-section: what you get with no model at all.

pm-jepa's own trained arms are read from its results/seeds.json so the table
shows this work next to the prior work at the same probe protocol.
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
from causaljepa.train import pick_device, represent_all  # noqa: E402


def random_encoder_arm(X, tr, te, Y, names, *, causal, device, seed,
                       d_model=128, n_layers=4):
    """Probe an UNTRAINED encoder. No optimiser, no steps, no EMA.

    This is the floor. Random projections of a structured input are a
    surprisingly strong baseline, which is exactly why it belongs in the table.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, T, K, C = X.shape
    model = CausalJEPA(T, K, C, d_model=d_model, n_layers=n_layers,
                       causal_context=causal, causal_target=causal).to(device)
    model.eval()
    t0 = time.time()
    f = represent_all(model, X, device)
    r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
    r["seconds"] = round(time.time() - t0, 1)
    del model
    return r


def pm_jepa_reference(root):
    """pm-jepa's trained arms, straight from its own seeds.json. Read-only."""
    p = pathlib.Path(root) / "results" / "seeds.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}

    def walk(node, trail=()):
        if isinstance(node, dict):
            rm = node.get("ridge_mean")
            if isinstance(rm, (int, float)) and trail:
                out[trail[-1]] = {
                    "ridge_mean": float(rm),
                    "ridge_std": float(node.get("ridge_std") or 0.0),
                    "mlp_mean": float(node.get("mlp_mean") or float("nan")),
                    "mlp_std": float(node.get("mlp_std") or 0.0),
                }
            for k, v in node.items():
                walk(v, trail + (k,))

    walk(raw)
    return {k: v for k, v in out.items() if k.startswith("jepa-")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"),
                    help="checkout of the pm-jepa corpus repo; "
                         "defaults to $PM_JEPA_ROOT then ../pm-jepa")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, Y, owner, names = cj_data.load_corpus(args.pm_jepa_root)
    tr, te = cj_data.split_by_event(owner)
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)

    res = {"config": vars(args), "device": device,
           "corpus": {"shape": list(X.shape), "n_train": int(tr.sum()),
                      "n_test": int(te.sum()), "target_names": list(names)},
           "arms": {}}

    base_p = pathlib.Path(args.pm_jepa_root) / "results" / "baselines.json"
    if base_p.exists():
        for k, v in json.loads(base_p.read_text()).items():
            res["arms"][k + " (control)"] = {
                "ridge_mean": v["ridge_mean"], "ridge_std": 0.0,
                "mlp_mean": v["mlp_mean"], "mlp_std": 0.0,
                "source": "pm-jepa/results/baselines.json", "seeds": 1}

    for causal in (False, True):
        label = "random-encoder ({})".format("causal" if causal else "bidir")
        runs = []
        for s in range(args.seeds):
            r = random_encoder_arm(X, tr, te, Y, names, causal=causal,
                                   device=device, seed=s,
                                   d_model=args.d_model, n_layers=args.layers)
            runs.append(r)
            print("  {}  seed {}  ridge {:+.4f}  mlp {:+.4f}  ({:.0f}s)".format(
                label, s, r["ridge_mean"], r["mlp_mean"], r["seconds"]), flush=True)
        rm = np.array([r["ridge_mean"] for r in runs])
        mm = np.array([r["mlp_mean"] for r in runs])
        res["arms"][label] = {
            "ridge_mean": float(rm.mean()), "ridge_std": float(rm.std()),
            "mlp_mean": float(mm.mean()), "mlp_std": float(mm.std()),
            "seeds": args.seeds, "trained": False,
            "per_target_mean": {n: float(np.mean([r["ridge"][n] for r in runs]))
                                for n in names},
            "runs": runs}

    for k, v in pm_jepa_reference(args.pm_jepa_root).items():
        v["source"] = "pm-jepa/results/seeds.json"
        res["arms"][k + " (pm-jepa)"] = v

    for fn, tag in (("causal_ablation", "temporal"), ("interpolation_arm", "interp")):
        p = pathlib.Path("results") / (fn + ".json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for code, arm in d["arms"].items():
            res["arms"]["{} [{}] (this work)".format(arm["label"].strip(), tag)] = {
                "ridge_mean": arm["ridge_mean"], "ridge_std": arm["ridge_std"],
                "mlp_mean": arm["mlp_mean"], "mlp_std": arm["mlp_std"],
                "seeds": d["config"]["seeds"], "source": str(p)}

    print("\n" + "=" * 78)
    print("{:<40}{:>18}{:>18}".format("arm", "ridge R^2", "mlp R^2"))
    print("=" * 78)
    for k, v in sorted(res["arms"].items(), key=lambda kv: -kv[1]["ridge_mean"]):
        print("{:<40}{:>+12.4f}+-{:.4f}{:>+12.4f}+-{:.4f}".format(
            k[:39], v["ridge_mean"], v.get("ridge_std", 0.0),
            v["mlp_mean"], v.get("mlp_std", 0.0)))
    print("=" * 78)

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
