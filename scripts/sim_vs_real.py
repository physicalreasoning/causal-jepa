#!/usr/bin/env python3
"""E3: why does the simulator work and the market not?

On slate-jepa's simulator the JEPA beat every control (+0.9014 ridge against an
identity control at +0.8142). On real Kalshi ladders the ordering inverts and
every arm loses. Nobody has measured why, and without that the whole programme's
negative result is "it did not work here" rather than an explanation.

Two measurements, because one alone would not settle it.

PART A, noise sweep. The simulator's observation noise is
`noise_sd = 0.006 / sqrt(liq)` inside `sim.observe`. Scaling it degrades the
observations without touching the latent state, so it isolates signal-to-noise
from everything else. Sweep it and find where the JEPA advantage over the
identity control crosses zero.

PART B, cross-sectional redundancy. My hypothesis for the structural difference:
the simulator's markets are HETEROGENEOUS (a moneyline, four spread lines, three
totals) encoding a 3-dimensional latent that no single market identifies, which
is exactly what forces a model to learn cross-market structure. A Kalshi strike
ladder is 24 strikes that are a smooth, near-deterministic function of two
numbers, spot and width, so one strike is nearly recoverable from its neighbours
by interpolation and there is very little for a model to infer that raw features
do not already carry.

`sibling_interpolation_gap` (ported from pm-jepa/model/diagnostics.py) measures
exactly that: the R^2 of predicting one market's log-odds from all the others in
the same bucket. It is computable on both datasets and needs no training, so it
is the honest common axis. Prediction registered before running: real Kalshi
should sit far higher than the simulator.

NOTE ON PROVENANCE. `observe_scaled` below is a fork of `sim.observe` with one
added parameter. slate-jepa is read-only for this repo (RESEARCH_PLAN.md rule 7),
and the noise level is an inline literal there with no way to reach it, so the
function is copied rather than imported. Everything else, the latent process, the
pricing, the model, the training loop and the probes, is imported from slate-jepa
so the numbers stay comparable to `poc/results/poc.json`.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SLATE_POC = pathlib.Path("/Users/nikita/slate-jepa/poc")
sys.path.insert(0, str(SLATE_POC))

from slatejepa import baselines, probes, sim  # noqa: E402
from slatejepa.train import pick_device, represent_all, train_jepa  # noqa: E402

from causaljepa import data as cj_data  # noqa: E402


# ---------------------------------------------------------------- part A

def observe_scaled(p_true, rng, noise_scale=1.0):
    """Fork of sim.observe with a tunable quote-noise multiplier.

    Only `noise_sd` is parameterised. Spread, depth, imbalance and flow are left
    exactly as slate-jepa generates them, so `noise_scale=1.0` reproduces the
    original dataset bit for bit given the same rng stream.
    """
    n_g, n_b, k = p_true.shape
    liq = np.ones(k)
    liq[0] = 1.0
    liq[1:1 + len(sim.SPREAD_LINES)] = [0.35, 0.7, 0.7, 0.35]
    liq[1 + len(sim.SPREAD_LINES):] = [0.5, 0.8, 0.5]
    liq = liq[None, None, :]

    noise_sd = (0.006 * noise_scale) / np.sqrt(liq)          # <-- the only change
    mid = np.clip(p_true + rng.normal(0.0, 1.0, p_true.shape) * noise_sd, 0.01, 0.99)

    edge = np.minimum(mid, 1.0 - mid)
    half = sim.TICK * (0.5 + 1.2 / liq) * (1.0 + 0.8 * np.exp(-edge / 0.06))
    half = np.clip(half, sim.TICK * 0.5, 0.06)

    lvl = np.arange(sim.N_LEVELS)[None, None, None, :]
    bid_off = half[..., None] + lvl * sim.TICK
    ask_off = half[..., None] + lvl * sim.TICK

    depth_scale = liq[..., None] * 2000.0
    bid_sz = rng.lognormal(0.0, 0.6, (n_g, n_b, k, sim.N_LEVELS)) * depth_scale * (1.0 - 0.12 * lvl)
    ask_sz = rng.lognormal(0.0, 0.6, (n_g, n_b, k, sim.N_LEVELS)) * depth_scale * (1.0 - 0.12 * lvl)

    imb = (bid_sz.sum(-1) - ask_sz.sum(-1)) / (bid_sz.sum(-1) + ask_sz.sum(-1) + 1e-9)
    dmid = np.diff(mid, axis=1, prepend=mid[:, :1, :])
    n_trades = rng.poisson(np.clip(6.0 * liq, 0.5, None) * np.ones_like(mid))
    signed_vol = dmid * depth_scale[..., 0] * 3.0 + rng.normal(0.0, 60.0, mid.shape)

    feats = [
        sim._logit(mid)[..., None], (half / sim.TICK)[..., None],
        np.log1p(bid_sz), np.log1p(ask_sz), bid_off / sim.TICK, ask_off / sim.TICK,
        imb[..., None], np.log1p(n_trades)[..., None], (signed_vol / 1000.0)[..., None],
    ]
    return np.concatenate(feats, axis=-1).astype(np.float32)


def make_dataset_scaled(n_games, window, windows_per_game, seed, noise_scale):
    """sim.make_dataset with a noise multiplier. Same rng consumption order."""
    rng = np.random.default_rng(seed)
    latent = sim.simulate_games(n_games, rng)
    p = sim.true_prices(latent)
    obs_full = observe_scaled(p, rng, noise_scale)

    max_start = sim.N_BUCKETS_FULL + 1 - window
    starts = rng.integers(0, max_start, size=(n_games, windows_per_game))
    g_idx = np.repeat(np.arange(n_games), windows_per_game)
    s_idx = starts.reshape(-1)
    t_idx = s_idx[:, None] + np.arange(window)[None, :]
    obs = obs_full[g_idx[:, None], t_idx]
    end = t_idx[:, -1]
    tgt = np.stack([latent[n][g_idx, end] for n in sim.LATENT_NAMES], axis=-1)
    return obs.astype(np.float32), tgt.astype(np.float32)


# ---------------------------------------------------------------- part B

def sibling_r2(logodds: np.ndarray, ridge: float = 1.0) -> dict:
    """R^2 of predicting each column from all the others, same row.

    logodds: (N, K). Ported from pm-jepa/model/diagnostics.py::
    sibling_interpolation_gap, generalised to report every market rather than
    one. High values mean the cross-section is redundant: a held-out market is
    recoverable by interpolating its siblings, so there is little latent
    structure a model could add on top of raw features.
    """
    x = torch.from_numpy(np.ascontiguousarray(logodds)).double()
    n, k = x.shape
    out = []
    for j in range(k):
        y = x[:, j:j + 1]
        others = torch.cat([x[:, :j], x[:, j + 1:]], dim=1)
        others = torch.cat([others, torch.ones(n, 1, dtype=others.dtype)], dim=1)
        gram = others.T @ others + ridge * torch.eye(others.shape[1], dtype=others.dtype)
        w = torch.linalg.solve(gram, others.T @ y)
        resid = ((y - others @ w) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        out.append(float(1.0 - resid / total.clamp(min=1e-12)))
    a = np.array(out)
    return {"per_market": out, "mean": float(a.mean()),
            "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "k": int(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-scales", default="0.5,1,2,4,8,16")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arm", default="slate")
    ap.add_argument("--games-train", type=int, default=1200)
    ap.add_argument("--games-test", type=int, default=300)
    ap.add_argument("--windows-per-game", type=int, default=6)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pm-jepa-root", default="/Users/nikita/pm-jepa")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/sim_vs_real.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    scales = [float(s) for s in args.noise_scales.split(",") if s]
    names = list(sim.LATENT_NAMES)
    res = {"config": vars(args), "device": device, "sweep": [], "redundancy": {}}

    # ---------------- Part B first: it is cheap and it is the explanation ----
    print("=" * 78)
    print("PART B: cross-sectional redundancy (no training, both datasets)")
    print("=" * 78)

    obs_s, _ = make_dataset_scaled(400, args.window, 4, 12345, 1.0)
    sim_lo = obs_s[:, -1, :, 0]                       # (N, K) logit mid, last bucket
    r_sim = sibling_r2(sim_lo)
    res["redundancy"]["simulator"] = r_sim
    print("  simulator   K={:>3}  mean R^2 {:.4f}  median {:.4f}  min {:.4f}".format(
        r_sim["k"], r_sim["mean"], r_sim["median"], r_sim["min"]), flush=True)

    X, Y, owner, real_names = cj_data.load_corpus(args.pm_jepa_root)
    real_lo = X[:, -1, :, 0].astype(np.float64)       # (N, 24) log-odds mid
    r_real = sibling_r2(real_lo)
    res["redundancy"]["kalshi"] = r_real
    print("  kalshi      K={:>3}  mean R^2 {:.4f}  median {:.4f}  min {:.4f}".format(
        r_real["k"], r_real["mean"], r_real["median"], r_real["min"]), flush=True)
    res["redundancy"]["gap"] = r_real["mean"] - r_sim["mean"]
    print("  gap (kalshi - simulator): {:+.4f}".format(res["redundancy"]["gap"]), flush=True)

    # ---------------- Part A: the noise sweep -------------------------------
    print("\n" + "=" * 78)
    print("PART A: JEPA advantage over identity, versus observation noise")
    print("=" * 78)
    for ns in scales:
        obs_tr, y_tr = make_dataset_scaled(
            args.games_train, args.window, args.windows_per_game, 7, ns)
        obs_te, y_te = make_dataset_scaled(
            args.games_test, args.window, args.windows_per_game, 99, ns)

        f_tr = baselines.identity_features(obs_tr)
        f_te = baselines.identity_features(obs_te)
        ident = probes.run_probes(f_tr, y_tr, f_te, y_te, names, device="cpu")

        js = []
        for s in range(args.seeds):
            out = train_jepa(obs_tr, args.arm, device, steps=args.steps,
                             batch=args.batch, d_model=args.d_model,
                             n_layers=args.layers, seed=s)
            jr = probes.run_probes(
                represent_all(out["model"], obs_tr, device),
                y_tr,
                represent_all(out["model"], obs_te, device),
                y_te, names, device="cpu")
            js.append(jr)
            del out
        jm = float(np.mean([j["ridge_mean"] for j in js]))
        jsd = float(np.std([j["ridge_mean"] for j in js]))
        rec = {"noise_scale": ns,
               "identity_ridge": ident["ridge_mean"], "identity_mlp": ident["mlp_mean"],
               "jepa_ridge_mean": jm, "jepa_ridge_std": jsd,
               "jepa_mlp_mean": float(np.mean([j["mlp_mean"] for j in js])),
               "advantage": jm - ident["ridge_mean"], "seeds": args.seeds}
        res["sweep"].append(rec)
        print("  noise x{:<5g}  identity {:+.4f}   jepa {:+.4f}+-{:.4f}   "
              "advantage {:+.4f}".format(ns, ident["ridge_mean"], jm, jsd,
                                         rec["advantage"]), flush=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        del obs_tr, obs_te

    # ---------------- crossing point and verdict ----------------------------
    sw = sorted(res["sweep"], key=lambda r: r["noise_scale"])
    cross = None
    for a, b in zip(sw, sw[1:]):
        if a["advantage"] > 0 >= b["advantage"]:
            span = a["advantage"] - b["advantage"]
            frac = a["advantage"] / span if span > 1e-12 else 0.0
            cross = a["noise_scale"] + frac * (b["noise_scale"] - a["noise_scale"])
            break
    res["crossing_noise_scale"] = cross

    res["gate"] = {
        "criterion": ("does observation noise alone explain the sim-to-real gap, and "
                      "is the Kalshi cross-section more redundant than the simulator's"),
        "advantage_at_baseline_noise": sw[0]["advantage"] if sw else None,
        "crossing_noise_scale": cross,
        "redundancy_simulator": r_sim["mean"],
        "redundancy_kalshi": r_real["mean"],
        "kalshi_more_redundant": bool(r_real["mean"] > r_sim["mean"]),
    }

    print("\n" + "=" * 78)
    print("{:>10}{:>14}{:>18}{:>14}".format("noise", "identity", "jepa", "advantage"))
    print("-" * 78)
    for r in sw:
        print("{:>10g}{:>+14.4f}{:>+12.4f}+-{:.4f}{:>+14.4f}".format(
            r["noise_scale"], r["identity_ridge"], r["jepa_ridge_mean"],
            r["jepa_ridge_std"], r["advantage"]))
    print("=" * 78)
    print("JEPA advantage crosses zero at noise x{}".format(
        "{:.2f}".format(cross) if cross else "never in this range"))
    print("cross-sectional redundancy: simulator {:.4f}  kalshi {:.4f}  gap {:+.4f}".format(
        r_sim["mean"], r_real["mean"], r_real["mean"] - r_sim["mean"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
