#!/usr/bin/env python3
"""E10: is staleness the mechanism? Demonstrate it where the objective works.

Finding 19 measured why the Kalshi corpus fails: roughly 70% of strike-minutes
carry no quote movement at all and 64% carry no trade. Finding 12 measured that
the same objective wins comfortably on the simulator. Between them sits an
inference nobody has tested, namely that the first fact causes the second.

That inference is the load-bearing claim of this entire programme and it is
currently correlational. We observed thin data, we observed failure, and we
reasoned backwards. This makes it causal, in the one environment where the
objective demonstrably works and every knob is ours.

THE INTERVENTION. `observe_stale` forks `sim.observe` and freezes quotes. With
probability `p`, independently per game, per bucket and per market, a cell
carries forward the previous bucket's ENTIRE observation vector rather than
taking the new one. Freezes compound: a cell unlucky several buckets running
holds one stale quote across all of them, which is what an untouched strike on a
real ladder does. The LATENT STATE IS UNTOUCHED. The game still evolves, the
fair prices still move; only the observation goes stale. That is precisely the
real situation and it isolates staleness from every other property of the data.

WHAT IS BEING CONTROLLED. Freezing removes information, so of course everything
degrades. The question is not whether the JEPA gets worse. It is whether the
JEPA's ADVANTAGE OVER RAW FEATURES gets worse, because that advantage is what
the whole programme was trying to obtain, and both arms see identically frozen
data at every level.

ANCHORING TO KALSHI, honestly. The corpus measurement is the fraction of
strike-minutes whose intra-minute mid range is exactly zero: 0.666, with the
other quote channels between 0.699 and 0.750. The simulator has no intra-minute
structure, so the comparable statistic here is the realised fraction of
consecutive buckets whose observation is unchanged, which this script measures
and reports beside `p`. The two are analogous rather than identical, so the
Kalshi figure is drawn as a BAND rather than a point, and no claim below depends
on the third decimal of where it lands.

Pre-registered gate:
  the JEPA advantage over the identity control falls monotonically with
  staleness and is at or below zero by the Kalshi band
      -> staleness is sufficient to explain the programme's negative result, and
         findings 1 to 19 have a single mechanism rather than nineteen
  the advantage survives at Kalshi-level staleness
      -> staleness is NOT the explanation, the sim-to-real gap is something else
         again, and finding 19's reading is wrong
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
SLATE = pathlib.Path(os.environ.get("SLATE_JEPA_ROOT",
                                   os.path.expanduser("~/slate-jepa"))) / "poc"
sys.path.insert(0, str(SLATE))

from slatejepa import baselines, probes, sim  # noqa: E402
from slatejepa.train import pick_device, represent_all, train_jepa  # noqa: E402

# Kalshi's measured staleness, findings 19. A band, not a point: mid_range is
# 0.666 zero, the other quote channels run 0.699 to 0.750.
KALSHI_BAND = (0.666, 0.750)


def freeze(obs: np.ndarray, p: float, rng) -> np.ndarray:
    """Carry the previous bucket's observation forward with probability `p`.

    Applied sequentially so that freezes COMPOUND: a cell frozen at buckets t
    and t+1 shows the bucket t-1 quote at both, which is what an untouched
    strike does. Vectorised over games and markets; the loop is over buckets
    only, and it has to be a loop because each step depends on the last.
    """
    if p <= 0.0:
        return obs
    out = obs.copy()
    n_g, n_b, k, _d = out.shape
    frozen = rng.random((n_g, n_b, k)) < p
    for t in range(1, n_b):
        m = frozen[:, t]
        out[:, t][m] = out[:, t - 1][m]
    return out


def unchanged_fraction(obs: np.ndarray) -> float:
    """Realised fraction of consecutive buckets whose observation is identical.

    The statistic that anchors `p` to the Kalshi measurement. Reported rather
    than assumed, because compounding makes the realised value differ from `p`.
    """
    d = np.abs(np.diff(obs, axis=1)).max(axis=-1)
    return float((d == 0.0).mean())


def make_dataset(n_games, window, per_game, seed, stale, noise_scale=1.0):
    """sim.make_dataset with quotes frozen after observation, latent untouched."""
    rng = np.random.default_rng(seed)
    latent = sim.simulate_games(n_games, rng)
    p_true = sim.true_prices(latent)
    obs_full = sim.observe(p_true, rng)
    stale_rng = np.random.default_rng(seed + 90210)
    obs_full = freeze(obs_full, stale, stale_rng)

    max_start = sim.N_BUCKETS_FULL + 1 - window
    starts = rng.integers(0, max_start, size=(n_games, per_game))
    g = np.repeat(np.arange(n_games), per_game)
    s = starts.reshape(-1)
    t_idx = s[:, None] + np.arange(window)[None, :]
    obs = obs_full[g[:, None], t_idx]
    end = t_idx[:, -1]
    tgt = np.stack([latent[n][g, end] for n in sim.LATENT_NAMES], axis=-1)
    return (obs.astype(np.float32), tgt.astype(np.float32),
            unchanged_fraction(obs_full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="0,0.2,0.4,0.55,0.7,0.85")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arm", default="slate")
    ap.add_argument("--games-train", type=int, default=1200)
    ap.add_argument("--games-test", type=int, default=300)
    ap.add_argument("--per-game", type=int, default=6)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/staleness_sweep.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    levels = [float(x) for x in args.stale.split(",") if x]
    names = list(sim.LATENT_NAMES)
    res = {"config": vars(args), "device": device, "kalshi_band": KALSHI_BAND,
           "sweep": []}
    print("device {}  latent targets {}".format(device, names), flush=True)
    print("Kalshi reference band for realised staleness: {:.3f} to {:.3f}\n".format(
        *KALSHI_BAND), flush=True)

    for p in levels:
        t0 = time.time()
        tr_x, tr_y, frac_tr = make_dataset(
            args.games_train, args.window, args.per_game, 7, p)
        te_x, te_y, _ = make_dataset(
            args.games_test, args.window, args.per_game, 99, p)

        f_tr = baselines.identity_features(tr_x)
        f_te = baselines.identity_features(te_x)
        ident = probes.run_probes(f_tr, tr_y, f_te, te_y, names, device="cpu")

        js = []
        for s in range(args.seeds):
            out = train_jepa(tr_x, args.arm, device, steps=args.steps,
                             batch=args.batch, d_model=args.d_model,
                             n_layers=args.layers, seed=s)
            js.append(probes.run_probes(
                represent_all(out["model"], tr_x, device), tr_y,
                represent_all(out["model"], te_x, device), te_y,
                names, device="cpu"))
            del out
        jm = float(np.mean([j["ridge_mean"] for j in js]))
        jsd = float(np.std([j["ridge_mean"] for j in js]))
        rec = {"p": p, "realised_unchanged": frac_tr,
               "identity_ridge": ident["ridge_mean"],
               "jepa_ridge_mean": jm, "jepa_ridge_std": jsd,
               "advantage": jm - ident["ridge_mean"], "seeds": args.seeds,
               "seconds": round(time.time() - t0, 1)}
        res["sweep"].append(rec)
        print("  p {:<5g} realised {:.3f}   identity {:+.4f}   jepa {:+.4f}+-{:.4f}"
              "   advantage {:+.4f}   ({:.0f}s)".format(
                  p, frac_tr, ident["ridge_mean"], jm, jsd, rec["advantage"],
                  rec["seconds"]), flush=True)
        pathlib.Path(args.out).parent.mkdir(exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        del tr_x, te_x

    sw = sorted(res["sweep"], key=lambda r: r["realised_unchanged"])
    adv = [r["advantage"] for r in sw]
    xs = [r["realised_unchanged"] for r in sw]

    cross = None
    for a, b in zip(sw, sw[1:]):
        if a["advantage"] > 0 >= b["advantage"]:
            span = a["advantage"] - b["advantage"]
            fr = a["advantage"] / span if span > 1e-12 else 0.0
            cross = (a["realised_unchanged"]
                     + fr * (b["realised_unchanged"] - a["realised_unchanged"]))
            break

    def at(x):
        return float(np.interp(x, xs, adv))
    band = (at(KALSHI_BAND[0]), at(KALSHI_BAND[1]))
    monotone = all(adv[i] >= adv[i + 1] - 1e-9 for i in range(len(adv) - 1))
    dead = bool(max(band) <= 0.0)

    res["analysis"] = {
        "advantage_at_zero_staleness": adv[0],
        "crossing_realised_staleness": cross,
        "advantage_across_kalshi_band": band,
        "monotone_decreasing": monotone,
        "dead_by_kalshi_band": dead,
        "verdict": (
            "STALENESS IS THE MECHANISM: the advantage falls {}monotonically and "
            "is at or below zero across the Kalshi band ({:+.4f} to {:+.4f}). "
            "Findings 1-19 have one cause, and it is measurable in the data "
            "before any model is trained.".format(
                "" if monotone else "non-", band[0], band[1])
            if dead else
            "STALENESS IS NOT SUFFICIENT: the advantage is still {:+.4f} to "
            "{:+.4f} across the Kalshi band, so freezing quotes to real levels "
            "does not reproduce the real failure. Finding 19's reading is wrong "
            "and the sim-to-real gap is something else.".format(*band))}

    w = 84
    print("\n" + "=" * w)
    print("E10 STALENESS SWEEP  ({} seeds, {} steps, arm {})".format(
        args.seeds, args.steps, args.arm))
    print("=" * w)
    print("{:>8}{:>12}{:>14}{:>20}{:>14}".format(
        "p", "realised", "identity", "jepa", "advantage"))
    print("-" * w)
    for r in sw:
        print("{:>8g}{:>12.3f}{:>+14.4f}{:>+14.4f}+-{:.4f}{:>+14.4f}".format(
            r["p"], r["realised_unchanged"], r["identity_ridge"],
            r["jepa_ridge_mean"], r["jepa_ridge_std"], r["advantage"]))
    print("=" * w)
    print("advantage at zero staleness      {:+.4f}".format(adv[0]))
    print("crosses zero at realised         {}".format(
        "{:.3f}".format(cross) if cross else "never in this range"))
    print("across the Kalshi band {:.3f}-{:.3f}   {:+.4f} to {:+.4f}".format(
        KALSHI_BAND[0], KALSHI_BAND[1], band[0], band[1]))
    print("monotone decreasing              {}".format(monotone))
    print("\nVERDICT: {}".format(res["analysis"]["verdict"]))

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
