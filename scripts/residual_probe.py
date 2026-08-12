#!/usr/bin/env python3
"""E7c: is the training lift on `vol_forecast_error` reconstruction in disguise?

E7 found a matched-width training lift of +0.0433 (+7.73 pooled SD) on
`vol_forecast_error` at the 128-dim `last_patch` readout. That target was built
so the ladder's own volatility forecast is divided out, which is what made the
lift interesting. But the construction only divides it out ARITHMETICALLY:

    vol_forecast_error = log(realised) - log(implied)

and `log(implied)` is a function of `implied_width`, which is derivable from the
input. E7 measured a +0.1392 training lift on `implied_width` at the same
readout. So an encoder that merely reconstructs the ladder better would predict
the `log(implied)` HALF of this target better, and score a higher R^2 without
having learned anything whatsoever about future volatility.

That is the same confound, in a new costume, that finding 10 caught in the
headline metric. It has to be ruled out before the lift means anything.

THE TEST. Remove what raw features already explain, then probe what is left:

    1. Fit ridge from a raw control to the target, ON TRAIN ONLY.
    2. Subtract its prediction from the target on both splits.
    3. Probe every arm against the residual.

Whatever raw features can compute from the input, including all of
`log(implied)`, is gone from the residual. An arm scoring above zero on it
carries structure about the target that the raw cross-section does not. A
TRAINING LIFT on it cannot be reconstruction, because reconstruction is exactly
what the raw control already did.

The residualising control is fit on train and applied to test, never fit on
test, so no information crosses the split.

SANITY CHECK, printed and asserted: the residualising control itself must score
approximately zero on its own residual. If it does not, the residualisation did
not work and nothing below means anything.

ALSO RUNS ON THE CROSS-ASSET CORPUS via `--corpus cross`, where the target is
`realised_corr` and the control subtracted is the both-ladder raw readout. E8
measured a +0.0819 (+4.54 pooled SD) matched-width lift there, on a target E8a
proved needs both assets. It failed this control too.

THE VERDICT NEEDS THREE CONDITIONS, AND IT TOOK TWO FAILURES TO FIND THEM ALL.

  1. A matched-width lift above 2 pooled SD. As first written this was the ONLY
     condition, and it fired on a lift running from -0.0173 to -0.0023: both
     endpoints negative R^2, so neither arm predicts the residual better than
     its own mean. Being less bad at a task neither model can do is not
     evidence of carrying information.

  2. The trained arm must reach a residual R^2 significantly above ZERO. Added
     after failure 1. Not sufficient either: on the cross-asset run conditions
     1 and 2 were satisfied by DIFFERENT ARMS, one clearing zero while a second,
     entirely negative one supplied the lift.

  3. That trained arm must also BEAT THE BEST UNTRAINED ARM on the residual.
     This is what actually decided it. On the cross-asset run the best untrained
     arm scored +0.0077 against the best trained arm's +0.0042, so the small
     surviving signal is the architecture behaving as a random projection, not
     anything training produced.

Both original verdicts are preserved as run, in `results/residual_probe.json`
and `results/residual_probe_cross.json`, with corrections beside them under
`verdict_review` rather than quietly overwritten. Counting E1's "at any
checkpoint" and E2's "the mechanism helps", four criteria in this programme have
now failed on their wording. That rate is itself a finding: a threshold stated
before seeing the data is still only as good as the failure modes its author
imagined, which is the argument for keeping the original next to the correction
instead of editing history.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from causaljepa import data as cj_data  # noqa: E402
from causaljepa import probes, targets as cj_targets  # noqa: E402
from causaljepa.model import CausalJEPA  # noqa: E402
from causaljepa.train import pick_device, train_arm  # noqa: E402

import readout_ablation as e4  # noqa: E402

SELF_R2_BAR = 0.02      # the residualising control must score below this on its own residual


def residualise(f_tr, f_te, y_tr, y_te):
    """Subtract a raw-feature ridge fit (train only) from the target.

    Returns (resid_tr, resid_te, r2_of_the_control). `ridge_probe` already
    standardises features and target internally and picks lambda on a held-out
    slice of train, so this reuses it rather than hand-rolling a second ridge
    with different conventions.
    """
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)
    # Refit here rather than reuse ridge_probe's internal weights, because we
    # need predictions on BOTH splits and it only returns test R^2.
    from causaljepa.probes import RIDGE_GRID, _standardise, _r2
    xtr, xte = _standardise(f_tr, f_te)
    ymu, ysd = y_tr.mean(0, keepdims=True), y_tr.std(0, keepdims=True) + 1e-6
    ys = (y_tr - ymu) / ysd

    n_val = max(1, int(len(xtr) * 0.2))
    xa, ya = xtr[:-n_val], ys[:-n_val]
    xv, yv = xtr[-n_val:], ys[-n_val:]
    xa_t = torch.from_numpy(xa).double()
    gram = xa_t.T @ xa_t
    rhs = xa_t.T @ torch.from_numpy(ya).double()
    eye = torch.eye(gram.shape[0], dtype=torch.float64)
    best_lam, best = RIDGE_GRID[0], -np.inf
    for lam in RIDGE_GRID:
        w = torch.linalg.solve(gram + lam * eye, rhs).numpy()
        sc = _r2(xv @ w, yv).mean()
        if sc > best:
            best_lam, best = lam, sc

    xall = torch.from_numpy(xtr).double()
    w = torch.linalg.solve(xall.T @ xall + best_lam * eye,
                           xall.T @ torch.from_numpy(ys).double()).numpy()
    pred_tr = (xtr @ w) * ysd + ymu
    pred_te = (xte @ w) * ysd + ymu
    return (y_tr - pred_tr), (y_te - pred_te), float(_r2(pred_te, y_te)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-jepa-root",
                    default=os.environ.get("PM_JEPA_ROOT", "../pm-jepa"))
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--corpus", default="single", choices=("single", "cross"),
                    help="'single' probes E7's one-asset targets; 'cross' probes "
                         "E8's BTC+ETH corpus, where the control to subtract is "
                         "the both-ladder raw readout")
    ap.add_argument("--target", default="vol_forecast_error")
    ap.add_argument("--control", default="raw_identity",
                    help="raw control whose prediction is subtracted out")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/residual_probe.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.corpus == "cross":
        from causaljepa import crossasset as ca
        import crossasset_headroom as e8a
        X, Y, owner, nn_, cstats = ca.load_corpus(
            args.pm_jepa_root, horizon=args.horizon)
        # Same four views E8 controlled against, flattened into one dict so the
        # residualising control can be named on the command line.
        raw = {"{}|{}".format(v, r): f
               for v, V in e8a.views(X).items()
               for r, f in e8a.raw_features(V).items()}
        if args.target == "vol_forecast_error":
            args.target = "realised_corr"
        if args.control == "raw_identity":
            args.control = "both|last_4min"
    else:
        X, _Yo, Y, owner, _on, nn_, cstats = cj_targets.load_corpus(
            args.pm_jepa_root, horizon=args.horizon)
        raw = e4.raw_controls(X)
    tr, te = cj_data.split_by_event(owner)
    ti = list(nn_).index(args.target)
    y = Y[:, ti].astype(np.float64)
    ctl = raw[args.control].astype(np.float64)
    r_tr, r_te, ctl_r2 = residualise(ctl[tr], ctl[te], y[tr], y[te])
    print("device {}  corpus {}  train {} / test {}".format(
        device, X.shape, tr.sum(), te.sum()), flush=True)
    print("target {}  residualised against {} (dim {}), which explains "
          "R^2 {:+.4f} of it".format(args.target, args.control,
                                     ctl.shape[1], ctl_r2), flush=True)

    names = [args.target + "_residual"]
    res = {"config": vars(args), "device": device, "corpus": cstats,
           "control_r2_on_target": ctl_r2, "self_r2_bar": SELF_R2_BAR,
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "runs": []}

    def probe(f):
        f = f.astype(np.float64)
        return probes.run_probes(f[tr], r_tr, f[te], r_te, names, device="cpu")

    print("\n--- raw controls on the residual ---", flush=True)
    for name, f in raw.items():
        t0 = time.time()
        r = probe(f)
        rec = {"model": "raw", "readout": name, "seed": None,
               "dim": int(f.shape[1]), "ridge": r["ridge"][names[0]],
               "mlp": r["mlp"][names[0]], "seconds": round(time.time() - t0, 1)}
        res["runs"].append(rec)
        print("  {:<9} {:<15} dim {:>5}  ridge {:+.4f}  mlp {:+.4f}".format(
            "raw", name, rec["dim"], rec["ridge"], rec["mlp"]), flush=True)

    self_r2 = next(r["ridge"] for r in res["runs"] if r["readout"] == args.control)
    res["self_r2"] = self_r2
    # ONE-SIDED ON PURPOSE. The failure this guards against is the control still
    # predicting its own residual, which would mean the subtraction did not
    # remove its information and nothing downstream is interpretable. A NEGATIVE
    # self-R^2 is the opposite of that failure: the control cannot predict the
    # residual even slightly. It shows up whenever the control is wide relative
    # to the sample (768 features against 2,654 training rows on the cross-asset
    # corpus gives -0.0341), because refitting on train then picks up noise that
    # does not generalise. Penalising it would abort a perfectly clean run; the
    # first version of this check was two-sided and did exactly that.
    res["self_r2_note"] = (
        "negative self-R^2 means the control cannot re-predict its own residual, "
        "which is the intended outcome; only a positive value invalidates the run")
    assert self_r2 < SELF_R2_BAR, (
        "RESIDUALISATION FAILED: the control {} still scores ridge R^2 {:+.4f} on "
        "its own residual, so its information was not removed. Nothing "
        "downstream is interpretable.".format(args.control, self_r2))

    for seed in range(args.seeds):
        print("\n--- seed {} ---".format(seed), flush=True)
        model, _h, _s = train_arm(
            X, tr, strategy="temporal", causal_context=False,
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
            for name, f in e4.extract(m, X, device).items():
                r = probe(f)
                rec = {"model": tag, "readout": name, "seed": seed,
                       "dim": int(f.shape[1]), "ridge": r["ridge"][names[0]],
                       "mlp": r["mlp"][names[0]]}
                res["runs"].append(rec)
                print("  {:<9} {:<15} dim {:>5}  ridge {:+.4f}  mlp {:+.4f}".format(
                    tag, name, rec["dim"], rec["ridge"], rec["mlp"]), flush=True)
        del model, untrained
        pathlib.Path(args.out).parent.mkdir(exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    agg = {}
    for r in res["runs"]:
        if r["seed"] is None:
            continue
        agg.setdefault((r["model"], r["readout"]), []).append(r)
    summary = {}
    for (tag, ro), rs in agg.items():
        a = np.array([x["ridge"] for x in rs])
        m = np.array([x["mlp"] for x in rs])
        summary["{}|{}".format(tag, ro)] = {
            "model": tag, "readout": ro, "dim": rs[0]["dim"], "seeds": len(rs),
            "ridge_mean": float(a.mean()), "ridge_std": float(a.std()),
            "mlp_mean": float(m.mean()), "mlp_std": float(m.std())}
    res["summary"] = summary

    lift = {}
    for v in summary.values():
        if v["model"] != "trained":
            continue
        u = summary.get("untrained|" + v["readout"])
        if not u:
            continue
        sd = (v["ridge_std"] ** 2 + u["ridge_std"] ** 2) ** 0.5
        lift[v["readout"]] = {
            "dim": v["dim"], "trained": v["ridge_mean"],
            "untrained": u["ridge_mean"],
            "lift": v["ridge_mean"] - u["ridge_mean"],
            "in_sd": ((v["ridge_mean"] - u["ridge_mean"]) / sd
                      if sd > 1e-12 else float("nan"))}
    res["training_lift_at_matched_readout"] = lift

    best = max(lift.values(), key=lambda v: v["in_sd"]) if lift else None
    lift_survives = bool(best and args.seeds >= 2 and np.isfinite(best["in_sd"])
                         and best["in_sd"] > 2.0 and best["lift"] > 0)

    # SECOND, NECESSARY CONDITION. The first run of this script reported
    # "NOT RECONSTRUCTION" on a lift running from -0.0173 to -0.0023: both
    # endpoints negative, so neither arm predicted the residual better than its
    # own mean, and being less bad at an impossible task proves nothing. A lift
    # is only evidence of carried information if the TRAINED arm actually
    # reaches a positive R^2. That original verdict is preserved as-run in
    # `results/residual_probe.json`, with the correction beside it under
    # `verdict_review`, per docs/RESEARCH_PLAN.md rule 1.
    bt = max((v for v in summary.values() if v["model"] == "trained"),
             key=lambda v: v["ridge_mean"], default=None)
    t_vs_zero, above_zero = float("nan"), False
    if bt and bt["seeds"] >= 2:
        se = bt["ridge_std"] / (bt["seeds"] ** 0.5)
        if se > 1e-12:
            t_vs_zero = bt["ridge_mean"] / se
            # Two-tailed t at p<0.05 for the small seed counts used here.
            t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(bt["seeds"], 2.776)
            above_zero = bool(t_vs_zero > t_crit)

    # THIRD CONDITION, and it is the one that decided the cross-asset run. The
    # criterion above can be satisfied by TWO DIFFERENT ARMS: one clearing zero
    # while a second, entirely negative, supplies the lift. On E8 it was exactly
    # that, and worse, the best UNTRAINED arm outscored every trained arm on the
    # residual, which means the small surviving signal is the architecture
    # acting as a random projection rather than anything training produced. A
    # trained arm must beat the untrained arms on the residual, not merely beat
    # zero.
    bu = max((v for v in summary.values() if v["model"] == "untrained"),
             key=lambda v: v["ridge_mean"], default=None)
    beats_untrained = bool(bt and bu and bt["ridge_mean"] > bu["ridge_mean"])
    res_beats = {"best_trained": bt["readout"] if bt else None,
                 "best_untrained": bu["readout"] if bu else None,
                 "trained_beats_untrained_on_residual": beats_untrained}
    survives = bool(lift_survives and above_zero and beats_untrained)
    res["verdict"] = {
        "decidable": args.seeds >= 2,
        "best_lift": best,
        "lift_above_untrained": lift_survives,
        "residual_ranking": res_beats,
        "best_trained_residual": (
            {"readout": bt["readout"], "dim": bt["dim"],
             "ridge_mean": bt["ridge_mean"], "ridge_std": bt["ridge_std"],
             "t_vs_zero": t_vs_zero, "significant_above_zero": above_zero}
            if bt else None),
        "survives_residualisation": survives,
        "statement": (
            "NOT RECONSTRUCTION: the matched-width training lift survives "
            "removal of everything the raw cross-section explains AND the "
            "trained arm reaches a residual R^2 above zero, so the encoder "
            "carries structure about {} that raw features do "
            "not.".format(args.target) if survives else
            "RECONSTRUCTION: the training lift does not survive. Best trained arm "
            "on the residual t = {:.2f} against zero, and trained beats untrained "
            "on the residual: {}. Whatever the lift measured, it is not "
            "information about {} beyond what the raw cross-section already "
            "carries.".format(t_vs_zero, beats_untrained, args.target))}

    w = 84
    print("\n" + "=" * w)
    print("E7c RESIDUAL PROBE  target {} minus what {} explains".format(
        args.target, args.control))
    print("=" * w)
    print("{:<11}{:<16}{:>6}{:>20}{:>20}".format("model", "readout", "dim", "ridge", "mlp"))
    print("-" * w)
    for v in sorted(summary.values(), key=lambda v: -v["ridge_mean"]):
        print("{:<11}{:<16}{:>6}{:>+14.4f}+-{:.4f}{:>+14.4f}+-{:.4f}".format(
            v["model"], v["readout"], v["dim"], v["ridge_mean"], v["ridge_std"],
            v["mlp_mean"], v["mlp_std"]))
    print("=" * w)
    print("\nMATCHED-WIDTH LIFT ON THE RESIDUAL")
    print("{:<16}{:>6}{:>12}{:>12}{:>12}{:>10}".format(
        "readout", "dim", "trained", "untrained", "lift", "in sd"))
    for ro, v in sorted(lift.items(), key=lambda kv: -kv[1]["in_sd"]):
        print("{:<16}{:>6}{:>+12.4f}{:>+12.4f}{:>+12.4f}{:>10.2f}".format(
            ro, v["dim"], v["trained"], v["untrained"], v["lift"], v["in_sd"]))
    print("=" * w)
    print("sanity: {} on its own residual: ridge {:+.4f} (bar {:.2f}) -> ok".format(
        args.control, self_r2, SELF_R2_BAR))
    print("VERDICT: {}".format(res["verdict"]["statement"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
