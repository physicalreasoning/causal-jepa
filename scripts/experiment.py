#!/usr/bin/env python3
"""Driver for the causal 2x2: (causal_context) x (causal_target), N seeds each.

The four cells are not four unrelated arms; they are a factorial design and only
make sense read together:

    ctx=bidir, tgt=bidir     pm-jepa's setup, reproduced under our encoder
    ctx=bidir, tgt=causal    the interesting cell: fixes the target leak WITHOUT
                             changing what the predictor gets to see
    ctx=causal, tgt=bidir    causal context against a contaminated reference
    ctx=causal, tgt=causal   fully causal, the streamable configuration

Two tables come out. The first is probe R^2 against pm-jepa's controls, printed
with the controls interleaved rather than in a separate section, because the
result that matters is not "which arm won" but "did any arm clear the raw
cross-section at all". pm-jepa's best arm reached +0.377 against an identity
ridge of +0.449, i.e. every JEPA arm lost to the raw features. An arm here that
quietly lands at +0.30 must be visibly below the control line, not one page away
from it.

The second table is the causality diagnostics, which is what the experiment is
actually testing. H1 predicts target autocorrelation across the mask boundary
falls when the target encoder goes causal. H2 predicts copy_alignment falls and
copy_loss_ratio rises. Probe R^2 (H3) is the open question and is NOT tuned
towards; a negative result is reported as a negative result.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaljepa import probes  # noqa: E402
from causaljepa.data import load_corpus, split_by_event  # noqa: E402
from causaljepa.train import pick_device, represent_all, train_arm  # noqa: E402

# Fallbacks for pm-jepa/results/baselines.json, from SPEC section 0. Used only
# when that file cannot be read, so the control line is never missing from the
# table; a missing control is how an arm that loses to raw features gets read
# as a success.
CONTROL_FALLBACK = {
    "identity (control)": {"ridge_mean": 0.449, "mlp_mean": 0.505},
    "handcrafted (control)": {"ridge_mean": 0.460, "mlp_mean": 0.486},
}

# code -> (causal_context, causal_target). First letter is the CONTEXT encoder,
# second is the TARGET encoder; b = bidirectional, c = causal.
CELLS = {
    "bb": (False, False),
    "bc": (False, True),
    "cb": (True, False),
    "cc": (True, True),
}
CELL_ORDER = ("bb", "bc", "cb", "cc")


def cell_label(causal_context: bool, causal_target: bool) -> str:
    return "ctx={:<6} tgt={}".format("causal" if causal_context else "bidir",
                                     "causal" if causal_target else "bidir")


def load_controls(pm_jepa_root: pathlib.Path) -> dict:
    """pm-jepa's measured controls, read-only, with a hardcoded fallback."""
    p = pathlib.Path(pm_jepa_root) / "results" / "baselines.json"
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return dict(CONTROL_FALLBACK)
    out = {}
    for name in ("identity", "handcrafted"):
        if name in raw:
            out[name + " (control)"] = {
                "ridge_mean": float(raw[name]["ridge_mean"]),
                "mlp_mean": float(raw[name].get("mlp_mean", float("nan"))),
                "ridge": raw[name].get("ridge", {}),
            }
    return out or dict(CONTROL_FALLBACK)


def _fmt(v, spec="{:+.3f}", width=None, dash="-"):
    """Format a possibly-missing or NaN number without pretending it is zero."""
    if v is None or (isinstance(v, float) and v != v):
        s = dash
    else:
        s = spec.format(v)
    return s if width is None else "{:>{w}}".format(s, w=width)


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.std(vals)) if vals else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Causal 2x2 over (causal_context) x (causal_target).")
    ap.add_argument("--strategy", default="temporal",
                    help="masking strategy, one of causaljepa.masking.sample_mask")
    ap.add_argument("--cells", default="all",
                    help="comma list of cells to run from {} (context,target; "
                         "b=bidir c=causal), or 'all'".format(",".join(CELL_ORDER)))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.0,
                    help="inert; there is no regulariser, see train.jepa_loss")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-held-out", type=int, default=6)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="held constant across arms and seeds on purpose; "
                         "resplitting per seed would confound arm with split")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pm-jepa-root", default=str(pathlib.Path.home() / "pm-jepa"))
    ap.add_argument("--name", default=None,
                    help="results/<name>.json; defaults to causal2x2-<strategy>")
    ap.add_argument("--out", default=None, help="explicit output path, overrides --name")
    args = ap.parse_args(argv)

    if args.cells == "all":
        codes = list(CELL_ORDER)
    else:
        codes = [c.strip() for c in args.cells.split(",") if c.strip()]
        bad = [c for c in codes if c not in CELLS]
        if bad:
            ap.error("unknown cell code(s) {}; expected from {}".format(
                bad, ",".join(CELL_ORDER)))

    name = args.name or "causal2x2-{}".format(args.strategy)
    out_path = pathlib.Path(args.out) if args.out else ROOT / "results" / (name + ".json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    device = pick_device(args.device)
    X, Y, owner, names = load_corpus(args.pm_jepa_root, args.window, args.stride)
    tr, te = split_by_event(owner, frac=args.test_frac, seed=args.split_seed)
    controls = load_controls(pathlib.Path(args.pm_jepa_root))

    print("device {}   corpus {}   train {} / test {}".format(
        device, tuple(X.shape), int(tr.sum()), int(te.sum())), flush=True)
    print("strategy {}   cells {}   seeds {}   steps {}".format(
        args.strategy, ",".join(codes), args.seeds, args.steps), flush=True)

    res = {
        "config": vars(args),
        "device": device,
        "corpus": {"shape": list(X.shape), "n_train": int(tr.sum()),
                   "n_test": int(te.sum()), "target_names": list(names)},
        "controls": controls,
        "cells_run": codes,
        "arms": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for code in codes:
        cc, ct = CELLS[code]
        label = cell_label(cc, ct)
        runs = []
        for s in range(args.seeds):
            model, hist, secs = train_arm(
                X, tr, strategy=args.strategy, causal_context=cc,
                causal_target=ct, device=device, steps=args.steps,
                batch=args.batch, lr=args.lr, lam=args.lam,
                d_model=args.d_model, n_layers=args.layers,
                n_held_out=args.n_held_out, seed=s, log_every=args.log_every,
                verbose=(s == 0))
            f = represent_all(model, X, device)
            r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device="cpu")
            final = hist[-1] if hist else {}
            runs.append({"seed": s, "ridge": r["ridge"], "mlp": r["mlp"],
                         "ridge_mean": r["ridge_mean"], "mlp_mean": r["mlp_mean"],
                         "ridge_lambda": r["ridge_lambda"],
                         "seconds": round(secs, 1),
                         "diagnostics": final, "history": hist})
            print("   {}  seed {}  ridge {:+.3f}  mlp {:+.3f}  copy_align {}  ({:.0f}s)"
                  .format(label, s, r["ridge_mean"], r["mlp_mean"],
                          _fmt(final.get("copy_alignment")), secs), flush=True)
            del model

        diag_keys = sorted({k for r in runs for k in r["diagnostics"]} - {"step"})
        res["arms"][code] = {
            "code": code, "label": label,
            "causal_context": cc, "causal_target": ct,
            "runs": runs,
            "ridge_mean": _mean([r["ridge_mean"] for r in runs]),
            "ridge_std": _std([r["ridge_mean"] for r in runs]),
            "mlp_mean": _mean([r["mlp_mean"] for r in runs]),
            "mlp_std": _std([r["mlp_mean"] for r in runs]),
            "per_target_mean": {n: _mean([r["ridge"][n] for r in runs]) for n in names},
            "per_target_std": {n: _std([r["ridge"][n] for r in runs]) for n in names},
            "diagnostics_mean": {k: _mean([r["diagnostics"].get(k) for r in runs])
                                 for k in diag_keys},
            "diagnostics_std": {k: _std([r["diagnostics"].get(k) for r in runs])
                                for k in diag_keys},
        }

    _print_probe_table(res, controls, names, args)
    _print_diagnostic_table(res)
    _print_hypothesis_readout(res)

    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print("wrote {}".format(out_path))
    return res


def _print_probe_table(res, controls, names, args):
    width = 22 + 16 * len(names) + 20
    print("\n" + "=" * width)
    print("FROZEN-ENCODER PROBE R^2 vs pm-jepa CONTROLS  ({}, {} seeds)".format(
        args.strategy, args.seeds))
    print("=" * width)
    hdr = "{:<22}".format("arm") + "".join("{:>16}".format(n[:15]) for n in names)
    print(hdr + "{:>11}{:>9}".format("mean", "sd"))
    print("-" * width)

    ident = None
    for label, c in controls.items():
        if "identity" in label:
            ident = c["ridge_mean"]
        row = "{:<22}".format(label)
        per = c.get("ridge", {})
        row += "".join(_fmt(per.get(n), width=16) for n in names)
        row += _fmt(c["ridge_mean"], width=11) + "{:>9}".format("-")
        print(row)
    print("-" * width)

    for code in CELL_ORDER:
        arm = res["arms"].get(code)
        if arm is None:
            continue
        row = "{:<22}".format(arm["label"])
        row += "".join(_fmt(arm["per_target_mean"][n], width=16) for n in names)
        row += _fmt(arm["ridge_mean"], width=11)
        row += _fmt(arm["ridge_std"], "{:.3f}", width=9)
        print(row)
    print("=" * width)

    if ident is not None and res["arms"]:
        best_code = max(res["arms"], key=lambda c: res["arms"][c]["ridge_mean"])
        best = res["arms"][best_code]
        gap = best["ridge_mean"] - ident
        sd = best["ridge_std"]
        verdict = "CLEARS" if gap > 0 else "LOSES TO"
        # A single seed has zero spread, and dividing by it manufactures a
        # six-figure "sd" that reads as overwhelming evidence from one run.
        # Say the seed count instead.
        in_sd = ("{:.1f} sd".format(gap / sd) if sd == sd and sd > 1e-4
                 else "seed spread too small to quote, {} seed(s)".format(args.seeds))
        print("best arm ({}) {} the identity control by {:+.3f} ({})".format(
            best["label"].strip(), verdict, gap, in_sd))


def _print_diagnostic_table(res):
    cols = [("copy_alignment", "copy_align", "{:+.3f}"),
            ("copy_loss_ratio", "copy_ratio", "{:.3f}"),
            ("extrap_loss_ratio", "extrap_rat", "{:.3f}"),
            ("interp_loss_ratio", "interp_rat", "{:.3f}"),
            ("interp_advantage", "interp_adv", "{:+.4f}"),
            ("lag_1", "autoc_lag1", "{:+.3f}"),
            ("cross_boundary", "autoc_xbnd", "{:+.3f}"),
            ("eff_rank", "eff_rank", "{:.1f}"),
            ("reg_grad_norm", "reg_gnorm", "{:.2e}")]
    width = 22 + 12 * len(cols)
    print("\n" + "=" * width)
    print("CAUSALITY DIAGNOSTICS AT FINAL STEP  (seed mean)")
    print("=" * width)
    print("{:<22}".format("arm") + "".join("{:>12}".format(c[1]) for c in cols))
    print("-" * width)
    for code in CELL_ORDER:
        arm = res["arms"].get(code)
        if arm is None:
            continue
        row = "{:<22}".format(arm["label"])
        row += "".join(_fmt(arm["diagnostics_mean"].get(k), f, width=12)
                       for k, _, f in cols)
        print(row)
    print("=" * width)


def _print_hypothesis_readout(res):
    """Report H1/H2 as a signed contrast, not as a pass/fail badge.

    The contrast is over the TARGET encoder flag, averaged over the context
    flag, because that is the factor the hypotheses are about. Printing the
    predicted sign next to the observed one keeps a result that came out the
    wrong way visible instead of buried in the table above.
    """
    bidir = [a for a in res["arms"].values() if not a["causal_target"]]
    causal = [a for a in res["arms"].values() if a["causal_target"]]
    if not bidir or not causal:
        print("\n(H1/H2 contrast needs both target arms; skipped)")
        return

    def avg(group, key):
        return _mean([a["diagnostics_mean"].get(key) for a in group])

    rows = [("H1 target_autocorr cross_boundary", "cross_boundary", "lower"),
            ("H1 target_autocorr lag_1", "lag_1", "lower"),
            ("H2 copy_alignment", "copy_alignment", "lower"),
            ("H2 copy_loss_ratio", "copy_loss_ratio", "higher"),
            ("H3 probe ridge_mean", None, "open")]
    print("\n" + "=" * 78)
    print("TARGET-ENCODER CONTRAST  (causal target minus bidirectional target)")
    print("=" * 78)
    print("{:<34}{:>11}{:>11}{:>11}{:>11}".format(
        "quantity", "bidir", "causal", "delta", "predicted"))
    print("-" * 78)
    for title, key, direction in rows:
        if key is None:
            b = _mean([a["ridge_mean"] for a in bidir])
            c = _mean([a["ridge_mean"] for a in causal])
        else:
            b, c = avg(bidir, key), avg(causal, key)
        delta = c - b
        print("{:<34}{}{}{}{:>11}".format(
            title, _fmt(b, width=11), _fmt(c, width=11),
            _fmt(delta, "{:+.4f}", width=11), direction))
    print("=" * 78)
    print("H3 is an open question by design; do not tune towards a causal win.")


if __name__ == "__main__":
    main()
