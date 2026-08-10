#!/usr/bin/env python3
"""Read the finished results files and print the H1-H4 readout.

Kept separate from `experiment.py` so the hypotheses can be re-read against the
saved numbers without re-running a single training step. Everything printed here
is recomputed from `results/*.json`; nothing is retyped.

Two reporting rules are enforced in code rather than left to the writer:

  1. Every arm figure is mean +/- std across seeds. A bare mean hides that the
     H1 contrast at this step count is smaller than the seed spread, which is the
     single most important caveat on the whole experiment.
  2. `cross_boundary` and `lag_L` are reported alongside `baseline_cos`, the
     anisotropy floor. A cosine of 0.95 against a floor of 0.94 is not evidence
     of leakage, and quoting the level without the floor is how it would be
     mistaken for evidence.
"""
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CELL_ORDER = ("bb", "bc", "cb", "cc")


def _f(v, spec="{:+.3f}", w=None, dash="-"):
    s = dash if v is None or (isinstance(v, float) and v != v) else spec.format(v)
    return s if w is None else "{:>{w}}".format(s, w=w)


def _pm(m, s, spec="{:+.3f}", w=16):
    if m is None or m != m:
        return "{:>{w}}".format("-", w=w)
    return "{:>{w}}".format(spec.format(m) + " +/-" + "{:.3f}".format(s), w=w)


def seed_vals(arm, key):
    return [r["diagnostics"].get(key) for r in arm["runs"]]


def welch(a, b):
    """Welch t on the seed means. Small n, so this is a sanity gauge, not a p."""
    a = [x for x in a if x is not None and x == x]
    b = [x for x in b if x is not None and x == x]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    den = math.sqrt(va / len(a) + vb / len(b))
    if den <= 0:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / den


def probe_table(res, title):
    names = res["corpus"]["target_names"]
    ctrl = res["controls"]
    w = 24 + 16 * len(names) + 20
    print("\n" + "=" * w)
    print(title)
    print("=" * w)
    print("{:<24}".format("arm") + "".join("{:>16}".format(n[:15]) for n in names)
          + "{:>20}".format("ridge mean +/- sd"))
    print("-" * w)
    for label, c in ctrl.items():
        per = c.get("ridge", {})
        print("{:<24}".format(label)
              + "".join(_f(per.get(n), w=16) for n in names)
              + "{:>20}".format(_f(c["ridge_mean"])))
    print("-" * w)
    for code in CELL_ORDER:
        a = res["arms"].get(code)
        if a is None:
            continue
        print("{:<24}".format(a["label"])
              + "".join(_f(a["per_target_mean"][n], w=16) for n in names)
              + "{:>20}".format("{:+.3f} +/- {:.3f}".format(a["ridge_mean"], a["ridge_std"])))
    print("=" * w)


def diag_table(res, title):
    cols = [("copy_alignment", "copy_align", "{:+.3f}"),
            ("copy_loss_ratio", "copy_ratio", "{:.3f}"),
            ("extrap_loss_ratio", "extrap_rat", "{:.3f}"),
            ("interp_loss_ratio", "interp_rat", "{:.3f}"),
            ("interp_advantage", "interp_adv", "{:+.4f}"),
            ("cross_boundary", "autoc_xbnd", "{:+.4f}"),
            ("lag_1", "autoc_lag1", "{:+.4f}"),
            ("baseline_cos", "floor_cos", "{:+.4f}"),
            ("eff_rank", "eff_rank", "{:.1f}")]
    w = 24 + 17 * len(cols)
    print("\n" + "=" * w)
    print(title)
    print("=" * w)
    print("{:<24}".format("arm") + "".join("{:>17}".format(c[1]) for c in cols))
    print("-" * w)
    for code in CELL_ORDER:
        a = res["arms"].get(code)
        if a is None:
            continue
        row = "{:<24}".format(a["label"])
        for k, _, spec in cols:
            row += _pm(a["diagnostics_mean"].get(k), a["diagnostics_std"].get(k),
                       spec, w=17)
        print(row)
    print("=" * w)


def contrast(res, title):
    """causal target minus bidirectional target, paired within context flag.

    Pairing matters: (bb vs bc) and (cb vs cc) each hold the context encoder
    fixed, so the difference is attributable to the target flag alone. Averaging
    all four cells into two groups would let a context effect ride along.
    """
    rows = [("cross_boundary", "H1 autocorr cross_boundary", "lower"),
            ("lag_1", "H1 autocorr lag_1", "lower"),
            ("lag_3", "H1 autocorr lag_3", "lower"),
            ("baseline_cos", "   (floor: baseline_cos)", "n/a"),
            ("boundary_excess", "   boundary_excess", "~0 if mask-blind"),
            ("copy_alignment", "H2 copy_alignment", "lower"),
            ("copy_loss_ratio", "H2 copy_loss_ratio", "higher"),
            ("interp_advantage", "H2 interp_advantage", "lower"),
            (None, "H3 probe ridge_mean", "open")]
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print("{:<30}{:>18}{:>18}{:>14}{:>10}{:>16}".format(
        "quantity", "tgt=bidir", "tgt=causal", "delta", "welch t", "predicted"))
    print("-" * 104)
    for key, label, pred in rows:
        bv, cv = [], []
        for bcode, ccode in (("bb", "bc"), ("cb", "cc")):
            b, c = res["arms"].get(bcode), res["arms"].get(ccode)
            if not b or not c:
                continue
            if key is None:
                bv += [r["ridge_mean"] for r in b["runs"]]
                cv += [r["ridge_mean"] for r in c["runs"]]
            else:
                bv += seed_vals(b, key)
                cv += seed_vals(c, key)
        bv = [x for x in bv if x is not None and x == x]
        cv = [x for x in cv if x is not None and x == x]
        if not bv or not cv:
            print("{:<30}{:>18}{:>18}{:>14}{:>10}{:>16}".format(
                label, "-", "-", "-", "-", pred))
            continue
        bm, bs = float(np.mean(bv)), float(np.std(bv))
        cm, cs = float(np.mean(cv)), float(np.std(cv))
        print("{:<30}{:>18}{:>18}{:>14}{:>10}{:>16}".format(
            label,
            "{:+.4f} +/-{:.4f}".format(bm, bs),
            "{:+.4f} +/-{:.4f}".format(cm, cs),
            "{:+.4f}".format(cm - bm),
            _f(welch(cv, bv), "{:+.2f}"),
            pred))
    print("=" * 104)


def derived(rec):
    """Quantities the history does not store but H1 and H2 cannot be read without.

    `*_over_floor`: transformer embeddings are anisotropic and sit at high cosine
    to everything, so a target autocorrelation of 0.94 is meaningless until the
    floor is subtracted. `baseline_cos` is that floor, the cosine between the
    same (patch, strike) slot in two different windows. If the causal arm lowers
    the raw autocorrelation and the floor by the same amount, it has changed the
    embedding geometry globally and has removed no leakage at all; only the
    excess over the floor is the H1 quantity. `diagnostics.py` says this
    explicitly and the raw contrast alone would let it be missed.

    `copy_loss_ratio` is model loss over oracle loss, and a fall can mean the
    model got better or the oracle got worse. H2 predicts a rise, so the sign
    matters and the two must be read apart; both terms are already recorded.
    """
    out = dict(rec)
    floor = rec.get("baseline_cos")
    if floor is not None and floor == floor:
        for k in ("cross_boundary", "lag_1", "lag_3", "within_side"):
            v = rec.get(k)
            if v is not None and v == v:
                out[k + "_over_floor"] = v - floor
    return out


def paired_contrast(res, title):
    """Per-seed differences, causal target minus bidirectional target.

    This is the test the design actually supports, and the unpaired version
    above understates it badly. `train_arm` seeds torch, numpy and the mask
    generator from the arm's seed alone, so seed s in cell `bb` and seed s in
    cell `bc` start from identical weights and then see the identical sequence
    of batches and the identical masks. The only thing that differs is the
    target encoder's attention mask. Differencing within a seed therefore
    cancels initialisation and batch-order noise, which is most of the spread in
    the unpaired table; what is left is the intervention.

    Pairs are formed within a context flag as well, (bb,bc) and (cb,cc), so a
    context effect cannot leak into the target contrast. n = 2 * n_seeds.

    The t statistic is a one-sample t on the differences. With n=8 it is a
    direction-and-consistency gauge, not a p-value to quote; a contrast whose
    every pair shares a sign is real in a way that a mean +/- sd cannot show.
    """
    rows = [("cross_boundary", "H1 autocorr cross_boundary", "lower"),
            ("lag_1", "H1 autocorr lag_1", "lower"),
            ("lag_3", "H1 autocorr lag_3", "lower"),
            ("baseline_cos", "   (floor: baseline_cos)", "n/a"),
            ("cross_boundary_over_floor", "H1 xbound OVER FLOOR", "lower"),
            ("lag_3_over_floor", "H1 lag_3 OVER FLOOR", "lower"),
            ("boundary_excess", "   boundary_excess", "~0 if mask-blind"),
            ("copy_alignment", "H2 copy_alignment", "lower"),
            ("copy_loss_ratio", "H2 copy_loss_ratio", "higher"),
            ("model_loss_on_copyable", "   H2 numerator: model loss", "n/a"),
            ("extrap_oracle_loss", "   H2 denominator: copy loss", "n/a"),
            ("interp_advantage", "H2 interp_advantage", "lower"),
            ("interp_loss_ratio", "H2 interp_loss_ratio", "higher"),
            ("interp_oracle_loss", "   interp oracle loss", "n/a"),
            ("eff_rank", "   eff_rank", "n/a"),
            (None, "H3 probe ridge_mean", "open")]
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print("{:<30}{:>16}{:>10}{:>10}{:>12}{:>16}".format(
        "quantity (causal - bidir)", "mean delta", "sd", "t", "signs", "predicted"))
    print("-" * 104)
    for key, label, pred in rows:
        d = []
        for bcode, ccode in (("bb", "bc"), ("cb", "cc")):
            b, c = res["arms"].get(bcode), res["arms"].get(ccode)
            if not b or not c:
                continue
            for rb, rc in zip(b["runs"], c["runs"]):
                if key is None:
                    x, y = rb["ridge_mean"], rc["ridge_mean"]
                else:
                    x = derived(rb["diagnostics"]).get(key)
                    y = derived(rc["diagnostics"]).get(key)
                if x is None or y is None or x != x or y != y:
                    continue
                d.append(y - x)
        if len(d) < 2:
            print("{:<30}{:>16}{:>10}{:>10}{:>12}{:>16}".format(
                label, "-", "-", "-", "-", pred))
            continue
        d = np.array(d)
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("nan")
        signs = "{}+/{}- of {}".format(int((d > 0).sum()), int((d < 0).sum()), len(d))
        print("{:<30}{:>16}{:>10}{:>10}{:>12}{:>16}".format(
            label, "{:+.5f}".format(d.mean()), "{:.5f}".format(sd),
            _f(t, "{:+.2f}"), signs, pred))
    print("=" * 104)


def ctx_contrast(res, title):
    """The other margin of the 2x2: causal CONTEXT minus bidirectional context.

    Reported because the temporal table showed the context flag moving probe R^2
    and effective rank more than the target flag did, and a 2x2 whose second
    margin is never printed is a 2x2 that was run for nothing. Pairs are
    (bb,cb) and (bc,cc), holding the target encoder fixed.
    """
    rows = [("cross_boundary", "autocorr cross_boundary"),
            ("copy_alignment", "copy_alignment"),
            ("copy_loss_ratio", "copy_loss_ratio"),
            ("eff_rank", "eff_rank"),
            (None, "probe ridge_mean")]
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print("{:<30}{:>16}{:>10}{:>10}{:>12}".format(
        "quantity (causal - bidir)", "mean delta", "sd", "t", "signs"))
    print("-" * 104)
    for key, label in rows:
        d = []
        for bcode, ccode in (("bb", "cb"), ("bc", "cc")):
            b, c = res["arms"].get(bcode), res["arms"].get(ccode)
            if not b or not c:
                continue
            for rb, rc in zip(b["runs"], c["runs"]):
                if key is None:
                    x, y = rb["ridge_mean"], rc["ridge_mean"]
                else:
                    x = derived(rb["diagnostics"]).get(key)
                    y = derived(rc["diagnostics"]).get(key)
                if x is None or y is None or x != x or y != y:
                    continue
                d.append(y - x)
        if len(d) < 2:
            print("{:<30}{:>16}{:>10}{:>10}{:>12}".format(label, "-", "-", "-", "-"))
            continue
        d = np.array(d)
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("nan")
        print("{:<30}{:>16}{:>10}{:>10}{:>12}".format(
            label, "{:+.5f}".format(d.mean()), "{:.5f}".format(sd),
            _f(t, "{:+.2f}"),
            "{}+/{}- of {}".format(int((d > 0).sum()), int((d < 0).sum()), len(d))))
    print("=" * 104)


def main():
    files = [("results/causal_ablation.json", "TEMPORAL"),
             ("results/interpolation_arm.json", "INTERPOLATION")]
    for rel, tag in files:
        p = ROOT / rel
        if not p.exists():
            print("missing {}".format(p))
            continue
        res = json.loads(p.read_text())
        cfg = res["config"]
        print("\n\n" + "#" * 104)
        print("# {}  strategy={} seeds={} steps={} d_model={} layers={} batch={} device={}".format(
            tag, cfg["strategy"], cfg["seeds"], cfg["steps"], cfg["d_model"],
            cfg["layers"], cfg["batch"], res["device"]))
        print("#" * 104)
        probe_table(res, "PROBE R^2 vs pm-jepa CONTROLS  ({})".format(tag))
        diag_table(res, "CAUSALITY DIAGNOSTICS, FINAL STEP, mean +/- sd over {} seeds".format(
            cfg["seeds"]))
        contrast(res, "TARGET-ENCODER CONTRAST, unpooled seeds  {}".format(tag))
        paired_contrast(res, "TARGET-ENCODER CONTRAST, PAIRED BY SEED  {}  (the powered test)".format(tag))
        ctx_contrast(res, "CONTEXT-ENCODER CONTRAST, PAIRED BY SEED  {}".format(tag))

    cb = ROOT / "results" / "cache_bench.json"
    if cb.exists():
        d = json.loads(cb.read_text())
        print("\n\n" + "#" * 104)
        print("# H4 CACHE BENCHMARK")
        print("#" * 104)
        print("{:<6}{:>7}{:>8}{:>12}{:>16}{:>16}{:>10}{:>14}{:>14}".format(
            "dev", "P", "exact", "maxdiff", "full ms/tick", "cache ms/tick",
            "speedup", "full tok/s", "cache tok/s"))
        print("-" * 104)
        for r in d["rows"]:
            print("{:<6}{:>7}{:>8}{:>12.1e}{:>16.3f}{:>16.3f}{:>10.2f}{:>14.0f}{:>14.0f}"
                  .format(r["device"], r["P"], str(r["exact_vs_full_reencode"]),
                          r["max_abs_diff"], 1e3 * r["full_s_per_tick_mean"],
                          1e3 * r["cache_s_per_tick_mean"], r["speedup"],
                          r["full_tokens_per_s"], r["cache_tokens_per_s"]))
        print("-" * 104)
        print("regime: {}".format(d["regime"]))


if __name__ == "__main__":
    main()
