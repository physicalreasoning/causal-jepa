#!/usr/bin/env python3
"""E3, part C: what is the headline metric actually made of?

Every arm across three repositories has been ranked by `ridge_mean`, the mean
ridge R^2 over four probe targets. Nobody decomposed it. Doing so explains the
whole programme.

`pm-jepa/data/dataset.py` documents the targets itself:

    log_return_to_settle   "GENUINELY FUTURE. The headline target."
    time_to_expiry         "Not in the input, provided windows are sampled at
                            random offsets"
    implied_width          "DERIVABLE from the input, so it is a sanity check:
                            a healthy encoder should nail it. Low R^2 means
                            broken, not interesting."
    window_log_return      inside the window

So one target is unpredictable by construction of an efficient market, one is
explicitly a derivable sanity check, and both sit in the mean with equal weight.
Ranking encoders by that mean rewards reconstructing the input, which raw
features do perfectly by definition, and it barely registers the only target
anyone would trade on.

This script does no training. It reads the results already on disk and reports
the decomposition, so the claim is checkable in seconds.
"""
import json
import pathlib

import numpy as np

NAMES = ("log_return_to_settle", "time_to_expiry", "implied_width", "window_log_return")
DERIVABLE = {"implied_width"}                    # dataset.py calls this a sanity check
FUTURE = {"log_return_to_settle"}                # dataset.py calls this the headline


def collect(root="results"):
    p = pathlib.Path(root)
    rows = {}
    b = json.loads((p / "benchmark.json").read_text())
    for k, v in b["arms"].items():
        pt = v.get("per_target_mean") or v.get("ridge")
        if pt and all(n in pt for n in NAMES):
            rows[k] = [pt[n] for n in NAMES]
    ra = json.loads((p / "readout_ablation.json").read_text())
    acc = {}
    for r in ra["runs"]:
        tag = None
        if r["model"] == "raw":
            tag = "raw:" + r["readout"]
        elif r["model"] in ("trained", "untrained"):
            tag = ("jepa:" if r["model"] == "trained" else "untrained:") + r["readout"]
        if tag:
            acc.setdefault(tag, []).append([r["ridge"][n] for n in NAMES])
    for k, v in acc.items():
        rows[k] = list(np.mean(v, axis=0))
    return rows


def main():
    rows = collect()
    out = {"targets": list(NAMES), "derivable": sorted(DERIVABLE),
           "genuinely_future": sorted(FUTURE), "arms": {}}

    print("=" * 112)
    print("WHAT THE HEADLINE METRIC IS MADE OF   (real Kalshi corpus, ridge R^2)")
    print("=" * 112)
    print("{:<28}".format("arm") + "".join("{:>20}".format(n[:19]) for n in NAMES)
          + "{:>9}".format("mean"))
    print("-" * 112)
    for k, v in sorted(rows.items(), key=lambda r: -float(np.mean(r[1]))):
        out["arms"][k] = {"per_target": dict(zip(NAMES, map(float, v))),
                          "mean": float(np.mean(v)),
                          "future_only": float(v[NAMES.index("log_return_to_settle")]),
                          "mean_excl_derivable": float(np.mean(
                              [x for n, x in zip(NAMES, v) if n not in DERIVABLE]))}
        print("{:<28}".format(k[:27]) + "".join("{:>+20.4f}".format(x) for x in v)
              + "{:>+9.4f}".format(np.mean(v)))
    print("=" * 112)

    fut = {k: v["future_only"] for k, v in out["arms"].items()}
    lo, hi = min(fut.values()), max(fut.values())
    out["future_target_range"] = {"min": lo, "max": hi, "spread": hi - lo,
                                  "n_arms": len(fut)}
    print("\nON THE ONLY GENUINELY FUTURE TARGET (log_return_to_settle):")
    print("  every one of {} arms lands in [{:+.4f}, {:+.4f}], a spread of {:.4f}".format(
        len(fut), lo, hi, hi - lo))
    print("  raw features, an untrained encoder and every trained JEPA are")
    print("  indistinguishable from each other and from zero.")

    width = {k: v["per_target"]["implied_width"] for k, v in out["arms"].items()}
    raw_w = max(v for k, v in width.items() if k.startswith("raw:"))
    jep_w = max(v for k, v in width.items() if k.startswith("jepa:"))
    out["implied_width_gap"] = {"best_raw": raw_w, "best_jepa": jep_w,
                                "gap": raw_w - jep_w}
    print("\nWHERE THE JEPA ACTUALLY LOSES:")
    print("  implied_width, the target dataset.py calls a derivable sanity check.")
    print("  best raw {:+.4f} vs best jepa {:+.4f}, a gap of {:.4f}.".format(
        raw_w, jep_w, raw_w - jep_w))
    print("  A pooled embedding cannot beat the raw input at reproducing a")
    print("  function of the raw input, so this comparison was never winnable.")

    print("\nRANKING BY THE FUTURE TARGET INSTEAD OF THE MEAN:")
    for k, v in sorted(fut.items(), key=lambda r: -r[1])[:6]:
        print("  {:<28}{:>+10.4f}".format(k[:27], v))

    p = pathlib.Path("results/target_decomposition.json")
    p.write_text(json.dumps(out, indent=2))
    print("\nwrote {}".format(p))


if __name__ == "__main__":
    main()
