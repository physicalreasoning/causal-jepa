#!/usr/bin/env python3
"""Standing verification for causaljepa/data.py and causaljepa/probes.py.

Two things can rot silently and invalidate every comparison to pm-jepa: the
corpus changing shape underneath us, and the probe drifting from the original.
Both are checked here against hard numbers rather than eyeballed, so a
regression fails loudly instead of producing plausible-looking R^2.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PM_JEPA = pathlib.Path("/Users/nikita/pm-jepa")
sys.path.insert(0, str(ROOT))

from causaljepa import data as cjdata          # noqa: E402
from causaljepa import probes as cjprobes      # noqa: E402

EXPECT_WINDOWS = 25818
EXPECT_SHAPE = (25818, 24, 24, 4)
EXPECT_TRAIN, EXPECT_TEST = 20506, 5312


def _tree_fingerprint(root: pathlib.Path):
    """(path, size, mtime_ns) for every file under root, to prove read-only use."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            st = p.stat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


def _load_original_probes():
    spec = importlib.util.spec_from_file_location(
        "pm_jepa_probes_original", PM_JEPA / "model" / "probes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    fails = []

    print("=" * 72)
    print("A. corpus load (pm-jepa loader, via path injection)")
    print("=" * 72)
    before = _tree_fingerprint(PM_JEPA)

    X, Y, owner, names = cjdata.load_corpus(PM_JEPA, window=24, stride=4)
    after = _tree_fingerprint(PM_JEPA)

    print("  X       {}  {}".format(X.shape, X.dtype))
    print("  Y       {}  {}".format(Y.shape, Y.dtype))
    print("  owner   {}  {}   events {:,}".format(owner.shape, owner.dtype,
                                                  len(np.unique(owner))))
    print("  targets {}".format(names))
    print("  memory  {:.1f} MB".format(X.nbytes / 1e6))

    if tuple(X.shape) != EXPECT_SHAPE:
        fails.append("SHAPE MISMATCH: got {} expected {}".format(tuple(X.shape), EXPECT_SHAPE))
    if len(X) != EXPECT_WINDOWS:
        fails.append("WINDOW COUNT MISMATCH: got {} expected {}".format(len(X), EXPECT_WINDOWS))

    print()
    print("  read-only check against the pm-jepa tree")
    added = set(after) - set(before)
    changed = {k for k in set(before) & set(after) if before[k] != after[k]}
    removed = set(before) - set(after)
    print("    files added {}  changed {}  removed {}".format(
        len(added), len(changed), len(removed)))
    if added or changed or removed:
        fails.append("MUTATED pm-jepa TREE: +{} ~{} -{}".format(
            sorted(added)[:3], sorted(changed)[:3], sorted(removed)[:3]))

    print()
    print("=" * 72)
    print("B. split_by_event")
    print("=" * 72)
    tr, te = cjdata.split_by_event(owner, frac=0.2, seed=0)
    print("  train {:>7,}   test {:>7,}   total {:>7,}".format(
        int(tr.sum()), int(te.sum()), len(owner)))
    print("  train events {:,}   test events {:,}   overlap {}".format(
        len(np.unique(owner[tr])), len(np.unique(owner[te])),
        len(set(owner[tr].tolist()) & set(owner[te].tolist()))))
    if int(tr.sum()) != EXPECT_TRAIN or int(te.sum()) != EXPECT_TEST:
        fails.append("SPLIT MISMATCH: got {}/{} expected {}/{}".format(
            int(tr.sum()), int(te.sum()), EXPECT_TRAIN, EXPECT_TEST))
    if set(owner[tr].tolist()) & set(owner[te].tolist()):
        fails.append("EVENT LEAK: an event id appears in both splits")

    lc = cjdata._pm_jepa_module(PM_JEPA, "load_corpus")
    for seed in (0, 1, 7):
        a_tr, a_te = cjdata.split_by_event(owner, 0.2, seed)
        b_tr, b_te = lc.split_by_event(owner, 0.2, seed)
        same = bool((a_tr == b_tr).all() and (a_te == b_te).all())
        print("  seed {}: bit-identical to pm-jepa split_by_event -> {}".format(seed, same))
        if not same:
            fails.append("SPLIT PARITY FAILED at seed {}".format(seed))

    print()
    print("=" * 72)
    print("C. per-target mean / std (full corpus)")
    print("=" * 72)
    for i, n in enumerate(names):
        print("  {:<22} mean {:>14.6f}   std {:>14.6f}".format(
            n, float(Y[:, i].mean()), float(Y[:, i].std())))
    print()
    print("  train split")
    for i, n in enumerate(names):
        print("  {:<22} mean {:>14.6f}   std {:>14.6f}".format(
            n, float(Y[tr, i].mean()), float(Y[tr, i].std())))
    print("  test split")
    for i, n in enumerate(names):
        print("  {:<22} mean {:>14.6f}   std {:>14.6f}".format(
            n, float(Y[te, i].mean()), float(Y[te, i].std())))

    print()
    print("=" * 72)
    print("D. probe parity vs pm-jepa/model/probes.py")
    print("=" * 72)
    orig = _load_original_probes()
    print("  RIDGE_GRID ours {}".format(cjprobes.RIDGE_GRID))
    print("  RIDGE_GRID pm   {}".format(orig.RIDGE_GRID))
    if cjprobes.RIDGE_GRID != orig.RIDGE_GRID:
        fails.append("RIDGE GRID DIFFERS")

    ours_src = (ROOT / "causaljepa" / "probes.py").read_text()
    pm_src = (PM_JEPA / "model" / "probes.py").read_text()
    ours_body = ours_src.split('"""', 2)[2]
    pm_body = pm_src.split('"""', 2)[2]
    print("  code body sha256 ours {}".format(hashlib.sha256(ours_body.encode()).hexdigest()[:16]))
    print("  code body sha256 pm   {}".format(hashlib.sha256(pm_body.encode()).hexdigest()[:16]))
    if ours_body != pm_body:
        fails.append("PROBE CODE BODY DIFFERS FROM pm-jepa")

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial, (n_tr, n_te, dim, k) in enumerate(
            [(400, 120, 32, 4), (1500, 400, 96, 4), (300, 90, 16, 2)]):
        w_true = rng.normal(size=(dim, k))
        x_tr = rng.normal(size=(n_tr, dim)).astype(np.float64)
        x_te = rng.normal(size=(n_te, dim)).astype(np.float64)
        y_tr = x_tr @ w_true + 0.3 * rng.normal(size=(n_tr, k))
        y_te = x_te @ w_true + 0.3 * rng.normal(size=(n_te, k))

        r2_a, lam_a = cjprobes.ridge_probe(x_tr, y_tr, x_te, y_te)
        r2_b, lam_b = orig.ridge_probe(x_tr, y_tr, x_te, y_te)
        dev = float(np.max(np.abs(r2_a - r2_b)))
        worst = max(worst, dev)
        print("  trial {}  n_tr={:<5} dim={:<3} k={}  lambda {} vs {}  max|dR2| = {:.3e}".format(
            trial, n_tr, dim, k, lam_a, lam_b, dev))
        print("           ours {}".format(np.array2string(r2_a, precision=12)))
        print("           pm   {}".format(np.array2string(r2_b, precision=12)))
        if lam_a != lam_b:
            fails.append("LAMBDA DIFFERS on trial {}".format(trial))

    # Real corpus, identity features, so parity is checked on the data we ship.
    flat_tr = X[tr].reshape(int(tr.sum()), -1)[:, ::64].astype(np.float64)
    flat_te = X[te].reshape(int(te.sum()), -1)[:, ::64].astype(np.float64)
    r2_a, lam_a = cjprobes.ridge_probe(flat_tr, Y[tr].astype(np.float64),
                                       flat_te, Y[te].astype(np.float64))
    r2_b, lam_b = orig.ridge_probe(flat_tr, Y[tr].astype(np.float64),
                                   flat_te, Y[te].astype(np.float64))
    dev = float(np.max(np.abs(r2_a - r2_b)))
    worst = max(worst, dev)
    print("  real corpus (subsampled identity feats, dim={})  lambda {} vs {}  max|dR2| = {:.3e}".format(
        flat_tr.shape[1], lam_a, lam_b, dev))
    print("           ours {}".format(np.array2string(r2_a, precision=12)))
    print("           pm   {}".format(np.array2string(r2_b, precision=12)))

    print()
    print("  WORST ABSOLUTE DEVIATION ACROSS ALL TRIALS: {:.3e}  (threshold 1e-9)".format(worst))
    if worst > 1e-9:
        fails.append("PROBE PARITY FAILED: worst deviation {:.3e}".format(worst))

    mlp_a = cjprobes.mlp_probe(x_tr, y_tr, x_te, y_te, steps=60, seed=3)
    mlp_b = orig.mlp_probe(x_tr, y_tr, x_te, y_te, steps=60, seed=3)
    mdev = float(np.max(np.abs(mlp_a - mlp_b)))
    print("  mlp_probe (seed 3, 60 steps) max|dR2| = {:.3e}".format(mdev))
    if mdev > 1e-9:
        fails.append("MLP PARITY FAILED: {:.3e}".format(mdev))

    rp_a = cjprobes.run_probes(x_tr, y_tr, x_te, y_te, ("a", "b", "c", "d"))
    rp_b = orig.run_probes(x_tr, y_tr, x_te, y_te, ("a", "b", "c", "d"))
    print("  run_probes ridge_mean ours {!r}  pm {!r}  delta {:.3e}".format(
        rp_a["ridge_mean"], rp_b["ridge_mean"],
        abs(rp_a["ridge_mean"] - rp_b["ridge_mean"])))
    if abs(rp_a["ridge_mean"] - rp_b["ridge_mean"]) > 1e-9:
        fails.append("run_probes PARITY FAILED")

    print()
    print("=" * 72)
    print("E. pm-jepa control baselines (for side-by-side reporting)")
    print("=" * 72)
    base = json.loads((PM_JEPA / "results" / "baselines.json").read_text())
    print("  {:<14} {:>6} {:>12} {:>12}".format("arm", "dim", "ridge_mean", "mlp_mean"))
    for arm, blk in base.items():
        print("  {:<14} {:>6} {:>12.6f} {:>12.6f}".format(
            arm, blk["dim"], blk["ridge_mean"], blk["mlp_mean"]))

    print()
    print("=" * 72)
    if fails:
        print("FAILURES ({}):".format(len(fails)))
        for f in fails:
            print("  !! {}".format(f))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
