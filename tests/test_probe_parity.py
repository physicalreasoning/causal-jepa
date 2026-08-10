"""SPEC test 6: our ridge probe must reproduce pm-jepa's to 1e-9.

The only reason to port `probes.py` instead of writing a cleaner one is that
every number in this repo gets reported next to pm-jepa's controls (ridge
+0.449 identity, +0.460 handcrafted, best JEPA arm +0.377). A probe that picks
a different lambda, standardises differently, or cuts the validation split at a
different index produces R^2 on a scale that is not those numbers' scale, and
nothing in the pipeline would complain. This test is the tripwire.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PM_JEPA = pathlib.Path("/Users/nikita/pm-jepa")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causaljepa import probes as cjprobes  # noqa: E402


def _original():
    """Load pm-jepa's probes.py off disk, by path, as the reference oracle."""
    path = PM_JEPA / "model" / "probes.py"
    if not path.is_file():
        pytest.skip("pm-jepa reference tree not present at {}".format(PM_JEPA))
    spec = importlib.util.spec_from_file_location("pm_jepa_probes_reference", path)
    mod = importlib.util.module_from_spec(spec)
    was_writing = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = was_writing
    return mod


def _synthetic(seed, n_tr, n_te, dim, k):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(dim, k))
    x_tr = rng.normal(size=(n_tr, dim))
    x_te = rng.normal(size=(n_te, dim))
    y_tr = x_tr @ w + 0.3 * rng.normal(size=(n_tr, k))
    y_te = x_te @ w + 0.3 * rng.normal(size=(n_te, k))
    return x_tr, y_tr, x_te, y_te


def test_ridge_grid_identical():
    assert cjprobes.RIDGE_GRID == _original().RIDGE_GRID


def test_probe_source_is_verbatim():
    """Byte-compare everything below the module docstring.

    Numeric parity on a handful of draws is necessary but not sufficient; two
    implementations can agree on well-conditioned Gaussian data and diverge on
    the rank-deficient features a collapsed encoder produces. Comparing the
    source removes that whole class of doubt.
    """
    orig_path = PM_JEPA / "model" / "probes.py"
    if not orig_path.is_file():
        pytest.skip("pm-jepa reference tree not present")
    ours = (ROOT / "causaljepa" / "probes.py").read_text().split('"""', 2)[2]
    theirs = orig_path.read_text().split('"""', 2)[2]
    assert ours == theirs


@pytest.mark.parametrize("seed,n_tr,n_te,dim,k", [
    (0, 400, 120, 32, 4),
    (1, 1500, 400, 96, 4),
    (2, 300, 90, 16, 2),
    (3, 260, 80, 200, 4),      # dim > n, so the ridge term is load bearing
])
def test_ridge_probe_parity(seed, n_tr, n_te, dim, k):
    orig = _original()
    x_tr, y_tr, x_te, y_te = _synthetic(seed, n_tr, n_te, dim, k)
    r2_a, lam_a = cjprobes.ridge_probe(x_tr, y_tr, x_te, y_te)
    r2_b, lam_b = orig.ridge_probe(x_tr, y_tr, x_te, y_te)
    assert lam_a == lam_b
    dev = float(np.max(np.abs(r2_a - r2_b)))
    assert dev < 1e-9, "ridge R^2 deviates by {:.3e}".format(dev)


def test_ridge_probe_parity_on_degenerate_features():
    """Duplicated and constant columns, the shape a collapsed encoder emits."""
    orig = _original()
    x_tr, y_tr, x_te, y_te = _synthetic(7, 500, 150, 8, 4)
    x_tr = np.concatenate([x_tr, x_tr, np.zeros((len(x_tr), 3))], axis=1)
    x_te = np.concatenate([x_te, x_te, np.zeros((len(x_te), 3))], axis=1)
    r2_a, lam_a = cjprobes.ridge_probe(x_tr, y_tr, x_te, y_te)
    r2_b, lam_b = orig.ridge_probe(x_tr, y_tr, x_te, y_te)
    assert lam_a == lam_b
    assert float(np.max(np.abs(r2_a - r2_b))) < 1e-9


def test_mlp_and_run_probes_parity():
    orig = _original()
    x_tr, y_tr, x_te, y_te = _synthetic(4, 600, 200, 24, 4)
    a = cjprobes.mlp_probe(x_tr, y_tr, x_te, y_te, steps=80, seed=11)
    b = orig.mlp_probe(x_tr, y_tr, x_te, y_te, steps=80, seed=11)
    assert float(np.max(np.abs(a - b))) < 1e-9

    names = ("log_return_to_settle", "time_to_expiry", "implied_width",
             "window_log_return")
    ra = cjprobes.run_probes(x_tr, y_tr, x_te, y_te, names)
    rb = orig.run_probes(x_tr, y_tr, x_te, y_te, names)
    assert ra.keys() == rb.keys()
    assert ra["ridge_lambda"] == rb["ridge_lambda"]
    for key in ("ridge_mean", "mlp_mean"):
        assert abs(ra[key] - rb[key]) < 1e-9
    for head in ("ridge", "mlp"):
        assert ra[head].keys() == rb[head].keys()
        for n in names:
            assert abs(ra[head][n] - rb[head][n]) < 1e-9
