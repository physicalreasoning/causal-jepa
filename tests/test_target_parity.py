"""E7 targets must be a SUBSET of the published corpus, not a new slicing.

The whole argument of E7 rests on one claim: the original four targets and the
four new ones are measured on the SAME windows, so a difference in the table is
a difference in the targets rather than a difference in the data. If
`causaljepa/targets.py` enumerated windows even slightly differently from
`pm-jepa/data/dataset.py` -- a different stride origin, an off-by-one on the
window end, a different validity filter -- then E7's within-run control would be
comparing two different corpora and every conclusion drawn from it would be
wrong, silently and unfalsifiably.

This is the tripwire for that, in the same spirit as `test_probe_parity.py`. It
reconstructs the survivor mask independently and asserts the recomputed original
targets are bit-comparable to the ones pm-jepa's own builder produces.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causaljepa import targets as cjt  # noqa: E402

WINDOW, STRIDE, HORIZON = 24, 4, 10


def _corpus_root():
    from causaljepa import data as cj_data
    try:
        return cj_data._resolve_root(None)
    except FileNotFoundError:
        pytest.skip("no corpus available")


def _events(root, limit=40):
    """Cached events long enough to yield at least one horizon-bearing window."""
    cache = pathlib.Path(root) / "data_cache" / "event_arrays"
    out = []
    for f in sorted(cache.glob("*.npz")):
        try:
            with np.load(f) as z:
                obs, ts, state = z["obs"], z["ts"], z["state"]
                settle = float(z["settle"][0])
        except (ValueError, OSError, KeyError):
            continue
        if obs.shape[0] >= WINDOW + HORIZON:
            out.append((obs, ts, state, settle))
        if len(out) >= limit:
            break
    if not out:
        pytest.skip("no event long enough for window+horizon")
    return out


def _reference_dataset(root):
    """The window builder that actually produced the corpus.

    Loaded through `causaljepa.data`'s own import machinery rather than by file
    path. `data/dataset.py` does `from . import kalshi, slate`, a relative
    import that only resolves when the module is loaded as part of its package,
    so a bare `spec_from_file_location` on it raises ImportError. Going through
    the loader module the same way `load_corpus` does gets the package context
    right and, more importantly, guarantees this test compares against the
    exact module that built `Y`, not a second copy of it that could have been
    resolved from a different `sys.path` entry.
    """
    from causaljepa import data as cj_data
    lc = (cj_data._local_loader() if cj_data._is_local_cache(root)
          else cj_data._pm_jepa_module(root, "load_corpus"))
    ds = getattr(lc, "dataset", None)
    if ds is None or not hasattr(ds, "windows_from_event"):
        pytest.skip("reference window builder not reachable; parity untestable")
    return ds


def test_original_targets_match_pm_jepa_on_surviving_windows():
    """The recomputed original targets must equal pm-jepa's, window for window."""
    root = _corpus_root()
    ds = _reference_dataset(root)
    checked = 0
    for obs, ts, state, settle in _events(root):
        built = cjt.windows_from_event(obs, ts, state, settle,
                                       WINDOW, STRIDE, HORIZON)
        if built is None:
            continue
        _x_sub, y_orig, _y_new = built
        x_full, y_full = ds.windows_from_event(obs, ts, state, settle,
                                               WINDOW, STRIDE)
        if len(x_full) == 0:
            continue

        # Rebuild the survivor mask independently of targets.py.
        T = obs.shape[0]
        keep = []
        for i, start in enumerate(range(0, T - WINDOW + 1, STRIDE)):
            end = start + WINDOW - 1
            fut = end + HORIZON
            if fut > T - 1:
                continue
            spot, width, t_rem = (float(v) for v in state[end])
            if spot <= 0 or float(state[start][0]) <= 0 or width <= 0 or t_rem <= 0:
                continue
            dt = float(ts[fut] - ts[end])
            if dt <= 0 or dt > 2.0 * HORIZON * 60.0:
                continue
            if (np.asarray(state[end:fut + 1, 0]) <= 0).any():
                continue
            keep.append(i)

        assert len(keep) == len(y_orig), (
            "survivor count disagrees: mask {} vs targets.py {}".format(
                len(keep), len(y_orig)))
        np.testing.assert_allclose(y_orig, y_full[keep], rtol=0, atol=1e-6)
        np.testing.assert_allclose(_x_sub, x_full[keep], rtol=0, atol=0)
        checked += len(keep)
    assert checked > 0, "no windows were actually compared"


def test_surviving_windows_are_a_strict_subset():
    """Requiring a future must remove windows, never add or reorder them."""
    root = _corpus_root()
    ds = _reference_dataset(root)
    saw_drop = False
    for obs, ts, state, settle in _events(root, limit=25):
        built = cjt.windows_from_event(obs, ts, state, settle,
                                       WINDOW, STRIDE, HORIZON)
        x_full, _ = ds.windows_from_event(obs, ts, state, settle, WINDOW, STRIDE)
        n_sub = 0 if built is None else len(built[0])
        assert n_sub <= len(x_full)
        saw_drop = saw_drop or n_sub < len(x_full)
    assert saw_drop, "horizon filter dropped nothing; the test is vacuous"


def test_vol_forecast_error_matches_its_definition():
    """Recompute log(realised / implied) by hand for every checked window."""
    root = _corpus_root()
    for obs, ts, state, settle in _events(root, limit=10):
        built = cjt.windows_from_event(obs, ts, state, settle,
                                       WINDOW, STRIDE, HORIZON)
        if built is None:
            continue
        _x, _yo, y_new = built
        T = obs.shape[0]
        row = 0
        for start in range(0, T - WINDOW + 1, STRIDE):
            end = start + WINDOW - 1
            fut = end + HORIZON
            if fut > T - 1:
                continue
            spot, width, t_rem = (float(v) for v in state[end])
            if spot <= 0 or float(state[start][0]) <= 0 or width <= 0 or t_rem <= 0:
                continue
            dt = float(ts[fut] - ts[end])
            if dt <= 0 or dt > 2.0 * HORIZON * 60.0:
                continue
            path = np.asarray(state[end:fut + 1, 0], dtype=np.float64)
            if (path <= 0).any():
                continue
            rv = float(np.sqrt((np.diff(np.log(path)) ** 2).sum()))
            iv = float(width / spot * np.sqrt(dt / t_rem))
            expect = np.log((rv + 1e-6) / (iv + 1e-6))
            assert abs(float(y_new[row, 1]) - expect) < 1e-5
            assert abs(float(y_new[row, 0]) - rv) < 1e-8
            assert abs(float(y_new[row, 3])
                       - float(np.log(path[-1] / path[0]))) < 1e-8
            assert abs(float(y_new[row, 2]) - abs(float(np.log(path[-1] / path[0])))) < 1e-8
            row += 1
        assert row == len(y_new)


def test_targets_are_finite():
    root = _corpus_root()
    X, Yo, Yn, owner, on, nn_, stats = cjt.load_corpus(
        root, window=WINDOW, stride=STRIDE, horizon=HORIZON)
    assert np.isfinite(Yo).all(), "original targets contain non-finite values"
    assert np.isfinite(Yn).all(), "new targets contain non-finite values"
    assert len(X) == len(Yo) == len(Yn) == len(owner)
    assert stats["events_contributing"] > 0
    assert list(on) == list(cjt.ORIG_TARGET_NAMES)
    assert list(nn_) == list(cjt.NEW_TARGET_NAMES)


def test_signed_return_is_centred():
    """The negative control must look like a martingale, or it is not a control.

    A ten-minute forward log return should have a mean small relative to its own
    spread. A drifting control would make the leak assertion in
    `scripts/hidden_state_targets.py` meaningless.
    """
    root = _corpus_root()
    _X, _Yo, Yn, _o, _on, nn_, _s = cjt.load_corpus(
        root, window=WINDOW, stride=STRIDE, horizon=HORIZON)
    col = Yn[:, list(nn_).index("fwd_signed_return")].astype(np.float64)
    assert abs(col.mean()) < 0.05 * col.std(), (
        "fwd_signed_return drifts: mean {:.3e} vs std {:.3e}".format(
            col.mean(), col.std()))


def test_longer_horizon_keeps_fewer_windows():
    root = _corpus_root()
    counts = []
    for h in (5, 10, 20):
        X, _Yo, _Yn, _o, _on, _nn, _s = cjt.load_corpus(
            root, window=WINDOW, stride=STRIDE, horizon=h)
        counts.append(len(X))
    assert counts[0] > counts[1] > counts[2], counts
