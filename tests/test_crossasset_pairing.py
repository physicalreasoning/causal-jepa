"""The cross-asset corpus must pair real minutes, not nearby ones.

Finding 16 rests entirely on the claim that a BTC row and an ETH row in the same
window describe THE SAME MINUTE of market. Minutes that failed `validate_ladder`
were dropped independently per asset when each event was cached, so two events
covering the same nominal hour routinely disagree about which minutes exist. If
the alignment were on position rather than on timestamp, a BTC minute would be
paired with a different ETH minute, and that manufactures cross-asset
correlation structure out of nothing. Every number in finding 16 would be an
artefact and nothing would complain.

This is the tripwire for that.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causaljepa import crossasset as ca  # noqa: E402


def _root():
    from causaljepa import data as cj_data
    try:
        root = cj_data._resolve_root(None)
    except FileNotFoundError:
        pytest.skip("no corpus available")
    if not (pathlib.Path(root) / "data_cache" / "event_list").is_dir():
        pytest.skip("no event_list cache; series labels unrecoverable")
    return root


def test_series_index_resolves_and_is_multi_asset():
    idx = ca.series_index(_root())
    assert idx, "no cached event resolved to a series"
    by_series = {}
    for _k, (s, _t) in idx.items():
        by_series[s] = by_series.get(s, 0) + 1
    assert len(by_series) >= 2, (
        "cross-asset work needs at least two series, found {}".format(by_series))
    for s, n in by_series.items():
        assert n > 0 and s.startswith("KX")


def test_pairs_are_one_to_one():
    pairs = ca.pair_events(_root(), min_shared=34)
    if not pairs:
        pytest.skip("no pairs at this threshold")
    a = [p["a_key"] for p in pairs]
    b = [p["b_key"] for p in pairs]
    assert len(set(a)) == len(a), "an event was reused on the A side"
    assert len(set(b)) == len(b), "an event was reused on the B side"
    assert not set(a) & set(b), "an event appears on both sides"


def test_alignment_is_on_timestamps_not_position():
    """Both legs of an aligned pair must carry identical timestamps.

    This is the assertion that matters. It is not enough that the two arrays
    have the same length: they must refer to the same wall-clock minutes.
    """
    root = _root()
    pairs = ca.pair_events(root, min_shared=34)
    if not pairs:
        pytest.skip("no pairs at this threshold")
    checked = 0
    for pr in pairs[:25]:
        al = ca._aligned(root, pr)
        if al is None:
            continue
        obs_a, obs_b, st_a, st_b, ts = al
        assert len(obs_a) == len(obs_b) == len(st_a) == len(st_b) == len(ts)
        assert np.all(np.diff(ts) > 0), "aligned timestamps are not strictly increasing"

        # Re-derive the intersection independently and require an exact match.
        ga = ca._load(root, pr["a_key"])
        gb = ca._load(root, pr["b_key"])
        shared = np.intersect1d(ga[1], gb[1])
        np.testing.assert_array_equal(ts, np.sort(shared))

        # And every aligned row must sit at the timestamp it claims, on BOTH legs.
        ia = {int(t): i for i, t in enumerate(ga[1])}
        ib = {int(t): i for i, t in enumerate(gb[1])}
        for j in range(0, len(ts), max(1, len(ts) // 5)):
            t = int(ts[j])
            np.testing.assert_array_equal(obs_a[j], ga[0][ia[t]])
            np.testing.assert_array_equal(obs_b[j], gb[0][ib[t]])
        checked += 1
    assert checked > 0, "no pair was actually checked"


def test_both_ladders_are_present_and_ordered():
    """X must be BTC strikes then ETH strikes, and the views must recover them."""
    root = _root()
    X, Y, owner, names, stats = ca.load_corpus(root, horizon=10)
    assert X.shape[2] % 2 == 0
    k = X.shape[2] // 2
    assert len(X) == len(Y) == len(owner)
    assert list(names) == list(ca.CROSS_TARGET_NAMES)
    assert np.isfinite(Y).all(), "cross-asset targets contain non-finite values"
    # The two halves must not be identical, or the pairing collapsed.
    assert not np.allclose(X[:, :, :k, :], X[:, :, k:, :]), (
        "both halves of X are the same ladder; the second asset is missing")
    assert stats["pairs_contributing"] > 0


def test_realised_corr_is_in_range_and_not_degenerate():
    root = _root()
    _X, Y, _o, names, _s = ca.load_corpus(root, horizon=10)
    c = Y[:, list(names).index("realised_corr")].astype(np.float64)
    assert (c >= -1.0).all() and (c <= 1.0).all()
    assert c.std() > 0.05, "realised_corr is nearly constant; it cannot rank anything"


def test_signed_spread_return_is_centred():
    """The negative control must look like a martingale."""
    root = _root()
    _X, Y, _o, names, _s = ca.load_corpus(root, horizon=10)
    c = Y[:, list(names).index("fwd_signed_spread_return")].astype(np.float64)
    assert abs(c.mean()) < 0.1 * c.std(), (
        "fwd_signed_spread_return drifts: mean {:.3e} vs std {:.3e}".format(
            c.mean(), c.std()))
