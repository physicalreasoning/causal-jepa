"""SPEC test 7: loading the corpus must not touch the pm-jepa tree.

pm-jepa is the control arm. If this repo ever writes into it, whether a stray
`__pycache__`, a re-emitted `results/corpus_shape.json`, or a rebuilt npz, then
the baselines we compare against stop being the baselines that were published
and the comparison quietly becomes circular. The failure is invisible at read
time, so it has to be asserted mechanically.

Two things make this test easy to write in a form that proves nothing, and both
are guarded against explicitly below.

First, `load_corpus` memoises on (root, window, stride). An earlier fixture in
the same session warms that cache, so a later call returns in 0.0000s without
importing a module or opening a single npz. Fingerprinting around *that* call
measures a no-op. The load inside the fingerprint window is therefore forced
COLD: the corpus memo, the module memo, and the `sys.modules` entries pm-jepa's
importer created are all dropped first, so the module import and the ~3,400-file
npz walk genuinely re-run where the fingerprint can see them.

Second, a fingerprint differ that never fires reports "no changes" on a tree
that is being rewritten under it. So the differ is itself tested, on a scratch
directory, against an addition, a content change, and a deletion.

Also asserts the corpus fingerprint. A silent change in window count would move
every arm's R^2 without moving any code.
"""
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PM_JEPA = pathlib.Path(
    os.environ.get("PM_JEPA_ROOT", "../pm-jepa")).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causaljepa import data as cjdata  # noqa: E402

EXPECT_SHAPE = (25818, 24, 24, 4)
EXPECT_TRAIN, EXPECT_TEST = 20506, 5312


def _fingerprint(root: pathlib.Path):
    """Every file's (size, mtime_ns). Directories are not tracked on purpose.

    A directory's mtime moves when a child is created and moves back to nothing
    meaningful when it is removed, so tracking it produces failures that are
    real writes only some of the time. Tracking the files themselves catches an
    added `.pyc` just as well, because the `.pyc` is a file.
    """
    out = {}
    for p in root.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file():
            st = p.stat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


def _diff(before, after):
    return (sorted(set(after) - set(before)),
            sorted(set(before) - set(after)),
            sorted(k for k in set(before) & set(after) if before[k] != after[k]))


def _force_cold():
    """Drop every memo so the next `load_corpus` really re-imports and re-reads.

    Without this the call under test is a dictionary lookup, and the test
    degrades into asserting that a dictionary lookup does not write to disk.
    """
    cjdata._CORPUS_CACHE.clear()
    cjdata._MODULE_CACHE.clear()
    for name in [n for n in list(sys.modules)
                 if n.startswith("pm_jepa_") or n == "data" or n.startswith("data.")]:
        del sys.modules[name]


def _cold_load_watching_the_import(sentinel=False):
    """Force a cold load and report `sys.dont_write_bytecode` as seen at import.

    Returns the list of flag values observed at the moment each pm-jepa module
    body executed. An empty list means no module was executed, i.e. the load was
    served from a memo and whatever the caller went on to assert was vacuous;
    callers check for that explicitly rather than trusting `_force_cold`.

    `sys.dont_write_bytecode` is pinned to `sentinel` first so that "the guard
    restored the flag" is a statement about a value this function chose. Reading
    the ambient value instead makes the check order-dependent: an earlier test
    that already leaked the flag would make the leak look like the status quo
    and the assertion would pass on exactly the runs it exists to fail.
    """
    _force_cold()
    real_spec = importlib.util.spec_from_file_location
    seen = []

    def spy(name, path):
        spec = real_spec(name, path)
        real_exec = spec.loader.exec_module

        def wrapped(mod):
            seen.append(sys.dont_write_bytecode)
            return real_exec(mod)

        spec.loader.exec_module = wrapped
        return spec

    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = sentinel
    importlib.util.spec_from_file_location = spy
    try:
        cjdata.load_corpus(PM_JEPA, window=24, stride=4)
        restored = sys.dont_write_bytecode
    finally:
        importlib.util.spec_from_file_location = real_spec
        sys.dont_write_bytecode = original
    return seen, restored


@pytest.fixture(scope="module")
def corpus():
    if not PM_JEPA.is_dir():
        pytest.skip("pm-jepa reference tree not present at {}".format(PM_JEPA))
    return cjdata.load_corpus(PM_JEPA, window=24, stride=4)


def test_fingerprint_differ_actually_detects(tmp_path):
    """The differ must fire on a tree that changed, or 'no changes' means nothing."""
    (tmp_path / "sub").mkdir()
    keep = tmp_path / "sub" / "keep.txt"
    keep.write_text("original")
    doomed = tmp_path / "sub" / "doomed.txt"
    doomed.write_text("x")

    before = _fingerprint(tmp_path)
    assert _diff(before, _fingerprint(tmp_path)) == ([], [], []), "differ fires on a quiet tree"

    (tmp_path / "sub" / "__pycache__").mkdir()
    (tmp_path / "sub" / "__pycache__" / "m.cpython-39.pyc").write_bytes(b"\x00\x01")
    keep.write_text("rewritten, different length")
    doomed.unlink()

    added, removed, changed = _diff(before, _fingerprint(tmp_path))
    assert any(p.endswith(".pyc") for p in added), added
    assert [p for p in removed if p.endswith("doomed.txt")], removed
    assert [p for p in changed if p.endswith("keep.txt")], changed


def test_no_data_mutation(corpus):
    """Fingerprint a genuinely COLD corpus load, not a memoised no-op."""
    before = _fingerprint(PM_JEPA)
    seen, _ = _cold_load_watching_the_import()
    cjdata.split_by_event(corpus[2], 0.2, 0)
    after = _fingerprint(PM_JEPA)

    # Proof that the fingerprinted region contained a real import and a real
    # corpus walk. Asserting `"pm_jepa_load_corpus" in sys.modules` here instead
    # would be satisfied by the module the fixture imported minutes ago, so a
    # regression that made the load permanently memoised would go unnoticed.
    assert seen, "load was served from a memo; this test measured nothing"

    added, removed, changed = _diff(before, after)
    assert not added, "wrote new files into pm-jepa: {}".format(added[:5])
    assert not removed, "deleted files from pm-jepa: {}".format(removed[:5])
    assert not changed, "modified files in pm-jepa: {}".format(changed[:5])


def test_pm_jepa_import_runs_with_bytecode_writing_off(corpus):
    """Assert the guard's BEHAVIOUR, because its effect is invisible here.

    `data.py` flips `sys.dont_write_bytecode` around the pm-jepa import so that
    CPython cannot drop `.pyc` files into a tree this project declares read-only.
    On a machine whose `pm-jepa/__pycache__` is already warm and current, removing
    that flip changes nothing observable: importlib only writes when the cached
    bytecode is absent or stale, so the fingerprint test above stays green even
    with the guard deleted. It is a clean checkout, or a checkout whose sources
    are newer than their caches, where the guard actually earns its keep, and
    that is precisely the case a fingerprint on this machine cannot exercise.

    So the flag is observed at the moment the pm-jepa module body executes,
    which also covers the nested `from data import dataset` that happens inside
    it. This is the assertion that fails if someone deletes the guard.
    """
    seen, restored = _cold_load_watching_the_import(sentinel=False)
    assert seen, "pm-jepa module was never executed; the cold-load reset failed"
    assert all(seen), "pm-jepa imported with bytecode writing ON: {}".format(seen)
    # Restored to the value the helper pinned, not merely left self-consistent.
    # A guard that flips the flag and forgets to put it back would otherwise
    # disable bytecode caching for every later test in the process.
    assert restored is False, "import guard leaked its setting: {}".format(restored)


def test_corpus_fingerprint(corpus):
    X, Y, owner, names = corpus
    assert tuple(X.shape) == EXPECT_SHAPE, tuple(X.shape)
    assert X.dtype == np.float32
    assert Y.shape == (EXPECT_SHAPE[0], 4)
    assert owner.shape == (EXPECT_SHAPE[0],)
    assert names == ["log_return_to_settle", "time_to_expiry",
                     "implied_width", "window_log_return"]


def test_split_sizes_and_no_event_leak(corpus):
    owner = corpus[2]
    tr, te = cjdata.split_by_event(owner, frac=0.2, seed=0)
    assert tr.dtype == np.bool_ and te.dtype == np.bool_
    assert int(tr.sum()) == EXPECT_TRAIN, int(tr.sum())
    assert int(te.sum()) == EXPECT_TEST, int(te.sum())
    assert int(tr.sum()) + int(te.sum()) == len(owner)
    assert not (tr & te).any()
    # The whole point: an event id may not appear on both sides, because
    # windows within an event overlap by window - stride minutes.
    assert not (set(owner[tr].tolist()) & set(owner[te].tolist()))


def test_split_matches_pm_jepa(corpus):
    """Bit-identical splits, or our train set is not pm-jepa's train set."""
    owner = corpus[2]
    lc = cjdata._pm_jepa_module(PM_JEPA, "load_corpus")
    for seed in (0, 1, 7):
        a_tr, a_te = cjdata.split_by_event(owner, 0.2, seed)
        b_tr, b_te = lc.split_by_event(owner, 0.2, seed)
        assert (a_tr == b_tr).all() and (a_te == b_te).all()
