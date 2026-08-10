"""Corpus access for causal-jepa, borrowed wholesale from pm-jepa.

This project forks pm-jepa's model, not its data. Every number we report has to
sit next to pm-jepa's prior results, and that comparison is only honest if both
projects see the identical windows in the identical order with the identical
train/test split. So this module does not build windows; it imports pm-jepa's
`load_corpus` module off disk and calls it. Reimplementing `windows_from_event`
here would be a second chance to disagree about window starts, target offsets or
event filtering, and any such disagreement would silently invalidate the
comparison instead of raising.

The corpus also stays where it is. `data_cache/event_arrays/*.npz` is 45 MB of
snapshotted Kalshi ladders that cannot be rebuilt offline; copying it into this
repo would create a second copy that can drift from the one pm-jepa trains on.
Everything below is read-only with respect to the pm-jepa tree: we import a
module and call `build()`, which only reads. We never call pm-jepa's `main()`,
which writes `results/corpus_shape.json`.
"""
import importlib.util
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np

# Loading the corpus walks ~3,400 npz files, which is a few seconds. Tests and
# multi-arm sweeps call this repeatedly in one process, so memoise on the exact
# (root, window, stride) that produced the arrays. The arrays handed back are
# the cached objects themselves; treat them as read-only, because mutating them
# would poison every later caller in the same process.
_CORPUS_CACHE: Dict[Tuple[str, int, int], tuple] = {}
_MODULE_CACHE: Dict[str, object] = {}


def _pm_jepa_module(pm_jepa_root, name: str):
    """Import a top-level pm-jepa module by file path, under a namespaced key.

    Imported by path rather than by plain `import load_corpus` for two reasons.
    First, pm-jepa's module names (`load_corpus`, `train`, `analyze`) are
    generic enough to collide with something else already in `sys.modules`, and
    a silent collision would hand us the wrong `build()`. Second, the namespaced
    key makes it obvious in a traceback that the frame belongs to pm-jepa and
    not to this repo.

    The pm-jepa root still has to go on `sys.path`, because `load_corpus.py`
    itself does `from data import dataset`, an absolute import that must resolve
    inside the pm-jepa tree.
    """
    root = pathlib.Path(pm_jepa_root).expanduser().resolve()
    key = "{}::{}".format(root, name)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    path = root / "{}.py".format(name)
    if not path.is_file():
        raise FileNotFoundError(
            "expected pm-jepa's {}.py at {}; pass the pm-jepa repo root".format(name, path)
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    spec = importlib.util.spec_from_file_location("pm_jepa_{}".format(name), path)
    if spec is None or spec.loader is None:
        raise ImportError("could not build an import spec for {}".format(path))
    module = importlib.util.module_from_spec(spec)

    # Importing pm-jepa's modules pulls in `data/dataset.py` by ordinary import,
    # and CPython would drop `.pyc` files into `pm-jepa/data/__pycache__/`. That
    # is a write into a tree this project has declared read-only, and it is
    # exactly what `test_no_data_mutation` looks for; on a machine where the
    # caches happen to be warm the write would not show up and the guarantee
    # would be accidental rather than enforced. Suppress bytecode writing for
    # the duration of the import and restore the interpreter's setting after.
    was_writing = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        # Register before exec so that a module doing relative-ish self-reference
        # during import does not re-enter and load a second copy.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = was_writing

    _MODULE_CACHE[key] = module
    return module


def load_corpus(pm_jepa_root, window: int = 24, stride: int = 4
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """-> (X, Y, owner, target_names), straight out of pm-jepa's own builder.

    X is (N, window, K=24, C=4) float32, Y is (N, 4) float32, owner is (N,) int
    event ids. At window=24, stride=4 this is 25,818 windows over 3,127 events;
    if you get a different count the cache has changed underneath both projects
    and the comparison to pm-jepa's results is no longer valid.

    `target_names` comes from `data.dataset.TARGET_NAMES` rather than being
    hard-coded, so a reordering of the target columns upstream cannot silently
    mislabel our probe R^2 tables.
    """
    root = str(pathlib.Path(pm_jepa_root).expanduser().resolve())
    key = (root, int(window), int(stride))
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]

    lc = _pm_jepa_module(root, "load_corpus")
    # `build` returns (X, Y, owner, n_short); n_short is the count of events too
    # short to yield a single window, useful for the log but not for training.
    X, Y, owner, _n_short = lc.build(window=window, stride=stride)

    # Read TARGET_NAMES off the module pm-jepa itself imported (`load_corpus.py`
    # does `from data import dataset`) rather than importing `data.dataset` a
    # second time here. A second import could resolve against a different
    # `sys.path` entry and hand back a different column order than the one that
    # actually produced Y, which would mislabel every probe R^2 we publish.
    target_names = list(lc.dataset.TARGET_NAMES)
    out = (X, Y, owner, target_names)
    _CORPUS_CACHE[key] = out
    return out


def split_by_event(owner: np.ndarray, frac: float = 0.2, seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out whole events. -> (train_mask, test_mask), both (N,) bool.

    Windows inside one event overlap by `window - stride` minutes, so a
    per-window split puts near-duplicate rows on both sides and inflates every
    probe R^2 we report. Splitting on the event id is the only version of this
    that measures generalisation to an unseen hour of the market.

    Reimplemented here rather than delegated so that this module has no import
    cycle back into pm-jepa for a pure function, but the shuffle is the same
    `default_rng(seed)` over `np.unique(owner)`, so for a given seed the masks
    are bit-identical to `pm-jepa/load_corpus.py:split_by_event`. There is a
    parity assertion for exactly this in the test suite; if it ever fails, the
    prior results are no longer a valid control.
    """
    ids = np.unique(owner)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * frac))
    test_ids = set(ids[:n_test].tolist())
    is_test = np.array([o in test_ids for o in owner])
    return ~is_test, is_test


def summarise(X: np.ndarray, Y: np.ndarray, owner: np.ndarray,
              target_names: List[str], tr: np.ndarray, te: np.ndarray) -> Dict:
    """Config block for `results/*.json`, so every run records what it trained on.

    A result file with a bare R^2 and no corpus fingerprint is unfalsifiable
    later: nobody can tell whether an arm improved or the corpus grew. Inline
    this next to the model config in every artefact we write.
    """
    return {
        "windows": int(len(X)),
        "obs_shape": [int(v) for v in X.shape],
        "events": int(len(np.unique(owner))),
        "train": int(tr.sum()),
        "test": int(te.sum()),
        "targets": {
            n: {"mean": float(Y[:, i].mean()), "std": float(Y[:, i].std())}
            for i, n in enumerate(target_names)
        },
    }
