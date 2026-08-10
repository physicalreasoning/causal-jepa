"""Put the repo root on `sys.path` before any test imports `causaljepa`.

There is no `pip install -e .` here and there is not going to be: the package has
to be importable from a bare checkout so that a reviewer can run the suite
without building anything. Doing it in a root conftest rather than in each test
file also fixes the import ORDER, which matters more than it looks. Loading the
corpus puts pm-jepa's repo root on `sys.path` so its `load_corpus.py` can find
`data/dataset.py`, and from that moment pm-jepa's own top-level `train.py` and
`model/` are importable under bare names. With this repo's root already first,
`causaljepa.*` still wins.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
