"""causal-jepa: a JEPA over Kalshi strike ladders with a switchable causal axis.

This package forks pm-jepa's model and reuses its corpus, its probes and its
diagnostics unchanged. The single new degree of freedom is that the TIME axis of
the encoder can be masked causally, independently for the context tower and the
target tower, which is what makes the 2x2 in `scripts/experiment.py` a
controlled experiment rather than four unrelated runs.

The re-exports below exist so that a caller cannot accidentally reach a second
copy of one of these symbols. `causaljepa.data` puts pm-jepa's repo root at the
FRONT of `sys.path`, because pm-jepa's `load_corpus.py` does `from data import
dataset` and that absolute import has to resolve inside their tree. From that
moment on, in the same process, a bare `import train` resolves to
`/Users/nikita/pm-jepa/train.py` and a bare `import load_corpus` to theirs;
verified, not hypothetical. Nothing raises, and the two `train` modules have
similarly named functions with different defaults. (`probes` and `diagnostics`
happen to be safe today only because they sit under pm-jepa's `model/` package
rather than at its root, which is luck, not design.) Always go through
`causaljepa.<module>`.

Import order here is dependency order (cache before model, diagnostics and
masking before train) so that a partially initialised package can never hand a
submodule a half-built sibling.
"""
from . import cache
from . import masking
from . import diagnostics
from . import model
from . import probes
from . import data
from . import train

from .cache import KVCache, encode_incremental
from .data import load_corpus, split_by_event
from .diagnostics import (
    copy_diagnostics,
    dimension_stats,
    directional_copy_diagnostics,
    effective_rank,
    nearest_visible_reference,
    target_autocorrelation,
)
from .masking import interpolation_span, sample_mask
from .model import CausalJEPA, FactorisedEncoder
from .probes import mlp_probe, ridge_probe, run_probes
from .train import represent_all, train_arm

__version__ = "0.1.0"

__all__ = [
    "CausalJEPA",
    "FactorisedEncoder",
    "KVCache",
    "cache",
    "copy_diagnostics",
    "data",
    "diagnostics",
    "dimension_stats",
    "directional_copy_diagnostics",
    "effective_rank",
    "encode_incremental",
    "interpolation_span",
    "load_corpus",
    "masking",
    "mlp_probe",
    "model",
    "nearest_visible_reference",
    "probes",
    "represent_all",
    "ridge_probe",
    "run_probes",
    "sample_mask",
    "split_by_event",
    "target_autocorrelation",
    "train",
    "train_arm",
]
