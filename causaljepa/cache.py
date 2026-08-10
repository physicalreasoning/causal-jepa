"""KV cache for the TIME-attention sublayers of a causal FactorisedEncoder.

Why only the time sublayers: a factorised block attends twice, once across
strikes inside a single patch and once across patches inside a single strike.
The strike sublayer never reads a token from another patch, so re-running it on
a newly arrived patch costs nothing and reproduces the original activations
exactly. Only the time sublayer looks backwards, and only its K/V need to
survive between calls.

The correctness hazard this file guards against is a silent one. If the cache
holds the wrong keys, or the positional offset is off by one, incremental
encoding still returns plausible-looking embeddings; nothing raises. The only
way to know is to compare against a full re-encode, which is what
`tests/test_cache_exactness.py` does at atol=1e-4. Treat any failure there as a
bug in this file, never as a tolerance to relax.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Per-layer K/V for the TIME-attention sublayers only.

    Layout is (B*K, heads, p_seen, head_dim): the time sublayer flattens batch
    and strike into one sequence-batch axis, so one strike ladder at one batch
    element is one independent sequence over patches. `n_seen` is the number of
    patches already absorbed, and it doubles as the positional offset for the
    next call; keeping it here rather than at the call site is what stops the
    two from drifting apart.

    Strike attention is within-patch and needs no cache.
    """

    k: List[Tensor] = field(default_factory=list)
    v: List[Tensor] = field(default_factory=list)
    n_seen: int = 0

    @classmethod
    def empty(cls) -> "KVCache":
        return cls(k=[], v=[], n_seen=0)

    def reset(self) -> None:
        """Drop every stored key and value and rewind the patch counter.

        Reusing a cache across two different windows without resetting is the
        easy mistake: the second window would attend to the first window's past
        and produce embeddings no full forward could ever match.
        """
        self.k.clear()
        self.v.clear()
        self.n_seen = 0

    def __len__(self) -> int:
        return self.n_seen


def _resolve_encoder(model):
    """Accept either a FactorisedEncoder or a CausalJEPA and return the encoder.

    Callers stream through a whole JEPA more often than through a bare encoder,
    and silently caching the wrong tower (target instead of context) would be
    invisible in the output shapes, so the resolution rule is explicit here.
    """
    if hasattr(model, "causal"):
        return model
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "causal"):
        return enc
    raise TypeError(
        "encode_incremental needs a FactorisedEncoder or a module exposing one "
        "as `.encoder`, got {}".format(type(model).__name__))


def encode_incremental(
    model,
    obs_window: Tensor,
    cache: Optional[KVCache] = None,
    start_patch: Optional[int] = None,
) -> Tuple[Tensor, KVCache]:
    """Encode only the NEW patches in `obs_window`, reusing `cache`.

    Two call conventions, because streaming code and test code want different
    things:

      * `start_patch=None` (default): `obs_window` is the window PREFIX from
        minute 0. Patches already counted in `cache.n_seen` are skipped, so the
        caller can hand over a growing buffer and let the cache decide what is
        new. This is the convention the exactness test uses.
      * `start_patch=p`: `obs_window` holds exactly the minutes of the patches
        starting at absolute patch index p, and p must equal `cache.n_seen`.
        This is for true streaming, where the old minutes have been dropped.

    Returns (tokens_for_new_patches, updated_cache) with tokens shaped
    (B, n_new, K, d). Requires the encoder to be causal: with bidirectional time
    attention an already-emitted token is not final, so no cache can be exact
    and returning one anyway would be a lie.

    SPEC NOTE: section 3 fixes the signature at (model, obs_window, cache);
    `start_patch` is appended as an optional keyword so positional callers are
    unaffected.
    """
    enc = _resolve_encoder(model)
    if not enc.causal:
        raise ValueError(
            "encode_incremental requires a causal encoder; with bidirectional "
            "time attention a future patch rewrites past tokens, so no KV cache "
            "can reproduce a full re-encode")
    if cache is None:
        cache = KVCache.empty()

    stride, length = enc.patch_stride, enc.patch_length
    t = obs_window.shape[1]

    if start_patch is not None:
        if start_patch != cache.n_seen:
            raise ValueError(
                "start_patch={} does not continue the cache at n_seen={}".format(
                    start_patch, cache.n_seen))
        chunk = obs_window
    else:
        p_avail = 0 if t < length else (t - length) // stride + 1
        n_new = p_avail - cache.n_seen
        if n_new <= 0:
            b, _, k, _ = obs_window.shape
            empty = obs_window.new_zeros(b, 0, k, enc.d_model)
            return empty, cache
        first = cache.n_seen * stride
        last = (p_avail - 1) * stride + length
        chunk = obs_window[:, first:last]

    return enc(chunk, cache=cache), cache
