"""SPEC test 1: incremental encoding must equal a full re-encode, atol=1e-4.

This is the single most important test in the repo, and the reason is that its
failure mode is silent. A cache that stores the wrong keys, or that is off by one
in the positional offset, still returns embeddings of the right shape and the
right rough magnitude; a training run built on it converges to something, and the
only symptom is that streaming inference disagrees with batch inference by an
amount nobody measures. If this test fails, the bug is in `cache.py` or in the
offset handling of `FactorisedEncoder.forward`. Do not relax the tolerance.

The teeth half matters as much. `atol=1e-4` on float32 activations is a bar that
a broken cache could conceivably clear by accident if the encoder barely mixed
across patches, so the same comparison is run against a BIDIRECTIONAL forward,
where exactness is impossible in principle, and asserted to fail by orders of
magnitude.
"""
import pytest
import torch

from causaljepa.cache import KVCache, encode_incremental
from causaljepa.model import CausalJEPA, FactorisedEncoder

ATOL = 1e-4
B, T, K, C = 3, 24, 24, 4


def _encoder(causal, seed=0, d_model=64, n_layers=3):
    torch.manual_seed(seed)
    return FactorisedEncoder(K, C, T // 4, 4, d_model=d_model, n_layers=n_layers,
                             n_heads=8, causal=causal).eval()


@pytest.fixture(scope="module")
def obs():
    torch.manual_seed(99)
    return torch.randn(B, T, K, C)


@pytest.mark.parametrize("chunk", [1, 2, 3, 6])
def test_cache_exactness(obs, chunk):
    """Feed the window `chunk` patches at a time and rebuild the whole thing."""
    enc = _encoder(causal=True)
    with torch.no_grad():
        full = enc(obs)
        cache, pieces = KVCache.empty(), []
        for p in range(0, 6, chunk):
            tok, cache = encode_incremental(enc, obs[:, : (p + chunk) * 4], cache)
            pieces.append(tok)
    inc = torch.cat(pieces, dim=1)
    assert inc.shape == full.shape
    dev = float((inc - full).abs().max())
    assert dev < ATOL, "chunk={} deviates by {:.3e}".format(chunk, dev)


def test_cache_exactness_true_streaming(obs):
    """The form a real streaming caller uses: only the NEW minutes are handed in.

    Worth testing separately from the growing-prefix form because it is the only
    one that exercises `start_patch`, and it is the one where an offset bug has
    nowhere to hide: the encoder never sees the old minutes at all, so every
    piece of the past has to come from the cache.
    """
    enc = _encoder(causal=True)
    with torch.no_grad():
        full = enc(obs)
        cache, pieces = KVCache.empty(), []
        for p in range(6):
            tok, cache = encode_incremental(enc, obs[:, p * 4:(p + 1) * 4], cache,
                                            start_patch=p)
            pieces.append(tok)
    dev = float((torch.cat(pieces, dim=1) - full).abs().max())
    assert dev < ATOL, "streaming form deviates by {:.3e}".format(dev)
    assert cache.n_seen == 6


def test_cache_refuses_bidirectional(obs):
    enc = _encoder(causal=False)
    with pytest.raises(ValueError):
        encode_incremental(enc, obs)
    with pytest.raises(ValueError):
        enc(obs, cache=KVCache.empty())


def test_cache_exactness_has_teeth(obs):
    """The same cached path, same weights, against a bidirectional full forward.

    Proves the 1e-4 bar is discriminating rather than trivially met. The guard in
    `encode_incremental` is lifted by hand here; that is the point of the test and
    not a pattern to copy.
    """
    enc = _encoder(causal=False)
    with torch.no_grad():
        full_bidir = enc(obs)
    enc.causal = True
    with torch.no_grad():
        cache, pieces = KVCache.empty(), []
        for p in range(6):
            tok, cache = encode_incremental(enc, obs[:, : (p + 1) * 4], cache)
            pieces.append(tok)
    dev = float((torch.cat(pieces, dim=1) - full_bidir).abs().max())
    assert dev > 100 * ATOL, "bar is not discriminating: {:.3e}".format(dev)


def test_cache_reset_clears_history(obs):
    """A reused cache must not attend to the previous window's past.

    Cheap to get wrong in a streaming loop and impossible to see in the output,
    so the failure is asserted directly: a cache carried over without `reset()`
    would make the second window's first patch attend to six phantom patches.
    """
    enc = _encoder(causal=True)
    cache = KVCache.empty()
    with torch.no_grad():
        encode_incremental(enc, obs, cache)
        assert cache.n_seen == 6 and len(cache.k) == enc.n_layers
        cache.reset()
        assert cache.n_seen == 0 and not cache.k and not cache.v
        again, cache = encode_incremental(enc, obs, cache)
        fresh, _ = encode_incremental(enc, obs, KVCache.empty())
    assert float((again - fresh).abs().max()) == 0.0


def test_cache_streams_the_context_tower_of_a_jepa(obs):
    """`encode_incremental(model, ...)` must resolve to the CONTEXT encoder.

    Silently caching the target tower instead would produce embeddings of the
    right shape from the wrong network, which no shape check catches.
    """
    torch.manual_seed(5)
    m = CausalJEPA(T, K, C, d_model=32, n_layers=2,
                   causal_context=True, causal_target=False)
    m.eval()
    with torch.no_grad():
        tok, _ = encode_incremental(m, obs)
        direct = m.encode(obs, None, target=False)
    assert float((tok - direct).abs().max()) < ATOL
    with pytest.raises(ValueError):
        encode_incremental(m.target_encoder, obs)      # bidirectional target
    with pytest.raises(TypeError):
        encode_incremental(torch.nn.Linear(2, 2), obs)
