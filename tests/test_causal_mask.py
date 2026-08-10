"""SPEC test 2: perturbing patch p must not move any token before p.

Reading the attention mask and agreeing that it looks lower-triangular is not
proof. The mask is built in one place, the positional offset in another, and the
factorised block runs a SECOND attention (over strikes) that is bidirectional by
design; a leak introduced by that sublayer, or by a residual path that mixes
patches outside attention entirely, would be invisible to any inspection of the
time mask. Perturbing the input and watching what moves is device-independent,
implementation-independent evidence, and it is bit-exact rather than approximate:
under a correct causal mask the earlier tokens are not merely close, they are the
same floats.

The bidirectional control is not decoration. It is what distinguishes "the past
does not move" from "nothing moves", which is what a dead encoder would report.
"""
import torch

from causaljepa.model import CausalJEPA, FactorisedEncoder

T, K, C = 24, 24, 4
PERTURBED_PATCH = 3


def _pair(causal, seed=0):
    torch.manual_seed(seed)
    enc = FactorisedEncoder(K, C, 6, 4, d_model=64, n_layers=3, n_heads=8,
                            causal=causal).eval()
    torch.manual_seed(seed + 1)
    a = torch.randn(2, T, K, C)
    b = a.clone()
    b[:, PERTURBED_PATCH * 4:(PERTURBED_PATCH + 1) * 4] += 5.0
    with torch.no_grad():
        return enc(a), enc(b)


def test_causal_mask_no_future_leak():
    za, zb = _pair(causal=True)
    past_a, past_b = za[:, :PERTURBED_PATCH], zb[:, :PERTURBED_PATCH]
    assert torch.equal(past_a, past_b), \
        "past moved by {:.3e}".format(float((past_a - past_b).abs().max()))
    # The perturbation has to actually do something, or the test is vacuous.
    moved = float((za[:, PERTURBED_PATCH:] - zb[:, PERTURBED_PATCH:]).abs().max())
    assert moved > 1e-3, "perturbation had no effect anywhere: {:.3e}".format(moved)


def test_bidirectional_does_leak():
    za, zb = _pair(causal=False)
    leak = float((za[:, :PERTURBED_PATCH] - zb[:, :PERTURBED_PATCH]).abs().max())
    assert leak > 1e-3, "bidirectional encoder did not leak: {:.3e}".format(leak)


def test_causal_prefix_consistency():
    """enc(prefix)[:, :p] must equal enc(full)[:, :p], with no cache involved.

    A separate statement from the perturbation test: that one says later inputs
    do not change earlier outputs, this one says the mere PRESENCE of later
    patches does not, which is what makes the encoder usable on a partial window
    at inference time.
    """
    torch.manual_seed(3)
    enc = FactorisedEncoder(K, C, 6, 4, d_model=64, n_layers=3, n_heads=8,
                            causal=True).eval()
    obs = torch.randn(2, T, K, C)
    with torch.no_grad():
        assert torch.equal(enc(obs[:, :16]), enc(obs)[:, :4])


def test_target_tower_causality_is_independent_of_context():
    """The 2x2 is only a controlled experiment if the flags really are separate.

    The interesting cell is causal_target=True with causal_context=False. If the
    deepcopy in `CausalJEPA.__init__` ever tied the two towers' flags together,
    every cell would still train and the experiment would silently become a 1x2.
    """
    torch.manual_seed(7)
    m = CausalJEPA(T, K, C, d_model=32, n_layers=2,
                   causal_context=False, causal_target=True).eval()
    obs = torch.randn(2, T, K, C)
    pert = obs.clone()
    pert[:, PERTURBED_PATCH * 4:(PERTURBED_PATCH + 1) * 4] += 5.0
    with torch.no_grad():
        ctx_a = m.encode(obs, None, target=False)
        ctx_b = m.encode(pert, None, target=False)
        tgt_a = m.encode(obs, None, target=True)
        tgt_b = m.encode(pert, None, target=True)
    ctx_leak = float((ctx_a[:, :PERTURBED_PATCH] - ctx_b[:, :PERTURBED_PATCH]).abs().max())
    tgt_leak = float((tgt_a[:, :PERTURBED_PATCH] - tgt_b[:, :PERTURBED_PATCH]).abs().max())
    assert ctx_leak > 1e-3, "bidirectional context did not leak: {:.3e}".format(ctx_leak)
    assert tgt_leak == 0.0, "causal target leaked: {:.3e}".format(tgt_leak)


def test_towers_start_from_identical_weights():
    """Otherwise a between-arm difference could be an initialisation difference."""
    torch.manual_seed(11)
    m = CausalJEPA(T, K, C, d_model=32, n_layers=2,
                   causal_context=False, causal_target=True)
    for a, b in zip(m.encoder.parameters(), m.target_encoder.parameters()):
        assert torch.equal(a, b)
    assert not any(p.requires_grad for p in m.target_encoder.parameters())
