"""SPEC tests 3 and 4: patch grid geometry, and the shape of every forward key.

The shape contract is load-bearing well beyond "does it run". Every ported
diagnostic reads (B, P*K, d) as patch-major and strike-minor; hand it the
transposed grid and `copy_diagnostics` reports temporal masking as 0% copyable
and slate masking as 100%, exactly backwards, without raising. The geometry test
is the same hazard one level up: if `patch_mask` disagreed with the sampler about
where minute 16 lives, the copy oracle would quote a reference patch that had
already seen the answer, and the copy ratio would come out flatteringly low.
"""
import pytest
import torch

from causaljepa.masking import interpolation_span, sample_mask
from causaljepa.model import CausalJEPA

T, K, C, D = 24, 24, 4, 64
B = 3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CausalJEPA(T, K, C, d_model=D, n_layers=2,
                      causal_context=False, causal_target=True)


def test_patch_mask_geometry(model):
    assert model.n_patches == 6
    assert model.patch_length == 4 and model.patch_stride == 4

    sm = sample_mask(B, T, K, "temporal", patch_length=model.patch_length)
    # minutes 18..23 are masked at horizon_frac 0.25, which is patches 4 and 5
    assert sm[:, :18].sum() == 0
    assert bool(sm[:, 18:].all())
    pm = model.patch_mask(sm)
    assert pm.shape == (B, 6, K)
    masked = sorted(torch.nonzero(pm[0].any(dim=-1)).flatten().tolist())
    assert masked == [4, 5], masked


def test_patch_mask_geometry_interpolation(model):
    """The new arm has to leave whole visible patches on BOTH sides.

    If the hole straddled a boundary, `patch_mask`'s any() rule would widen it to
    the neighbouring patches and the interpolation arm would quietly become a
    wider temporal one, with no visible symptom other than a shifted loss.
    """
    assert interpolation_span(T, 0.25, 4) == (8, 16)
    sm = sample_mask(B, T, K, "interpolation", patch_length=4)
    pm = model.patch_mask(sm)
    masked = sorted(torch.nonzero(pm[0].any(dim=-1)).flatten().tolist())
    assert masked == [2, 3], masked
    assert bool((~pm[:, [0, 1, 4, 5]]).all())


def test_patch_mask_is_conservative(model):
    """One masked minute contaminates its whole patch, never the reverse."""
    sm = torch.zeros(1, T, K, dtype=torch.bool)
    sm[0, 9, 3] = True                      # one minute, one strike, inside patch 2
    pm = model.patch_mask(sm)
    assert int(pm.sum()) == 1
    assert bool(pm[0, 2, 3])


@pytest.mark.parametrize("strategy", ["temporal", "slate", "contiguous", "mixed",
                                      "random", "interpolation"])
def test_shapes(model, strategy):
    torch.manual_seed(1)
    obs = torch.randn(B, T, K, C)
    sm = sample_mask(B, T, K, strategy, patch_length=model.patch_length)
    out = model(obs, sm)
    assert set(out) == {"pred", "target", "context", "mask_flat", "patch_mask"}
    assert out["pred"].shape == (B, 6 * K, D)
    assert out["target"].shape == (B, 6 * K, D)
    assert out["context"].shape == (B, 6 * K, D)
    assert out["mask_flat"].shape == (B, 6 * K)
    assert out["mask_flat"].dtype == torch.bool
    assert out["patch_mask"].shape == (B, 6, K)
    assert out["patch_mask"].dtype == torch.bool
    # mask_flat is patch-major flattening of patch_mask, not some other order.
    assert torch.equal(out["mask_flat"], out["patch_mask"].reshape(B, -1))
    assert not out["target"].requires_grad
    assert out["pred"].requires_grad


def test_represent_shape(model):
    torch.manual_seed(2)
    z = model.represent(torch.randn(B, T, K, C))
    assert z.shape == (B, D)
    assert torch.isfinite(z).all()


def test_masked_minutes_are_zero_filled(model):
    """pm-jepa zero-fills rather than dropping, and the diagnostics assume it.

    If a masked minute still reached the encoder, the context tower would see the
    answer and every copy diagnostic in the repo would be measuring nothing.
    """
    torch.manual_seed(4)
    obs = torch.randn(B, T, K, C)
    sm = sample_mask(B, T, K, "temporal", patch_length=4)
    other = obs.clone()
    other[sm] = 99.0
    with torch.no_grad():
        a = model.encode(obs, sm, target=False)
        b = model.encode(other, sm, target=False)
    assert torch.equal(a, b)


def test_ema_moves_target_towards_context(model):
    """The target must move (1 - m) of the way to the context, for asymmetric m.

    Deliberately NOT tested at momentum 0.5: there `m` and `1 - m` are the same
    number, so `pt.mul_(m).add_(po, alpha=1-m)` and the inverted
    `pt.mul_(1-m).add_(po, alpha=m)` produce identical weights and the test
    cannot tell a correct EMA from one that tracks the context almost entirely.
    That inversion would make the target encoder a near-copy of the context
    encoder, collapsing the JEPA to a trivial self-prediction, and every copy
    diagnostic in the repo would still look plausible.
    """
    torch.manual_seed(6)
    m = CausalJEPA(T, K, C, d_model=32, n_layers=1)
    before = m.target_encoder.embed.weight.detach().clone()
    with torch.no_grad():
        m.encoder.embed.weight.add_(1.0)
    ctx = m.encoder.embed.weight.detach().clone()
    assert not torch.allclose(before, ctx)

    for mom in (0.9, 0.996):
        m.target_encoder.embed.weight.data.copy_(before)
        m.update_target(momentum=mom)
        after = m.target_encoder.embed.weight.detach()
        assert torch.allclose(after, mom * before + (1.0 - mom) * ctx, atol=1e-7), mom
        # A high momentum must stay CLOSE to where it was, not jump to context.
        assert (after - before).abs().max() < (after - ctx).abs().max()
