"""SPEC test 5: the directional copy oracle, on data with a known copy structure.

`copy_diagnostics` reports one number for "could you have solved this by copying",
and under a middle-of-window hole that number cannot tell an extrapolating model
from an interpolating one: almost every masked position has a visible past, so
the forward reference never enters the average. The split oracle exists to make
the direction visible, and H2 is stated in terms of it. So the test plants the
answer: build targets where the masked patches are literally a copy of the patch
AFTER the hole, and `interp_advantage` must be positive; plant a copy of the
patch BEFORE it, and the same number must be negative.

The nan half is not defensive padding. These run inside the training loop at
every log step, on whatever mask the sampler produced; `temporal` masking has no
forward reference at all, and a raise there would kill an overnight sweep.
"""
import torch

from causaljepa.diagnostics import (
    copy_diagnostics, directional_copy_diagnostics, effective_rank,
    target_autocorrelation)
from causaljepa.masking import sample_mask

B, P, K, D = 8, 6, 24, 16


def _interp_mask():
    """Patch-level mask with the hole at patches 2 and 3, both sides visible."""
    pm = torch.zeros(B, P, K, dtype=torch.bool)
    pm[:, 2:4, :] = True
    return pm


def _planted(copy_from):
    """Targets where the masked patches are an exact copy of patch `copy_from`."""
    torch.manual_seed(0)
    tgt = torch.randn(B, P, K, D)
    tgt[:, 2] = tgt[:, copy_from]
    tgt[:, 3] = tgt[:, copy_from]
    return tgt.reshape(B, P * K, D)


def test_directional_diagnostics_forward_copy():
    """Masked == the patch AFTER the hole, so interpolation is the easy solution."""
    tgt = _planted(copy_from=4)
    pred = torch.zeros_like(tgt)
    d = directional_copy_diagnostics(pred, tgt, _interp_mask())
    assert d["interp_advantage"] > 0.1, d["interp_advantage"]
    assert d["interp_oracle_loss"] < 1e-9
    assert d["extrap_oracle_loss"] > 0.1
    assert d["paired_frac"] == 1.0


def test_directional_diagnostics_backward_copy():
    """Mirror image: the sign has to flip, or the split is not reading direction."""
    tgt = _planted(copy_from=1)
    pred = torch.zeros_like(tgt)
    d = directional_copy_diagnostics(pred, tgt, _interp_mask())
    assert d["interp_advantage"] < -0.1, d["interp_advantage"]
    assert d["extrap_oracle_loss"] < 1e-9


def test_directional_diagnostics_no_planted_copy():
    """Independent patches: neither direction should look meaningfully easier."""
    torch.manual_seed(1)
    tgt = torch.randn(B, P * K, D)
    pred = torch.zeros_like(tgt)
    d = directional_copy_diagnostics(pred, tgt, _interp_mask())
    assert abs(d["interp_advantage"]) < 0.05, d["interp_advantage"]


def test_directional_diagnostics_nan_safe_under_temporal():
    """`temporal` masking has no forward reference, so interp keys must be nan.

    Reporting 0.0 instead would read as "interpolation is exactly as easy as
    extrapolation", which is a claim, not a missing measurement.
    """
    torch.manual_seed(2)
    sm = sample_mask(B, 24, K, "temporal", patch_length=4)
    pm = sm.reshape(B, P, 4, K).any(dim=2)
    tgt = torch.randn(B, P * K, D)
    d = directional_copy_diagnostics(tgt, tgt, pm)
    assert d["interp_frac"] == 0.0
    assert d["interp_loss_ratio"] != d["interp_loss_ratio"]
    assert d["interp_advantage"] != d["interp_advantage"]
    assert d["extrap_frac"] == 1.0
    assert d["extrap_loss_ratio"] == d["extrap_loss_ratio"]


def test_directional_diagnostics_nan_safe_on_empty_and_full_masks():
    torch.manual_seed(3)
    tgt = torch.randn(B, P * K, D)
    for pm in (torch.zeros(B, P, K, dtype=torch.bool),
               torch.ones(B, P, K, dtype=torch.bool)):
        d = directional_copy_diagnostics(tgt, tgt, pm)
        c = copy_diagnostics(tgt, tgt, pm)
        assert isinstance(d, dict) and isinstance(c, dict)
        assert d["interp_advantage"] != d["interp_advantage"]


def test_alignment_is_a_cosine():
    """Every *_alignment key must be a cosine, so out-of-range is a wiring bug."""
    torch.manual_seed(4)
    tgt = torch.randn(B, P * K, D)
    pred = torch.randn(B, P * K, D)
    pm = _interp_mask()
    c = copy_diagnostics(pred, tgt, pm)
    d = directional_copy_diagnostics(pred, tgt, pm)
    for k, v in list(c.items()) + list(d.items()):
        if k.endswith("alignment"):
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6, (k, v)


def test_target_autocorrelation_detects_planted_leakage():
    """Orthogonal patches read zero; a leaky target that mixes them reads high.

    This is the H1 instrument. If it could not separate these two synthetic
    cases it could not separate the two arms either, and an H1 null would be
    uninterpretable.
    """
    torch.manual_seed(5)
    basis = torch.linalg.qr(torch.randn(D, P))[0][:, :P].T          # (P, D)
    clean = basis.view(1, P, 1, D).expand(B, P, K, D).contiguous()
    leaky = clean.clone()
    leaky[:, 2:4] = clean.mean(dim=1, keepdim=True)                 # masked = mean of all

    pm = _interp_mask()
    a = target_autocorrelation(clean.reshape(B, P * K, D), pm)
    b = target_autocorrelation(leaky.reshape(B, P * K, D), pm)
    assert abs(a["cross_boundary"]) < 1e-5, a["cross_boundary"]
    assert b["cross_boundary"] > 0.3, b["cross_boundary"]
    assert b["lag_1"] > a["lag_1"]
    for L in range(1, 6):
        assert "lag_{}".format(L) in a


def test_target_autocorrelation_nan_safe():
    torch.manual_seed(6)
    tgt = torch.randn(B, P * K, D)
    r = target_autocorrelation(tgt, torch.zeros(B, P, K, dtype=torch.bool))
    assert r["cross_boundary"] != r["cross_boundary"]
    assert r["lag_1"] == r["lag_1"]                 # lags still defined
    one = target_autocorrelation(torch.randn(1, 1 * K, D),
                                 torch.zeros(1, 1, K, dtype=torch.bool))
    assert one["lag_1"] != one["lag_1"]             # fewer than two patches
    assert one["baseline_cos"] != one["baseline_cos"]   # batch of one


def test_effective_rank_bounds():
    torch.manual_seed(7)
    assert effective_rank(torch.randn(512, 32)) > 25
    low = torch.randn(512, 4) @ torch.randn(4, 32)
    assert effective_rank(low) < 6
