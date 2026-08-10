"""Mask samplers over the (time, market) grid.

`temporal` is the standard forward-prediction objective and the one exposed to
martingale collapse: because each market's price is a martingale, the target at
t+F is well approximated by copying the embedding at t.

`slate` holds out entire markets across the whole window. There is no copy
solution: the model has to infer the held-out market's state from its siblings,
which requires recovering the shared latent state.

`interpolation` is new in causal-jepa. It is the arm the causal intervention is
aimed at: the hole sits in the MIDDLE of the window, so a bidirectional encoder
can reach visible context on both temporal sides and solve the objective by
interpolating between them, while a causal encoder can only extrapolate forwards
from the past side. Every other strategy here is ported verbatim from pm-jepa so
that the two projects' numbers remain comparable; do not "improve" them.
"""
from typing import Optional

import torch


def sample_mask(
    batch: int,
    window: int,
    n_markets: int,
    strategy: str,
    generator: Optional[torch.Generator] = None,
    horizon_frac: float = 0.25,
    n_held_out: int = 2,
    random_frac: float = 0.25,
    device: str = "cpu",
    patch_length: int = 4,
) -> torch.Tensor:
    """-> bool (batch, window, n_markets). True = masked (a prediction target).

    `patch_length` is appended AFTER `device` on purpose: every pm-jepa caller
    passes the first arguments positionally, so inserting a parameter anywhere
    earlier would silently reinterpret `device` as a patch length. It is only
    read by `interpolation`, which has to align its hole to patch boundaries;
    a hole that straddles a boundary would leave half-masked patches, and
    `patch_mask` treats a half-masked patch as fully masked, which would quietly
    widen the hole past what `horizon_frac` asked for.
    """
    m = torch.zeros(batch, window, n_markets, dtype=torch.bool, device=device)
    horizon = max(1, int(round(window * horizon_frac)))

    if strategy == "temporal":
        m[:, window - horizon:, :] = True

    elif strategy == "slate":
        idx = _held_out_indices(batch, n_markets, n_held_out, generator, device)
        m.scatter_(2, idx.unsqueeze(1).expand(batch, window, n_held_out), True)

    elif strategy == "mixed":
        m[:, window - horizon:, :] = True
        idx = _held_out_indices(batch, n_markets, n_held_out, generator, device)
        m.scatter_(2, idx.unsqueeze(1).expand(batch, window, n_held_out), True)

    elif strategy == "contiguous":
        # Hold out an ADJACENT block of strikes. slate-jepa measured that a
        # randomly held-out market is 93-99% linearly recoverable from its
        # siblings, so random slate masking leaves an interpolation escape
        # hatch. A contiguous block removes the nearby anchors that make
        # interpolation easy. On a no-arbitrage strike ladder, which is smooth
        # by construction, this matters more than it did on sports markets.
        width = min(max(n_held_out, 1), n_markets - 1)
        starts = torch.randint(0, n_markets - width + 1, (batch,),
                               generator=generator, device=device)
        offs = torch.arange(width, device=device)
        idx = starts.unsqueeze(1) + offs.unsqueeze(0)          # (batch, width)
        m.scatter_(2, idx.unsqueeze(1).expand(batch, window, width), True)

    elif strategy == "interpolation":
        t0, t1 = interpolation_span(window, horizon_frac, patch_length)
        m[:, t0:t1, :] = True

    elif strategy == "random":
        r = torch.rand(batch, window, n_markets, generator=generator, device=device)
        m = r < random_frac

    else:
        raise ValueError("unknown masking strategy: {}".format(strategy))

    # Guarantee at least one visible and one masked token per sample.
    flat = m.view(batch, -1)
    empty = ~flat.any(dim=1)
    if empty.any():
        flat[empty, -1] = True
    full = flat.all(dim=1)
    if full.any():
        flat[full, 0] = False
    return flat.view(batch, window, n_markets)


def interpolation_span(window: int, horizon_frac: float = 0.25,
                       patch_length: int = 4) -> tuple:
    """-> (t0, t1) minute bounds of the middle hole, patch-aligned and centred.

    Deterministic on purpose, exactly like `temporal`. The point of this arm is
    to be `temporal`'s mirror image: same number of held-out patches, differing
    only in whether visible context exists on the future side. If the hole
    wandered per sample, the diagnostics that average over the mask boundary
    would be averaging over a different geometry every step, and the H1
    comparison between the causal and bidirectional target encoders would be
    reading mask jitter rather than leakage.

    At window=24, patch_length=4, horizon_frac=0.25 this is patches 2 and 3,
    i.e. minutes 8..15, leaving patches 0,1 before and 4,5 after.

    Degenerate grids are handled by clamping rather than raising, because the
    sampler is called inside the training loop: with fewer than three patches
    there is no room for context on both sides and the hole simply lands at the
    front, which is `temporal` reflected. Callers that need the two-sided
    guarantee should assert P >= 3 themselves.
    """
    n_patches = max(1, window // patch_length)
    width = max(1, int(round(n_patches * horizon_frac)))
    width = min(width, max(1, n_patches - 2))
    start = (n_patches - width) // 2
    if n_patches >= 3:
        # Keep at least one whole visible patch on each side of the hole.
        start = min(max(start, 1), n_patches - 1 - width)
    t0 = start * patch_length
    t1 = min(window, (start + width) * patch_length)
    return t0, t1


def _held_out_indices(batch, n_markets, n_held_out, generator, device):
    n_held_out = min(n_held_out, n_markets - 1)
    scores = torch.rand(batch, n_markets, generator=generator, device=device)
    return scores.argsort(dim=1)[:, :n_held_out]
