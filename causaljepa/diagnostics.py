"""Collapse diagnostics, including the martingale-collapse detector.

Standard SSL practice checks for REPRESENTATIONAL collapse: all inputs mapping
to the same vector. Effective rank and per-dimension variance catch that.

Neither catches the failure mode that matters on financial data. Because prices
are martingales, E[z_{t+dt} | F_t] ~ z_t, so a model can drive the JEPA loss down
by learning the identity map. The identity map has FULL rank and FULL variance,
so it passes every standard check while encoding no state at all.

`copy_diagnostics` compares the model against an explicit copy oracle: predict
each masked token using the target embedding of the nearest visible token of the
same market. `loss_ratio` near 1.0 means the model has learned nothing that
copying would not give you.

causal-jepa adds a second layer of suspicion, because in pm-jepa the copy oracle
itself is contaminated. The target encoder there is bidirectional, so a target
embedding at a VISIBLE patch was computed with attention over the MASKED patches.
The oracle's reference therefore already contains the answer it is being scored
against, and part of what looked like martingale collapse may be an artefact of
the target encoder rather than a property of the price process. Two new
measurements separate those:

  `directional_copy_diagnostics` splits the oracle into a strictly-backward
  extrapolation copy, which a causal encoder can also do, and a forward
  interpolation copy, which only a bidirectional one can. If interpolation is
  the easier solution, `interp_advantage` is positive and the objective is
  softer than it looks.

  `target_autocorrelation` measures similarity between target embeddings at a
  fixed patch lag and across the mask boundary. This is the direct test of H1:
  at identical mask geometry, a bidirectional target encoder should read higher
  than a causal one, because both members of every pair were computed from the
  same full-window attention. It is a between-arm difference, never a level;
  the docstring there says why, and what the number does NOT mean.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_NAN = float("nan")


def effective_rank(x: torch.Tensor, eps: float = 1e-9) -> float:
    """exp(entropy of the normalised singular value spectrum), mean-centred.

    Measures how many independent DIRECTIONS OF VARIATION the representation
    uses. Because it centres first, it is blind to the degenerate case where
    every input maps to the same constant vector: after centring that is pure
    isotropic noise and scores as full rank. `dimension_stats` is what catches
    that case, via near-zero per-dimension standard deviation.

    Report both. Neither catches martingale collapse; see copy_diagnostics.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] > 4096:
        x = x[torch.randperm(x.shape[0])[:4096]]
    sv = torch.linalg.svdvals(x)
    p = sv / (sv.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def dimension_stats(x: torch.Tensor) -> Dict[str, float]:
    flat = x.reshape(-1, x.shape[-1]).float()
    std = flat.std(dim=0)
    return {
        "mean_dim_std": float(std.mean()),
        "min_dim_std": float(std.min()),
        "dead_dims": float((std < 1e-3).float().mean()),
    }


def nearest_visible_reference(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Copy oracle. -> (B, T, K, d), and a validity flag via `has_reference`.

    For each (b, t, k) returns the target embedding of the nearest visible bucket
    of the SAME market, preferring the most recent one at or before t.
    """
    b, t, k = mask.shape
    d = target.shape[-1]
    tgt = target.reshape(b, t, k, d)
    visible = ~mask

    # cummax/cummin have no MPS kernel as of torch 2.6, and this is a cheap
    # diagnostic, so the index arithmetic runs on CPU and moves back.
    vis_cpu = visible.cpu()
    idx = torch.arange(t).view(1, t, 1).expand(b, t, k)

    back = torch.where(vis_cpu, idx, torch.full_like(idx, -1))
    last_vis = torch.cummax(back, dim=1).values                       # -1 if none yet

    fwd = torch.where(vis_cpu, idx, torch.full_like(idx, t))
    next_vis = torch.flip(torch.cummin(torch.flip(fwd, [1]), dim=1).values, [1])

    ref_cpu = torch.where(last_vis >= 0, last_vis, next_vis)
    valid = ((last_vis >= 0) | (next_vis < t)).to(mask.device)
    ref = ref_cpu.clamp(0, t - 1).to(mask.device)

    gathered = torch.gather(tgt, 1, ref.unsqueeze(-1).expand(b, t, k, d))
    return gathered, valid


def copy_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Model loss vs copy-oracle loss on the same masked positions."""
    b, t, k = mask.shape
    d = target.shape[-1]
    pred4 = pred.reshape(b, t, k, d)
    tgt4 = target.reshape(b, t, k, d)

    ref, valid = nearest_visible_reference(target, mask)
    sel = mask & valid
    if sel.sum() == 0:
        return {"copy_loss_ratio": float("nan"), "copy_alignment": float("nan"),
                "model_loss_on_copyable": float("nan"), "copyable_frac": 0.0}

    s = sel.unsqueeze(-1).float()
    n = s.sum() * d

    model_l = (F.smooth_l1_loss(pred4, tgt4, beta=beta, reduction="none") * s).sum() / n
    copy_l = (F.smooth_l1_loss(ref, tgt4, beta=beta, reduction="none") * s).sum() / n

    align = F.cosine_similarity(pred4, ref, dim=-1)
    align = float((align * sel.float()).sum() / sel.float().sum().clamp(min=1))

    return {
        "copy_loss_ratio": float(model_l / copy_l.clamp(min=1e-9)),
        "copy_alignment": align,
        "model_loss_on_copyable": float(model_l),
        "copyable_frac": float(sel.float().sum() / mask.float().sum().clamp(min=1)),
    }


# ---------------------------------------------------------------------------
# directional copy oracle
# ---------------------------------------------------------------------------


def directional_references(target: torch.Tensor, mask: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (back_ref, back_valid, fwd_ref, fwd_valid), each over (B, T, K[, d]).

    `nearest_visible_reference` collapses both directions into one oracle and
    prefers the past, which makes it impossible to tell an extrapolating model
    from an interpolating one: under middle-of-window masking almost every
    masked position has a past reference, so the forward side never shows up in
    the number. These two references are kept separate for exactly that reason.

    Both lookups are STRICT. The exclusive shift matters even though we only
    score masked positions, where a self-reference cannot occur anyway: without
    it the same helper would silently return the position itself when reused on
    visible positions, and the resulting oracle loss of zero would read as a
    perfect copy solution rather than as a bug.

    cummax/cummin still have no MPS kernel as of torch 2.6, so the index
    arithmetic runs on CPU and the results move back, same as the ported oracle.
    """
    b, t, k = mask.shape
    d = target.shape[-1]
    tgt = target.reshape(b, t, k, d)
    visible = ~mask

    vis_cpu = visible.cpu()
    idx = torch.arange(t).view(1, t, 1).expand(b, t, k)

    back = torch.where(vis_cpu, idx, torch.full_like(idx, -1))
    last_vis = torch.cummax(back, dim=1).values
    prev_vis = torch.cat(
        [torch.full_like(last_vis[:, :1], -1), last_vis[:, :-1]], dim=1)   # strictly before

    fwd = torch.where(vis_cpu, idx, torch.full_like(idx, t))
    next_vis = torch.flip(torch.cummin(torch.flip(fwd, [1]), dim=1).values, [1])
    succ_vis = torch.cat(
        [next_vis[:, 1:], torch.full_like(next_vis[:, :1], t)], dim=1)     # strictly after

    back_valid = (prev_vis >= 0).to(mask.device)
    fwd_valid = (succ_vis < t).to(mask.device)
    back_idx = prev_vis.clamp(0, t - 1).to(mask.device)
    fwd_idx = succ_vis.clamp(0, t - 1).to(mask.device)

    back_ref = torch.gather(tgt, 1, back_idx.unsqueeze(-1).expand(b, t, k, d))
    fwd_ref = torch.gather(tgt, 1, fwd_idx.unsqueeze(-1).expand(b, t, k, d))
    return back_ref, back_valid, fwd_ref, fwd_valid


def _oracle_scores(pred4, tgt4, ref, sel, beta):
    """-> (model_loss, oracle_loss, alignment) over `sel`, or nans if `sel` is empty."""
    n_sel = sel.sum()
    if n_sel == 0:
        return _NAN, _NAN, _NAN
    d = tgt4.shape[-1]
    s = sel.unsqueeze(-1).float()
    n = s.sum() * d
    model_l = (F.smooth_l1_loss(pred4, tgt4, beta=beta, reduction="none") * s).sum() / n
    oracle_l = (F.smooth_l1_loss(ref, tgt4, beta=beta, reduction="none") * s).sum() / n
    align = F.cosine_similarity(pred4, ref, dim=-1)
    align = (align * sel.float()).sum() / sel.float().sum().clamp(min=1)
    return float(model_l), float(oracle_l), float(align)


def directional_copy_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Split the copy oracle by DIRECTION. Returns:

        extrap_alignment, extrap_loss_ratio, extrap_frac   # nearest visible STRICTLY BEFORE
        interp_alignment, interp_loss_ratio, interp_frac   # nearest visible AFTER (when one exists)
        interp_advantage   # extrap_loss - interp_loss, > 0 means interpolation is the easier solution

    Why this exists: `copy_diagnostics` reports one ratio, and under
    `interpolation` masking that ratio cannot distinguish a model that has
    learned the martingale from one that is quietly reading the future through a
    bidirectional target encoder. Splitting the oracle makes the second case
    visible as a positive `interp_advantage`, and H2 predicts that number falls
    towards zero once the target encoder is causal.

    `interp_advantage` is deliberately computed on the PAIRED subset of masked
    positions where both references exist, not as the difference of the two
    headline losses. The two direction sets are not the same positions: under
    `temporal` masking every masked patch has a past reference and none has a
    future one, so differencing unpaired means would compare the last patch of
    the window against nothing and report a number driven entirely by which
    positions happened to qualify. On the paired subset the comparison is
    within-position and the sign is interpretable. `paired_frac` says how much
    of the mask that subset covers; when it is 0.0 the advantage is nan by
    construction, which is the honest answer for one-sided masking.

    Extra keys beyond the contract (`extrap_oracle_loss`, `interp_oracle_loss`,
    `paired_frac`, `model_loss_paired`) are reported so a suspicious ratio can be
    traced to its numerator or its denominator without a rerun.
    """
    b, t, k = mask.shape
    d = target.shape[-1]
    pred4 = pred.reshape(b, t, k, d)
    tgt4 = target.reshape(b, t, k, d)

    out = {
        "extrap_alignment": _NAN, "extrap_loss_ratio": _NAN, "extrap_frac": 0.0,
        "interp_alignment": _NAN, "interp_loss_ratio": _NAN, "interp_frac": 0.0,
        "interp_advantage": _NAN,
        "extrap_oracle_loss": _NAN, "interp_oracle_loss": _NAN,
        "model_loss_paired": _NAN, "paired_frac": 0.0,
    }
    n_masked = mask.sum()
    if n_masked == 0:
        return out

    back_ref, back_valid, fwd_ref, fwd_valid = directional_references(target, mask)
    back_sel = mask & back_valid
    fwd_sel = mask & fwd_valid
    denom = mask.float().sum().clamp(min=1)

    model_l, oracle_l, align = _oracle_scores(pred4, tgt4, back_ref, back_sel, beta)
    out["extrap_alignment"] = align
    out["extrap_oracle_loss"] = oracle_l
    out["extrap_frac"] = float(back_sel.float().sum() / denom)
    if oracle_l == oracle_l:
        out["extrap_loss_ratio"] = float(model_l / max(oracle_l, 1e-9))

    model_l, oracle_l, align = _oracle_scores(pred4, tgt4, fwd_ref, fwd_sel, beta)
    out["interp_alignment"] = align
    out["interp_oracle_loss"] = oracle_l
    out["interp_frac"] = float(fwd_sel.float().sum() / denom)
    if oracle_l == oracle_l:
        out["interp_loss_ratio"] = float(model_l / max(oracle_l, 1e-9))

    both = back_sel & fwd_sel
    out["paired_frac"] = float(both.float().sum() / denom)
    if both.sum() > 0:
        m_p, back_l, _ = _oracle_scores(pred4, tgt4, back_ref, both, beta)
        _, fwd_l, _ = _oracle_scores(pred4, tgt4, fwd_ref, both, beta)
        out["model_loss_paired"] = m_p
        out["interp_advantage"] = float(back_l - fwd_l)
    return out


# ---------------------------------------------------------------------------
# target-side leakage (H1)
# ---------------------------------------------------------------------------


def target_autocorrelation(target: torch.Tensor, patch_mask: torch.Tensor,
                           max_lag: int = 5) -> Dict[str, float]:
    """Mean cosine similarity between target embeddings at patch lag L, same
    strike. Key H1 measurement: a bidirectional target encoder inflates this
    across the mask boundary. Returns {"lag_1": ..., ..., "cross_boundary": ...}
    where cross_boundary averages pairs straddling the visible/masked split.

    HOW TO READ THIS, or H1 is not falsifiable. Every key here is a BETWEEN-ARM
    statistic. Compare `lag_L` and `cross_boundary` for the bidirectional and
    causal target encoders at the SAME masking strategy, seed and batch, and
    report the difference. Never read the sign or size of one arm alone, and
    never compare across masking strategies. Straddling pairs are systematically
    further apart in time than same-side pairs, so `cross_boundary` moves when
    the hole moves; under `temporal` masking the straddling set has mean lag
    about 2.8 while the same-side set is dominated by lag 1. Holding the mask
    fixed makes the pair set identical in both arms, and every such confound
    then cancels in the difference. That, not the absolute level, is H1.

    The effect is small and only exists in a TRAINED encoder. Measured on
    pm-jepa's own PatchTST target with `interpolation` masking (see
    `scripts/verify_h1_measurable.py`), varying how much the attention sublayers
    mix across patches, `bidir - causal` at lag 3 goes +0.0002 at random init,
    +0.0141 at 16x mixing, +0.0295 at 64x. At random init, zeroing patches 4 and
    5 moves patch 3's target embedding by 0.005 of its norm: a pre-norm residual
    stream starts near the identity, so there is almost no leakage to detect
    before training. An H1 null measured on an undertrained target encoder is
    a null about the encoder, not about causality.

    `boundary_excess` is NOT the H1 statistic, and it is here to prove which
    leakage mechanism is operating. It is the lag-controlled contrast between
    straddling and same-side pairs: at each lag it differences the two, then
    averages over lags weighted by the straddling pair counts, dropping lags
    where either pair type is missing. Because pm-jepa's target encoder is
    MASK-BLIND (it is called with `strike_mask=None`, so it does not know where
    the hole is), bidirectional leakage inflates every pair in the window by
    roughly the same amount, and this contrast cancels it: measured across the
    same mixing sweep, `boundary_excess` separates the arms by +0.00035 at 64x
    mixing while `cross_boundary` separates them by +0.0155. Expect it near zero
    in BOTH arms; that is the mask-blind signature, not a failed H1. If it ever
    does move between arms, the leakage is boundary-specific rather than
    window-wide, which would mean the target encoder is seeing the mask and the
    experiment is measuring something other than what H1 describes.

    Its null is not zero either, so read only the between-arm difference. On a
    synthetic zero-started random walk with no leakage planted at all,
    `boundary_excess` reads +0.002 under `interpolation` masking and +0.134
    under `temporal` masking, purely because a walk is not stationary in the
    patch index and a deterministic hole makes late-window pairs
    disproportionately straddling.

    `boundary_excess_matched` removes that position confound by contrasting
    straddling against same-side instances of the SAME (p, q) pair. It is
    zero-referenced, but it only exists when the mask varies across samples or
    strikes; under `temporal` and `interpolation` the mask is identical for
    every (b, k), so a given pair is either always straddling or never, and this
    key is nan. That is the honest answer: for the deterministic arms there is
    no in-sample control and the causal arm is the only available null, which is
    exactly why the experiment is a 2x2 rather than one run.

    `baseline_cos` is the floor: mean cosine between the same (patch, strike)
    slot in different windows of the batch. Transformer embeddings are anisotropic
    and routinely sit at cosine 0.9 to everything, so a `lag_1` of 0.95 means
    nothing until you know the floor. If `baseline_cos` is within noise of the
    lag figures, the encoder has collapsed and none of these numbers are
    measuring leakage; a rising `lag_L` alongside a rising `baseline_cos` is
    collapse, not leakage, and only the gap between them is evidence for H1.

    Nan-safe on every degenerate input: fewer than two patches, an all-visible
    or all-masked grid, or a batch of one all return nan for the affected keys
    rather than raising, because this runs inside the training loop and a mask
    that happens to be one-sided for one step must not kill the run.
    """
    b, p, k = patch_mask.shape
    d = target.shape[-1]
    tgt = target.reshape(b, p, k, d).float()
    max_lag = max(1, int(max_lag))

    out: Dict[str, float] = {"lag_{}".format(L): _NAN for L in range(1, max_lag + 1)}
    out["cross_boundary"] = _NAN
    out["within_side"] = _NAN
    out["boundary_excess"] = _NAN
    out["boundary_excess_matched"] = _NAN
    out["cross_boundary_pairs"] = 0.0
    out["baseline_cos"] = _NAN

    if b > 1:
        # Same slot, different window. Rolling by one is a cheap unbiased-enough
        # pairing; it needs no RNG, so the floor is reproducible across runs.
        floor = F.cosine_similarity(tgt, tgt.roll(1, dims=0), dim=-1)
        out["baseline_cos"] = float(floor.mean())

    if p < 2:
        return out

    pm = patch_mask.to(tgt.device)
    cross_sum, cross_n = {}, {}
    same_sum, same_n = {}, {}
    matched_num, matched_w = 0.0, 0.0

    # P is 6 here, so 15 pair slices; the explicit loop costs nothing and keeps
    # the per-pair control (which a vectorised lag-diagonal cannot express).
    for lag in range(1, min(max_lag, p - 1) + 1):
        lag_sum, lag_n = 0.0, 0.0
        cross_sum[lag] = cross_n[lag] = same_sum[lag] = same_n[lag] = 0.0
        for i in range(p - lag):
            j = i + lag
            cos = F.cosine_similarity(tgt[:, i], tgt[:, j], dim=-1)      # (b, k)
            st = (pm[:, i] != pm[:, j]).float()                          # (b, k)
            n_s = float(st.sum())
            n_o = float(st.numel() - n_s)
            s_sum = float((cos * st).sum())
            o_sum = float((cos * (1.0 - st)).sum())

            lag_sum += s_sum + o_sum
            lag_n += float(cos.numel())
            cross_sum[lag] += s_sum
            cross_n[lag] += n_s
            same_sum[lag] += o_sum
            same_n[lag] += n_o

            # Pair-matched contrast: same (i, j), so lag AND patch position are
            # both held fixed and only boundary membership varies.
            if n_s > 0 and n_o > 0:
                matched_num += n_s * (s_sum / n_s - o_sum / n_o)
                matched_w += n_s

        if lag_n > 0:
            out["lag_{}".format(lag)] = lag_sum / lag_n

    tot_cross_n = sum(cross_n.values())
    out["cross_boundary_pairs"] = tot_cross_n
    if tot_cross_n > 0:
        out["cross_boundary"] = sum(cross_sum.values()) / tot_cross_n

    # Lag-matched contrast. Weight each lag by how many straddling pairs it
    # contributes, so the control set has the straddling set's lag profile.
    lags = [L for L in cross_n if cross_n[L] > 0 and same_n[L] > 0]
    w = sum(cross_n[L] for L in lags)
    if w > 0:
        cross_m = sum(cross_sum[L] for L in lags) / w
        within_m = sum(cross_n[L] * (same_sum[L] / same_n[L]) for L in lags) / w
        out["within_side"] = within_m
        out["boundary_excess"] = cross_m - within_m
    if matched_w > 0:
        out["boundary_excess_matched"] = matched_num / matched_w
    return out
