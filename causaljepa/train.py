#!/usr/bin/env python3
"""Training loop for one cell of the causal 2x2.

Everything here that is not the causality flag is a line-for-line copy of
`pm-jepa/train.py`: AdamW lr=1e-3 wd=0.02 betas=(0.9,0.95), a 100-step linear
warmup multiplied by a cosine decay, grad clip 1.0, and an EMA momentum
cosine-ramped from 0.996 up to a 0.9999 cap. That parity is the whole point. If
the optimiser or the schedule drifted, a difference between our arms and
pm-jepa's published numbers would be uninterpretable: we could not say whether
causality moved the metric or whether a different warmup did.

Two deliberate departures from pm-jepa, both documented at the point of use:

  1. No SIGReg. In pm-jepa the regulariser was evaluated on `out["target"]`,
     which is detached, so `lam * sigreg` had no path to any parameter and
     contributed exactly zero gradient in every run they published. The loss
     here is plain smooth-L1 on masked positions and nothing else.

  2. `reg_grad_norm` is logged at every log step. It is 0.0 while there is no
     regulariser, and it is computed by actually differentiating the
     regulariser term against the trainable parameters when one exists. This is
     the tripwire for failure (1): a regulariser that is inert reports 0.0 in
     the history, in every record, and can no longer hide behind a loss curve
     that looks plausible because the prediction term is doing all the work.

Diagnostics run on CPU copies of the forward pass. `torch.linalg.svdvals` has no
MPS kernel in torch 2.6 and `cummax` inside the copy oracle does not either, so
half the diagnostic block would hop to CPU regardless; doing it uniformly keeps
the numbers identical across devices and keeps device-specific fallbacks out of
the diagnostic code.
"""
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from causaljepa.diagnostics import (
    copy_diagnostics,
    dimension_stats,
    directional_copy_diagnostics,
    effective_rank,
    target_autocorrelation,
)
from causaljepa.masking import sample_mask
from causaljepa.model import CausalJEPA

WARMUP_STEPS = 100
EMA_BASE = 0.996
EMA_CAP = 0.9999

# SPEC NOTE (section 8): `lam` is in the mandated `train_arm` signature, but the
# same section forbids porting SIGReg, so there is no regulariser for `lam` to
# weight and it cannot change any gradient. Implemented as written; kept because
# dropping it would break config parity with pm-jepa's results files. The
# startup line and `reg_grad_norm` together make its inertness visible rather
# than implicit.

# SPEC NOTE (section 10): the SPEC says Python 3.11+, but the only interpreter
# on this machine with torch 2.6 installed is CPython 3.9.6, where PEP 604
# annotations (`Tensor | None`) raise TypeError at def time. This module sticks
# to `typing.Optional` so it imports on the interpreter that actually exists.


def pick_device(req: str = "auto") -> str:
    if req != "auto":
        return req
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lr_multiplier(step: int, total_steps: int) -> float:
    """LambdaLR multiplier: 100-step linear warmup times cosine decay.

    Transcribed from pm-jepa/train.py rather than rewritten, including the
    `(s + 1) / 100` off-by-one that makes the first step run at 0.01x rather
    than 0.0x. Tidying that would silently change the first hundred steps and
    break comparability with their published arms.
    """
    return (min(1.0, (step + 1) / WARMUP_STEPS)
            * 0.5 * (1 + np.cos(np.pi * min(step / total_steps, 1.0))))


def ema_momentum(step: int, total_steps: int,
                 base: float = EMA_BASE, cap: float = EMA_CAP) -> float:
    """Target-encoder momentum, cosine-ramped from `base` to `cap`.

    Also transcribed verbatim. Note the ramp is driven by the raw `step`, not by
    the scheduler's counter, and that the cap binds only at the very end; both
    details are pm-jepa's and both are load-bearing for parity.
    """
    m = 1.0 - (1.0 - base) * (0.5 * (1 + np.cos(np.pi * step / total_steps)))
    return float(min(m, cap))


def jepa_loss(pred: torch.Tensor, target: torch.Tensor, mask_flat: torch.Tensor,
              beta: float = 1.0) -> Dict[str, Any]:
    """Smooth-L1 on masked positions, normalised the way pm-jepa normalises it.

    The denominator is (number of masked tokens) x (embedding width), not the
    element count of the full tensor, so the loss scale does not move when the
    masking strategy changes the masked fraction. Arms with different masking
    ratios would otherwise be reported on different scales.

    `reg` is returned as None to say explicitly that there is no regulariser,
    rather than returning a zero tensor that would be indistinguishable from a
    regulariser that happens to evaluate to zero.
    """
    sel = mask_flat.unsqueeze(-1)
    n = mask_flat.sum().clamp(min=1)
    err = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    pred_loss = (err * sel).sum() / (n * pred.shape[-1])
    return {"loss": pred_loss, "pred_loss": pred_loss.detach(), "reg": None}


def regulariser_grad_norm(reg: Optional[torch.Tensor],
                          params: Sequence[torch.Tensor]) -> float:
    """L2 norm of the gradient the regulariser contributes to the parameters.

    Call this BEFORE `backward()`; it uses `torch.autograd.grad` with
    `retain_graph=True`, so it reads the graph without touching `.grad` and the
    subsequent backward is unaffected.

    Returns 0.0 for the two ways a regulariser can be inert: absent, or built
    from tensors detached from the parameter graph. The second is what happened
    in pm-jepa, where SIGReg was applied to the detached target. `allow_unused`
    is on because a live regulariser may legitimately touch only a subset of
    parameters; a None gradient there contributes nothing to the norm, which is
    correct, whereas raising would be wrong.
    """
    if reg is None:
        return 0.0
    if not (torch.is_tensor(reg) and reg.requires_grad):
        return 0.0
    grads = torch.autograd.grad(reg, list(params), retain_graph=True,
                                allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum())
    return float(total ** 0.5)


def _merge(rec: Dict[str, float], extra: Dict[str, float], source: str) -> None:
    """Flatten a diagnostic dict into the history record, refusing collisions.

    A silently overwritten key is the same class of bug as the inert
    regulariser: the history keeps reporting a plausible number that is not the
    number you think it is. Two diagnostics sharing a key name is a defect in
    the diagnostics module, so fail loudly at the first log step rather than
    after an overnight sweep.

    SPEC NOTE (section 8): the history is required to carry every key from all
    four diagnostics in one flat record, so `target_autocorrelation`'s generic
    `lag_1..lag_5` land unprefixed next to the copy metrics. That is what makes
    a collision plausible in the first place, hence this guard rather than a
    plain `dict.update`.
    """
    clash = sorted(set(extra) & set(rec))
    if clash:
        raise ValueError("{} would overwrite history keys {}".format(source, clash))
    rec.update(extra)


def train_arm(X: np.ndarray, tr: np.ndarray, *, strategy: str,
              causal_context: bool, causal_target: bool, device: str,
              steps: int, batch: int, lr: float, lam: float, d_model: int,
              n_layers: int, n_held_out: int, seed: int, log_every: int,
              verbose: bool = True) -> Tuple[nn.Module, List[dict], float]:
    """Train one arm; returns (model, history, seconds).

    `lam` is accepted for signature and config parity with pm-jepa and is
    currently INERT: there is no regulariser to weight. It is not quietly
    dropped, it is printed at startup and every history record carries
    `reg_grad_norm == 0.0` so the reader can see that nothing is being
    regularised. Passing a nonzero `lam` will not change a single gradient.

    History records, at every log step: step, pred_loss, eff_rank,
    reg_grad_norm, every key from copy_diagnostics, every key from
    directional_copy_diagnostics, every key from target_autocorrelation, and
    dimension_stats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, T, K, C = X.shape
    model = CausalJEPA(T, K, C, d_model=d_model, n_layers=n_layers,
                       causal_context=causal_context,
                       causal_target=causal_target).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.02, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, steps))

    idx_pool = np.flatnonzero(tr)
    Xt = torch.from_numpy(X)
    gen = torch.Generator().manual_seed(seed + 1)
    hist, t0 = [], time.time()

    tag = "{}|ctx={}|tgt={}".format(strategy,
                                    "causal" if causal_context else "bidir",
                                    "causal" if causal_target else "bidir")
    if verbose:
        print("   [{}] no regulariser; lam={} is inert, reg_grad_norm will be 0.0"
              .format(tag, lam), flush=True)
    if steps < WARMUP_STEPS:
        # A short run is not a small version of a long run, and the difference
        # is large enough to be mistaken for a null result. The warmup length is
        # a fixed 100 steps (pm-jepa's, kept for parity) while the cosine decay
        # is scaled to `steps`, so at steps=30 the two overlap and the learning
        # rate never exceeds 0.085x its nominal value; at steps=800 it reaches
        # 0.96x. Every arm then finishes near its initialisation, where the
        # pre-norm residual stream barely mixes across patches, so the causal and
        # bidirectional cells produce nearly identical diagnostics. That reads as
        # "causality changes nothing", which is a statement about the schedule
        # and not about H1. Warn rather than refuse; smoke runs want this.
        print("   [{}] WARNING: steps={} is inside the {}-step warmup, peak lr is "
              "{:.3f}x nominal. Diagnostics from this run measure an untrained "
              "encoder and must not be read as an H1 or H2 result.".format(
                  tag, steps, WARMUP_STEPS,
                  max(lr_multiplier(s, steps) for s in range(steps))), flush=True)

    for step in range(steps):
        sel = torch.from_numpy(np.random.choice(idx_pool, batch, replace=False))
        obs = Xt[sel].to(device)
        # `patch_length` is threaded from the model rather than left to the
        # sampler's default. Only `interpolation` reads it, and it reads it to
        # align the hole to patch boundaries. The two currently agree at 4, so a
        # drift would not raise: it would move the hole off the grid, and
        # `patch_mask`'s any() rule would widen it to the neighbouring patches,
        # turning the interpolation arm into a wider temporal one with no symptom
        # beyond a shifted loss.
        mk = sample_mask(batch, T, K, strategy, generator=gen,
                         n_held_out=n_held_out, device="cpu",
                         patch_length=model.patch_length).to(device)
        out = model(obs, mk)
        loss = jepa_loss(out["pred"], out["target"], out["mask_flat"])

        logging_now = (step + 1) % log_every == 0 or step == 0
        reg_gn = regulariser_grad_norm(loss["reg"], params) if logging_now else 0.0

        opt.zero_grad(set_to_none=True)
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target(momentum=ema_momentum(step, steps))

        if logging_now:
            with torch.no_grad():
                pred_c = out["pred"].detach().cpu()
                tgt_c = out["target"].detach().cpu()
                ctx_c = out["context"].detach().cpu()
                pm_c = out["patch_mask"].detach().cpu()

                # `lr_mult` and `ema_m` are recorded, not just applied. A run
                # whose diagnostics came out flat is ambiguous between "causality
                # does nothing" and "this run never left its initialisation", and
                # the schedule is the only thing that tells them apart. Without
                # it in the artefact, that has to be reconstructed from the step
                # count and a reading of this file.
                rec = {"step": step + 1,
                       "pred_loss": float(loss["pred_loss"]),
                       "eff_rank": effective_rank(ctx_c),
                       "reg_grad_norm": reg_gn,
                       "lr_mult": lr_multiplier(step, steps),
                       "ema_m": ema_momentum(step, steps)}
                _merge(rec, copy_diagnostics(pred_c, tgt_c, pm_c), "copy_diagnostics")
                _merge(rec, directional_copy_diagnostics(pred_c, tgt_c, pm_c),
                       "directional_copy_diagnostics")
                _merge(rec, target_autocorrelation(tgt_c, pm_c), "target_autocorrelation")
                _merge(rec, dimension_stats(ctx_c), "dimension_stats")
            hist.append(rec)
            if verbose:
                # `.get` rather than `[]`: this line is cosmetic, and a renamed
                # diagnostic key should not kill a training run that has already
                # recorded the number correctly in `rec`.
                nan = float("nan")
                print("   [{}] {:>5}  pred {:.4f}  copy_align {:+.3f}  copy_ratio {:.3f}  "
                      "interp_adv {:+.4f}  xbound {:+.3f}  rank {:.1f}".format(
                          tag, rec["step"], rec["pred_loss"],
                          rec.get("copy_alignment", nan), rec.get("copy_loss_ratio", nan),
                          rec.get("interp_advantage", nan), rec.get("cross_boundary", nan),
                          rec["eff_rank"]),
                      flush=True)
    return model, hist, time.time() - t0


@torch.no_grad()
def represent_all(model: nn.Module, X: np.ndarray, device: str,
                  batch: int = 256) -> np.ndarray:
    """Frozen representations for the probe, in eval mode and in float64.

    Copied from pm-jepa so the probe sees exactly the same preprocessing. The
    float64 cast matters: `ridge_probe` solves a normal-equation system whose
    Gram matrix is badly conditioned at d_model=128, and float32 there moves
    R^2 in the third decimal, which is the resolution the 2x2 is being read at.
    """
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model.represent(torch.from_numpy(X[i:i + batch]).to(device))
                   .float().cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float64)
