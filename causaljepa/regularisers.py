"""Anti-collapse regularisers that are actually in the gradient.

This module exists because neither of the two regularisers this programme has
claimed to test was ever applied.

  pm-jepa    `model/sigreg.py:jepa_loss_sigreg` computes SIGReg on `target`,
             which `encoder.py:forward` builds under `torch.no_grad()` and
             returns `.detach()`ed. Verified: the total parameter-gradient norm
             is bit-identical at lam = 0, 0.05 and 100.0.
  slate-jepa `poc/slatejepa/losses.py` computes both VICReg terms on the same
             detached target; the variable is even named `ctx`, preserving the
             original intent. Verified: bit-identical at weights (0, 0),
             (0.04, 0.004) and (1000, 1000).

So the published claim that these mechanisms fail to prevent martingale collapse
was never tested. Everything here takes the ONLINE context representation, which
carries gradient, and every call site is expected to assert `reg_grad_norm > 0`
before the run counts for anything.

SIGReg is reimplemented from the LeJEPA paper (Balestriero & LeCun, arXiv
2511.08544) rather than vendored: the reference implementation is CC BY-NC 4.0,
which is not an OSI licence. Methods are not copyrightable. Cite the paper.
"""
from typing import Dict, Optional

import torch


def _epps_pulley(z: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """Epps-Pulley normality statistic per projection, in closed form.

    Compares the empirical characteristic function of standardised samples
    against the Gaussian one on a small frequency grid. Smooth and
    differentiable, unlike rank-based tests, which matters because this is a
    training loss rather than a diagnostic.

    z: (n_samples, n_proj), already standardised per projection. -> (n_proj,)
    """
    ts = torch.linspace(0.4, 2.0, n_freq, device=z.device, dtype=z.dtype)
    arg = z.unsqueeze(-1) * ts.view(1, 1, -1)
    re = torch.cos(arg).mean(dim=0)
    im = torch.sin(arg).mean(dim=0)
    target = torch.exp(-0.5 * ts.pow(2)).view(1, -1)
    w = torch.exp(-0.5 * ts.pow(2)).view(1, -1)
    return (((re - target) ** 2 + im ** 2) * w).sum(dim=-1)


def sigreg(x: torch.Tensor, n_proj: int = 256, n_freq: int = 8,
           generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """Sketched isotropic-Gaussian penalty. x: (..., d), flattened to (n, d).

    Scale is whitened per projection but the covariance is NOT whitened
    globally: full whitening would drive the penalty to zero trivially by
    hiding anisotropy in the transform, which is the opposite of the intent.

    Known limit, measured in pm-jepa and unchanged here: a uniform distribution
    scores as low as an isotropic Gaussian, because in high dimensions a random
    1-D projection of a uniform is near-Gaussian by CLT. Sketched normality is
    a weaker constraint than "the embedding is Gaussian" sounds.
    """
    x = x.reshape(-1, x.shape[-1])
    n, d = x.shape
    if n < 8:
        z = x.new_zeros(())
        return {"reg": z, "normality": z, "scale_dev": z, "mean_norm": z}

    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    dirs = torch.randn(d, n_proj, device=x.device, dtype=x.dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)

    proj = xc @ dirs
    sd = proj.std(dim=0, keepdim=True).clamp(min=1e-6)
    normality = _epps_pulley(proj / sd, n_freq=n_freq).mean()
    scale_dev = sd.squeeze(0).log().var()
    mean_norm = mu.pow(2).sum() / d

    return {"reg": normality + scale_dev + mean_norm,
            "normality": normality.detach(), "scale_dev": scale_dev.detach(),
            "mean_norm": mean_norm.detach()}


def vicreg(x: torch.Tensor, var_weight: float = 1.0,
           cov_weight: float = 0.1) -> Dict[str, torch.Tensor]:
    """VICReg variance and covariance terms. x: (..., d).

    Ported from slate-jepa's formulation, with the one change that matters:
    it is applied to whatever `x` you pass, and the call site passes the online
    context rather than the detached target.

    The relative weighting of the two terms is folded in here so a sweep has a
    single knob, matching SIGReg's single-lambda interface. slate-jepa used
    (0.04, 0.004), the same 10:1 ratio.
    """
    flat = x.reshape(-1, x.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)

    std = torch.sqrt(flat.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()

    m = flat.shape[0]
    cov = (flat.T @ flat) / max(m - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / flat.shape[-1]

    return {"reg": var_weight * var_loss + cov_weight * cov_loss,
            "var_loss": var_loss.detach(), "cov_loss": cov_loss.detach()}


def build(name: str, x: torch.Tensor,
          generator: Optional[torch.Generator] = None) -> Optional[Dict[str, torch.Tensor]]:
    """Dispatch. `name` in {none, sigreg, vicreg}.

    Returns None for "none" rather than a zero tensor, so that an absent
    regulariser is distinguishable from one that merely evaluates to zero.
    `regulariser_grad_norm` relies on that distinction.
    """
    if name in (None, "none", ""):
        return None
    if name == "sigreg":
        return sigreg(x, generator=generator)
    if name == "vicreg":
        return vicreg(x)
    raise ValueError("unknown regulariser: {}".format(name))
