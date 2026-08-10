"""Factorised (strike x time) transformer, and the JEPA built on top of it.

pm-jepa's encoder is PatchTST with `channel_attention=True`, which flattens the
(strike x patch) grid into one sequence and attends over all of it at once.
That is a fine bidirectional encoder and a useless causal one: there is no
sublayer whose sequence axis is time alone, so a causal mask over that flat
sequence would also forbid a strike from seeing its siblings at the same
instant. That is not the constraint we want. The strike ladder is a
cross-section, not a sequence; only the time axis has a direction.

So attention is split. Each block attends twice:

    strike attention   within one patch, across all K strikes, ALWAYS
                       bidirectional. This is the sublayer that does the work
                       of inferring a held-out strike from its siblings, which
                       is the whole point of slate masking.
    time attention     within one strike, across all P patches, causal or
                       bidirectional per `causal`.

Two properties fall out. The time axis becomes maskable, which is the H1/H2
intervention, and it becomes cacheable, because a causal time sublayer is the
only thing in the block that reads backwards (see cache.py).

Axis order is PATCH first, STRIKE second, matching pm-jepa's `_tokens()`.
Returning (B, K, P, d) instead silently inverts every ported diagnostic:
`copy_diagnostics` would read temporal masking as 0% copyable and slate masking
as 100%, exactly backwards, and nothing would raise.
"""
import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .cache import KVCache


class _MultiHeadAttention(nn.Module):
    """Plain multi-head attention, written out rather than wrapped.

    `nn.MultiheadAttention` and `nn.TransformerEncoder` do not hand back the
    keys and values they computed, and the whole cache story in cache.py needs
    exactly those. Attention is also computed with an explicit matmul/softmax
    instead of `scaled_dot_product_attention` so that the full and incremental
    paths run the same arithmetic: a fused kernel is free to pick a different
    algorithm for a length-1 query than for a length-6 one, and the exactness
    bar is atol=1e-4, not "close enough".
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model {} not divisible by n_heads {}".format(
                d_model, n_heads))
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        s, l, _ = x.shape
        return x.view(s, l, self.h, self.dh).transpose(1, 2)      # (S, h, L, dh)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
                past: Optional[Tuple[Tensor, Tensor]] = None,
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """x (S, L, d) where S folds every axis that is not the sequence axis.

        `past` is the (k, v) already seen on this sequence; the returned k, v
        are the concatenation, i.e. what the caller should store back. Returning
        them unconditionally, even when there is no cache, keeps one code path
        for both modes so the cached one cannot quietly diverge.
        """
        s, l, d = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)

        scores = (q @ k.transpose(-2, -1)) * self.scale            # (S, h, L, L_kv)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(s, l, d)
        return self.out_proj(out), k, v


class _FactorisedBlock(nn.Module):
    """strike attention -> time attention -> FFN, all pre-norm residual.

    Strike before time is deliberate. The strike sublayer mixes only within a
    patch, so whatever order the patches arrive in, its output for a given patch
    is identical; that is what lets the cache store the time sublayer's K/V and
    still be exact. Putting time first would work too, but then a later patch
    would enter the time sublayer before it had seen its own cross-section, and
    the block would be doing cross-strike inference on a stale representation.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm_strike = nn.LayerNorm(d_model)
        self.strike_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm_time = nn.LayerNorm(d_model)
        self.time_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def forward(self, x: Tensor, time_mask: Optional[Tensor] = None,
                past: Optional[Tuple[Tensor, Tensor]] = None,
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """x (B, P, K, d) -> same, plus the time sublayer's K/V for the cache."""
        b, p, k, d = x.shape

        h = self.norm_strike(x).reshape(b * p, k, d)               # sequence = strikes
        a, _, _ = self.strike_attn(h)
        x = x + a.reshape(b, p, k, d)

        h = self.norm_time(x).permute(0, 2, 1, 3).reshape(b * k, p, d)   # sequence = patches
        a, new_k, new_v = self.time_attn(h, time_mask, past)
        x = x + a.reshape(b, k, p, d).permute(0, 2, 1, 3)

        x = x + self.ffn(self.norm_ffn(x))
        return x, new_k, new_v


class FactorisedEncoder(nn.Module):
    """Patchify (B,T,K,C) into (B,P,K,d) and run alternating strike/time attention."""

    def __init__(self, n_strikes: int, n_features: int, n_patches: int,
                 patch_length: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 8, causal: bool = False,
                 patch_stride: Optional[int] = None):
        """SPEC NOTE: section 2 fixes the signature through `causal`.
        `patch_stride` is appended as an optional trailing keyword because the
        constructor otherwise has no way to learn it, and the patch grid is not
        recoverable from (n_patches, patch_length) alone once stride and length
        differ. It defaults to `patch_length`, the non-overlapping case, so
        every positional call in the SPEC keeps its meaning.
        """
        super().__init__()
        self.K, self.C = n_strikes, n_features
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.patch_stride = patch_length if patch_stride is None else patch_stride
        self.d_model = d_model
        self.n_layers = n_layers
        self.causal = causal

        self.embed = nn.Linear(patch_length * n_features, d_model)
        self.pos_patch = nn.Parameter(torch.zeros(n_patches, d_model))
        self.pos_strike = nn.Parameter(torch.zeros(n_strikes, d_model))
        nn.init.normal_(self.pos_patch, std=0.02)
        nn.init.normal_(self.pos_strike, std=0.02)

        self.blocks = nn.ModuleList([
            _FactorisedBlock(d_model, n_heads, 2 * d_model) for _ in range(n_layers)])
        # Final norm lives inside the encoder rather than beside it as in
        # pm-jepa, so that EMA-copying the target tower cannot forget it.
        self.out_norm = nn.LayerNorm(d_model)
        self._mask_cache: Dict[tuple, Tensor] = {}

    def _patchify(self, obs: Tensor) -> Tensor:
        """(B,T,K,C) -> (B,P,K,patch_length*C), patch p covering minutes
        `p*stride .. p*stride + patch_length - 1`.

        Minutes past the last whole patch are dropped rather than padded: a
        half-filled patch would be a token whose statistics differ from every
        other token, and the encoder has no way to signal that.
        """
        b, t, k, c = obs.shape
        if c != self.C or k != self.K:
            raise ValueError("expected (B,T,{},{}) got {}".format(
                self.K, self.C, tuple(obs.shape)))
        if t < self.patch_length:
            raise ValueError("window of {} minutes is shorter than one patch ({})".format(
                t, self.patch_length))
        # unfold gives (B, P, K, C, patch_length); features vary slowest so the
        # embedding sees each strike's C features laid out per minute.
        u = obs.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        b, p, k, c, pl = u.shape
        return u.reshape(b, p, k, c * pl)

    def _time_mask(self, p: int, offset: int, device) -> Tensor:
        """Bool (1, 1, p, offset+p), True where attention is FORBIDDEN.

        Query i is the absolute patch `offset + i`, so it may read keys 0 ..
        offset+i. With an empty cache offset is 0 and this is the ordinary
        lower-triangular mask; the offset form is what makes a cached step read
        its own history and no further.
        """
        key = (p, offset, str(device))
        m = self._mask_cache.get(key)
        if m is None or m.device != torch.device(device):
            q = torch.arange(p, device=device).unsqueeze(1) + offset
            kk = torch.arange(offset + p, device=device).unsqueeze(0)
            m = (kk > q).view(1, 1, p, offset + p)
            self._mask_cache[key] = m
        return m

    def forward(self, obs: Tensor, strike_mask: Optional[Tensor] = None,
                cache: Optional[KVCache] = None) -> Tensor:
        """obs (B,T,K,C); strike_mask (B,T,K) bool, True = hidden -> (B,P,K,d).

        Patch-first, strike-second axis order. This matches pm-jepa's
        `_tokens()` convention and every ported diagnostic assumes it.

        When `cache` is given, `obs` is read as the minutes of the NEW patches
        only and positions are offset by `cache.n_seen`. Go through
        `causaljepa.cache.encode_incremental` rather than calling this form
        directly; it is the piece that works out which minutes are actually new,
        and getting that wrong shifts the positional embedding without raising.
        """
        if strike_mask is not None:
            obs = obs.masked_fill(strike_mask.unsqueeze(-1), 0.0)

        x = self.embed(self._patchify(obs))                        # (B,P,K,d)
        p = x.shape[1]
        offset = 0
        if cache is not None:
            if not self.causal:
                raise ValueError(
                    "a KV cache is only sound for a causal encoder; bidirectional "
                    "time attention rewrites earlier tokens when a patch arrives")
            offset = cache.n_seen
        if offset + p > self.n_patches:
            raise ValueError("{} patches at offset {} overruns the {}-patch grid".format(
                p, offset, self.n_patches))

        x = (x
             + self.pos_patch[offset:offset + p].view(1, p, 1, self.d_model)
             + self.pos_strike.view(1, 1, self.K, self.d_model))

        time_mask = self._time_mask(p, offset, x.device) if self.causal else None
        for i, block in enumerate(self.blocks):
            past = None
            if cache is not None and i < len(cache.k):
                past = (cache.k[i], cache.v[i])
            x, new_k, new_v = block(x, time_mask, past)
            if cache is not None:
                if i < len(cache.k):
                    cache.k[i], cache.v[i] = new_k, new_v
                else:
                    cache.k.append(new_k)
                    cache.v.append(new_v)
        if cache is not None:
            cache.n_seen = offset + p
        return self.out_norm(x)


def _predictor_stack(d_model: int, n_layers: int, n_heads: int, ff_mult: int = 2):
    """The predictor keeps pm-jepa's `nn.TransformerEncoder` verbatim.

    Only the encoder had to be rewritten by hand, and only because the cache
    needs its keys and values. The predictor is never cached and never causal,
    so reimplementing it would add a way for the two repos' numbers to differ
    without adding anything.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class CausalJEPA(nn.Module):
    """Context encoder + EMA target encoder + narrow predictor.

    `causal_context` and `causal_target` are separate on purpose. The cell that
    tests H1 and H2 is `causal_target=True, causal_context=False`: it removes
    the future contamination from the reference the copy oracle is scored
    against, without changing what the predictor is allowed to see. Tying the
    two flags together would confound "the target stopped leaking" with "the
    context got weaker", and the prior pm-jepa results are not strong enough to
    survive that ambiguity.
    """

    def __init__(self, n_minutes: int, n_strikes: int, n_features: int,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
                 patch_length: int = 4, patch_stride: int = 4,
                 pred_dim: int = 96, pred_layers: int = 2, ema: float = 0.996,
                 causal_context: bool = False, causal_target: bool = False):
        super().__init__()
        self.K, self.C, self.T = n_strikes, n_features, n_minutes
        self.d_model, self.ema = d_model, ema
        self.patch_length, self.patch_stride = patch_length, patch_stride
        self.causal_context, self.causal_target = causal_context, causal_target

        if n_minutes < patch_length:
            raise ValueError("window {} shorter than one patch {}".format(
                n_minutes, patch_length))
        self.n_patches = (n_minutes - patch_length) // patch_stride + 1

        self.encoder = FactorisedEncoder(
            n_strikes, n_features, self.n_patches, patch_length, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, causal=causal_context,
            patch_stride=patch_stride)

        self.pred_in = nn.Linear(d_model, pred_dim)
        self.predictor = _predictor_stack(pred_dim, pred_layers, 4)
        self.pred_out = nn.Linear(pred_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(self.n_patches, n_strikes, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

        # Deep-copy then flip the flag, so both towers start from identical
        # weights exactly as pm-jepa does. The flag only selects an attention
        # mask at forward time, so the two arms of the 2x2 differ in what they
        # may look at and in nothing else at initialisation.
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.causal = causal_target
        self.target_encoder._mask_cache = {}
        for p in self._target_params():
            p.requires_grad_(False)

    def _target_params(self):
        for p in self.target_encoder.parameters():
            yield p

    @torch.no_grad()
    def update_target(self, momentum: Optional[float] = None) -> None:
        m = self.ema if momentum is None else momentum
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
        for bo, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.copy_(bo)

    def encode(self, obs: Tensor, strike_mask: Optional[Tensor] = None,
               target: bool = False) -> Tensor:
        """obs (B,T,K,C); strike_mask (B,T,K) bool, True = hidden from context.

        Masked minutes are zero-filled in the input rather than dropped, exactly
        as pm-jepa does, so the token grid keeps its shape and the patch/strike
        positions of a masked token still mean what they meant.
        """
        enc = self.target_encoder if target else self.encoder
        return enc(obs, strike_mask)

    def patch_mask(self, strike_mask: Tensor) -> Tensor:
        """(B,T,K) minute-level mask -> (B,P,K) patch-level.

        A patch counts as masked if ANY minute inside it was masked, which is
        the conservative direction: a partially hidden patch is never treated as
        visible context, so the copy oracle can never quote a reference that had
        a peek at the answer.
        """
        b, t, k = strike_mask.shape
        out = strike_mask.new_zeros(b, self.n_patches, k)
        for p in range(self.n_patches):
            lo = p * self.patch_stride
            hi = min(lo + self.patch_length, t)
            if hi > lo:
                out[:, p, :] = strike_mask[:, lo:hi].any(dim=1)
        return out

    def forward(self, obs: Tensor, strike_mask: Tensor) -> Dict[str, Tensor]:
        ctx = self.encode(obs, strike_mask, target=False)          # (B,P,K,d)
        with torch.no_grad():
            tgt = self.encode(obs, None, target=True)

        pm = self.patch_mask(strike_mask)                          # (B,P,K)
        filled = torch.where(pm.unsqueeze(-1),
                             self.mask_token.view(1, 1, 1, -1).expand_as(ctx), ctx)
        b = obs.shape[0]
        z = self.pred_in(filled + self.pos.unsqueeze(0))
        z = z.reshape(b, self.n_patches * self.K, -1)
        pred = self.pred_out(self.predictor(z)).reshape(b, self.n_patches, self.K, -1)

        return {"pred": pred.reshape(b, -1, self.d_model),
                "target": tgt.reshape(b, -1, self.d_model).detach(),
                "context": ctx.reshape(b, -1, self.d_model),
                "mask_flat": pm.reshape(b, -1),
                "patch_mask": pm}

    @torch.no_grad()
    def represent(self, obs: Tensor) -> Tensor:
        """Frozen representation for probing: mean over the token grid -> (B, d)."""
        return self.encode(obs, None, target=False).mean(dim=(1, 2))
