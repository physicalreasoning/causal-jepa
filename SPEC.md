# causal-jepa — build specification

**This file is the interface contract.** Several agents build different modules in
parallel. Every signature below is fixed. Do not change a signature; if one looks
wrong, implement it as written and note the objection in a `# SPEC NOTE:` comment.

---

## 0. The hypothesis under test

`pm-jepa` trains a JEPA on Kalshi strike ladders. Its target encoder is
**bidirectional**: `encode(obs, strike_mask=None, target=True)` runs unmasked
attention over the entire 24-minute window. Under `temporal` masking (patches 4-5
of 6 held out), the target embedding at a *visible* patch is therefore computed
with attention over the *masked* patches too.

Consequence: `target[p=3]` and `target[p=5]` both contain information from all six
patches, so they are more similar to each other than two causally-computed targets
would be. The copy oracle in `model/diagnostics.py` measures the model against
`target` at the nearest visible patch. So the oracle's reference is
future-contaminated, and the copy solution is **partly an artefact of the
bidirectional target encoder** rather than purely a property of the martingale.

**H1 (leakage).** A causal target encoder reduces target-to-target similarity
across the mask boundary. Measurable as: `target_autocorr(bidir) > target_autocorr(causal)`
at the same lag.

**H2 (intervention).** Under a causal target encoder, `copy_alignment` falls and
`copy_loss_ratio` rises: the copy baseline gets genuinely harder to beat.

**H3 (utility).** Whether probe R^2 improves is the open empirical question. A
negative result is a publishable result here. Do not tune to make causal win.

**H4 (cache).** Causal patch attention makes incremental encoding exact. A KV
cache must reproduce full re-encode to float tolerance, and must be faster for
streaming stride-1 inference.

Prior result to beat, from `pm-jepa/results/`: controls sit at ridge +0.449
(identity) and +0.460 (handcrafted); the best JEPA arm reached +0.377, i.e. every
JEPA arm **lost to the raw cross-section**. We are not expecting to win. We are
testing whether causality changes the copy diagnostics and the cacheability.

---

## 1. Data layout (fixed by the existing corpus)

```
obs   float32 (N, T=24, K=24, C=4)   N ~ 25,818 windows
Y     float32 (N, 4)                 probe targets
owner int     (N,)                   event id, for leak-free splitting
```

Feature axis C: see `pm-jepa/data/dataset.py`. Target names come from
`dataset.TARGET_NAMES`. Corpus is cached at `pm-jepa/data_cache/event_arrays/*.npz`
and assembled by `pm-jepa/load_corpus.py:build(window, stride)`.

Token grid after patching: **P=6 patches x K=24 strikes = 144 tokens**, d_model=128.
`patch_length=4, patch_stride=4`, so patch p covers minutes `4p .. 4p+3`.

---

## 2. Module: `causaljepa/model.py`

Replace PatchTST with an explicit factorised transformer so the time axis can be
masked and cached. Alternating blocks: **strike attention** (within one patch,
across all K strikes, always bidirectional) then **time attention** (within one
strike, across all P patches, causal or bidirectional per the flag).

```python
class FactorisedEncoder(nn.Module):
    def __init__(self, n_strikes: int, n_features: int, n_patches: int,
                 patch_length: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 8, causal: bool = False): ...

    def forward(self, obs: Tensor,                  # (B, T, K, C)
                strike_mask: Tensor | None = None,  # (B, T, K) bool, True = hidden
                cache: "KVCache | None" = None,
                ) -> Tensor:                        # (B, P, K, d)
        """Patch-first, strike-second axis order. This matches pm-jepa's
        `_tokens()` convention and every ported diagnostic assumes it."""
```

- `causal=True` applies a causal mask **on the time axis only**. Strike attention
  stays bidirectional in both arms; the strike ladder is a cross-section, not a
  sequence.
- Masked positions are zero-filled in the input, exactly as pm-jepa does
  (`obs.masked_fill(strike_mask.unsqueeze(-1), 0.0)`).
- Use `norm_first=True`, GELU, `dropout=0.0`, `ffn_dim = 2 * d_model`.

```python
class CausalJEPA(nn.Module):
    """Context encoder + EMA target encoder + narrow predictor."""
    def __init__(self, n_minutes, n_strikes, n_features, d_model=128,
                 n_layers=4, n_heads=8, patch_length=4, patch_stride=4,
                 pred_dim=96, pred_layers=2, ema=0.996,
                 causal_context: bool = False,
                 causal_target: bool = False): ...

    def forward(self, obs, strike_mask) -> dict:
        # keys, all matching pm-jepa/model/encoder.py exactly:
        #   pred      (B, P*K, d)
        #   target    (B, P*K, d)   detached
        #   context   (B, P*K, d)
        #   mask_flat (B, P*K)      bool
        #   patch_mask(B, P, K)     bool

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None: ...

    @torch.no_grad()
    def represent(self, obs: Tensor) -> Tensor:   # (B, d) mean over token grid
```

`causal_context` and `causal_target` are **separate flags**. The 2x2 is the
experiment: the interesting cell is `causal_target=True, causal_context=False`,
which fixes the leak without changing what context the predictor sees.

```python
def patch_mask(self, strike_mask: Tensor) -> Tensor:  # (B,T,K) -> (B,P,K)
    """A patch is masked if ANY minute inside it is masked (conservative)."""
```

---

## 3. Module: `causaljepa/cache.py`

```python
@dataclass
class KVCache:
    """Per-layer K/V for the TIME-attention sublayers only.
    Strike attention is within-patch and needs no cache."""
    k: list[Tensor]   # per layer, (B*K, heads, p_seen, head_dim)
    v: list[Tensor]
    n_seen: int

    def reset(self) -> None: ...

def encode_incremental(model, obs_window, cache=None) -> tuple[Tensor, KVCache]:
    """Encode only the NEW patches in obs_window given cache, return
    (tokens_for_new_patches, updated_cache). Requires model.causal is True;
    raise ValueError otherwise."""
```

**Exactness requirement.** For a causal encoder, feeding patches one at a time
through `encode_incremental` must equal `forward()` on the full window within
`atol=1e-4` (float32). This is the single most important test in the repo. If it
does not hold, the cache is wrong; do not weaken the tolerance to make it pass.

---

## 4. Module: `causaljepa/masking.py`

Port `pm-jepa/model/masking.py:sample_mask` verbatim (strategies `temporal`,
`slate`, `contiguous`, `mixed`, `random`) so results are comparable. Add one:

- `"interpolation"`: mask a contiguous block of patches in the **middle** of the
  window, leaving visible context on both sides. This is the arm where an
  interpolation copy is available. Under `causal=True` the model cannot use the
  future side, which is precisely the intervention.

Keep the same guarantee: at least one visible and one masked token per sample.

---

## 5. Module: `causaljepa/diagnostics.py`

Port `effective_rank`, `dimension_stats`, `copy_diagnostics` and
`nearest_visible_reference` from `pm-jepa/model/diagnostics.py` unchanged. Add:

```python
def directional_copy_diagnostics(pred, target, mask, beta=1.0) -> dict:
    """Split the copy oracle by DIRECTION. Returns:
        extrap_alignment, extrap_loss_ratio, extrap_frac   # nearest visible STRICTLY BEFORE
        interp_alignment, interp_loss_ratio, interp_frac   # nearest visible AFTER (when one exists)
        interp_advantage   # extrap_loss - interp_loss, > 0 means interpolation is the easier solution
    """

def target_autocorrelation(target, patch_mask, max_lag=5) -> dict[str, float]:
    """Mean cosine similarity between target embeddings at patch lag L, same
    strike. Key H1 measurement: a bidirectional target encoder inflates this
    across the mask boundary. Return {"lag_1": ..., ..., "cross_boundary": ...}
    where cross_boundary averages pairs straddling the visible/masked split."""
```

Both must handle the `mask.sum() == 0` and no-valid-reference cases by returning
`nan`, never raising.

---

## 6. Module: `causaljepa/data.py`

```python
def load_corpus(pm_jepa_root: str | Path, window: int = 24, stride: int = 4
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """-> (X, Y, owner, target_names). Import pm-jepa's own loader by path
    injection; do NOT reimplement window construction and do NOT copy the data."""

def split_by_event(owner, frac=0.2, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Whole events held out. Windows from one event share minutes and leak."""
```

Read-only with respect to `pm-jepa`. Never write into that directory.

---

## 7. Module: `causaljepa/probes.py`

Port `pm-jepa/model/probes.py` verbatim: `ridge_probe`, `mlp_probe`, `run_probes`.
Identical ridge grid, identical standardisation, identical val split. Any deviation
makes the numbers non-comparable to the prior results.

---

## 8. Module: `causaljepa/train.py`

```python
def train_arm(X, tr, *, strategy: str, causal_context: bool, causal_target: bool,
              device: str, steps: int, batch: int, lr: float, lam: float,
              d_model: int, n_layers: int, n_held_out: int, seed: int,
              log_every: int, verbose: bool = True
              ) -> tuple[nn.Module, list[dict], float]:
    """Returns (model, history, seconds). History records step, pred_loss,
    eff_rank, all copy diagnostics, all directional diagnostics, target
    autocorrelation, and dimension stats."""
```

Optimiser, schedule and EMA ramp must match `pm-jepa/train.py` exactly: AdamW
`lr=1e-3 wd=0.02 betas=(0.9,0.95)`, 100-step linear warmup times cosine decay,
grad clip 1.0, EMA momentum cosine-ramped from 0.996 to at most 0.9999.

**Loss.** Use plain smooth-L1 on masked positions. Do NOT port SIGReg: in pm-jepa
it was computed on a detached target and contributed exactly zero gradient. If a
regulariser is added later it must be verified to produce nonzero grad first.
Record `reg_grad_norm` in the history so this can never silently recur.

---

## 9. Tests: `tests/`

Mandatory, all must pass:

1. `test_cache_exactness` — incremental == full re-encode, `atol=1e-4`. Also assert
   it **fails** for `causal=False`, proving the test has teeth.
2. `test_causal_mask_no_future_leak` — perturb the input at patch p, assert tokens
   at patches < p are bit-identical under `causal=True` and do change under
   `causal=False`. This is the real proof of causality, stronger than reading the
   mask.
3. `test_patch_mask_geometry` — T=24, patch=4 -> P=6; `temporal` masks patches 4,5.
4. `test_shapes` — every `forward()` key has the shape in section 2.
5. `test_directional_diagnostics` — on synthetic data with a known copy structure,
   `interp_advantage > 0` when interpolation is available and `nan`-safe when not.
6. `test_probe_parity` — `ridge_probe` reproduces pm-jepa's on identical input to
   1e-9.
7. `test_no_data_mutation` — corpus loading does not write to the pm-jepa tree.

Run with `python -m pytest tests/ -q`. Target: under 60s on CPU.

---

## 10. Conventions

- Python 3.11+, torch 2.6, numpy 1.26. Device `mps` on this machine.
- No em-dashes in any prose, comment, or docstring. Use commas, semicolons, or hyphens.
- Docstrings explain **why**, in the voice of `pm-jepa`: state the failure mode the
  code guards against, not what the code does.
- Every result written to `results/*.json` with the full config inlined.
- Never modify anything under `/Users/nikita/pm-jepa` or `/Users/nikita/slate-jepa`.
