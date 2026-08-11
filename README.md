# causal-jepa

**Causal-axis joint-embedding predictive architectures on prediction-market strike ladders, and
what happens when you attack your own diagnostics.**

This repository contains the code, raw results and full findings record for a negative result. A
JEPA trained on Kalshi strike ladders does not learn a representation that beats raw features,
and the four explanations we pre-registered for why were, in order: refuted, refuted, refuted,
and correct only after we decomposed the metric we had been ranking everything by.

The programme is closed. It is published because the way it failed is more useful than the way it
would have succeeded.

---

## Headline results

| Question | Answer |
|---|---|
| Is the copy diagnostic an artefact of our own encoder? | **No.** Causal-target ablation, 4 seeds, Welch *t* = 0.09 |
| Does more training help? | **No.** 32x compute drives the nonlinear probe 23 pooled SD *below* an untrained encoder |
| Does effective rank track representation quality here? | **No.** It nearly triples while utility falls |
| Do anti-collapse regularisers prevent copying? | **Yes**, once actually in the gradient. Copy alignment 0.965 to 0.776, and it buys nothing |
| Does the copy diagnostic predict downstream utility? | **No.** Over a 0.198 range its correlation with probe R² is **+0.390**, the wrong sign |
| Is the objective broken? | **No.** On a simulator with real hidden state it beats raw features, and by *more* as noise rises |
| Why did it fail on market data? | The probe target set contained no recoverable hidden state |

---

## Background

Joint-embedding predictive architectures [1,2,3] learn representations by predicting the
embeddings of masked inputs rather than the inputs themselves. Applied to financial time series
this runs into a specific failure mode. If prices are close to a martingale, then
`E[p_{t+Δ} | F_t] ≈ p_t`, and if the encoder is smooth in price, a model can drive the objective
down by learning something very near the identity map. That map has full rank and full
per-dimension variance, so it passes every standard collapse check while encoding nothing about
market state.

We call this *martingale collapse*, and the predecessor repositories built an explicit **copy
oracle** to detect it: score the model against a copy of the nearest visible embedding, and
report the cosine alignment and the loss ratio. On real Kalshi ladders it reported alignment of
0.964.

This repository was built to answer one question about that number, and ended up answering
several others.

---

## Method

### The intervention

The predecessor's target encoder ran unmasked attention over the whole window. Under temporal
masking the target at a *visible* patch was therefore computed with attention over the *masked*
patches, so the copy oracle's reference embedding was contaminated with the future. If that
contamination was what made copying easy, the diagnostic was measuring the architecture rather
than the market.

We replace the PatchTST backbone [4] with an explicit factorised encoder so the time axis can be
masked and cached: alternating **strike attention** (within one patch, across all strikes, always
bidirectional, since a strike ladder is a cross-section rather than a sequence) and **time
attention** (within one strike, across patches, causal or bidirectional per flag). Context and
target towers carry independent causality flags, giving a 2x2.

Causality also makes incremental encoding exact, so `causaljepa/cache.py` implements a KV cache
for the time-attention sublayers, in the manner of segment-level recurrence [5].

### Evaluation

A JEPA cannot be evaluated by its own loss: the loss lives in an embedding space the model
controls, and a collapsed or identity encoder scores well. We freeze the encoder and fit probes
against exogenous targets [6], reporting both ridge (is the information linearly accessible) and
MLP (is it present at all).

Three classes of control, and the third is the one that mattered:

1. **Raw-feature controls** at several widths, including an exact reproduction of the
   predecessor's identity control.
2. **Dimension-matched controls.** Any comparison across differing readout width is untrustworthy
   until width is matched. This bit us twice; see `FINDINGS.md` 9 and 11.
3. **An untrained encoder** of identical architecture, at the same seeds, in every experiment. A
   model that cannot separate from its own initialisation has learned nothing.

### Pre-registration

`RESEARCH_PLAN.md` states each experiment's decision gate *before* it ran, plus a kill criterion
for the programme. Two gates turned out to be mis-specified; both are recorded as written, with
the correction beside them rather than replacing them.

---

## Results

Probe R² on the frozen representation, real Kalshi corpus, 25,818 windows split by event.

| Arm | dim | ridge R² | MLP R² |
|---|---|---|---|
| raw, full window | 2304 | **+0.5091** | **+0.5586** |
| raw, last 4 minutes | 384 | +0.5066 | +0.5590 |
| raw, last minute | 96 | +0.4754 | +0.5341 |
| JEPA, `concat_strikes` | 3072 | +0.4529 ± 0.0020 | +0.4213 ± 0.0082 |
| identity control (predecessor's bar) | 96 | +0.4491 | +0.5050 |
| **untrained encoder**, `concat_strikes` | 3072 | +0.4423 ± 0.0007 | +0.4102 ± 0.0143 |
| JEPA, `mean_all` | 128 | +0.3919 ± 0.0058 | +0.3797 ± 0.0049 |
| **untrained encoder**, `mean_all` | 128 | +0.3456 ± 0.0067 | +0.3802 ± 0.0049 |

Two things to read from this. Training produces a large, highly significant ridge lift at matched
width (+0.0463, *t* = 9.09) and no MLP lift at `mean_all`. And at 3072 dimensions the trained and
untrained arms are nearly identical, so most of that column is readout width rather than
learning.

### Metric decomposition

The score every arm in three repositories was ranked by is a mean over four probe targets. The
predecessor's own `dataset.py` documents one of them as *"DERIVABLE from the input, so it is a
sanity check ... Low R² means broken, not interesting."*

| Arm | `log_return_to_settle` | `time_to_expiry` | `implied_width` | `window_log_return` | mean |
|---|---|---|---|---|---|
| raw, full window | +0.0071 | +0.7935 | **+0.9320** | +0.3037 | +0.5091 |
| raw, last 4 minutes | +0.0510 | +0.7437 | **+0.9407** | +0.2910 | +0.5066 |
| JEPA, `concat_strikes` | +0.0282 | +0.8109 | +0.7552 | +0.2172 | +0.4529 |
| untrained, `mean_all` | −0.0001 | +0.6781 | +0.5785 | +0.1257 | +0.3456 |

On `log_return_to_settle`, the only genuinely future target, **all 16 arms land in
[−0.0043, +0.0510]** and are indistinguishable from each other and from zero. Nearly the entire
raw-versus-learned gap lives in the derivable sanity check. The metric was largely measuring
input reconstruction.

### Convergence

3 seeds, 19,200 steps, one cosine schedule, probed at six checkpoints.

| step | ridge | MLP | MLP vs untrained floor | copy alignment | effective rank |
|---|---|---|---|---|---|
| floor | +0.3443 | +0.3829 | | | |
| 600 | +0.4006 | +0.3966 | +3.30 sd | 0.972 | 36.4 |
| 2400 | +0.4003 | +0.3198 | −1.64 sd | 0.951 | 64.1 |
| 19200 | +0.3965 | +0.3055 | **−23.17 sd** | 0.940 | 93.8 |

### Regularisers

SIGReg [3] and the VICReg variance/covariance pair [7] were both computed on a **detached target**
in the predecessor repositories, contributing exactly zero gradient to every run either ever did.
Computed on the online context they work, and change nothing.

| arm | copy alignment | Δ | *t* | ridge | MLP |
|---|---|---|---|---|---|
| none | +0.9655 | | | +0.3917 | +0.3818 |
| SIGReg λ=1 | +0.8060 | −0.1595 | −38.0 | +0.3772 | +0.3666 |
| VICReg λ=1 | +0.7758 | **−0.1897** | −49.8 | +0.3811 | +0.3439 |

Across all 27 runs copy alignment spans 0.198 while ridge R² spans 0.0346, with a **positive**
correlation of +0.390 between them.

### Cache

Exact to 1.2e-06 (float32) and 2.7e-15 (float64) against a full forward pass, with past tokens
bit-identical under perturbation. Speedup is 1.22x at this model's six-patch window and 30.9x at
P=512 on CPU. It is valid only for an append-only patch stream: at `patch_stride=4` a one-minute
slide rewrites every patch, so the streaming case that motivated it cannot use it.

Full record with objections and falsification criteria: **[`FINDINGS.md`](FINDINGS.md)**.

---

## Repository layout

```
causaljepa/          library
  model.py           factorised encoder, EMA target, predictor; independent causality flags
  cache.py           KV cache for the time-attention sublayers
  masking.py         temporal / slate / contiguous / interpolation samplers
  diagnostics.py     copy oracle, directional split, target autocorrelation, effective rank
  regularisers.py    SIGReg and VICReg, applied to the ONLINE context
  probes.py          ridge and MLP probes, ported verbatim for comparability
  data.py            corpus loader (reads the predecessor corpus in place)
  train.py           training loop; records reg_grad_norm every step
scripts/             one experiment per file, each writing results/<name>.json
tests/               49 tests, including cache exactness and a no-mutation guard
results/             every raw result, with the full config inlined
docs/EVALS.md        evaluation protocol and what each measurement cannot tell you
docs/REPRODUCE.md    exact commands, runtimes and expected outputs
RESEARCH_PLAN.md     pre-registered gates and the kill criterion
FINDINGS.md          the findings record, including the corrections
SPEC.md              the interface contract the implementation was built against
```

---

## Installation

```bash
git clone https://github.com/physicalreasoning/causal-jepa
cd causal-jepa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch 2.6, NumPy 1.26. Developed on Apple Silicon (MPS); CUDA and CPU
paths work but timings in `docs/REPRODUCE.md` are MPS.

## Data availability

**The corpus is not in this repository and is not currently public.** It is built from Kalshi's
free unauthenticated API by the predecessor repository, which is private, and consists of 25,818
windows of reconstructed 24-strike ladders over settled hourly crypto events.

Point the code at a checkout with:

```bash
export PM_JEPA_ROOT=/path/to/pm-jepa       # must contain load_corpus.py and results/baselines.json
export SLATE_JEPA_ROOT=/path/to/slate-jepa # only needed for the simulator experiment
```

Without it, the 36 tests that do not touch the corpus still run and the other 13 skip:

```bash
python3 -m pytest tests/ -q
```

Everything in `results/` was produced from that corpus and is complete, so the findings are
checkable without it even though the runs are not repeatable without it. This is a real
limitation and we state it plainly rather than implying reproducibility we cannot offer.

## Reproduction

```bash
python3 scripts/readout_ablation.py   --seeds 4                  # E4, ~10 min
python3 scripts/convergence.py        --seeds 3 --steps 19200    # E1, ~2.0 h
python3 scripts/regulariser_sweep.py  --seeds 3 --steps 600      # E2, ~1.8 h
python3 scripts/sim_vs_real.py        --seeds 2                  # E3
python3 scripts/target_decomposition.py                          # E3c, no training
python3 scripts/benchmark.py          --seeds 4                  # controls, ~1 min
python3 scripts/cache_bench.py                                   # cache scaling
```

Each writes `results/<name>.json` with the full configuration inlined and prints its
pre-registered gate outcome. See `docs/REPRODUCE.md` for expected values and what to check if a
number differs.

---

## Limitations

- **One corpus, one asset class, one venue.** Kalshi hourly crypto events. Nothing here is known
  to transfer.
- **Small models.** 1.8M parameters, d_model 128, 4 layers. The convergence result rules out a
  longer run at this size; it does not rule out a larger one.
- **The negative result is about this target set**, not about prediction markets in general.
  Finding 10 shows the targets cannot detect learned structure; a target set containing genuine
  hidden state has not been tried.
- **Two pre-registered gates were mis-specified** and are recorded as such. Both would have
  produced the wrong conclusion if followed literally.
- **The absolute paths in `results/*.json` were normalised** to `$PM_JEPA_ROOT` before
  publication. No other field was altered.

## What would restart this

A better target set, not a bigger model. Something with recoverable hidden state: realised
volatility over a future window, or the settlement of a different correlated market. That is a
new pre-registration, not a continuation of this one.

---

## References

1. Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive
   Architecture (I-JEPA).* arXiv:2301.08243, 2023.
2. Bardes, A. et al. *Revisiting Feature Prediction for Learning Visual Representations from Video
   (V-JEPA).* arXiv:2404.08471, 2024.
3. Balestriero, R. and LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without
   the Heuristics.* arXiv:2511.08544, 2025. SIGReg is reimplemented here from the paper; the
   reference implementation is CC BY-NC 4.0 and is not vendored.
4. Nie, Y. et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
   (PatchTST).* arXiv:2211.14730, 2022.
5. Dai, Z. et al. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.*
   arXiv:1901.02860, 2019.
6. Alain, G. and Bengio, Y. *Understanding Intermediate Layers Using Linear Classifier Probes.*
   arXiv:1610.01644, 2016.
7. Bardes, A., Ponce, J. and LeCun, Y. *VICReg: Variance-Invariance-Covariance Regularization for
   Self-Supervised Learning.* arXiv:2105.04906, 2021.
8. Grill, J.-B. et al. *Bootstrap Your Own Latent (BYOL).* arXiv:2006.07733, 2020. Source of the
   EMA target-encoder construction used here.
9. Assran, M. et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
   Planning.* arXiv:2506.09985, 2025.

## Citation

```bibtex
@misc{causaljepa2026,
  title  = {Causal-axis JEPAs on prediction-market strike ladders:
            a negative result and the diagnostics that failed to predict it},
  author = {Physical Reasoning},
  year   = {2026},
  url    = {https://github.com/physicalreasoning/causal-jepa}
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
