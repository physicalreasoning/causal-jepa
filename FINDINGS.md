# Findings

Everything below is traceable to a file in `results/`. Four seeds per arm, 600 steps,
d_model 128, 4 layers, on the 25,818-window Kalshi corpus that `pm-jepa` built. Significance
is Welch's t on 4-seed samples; `|t| > 2.4` is roughly p < 0.05 two-sided at df ~ 6. With four
seeds and several comparisons, treat every marginal t as suggestive and nothing more.

---

## 1. The hypothesis that started this repo is refuted

**Claim.** `pm-jepa`'s target encoder runs unmasked bidirectional attention over the whole
window, so its copy oracle is scored against future-contaminated reference embeddings. We
suspected the copy solution it found was therefore partly an artefact of the model rather than
a property of the market.

**It is not.** Making the target tower causal, which removes the contamination entirely,
changes nothing that matters:

| metric (temporal masking) | tgt=bidir | tgt=causal | delta | t |
|---|---|---|---|---|
| `copy_alignment` | +0.9641 +- 0.0040 | +0.9616 +- 0.0045 | -0.0025 | -0.73 |
| ridge R^2 | +0.3919 +- 0.0058 | +0.3924 +- 0.0069 | +0.0005 | 0.09 |
| MLP R^2 | +0.3797 +- 0.0049 | +0.3809 +- 0.0055 | +0.0012 | 0.28 |

Copy alignment stays pinned at 0.96. The probe difference is `t = 0.09`, which is as null as a
result gets. Source: `results/causal_ablation.json`, `results/benchmark.json`.

**The leak is real, it is just too small to matter.** A weight-matched evaluation, identical
weights read out both ways, shows the two target towers genuinely differ, most at early patches
where causality removes the most context:

| patch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| cos(bidir, causal) | 0.9812 | 0.9902 | 0.9943 | 0.9969 | 0.9985 | 0.9995 |
| relative L2 | 0.1271 | 0.0896 | 0.0657 | 0.0481 | 0.0315 | 0.0163 |

Cross-boundary target autocorrelation falls from 0.9483 to 0.9377 under temporal masking. So H1
holds directionally: the bidirectional target encoder does inflate target similarity across the
mask boundary. But 0.011 of autocorrelation is not what was holding copy alignment at 0.96.
Source: `results/h1_weight_matched.json`.

**Strongest objection.** 600 steps is short, and a longer run might separate the arms. Against
that: the arms are indistinguishable at every logged step, not just the last, and the effect
size we are chasing (a 0.011 shift in target autocorrelation) is an order of magnitude below
the quantity it would have to explain (copy alignment sitting at 0.96 rather than near 0).

**What would falsify this.** A run at 5,000+ steps where `copy_alignment` separates by more
than 0.02 between target arms, or a masking scheme where the contaminated and uncontaminated
oracles disagree about which positions are copyable at all.

---

## 2. The MLP probe cannot distinguish a trained encoder from an untrained one

> **Superseded in part by finding 9.** This holds only for the `mean_all` readout. At the same
> 128 dimensions, `last_patch` shows the trained encoder beating its own initialisation by
> +0.0545 at t = 7.73. The null was our pooling, not the objective. Finding 8 separately closes
> the "600 steps may be too few" objection below: it is not, and more training makes it worse.

This was the result that mattered, and it was not what we set out to test.

An untrained encoder, identical architecture, no optimiser, no steps, no EMA, mean-pooled over
the same token grid:

| arm | ridge R^2 | MLP R^2 |
|---|---|---|
| random-encoder (bidir), untrained | +0.3456 +- 0.0067 | +0.3802 +- 0.0049 |
| ctx=bidir tgt=bidir, 600 steps | +0.3919 +- 0.0058 | +0.3797 +- 0.0049 |
| **delta from training** | **+0.0463** (t = 9.09) | **-0.0005** (t = -0.12) |

Same architecture, same corpus, same probe protocol, same seeds. Training moves the ridge probe
by a large and highly significant margin and moves the MLP probe by nothing at all.

The natural reading: the JEPA objective is not adding information to the representation, it is
**rearranging information the random projection already carried into a more linearly accessible
form**. The MLP probe, which does not care about linear accessibility, sees no difference.

Our best arm does not change the picture. `ctx=causal tgt=bidir` reaches MLP +0.3872 +- 0.0192
against the untrained +0.3802 +- 0.0049, a gap of +0.0070 at `t = 0.61`. Still null.

**Strongest objection.** A random transformer over structured input is a strong random-features
baseline, and beating it is a high bar that plenty of legitimate SSL methods clear only slowly.
600 steps may simply be too few. This objection is fair and we cannot rule it out from this
data. What we can say is that whatever the JEPA learned in 600 steps is invisible to a
nonlinear probe, and that no published `pm-jepa` result ever ran this control.

**What would falsify this.** A longer training run where MLP R^2 separates from the untrained
baseline by more than 2 seed-standard-deviations. That run is the single highest-value next
experiment in this repo.

---

## 3. Every arm still loses to the raw cross-section

> **Bar corrected by finding 9.** The controls below are the ones this programme inherited, and
> they are not the strongest trivial baselines. Raw features at 2304 dimensions reach +0.5091
> ridge and +0.5586 MLP. The real gap is 0.06 and 0.14, wider than what this section states.

The controls, measured in `pm-jepa/results/baselines.json` before any model existed, remain
unbeaten:

| arm | ridge R^2 | MLP R^2 |
|---|---|---|
| temporal (control) | +0.4600 | +0.5108 |
| handcrafted (control) | +0.4595 | +0.4858 |
| identity (control) | +0.4491 | +0.5050 |
| ctx=causal tgt=bidir (best trained, this work) | +0.4018 +- 0.0034 | +0.3872 +- 0.0192 |
| ctx=bidir tgt=bidir (pm-jepa configuration, our code) | +0.3919 +- 0.0058 | +0.3797 +- 0.0049 |
| random-encoder (bidir), untrained | +0.3456 +- 0.0067 | +0.3802 +- 0.0049 |
| jepa-temporal (pm-jepa, PatchTST) | +0.3771 +- 0.0095 | +0.3227 +- 0.0128 |
| jepa-slate (pm-jepa, PatchTST) | +0.3426 +- 0.0059 | +0.3160 +- 0.0042 |
| jepa-contiguous (pm-jepa, PatchTST) | +0.3394 +- 0.0064 | +0.3263 +- 0.0045 |

Identity beats our best arm by +0.0473 ridge (t = 23.77) and +0.1178 MLP (t = 10.62). The gap
is not closing and this work did not close it.

Note on the `pm-jepa` rows: those arms use a PatchTST backbone, ours uses a hand-written
factorised encoder, so any comparison across that line is architecture-confounded. The
within-architecture comparisons in findings 1 and 2 are the clean ones. What is not confounded
is that all three `pm-jepa` arms score below our untrained encoder on the MLP probe
(+0.3160 to +0.3263 against +0.3802).

---

## 4. Causal context, not causal target, is where the small effect lives

Making the **context** tower causal is the only intervention that moved anything:

| metric | ctx=bidir | ctx=causal | delta | t |
|---|---|---|---|---|
| ridge R^2 | +0.3919 +- 0.0058 | +0.4018 +- 0.0034 | +0.0099 | 2.56 |
| MLP R^2 | +0.3797 +- 0.0049 | +0.3872 +- 0.0192 | +0.0075 | 0.65 |
| `copy_alignment` | +0.9641 +- 0.0040 | +0.9539 +- 0.0035 | -0.0102 | -3.42 |
| `cross_boundary` | +0.9441 +- 0.0077 | +0.9079 +- 0.0126 | -0.0362 | -4.90 |
| `model_loss_on_copyable` | 0.0353 | 0.0546 | +0.0193 | |
| `eff_rank` | 34.85 | 37.27 | +2.42 | |

The prediction task genuinely got harder (loss on copyable positions rose 55%), the
representation got higher-rank, copy alignment fell, and the ridge probe moved marginally. The
MLP probe did not move. Read conservatively, this is one marginal ridge effect at t = 2.56
among six comparisons, and we would not defend it as significant without more seeds.

---

## 5. Interpolation is the easier solution, and causality widens the gap

Under `interpolation` masking, where visible context exists on both temporal sides:

| metric | tgt=bidir | tgt=causal |
|---|---|---|
| `extrap_loss_ratio` | 0.7710 +- 0.0333 | 0.7054 +- 0.0338 |
| `interp_loss_ratio` | 0.8104 +- 0.0211 | 0.7988 +- 0.0187 |
| `interp_advantage` | +0.0026 +- 0.0025 | +0.0072 +- 0.0041 |

`interp_advantage > 0` means the interpolation oracle is a harder baseline to beat than the
extrapolation oracle, which confirms the ordering we predicted: averaging two temporal
neighbours is a better solution than extrapolating from one. Making the target causal widens
the advantage from +0.0026 to +0.0072, as it should, since it makes the extrapolation side
genuinely harder while leaving interpolation available to the predictor.

The magnitudes are tiny. This is a confirmed mechanism, not a large effect.

---

## 6. The KV cache is exact, and the use case that motivated it does not exist

**Exactness holds.** Feeding patches one at a time through `encode_incremental` reproduces a
full forward pass to `1.2e-06` in float32 and `2.7e-15` in float64, against a `1e-4` bar. The
float32 residual is accumulation noise, not a structural mismatch; the float64 figure is what
proves that.

The test has teeth. Lifting the causality guard and running the same cached path against a
bidirectional encoder puts the deviation four orders of magnitude over the bar. Perturbing the
input at patch 3 leaves tokens at patches 0 to 2 **exactly** bit-identical under `causal=True`
and moves them by order 1 under `causal=False`.

Two of those four quantities are seed-stable and two are not, which matters for how they should
be quoted. Across seeds 0, 1, 2 at d_model 128 and 4 layers:

| | seed 0 | seed 1 | seed 2 | stable? |
|---|---|---|---|---|
| causal exactness | 1.43e-06 | 1.19e-06 | 1.43e-06 | yes, always far under the bar |
| past-token drift, causal | 0.0 | 0.0 | 0.0 | yes, exactly zero |
| bidirectional deviation | 2.70 | 3.12 | 2.85 | no, order 1 |
| past-token drift, bidirectional | 1.33 | 1.00 | 0.96 | no, order 1 |

Quote the first two as figures and the last two as orders of magnitude. Earlier drafts of this
file gave `2.78` and `1.14` as if they were constants; they are single draws from the third and
fourth rows, and other scripts in this repo at other shapes report 2.46 to 2.67 and 0.74 to
1.21. Source: `tests/test_cache_exactness.py`, `tests/test_causal_mask.py`, 49 tests passing.

**The speedup is real only at scale.** Append-only patch stream, one new patch per tick:

| P | 6 | 32 | 128 | 512 |
|---|---|---|---|---|
| MPS speedup | 1.22x | 1.16x | 1.26x | 4.19x |
| CPU speedup | 2.15x | 4.09x | 7.65x | 30.92x |

At this model's actual shape, P = 6, the cache buys 1.22x on MPS. That is engineering theatre.
The honest claim is about asymptotics, not about this model.

**The bigger problem.** The cache is only valid for an **append-only patch stream**. With
`patch_stride = 4`, a one-minute slide of the window rewrites the contents of every patch, so
the streaming case that motivated the whole project, a new market tick arriving and the window
sliding by one minute, cannot use the cache at all. Getting that case would need
`patch_stride = 1`, or a patch grid anchored to absolute time rather than to the window.
Source: `results/cache_bench.json`, field `regime`.

So H4 is supported as stated and useless as motivated. We built a correct answer to a question
that this model does not ask.

---

## 7. Both regularisers in the prior work were inert, and this repo cannot repeat that

`pm-jepa` computed SIGReg on a detached target under `torch.no_grad()`, so it contributed
exactly zero gradient; every run was effectively `lam = 0`. `slate-jepa` computed VICReg the
same way. Neither repo's regulariser ever influenced training.

This repo ships no regulariser and records `reg_grad_norm` in every history entry, which reads
0.0 by construction in every run here. Any future regulariser that fails to produce gradient
will show up as a zero in the logs rather than as a silent no-op.

---

## 8. More training makes it worse, and effective rank moves opposite to utility

Added 2026-08-10 from `results/convergence.json`, E1 in `RESEARCH_PLAN.md`. 3 seeds, 19,200
steps, one long cosine schedule probed at six checkpoints. This retires finding 2's main caveat.

| step | ridge | MLP | MLP vs untrained floor | copy_align | eff_rank |
|---|---|---|---|---|---|
| floor | +0.3443 +- 0.0073 | +0.3829 +- 0.0016 | | | |
| 600 | +0.4006 | +0.3966 | **+3.30 sd** | 0.972 | 36.4 |
| 1200 | +0.4047 | +0.3860 | +0.21 sd | 0.969 | 47.1 |
| 2400 | +0.4003 | +0.3198 | -1.64 sd | 0.951 | 64.1 |
| 4800 | +0.3932 | +0.3307 | -1.49 sd | 0.941 | 80.0 |
| 9600 | +0.3961 | +0.3089 | -3.44 sd | 0.925 | 92.3 |
| 19200 | +0.3965 | +0.3055 | **-23.17 sd** | 0.940 | 93.8 |

**32x more compute does not help.** Ridge is flat, +0.4006 to +0.3965. The MLP probe peaks at
the first checkpoint and falls 0.091, ending 23 pooled standard deviations **below** the
untrained encoder. The 600-step budget that every prior conclusion rests on was not a constraint;
it was near the best point on the curve. Finding 2's stated threat, "600 steps may simply be too
few", is closed.

**Effective rank triples while utility falls**, 36.4 to 93.8. Rank is not merely blind to this
failure mode, it moves in the opposite direction to the thing we care about. It must never be
read as a health signal on this data.

**Copying falls and it does not help.** `copy_alignment` drops 0.972 to 0.940 with training while
the probes get worse. Third independent confirmation that the copy diagnostic does not predict
downstream utility.

**The pre-registered gate fired and was wrong, and that is recorded rather than patched.** The
criterion read "exceeds the floor by > 2 pooled SD at any checkpoint", which the transient +3.30
at step 600 satisfies. "Any checkpoint" cannot answer whether more training helps. The recorded
`gate` in the JSON is left exactly as the run produced it; the correction lives beside it in
`gate_review`, and `RESEARCH_PLAN.md` now shows the old wording struck rather than replaced. A
plan that silently edits its criteria after seeing data is worth nothing.

## 9. The readout was hiding half the result, and the bar was too low

Added 2026-08-10 from `results/readout_ablation.json`, E4. This **corrects finding 2**.

Finding 2 reported that training buys linear accessibility and nothing an MLP probe can see. That
is true only for `mean_all`, the readout averaging over both the patch and strike axes:

| readout | dim | trained MLP | untrained MLP | lift | t |
|---|---|---|---|---|---|
| `last_patch` | 128 | +0.4217 | +0.3671 | **+0.0545** | **7.73** |
| `mean_max` | 256 | +0.4158 | +0.3432 | **+0.0726** | **6.02** |
| `concat_patches` | 768 | +0.4108 | +0.3719 | +0.0389 | 2.41 |
| `mean_all` | 128 | +0.3797 | +0.3802 | -0.0005 | -0.12 |
| `concat_strikes` | 3072 | +0.4213 | +0.4102 | +0.0111 | 1.17 |

`last_patch` is the same width as `mean_all`, so this is not a dimension effect. The encoder does
learn something a nonlinear probe can see; our pooling was destroying it.

**And the bar was too low.** The programme measured against pm-jepa's identity control at
+0.4491, which is not the strongest trivial baseline, nor even the strongest at its own width:
`raw_last_min` at 96 dimensions scores +0.4754, and the raw window at 2304 dimensions scores
+0.5091 ridge and +0.5586 MLP. Best JEPA readout is +0.4529 and +0.4213, so the real gap is 0.06
on ridge and 0.14 on MLP, **wider** than finding 3 reported. Correcting the first error made the
overall result worse.

**The gate nearly fired spuriously.** `concat_strikes` at 3072 dimensions beats the 96-dimensional
identity control, which reads as "readout artefact, stop everything". But the untrained encoder at
that same width scores +0.4423, so almost all of it was dimension. Dimension-matched raw controls
are what turned a false positive into the correct verdict.

## 10. The headline metric was mostly measuring input reconstruction

Added 2026-08-10 from `results/target_decomposition.json`, E3 part C. **This explains the entire
programme**, and it retires findings 3 and 9 as framed.

Every arm across three repositories has been ranked by `ridge_mean`, the mean ridge R² over four
probe targets. Nobody decomposed it. `pm-jepa/data/dataset.py` documents the targets itself:

| target | what dataset.py says about it |
|---|---|
| `log_return_to_settle` | "GENUINELY FUTURE. The headline target." |
| `time_to_expiry` | "Not in the input, provided windows are sampled at random offsets" |
| `implied_width` | "DERIVABLE from the input, so it is a sanity check... **Low R² means broken, not interesting**" |
| `window_log_return` | inside the window |

Decomposed, across 16 arms including raw features, untrained encoders and every trained JEPA:

| arm | log_return_to_settle | time_to_expiry | implied_width | window_log_return | mean |
|---|---|---|---|---|---|
| raw, full window | +0.0071 | +0.7935 | **+0.9320** | +0.3037 | +0.5091 |
| raw, last 4 min | +0.0510 | +0.7437 | **+0.9407** | +0.2910 | +0.5066 |
| jepa, concat_strikes | +0.0282 | +0.8109 | +0.7552 | +0.2172 | +0.4529 |
| jepa, mean_all | +0.0109 | +0.7633 | +0.6518 | +0.1415 | +0.3919 |
| untrained, mean_all | -0.0001 | +0.6781 | +0.5785 | +0.1257 | +0.3456 |

**On the only genuinely future target, all 16 arms land in [-0.0043, +0.0510].** Raw features,
an untrained encoder and every trained JEPA are indistinguishable from each other and from zero.
Nothing in this programme ever predicted anything about the future.

**Where the JEPA actually loses is `implied_width`**, best raw +0.9407 against best JEPA +0.7552,
a gap of 0.1854. That is the target the code itself calls a derivable sanity check whose low
values mean "broken, not interesting". A pooled 128-dimensional embedding cannot beat the raw
2304-dimensional input at reproducing a function of that input, so the comparison was never
winnable, and winning it would have meant nothing.

So "every JEPA arm loses to raw features" decomposes into: it reconstructs a derivable quantity
less perfectly than the thing it is a compression of, and on the target anyone would trade on,
everything including the raw data scores zero.

**This is also the sim-to-real gap.** The simulator's probe targets are genuine hidden state that
no single market identifies: `score_diff`, `score_total`, `time_remaining` sit at +0.68 to +0.97
and the JEPA beats identity on all three (+0.8819 against +0.7769 on `score_total`, +0.8533
against +0.7057 on `time_remaining`). There the objective has something to learn and it learns
it. On the real corpus there is no equivalent hidden state in the target set: one target is
unpredictable by construction of an efficient market and one is a function of the input.

**What this does not say.** That prediction-market ladders contain no learnable structure, only
that this target set cannot detect any. A target set with genuine hidden state, for instance
realised volatility over a future window or the settlement of a *different* correlated market,
would be a real test. That test has not been run.

## 11. Cross-sectional redundancy does not explain the gap, and my prediction was wrong

Added 2026-08-10 from `results/_redundancy_kmatched.json`, E3 part B.

I pre-registered the hypothesis that a Kalshi strike ladder is far more cross-sectionally
redundant than the simulator's heterogeneous market slate, so there would be less latent
structure for a model to add. On the native cross-sections that looked confirmed: mean sibling
R² of 0.9850 on Kalshi against 0.9540 on the simulator, a residual of 1.5% against 4.6%.

**It is a predictor-count artefact.** Kalshi has 24 markets and the simulator has 8, so
predicting one column from 23 others is easier than from 7 for reasons having nothing to do with
market structure. Matching K:

| dataset | K | mean R² | residual |
|---|---|---|---|
| simulator, native | 8 | 0.9540 | 0.0460 |
| kalshi, native | 24 | 0.9850 | 0.0150 |
| kalshi, subsampled to 8 evenly spaced | 8 | 0.9284 | **0.0716** |
| kalshi, contiguous middle 8 | 8 | 0.9659 | 0.0341 |

At matched width Kalshi has *more* residual cross-sectional structure than the simulator when the
strikes are spread across the ladder, and slightly less when they are adjacent. The hypothesis is
refuted; the difference is a function of how many rungs you look at and how far apart they are.

This is the second confound of exactly this shape in one day, after the readout-width confound in
finding 9. Both were caught only by adding a matched control. On this data, any comparison
between things of different dimension is untrustworthy until the dimension is matched.

## 12. Observation noise does not explain the gap either, and the JEPA is noise-robust

Added 2026-08-10 from `results/sim_vs_real.json`, E3 part A. The second of two pre-registered
explanations, and the second one refuted.

Scaling the simulator's quote noise degrades the observations without touching the latent state.
The plan predicted the JEPA's advantage over the identity control would shrink and cross zero
somewhere, giving a threshold to locate the real corpus against. It does the opposite:

| noise multiplier | identity | jepa-slate | advantage |
|---|---|---|---|
| 0.5x | +0.8239 | +0.8928 +- 0.0096 | +0.0690 |
| 1x | +0.8229 | +0.8823 +- 0.0187 | +0.0594 |
| 2x | +0.8153 | +0.8791 +- 0.0116 | +0.0638 |
| 4x | +0.7952 | +0.8618 +- 0.0006 | +0.0666 |
| 8x | +0.7538 | +0.8466 +- 0.0014 | +0.0928 |
| 16x | +0.6734 | +0.8237 +- 0.0004 | **+0.1502** |

**The advantage never crosses zero, and across a 32x range of noise it more than doubles.** Raw
features degrade quickly under noise, +0.8239 to +0.6734, while the learned representation barely
moves, +0.8928 to +0.8237. That is the expected behaviour of an encoder that averages over a
window and a cross-section, and it is a genuine positive result about the objective: **where
there is real latent state to find, a JEPA is markedly more noise-robust than raw features, and
the noisier the data the more it is worth.**

It also means observation SNR cannot be the sim-to-real story. Both pre-registered explanations
for the gap, redundancy in finding 11 and noise here, are refuted. What is left is finding 10:
the real corpus's target set contains no hidden state to recover.

## What this work did not establish

- ~~Whether a longer run separates the trained encoder from its random initialisation.~~
  **Closed by finding 8.** It does not. 32x more compute leaves ridge flat and drives the MLP
  probe 23 pooled standard deviations below the untrained floor.
- Whether any of this transfers off the Kalshi hourly-crypto corpus. One corpus, one asset
  class, one venue.
- Whether the marginal causal-context effect in finding 4 survives more seeds. At t = 2.56 with
  n = 4 and six comparisons, it probably does not.
- Anything about trading. No arm beats the raw cross-section, so none of this is deployable.
