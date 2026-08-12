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

Added 2026-08-10 from `results/convergence.json`, E1 in `docs/RESEARCH_PLAN.md`. 3 seeds, 19,200
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
`gate_review`, and `docs/RESEARCH_PLAN.md` now shows the old wording struck rather than replaced. A
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
values mean "broken, not interesting". A compressed embedding cannot beat the raw input at
reproducing a function of that input, so the comparison was never winnable, and winning it would
have meant nothing.

*Attribution note, corrected 2026-08-10.* An earlier version of this paragraph said "a pooled
128-dimensional embedding cannot beat the raw 2304-dimensional input", pairing the two numbers
above with the wrong arms. +0.9407 is `raw_last_4min` at 384 dimensions, not the 2304-dimensional
window, which scores +0.9320; +0.7552 is `concat_strikes` at 3072 dimensions, not the 128-dimensional
pooled readout, which scores +0.6518. The argument is unaffected, the arm labels were wrong.

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

## 13. The regularisers do prevent copying. It buys nothing.

Added 2026-08-10 from `results/regulariser_sweep.json`, E2. **This refutes a published claim in
`pm-jepa/FINDINGS.md`**, and it is the fourth and sharpest confirmation that the copy diagnostic
does not predict utility.

Neither repo's regulariser was ever in the gradient, so "SIGReg does not prevent martingale
collapse" described an experiment nobody ran. Computed on the online context, where it carries
gradient, both regularisers work. 3 seeds, 600 steps, every arm asserted live before it counted.

| arm | copy_alignment | Δ vs baseline | t | ridge | Δ | t | MLP | Δ | t |
|---|---|---|---|---|---|---|---|---|---|
| none | +0.9655 | | | +0.3917 | | | +0.3818 | | |
| sigreg 0.01 | +0.9378 | -0.0277 | -9.3 | **+0.4042** | +0.0126 | 2.6 | +0.3850 | +0.0032 | 0.5 |
| sigreg 0.05 | +0.8995 | -0.0660 | -23.1 | +0.3958 | +0.0041 | 0.8 | +0.3746 | -0.0072 | -0.9 |
| sigreg 0.25 | +0.8435 | -0.1219 | -42.8 | +0.3879 | -0.0038 | -0.8 | +0.3521 | -0.0298 | -2.9 |
| sigreg 1 | +0.8060 | **-0.1595** | -38.0 | +0.3772 | -0.0144 | -2.5 | +0.3666 | -0.0152 | -2.8 |
| vicreg 0.05 | +0.9019 | -0.0636 | -9.5 | +0.3901 | -0.0015 | -0.3 | +0.3745 | -0.0073 | -0.5 |
| vicreg 0.25 | +0.8236 | -0.1418 | -19.2 | +0.3922 | +0.0005 | 0.1 | +0.3303 | -0.0516 | -2.3 |
| vicreg 1 | +0.7758 | **-0.1897** | -49.8 | +0.3811 | -0.0106 | -1.9 | +0.3439 | -0.0380 | -4.6 |

**The claim is refuted.** Copy alignment falls from 0.9655 to 0.7758, an absolute drop of 0.19 at
t = -49.8. Both mechanisms do exactly what they advertise once they are actually applied. The
structural argument in `pm-jepa/model/sigreg.py`, that a copy of a Gaussian-ish input is itself
Gaussian-ish so sketched normality cannot see copying, turns out to be wrong as a claim about the
optimisation even though it is right as a claim about the statistic.

**And it buys nothing.** Ridge moves between -0.0144 and +0.0126 across the whole sweep, mostly
inside noise. The MLP probe is flat or significantly *worse*, down 0.0380 at t = -4.6 for the
strongest VICReg. Nothing here closes any part of the gap to raw features.

**The sharpest way to state it.** Across all 27 runs, `copy_alignment` spans 0.7709 to 0.9690, a
range of 0.198, while `ridge_mean` spans only 0.0346. The correlation between them is **+0.390**,
and with the MLP probe **+0.366**. Both positive: within this sweep, copying *more* is weakly
associated with scoring *better*. The diagnostic is not merely uninformative about downstream
utility, over this range it points the wrong way.

So the copy oracle is now demonstrably manipulable. You can set it to almost any value you like
with one hyperparameter, and downstream quality does not follow. That retires it as a model
selection signal, while leaving intact what it was originally built for: detecting that copying
is happening at all.

**The gate wording was wrong again.** E2's criterion said dropping below 0.90 without destroying
probe R² means "the mechanism does help". It fired, and "help" is the wrong word: the mechanism
reduces copying and does not help the representation. Same class of mis-specification as E1's
"any checkpoint". Recorded rather than patched, in `results/regulariser_sweep.json:gate`.

## 14. Better targets make the encoder look like it is learning. It is not.

Added 2026-08-12 from `results/hidden_state_targets.json`, `results/hidden_state_targets_h5.json`
and `results/residual_probe.json`, E7 and E7c. **This closes the one question the kill criterion
left open**, and it does so by refuting my own intermediate result.

The plan said the single thing that would restart this programme was not a bigger model but a
better target set, because finding 10 showed the old one could not detect learned structure. E7
built that target set: four targets with genuine hidden state at a 10-step forward horizon,
17,449 windows over 2,727 events, headed by `vol_forecast_error`, which divides the ladder's own
volatility forecast out of future realised volatility. A parity test asserts the windows are an
exact subset of the published corpus, so old and new targets are measured on identical data.

**And a training lift appeared, at every readout, at both horizons.**

| readout | dim | untrained → trained | lift, h=10 | lift, h=5 |
|---|---|---|---|---|
| `last_patch` | 128 | +0.0358 → +0.0791 | **+0.0433 (+7.73 sd)** | +0.0443 (+3.76 sd) |
| `concat_patches` | 768 | +0.0487 → +0.0781 | +0.0294 (+3.10 sd) | +0.0363 (+3.50 sd) |
| `concat_strikes` | 3072 | +0.1022 → +0.1206 | +0.0184 (+3.15 sd) | +0.0143 (+1.68 sd) |
| `mean_all` | 128 | +0.0644 → +0.0776 | +0.0132 (+0.89 sd) | +0.0153 (+1.98 sd) |
| `mean_max` | 256 | +0.0677 → +0.0717 | +0.0040 (+0.62 sd) | +0.0177 (+2.12 sd) |

This is the first time in the programme that training moved a probe in the right direction on
something other than a target documented as derivable. It replicates at an independent horizon to
the third decimal at `last_patch`, +0.0443 against +0.0433. The martingale control stayed clean
throughout: the worst `fwd_signed_return` ridge R² across every arm is +0.0064 at h=10 and +0.0042
at h=5, against a pre-registered void-the-run bar of +0.05.

**It is still reconstruction.** `vol_forecast_error` is `log(realised) - log(implied)`, and
`log(implied)` is a function of `implied_width`, which is derivable from the input. The same run
measures a **+0.1392 (+6.30 sd)** training lift on `implied_width` itself at that same readout. So
an encoder that merely represents the ladder better predicts the `log(implied)` half of the target
better, and scores higher having learned nothing about the future.

E7c settles it by removing what raw features already explain. Residualising against the 96-dim
`raw_identity` control, which explains +0.1754 of the target and scores -0.0049 on its own
residual, then re-probing every arm:

| model | readout | dim | residual ridge R² | t vs zero |
|---|---|---|---|---|
| trained | `last_patch` | 128 | +0.0038 ± 0.0045 | 1.68 |
| trained | `mean_all` | 128 | +0.0007 ± 0.0068 | 0.21 |
| trained | `concat_strikes` | 3072 | -0.0023 ± 0.0027 | -1.71 |
| untrained | `concat_strikes` | 3072 | -0.0173 ± 0.0031 | -10.98 |

**No arm reaches a residual R² distinguishable from zero**, the best being t = 1.68 against a
critical value of 3.182 at 4 seeds. Whatever training did, it added nothing about
`vol_forecast_error` beyond what 96 raw numbers already carry.

**Raw features still win outright anyway**, before any of this subtlety. On `vol_forecast_error`
the 96-dim `raw_identity` control scores +0.1747 against +0.1206 for the best trained arm at 3072
dims, a margin of 13.58 pooled SD; at h=5 it is +0.2363 against +0.1771, 8.69 SD. Note the
reversal from the old targets, where width helped and made the 3072-dim readout look strong: here
`raw_full` at 2304 dims falls to +0.1170, and the narrowest control wins.

**The methodological point, which is the transferable part.** Dividing the confound out
*arithmetically* did not remove it. `vol_forecast_error` was designed so the market's forecast
cancels, and it still carried enough derivable structure to manufacture a 7.73 sigma effect. Only
the empirical residualisation removed it, and only because the control was verified to score zero
on its own residual. A target is not clean because its algebra says so.

**A third gate failed on its wording.** E7c's criterion asked only for a matched-width lift above
2 pooled SD, never that the trained arm reach a *positive* residual R². It fired on `concat_strikes`,
whose lift runs from -0.0173 to -0.0023: both endpoints negative, so neither arm predicts the
residual better than its own mean, and being less bad at an impossible task is not evidence. Same
class as E1's "at any checkpoint" and E2's "the mechanism helps". Recorded as run in
`results/residual_probe.json:verdict`, corrected beside it under `verdict_review`.

**So the negative result stands, and stands harder.** It is not an artefact of a badly chosen
target set. Given targets built specifically to contain hidden state, verified discriminative
before use, replicated across two horizons, with the derivable part removed empirically rather
than by assertion, the encoder carries nothing the raw cross-section does not.

## 15. Which regime a probe target is in is knowable before training anything

Added 2026-08-12 from `results/horizon_headroom.json`, E7b. Raw features only, no training, about
a minute per horizon.

A probe target can only rank representations if it sits between two failure modes. If raw features
already explain it, gains on it measure input reconstruction. If nothing explains it, it cannot
separate a good representation from a bad one. Both are measurable in advance, for free, by
fitting raw features to the target and looking at what is left.

Best raw ridge R², across horizons:

| target | h=5 | h=10 | h=15 | h=20 | regime |
|---|---|---|---|---|---|
| `implied_width` | +0.9443 | +0.9684 | +0.9789 | +0.9853 | DERIVABLE |
| `time_to_expiry` | +0.7811 | +0.8025 | +0.8363 | +0.8206 | DERIVABLE |
| `log_return_to_settle` | -0.0483 | -0.1346 | -0.0358 | -0.1323 | UNPREDICTABLE |
| `vol_forecast_error` | +0.2363 | +0.1747 | +0.0719 | -0.0150 | discriminative to h=15 |
| `fwd_abs_return` | +0.1553 | +0.1873 | +0.1229 | +0.1013 | DISCRIMINATIVE |

**Three of the four targets this entire programme was scored on are in regimes where a probe
comparison cannot carry information**, at every horizon tested, and the classification needs no
model, no training and no GPU. Finding 10 took a decomposition experiment and most of a programme
to establish what this table shows in a minute. Running it first would have redirected the work
before the encoder was written.

It also answers the obvious objection to E7. `vol_forecast_error` scores *higher* at h=5 (+0.2363)
than at the h=10 that E7 used (+0.1747), so 10 was not the flattering choice; the finding-14 lift
was then replicated at h=5 anyway.

## 16. Realised correlation is predictable from paired ladders, and the ladder is not what predicts it

Added 2026-08-12 from `results/crossasset_headroom.json`, E8a. Raw features only, no training.
**This is the one unambiguously positive result in the programme**, and it is not about the model.

The cached corpus was never one asset. It holds 1,675 KXBTCD and 1,661 KXETHD events over the same
ten weeks, and 99.1% of BTC events have a time-overlapping ETH event. The series label is recoverable
by recomputing `build_corpus.py`'s cache keys, so this needed no network access. Pairing on the
intersection of actual timestamps, rather than nominal event windows, gives 1,021 usable pairs at
`window=24, horizon=10`; the binding constraint is that the median pair shares only 37 valid minutes.

A strike ladder is a **marginal** distribution: it prices where one asset lands. The **dependence**
between two assets is in neither ladder at any strike. That makes realised correlation the first
target in this corpus that no single cross-section identifies, and the data agrees:

| target, h=10 | BTC only | ETH only | both | gain vs width-matched single |
|---|---|---|---|---|
| **`realised_corr`** | +0.1867 | +0.2520 | **+0.3693** | **+0.1064** |
| `eth_vol_forecast_error` | +0.0109 | +0.1363 | +0.1302 | -0.0048 |
| `fwd_abs_spread_return` | +0.0174 | +0.1519 | +0.1499 | -0.0014 |
| `vol_ratio_forecast_error` | +0.0211 | +0.1293 | +0.1189 | -0.0076 |

The gain is measured against **ETH duplicated to the same 48 strikes**, so it is the second asset
rather than the extra columns; E4 is why that control is not optional. It rises monotonically with
horizon, +0.0271 at h=3, +0.0794 at h=5, +0.1064 at h=10, consistent with realised correlation being
better estimated over longer windows.

Targets that should not need both assets get no gain from the second ladder, and three of them
come in slightly negative. Only the quantity that is theoretically joint behaves as though it is
joint, so this is not a width artefact.

### The correction, added the same day: the ladder is not doing the work

**This finding was first written up claiming the above was "the one unambiguously positive result
in the programme". That claim was wrong, and the control that shows it had not been run.** The
comparison against `eth_doubled` holds dimension fixed but starves the comparison of the second
asset *by construction*, and a correlation cannot be computed from one asset at all. So the
+0.1064 "cross-asset gain" is mostly the tautology that computing a correlation takes two things.

The control that was missing is both assets at minimal feature count:

| predictor for `realised_corr`, h=10 | dim | ridge R² |
|---|---|---|
| within-window realised correlation, **one number** | 1 | **+0.2718** |
| log implied volatility of each asset | 2 | +0.2128 |
| the above plus each asset's within-window realised volatility | 4 | +0.2214 |
| **five numbers**: both implied vols, both realised vols, lagged correlation | 5 | **+0.3462** |
| the 768-dim both-ladder readout | 768 | +0.3693 |

Bootstrap over the test set, 2,000 resamples: the ladder readout's advantage over those five
numbers is **+0.0234, 95% CI [-0.0389, +0.0840], P(>0) = 0.775**. Not distinguishable from zero.

So what is actually true is much smaller than what was claimed. Future realised correlation is
predictable, mostly because **correlation persists**, one lagged number carrying +0.2718 of it, and
partly because it rises with volatility. Both are textbook. The 24-strike cross-sections that this
whole repository is about contribute nothing detectable on top.

**This is the fifth correction in this document and the first that is purely self-inflicted**: not
a mis-worded gate, but a result written up before its control existed. It was caught by asking the
question this programme asks of everything else, what does a trivial baseline get, roughly an hour
after the claim was committed. That the same author who wrote findings 9, 10, 14 and 17 still
shipped it is the most honest evidence in this document that the discipline has to be procedural
rather than remembered.

## 17. The JEPA loses even where the latent provably exists, and the lift is reconstruction again

Added 2026-08-12 from `results/crossasset_jepa.json` and `results/residual_probe_cross.json`, E8
and E8c.

Finding 16 supplies what every previous experiment lacked: a target with a real cross-sectional
latent, verified before training. The pairing also leaves only 2,654 training windows against E7's
13,763, so wide raw readouts overfit and a compressed representation has more room here than
anywhere else in the programme. These are the most favourable conditions a JEPA gets on this corpus.

**It produced the largest training signal in the programme.** Matched-width lift on `realised_corr`:

| readout | dim | untrained → trained | lift |
|---|---|---|---|
| `last_patch` | 128 | +0.1280 → +0.2099 | **+0.0819 (4.54 sd)** |
| `concat_patches` | 768 | +0.1452 → +0.1983 | +0.0531 (3.27 sd) |
| `concat_strikes` | 6144 | +0.3259 → +0.3457 | +0.0198 (3.38 sd) |

**Raw features still win**, +0.3693 at 768 dims against +0.3457 for the best trained arm at 6144,
a margin of 6.58 pooled SD.

**And the lift is reconstruction, exactly as in finding 14.** Residualising against the both-ladder
raw readout, which explains +0.3693 of the target:

| model | readout | dim | residual ridge R² |
|---|---|---|---|
| **untrained** | `mean_all` | 128 | **+0.0077 ± 0.0018** |
| trained | `last_patch` | 128 | +0.0042 ± 0.0022 |
| trained | `concat_strikes` | 6144 | -0.0555 ± 0.0065 |
| untrained | `concat_strikes` | 6144 | -0.0813 ± 0.0027 |

**The best untrained arm outscores every trained arm on the residual.** A random-initialised
encoder carries more of what raw features miss than any trained one does, so the small surviving
signal is the architecture acting as a random projection, not a product of training.

**The generalisable result, and it is the most useful thing in this document.** The matched-width
training lift is the standard evidence that self-supervised pretraining worked. It has now been
measured at **+7.73 pooled SD** (finding 14) and **+4.54 pooled SD** (here), on different targets,
different corpora and different input geometries, and **both times it vanished under a control that
costs one extra minute**. Twice is a pattern, not an accident: an encoder trained to predict masked
parts of its input gets better at representing that input, and any target with a component the input
can compute will reward that without a single bit of new information having been learned. A lift
over an untrained encoder at matched width is necessary evidence of learning something useful. It is
nowhere near sufficient.

### Prior work, checked after the fact, which is the wrong order

Before writing this up as a methods contribution we checked whether it was one. It is not, and the
honest statement is that **both controls this programme arrived at the hard way are established
practice**:

- **Random and untrained encoder baselines** are standard in the probing literature. Hewitt and
  Liang [[9]](#references) formalise the concern as *selectivity*, using control tasks that "can
  only be learned by the probe itself" to separate what the representation carries from what the
  probe can memorise. Belinkov [[10]](#references) surveys the shortcomings more broadly.
- **Residualising a probe target against a confound predictor**, which is exactly what E7c and E8c
  do, is an existing diagnostic: fit a confound-only predictor, define a residualised target, and
  re-probe, where a large drop indicates confound-mediated signal.
- **The untrained encoder being a strong baseline** is itself documented. Asano et al.
  [[11]](#references) show the early layers of several self-supervised methods can be learned from
  a *single image* as well as from millions, so a large part of what looks like learned structure
  is architecture and augmentation rather than data.

What this programme adds is not the method but the magnitude, and the mechanism for one family.
A **+7.73** and a **+4.54 pooled SD** lift over a matched-width untrained encoder, on different
targets and different corpora, both residualising to nothing, is a sharper demonstration than we
found elsewhere. And the mechanism is specific to masked-prediction objectives rather than generic:
the pretraining task *is* input reconstruction, so it will inflate any target with an
input-computable component, by construction and every time. That is a narrower and more predictable
failure than "confounds exist".

The uncomfortable part is the ordering. This repository spent five experiments discovering, at
considerable cost, controls that were already in the literature it should have read first.

**A fourth criterion failed on its wording**, and this one needed two rounds. E8c's check required a
lift above 2 pooled SD plus a trained arm above zero, and both were satisfied **by different arms**,
one clearing zero while a second, entirely negative, supplied the lift. The condition that actually
decides it, that a trained arm must beat the best *untrained* arm on the residual, was missing.
Recorded as run in `results/residual_probe_cross.json:verdict`, corrected under `verdict_review`.
Four wording failures across sixteen gates is itself worth reporting: a threshold fixed in advance
is still only as good as the failure modes its author imagined.

## 18. The corpus discards most of what the API returns, and the sim-versus-real comparison never controlled for it

Added 2026-08-12. Structural facts about the data pipeline, verified against a live API response and
against the corpus itself. **This qualifies finding 12 and E3, and it is the most consequential
limitation in this document.**

Per strike per minute, Kalshi's candlestick endpoint returns **ten fields**:

```
yes_bid:  open, high, low, close        yes_ask:  open, high, low, close
volume,  open_interest
```

`data/kalshi.py::candle_quote` reads **two of them**, both `close_dollars`. The intra-minute high
and low of each side, which is the realised range and the cheapest good volatility estimator
available below the minute grid, is dropped at ingest. E7's headline targets were **volatility**
targets. We asked the model to predict volatility having thrown the range away.

A live event pulled 2026-08-12 (`KXBTCD-26AUG1213`) carries **188 strikes**. `resample_ladder`
interpolates onto a fixed 24-point grid, so for such events most of the cross-section is discarded
and `np.interp` smooths what remains. The claim made repeatedly in this repo, that a Kalshi ladder
is "a smooth near-deterministic function of two numbers", is therefore partly a statement about our
interpolation rather than about the market.

**And the confound that matters most.** The simulator every arm was benchmarked against has **25
features per market**: logit mid, half-spread, five bid sizes, five ask sizes, five bid offsets,
five ask offsets, imbalance, trade count, signed volume. The Kalshi corpus has **4 derived
channels**. Finding 12 concluded the objective works on the simulator and fails on the market, and
attributed the difference to the market. That comparison was never controlled for feature richness,
and it cannot support the weight this programme put on it.

**The current corpus is also richer than this repo claimed.** Persistence of principal components
across ten minutes, measured on the committed corpus:

| channel | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 |
|---|---|---|---|---|---|---|
| `survival_prob` | +0.919 | +0.734 | +0.494 | +0.223 | +0.269 | +0.402 |
| `quoted_spread` | +0.537 | +0.119 | +0.251 | +0.067 | +0.286 | +0.160 |
| `log_volume` | +0.642 | +0.813 | +0.184 | +0.369 | +0.089 | +0.221 |
| `log_open_interest` | **+0.946** | +0.596 | **+0.820** | **+0.822** | +0.562 | **+0.680** |

Open interest carries at least six persistent components. Only `survival_prob` looks like the
two-degrees-of-freedom object this repo described, and every probe target ever built here was
price-derived, so nothing asked the model about the rest.

**What this does and does not overturn.** It does not rescue any specific result: findings 14 and 17
compare arms on identical inputs, so a poorer input set penalises the raw controls and the encoder
equally. What it does is bound the scope. The honest statement is **not** "prediction-market ladders
carry too little structure for a representation learner". It is "the four channels this corpus
keeps, on a 24-point interpolated grid, carry too little structure", and the gap between those two
sentences is large and was never measured.

Order-book depth and signed flow, which the simulator has and this corpus lacks entirely, are
probably not recoverable retroactively: candles are what the API serves historically. Testing that
half would require recording depth forward from now.

**The empirical half of this is still open.** A first attempt to test it, rebuilding 109 events with
eight channels on a 48-point grid and comparing against the four-channel 24-point build on the same
events, is **discarded as invalid**. It carried a built-in control, a `lean` arm that had to
reproduce the corpus's own `raw_last_min` number of +0.1487 on `vol_forecast_error`, and that arm
returned -0.5459. Two causes, both in the test rather than in the data: 911 windows against the
corpus's 17,449, so 678 training rows against up to 1,536 features drove every arm negative; and the
`lean` arm recomputed spot and width with a crude quantile instead of calling `slate.implied_spot`
and `slate.implied_width`, rejecting 566 minutes the real builder keeps. It was never the same
feature set, so it never tested the question.

Doing this properly means a full corpus rebuild through the existing builder with the extra fields
retained, roughly 10,000 API calls, and it has not been run. **Until it is, the structural facts
above stand and the empirical claim that richer features would help is unsupported in either
direction.**

## 19. The discarded fields were recovered. They are empty, because the market is thin.

Added 2026-08-12 from `results/rich_vs_lean.json`, E9. **This closes the question finding 18
opened**, and it replaces this repository's original hand-wave about dimensionality with a measured
reason.

Finding 18 established that the builder reads 2 of the 10 fields each candle carries. So we rebuilt
534 events keeping all of them, through slate's own ladder builder, validator, `implied_spot`,
`implied_width` and `resample_ladder`, so that the lean arm is **bit-identical to the committed
corpus** on overlapping events: 6 of 6 checked matched exactly on `obs`, `ts` and `state`. Identical
minutes, identical windows, identical targets; only the channel count differs.

Width is controlled. Rich has 10 channels against lean's 4, so `lean_tiled` repeats the lean
channels to the same 10, carrying identical information at identical dimension.

| target | lean | lean_tiled | rich | rich − tiled |
|---|---|---|---|---|
| `fwd_realised_vol` | +0.1840 | +0.2061 | +0.1816 | **-0.0245** |
| `vol_forecast_error` | +0.2407 | +0.2361 | +0.1893 | **-0.0468** |
| `fwd_abs_return` | +0.1275 | +0.1218 | +0.1012 | -0.0206 |
| `fwd_signed_return` | -0.0320 | -0.0430 | -0.0791 | -0.0361 |

**The extra fields do not help on any target.** Repeating the four channels the corpus already keeps
beats adding six genuinely new ones, at matched dimension.

**And here is why**, which is the part worth keeping:

| channel | fraction exactly zero |
|---|---|
| `bid_range` | 0.699 |
| `ask_range` | 0.699 |
| `mid_range` | 0.666 |
| `mid_drift` | 0.724 |
| `bid_open_to_close` | 0.747 |
| `ask_open_to_close` | 0.750 |
| `log_volume` | 0.644 |

On roughly **70% of strike-minutes the quote does not move at all**, and on **64% nothing trades**.
The intra-minute range that should have been the cheapest good volatility estimator is mostly
literally zero. The fields exist in the API and carry almost no information on this venue, because
most strikes on an hourly crypto ladder are untouched from one minute to the next.

**So the constraint is liquidity, not the builder.** That is a better-supported statement than the
one this repo made for most of its life. "A Kalshi ladder is a smooth function of two numbers" was
partly an artefact of our own interpolation, as finding 18 showed. The defensible version is: *these
ladders are thin, most of the cross-section is stale at any given minute, and no amount of keeping
more fields changes that.* Finding 18's structural criticisms stand; its implied hope that richer
ingest would rescue the result does not.

**What this still does not test.** Order-book depth and signed order flow, which the simulator has
and which are the features that actually predict short-horizon moves, are not in the candles at all
and cannot be backfilled. Testing those means recording depth forward from now. Given that 70% of
minutes show no quote movement whatsoever, the prior on finding much there should be low.

## 20. Staleness is not the mechanism. My inference was wrong, and the simulator says so.

Added 2026-08-12 from `results/staleness_sweep.json`, E10. **This refutes finding 19's causal
reading**, which was mine and which I stated with more confidence than the evidence supported.

Finding 19 measured that ~70% of Kalshi strike-minutes carry no quote movement, and concluded the
constraint is liquidity. The measurement stands. The *causal* step, that thinness is what killed the
objective, was an inference from two facts sitting next to each other, and it is now tested.

`observe_stale` freezes quotes in the simulator with probability `p`, compounding across buckets, so
an unlucky cell holds one stale quote for several minutes exactly as an untouched strike does. **The
latent is untouched**: target standard deviations are identical at every level (21.04, 37.80,
406.16), so games unfold as before and only observation degrades. Both arms see identically frozen
data, so what is tracked is the JEPA's *advantage*, not its absolute score.

| realised staleness | identity | JEPA | advantage |
|---|---|---|---|
| 0.000 | +0.8229 | +0.8823 | **+0.0594** |
| 0.201 | +0.8156 | +0.8685 | +0.0529 |
| 0.401 | +0.8101 | +0.8630 | +0.0529 |
| 0.550 | +0.7999 | +0.8440 | +0.0441 |
| **0.701** | +0.7830 | +0.8235 | **+0.0405** |
| 0.850 | +0.7303 | +0.7566 | +0.0263 |

Staleness costs roughly half the advantage across the full range and **never crosses zero**. Across
the Kalshi band the JEPA still beats raw features by **+0.0413 to +0.0358**, comparable to the
+0.0594 it enjoys on pristine data. Freezing quotes to real levels does not reproduce the real
failure. *(The "monotone decreasing" flag reads False only because 0.2 and 0.4 tie at +0.0529; the
trend is otherwise strictly down.)*

**So the sim-to-real gap remains unexplained after six attempts.** Ruled out: observation noise
(finding 12), cross-sectional redundancy (finding 11), the target set (finding 14), a genuine
cross-sectional latent (finding 17), ingest poverty (finding 19), and now staleness.

### The hypothesis that survives, stated as a hypothesis

The simulator and this programme were never doing the same task.

The simulator probes `score_diff`, `score_total` and `time_remaining` **at the window end**, which is
*current hidden state*. Prices are a deterministic function of that state plus noise, so recovering
it is an **inversion** problem: invert a known forward map through noise. Finding 12 is exactly what
inversion looks like, with the advantage *growing* as noise rises, because that is when inverting
beats reading the surface.

Every Kalshi target that mattered was a **forecast**: what happens next in a near-efficient market.
That is not inversion, and no representation can manufacture information the present does not carry.

The programme's own numbers fit. Ranked by matched-width lift at `last_patch`:

| lift | target | what it is |
|---|---|---|
| +0.1392 | `implied_width` | current state |
| +0.1202 | `time_to_expiry` | current state |
| +0.0433 | `vol_forecast_error` | future, **and E7c showed this lift was reconstruction** |
| +0.0233 | `fwd_realised_vol` | future |
| -0.0117 | `fwd_signed_return` | future |
| -0.0318 | `log_return_to_settle` | future |

The two current-state targets take three times the lift of the best future one, the only negative
lifts are on future targets, and the one future target that looked strong was shown by its own
control to be the encoder representing its input better. **Every real training lift in this
programme is on current state or on input reconstruction. Not one survives on a genuinely future
quantity.**

If that is right, the JEPA objective was never the wrong tool for a badly built corpus. It was the
right tool for a filtering problem, pointed at a forecasting problem. **This is untested.** It
predicts that a Kalshi target which is genuinely current-but-hidden, rather than future, would show
a surviving lift, and no such target has been built.

## 21. The current/forecast split is wrong. And the encoder does learn something real, on the target we never checked.

Added 2026-08-12 from `results/state_vs_forecast.json`, E11. **This refutes finding 20's
hypothesis and corrects this programme's headline claim.** Two results, and the second matters more.

### The hypothesis is refuted

Finding 20 predicted that current-but-hidden targets would show surviving residual lifts where
forward targets do not. Current and forward targets were run together, on the same windows,
encoders and probes, each residualised against the strongest raw control:

| target | kind | trained | untrained | lift | in sd | survives |
|---|---|---|---|---|---|---|
| `spot_denoise` | current | +0.0065 | +0.0151 | -0.0086 | -1.43 | no |
| `width_denoise` | current | +0.0274 | +0.0184 | +0.0090 | 2.24 | yes |
| `vol_state_error` | current | +0.0257 | +0.0151 | +0.0107 | 1.42 | no |
| **`fwd_realised_vol`** | **future** | **+0.1528** | **+0.1251** | **+0.0278** | **3.40** | **yes** |
| `fwd_signed_return` | future | +0.0009 | +0.0021 | -0.0011 | -0.36 | no |

One of three current targets survives and one of two forward ones does, and **the strongest
surviving lift is on a FUTURE target**, which is the opposite of the prediction. Current versus
forecast is not the axis. Finding 20's reading joins staleness, the target set, redundancy, noise
and ingest poverty on the refuted list.

### The result that matters: a lift finally survives

**`fwd_realised_vol` is the first target in this programme where a training lift survives
residualisation**, and it does so under all three conditions that E7c and E8c needed two failures to
get right: the same arm clears zero (t = 119.3), beats the best untrained arm, and exceeds 2 pooled
SD. It is not one lucky readout. Every readout shows it, at matched width:

| readout | dim | untrained | trained |
|---|---|---|---|
| `concat_strikes` | 3072 | +0.1251 | +0.1528 |
| `concat_patches` | 768 | +0.0600 | +0.1037 |
| `mean_max` | 256 | +0.0328 | +0.0944 |
| `mean_all` | 128 | +0.0471 | +0.0845 |
| `last_patch` | 128 | +0.0321 | +0.0730 |

Two separate things are visible here, and both are real. The **untrained** encoder already scores
+0.1251 on the residual, so the architecture as a random projection carries substantial information
about future realised volatility that the best raw control misses. And **training adds +0.0278 on
top of that**, after everything raw explains has been removed.

### The headline was measured on the wrong target

E7 reported "raw features still win" and E7c reported "reconstruction", and both were true **of
`vol_forecast_error`**, the target this programme designated as its headline. On `fwd_realised_vol`,
in the same E7 run, the numbers were already there and were never residualised:

| arm | dim | ridge R² |
|---|---|---|
| trained `concat_strikes` | 3072 | **+0.2292** |
| untrained `concat_strikes` | 3072 | +0.2013 |
| best raw (`raw_last_4min`) | 384 | +0.1557 |

The trained encoder beats the best raw control by +0.0735 there, and E11 now shows roughly +0.028 of
that survives removing everything raw explains. The reason this was missed is a choice made for good
reasons that turned out wrong: `vol_forecast_error` was designated the headline **because** it
looked cleaner, having the market's own forecast divided out, and `fwd_realised_vol` was discounted
in `causaljepa/targets.py` as "partly forecastable, so not a clean test on its own". The clean-looking
target was the one whose lift was reconstruction. The messy one carried the real signal.

### What this does and does not claim

It does **not** overturn finding 8: at 19,200 steps the probes still degrade, and this is measured at
600. It does not make anything tradeable; +0.028 residual R² on reconstructed one-minute volatility
is not an edge after costs. It is one corpus, one asset class, one venue, at one budget.

What it does establish is narrow and, after twenty findings of the opposite, worth stating plainly:
**on this corpus the JEPA objective extracts information about future realised volatility that the
raw cross-section does not carry, the effect is consistent across all five readouts, and it survives
the control that killed every previous candidate.**

## What this work did not establish

- ~~Whether a longer run separates the trained encoder from its random initialisation.~~
  **Closed by finding 8.** It does not. 32x more compute leaves ridge flat and drives the MLP
  probe 23 pooled standard deviations below the untrained floor.
- ~~Whether a target set with genuine hidden state would change the verdict.~~ **Closed by
  finding 14.** It does not. Targets verified discriminative before use, replicated at two
  horizons, produce a training lift that does not survive removal of what raw features explain.
- ~~Whether a target needing a genuine cross-sectional latent would change the verdict.~~
  **Closed by finding 17.** It does not, on the most favourable conditions this corpus offers.
- Whether any of this transfers off Kalshi hourly crypto. Two assets now, but one venue and one
  asset class. Finding 15 gives the cheap way to check before committing to another corpus.
- Whether realised correlation between paired ladders is *tradeable*. Finding 16 says it is
  recoverable at +0.3693, but mostly by correlation persistence rather than by anything in the
  ladder, and nothing here touches costs, capacity or execution.
- Whether the ladder cross-section carries *any* dependence information. The +0.0234 it adds over
  five plain numbers has a bootstrap interval straddling zero at n = 665 test windows. A larger
  paired corpus could resolve it; this one cannot.
- Whether the marginal causal-context effect in finding 4 survives more seeds. At t = 2.56 with
  n = 4 and six comparisons, it probably does not.
- Anything about trading. No arm beats the raw cross-section, so none of this is deployable.
