# causal-jepa

**600 steps of JEPA training on this corpus buys linear accessibility and no new information.**

An untrained encoder of identical architecture, no optimiser, no steps, no EMA, mean-pooled over the
same token grid, scores ridge **+0.3456 +/- 0.0067** and MLP **+0.3802 +/- 0.0049** on the
frozen-representation probe. The trained `ctx=bidir tgt=bidir` arm, which is `pm-jepa`'s
configuration reproduced in this code, scores ridge **+0.3919 +/- 0.0058** and MLP
**+0.3797 +/- 0.0049**. So training moves the ridge probe by **+0.0463** (Welch t = 9.09) and the MLP
probe by **-0.0005** (t = -0.12). Same architecture, same corpus, same probe protocol, same seeds.
Source: `results/benchmark.json`, written by `scripts/benchmark.py`.

The reading we can defend: the objective is not adding information to the representation, it is
rearranging information the random projection already carried into a more linearly accessible form.
A nonlinear probe sees no difference at all. The strongest objection is that a random transformer
over structured input is a strong random-features baseline and 600 steps may simply be too few to
beat it; we cannot rule that out from this data, and a longer run is the highest-value next
experiment in this repo. What we can say is that whatever the JEPA learned in 600 steps is invisible
to a nonlinear probe, and that no published `pm-jepa` result ever ran this control.

Everything here still loses to the raw cross-section. Identity is +0.4491 ridge / +0.5050 MLP; the
best arm in this repo is +0.4018 / +0.3872.

---

## The question this repo was built to answer, and its answer

**Hypothesis, tested and refuted.** `pm-jepa` trains its target encoder with unmasked attention over
the whole 24-minute window, so under temporal masking the target at a *visible* patch is computed
with attention over the *masked* patches too, and the copy oracle it is scored against is
future-contaminated. We suspected the copy solution it found was partly an artefact of the model
rather than a property of the market. This repo forks that model, makes the time axis of the context
tower and the target tower independently causal, and runs the 2x2.

Making the target tower causal removes the contamination entirely and changes nothing that matters:

| metric (temporal masking) | tgt=bidir | tgt=causal | delta | t |
|---|---|---|---|---|
| `copy_alignment` | +0.9641 +/- 0.0040 | +0.9616 +/- 0.0045 | -0.0025 | -0.73 |
| ridge R^2 | +0.3919 +/- 0.0058 | +0.3924 +/- 0.0069 | +0.0005 | +0.09 |
| MLP R^2 | +0.3797 +/- 0.0049 | +0.3809 +/- 0.0055 | +0.0012 | +0.28 |

Copy alignment stays pinned at 0.96. The probe difference is `t = 0.09`, which is as null as a result
gets. Sources: `results/causal_ablation.json`, `results/benchmark.json`. All `t` in this document are
Welch on 4-seed samples, recomputed from the results files; `|t| > 2.4` is roughly p < 0.05
two-sided at df ~ 6, and with four seeds and several comparisons every marginal `t` is suggestive
and nothing more.

**The leak is real, it is just too small to matter.** A weight-matched evaluation, identical weights
read out both ways with no retraining, shows the two target towers genuinely differ, most at early
patches where causality removes the most context:

| patch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| cos(bidir, causal) | 0.9812 | 0.9902 | 0.9943 | 0.9969 | 0.9985 | 0.9995 |
| relative L2 | 0.1271 | 0.0896 | 0.0657 | 0.0481 | 0.0315 | 0.0163 |

Cross-boundary target autocorrelation falls from 0.9483 to 0.9377 under temporal masking, so the
bidirectional target encoder does inflate target similarity across the mask boundary. But 0.011 of
autocorrelation is not what was holding copy alignment at 0.96. Source:
`results/h1_weight_matched.json`.

Everything here reads pm-jepa's corpus, probes and diagnostics in place. Nothing under
`/Users/nikita/pm-jepa` is written to, and `tests/test_no_data_mutation.py` fingerprints the tree to
keep it that way.

---

## Status

| | |
|---|---|
| Factorised (strike x time) encoder with a switchable causal axis | Done |
| KV cache for the time-attention sublayers | Done, exact to 1.9e-06 |
| `interpolation` masking, the arm the intervention is aimed at | Done |
| Directional copy oracle and target autocorrelation | Done |
| **2x2 x 4 seeds x 2 strategies, 600 steps, real corpus** | **Done, 32 arms, 43.5 min on MPS** |
| **Untrained random-encoder control, 2 flags x 4 seeds** | **Done, about 60s on MPS** |
| **Headline: 600 steps of training vs its own initialisation** | **ridge +0.0463 (t = 9.09), MLP -0.0005 (t = -0.12)** |
| **H1 leakage** | **Inconclusive; the stated metric does not survive its own control** |
| **H2 intervention** | **Mechanism supported, stated metric refuted** |
| **H3 utility** | **Refuted for the target flag. Every arm still loses to identity** |
| **H4 cache** | **Exact in all 8 configurations; 1.22x at the real shape, and the streaming case that motivated it cannot use it** |
| Test suite | 49 passing in 6.9s |
| Money spent | **$0** |

Corpus: 25,818 windows of shape `(24 minutes, 24 strikes, 4 channels)` from pm-jepa's cached snapshot,
split **by event** into 20,506 train and 5,312 test. Model: 994,336 trainable parameters,
`d_model=128`, 4 layers, 8 heads, `patch_length=patch_stride=4`, so P=6 patches x K=24 strikes = 144
tokens.

---

## Benchmark: every arm against the controls that matter

Ridge and MLP R^2, mean over the four probe targets, mean +/- sd over seeds. The controls are
pm-jepa's, measured in `pm-jepa/results/baselines.json` before any model existed, on the identical
corpus and the identical event split. Recomputed into `results/benchmark.json` by
`scripts/benchmark.py`.

| arm | ridge R^2 | MLP R^2 |
|---|---|---|
| temporal (control) | +0.4600 | +0.5108 |
| handcrafted (control) | +0.4595 | +0.4858 |
| identity (control) | +0.4491 | +0.5050 |
| ctx=causal tgt=bidir (best trained, this work) | +0.4018 +/- 0.0034 | +0.3872 +/- 0.0192 |
| ctx=bidir tgt=bidir (pm-jepa configuration, our code) | +0.3919 +/- 0.0058 | +0.3797 +/- 0.0049 |
| jepa-temporal (pm-jepa, PatchTST) | +0.3771 +/- 0.0095 | +0.3227 +/- 0.0128 |
| **random-encoder (bidir), untrained** | **+0.3456 +/- 0.0067** | **+0.3802 +/- 0.0049** |
| jepa-slate (pm-jepa, PatchTST) | +0.3426 +/- 0.0059 | +0.3160 +/- 0.0042 |
| jepa-contiguous (pm-jepa, PatchTST) | +0.3394 +/- 0.0064 | +0.3263 +/- 0.0045 |
| random-encoder (causal), untrained | +0.3381 +/- 0.0079 | +0.3724 +/- 0.0076 |

Three things to read off it.

1. **Identity beats our best arm by +0.0473 ridge (t = 23.77) and +0.1178 MLP (t = 10.62).** The gap
   is not closing and this work did not close it. The controls have no seed spread because they are
   deterministic feature maps, so those `t` values use the arm's spread alone.
2. **The untrained encoder ties every trained arm on the MLP probe.** +0.3802 against +0.3797 for the
   pm-jepa configuration and +0.3872 for our best arm (a gap of +0.0070 at t = 0.61). Across all
   eight trained arms the largest separation from the untrained encoder on the MLP probe is
   `t = 1.09`. Training separates only on ridge.
3. **The `pm-jepa` rows are architecture-confounded.** Those arms use a PatchTST backbone; ours uses
   a hand-written factorised encoder, so any comparison across that line mixes architecture with
   everything else. The within-architecture comparisons, trained versus untrained and the 2x2 cells
   against each other, are the clean ones. What is *not* confounded is that all three `pm-jepa` arms
   score below our untrained encoder on the MLP probe, +0.3160 to +0.3263 against +0.3802.

---

## The four hypotheses, results first

### H1 (leakage): INCONCLUSIVE, and the stated metric does not survive its own control

**Predicted:** a causal target encoder lowers target-to-target similarity across the mask boundary.

Raw target autocorrelation falls exactly as predicted, in every seed of both strategies. Then the
control eats it. `baseline_cos`, the anisotropy floor that `diagnostics.py` warns must be subtracted
before any cosine is read as leakage, falls almost as far.

Paired per-seed contrast, causal target minus bidirectional target, n=8 (2 context levels x 4 seeds):

| quantity | temporal delta (t, signs) | interpolation delta (t, signs) | predicted |
|---|---|---|---|
| autocorr `cross_boundary` | -0.0101 (-4.75, 0+/8-) | -0.0115 (-8.91, 0+/8-) | lower, yes |
| *floor: `baseline_cos`* | *-0.0090 (-3.73)* | *-0.0155 (-5.55)* | *n/a* |
| **`cross_boundary` over floor** | **-0.0012 (-1.53, 2+/6-)** | **+0.0040 (+1.55, 5+/3-)** | lower, **no** |
| **`lag_3` over floor** | **-0.0025 (-3.12, 0+/8-)** | -0.0016 (-0.59, 3+/5-) | lower, partly |
| `boundary_excess` | **+0.0078 (+4.55, 8+/0-)** | -0.0007 (-4.15, 1+/7-) | ~0 if mask-blind |

Floor-corrected, `cross_boundary` is null in temporal and the *wrong sign* in interpolation. Only
temporal `lag_3` over floor survives, at -0.0025, roughly a fifth of the raw number. The reading is
that a causal target encoder changes the embedding geometry **globally** rather than specifically
across the mask boundary. H1's mechanism may well be real; the measurement as specified conflates it
with an isotropy shift. **The H1 null is "not detectable here", not "shown to be zero".**

The one place the H1 story does get support is `boundary_excess`, which per the diagnostics docstring
is the signature of boundary-specific rather than window-wide leakage. It moves +0.0078 in temporal,
8 seeds out of 8. It does not move that way in interpolation.

### H2 (intervention): mechanism SUPPORTED, stated metric REFUTED

**Predicted:** `copy_alignment` falls and `copy_loss_ratio` rises, i.e. the copy baseline gets
genuinely harder to beat.

| quantity | temporal delta (t, signs) | interpolation delta (t, signs) | predicted |
|---|---|---|---|
| `copy_alignment` | -0.0013 (-2.42, 2+/6-) | **-0.0066 (-6.42, 0+/8-)** | lower, yes |
| `copy_loss_ratio` | -0.0332 (-4.61, 0+/8-) | -0.0486 (-7.28, 0+/8-) | higher, **no** |
| numerator, model loss | -0.00003 (-0.16, ns) | +0.0033 (+7.09, 8+/0-) | n/a |
| denominator, copy-oracle loss | **+0.0022 (+4.17, 8+/0-)** | **+0.0083 (+8.47, 8+/0-)** | n/a |
| `interp_advantage` | n/a, one-sided mask | **+0.0040 (+6.86, 8+/0-)** | lower, **no** |

`copy_alignment` falls as predicted, and it falls **5x harder in the interpolation arm**, which is
exactly the strategy ordering H2 called for: that is the arm where a bidirectional target has a
future side to copy from.

`copy_loss_ratio` falls where H2 said it would rise, 8/8 in both strategies. The decomposition
vindicates the mechanism and indicts the metric: the copy **oracle's own loss rose** 8/8 in both arms,
so copying really did get harder. The ratio fell only because the model's loss rose by less. "The copy
baseline gets genuinely harder to beat" is true; "`copy_loss_ratio` rises" is false, because a ratio
cannot express it.

The sharpest negative is `interp_advantage`, which went **up** where H2 predicted down, +0.0040 with
t=+6.86 and all 8 seeds agreeing. The mechanism is clean and it is a design finding, not noise: the
target encoder is called with `strike_mask=None`, so a causal target at a **post-hole** patch has
already integrated the hole's own minutes, while a causal target at a **pre-hole** patch cannot.
Making the target causal therefore *strengthens* the forward reference relative to the backward one.
**A causal target does not close the interpolation escape hatch. Only a causal context can.**

### H3 (utility): REFUTED for the target flag, and the trained/untrained gap is the real story

**Predicted:** open question, negative result acceptable, do not tune to make causal win.

Paired probe delta for the target flag is null in both strategies: -0.0001 (t=-0.24) temporal,
+0.0011 (t=+0.62) interpolation. Fixing the leak costs nothing and buys nothing.

The flag that *does* move probe R^2 is the one the experiment was not aimed at. Causal **context**
minus bidirectional context, paired by seed:

| quantity | temporal | interpolation |
|---|---|---|
| probe `ridge_mean` | **+0.0093 (t=+6.11, 8+/0-)** | +0.0043 (t=+1.89, 6+/2-) |
| `eff_rank` | +2.21 (t=+5.56, 8+/0-) | +3.96 (t=+10.47, 8+/0-) |
| `copy_alignment` | -0.0089 (t=-12.15, 0+/8-) | -0.0019 (t=-3.03, 0+/8-) |

The mechanism is at least coherent: under temporal masking the causal-context arm's loss on copyable
positions rises from 0.0353 to 0.0546, a 55% harder prediction task, and `eff_rank` rises from 34.85
to 37.27. But the numbers above are the **paired** contrast. The unpaired Welch on the same two cells
is `t = 2.56`, which is the number to quote if the pairing is not trusted, and read conservatively
that is one marginal ridge effect among six comparisons at n=4. We would not defend it as significant
without more seeds. Both flags are dwarfed by the trained-versus-untrained contrast at the top of this
file, and every one of the 8 arms loses to both of pm-jepa's controls.

### H4 (cache): SUPPORTED as stated, useless as motivated

**Predicted:** incremental encoding is exact for a causal encoder and faster for streaming inference.

Exact in all 8 shape/device configurations at a worst case of 1.9e-06, roughly 50x inside the 1e-4
bar, and faster in all 8. `scripts/integration_check.py` and `tests/test_cache_exactness.py` both run
the has-teeth control: the same cached path against a **bidirectional** forward misses by 2.673176,
which is 26,732x the bar at that script's shapes.

The speedup at this model's actual shape, P=6, is 1.22x on MPS. More to the point, the cache is only
valid for an **append-only patch stream**: with `patch_stride=4` a one-minute slide of the window
rewrites the contents of every patch, so the streaming case that motivated the work, a new market
tick arriving and the window sliding by one minute, cannot use the cache at all. We built a correct
answer to a question this model does not ask.

---

## Probe R^2 by masking strategy

Ridge R^2, mean over 4 targets, mean +/- sd over 4 seeds. Controls are pm-jepa's, recomputed here on
the identical corpus and identical event split.

| arm | temporal | interpolation |
|---|---|---|
| **identity (control)** | **+0.449** | **+0.449** |
| **handcrafted (control)** | **+0.460** | **+0.460** |
| pm-jepa's `jepa-temporal` (PatchTST, 3 seeds), architecture-confounded | +0.377 | no such arm |
| ctx=bidir, tgt=bidir *(pm-jepa's configuration, reproduced)* | +0.392 +/- 0.006 | +0.390 +/- 0.007 |
| ctx=bidir, tgt=causal *(the key cell: fixes the leak, context unchanged)* | +0.392 +/- 0.007 | +0.394 +/- 0.009 |
| ctx=causal, tgt=bidir | **+0.402 +/- 0.003** | **+0.397 +/- 0.004** |
| ctx=causal, tgt=causal *(fully streamable)* | +0.401 +/- 0.004 | +0.395 +/- 0.004 |
| random-encoder (bidir), untrained; no masking strategy applies | +0.346 | +0.346 |

**Every arm loses to both controls.** The best is -0.047 below identity. Our arms do sit above
pm-jepa's PatchTST arms, but that comparison crosses an architecture boundary and says nothing about
causality; the target flag alone moves R^2 by -0.0001.

Per-target breakdown, temporal strategy, so the mean is not hiding a trade:

| arm | log_return | time_to_expiry | implied_width | window_return | **mean** |
|---|---|---|---|---|---|
| identity | +0.015 | +0.739 | **+0.803** | **+0.238** | **+0.449** |
| handcrafted | +0.020 | **+0.822** | +0.759 | +0.237 | **+0.460** |
| ctx=bidir tgt=bidir | +0.011 | +0.763 | +0.652 | +0.141 | +0.392 |
| ctx=bidir tgt=causal | +0.010 | +0.766 | +0.652 | +0.141 | +0.392 |
| ctx=causal tgt=bidir | +0.009 | +0.767 | +0.654 | +0.177 | +0.402 |
| ctx=causal tgt=causal | +0.009 | +0.765 | +0.654 | +0.176 | +0.401 |
| random-encoder (bidir), untrained | -0.000 | +0.678 | +0.579 | +0.126 | +0.346 |

This is pm-jepa's shape exactly: the JEPA buys a little on `time_to_expiry`, the target that needs the
ladder's shape integrated over time, and pays for it on `implied_width` and `window_log_return`.
Causality does not change that trade. Against the untrained encoder, training's ridge gain is almost
entirely `time_to_expiry` +0.085 and `implied_width` +0.073, the two targets the identity control
already scores highest on, with +0.016 on `window_log_return` and +0.011 on `log_return_to_settle`.
`log_return_to_settle` remains dead for every method including both controls, which is the correct
null for an efficient market and is what pm-jepa measured first, before any model existed.

## Copy diagnostics at the final step, mean +/- sd over 4 seeds

| arm | copy_align | copy_ratio | interp_ratio | interp_adv | autoc_xbnd | floor_cos | eff_rank |
|---|---|---|---|---|---|---|---|
| **temporal** | | | | | | | |
| ctx=bidir tgt=bidir | +0.964 +/-0.004 | 0.787 +/-0.044 | n/a | n/a | +0.9441 | +0.6551 | 34.9 |
| ctx=bidir tgt=causal | +0.962 +/-0.005 | 0.735 +/-0.047 | n/a | n/a | +0.9292 | +0.6407 | 35.5 |
| ctx=causal tgt=bidir | +0.954 +/-0.003 | 0.813 +/-0.049 | n/a | n/a | +0.9079 | +0.6453 | 37.3 |
| ctx=causal tgt=causal | +0.954 +/-0.003 | 0.798 +/-0.047 | n/a | n/a | +0.9025 | +0.6417 | 37.5 |
| **interpolation** | | | | | | | |
| ctx=bidir tgt=bidir | +0.961 +/-0.005 | 0.771 +/-0.033 | 0.810 +/-0.021 | +0.0026 | +0.9464 | +0.6425 | 28.6 |
| ctx=bidir tgt=causal | +0.954 +/-0.008 | 0.705 +/-0.034 | 0.799 +/-0.019 | +0.0072 | +0.9334 | +0.6311 | 29.5 |
| ctx=causal tgt=bidir | +0.959 +/-0.006 | 0.726 +/-0.044 | 0.766 +/-0.019 | +0.0032 | +0.9404 | +0.6551 | 31.9 |
| ctx=causal tgt=causal | +0.953 +/-0.008 | 0.694 +/-0.042 | 0.773 +/-0.022 | +0.0067 | +0.9304 | +0.6355 | 34.1 |

**Martingale collapse replicates for a third time.** `slate-jepa` measured `copy_alignment` 0.964 to
0.997 across its arms on synthetic sports slates. `pm-jepa` measured 0.966 on real Kalshi ladders
with PatchTST and SIGReg. This project measures **0.964 +/- 0.004** on the same ladders with a
factorised encoder, no regulariser, and a causal option on both towers. Different encoder, no
regulariser, same number. The best any arm here does is 0.953, and it is the *interpolation* arm with
both towers causal.

A representative trace, `ctx=bidir tgt=bidir`, temporal, seed 0:

```
step     1   pred 0.4577   copy_align +0.072   rank 35.7
step   100   pred 0.0636   copy_align +0.933   rank 22.6
step   300   pred 0.0382   copy_align +0.966   rank 31.3
step   600   pred 0.0294   copy_align +0.969   rank 33.0
```

Prediction loss falls 15.6x, effective rank recovers to 33 of 128, `dead_dims` is 0.0 in every one of
the 224 history records, and the model is emitting a copy. Every conventional signal says the run went
well, and the probe still cannot separate the result from an untrained encoder except linearly.
`reg_grad_norm` is likewise exactly 0.0 in all 224 records, which is how we know no inert regulariser
crept back in: SIGReg in pm-jepa was computed on a detached target and contributed exactly zero
gradient, and `train.py` here deliberately does not port it.

## H4 cache benchmark

Append-only patch stream, one new patch per tick. B=1, K=24, d=128, 4 layers, 3 repeats.

| device | P | exact | max abs diff | full ms/tick | cache ms/tick | speedup | tok/s full -> cache |
|---|---|---|---|---|---|---|---|
| mps | **6 (real shape)** | yes | 0.0e+00 | 8.607 | 7.001 | **1.22x** | 2,852 -> 3,474 |
| mps | 32 | yes | 1.4e-06 | 9.703 | 8.376 | 1.16x | 2,503 -> 2,903 |
| mps | 128 | yes | 1.4e-06 | 10.811 | 8.397 | 1.26x | 2,276 -> 2,859 |
| mps | 512 | yes | 1.9e-06 | 166.930 | 39.885 | 4.19x | 144 -> 602 |
| cpu | **6 (real shape)** | yes | 1.4e-06 | 2.766 | 1.832 | **2.15x** | 6,012 -> 12,942 |
| cpu | 32 | yes | 1.4e-06 | 9.530 | 1.793 | 4.09x | 2,301 -> 9,414 |
| cpu | 128 | yes | 1.5e-06 | 19.087 | 2.452 | 7.65x | 1,237 -> 9,455 |
| cpu | 512 | yes | 1.7e-06 | 156.104 | 5.048 | **30.92x** | 154 -> 4,754 |

At the real shape the win is small, and that is the honest headline: five patches of history is not
enough work to amortise per-call overhead, and on MPS kernel-launch latency dominates until P=512.
**The scaling is the only claim worth making**, not the level: CPU goes 2.15x, 4.09x, 7.65x, 30.92x at
P = 6, 32, 128, 512.

The caveat is recorded inside `results/cache_bench.json` as well as here: this is an **append-only**
stream. `patch_stride=4` means a 1-minute slide rewrites every patch and invalidates the cache
completely, so these numbers do not transfer to a minute-level sliding window.

---

## Layout

```
causaljepa/model.py         FactorisedEncoder (strike attention, then time attention) + CausalJEPA.
                            causal_context and causal_target are separate flags; that 2x2 is the experiment
causaljepa/cache.py         KVCache over the TIME-attention sublayers only; strike attention needs none
causaljepa/masking.py       pm-jepa's samplers verbatim, plus `interpolation`: a patch-aligned hole in
                            the MIDDLE of the window, so a bidirectional encoder can interpolate across it
causaljepa/diagnostics.py   pm-jepa's copy oracle and effective rank verbatim, plus
                            directional_copy_diagnostics (extrapolation vs interpolation) and
                            target_autocorrelation (the H1 measurement, with its anisotropy floor)
causaljepa/data.py          pm-jepa's corpus by path injection, read-only; split_by_event
causaljepa/probes.py        pm-jepa/model/probes.py verbatim; any deviation makes the numbers incomparable
causaljepa/train.py         one cell of the 2x2; optimiser, schedule and EMA ramp match pm-jepa exactly

scripts/experiment.py       the 2x2 driver, writes results/<name>.json with the config inlined
scripts/benchmark.py        the arms against the controls that matter, including the UNTRAINED encoder;
                            writes results/benchmark.json. No training, probes only
scripts/h1_weightmatched.py H1 with the training trajectory removed: one arm, its target tower read out
                            both ways on fixed weights; writes results/h1_weight_matched.json
scripts/cache_bench.py      H4 sweep over shapes and devices
scripts/report.py           recomputes the 2x2 tables in this README from results/*.json; nothing is retyped
scripts/timing_probe.py     seconds-per-step at the real corpus size, used to pick 600 steps x 4 seeds
scripts/smoke.py            end-to-end run on a 300-window subset, every code path
scripts/integration_check.py  cross-module check written against the SPEC, not against any module's fixtures
scripts/verify_*.py         standing per-module verification harnesses

tests/                      the seven mandatory SPEC tests; 38 test functions, 49 cases after
                            parametrisation, 6.9s measured
SPEC.md                     the binding interface contract
FINDINGS.md                 the scientific record, written after all results were in
docs/EVALS.md               what each instrument measures and what it cannot tell you, including
                            the untrained-encoder control
docs/REPRODUCE.md           exact commands, expected runtimes, expected outputs
```

Always import through `causaljepa.<module>`. `causaljepa.data` puts pm-jepa's repo root at the front
of `sys.path`, because pm-jepa's `load_corpus.py` does `from data import dataset` and that absolute
import has to resolve inside their tree. From that moment on, in the same process, a bare
`import train` resolves to `/Users/nikita/pm-jepa/train.py`. Verified, not hypothetical; nothing
raises, and the two `train` modules have similarly named functions with different defaults. The root
`conftest.py` puts this repo first for the tests.

## Install

```bash
pip3 install -r requirements.txt
```

`torch==2.6.0`, `numpy==1.26.2`, `pytest`, and `pandas` (pm-jepa's `data/dataset.py` imports it; we
never call it, but the corpus cannot be loaded without it). The pins are exact in practice, not just
in spirit: torch 2.6 has no MPS kernel for `linalg.svdvals` or `cummax`, and both `diagnostics.py` and
`train.py` route around that by moving to CPU. A newer torch that grew those kernels would change
which device the diagnostics ran on, and reduction order differs between backends at the third
decimal, which is the resolution the 2x2 is read at.

The corpus is **not** rebuilt or copied. It is read in place from
`/Users/nikita/pm-jepa/data_cache/event_arrays/*.npz`; `.gitignore` blocks `*.npz` and `data_cache/`
so a stray second copy cannot drift from the snapshot pm-jepa's published baselines were measured on.

## Reproduce every number above

```bash
# 49 tests, 6.9s measured. Includes the two has-teeth controls: cache exactness must FAIL
# for causal=False, and perturbing patch p must move earlier patches under causal=False.
python3 -m pytest tests/ -q

# cross-module wiring, cache exactness recomputed from the public API, causality perturbation
python3 scripts/integration_check.py

# every code path on a 300-window subset, 30 steps, d_model 64
python3 scripts/smoke.py

# the budget model that chose 600 steps x 4 seeds
python3 scripts/timing_probe.py

# the two experiment files. 21m58s and 21m33s measured on MPS, 43.5 min total.
# All other settings are defaults: batch 64, lr 1e-3, d_model 128, 4 layers,
# n_held_out 6, window 24, stride 4, test_frac 0.2, split_seed 0.
python3 scripts/experiment.py --strategy temporal      --seeds 4 --steps 600 --name causal_ablation
python3 scripts/experiment.py --strategy interpolation --seeds 4 --steps 600 --name interpolation_arm

# the headline table, including the UNTRAINED encoder control. About 60 seconds on MPS,
# no training at all: 8 probe fits (2 causal flags x 4 seeds) plus the saved arms and controls.
# Writes results/benchmark.json.
python3 scripts/benchmark.py --seeds 4

# H1 with the training trajectory removed: one arm, target tower read out both ways
# on identical weights. 53s measured on MPS. Writes results/h1_weight_matched.json.
python3 scripts/h1_weightmatched.py

# H4
python3 scripts/cache_bench.py --shapes 6,32,128,512 --devices mps,cpu --repeats 3

# the 2x2 tables in this README, recomputed from the saved JSON without retraining anything
python3 scripts/report.py
```

`results/causal_ablation.json`, `results/interpolation_arm.json` and `results/cache_bench.json` carry
the full per-seed history with the config inlined, so `report.py` alone reproduces the 2x2 readout,
and `results/benchmark.json` carries the per-seed probe fits for the untrained control.
`results/_driver-exercise-60steps-not-a-result.json` is a driver exercise and is named so that nobody
quotes it.

**`--steps` below about 100 produces a fake null.** `WARMUP_STEPS=100` is a fixed constant, kept for
parity with pm-jepa, while the cosine decay scales to `--steps`, so short runs overlap the two and
never reach a real learning rate: the peak multiplier is 0.085 at 30 steps, 0.531 at 200, 0.963 at
800. Every arm then finishes near its initialisation, where a pre-norm residual stream barely mixes
across patches and the causal and bidirectional cells agree to 3 or 4 decimals. `train.py` warns at
`steps < 100` and records `lr_mult` in every history entry so a flat results file can be diagnosed
without rerunning it.

---

## What we did NOT establish

- **That the trained encoder carries information its own initialisation did not.** The MLP probe
  cannot tell them apart: -0.0005 at t = -0.12 for the pm-jepa configuration, +0.0070 at t = 0.61 for
  our best arm. This rests on 600 steps and it is the main threat to every other conclusion here. A
  longer run where MLP R^2 separates from the untrained baseline by more than 2 seed-standard
  deviations would falsify it, and that run is the single highest-value next experiment in this repo.
- **That a causal target encoder removes boundary-specific leakage.** Floor-corrected, the H1 effect
  is null in temporal (t=-1.53) and the wrong sign in interpolation (+0.0040). The one surviving
  signal, temporal `lag_3` over floor at -0.0025, and the one supporting statistic, `boundary_excess`
  at +0.0078, are both around the resolution this protocol reaches. **The H1 null is "not detectable
  here", not "shown to be zero".**
- **That `copy_loss_ratio` measures what H2 said it measures.** It went the wrong way, 8/8, in both
  strategies. A ratio cannot distinguish "the model got better" from "the oracle got worse", and here
  both terms rose. Report the numerator and denominator, which the history already stores.
- **That anything closes the interpolation escape hatch.** A causal target makes `interp_advantage`
  *larger*, because the target encoder is called with `strike_mask=None` and so a causal target at a
  post-hole patch has already seen the hole's minutes. Causal context is the untested candidate.
- **That the context-flag probe gain is about causality.** +0.0093 (t=+6.11, 8/8) is real and
  consistent, but it was not what the experiment was designed to isolate, no mechanism was tested, and
  it does not change the verdict: the best arm is still 0.047 below identity.
- **That 600 steps is enough.** The target-flag effects are 0.005 to 0.01 against a between-seed
  spread of about 0.007. The paired design is what makes them readable at all; the unpaired table in
  `report.py` shows most of them as noise. With 4 seeds the reported t values are
  direction-and-consistency gauges, not p-values to quote.
- **That the predictor's directionality does not matter.** The predictor is bidirectional in all four
  arms. The SPEC does not ask otherwise, and making it causal would confound H1 with H2.
- **That the cache helps a real sliding window.** It was benchmarked append-only. At `patch_stride=4`
  a 1-minute slide rewrites every patch and invalidates the whole cache.
- **That any of this generalises past `temporal` and `interpolation`.** `slate`, `contiguous`, `mixed`
  and `random` are ported and tested but were never run under the causal 2x2.
- **That any of this transfers off the Kalshi hourly-crypto corpus.** One corpus, one asset class, one
  venue, one event split.
- **Anything about trading.** No arm beats the raw cross-section, so none of this is deployable.
- Two diagnostic columns are redundant for these two strategies and should not be read as independent
  evidence. `extrap_loss_ratio == copy_loss_ratio` exactly, because `nearest_visible_reference` prefers
  the past and every masked position has a past reference. `boundary_excess_matched` is `nan`, because
  the mask is identical for every `(batch, strike)` and there is no in-sample control.
  `interp_advantage` never touches `pred` at all; it is oracle minus oracle, a pure target-side
  statistic, so two cells sharing `causal_target` report identical values early in training.
- The interpreter here is **CPython 3.9.6**, not the 3.11+ of SPEC section 10. No module uses PEP 604
  annotations, which is what makes that work; verified, not assumed.
- Everything pm-jepa's own honest-limits section says still applies. One sector, crypto only, forced
  by a rolling ~10-week retention window; roughly 1M parameters; 24-minute windows on a 24-point
  moneyness grid; a single corpus snapshot and a single event split (`split_seed=0`, held constant
  across arms and seeds on purpose, because resplitting per seed would confound arm with split).
