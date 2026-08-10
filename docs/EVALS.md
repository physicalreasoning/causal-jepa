# EVALS.md: what we measure, why, and what each measurement cannot tell us

This is the protocol document for `causal-jepa`. Every number in `results/*.json`
comes out of one of the five instruments below, applied to either a trained or an
untrained encoder. For each instrument: what it computes, what failure mode it was
built to catch, and, at equal length, what it does not license you to say. The
last section is the standing list of limitations of the suite as a whole.

The house rule this document exists to enforce: **an instrument that moves is not
a result until you know what its floor did.** Three of the five instruments here
have a floor that moves with the intervention, and in two of the three the floor
ate most of the effect.

The probe has a floor too, and it is the one that decided this repo's headline: an
**untrained encoder of the identical architecture** (section 2, "The untrained
encoder"). Measured against it, 600 steps of training move the ridge probe by
+0.0463 and the MLP probe by -0.0005. Read that subsection before any other number
in this file.

---

## 0. Why a JEPA cannot be evaluated by its own loss

`train.py` minimises smooth-L1 between the predictor's output and the EMA target
encoder's embedding at masked positions. Both sides of that comparison live in a
space the model controls. Two degenerate solutions score perfectly:

- **Representational collapse.** Every input maps to the same vector. Loss goes
  to zero and nothing is encoded.
- **Martingale collapse.** Kalshi ladder prices are martingales, so
  `E[z_{t+dt} | F_t] ~ z_t`, and the identity map drives the loss down without
  encoding any state. The identity map has *full* rank and *full* per-dimension
  variance, so it passes every standard SSL collapse check.

The first is caught by `effective_rank` and `dimension_stats` in
`causaljepa/diagnostics.py`. The second is invisible to both, which is why the
copy oracle (section 3) exists. Neither of them establishes that anything useful
was learned, which is why the probe (section 1) exists and is the only number
that speaks to representation quality.

`pred_loss` is logged, and it is used for exactly one purpose: confirming that a
run trained at all. In `results/causal_ablation.json` it falls 0.458 to 0.029
over 600 steps. A run where it does not fall is a broken run, not a null result.
It is never compared between arms, because the arms have different target
encoders and therefore different targets; a lower loss against an easier target
means nothing.

---

## 1. The frozen-representation probe

### What runs

```
model.represent(obs)            context encoder, strike_mask=None,
                                mean over the (P=6, K=24) token grid  -> (B, 128)
represent_all(...)              eval mode, batched at 256, cast to float64
probes.run_probes(...)          ridge and MLP on the frozen features
```

The encoder is frozen and never sees the probe targets. The features are the
*context* tower, not the target tower, because the context tower is the artefact
a downstream user would deploy.

`represent_all` casts to float64 before the probe. This is not cosmetic: at
d_model=128 the ridge normal equations are badly conditioned, and float32 there
moves R^2 in the third decimal, which is the resolution the 2x2 is read at.

### The ridge probe

Ported byte-for-byte from `pm-jepa/model/probes.py`; `tests/test_probe_parity.py`
asserts agreement to 1e-9 against the original loaded off disk, and also asserts
that the ported source text is verbatim. The reason for that severity is that the
whole point of this repo is to put its arms next to pm-jepa's published numbers.
A probe that differs in its ridge grid, its standardisation, or where it cuts the
validation split produces R^2 on a different scale, and the comparison silently
becomes meaningless rather than failing loudly.

- Features standardised with the **train** mean and standard deviation (+1e-6),
  applied unchanged to test. Targets standardised the same way and un-standardised
  before scoring.
- Lambda chosen on a held-out **tail** of train, the last 20% of rows, over the
  fixed grid `(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)`.
- Closed form, `torch.linalg.solve` in float64, refit on all of train at the
  selected lambda.
- R^2 per target, then the unweighted mean over the four targets. `ridge_mean` is
  the headline.

### The MLP probe

2x256 GELU, AdamW `lr=1e-3 wd=1e-4`, 800 steps, batch 512, seed 0, on CPU.

It is reported alongside ridge to separate two different claims: "the information
is present" (MLP) from "the information is linearly accessible" (ridge). It is
not a second headline. In practice on this corpus the MLP probe *underperforms*
ridge on learned features (+0.38 vs +0.39 in every arm) while *outperforming* it
on raw features (identity +0.505 vs +0.449), which is itself informative and is
discussed under headroom below.

That separation is what produced this repo's headline. Against an untrained
encoder of the same architecture, 600 steps of training move ridge by +0.0463 and
the MLP probe by -0.0005, i.e. the training changed linear accessibility and not
the information content the MLP can reach. See section 2, "The untrained
encoder".

### The split

`data.split_by_event(owner, frac=0.2, seed=0)` holds out **whole events**:
25,818 windows over 3,127 events becomes 20,506 train / 5,312 test.

Windows inside one event overlap by `window - stride` = 20 of 24 minutes. A
per-window split therefore puts near-duplicate rows on both sides and inflates
every R^2 reported. Splitting on the event id is the only version of this that
measures generalisation to an unseen hour of the market.
`tests/test_no_data_mutation.py::test_split_matches_pm_jepa` asserts the masks
are bit-identical to pm-jepa's own splitter, so the controls and the arms are
scored on the same rows.

`split_seed` is held at 0 across every arm and every seed on purpose. Resplitting
per seed would confound the arm with the split, and with a between-arm effect of
0.001 to 0.01 that confound would be larger than the signal.

### What the probe cannot tell you

- **Four targets, unequally interesting.** `ridge_mean` averages
  `log_return_to_settle` (about +0.01 for every trained arm and both controls, and
  the only target with real trading content), `time_to_expiry` (+0.74 to +0.82
  across arms and controls, nearly free), `implied_width` (+0.65 to +0.80) and
  `window_log_return` (+0.14 to +0.24). The headline is dominated by the two easy
  targets. An arm can move `ridge_mean` by moving `window_log_return` alone, which
  is what the causal-context margin does (+0.141 to +0.177 under temporal
  masking). Read the per-target columns. The untrained encoder makes the point
  again from the other side: its per-target row is -0.000 / +0.678 / +0.579 /
  +0.126, so almost all of training's +0.0463 ridge gain is `time_to_expiry`
  (+0.085) and `implied_width` (+0.073).
- **One split.** `split_seed=0` only. Seed variation here is over training seeds,
  not over splits, so the reported spread does not include split variance.
- **The validation tail is contiguous.** `ridge_probe` cuts the last 20% of the
  train array rather than sampling, and the train array is in corpus order, so
  the lambda is selected on a particular set of events rather than a random one.
  This is pm-jepa's behaviour and was kept for comparability, not because it is
  the better choice.
- **Dimension is not matched to the controls.** Identity is 96-dimensional,
  handcrafted 167, our representation 128. Ridge with a tuned lambda is fairly
  insensitive to this, but it is not a controlled comparison.
- **It measures a mean-pooled representation.** `represent` averages over all 144
  tokens. Anything the encoder knows that lives in the *variation* across the
  grid rather than its mean is invisible to this probe.

---

## 2. Control baselines, and why they are the real bar

From `pm-jepa/results/baselines.json`, computed by `pm-jepa/baselines.py` **before
any model existed**, on the identical corpus and the identical event split:

| control | what it is | dim | ridge | MLP |
|---|---|---|---|---|
| identity | raw observation, mean-pooled over the last 4 minutes | 96 | **+0.449** | +0.505 |
| temporal | identity plus the change across the window | 192 | +0.460 | +0.511 |
| handcrafted | probit ladder, its slope and curvature, spread, volume, open interest, realised vol | 167 | **+0.460** | +0.486 |

The controls are the bar, and not each other, because they answer the only
question that matters for a representation-learning claim: **did the encoder
learn anything the raw cross-section did not already contain?** A JEPA arm at
+0.40 has not. It does not matter that it beats another JEPA arm.

`experiment.py` prints the controls *interleaved* with the arms in the same
table, not in a separate section, and `_print_probe_table` prints an explicit
"best arm CLEARS / LOSES TO the identity control by ..." line. Both are
deliberate. A control one page away from the arms is a control that gets skipped.

The controls are loaded from pm-jepa's file at run time, with the SPEC's numbers
as a hardcoded fallback, so the control line can never be missing from a results
table. A missing control is how an arm that loses to raw features gets read as a
success.

### The untrained encoder: the most important control in this suite

**What runs.** `scripts/benchmark.py::random_encoder_arm` builds a `CausalJEPA`
of the identical architecture (`d_model=128`, 4 layers), calls `.eval()`, and
feeds it straight into `represent_all` and `probes.run_probes`. No optimiser, no
steps, no EMA, no masking, same mean-pool over the same (P=6, K=24) token grid,
same event split, same probe protocol. Four seeds, where the seed is the
initialisation. It is run for both settings of the causal flag. Written to
`results/benchmark.json`; about 60 seconds on MPS, no training at all.

**Why it outranks identity and handcrafted.** Those two answer the question that
matters for deployment: did the encoder learn anything the raw cross-section did
not already contain. This one answers the prior question: **did training do
anything at all.** A model that cannot beat its own initialisation has learned
nothing, and no loss curve, effective rank, dimension statistic or copy
diagnostic can rescue that; all of those are computed on the trained model alone
and none of them has an untrained reference. It is also the cheapest control in
the suite: about 60 seconds, and no training.

**What it revealed.** From `results/benchmark.json`, four seeds per row:

| arm | ridge R^2 | MLP R^2 |
|---|---|---|
| random-encoder (bidir), untrained | +0.3456 +/- 0.0067 | +0.3802 +/- 0.0049 |
| random-encoder (causal), untrained | +0.3381 +/- 0.0079 | +0.3724 +/- 0.0076 |
| ctx=bidir tgt=bidir, 600 steps (pm-jepa's configuration) | +0.3919 +/- 0.0058 | +0.3797 +/- 0.0049 |
| ctx=causal tgt=bidir, 600 steps (best arm here) | +0.4018 +/- 0.0034 | +0.3872 +/- 0.0192 |
| **delta: pm-jepa configuration minus untrained** | **+0.0463 (t = 9.09)** | **-0.0005 (t = -0.12)** |
| **delta: best arm minus untrained** | **+0.0562 (t = 12.95)** | **+0.0070 (t = 0.61)** |

Training moves the ridge probe by a large and highly significant margin and moves
the MLP probe by nothing. This is not a property of one arm: across all eight
trained arms in `results/causal_ablation.json` and `results/interpolation_arm.json`
the MLP deltas against the untrained encoder run from -0.0010 to +0.0070, and the
largest is `t = 1.09`. The reading we can defend: the JEPA objective is not adding
information to the representation, it is rearranging information the random
projection already carried into a more linearly accessible form. The MLP probe,
which does not care about linear accessibility, sees no difference.

**What this does not license you to say.**

- Not "random features match SSL in general". A random transformer over
  structured input is a strong random-features baseline, and beating it is a bar
  that legitimate SSL methods can take a while to clear. On `slate-jepa`'s
  synthetic slates the same comparison separated cleanly: random-encoder +0.765
  ridge against `jepa-temporal` +0.901 (`slate-jepa/poc/results/poc.json`). The
  claim here is narrower and is about this corpus at this budget.
- Not "the trained encoder contains no more information". It says a 2x256 GELU
  MLP trained for 800 steps on 20,506 rows cannot find any, on these four
  targets. Section 1 lists what that probe cannot see.
- Not "600 steps is enough to conclude". It is not, and that is limitation 1 in
  section 6.

**What would falsify it.** A longer training run where MLP R^2 separates from the
untrained baseline by more than 2 seed-standard-deviations. That run is the
single highest-value next experiment in this repo.

**Where it should have run before.** No published `pm-jepa` result includes this
control. Its three arms sit at MLP +0.3160 to +0.3263 against an untrained
+0.3802, though that particular comparison crosses the PatchTST / factorised
encoder boundary and is architecture-confounded. The within-architecture
comparison, the one in the table above, is not.

### The controls reproduce under this repo's pipeline

Not assumed, measured. Running pm-jepa's `identity_features` through
`causaljepa.data.load_corpus`, `causaljepa.data.split_by_event` and
`causaljepa.probes` gives **+0.4491** at `lambda=1000`, dim 96, against pm-jepa's
published +0.44908884. Per-target: +0.0152 / +0.7394 / +0.8033 / +0.2384, all
four exact to four decimals. The recipe is in `docs/REPRODUCE.md` section 8 and
should be re-run whenever the corpus cache changes.

### Headroom

Identity's MLP (+0.505) beats identity's ridge (+0.449), so there *is* nonlinear
structure in the raw features that a linear probe cannot reach. That gap is the
headroom a representation could in principle capture. Per target, though,
`time_to_expiry` sits at ridge +0.739 / MLP +0.843 and `implied_width` at +0.803 /
+0.840; there is very little left on either. The headroom that matters is on
`log_return_to_settle`, where identity is +0.015 ridge / +0.012 MLP, i.e. the
information is not there in the raw features and no probe of either kind finds
it. Every arm in this repo also sits at about +0.010 there.

### Where our arms land

| arm | temporal | interpolation |
|---|---|---|
| identity (control) | +0.449 | +0.449 |
| handcrafted (control) | +0.460 | +0.460 |
| pm-jepa's best of three arms at 3 seeds, `jepa-temporal` (PatchTST) | +0.377 +/- 0.009 | no such arm |
| ctx=bidir, tgt=bidir | +0.392 +/- 0.006 | +0.390 +/- 0.007 |
| ctx=bidir, tgt=causal | +0.392 +/- 0.007 | +0.394 +/- 0.009 |
| ctx=causal, tgt=bidir | **+0.402 +/- 0.003** | **+0.397 +/- 0.004** |
| ctx=causal, tgt=causal | +0.401 +/- 0.004 | +0.395 +/- 0.004 |
| random-encoder (bidir), untrained; strategy does not apply | +0.346 +/- 0.007 | +0.346 +/- 0.007 |

All eight arms lose to both controls. The best is 0.047 below identity. The
factorised encoder does sit above pm-jepa's PatchTST arms, but that comparison
crosses an architecture boundary, so it is not a causality result and not a
representation-quality result either. The `pm-jepa` row is from
`results/benchmark.json`, which reads their `results/seeds.json` (3 seeds per arm);
`jepa-slate` is +0.343 +/- 0.006 and `jepa-contiguous` +0.339 +/- 0.006 there.

### What the controls cannot tell you

- They are a bar, not a ceiling. Losing to identity does not prove the
  representation is worthless for a task the four probe targets do not cover.
- They were computed once, by pm-jepa, at `window=24, stride=4`. Change either
  and they are no longer the right bar; `data.load_corpus` is memoised on
  `(root, window, stride)` and the results files inline the config so this is
  auditable, but nothing enforces it.
- `identity` pools the last 4 minutes; our representation pools all 24. The
  controls are not matched on how much of the window they see.

---

## 3. The copy oracle, and the directional split

### The plain oracle

`copy_diagnostics(pred, target, patch_mask)`. For each masked patch it builds the
copy solution explicitly: the target embedding of the **nearest visible patch of
the same strike**, preferring the most recent one at or before the masked patch.
Then:

| key | meaning |
|---|---|
| `copy_loss_ratio` | model loss / oracle loss on the same positions. Near 1.0 means the model has learned nothing that copying would not give you. |
| `copy_alignment` | cosine between the model's prediction and the copy reference. Near 1.0 means the model *is* the copy, whatever the ratio says. |
| `model_loss_on_copyable` | the numerator alone, so a moving ratio can be attributed. |
| `copyable_frac` | fraction of masked positions with any reference at all. 1.0 in every run here. |

The two keys have to be read together, and this is the single most common
misreading. In `results/causal_ablation.json` the final `copy_alignment` is
+0.954 to +0.964 while `copy_loss_ratio` is 0.735 to 0.813. The model is
cosine-aligned with the copy solution to three decimals *and* beats it on loss by
19% to 27%. Alignment says the direction is the copy; the ratio says the magnitude
is not. Quoting either alone gives the opposite impression from the other.

### The directional split, new in this repo

`directional_copy_diagnostics(pred, target, mask)` splits the oracle by direction,
using strict lookups on both sides:

| key | reference | detects |
|---|---|---|
| `extrap_*` | nearest visible patch **strictly before** | the copy a causal encoder can also do. Forward extrapolation off the martingale. |
| `interp_*` | nearest visible patch **strictly after** | the copy only a bidirectional encoder can do. Reading backwards from the future side of the hole. |
| `interp_advantage` | `extrap_oracle_loss - interp_oracle_loss` on the **paired** subset | which direction is the easier solution. Positive means interpolation is easier, i.e. the objective is softer than it looks. |

Why the split exists: `copy_diagnostics` reports one number, and under
`interpolation` masking that number cannot distinguish a model that has learned
the martingale from one that is quietly reading the future through a bidirectional
target encoder. H2 predicts `interp_advantage` falls towards zero once the target
encoder is causal.

`interp_advantage` is computed on the paired subset where **both** references
exist, not as a difference of the two headline losses. Under `temporal` masking
every masked patch has a past reference and none has a future one, so differencing
unpaired means would compare the last patch of the window against nothing.
`paired_frac` reports the coverage: 0.0 under `temporal` (so `interp_advantage` is
`nan`, which is the honest answer), 1.0 under `interpolation`.

### Two structural redundancies, measured

Both were confirmed against the 224 history records in the results files, not
reasoned about:

1. **`extrap_loss_ratio == copy_loss_ratio`** under both `temporal` and
   `interpolation`. Max absolute difference across every record: 2.4e-07, i.e.
   float32 noise. `nearest_visible_reference` prefers the past, `extrap_frac` and
   `copyable_frac` are both 1.0, so the two oracles select the same reference at
   every scored position. The two columns in the diagnostic table are redundant
   for these strategies. They would separate under `random` or `slate` masking,
   where a masked position can lack a past reference.
2. **`interp_advantage` never touches `pred`.** It is oracle-minus-oracle, a pure
   target-side statistic. Two cells that share a `causal_target` value therefore
   report identical values early in training. This is correct and is easy to
   misread as "the flag is not wired".

### What pm-jepa established that the copy diagnostic does NOT predict

**Downstream utility.** This is the most important negative result carried into
this repo, and it is why the copy oracle is never used to rank arms.

From `pm-jepa/results/train.json`, three arms, one seed each, 1500 steps:

| pm-jepa arm | copy_loss_ratio | copy_alignment | ridge |
|---|---|---|---|
| jepa-temporal | **0.563** | +0.966 | +0.374 |
| jepa-contiguous | nan (no copy solution) | nan | +0.342 |
| jepa-slate | nan (no copy solution) | nan | **+0.383** |

The arm with the strongest anti-copy reading (temporal, beating the oracle by
44%) had the worse probe R^2 of the two arms that finished. The arms constructed
so that no copy solution exists at all, which is the strongest possible
intervention against martingale collapse, gained +0.009 on the probe and remained
0.066 below the identity control.

That single-run ordering does not even survive seeds. The multi-seed values this
repo reads from their `results/seeds.json`, mirrored into
`results/benchmark.json`, put both no-copy-solution arms **below** the copy-prone
one: `jepa-temporal` +0.3771 +/- 0.0095, `jepa-slate` +0.3426 +/- 0.0059,
`jepa-contiguous` +0.3394 +/- 0.0064. Either way the conclusion is the same and
the stronger version is the seeded one: removing the copy shortcut did not buy
representation quality, and every one of the three stayed at least 0.07 below the
identity control.

This repo reproduces the same dissociation under a much better-powered design.
The causal-target flag moves the copy diagnostics with 8/8 sign consistency in
both strategies, and moves the probe by nothing:

| quantity, paired causal minus bidir | temporal | interpolation |
|---|---|---|
| `copy_alignment` | -0.0013 (t=-2.42, 2+/6-) | -0.0066 (t=-6.42, **0+/8-**) |
| `copy_loss_ratio` | -0.0332 (t=-4.61, **0+/8-**) | -0.0486 (t=-7.28, **0+/8-**) |
| probe `ridge_mean` | -0.0001 (t=-0.24, 3+/5-) | +0.0011 (t=+0.62, 4+/4-) |

**Rule: never rank arms by `copy_loss_ratio` or `copy_alignment`.** They are
diagnostics of the *objective*, and they answer "is this task trivially solvable
by copying". They are not a proxy for representation quality, and two independent
experiments now say so.

### The H2 sign trap

H2 predicted `copy_loss_ratio` would **rise** under a causal target. It **fell**,
8/8, in both strategies. The decomposition, which is in the history precisely so
this can be settled without a rerun, vindicates the mechanism and indicts the
metric:

| | temporal | interpolation |
|---|---|---|
| numerator, model loss | -0.00003 (t=-0.16, ns) | +0.0033 (t=+7.09, 8+/0-) |
| denominator, copy oracle loss | **+0.0022 (t=+4.17, 8+/0-)** | **+0.0083 (t=+8.47, 8+/0-)** |

The copy oracle's own loss rose 8/8 in both arms. Copying genuinely did get
harder, which is what H2 claimed. The ratio fell because the model's loss rose by
less. **A ratio cannot express "both terms moved and the denominator moved
more".** Always read `model_loss_on_copyable` and `extrap_oracle_loss` next to
`copy_loss_ratio`.

### What the copy oracle cannot tell you

- It is a *sufficient* condition test, not a necessary one. A low ratio proves the
  model is not literally copying. It does not prove the model learned anything
  else; see the dissociation above.
- The reference is the **target encoder's** embedding, so when the target encoder
  changes, the oracle changes. Every between-arm comparison of a copy metric is a
  comparison against a moved bar. That is exactly why H2 is stated as a paired
  contrast and why both terms of the ratio are logged.
- `copy_alignment` saturates. Above about +0.95 it has very little resolution
  left, and every trained arm here is above +0.95.
- It is computed at the patch level, on `patch_mask`, not at the minute level.

---

## 4. `target_autocorrelation` and the leakage measurement (H1)

### What it computes

Mean cosine similarity between target embeddings at a fixed patch lag, same
strike, plus a set of controls:

| key | meaning |
|---|---|
| `lag_1` .. `lag_5` | mean cosine between patches L apart, all pairs, same strike |
| `cross_boundary` | mean cosine over pairs **straddling** the visible/masked split |
| `within_side` | the lag-weighted same-side control for `cross_boundary` |
| `boundary_excess` | `cross_boundary - within_side`, lag-matched |
| `boundary_excess_matched` | the same contrast within an identical `(p, q)` pair |
| `baseline_cos` | the **floor**: mean cosine between the same `(patch, strike)` slot in two different windows of the batch |
| `cross_boundary_pairs` | how many pairs went into the straddling average |

The H1 claim is that a bidirectional target encoder computes `target[p=3]` and
`target[p=5]` both with attention over all six patches, which makes them more
similar to each other than two causally-computed targets would be. A causal
target encoder should lower that similarity.

### How to read it, or H1 is not falsifiable

**Every key here is a between-arm statistic.** The absolute level is
uninterpretable. Straddling pairs are systematically further apart in time than
same-side pairs, so `cross_boundary` moves whenever the hole moves; under
`temporal` masking the straddling set has mean lag about 2.8 while the same-side
set is dominated by lag 1. Never compare across masking strategies, and never read
one arm's level.

Three things make the difference readable:

1. **The mask is held fixed.** `temporal` and `interpolation` are both
   deterministic, so the pair set is identical in both arms and every geometric
   confound cancels in the difference.
2. **The design is paired.** `train_arm` seeds torch, numpy and the mask generator
   from the arm's seed alone, so seed *s* in cell `bb` and seed *s* in cell `bc`
   start from identical weights and see identical batches and identical masks.
   Only the target encoder's attention mask differs. The per-seed difference
   cancels initialisation and batch-order noise, which is most of the unpaired
   spread. Compare the two tables: the unpaired Welch t on temporal
   `cross_boundary` is -0.97, the paired t on the same data is **-4.75**.
   `scripts/report.py` prints both, and the paired one is labelled "the powered
   test".
3. **The floor is subtracted.** Transformer embeddings are anisotropic and
   routinely sit at cosine 0.9 to everything. A `lag_1` of 0.95 means nothing
   until you know `baseline_cos`. `report.py:derived()` computes
   `*_over_floor` for exactly this reason.

### The floor ate the effect

This is the headline finding on H1 and the reason this document opens the way it
does.

| paired, causal minus bidir target | temporal | interpolation |
|---|---|---|
| `cross_boundary` raw | -0.0101 (t=-4.75, 0+/8-) | -0.0115 (t=-8.91, 0+/8-) |
| `baseline_cos` (the floor) | **-0.0090 (t=-3.73)** | **-0.0155 (t=-5.55)** |
| `cross_boundary` **over floor** | -0.0012 (t=-1.53, 2+/6-) | **+0.0040 (t=+1.55, 5+/3-)** |
| `lag_3` over floor | -0.0025 (t=-3.12, 0+/8-) | -0.0016 (t=-0.59, 3+/5-) |

The raw autocorrelation falls exactly as H1 predicted, 8/8 in both strategies.
The anisotropy floor falls almost as much, and in the interpolation arm it falls
*more*. Floor-corrected, the effect is null in temporal and the wrong sign in
interpolation. Only temporal `lag_3` over floor survives, at about a fifth of the
raw number.

Reading: the causal target encoder changes the embedding geometry **globally**,
not specifically across the mask boundary. H1's mechanism may be real, but the
metric as specified conflates it with an isotropy shift. The verdict recorded is
**inconclusive**, and specifically "not detectable at this protocol", not "shown
to be zero".

### `boundary_excess` and what its null means

`boundary_excess` is **not** the H1 statistic. It is here to identify which
leakage mechanism is operating. pm-jepa's target encoder, and ours, is
**mask-blind**: `forward()` calls the target tower with `strike_mask=None`, so it
does not know where the hole is. Bidirectional leakage then inflates every pair in
the window by roughly the same amount, and the straddling-minus-same-side contrast
cancels it. Expect it near zero in both arms; that is the mask-blind signature,
not a failed H1.

Measured: **+0.0078 (t=+4.55, 8+/0-)** under temporal, **-0.0007 (t=-4.15, 1+/7-)**
under interpolation. The temporal number is the one place H1's story gets support,
because a boundary-specific move is what you would see if the leakage were
boundary-specific rather than window-wide. It is also small enough, against a
statistic whose own null is not zero, to be worth no more than a sentence.

Its null is genuinely not zero: on a synthetic zero-started random walk with no
leakage planted at all, `boundary_excess` reads +0.002 under `interpolation` and
+0.134 under `temporal`, purely because a walk is not stationary in the patch
index and a deterministic hole makes late-window pairs disproportionately
straddling. Read only the between-arm difference.

`boundary_excess_matched` removes that position confound by contrasting straddling
against same-side instances of the *same* `(p, q)` pair, and is zero-referenced.
It requires the mask to vary across samples or strikes. Under `temporal` and
`interpolation` the mask is identical for every `(b, k)`, so a given pair is either
always straddling or never, and this key is **`nan` in all 224 history records**.
That is the honest answer: for the deterministic arms there is no in-sample
control, and the causal arm is the only available null. That is exactly why the
experiment is a 2x2 rather than one run.

### What the leakage measurement cannot tell you

- **It is a cosine, not an information measure.** It says pairs moved apart. It
  does not quantify how much future information the target embedding contains, and
  the two are not monotonically related.
- **It only exists in a trained encoder.** `scripts/verify_h1_measurable.py`
  measured this on pm-jepa's own PatchTST target: `bidir - causal` at lag 3 is
  +0.0002 at random init, +0.0141 at 16x attention mixing, +0.0295 at 64x. At
  random init, zeroing patches 4 and 5 moves patch 3's target embedding by 0.005
  of its norm, because a pre-norm residual stream starts near the identity. **An
  H1 null measured on an undertrained target encoder is a null about the encoder,
  not about causality.** `train_arm` warns when `steps < 100` for this reason
  (see section 3 of `docs/REPRODUCE.md`).
- **`baseline_cos` doubles as a collapse detector, and it is load-bearing.** A
  rising `lag_L` alongside a rising `baseline_cos` is collapse, not leakage. In
  the reported runs the floor sits at +0.63 to +0.66 against `lag_1` of +0.93 to
  +0.95, so there is a real gap and the arms are not collapsed. If those ever
  converge, none of the autocorrelation numbers are measuring leakage.
- **600 steps is short.** The floor-corrected effect is around 0.002, which is
  near the resolution this protocol reaches.

---

## 5. Cache exactness and the streaming benchmark (H4)

Two separate instruments with two separate jobs. Exactness is a correctness test
and gates the benchmark; the benchmark is a performance measurement and proves
nothing about correctness.

### Causality itself: the perturbation test

`tests/test_causal_mask.py::test_causal_mask_no_future_leak` is the real proof,
and it is deliberately stronger than reading the attention mask. Perturb the input
at patch 3 and measure how far the output tokens move:

| | patches < 3 | patches >= 3 |
|---|---|---|
| `causal=True` | **0.000e+00** (bit-identical) | 4.637e+00 |
| `causal=False` | 1.210e+00 | 4.665e+00 |

Reading a mask tensor tests that you wrote the mask you meant to write. This tests
that the mask is actually applied, on the path the model actually runs.
`test_bidirectional_does_leak` is the has-teeth half: if the bidirectional control
ever stopped leaking, the causal assertion would be vacuous.

### Cache exactness

The SPEC calls this the single most important test in the repo. For a causal
encoder, feeding patches through `encode_incremental` one at a time must equal
`forward()` on the full window within `atol=1e-4` in float32.

Measured, independently of the cache's own verification script:

| configuration | max abs difference |
|---|---|
| stride-1 incremental vs full re-encode | 1.19e-06 |
| 2-patch chunks | 0.00e+00 |
| true streaming (`start_patch`, old minutes dropped) | 1.19e-06 |
| on real corpus data, MPS-trained model | 9.54e-07 |
| **same cached path against a BIDIRECTIONAL forward** | **2.67e+00** |

The last row is the point. It is the has-teeth control, at **26,732 times** the
1e-4 bar. Without it, a test that passes proves only that the tolerance is loose.
`tests/test_cache_exactness.py` contains both halves plus a refusal test
(`encode_incremental` raises `ValueError` on a bidirectional encoder, because
with bidirectional time attention an already-emitted token is not final and
returning a cache anyway would be a lie) and a reset test.

**If exactness ever fails, that is a bug in `causaljepa/cache.py`. Do not relax
the tolerance.** The failure mode this guards against is silent: a wrong key or an
off-by-one positional offset still returns plausible-looking embeddings and
nothing raises.

### The streaming benchmark

Append-only patch stream: at tick p the encoder has seen patches 0..p and must
emit the tokens for patch p. `full` re-runs `forward()` on the whole prefix;
`cache` runs `encode_incremental`. B=1, K=24, d=128, 4 layers, median of 3 repeats
(1 at P=512), one untimed warm-up pass per path so MPS kernel specialisation does
not land in tick 1. Exactness is re-verified at every shape, because a cache that
is fast and wrong is worse than no cache.

| device | P | exact | max abs diff | full ms/tick | cache ms/tick | speedup | tok/s full -> cache |
|---|---|---|---|---|---|---|---|
| mps | **6 (real)** | yes | 0.0e+00 | 8.61 | 7.00 | **1.22x** | 2,852 -> 3,474 |
| mps | 32 | yes | 1.4e-06 | 9.70 | 8.38 | 1.16x | 2,503 -> 2,903 |
| mps | 128 | yes | 1.4e-06 | 10.81 | 8.40 | 1.26x | 2,276 -> 2,859 |
| mps | 512 | yes | 1.9e-06 | 166.93 | 39.89 | 4.19x | 144 -> 602 |
| cpu | **6 (real)** | yes | 1.4e-06 | 2.77 | 1.83 | **2.15x** | 6,012 -> 12,942 |
| cpu | 32 | yes | 1.4e-06 | 9.53 | 1.79 | 4.09x | 2,301 -> 9,414 |
| cpu | 128 | yes | 1.5e-06 | 19.09 | 2.45 | 7.65x | 1,237 -> 9,455 |
| cpu | 512 | yes | 1.7e-06 | 156.10 | 5.05 | **30.92x** | 154 -> 4,754 |

### The honest caveats, stated rather than discovered

- **At the real shape the win is small: 1.22x on MPS, 2.15x on CPU.** P=6 means
  five patches of history at most, which is not enough work to amortise the
  constant per-call cost. On MPS, kernel-launch latency dominates until P=512.
  This is the correct result at that shape and it is reported as the headline, not
  buried. The larger shapes are there to show the slope, not to flatter the small
  one. The slope is the actual result: CPU 2.15x, 4.09x, 7.65x, **30.92x** at
  P = 6, 32, 128, 512.
- **This is an append-only stream.** `patch_stride=4`, so a 1-minute slide of the
  corpus window rewrites the contents of every patch and invalidates the cache
  completely. The regime the cache serves is a *growing history*, not a *sliding
  window*. Anyone quoting these numbers for a sliding window would be quoting them
  for the wrong thing. This sentence is inlined into
  `results/cache_bench.json` under the `regime` key so it travels with the
  numbers.
- Batch is 1, which is the streaming case, and which also means the numbers are
  latency-bound rather than throughput-bound.
- No `torch.compile`, no quantisation, no other process control. These are wall
  times on a laptop.
- `tok/s` is defined as new tokens *emitted* over the sweep, `P * K * batch`, not
  tokens attended over. Both paths emit the same tokens; only the work differs.

---

## 6. Known limitations of this eval suite

Ordered roughly by how likely each one is to change a conclusion.

1. **The 600-step budget is the main threat to the conclusions, including the
   headline one.** The design spends its budget on seeds rather than steps,
   because the target-flag effects are 0.005 to 0.01 against a between-seed
   spread of about 0.007, so seeds buy resolution where steps do not. That trade
   is defensible for the 2x2 and it is exactly where the headline is weakest: the
   finding that 600 steps of training adds nothing an MLP probe can find (section
   2) rests on one budget, and a random transformer over structured input is a
   strong baseline that a legitimate method could need far more than 600 steps to
   clear. Nothing in this suite rules that out, and the arms are indistinguishable
   at every logged step rather than only at the last, which is consistent with
   both explanations. The floor-corrected H1 effect of about 0.002 is likewise at
   the edge of what this protocol resolves. H2 and H4 are robust at 8/8 sign
   consistency; the H1 null is "not detectable here", not "zero"; the
   trained-versus-untrained MLP null is "not detectable at 600 steps", not "the
   representation contains nothing".

2. **The H1 metric does not survive its own control.** `cross_boundary` is the
   metric the SPEC names, and the floor moves with it. This is a defect in the
   measurement, not in the run. A better instrument would compare target
   embeddings under matched isotropy, or measure mutual information with the
   masked patches directly rather than through a cosine.

3. **`boundary_excess_matched` is `nan` in every record.** The deterministic
   masks give no in-sample control for boundary-specific leakage. A strategy whose
   mask varies across `(b, k)`, `random` or `slate`, would populate it, and no arm
   here uses one.

4. **The predictor is bidirectional in all four cells.** The SPEC does not ask
   otherwise, and making it causal would confound H1 and H2 with a predictor
   change. So the 2x2 is over the two *encoders* only, and no arm in this repo is
   causal end to end in the sense a deployed streaming model would be.

5. **The target encoder is mask-blind.** `forward()` calls it with
   `strike_mask=None`. This is pm-jepa's behaviour, kept for parity, and it is
   what makes `boundary_excess` a mask-blindness detector rather than a leakage
   detector. It also produces the sharpest negative in the whole experiment:
   `interp_advantage` went **up** under a causal target, +0.0040, t=+6.86, 8+/0-,
   where H2 predicted down. The mechanism is clean. A causal target at a
   *post-hole* patch has already integrated the hole's own minutes; a causal target
   at a *pre-hole* patch cannot. Making the target causal therefore *strengthens*
   the forward reference relative to the backward one. **Causal target does not
   close the interpolation escape hatch; only causal context can.**

6. **One split seed, four training seeds, one corpus.** `split_seed=0` throughout.
   The reported spread is over training seeds only.

7. **`ridge_mean` averages four heterogeneous targets**, two of which are nearly
   saturated and one of which (`log_return_to_settle`) is near zero for every
   method including both controls. See section 1.

8. **`extrap_loss_ratio` and `copy_loss_ratio` are the same number** under both
   strategies actually run (max difference 2.4e-07). Two of the nine columns in
   the diagnostic table carry one column of information.

9. **The cache benchmark is append-only and does not model the deployment case.**
   See section 5.

10. **Interpreter is CPython 3.9.6**, not the 3.11+ the SPEC names, because that
    is the only interpreter on this machine with torch 2.6. Every module avoids
    PEP 604 annotations; this was verified, not assumed. The `SPEC NOTE` comments
    in `train.py` and `cache.py` record it.

11. **Diagnostics run on CPU copies** of the forward pass, because
    `torch.linalg.svdvals` and `cummax` have no MPS kernel in torch 2.6. This is
    done uniformly rather than conditionally so the numbers are identical across
    devices, but it means the diagnostics are measured on float32 values that made
    a device round trip. A future torch that grows those kernels would change
    reduction order at about the third decimal, which is the resolution the 2x2 is
    read at. This is why `requirements.txt` pins exact versions.

12. **`reg_grad_norm` is 0.0 in all 224 records**, by construction: there is no
    regulariser. It is logged as a tripwire, not a measurement. pm-jepa's SIGReg
    was computed on a detached target and contributed exactly zero gradient in
    every run they published; this field exists so that failure can never silently
    recur. It is not evidence about anything else.

13. **Dropping `out_norm` from the encoder is uncovered by the tests.** It is not
    a SPEC contract item, so no test was invented for it, but it is a real coverage
    gap: the final LayerNorm is what the cosine-based diagnostics and
    `effective_rank` implicitly assume about output scale.

14. **`sibling_interpolation_gap` was not ported** from pm-jepa. It is not in the
    SPEC section 5 list. The strike-axis analogue of the interpolation escape
    hatch is therefore unmeasured here.

---

## 7. Instrument summary

| instrument | what it catches | what it cannot say | where |
|---|---|---|---|
| `pred_loss` | a run that did not train | anything between arms | `train.py` |
| `effective_rank`, `dimension_stats` | representational collapse | martingale collapse | `diagnostics.py` |
| `copy_diagnostics` | martingale collapse, the copy shortcut | downstream utility (established twice) | `diagnostics.py` |
| `directional_copy_diagnostics` | which direction the copy comes from | anything about `pred` for `interp_advantage` | `diagnostics.py` |
| `target_autocorrelation` | between-arm target-side leakage | absolute leakage; anything without the floor | `diagnostics.py` |
| `ridge_probe`, `mlp_probe` | representation quality vs raw features | quality on tasks outside the four targets | `probes.py` |
| untrained random-encoder control | training that changed nothing a probe can read | whether a longer run would separate it | `scripts/benchmark.py` |
| cache exactness | a wrong or off-by-one cache | that the cache is fast | `tests/test_cache_exactness.py` |
| perturbation test | a causal mask that is not applied | how much leakage a bidirectional arm has | `tests/test_causal_mask.py` |
| `cache_bench` | streaming speedup and its slope | sliding-window performance | `scripts/cache_bench.py` |

To re-derive the 2x2 and H4 tables in this document from the saved JSON without
training a single step: `python3 scripts/report.py`. The benchmark table, the
untrained control included, is `python3 scripts/benchmark.py --seeds 4`, which
refits the probes but trains nothing, about 60 seconds on MPS. Nothing here was
retyped.
