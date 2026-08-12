# Research plan

An autonomous continuation of the prediction-market JEPA programme across `slate-jepa`,
`pm-jepa`, and `causal-jepa`.

Every experiment below states a **pre-registered gate**: the outcome that decides what runs next.
Gates are written before the run, not after, because the failure mode this programme has already
hit twice is interpreting a result after seeing it. There is also a **kill criterion** at the end.
A plan without one is a plan to keep going regardless of evidence.

---

## Where we actually are

| | |
|---|---|
| Copying is the martingale, not an encoder artefact | Settled. Causal-target ablation, 4 seeds, t = 0.09 |
| Training beats its own random init on ridge | Yes, +0.0463, t = 9.09 |
| Training beats its own random init on MLP | **No. -0.0005, t = -0.12** |
| Any arm beats the raw cross-section | No. Identity +0.4491 vs best arm +0.4018 |
| JEPA beats controls on synthetic data | Yes, +0.9014 vs identity +0.8142 |
| Either repo's regulariser was ever active | **No. Both computed on detached targets** |

Two facts drive everything below. First, **the whole programme rests on a 600-step budget**, and
the load-bearing null (MLP vs untrained) has never been run to convergence. Second, **the
objective works on the simulator and fails on the market**, and nobody has measured why.

---

## E4 first: is the readout hiding the representation?

Run this before anything expensive, because it is a confound in *every* result the programme has
produced, including the ones we just published.

`represent()` returns `encode(obs).mean(dim=(1, 2))`, a mean over both the patch and strike axes.
That is a 144-token grid crushed to one 128-vector. The identity control, by contrast, keeps all
24 strikes. If the pooling is what loses the information, then "JEPA does not beat raw features"
is a statement about our readout, not about the model.

**Run.** Reuse the already-trained encoders from `results/causal_ablation.json`. No retraining.
Probe six readouts: mean over both axes (current), last patch only, concatenate all patches,
concatenate all strikes, attention pooling with a learned query, and max over patches. Same ridge
and MLP protocol, same event-level split, 4 seeds.

**Gate.**
- Any readout puts a trained arm above identity (+0.4491 ridge): **the prior conclusion was a
  readout artefact.** Stop, correct `pm-jepa/FINDINGS.md` and the blog post, rerun the headline
  comparisons under the better readout before anything else proceeds.
- Best readout still below identity: pooling is exonerated, continue to E1.

**Cost.** Under an hour. Highest value-per-minute in the plan.

---

## E1: run it to convergence

The decisive experiment. Every conclusion in three repos is conditioned on 600 steps.

**Run.** One long run per seed, 3 seeds, at the reference config (`ctx=bidir, tgt=bidir`,
d_model 128, 4 layers). Checkpoint and probe at 600, 1200, 2400, 4800, 9600, 19200 steps. Probe
the untrained encoder at the same seeds as the floor. Record ridge, MLP, `copy_alignment`,
`eff_rank`, and `reg_grad_norm` at every checkpoint. Training one long run and probing at
checkpoints costs a sixth of training six separate runs; do it that way.

**Gate.** *(Wording corrected 2026-08-10 after the first run. The original said "at any
checkpoint", which a transient early peak satisfies without answering the question the gate was
for. The run itself is unaffected; see `results/convergence.json:gate_review`. Kept visible
rather than silently patched, because a plan that edits its own criteria after seeing data is
worth nothing.)*

- MLP R² **at the final checkpoint** exceeds the untrained floor by more than 2 pooled standard
  deviations, **and** the trajectory is non-decreasing over the last two checkpoints: **the
  objective does learn, and the 600-step null was a budget artefact.** Re-run E4 and the causal
  2x2 at that budget.
- No separation at the final checkpoint: **strong negative.** The JEPA objective adds no
  probe-visible information on this corpus at any budget we can afford. Proceed to E3, which
  becomes the explanation rather than a side quest.
- Separation appears at an early checkpoint and then reverses: **also a strong negative, and a
  more interesting one.** Report the peak and the reversal; the objective is destroying
  information it briefly had.

**Result (2026-08-10).** The third branch. 3 seeds, 19,200 steps. Ridge is flat from step 600
(+0.4006) to step 19,200 (+0.3965). MLP peaks at step 600 (+0.3966, +3.30 SD over floor) and
falls to +0.3055, ending **23 pooled SD below the untrained floor**. Effective rank nearly
triples, 36.4 to 93.8, while utility falls. `copy_alignment` falls 0.972 to 0.940 and the probes
get worse anyway. Proceed to E3.

**Cost.** 2.0 hours on MPS. Measured, not guessed: the four arms in `results/causal_ablation.json`
averaged 0.123 s/step at d_model 128 and 4 layers, so 19,200 steps times 3 seeds is 2.0 h. Run it
overnight.

---

## E2: test the regulariser that was never tested

`pm-jepa` computed SIGReg on a detached target and `slate-jepa` did the same with VICReg. Both
contributed exactly zero gradient. So the published claim that these mechanisms fail to prevent
martingale collapse **was never tested**, and it is still open.

**Run.** In `causal-jepa`, add a regulariser computed on the **online context** representation,
which carries gradient. Assert `reg_grad_norm > 0` before the run counts; the tripwire already
exists in `train.py` and this is what it was built for. Sweep `lam` over 0, 0.01, 0.05, 0.25, 1.0
for SIGReg and the VICReg variance/covariance pair, 3 seeds each, at whatever budget E1 says is
adequate.

**Gate.**
- An active regulariser drives `copy_alignment` below 0.90 without destroying probe R²: **the
  original claim was wrong and the mechanism does help.** This is a correction to a published
  finding and goes straight into a note.
- `copy_alignment` stays above 0.95 across the whole sweep: **the original claim is vindicated,
  now for real.** Also publishable, and it retires the question permanently.
- Probe R² collapses as `lam` rises: the regulariser trades representation for distribution
  shape. Report the frontier.

**Result (2026-08-10). The original claim is refuted, and it changes nothing.**

Run at 600 steps rather than 4,800, on E1's evidence that the probes peak at the first checkpoint
and degrade after; a longer run would have measured a worse model. 3 seeds, 9 configurations,
every regularised arm asserted `reg_grad_norm > 0` before it counted.

Both mechanisms work once they are in the gradient. `copy_alignment` falls from 0.9655
unregularised to 0.8060 under SIGReg and 0.7758 under VICReg, at t = -38 and t = -50. So "SIGReg
does not prevent martingale collapse" was never true; it was never tested.

And it buys nothing. Ridge moves between -0.0144 and +0.0126 across the sweep, the MLP probe is
flat or worse. Across all 27 runs `copy_alignment` spans 0.198 while `ridge_mean` spans 0.0346,
and their correlation is **+0.390**: within this sweep, copying more is weakly associated with
scoring better. See finding 13.

*Gate wording was wrong, again.* The criterion equated "drops below 0.90" with "the mechanism does
help". It fired, and help is the wrong word. Recorded rather than patched, as with E1.

**Cost.** 4.9 hours at 4,800 steps: 10 configurations times 3 seeds at the measured 0.123 s/step.
Scales linearly if E1 says a longer budget is needed, so check E1's answer before launching.

---

## E3: why does the simulator work and the market not?

Scientifically the most valuable experiment here, and the one that would explain the entire
programme.

On `slate-jepa`'s simulator the JEPA beat every control (+0.9014 against identity +0.8142,
handcrafted +0.8611, untrained +0.7653). On real Kalshi ladders the ordering inverts. The
simulator has a latent state that is genuinely there and genuinely recoverable. The market may
not, or its signal-to-noise may sit below what the objective can extract.

**Run.** The simulator exposes true latent state and a controllable noise level. Sweep the
latent signal-to-noise across roughly a decade, and at each point train the JEPA and measure
whether it beats the identity control. That yields a **threshold**: the SNR below which the
objective stops adding value. Then locate the real corpus on the same axis, by measuring the same
latent-recoverability statistic on Kalshi data that the simulator sweep varies.

**Result (2026-08-10). Neither pre-registered explanation survived; the answer came from a
third measurement the plan did not anticipate.**

- *Noise (part A): refuted, in the opposite direction.* The JEPA's advantage over identity never
  crosses zero and more than doubles across a 32x noise range, +0.0594 at baseline to +0.1502 at
  16x. Raw features degrade under noise and the learned representation does not, which is a
  genuine positive result about the objective rather than a null.
- *Redundancy (part B): refuted by its own control.* Kalshi looked far more redundant than the
  simulator, residual 1.5% against 4.6%, but that is a predictor-count artefact of 24 markets
  versus 8. At matched K, Kalshi has more residual structure, not less.
- *Target decomposition (part C): the answer.* The headline metric is a mean over four targets,
  one of which `dataset.py` itself calls a derivable sanity check and which is where nearly the
  whole raw-versus-JEPA gap lives. On the only genuinely future target, all 16 arms sit in
  [-0.0043, +0.0510]. There is no hidden state in the target set for the objective to find.

**Gate.**
- Real data sits below the threshold: **the programme's negative results are explained, and the
  explanation is a property of prediction markets rather than of our models.** That is the
  strongest paper available from this work, and it retires the whole line cleanly.
- Real data sits above the threshold and the JEPA still fails: the objective is not the
  bottleneck, the architecture or optimisation is. Continue to E6.

**Cost.** Sweep of maybe 8 SNR points times 3 seeds on the cheap synthetic model, plus one
statistic on the real corpus. A day.

---

## E5: make the cache claim honest (engineering, unblocked)

The cache is exact but only for an append-only patch stream. At `patch_stride = 4` a one-minute
slide rewrites every patch, so the streaming case that motivated it cannot use it.

**Run.** Either set `patch_stride = 1`, or anchor the patch grid to absolute time rather than to
the window so a sliding window appends rather than rewrites. Re-run `scripts/cache_bench.py` in
the true stride-1 streaming regime and report the honest speedup.

**Gate.** Independent of the science. Do it whenever the machine is otherwise idle. If stride-1
patching materially changes probe results, that is itself a finding and goes to E1's protocol.

---

## E6: scale, but only if E1 says there is something to scale

Everything so far is 1.8M parameters, d_model 128, 4 layers, 26k windows. Conditional on E1
showing separation, sweep d_model and depth to see whether the gap to the identity control closes
with capacity or is flat in it. A flat curve is a much stronger negative than a single point.

**Gate.** Only runs if E1 separates. Otherwise scaling a model that learns nothing learns nothing
faster.

---

## Kill criterion

Stop the programme and write it up as a negative result if **all three** hold:

1. E1 shows no MLP separation from the untrained floor by 19200 steps.
2. E4 shows no readout recovers a trained arm above the identity control.
3. E3 places the real corpus below the SNR threshold where the objective stops working.

### Status 2026-08-10: 3 of 3 met. The criterion has fired.

1. **Met, and worse than stated.** E1 shows no separation at convergence and active
   degradation: the MLP probe ends 23 pooled SD *below* the untrained floor after 32x more
   compute. `results/convergence.json`, finding 8.
2. **Met.** No readout puts a trained arm above dimension-matched raw features; the best JEPA
   readout is +0.4529 against raw +0.5091. `results/readout_ablation.json`, finding 9.
3. **Met by a different mechanism than the one the criterion anticipated.** The SNR framing was
   the wrong instrument. The decomposition in finding 10 shows the headline metric is mostly
   input reconstruction, and that on the only genuinely future target all 16 arms sit in
   [-0.0043, +0.0510], indistinguishable from zero and from raw data.
   `results/target_decomposition.json`.

Per the plan, the programme stops here and is written up as a negative result. Do not continue
on the grounds that a bigger model or a longer run might work; E1 and E4 exist to rule exactly
that out, and they did.

**The one thing that would restart it** is not a better model, it is a better target set. Finding
10 establishes that the current targets cannot detect learned structure: one is unpredictable by
construction of an efficient market and one is a function of the input. A target with genuine
hidden state, such as realised volatility over a future window or the settlement of a different
correlated market, would be a real test and has never been run. That is a new programme with a
new pre-registration, not a continuation of this one.

> **Run on 2026-08-12 as E7, under Pre-registration 2 below.** Realised volatility over a future
> window was the target, exactly as named here. It did not restart the programme. A training lift
> appeared and then failed the control that removes what raw features already explain; see finding
> 14. This paragraph is left as written because it was the correct call at the time, and because
> the experiment it specified is the reason the negative result can now be stated without the
> caveat it was carrying.

That combination means the JEPA objective cannot extract structure from prediction-market ladders
that raw features do not already carry, that the failure is a property of the data rather than of
our implementation, and that we have measured exactly why. Written honestly, that is a better
contribution than most positive results in this area, and it costs the lab nothing further to
stop there.

**Do not** continue past this on the grounds that a bigger model or a longer run might work. The
plan above is designed to rule that out before the question is asked.

---

# Pre-registration 2, 2026-08-12: E7, the target set

The kill criterion above fired and the programme it governed is closed. It named exactly one thing
that would restart the work, and it is the thing below:

> A target with genuine hidden state, such as realised volatility over a future window or the
> settlement of a different correlated market, would be a real test and has never been run. That is
> a new programme with a new pre-registration, not a continuation of this one.

This is that new pre-registration. It is written and committed **before the run**, and it governs
only E7. Nothing here reinterprets a gate above.

### The question

The old result is under-determined, and it has been from the start. Two readings survive every
measurement taken so far:

- **(a)** a JEPA learns nothing useful on this data, or
- **(b)** the target set contained no recoverable hidden state, so no encoder could have
  demonstrated anything and the comparison was never able to discriminate.

Finding 10 is what makes this live rather than pedantic: `implied_width` carried nearly the whole
apparent gap and is documented upstream as derivable from the input, while `log_return_to_settle`
put all 16 arms inside [-0.0043, +0.0510]. One target was a function of the input and the other was
unpredictable. A metric built from those two cannot rank representations, whatever the model does.

### The targets

`causaljepa/targets.py`, all built from arrays `build_corpus.py` already snapshots, at a forward
horizon of 10 steps measured in wall clock rather than array index:

| target | what it is | why it is here |
|---|---|---|
| `fwd_realised_vol` | realised volatility of `implied_spot` over the next horizon | genuinely future and path-dependent, but partly forecastable, so not a clean test alone |
| `vol_forecast_error` | `log(realised / implied)` over the same horizon | **the headline.** The ladder's own forecast is divided out by construction, so the part `implied_width` already carried is removed |
| `fwd_abs_return` | magnitude of the future move | coarser cousin of realised vol, no path dependence |
| `fwd_signed_return` | signed future move | **negative control.** A ten-minute near-martingale that nothing should predict |

### Hard precondition

Every arm must score at or below **+0.05 ridge R²** on `fwd_signed_return`. Anything above that has
seen the future through its features, and the run is void rather than interesting. This is an
assertion in the script, not a log line, on the E2 precedent.

### Gate

Stated in pooled SD across seeds, with a minimum of 2 seeds for the gate to be evaluated at all.
Below that the pooled SD is exactly zero, every margin is `nan`, and both branches would read as
"not met"; E1 and E2 were each decided by a gate whose wording admitted a degenerate case, so this
one refuses to return a verdict instead.

- Best trained arm beats the best raw control on `vol_forecast_error` by **more than 2 pooled SD**
  → the negative result was a property of the **target set**. The programme reopens here.
- **No** new target shows a matched-width training lift above 2 pooled SD
  → the negative result **generalises**: it survives contact with targets that do contain hidden
  state, and reading (a) is the right one.
- Anything between → report the frontier and claim nothing.

### What makes it a fair test

1. The arms are **imported** from `readout_ablation.py`, not reimplemented, so the raw controls,
   the readouts and the extraction are bit-identical to E4's. A difference in the table is a
   difference in the targets.
2. Both target blocks are probed on **the same windows**. The horizon requirement drops windows near
   the end of each event, so the original four are recomputed on the surviving subset and reported
   beside the new four; otherwise a change could be the subset rather than the targets.
3. The blocks are probed **separately**, so each picks its own ridge lambda. Eight mixed columns
   would let the new targets shift the lambda chosen for the old ones and silently break
   comparability with every published number in this repo.
4. Every trained arm is paired with an **untrained encoder at the same readout width**, per rule 5.

### Outcome, 2026-08-12: PARTIAL, then resolved against the model by E7c

The gate returned **PARTIAL** at both horizons, and the partial half was real: a matched-width
training lift of **+0.0433 (+7.73 pooled SD)** on `vol_forecast_error` at the 128-dim `last_patch`
readout, replicating at h=5 at **+0.0443 (+3.76 sd)**. Raw features nonetheless won the headline
target outright, 96 dims at +0.1747 against +0.1206 for the best trained arm at 3072 dims, a
margin of 13.58 pooled SD. The martingale precondition held everywhere, worst +0.0064 against the
+0.05 bar.

The lift did not survive its own control. **E7c** removed what the raw cross-section explains and
re-probed: no arm reaches a residual R² distinguishable from zero, best t = 1.68 against a critical
3.182. The lift is the encoder representing the INPUT better, which the same run confirms directly
with a +0.1392 training lift on `implied_width`. Full record in finding 14.

**E7c's own gate was mis-specified**, the third in this programme. It asked only for a lift above 2
pooled SD and never that the trained arm reach a positive residual R², so it fired on a lift running
between two negative numbers. Recorded as run in `results/residual_probe.json:verdict`, corrected
beside it under `verdict_review`, script criterion tightened for future runs.

**The programme stays closed, and the closure is now much stronger.** The one reading the kill
criterion could not exclude, that the target set was incapable of detecting learning, has been
tested directly and rejected. Do not reopen this on the grounds that some other target might work;
`scripts/horizon_headroom.py` will classify a candidate target in about a minute, and a target that
comes back DERIVABLE or UNPREDICTABLE cannot rank representations no matter what is trained on it.

### Stated in advance, because it bounds the claim

`implied_spot` is itself reconstructed from the ladder, so its increments carry reconstruction noise
and `fwd_realised_vol` overstates true realised volatility. This is identical across arms and does
not bias the comparison between them, but an absolute R² here is not an R² against Bitcoin's
realised volatility, and it will not be reported as one.

---

# Pre-registration 3, 2026-08-12: E8, the cross-sectional latent

The last untested idea in this repo was "the settlement of a different correlated market". This is
it, and it is the only version of the escape hatch still standing after E7.

### Why it is not just another lap

Findings 11 and 12 could not explain why the simulator wins and the market does not. The structural
difference that survives every test: the simulator observes a 3-dimensional latent through **eight
heterogeneous markets**, none of which identifies it alone, while a Kalshi BTC ladder is 24 strikes
that are a smooth function of **two numbers**. There was never a latent for the encoder to recover.

The cached corpus turns out to hold two assets, 1,675 KXBTCD and 1,661 KXETHD events over the same
ten weeks, 99.1% time-overlapping. A strike ladder is a **marginal** distribution; the **dependence**
between two assets is in neither ladder. That makes realised correlation a genuine joint latent, and
reproduces the simulator's geometry on real data.

### Gate, stated before the run

- **E8a, free:** if the best width-matched cross-asset gain on any discriminative target is below
  +0.02, the premise is wrong and we stop without training. Above it, proceed.
- **E8:** best trained arm beats the best raw control on `realised_corr` by more than 2 pooled SD
  → the JEPA wins where the latent exists, and we can name the property that was missing.
- No matched-width lift above 2 pooled SD → the objective fails even where the latent provably
  exists, a far stronger negative than findings 1 to 15.
- **E8c is mandatory regardless of outcome.** After finding 14, no lift on this corpus is reported
  as learning until it survives removal of what raw features explain.

### Outcome: premise confirmed, model still loses

**E8a PROCEEDED**, and it is finding 16, the one clean positive in the programme. `realised_corr`
scores +0.1867 from BTC alone, +0.2520 from ETH alone and +0.3693 from both, a **+0.1064** gain over
a width-matched duplicated ladder, rising monotonically with horizon. Targets that should not need
both assets show no gain, which is the control that makes it credible.

**E8 came back PARTIAL.** Largest training signal in the programme, +0.0819 (+4.54 pooled SD) at the
128-dim `last_patch` readout, and raw features still win by 6.58 pooled SD.

**E8c killed it**, exactly as E7c did. The best untrained arm scores +0.0077 on the residual against
the best trained arm's +0.0042. A random-initialised encoder carries more of what raw features miss
than any trained one. Finding 17.

**A fourth criterion failed on its wording**, and needed two rounds to fix: E8c's conditions were
satisfied by *different arms*, one clearing zero while a second, entirely negative, supplied the
lift. The missing condition was that a trained arm must beat the best *untrained* arm. Recorded as
run in `results/residual_probe_cross.json`, corrected under `verdict_review`.

**The programme is closed for good on this corpus.** Every escape hatch named in advance has now
been opened and found empty: more training (E1), the readout (E4), the regularisers (E2), the
signal-to-noise story (E3), the target set (E7), and the cross-sectional latent (E8). What remains
is not a different target on Kalshi, it is a different domain.

---

## Order of operations

```
E4  readout ablation        <1h    confound check, runs first, no retraining
E1  convergence sweep       2.0h   decisive, overnight
E2  live regulariser        4.9h   closes a published open question
E3  synthetic-to-real gap   ~1d    explains everything
E5  stride-1 cache          ~2h    idle-time engineering, unblocked
E6  scale                   ---    only if E1 separates
```

Timings for E1 and E2 are measured from `results/causal_ablation.json` at 0.123 s/step. E3 and E5
are estimates. Total for the decisive path, E4 through E3, is under two days of wall clock on one
laptop.

## Rules for autonomous execution

1. **Gates are binding.** Do not proceed past a gate whose criterion was not met, and do not
   reinterpret a criterion after seeing the number.
2. **Every run writes `results/*.json` with the full config inlined.** A result without its config
   is not a result.
3. **Report mean and standard deviation across seeds. Never a single seed.**
4. **Never tune toward a desired outcome.** If an arm wins only after hyperparameter search that
   the baseline did not get, it did not win.
5. **The untrained control is run in every experiment**, at the same seeds. It is the floor and it
   is cheap.
6. **`reg_grad_norm` is checked on every run.** The programme has already lost two repos' worth of
   claims to a silently inert regulariser.
7. **Never write into `pm-jepa` or `slate-jepa` during an experiment.** They are read-only sources
   of corpus and baselines. Corrections to them are separate, deliberate commits.
8. **Autonomy stops at the repo boundary.** Experiments, local commits, and pushes to
   `causal-jepa` proceed without asking. Anything that changes `pm-jepa`, `slate-jepa`, or the
   public site waits for a human.
9. **A negative result is a result.** Write it up with the same care as a positive one.
