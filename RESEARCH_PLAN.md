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

**Gate.**
- MLP R² of the trained encoder exceeds the untrained floor by more than 2 pooled standard
  deviations at any checkpoint: **the objective does learn, and the 600-step null was a budget
  artefact.** Record the step where separation appears, and re-run E4 and the causal 2x2 at that
  budget.
- No separation by 19200 steps: **strong negative.** The JEPA objective adds no probe-visible
  information on this corpus at any budget we can afford. Proceed to E3, which becomes the
  explanation rather than a side quest.

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

That combination means the JEPA objective cannot extract structure from prediction-market ladders
that raw features do not already carry, that the failure is a property of the data rather than of
our implementation, and that we have measured exactly why. Written honestly, that is a better
contribution than most positive results in this area, and it costs the lab nothing further to
stop there.

**Do not** continue past this on the grounds that a bigger model or a longer run might work. The
plan above is designed to rule that out before the question is asked.

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
