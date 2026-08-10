# REPRODUCE.md: exact commands, expected runtimes, expected outputs

Everything in `results/` and every table in `docs/EVALS.md` comes from the
commands below, in this order. Total wall clock for the full set is about **54
minutes**, of which 44 is the two training sweeps. The headline result, the
untrained-encoder control in section 6, costs about 60 seconds of that and trains
nothing.

All commands are run from the repo root:

```
cd /private/tmp/claude-501/-Users-nikita/8b0cffcc-15f7-4f97-bb17-9c01e31da432/scratchpad/causal-jepa
```

There is no `pip install -e .` and there is not going to be. `conftest.py` puts
the repo root on `sys.path` for the tests, and every script under `scripts/` does
the same for itself, so a bare checkout runs.

---

## 0. Prerequisites

| requirement | expected | how to check |
|---|---|---|
| interpreter | CPython **3.9.6** | `python3 -VV` |
| torch | **2.6.0**, MPS available | `python3 -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"` |
| numpy | **1.26.2** | `python3 -c "import numpy;print(numpy.__version__)"` |
| pandas | any 2.x | needed only because pm-jepa's `data/dataset.py` imports it |
| pytest | >= 7 | `python3 -m pytest --version` |
| pm-jepa checkout | `/Users/nikita/pm-jepa` | must contain `load_corpus.py` and `results/baselines.json` |
| corpus cache | `~/pm-jepa/data_cache/event_arrays/*.npz` | about 3,400 files, 45 MB |

The versions are pinned exactly in `requirements.txt`, not as bureaucracy. torch
2.6 has no MPS kernel for `linalg.svdvals` or `cummax`, and both
`causaljepa/diagnostics.py` and `causaljepa/train.py` route around that by moving
to CPU. A newer torch that grew those kernels would not break anything, but it
would change which device the diagnostics ran on, and reduction order differs
between backends at about the third decimal, which is the resolution the 2x2 is
read at.

**Do not rebuild the corpus.** `causaljepa/data.py` imports pm-jepa's own
`load_corpus.build()` by path injection and calls it. It never calls pm-jepa's
`main()`, which would write `results/corpus_shape.json`. The pm-jepa tree is
read-only for this project.

### A note on the shell before you start

This machine has an `rtk` proxy that rewrites and compresses Bash output. It has
been observed to turn a `pytest -v` run into the single line `Pytest: 49 passed`
**even through a file redirect**, and to print nothing for a `find` whose targets
demonstrably existed. Any command below whose exact output matters should be run
through a Python subprocess so the proxy is bypassed:

```
python3 -c "
import subprocess, sys, time
t = time.time()
p = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q'],
                   capture_output=True, text=True)
print('WALL %.1fs rc %d' % (time.time() - t, p.returncode))
print(p.stdout)
"
```

Substitute the command you want in the argument list. Everything reported in this
file was measured that way.

---

## 1. Test suite

```
python3 -m pytest tests/ -q
```

**Runtime:** 6.9s reported by pytest, 7.4s wall. Cold cache measured at 7.7s
total. Budget in the SPEC is 60s on CPU.

**Expected output:**

```
.................................................                        [100%]
49 passed in 6.93s
```

Breakdown, from `--collect-only` (parametrised cases counted separately):

| file | collected |
|---|---|
| `test_cache_exactness.py` | 9 |
| `test_causal_mask.py` | 5 |
| `test_directional_diagnostics.py` | 9 |
| `test_model_shapes.py` | 12 |
| `test_no_data_mutation.py` | 6 |
| `test_probe_parity.py` | 8 |
| **total** | **49** |

All seven mandatory SPEC section 9 tests are here, plus the bidirectional-control
halves of tests 1 and 2.

**If it fails:**

| failing test | what it means | what to do |
|---|---|---|
| `test_cache_exactness` | the KV cache is wrong: bad keys, or an off-by-one positional offset | fix `causaljepa/cache.py`. **Do not relax the tolerance.** The SPEC calls this the single most important test in the repo, and the failure mode is silent: a wrong cache still returns plausible embeddings. |
| `test_cache_exactness_has_teeth` | the *bidirectional* control stopped disagreeing with the cache | the causal assertion is now vacuous. Something has broken the bidirectional path, not the cache. |
| `test_causal_mask_no_future_leak` | the causal mask is not being applied on the path the model runs | fix `causaljepa/model.py`. Reading the mask tensor is not a substitute; this test perturbs the input and measures the output. |
| `test_probe_parity` | `causaljepa/probes.py` has drifted from pm-jepa's | every comparison to the +0.449 / +0.460 controls is invalid until this passes. The file is supposed to be byte-identical below the docstring. |
| `test_no_data_mutation` | corpus loading wrote into the pm-jepa tree | see section 11. Note that `git status` in pm-jepa **cannot** catch this on its own: its `.gitignore` lists `__pycache__/` and `*.py[cod]`. The fingerprint in this test is the only real detector. |
| `test_split_matches_pm_jepa` | our event split diverged from pm-jepa's | the controls are no longer scored on the same rows. Nothing downstream is comparable. |
| `test_ema_moves_target_towards_context` | the EMA update is wrong, possibly inverted | an inverted momentum makes the target encoder a near-copy of the context encoder, collapsing the JEPA to trivial self-prediction while every copy diagnostic still looks plausible. This test uses momentum 0.9 and 0.996 for that reason; at 0.5, `m` and `1-m` are the same number and a correct and an inverted EMA are indistinguishable. |

---

## 2. Cross-module integration check

```
python3 scripts/integration_check.py
```

**Runtime:** 2.8s. **Exit code 0** and a final line `all integration checks
passed`.

This checks the seams the unit tests do not, because each unit test owns one
module. Five sections:

1. cache exactness across three call conventions
2. causality by perturbation, both arms
3. streaming speed, about **3x** faster than full re-encode (3.00x and 3.07x on
   two runs; it is a wall-clock timing on a laptop and will not repeat exactly)
4. model / masking / diagnostics wiring: `n_patches` 6, `temporal` masks patches
   `[4, 5]`, `interpolation` masks `[2, 3]`, `interpolation_span(24, 0.25, 4)` is
   `(8, 16)`, `forward()` keys are exactly
   `['context', 'mask_flat', 'patch_mask', 'pred', 'target']` with shapes
   `pred/target/context (B, 144, d)`, `mask_flat (B, 144)`, `patch_mask (B, 6, 24)`,
   target detached, and all four 2x2 cells reach the encoders
5. nan safety of the whole diagnostic block on empty and full masks

**If a geometry line fails** (`temporal masks patches [4, 5]`, or the span), the
masking module and the model disagree about the patch grid. The likely cause is
`patch_length` drifting apart between `CausalJEPA` and `sample_mask`. They agree
at 4 only because `train.py` explicitly threads `model.patch_length` into
`sample_mask`; without that, a drift would move the `interpolation` hole off the
patch grid and `patch_mask`'s `any()` rule would silently widen it into a fatter
`temporal` arm, with no symptom beyond a shifted loss.

---

## 3. End-to-end smoke run

```
python3 scripts/smoke.py
```

Defaults: 300 windows, 30 steps, batch 32, d_model 64, 2 layers, device auto.

**Runtime:** 10.7s wall, of which 1.3s is the corpus load. Writes
`results/smoke.json`. **Exit code 0** and a final line `smoke run clean`.

**Expected output, the summary block:**

```
corpus (25818, 24, 24, 4) loaded in 1.3s, targets ['log_return_to_settle', 'time_to_expiry', 'implied_width', 'window_log_return']
subset 300 windows over 40 events, train 256 / test 44, device mps
...
arm                        pred_loss_fi pred_loss_la     eff_rank copy_alignme copy_loss_ra        lag_1 cross_bounda   ridge_mean
temporal|ctx=b|tgt=b            +0.5238      +0.3245     +21.8528      +0.5232      +4.4393      +0.9252      +0.9194      -2.3167
temporal|ctx=b|tgt=c            +0.5236      +0.3246     +21.8507      +0.5228      +4.4308      +0.9251      +0.9182      -2.3172
interpolation|ctx=b|tgt=c       +0.5232      +0.3240     +20.7636      +0.5258      +4.6834      +0.9251      +0.9222      -2.3200
interpolation|ctx=c|tgt=c       +0.5221      +0.3242     +22.2921      +0.5257      +4.6861      +0.9251      +0.9222      -2.3414
```

Plus, in the two causal-context arms, `cache exact on real corpus data ... max|delta| = 9.5e-07`.

### Two smoke numbers that look like bugs and are not

- **`ridge_mean` is about -2.3.** This is the 300-window subset, not a defect. The
  identity control on that same subset is only +0.077, with `time_to_expiry` at
  -1.64, because the 44 test windows come from held-out events with a different
  expiry distribution. On the full corpus the same pipeline gives the identity
  control +0.4491 (section 8) and a 60-step driver arm +0.326. Do not read a smoke
  ridge as a result.
- **The arms are nearly identical to 3 decimals.** At 30 steps every arm finishes
  near its initialisation, and a pre-norm residual stream barely mixes across
  patches there, so the causal and bidirectional cells cannot differ. `train_arm`
  prints a warning about this:

```
WARNING: steps=30 is inside the 100-step warmup, peak lr is 0.085x nominal.
Diagnostics from this run measure an untrained encoder and must not be read as
an H1 or H2 result.
```

  The cause is that `WARMUP_STEPS = 100` is a fixed constant (pm-jepa's, kept for
  parity) while the cosine decay is scaled to `steps`, so at short step counts the
  two overlap:

| steps | peak lr multiplier |
|---|---|
| 30 | 0.085 |
| 60 | 0.164 |
| 200 | 0.531 |
| 600 | 0.934 |
| 800 | 0.963 |

  **A short run manufactures a fake null.** `lr_mult` and `ema_m` are recorded in
  every history record so a flat results file can be diagnosed without a rerun.

---

## 4. Budget probe (optional, run before changing the protocol)

```
python3 scripts/timing_probe.py
```

**Runtime:** about 2 minutes. Takes no arguments.

Times steady-state seconds per step at the real corpus size and real width, with
warmup steps timed separately, plus the two fixed per-arm costs. Measured:

| strategy | ctx / tgt | steady s/step |
|---|---|---|
| temporal | bidir/bidir | 0.0662 +/- 0.0002 |
| temporal | bidir/causal | 0.0663 +/- 0.0001 |
| temporal | causal/causal | 0.0827 +/- 0.0167 |
| interpolation | all three | 0.086 +/- 0.009 |

Fixed per arm: `represent_all` 6.1s + `run_probes` 1.1s = 7.2s. Budget model:
`arm_seconds = 0.0856 * steps + 7.2`.

Run this before committing to a step count. On MPS the first forward pays graph
compilation and lazy allocator warmup; folding that into the estimate inflates the
per-step cost by enough to talk yourself out of a seed you could have afforded.

---

## 5. The two training sweeps

These are the experiment. 32 arms total: 4 cells x 4 seeds x 2 strategies.

```
python3 scripts/experiment.py --strategy temporal      --seeds 4 --steps 600 --log-every 100 --name causal_ablation
python3 scripts/experiment.py --strategy interpolation --seeds 4 --steps 600 --log-every 100 --name interpolation_arm
```

Every other flag is at its default and is inlined into the output file under
`config`: batch 64, lr 1e-3, lam 0.0 (inert), d_model 128, layers 4, n_held_out 6,
window 24, stride 4, test_frac 0.2, split_seed 0, device auto.

**Runtime:** 22.0 min and 21.6 min on MPS (measured 01:41:01 to 02:02:59 and
02:03:02 to 02:24:35). Per arm, 56s to 83s of training plus about 7s of probing;
16 arms per sweep.

**Writes:** `results/causal_ablation.json` (223 KB) and
`results/interpolation_arm.json` (234 KB), each with full per-seed history and the
config inlined.

**Expected header line:**

```
device mps   corpus (25818, 24, 24, 4)   train 20506 / test 5312
strategy temporal   cells bb,bc,cb,cc   seeds 4   steps 600
```

**Expected per-arm ridge means** (mean +/- sd over 4 seeds):

| cell | temporal | interpolation |
|---|---|---|
| `bb` ctx=bidir tgt=bidir | +0.392 +/- 0.006 | +0.390 +/- 0.007 |
| `bc` ctx=bidir tgt=causal | +0.392 +/- 0.007 | +0.394 +/- 0.009 |
| `cb` ctx=causal tgt=bidir | +0.402 +/- 0.003 | +0.397 +/- 0.004 |
| `cc` ctx=causal tgt=causal | +0.401 +/- 0.004 | +0.395 +/- 0.004 |

The driver ends each sweep by printing the probe table with pm-jepa's controls
interleaved, the diagnostics table, and the target-encoder contrast, and by
stating explicitly that the best arm **LOSES TO** the identity control.

`pred_loss` on seed 0, cell `bb`, temporal, at steps 1/100/200/300/400/500/600:
`0.4577, 0.0636, 0.0400, 0.0382, 0.0354, 0.0287, 0.0294`. The last-100-step change
is +1.7% to +7.3% across arms, which is single-batch noise, not descent; the loss
has flattened by 600.

### To run a subset

```
python3 scripts/experiment.py --strategy temporal --cells bb,bc --seeds 2 --steps 600
```

`--cells` takes any comma list from `bb,bc,cb,cc` (first letter is the context
encoder, second is the target encoder; `b` = bidirectional, `c` = causal). Note
that dropping `cb` and `cc` halves the paired contrast from n=8 to n=4 and removes
the context margin entirely.

---

## 6. The benchmark table, including the untrained encoder control

This is the headline result of the repo and the cheapest command in this file.

```
python3 scripts/benchmark.py --seeds 4
```

**Runtime:** about **60 seconds** on MPS. It trains nothing. The work is 8 probe
fits (2 causal flags x 4 seeds) on an untrained encoder, about 7s each, plus the
corpus load; the trained arms and the controls are read from JSON. Defaults:
`--d-model 128 --layers 4 --device auto --pm-jepa-root /Users/nikita/pm-jepa
--out results/benchmark.json`, which match the training sweeps in section 5 so
the rows are comparable.

**Writes:** `results/benchmark.json` (10 KB), with the per-seed probe fits and the
per-target means for the untrained arms.

**Requires:** `results/causal_ablation.json` and `results/interpolation_arm.json`
from section 5 for the trained rows, `pm-jepa/results/baselines.json` for the
controls and `pm-jepa/results/seeds.json` for pm-jepa's arms. Any missing input is
skipped rather than faked, so a short table means a missing file.

**Expected output:**

```
device mps  corpus (25818, 24, 24, 4)  train 20506 / test 5312
  random-encoder (bidir)  seed 0  ridge +0.3389  mlp +0.3807  (7s)
  random-encoder (bidir)  seed 1  ridge +0.3394  mlp +0.3841  (7s)
  random-encoder (bidir)  seed 2  ridge +0.3546  mlp +0.3839  (7s)
  random-encoder (bidir)  seed 3  ridge +0.3494  mlp +0.3721  (7s)
  random-encoder (causal)  seed 0  ridge +0.3276  mlp +0.3835  (7s)
  random-encoder (causal)  seed 1  ridge +0.3352  mlp +0.3737  (7s)
  random-encoder (causal)  seed 2  ridge +0.3493  mlp +0.3700  (7s)
  random-encoder (causal)  seed 3  ridge +0.3401  mlp +0.3623  (7s)

==============================================================================
arm                                              ridge R^2           mlp R^2
==============================================================================
temporal (control)                           +0.4600+-0.0000     +0.5108+-0.0000
handcrafted (control)                        +0.4595+-0.0000     +0.4858+-0.0000
identity (control)                           +0.4491+-0.0000     +0.5050+-0.0000
ctx=causal tgt=bidir [temporal] (this w      +0.4018+-0.0034     +0.3872+-0.0192
ctx=causal tgt=causal [temporal] (this       +0.4011+-0.0036     +0.3862+-0.0127
ctx=causal tgt=bidir [interp] (this wor      +0.3974+-0.0039     +0.3830+-0.0092
ctx=causal tgt=causal [interp] (this wo      +0.3949+-0.0044     +0.3839+-0.0084
ctx=bidir  tgt=causal [interp] (this wo      +0.3942+-0.0092     +0.3854+-0.0067
ctx=bidir  tgt=causal [temporal] (this       +0.3924+-0.0069     +0.3809+-0.0055
ctx=bidir  tgt=bidir [temporal] (this w      +0.3919+-0.0058     +0.3797+-0.0049
ctx=bidir  tgt=bidir [interp] (this wor      +0.3896+-0.0067     +0.3792+-0.0028
jepa-temporal (pm-jepa)                      +0.3771+-0.0095     +0.3227+-0.0128
random-encoder (bidir)                       +0.3456+-0.0067     +0.3802+-0.0049
jepa-slate (pm-jepa)                         +0.3426+-0.0059     +0.3160+-0.0042
jepa-contiguous (pm-jepa)                    +0.3394+-0.0064     +0.3263+-0.0045
random-encoder (causal)                      +0.3381+-0.0079     +0.3724+-0.0076
==============================================================================
wrote results/benchmark.json
```

**How to read it.** The row that matters is `random-encoder (bidir)`, the
untrained encoder of the identical architecture. Against it, 600 steps of
training move ridge by **+0.0463** (Welch t = 9.09) and MLP by **-0.0005**
(t = -0.12). Training bought linear accessibility and nothing a nonlinear probe
can find. See `docs/EVALS.md` section 2, "The untrained encoder".

The `jepa-*` rows use a PatchTST backbone and are architecture-confounded against
everything else in the table; the untrained-versus-trained comparison is
within-architecture and is the clean one.

**If the numbers move:** the untrained rows are pure forward passes of a freshly
seeded model, so they are the most reproducible in the repo. If they shift by more
than about 0.005, check `torch.__version__` and the device first, then re-run
section 8 to confirm the corpus and the probe have not changed.

---

## 7. H1 with the training trajectory removed

```
python3 scripts/h1_weightmatched.py
```

**Runtime:** 53s measured on MPS (the JSON records `seconds` 52.8). Defaults:
`--steps 600 --seed 0 --batch 256 --strategy temporal`. Trains one `bb` arm, then
reads its target tower out twice on identical weights, once bidirectional and once
causal, so the flip is a pure intervention with no retraining.

**Writes:** `results/h1_weight_matched.json` (3 KB).

**Expected numbers.** Per-patch cosine between the two readouts, and cross-boundary
target autocorrelation under each:

| patch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| cos(bidir, causal) | 0.9812 | 0.9902 | 0.9943 | 0.9969 | 0.9985 | 0.9995 |
| relative L2 | 0.1271 | 0.0896 | 0.0657 | 0.0481 | 0.0315 | 0.0163 |

`cross_boundary` 0.9483 bidirectional against 0.9377 causal under temporal
masking, `baseline_cos` 0.7062 against 0.6988, so 0.0033 of the 0.0107 raw move
survives the floor. The leak the repo was built around is real and about an order
of magnitude too small to explain a copy alignment of 0.96.

---

## 8. Reproduce the control line

Not part of any script, and worth running whenever the corpus cache changes,
because it is the only check that the controls this repo quotes are the controls
this repo's pipeline would compute.

```
python3 -c "
import sys, numpy as np
sys.path.insert(0, '.')
from causaljepa.data import load_corpus, split_by_event
from causaljepa import probes
X, Y, owner, names = load_corpus('/Users/nikita/pm-jepa')
tr, te = split_by_event(owner, frac=0.2, seed=0)
print('windows', len(X), 'events', len(np.unique(owner)), 'train', tr.sum(), 'test', te.sum())
f = X[:, -4:].mean(axis=1).reshape(len(X), -1).astype(np.float64)   # pm-jepa identity_features
r = probes.run_probes(f[tr], Y[tr], f[te], Y[te], names, device='cpu')
print('identity dim', f.shape[1], 'ridge_mean %.4f' % r['ridge_mean'], 'lambda', r['ridge_lambda'])
print({k: round(v, 4) for k, v in r['ridge'].items()})
"
```

**Runtime:** about 3s with the npz cache warm in the page cache, a few seconds
more cold. Most of it is the corpus load and the MLP probe.

**Expected output, exactly:**

```
windows 25818 events 3127 train 20506 test 5312
identity dim 96 ridge_mean 0.4491 lambda 1000.0
{'log_return_to_settle': 0.0152, 'time_to_expiry': 0.7394, 'implied_width': 0.8033, 'window_log_return': 0.2384}
```

pm-jepa's published figure is 0.44908884624454815 at `lambda=1000`, dim 96, with
the identical four per-target values. If this line does not reproduce, **stop**:
either the corpus changed or the probe drifted, and no comparison to +0.449 or
+0.460 is valid until it does.

---

## 9. Cache benchmark

```
python3 scripts/cache_bench.py --cpu-max-p 512
```

The `--cpu-max-p` flag is required to reproduce the published table; the default
is 128, which skips the CPU P=512 row where the strongest result lives.

**Runtime:** 3.6 min of benchmark time (0.6 + 2.6 + 12.6 + 106.3s on MPS, 0.2 +
1.2 + 8.9 + 83.1s on CPU) plus process startup. Writes
`results/cache_bench.json` (85 KB, with per-tick timings).

**Expected output:**

```
mps  P=6     exact=True  maxdiff=0.00e+00  full     8.61ms/tick  cache     7.00ms/tick  speedup  1.22x  tok/s     2852 ->     3474
mps  P=32    exact=True  maxdiff=1.4e-06   full     9.70ms/tick  cache     8.38ms/tick  speedup  1.16x  tok/s     2503 ->     2903
mps  P=128   exact=True  maxdiff=1.4e-06   full    10.81ms/tick  cache     8.40ms/tick  speedup  1.26x  tok/s     2276 ->     2859
mps  P=512   exact=True  maxdiff=1.9e-06   full   166.93ms/tick  cache    39.89ms/tick  speedup  4.19x  tok/s      144 ->      602
cpu  P=6     exact=True  maxdiff=1.4e-06   full     2.77ms/tick  cache     1.83ms/tick  speedup  2.15x  tok/s     6012 ->    12942
cpu  P=32    exact=True  maxdiff=1.4e-06   full     9.53ms/tick  cache     1.79ms/tick  speedup  4.09x  tok/s     2301 ->     9414
cpu  P=128   exact=True  maxdiff=1.5e-06   full    19.09ms/tick  cache     2.45ms/tick  speedup  7.65x  tok/s     1237 ->     9455
cpu  P=512   exact=True  maxdiff=1.7e-06   full   156.10ms/tick  cache     5.05ms/tick  speedup 30.92x  tok/s      154 ->     4754
```

`exact=True` in every row is the gate. The benchmark refuses to report a speedup
it has not first verified against a full re-encode at `atol=1e-4`, because a cache
that is fast and wrong is worse than no cache.

**Timings will not reproduce exactly.** These are wall times on a laptop, batch 1,
no process control, no `torch.compile`. The *shape* of the table is what should
reproduce: exactness in every row, a modest win at P=6, and the CPU speedup rising
monotonically 2.15x, 4.09x, 7.65x, 30.92x with P.

---

## 10. Re-derive every table without training

```
python3 scripts/report.py
```

**Runtime:** under a second. Reads `results/causal_ablation.json`,
`results/interpolation_arm.json` and `results/cache_bench.json` and prints, per
strategy: the probe table with controls, the diagnostics table with mean +/- sd,
the unpaired target contrast, the **paired** target contrast (labelled "the
powered test"), and the context contrast. Then the H4 table.

It does **not** read `results/benchmark.json`, so the benchmark table and the
untrained-encoder control are not in its output; those come from section 6.

Nothing in `docs/EVALS.md` was retyped. If a number in that document disagrees
with the output of `report.py` or `benchmark.py`, the script wins.

The paired contrast is the primary test. Its expected first rows, temporal:

```
H1 autocorr cross_boundary            -0.01014   0.00603     -4.75  0+/8- of 8           lower
   (floor: baseline_cos)              -0.00898   0.00681     -3.73  1+/7- of 8             n/a
H1 xbound OVER FLOOR                  -0.00116   0.00214     -1.53  2+/6- of 8           lower
H1 lag_3 OVER FLOOR                   -0.00250   0.00226     -3.12  0+/8- of 8           lower
```

---

## 11. Verify nothing was written into pm-jepa

Run after any sweep.

```
python3 -c "
import subprocess, sys
p = subprocess.run(['git', 'status', '--porcelain'], cwd='/Users/nikita/pm-jepa',
                   capture_output=True, text=True)
print('LINES:', len(p.stdout.splitlines()))
print(p.stdout)
"
```

**Expected: 0 lines.**

This is necessary but **not sufficient**. pm-jepa's `.gitignore` lists
`__pycache__/` and `*.py[cod]`, so a stray `.pyc` write would not appear here.
`tests/test_no_data_mutation.py::test_no_data_mutation` fingerprints the tree
(path, size, mtime) around a **cold** corpus load and is the real detector. It
forces the load cold by dropping the corpus memo, the module memo and pm-jepa's
`sys.modules` entries, so the import and the ~3,400-file npz walk genuinely re-run
inside the fingerprint window, and it asserts that they did. Check it with:

```
python3 -c "
import subprocess, sys
p = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_no_data_mutation.py',
                    '-q', '--durations=8'], capture_output=True, text=True)
print(p.stdout)
"
```

Expected: `1.24s call ... test_no_data_mutation`. If that duration ever collapses
to 0.00s, the memo was warm, the fingerprint window contained a dict lookup, and
the test has gone vacuous. The `assert seen, "load was served from a memo; this
test measured nothing"` inside it exists to catch exactly that, but the duration
is the cheap external check.

The guard itself lives in `causaljepa/data.py`: `sys.dont_write_bytecode` is set
to `True` around the pm-jepa import and restored afterwards. On this machine the
`__pycache__` directories are already warm, so importlib would not write anyway,
and deleting the guard changes nothing observable. It earns its keep on a clean
checkout, which is precisely the case this machine cannot exercise.
`test_pm_jepa_import_runs_with_bytecode_writing_off` asserts the flag's value at
the moment the pm-jepa module body executes, which does cover it.

---

## 12. Triage: what to check when a number comes out different

| symptom | first thing to check | then |
|---|---|---|
| corpus is not `(25818, 24, 24, 4)`, or events is not 3127 | the npz cache under `~/pm-jepa/data_cache/event_arrays/` changed | every comparison to pm-jepa's controls is void. Re-run section 8; if the identity control is no longer +0.4491, the controls must be recomputed with `python3 baselines.py` inside pm-jepa before any arm is interpreted. |
| a trained arm no longer beats the untrained encoder on ridge | `results/benchmark.json`, the `random-encoder (bidir)` row | the headline is that gap (+0.0463, t = 9.09). If it has closed, either the arms did not train or the untrained rows were computed at a different `--d-model` / `--layers` than the sweeps. Re-run section 6 with the defaults. |
| all four cells report identical diagnostics | `lr_mult` in the history records | if the last `lr_mult` is small, or `steps < 100`, the run never left warmup. This is the fake-null failure mode; see section 3. Re-run at 600 steps. |
| `pred_loss` does not fall | `reg_grad_norm`, and whether `nan` appears anywhere | `reg_grad_norm` must be exactly `0.0` in every record. Anything else means a regulariser crept back in and is now contributing gradient, which the SPEC forbids. |
| `ridge_mean` is large and negative | how many windows the run used | a subset of a few hundred windows gives about -2.3 for both the model and the identity control. Only full-corpus numbers are comparable. |
| `eff_rank` near 1, or `min_dim_std` near 0 | `dimension_stats` in the history | representational collapse. None of the copy or autocorrelation numbers mean anything in that state. |
| `baseline_cos` has risen to meet `lag_1` | the gap between them | that is collapse, not leakage. Expected healthy values: floor +0.63 to +0.66 against `lag_1` +0.93 to +0.95. |
| `interp_advantage` is `nan` | the masking strategy | `nan` is correct under `temporal`: `paired_frac` is 0.0 because no masked patch has a future reference. It is finite (`paired_frac` 1.0) only under `interpolation`. |
| `boundary_excess_matched` is `nan` | the masking strategy | correct and expected for both `temporal` and `interpolation`. The mask is identical for every `(b, k)`, so a given pair is either always straddling or never, and there is no in-sample control. |
| `extrap_loss_ratio` equals `copy_loss_ratio` | nothing | expected. Measured max difference 2.4e-07 across all 224 records. `nearest_visible_reference` prefers the past and every masked position has a past reference, so the two oracles select the same reference. |
| `copy_loss_ratio` moved but you cannot tell why | `model_loss_on_copyable` and `extrap_oracle_loss` | both terms are logged for exactly this. A falling ratio can mean the model improved or the oracle got harder; in the reported runs it was the latter. |
| the H1 contrast is smaller than expected | `baseline_cos` in the same records | subtract the floor. In the published runs the floor accounts for most of the raw `cross_boundary` move. `report.py:derived()` computes `*_over_floor`. |
| deltas differ from the published ones in the third decimal | device, torch version | MPS is not bitwise reproducible across torch versions, and the diagnostics take a CPU round trip. Expect agreement to about 3 decimals on the diagnostics and about 0.005 on `ridge_mean`. **The robust quantity is the sign-consistency column** (`0+/8-`), not the exact delta. If the signs flipped, that is a real difference; if only the fourth decimal moved, it is not. |
| a cache row reports `exact=False` | `causaljepa/cache.py` | a bug, not a tolerance. Do not weaken 1e-4. |
| a command prints suspiciously terse output | the `rtk` proxy | re-run it through the Python subprocess wrapper in section 0. |

---

## 13. Files this produces

| path | size | what it is |
|---|---|---|
| `results/causal_ablation.json` | 223 KB | temporal 2x2, 4 seeds, 600 steps, full per-seed history |
| `results/interpolation_arm.json` | 234 KB | interpolation 2x2, same protocol |
| `results/benchmark.json` | 10 KB | every arm against the controls, including the untrained encoder; per-seed probe fits |
| `results/h1_weight_matched.json` | 3 KB | H1 read out both ways on identical weights, per-patch cosines |
| `results/cache_bench.json` | 85 KB | H4, per-tick timings, `regime` caveat inlined |
| `results/smoke.json` | 2 KB | smoke summary, not a result |
| `results/_driver-exercise-60steps-not-a-result.json` | 22 KB | a 60-step driver exercise, named so nobody quotes it |

Every results file inlines its full config. A file with a bare R^2 and no corpus
fingerprint is unfalsifiable later: nobody can tell whether an arm improved or the
corpus grew.
