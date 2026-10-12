# Contributing to causal-jepa

This repository records a negative result: JEPA representations of Kalshi strike ladders never
beat the raw inputs. The programme is closed, but some of its findings are explicitly open
(finding 22, E12) or bounded in ways that are cheap to test, and those are the useful places to
contribute. Read [FINDINGS.md](FINDINGS.md), including "What this work did not establish", and
[docs/EVALS.md](docs/EVALS.md) before starting. Starter projects are listed on
[physicalreasoning.ai/open-problems](https://physicalreasoning.ai/open-problems/) and as issues
labelled `good first issue`, `help wanted`, `research` and `infra`.

## Propose work in an issue first

Open an issue, or comment on an existing one, before writing code. Say:

- the question, in one sentence;
- the corpus it runs on (the published snapshot, or a rebuild from Kalshi's public API);
- for research: the gate you would pre-register (see below), and the compute it needs;
- for infra: what changes and how it is tested.

A maintainer will reply on scope before anything is run.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/). There is no package to build; the repository
is importable from a bare checkout.

```bash
git clone https://github.com/physicalreasoning/causal-jepa && cd causal-jepa
uv venv --python 3.11
uv pip install -r requirements.txt
```

The pins in `requirements.txt` (torch 2.6.0, numpy 1.26.2) are the versions the reported numbers
were measured on; keep them unless an issue agrees otherwise.

The corpus is not committed. Rebuild it from Kalshi's unauthenticated public API, no key needed:

```bash
uv run python scripts/build_corpus.py --series KXBTCD --max-events 2000
```

Kalshi events settle and roll, so a rebuild covers different events than the snapshot in
`results/`. It reproduces the method, not the fourth decimal; say which corpus you used in every
result.

## Tests

CI runs the suite on Python 3.11 and 3.12:

```bash
uv run python -m pytest tests/ -q
```

Without a corpus, 36 tests run and 25 skip; the skipped ones need a corpus. There is no lint
step in CI; match the style of the surrounding code and do not reformat files you are not
otherwise changing.

## Research rules

1. **Pre-register before fetching data or training.** Add the experiment to
   [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) with its gate (what passes, what fails, what
   each outcome would mean) and push it before the run, as Pre-registrations 2 and 3 did. If a
   new corpus is involved, write the gate before building it.
2. **Check headroom first.** Before training anything against a new target or corpus, run the
   raw-feature headroom check (finding 15, `scripts/horizon_headroom.py`). A target that raw
   features already explain, or that nothing explains, cannot rank representations.
3. **Run the controls.** Every arm is compared with raw features, a dimension-matched control
   and an untrained encoder of the same architecture at the same seeds. A training lift counts
   only if it survives residualisation against the strongest raw control (findings 14 and 21).
4. **Sealed tests and finished runs are not re-run with new settings.** A completed gate stands
   as recorded. If a gate was mis-specified, the correction goes beside the original, with the
   reason, and the original result is kept.
5. **Publish negative results.** Every pre-registered run is written up in FINDINGS.md whatever
   it returns, with its raw output in `results/<name>.json`. Corrections go beside the original
   text, never in place of it.
6. **Honest uncertainty.** Gates here are in pooled standard deviations across seeds, which need
   at least two complete seeds; a seed that did not finish is dropped whole (finding 22). Any
   bootstrap resamples whole events or dates, never windows, because windows from one event
   overlap. If you compare anything with a market price, use the market priced at its bids
   (empty bids floored at 0.5 cents, renormalised), not the mid (isotherm FINDINGS §41).
7. **No data or secrets in git.** Never commit a corpus, `data_cache/`, `*.npz` files, keys or
   `.env` files. Result files inline their config; check them for absolute paths from your
   machine before committing.

## Commits and pull requests

- One change per pull request, linked to its issue.
- Commits are brief: a short subject in the imperative, no trailing period, a body only when the
  why is not obvious. Existing examples: `E10: staleness is not the mechanism, refuting my own
  finding-19 inference`, `Soften the corroboration wording`.
- No AI or tool trailers in commits or pull requests (no `Co-authored-by` lines for tools, no
  "generated with" lines).
- No em dashes in code, docs or commit messages; use commas, colons or parentheses.

Contributions are licensed under Apache-2.0, the license of this repository.
