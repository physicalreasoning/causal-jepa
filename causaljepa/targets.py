"""Probe targets with genuine hidden state.

WHY THIS EXISTS. `data/dataset.py` documents two of its four targets as
DERIVABLE from the input: `implied_width` is the scale of the implied
distribution, and `window_log_return` is a move that happened inside the window.
E3c found nearly the whole apparent gap between arms living in `implied_width`,
so the headline metric was largely measuring input reconstruction. On
`log_return_to_settle`, the one genuinely future target, all 16 arms landed in
[-0.0043, +0.0510], which is too narrow a band to rank anything.

That leaves the programme's negative result under-determined. Two readings fit
the evidence equally well:

  (a) a JEPA learns nothing useful on this data, or
  (b) the TARGET SET had no recoverable hidden state, so no encoder could have
      demonstrated anything and the comparison was never able to discriminate.

Nothing measured so far separates them, and the difference matters: (a) closes
the question, (b) says the question was never asked. This module builds targets
designed to contain hidden state, so the experiment that consumes it
(`scripts/hidden_state_targets.py`) can report them beside the original four ON
THE SAME WINDOWS and tell (a) from (b).

THE CONSTRUCTION. Everything comes from arrays `build_corpus.py` already
snapshots: `state` = (implied_spot, implied_width, seconds_to_settlement) and
`ts`. Nothing here re-downloads, and no field is used that the model can see,
because the model's input is `obs` alone.

    fwd_realised_vol     realised volatility of implied_spot over the next
                         `horizon` steps. GENUINELY FUTURE and path-dependent:
                         it is a property of a price path that has not happened
                         at window end. Partly forecastable from the input,
                         since volatility clusters and `implied_width` is the
                         market's own forecast of it, so it is not a clean test
                         on its own. It is reported to make that forecastable
                         part visible rather than to hide it.

    vol_forecast_error   log(realised / implied) over the same horizon. THE
                         HEADLINE. The ladder's own volatility forecast is
                         divided out, so the part of `fwd_realised_vol` that
                         `implied_width` already carries is removed BY
                         CONSTRUCTION. What remains is the market's forecast
                         error, which is exactly the quantity that cannot be
                         read off the input the way `implied_width` could. If a
                         learned representation is ever going to beat raw
                         features on this corpus, this is where it happens.

    fwd_abs_return       |log(spot_end+h / spot_end)|. Future move magnitude,
                         a coarser and more robust cousin of realised vol with
                         no dependence on the path in between.

    fwd_signed_return    log(spot_end+h / spot_end). The NEGATIVE CONTROL. A
                         near-martingale over ten minutes, so every arm should
                         score near zero. Anything that scores well here has
                         leaked the future, and the run is void rather than
                         interesting.

TIME IS WALL CLOCK, NOT ARRAY INDEX. Minutes that fail `validate_ladder` are
dropped when the event is cached, so `h` steps forward in the array is not `h`
minutes of market. Every horizon here is measured with `ts`, and windows whose
forward span stretches beyond `max_gap_factor` times the nominal horizon are
dropped rather than silently treated as contiguous. Scaling the implied
forecast to the horizon uses the same true elapsed seconds.

KNOWN LIMITATION, stated because it bounds the claim. `implied_spot` is itself
reconstructed from the ladder, so its increments carry reconstruction noise on
top of the underlying move. `fwd_realised_vol` therefore overstates true
realised volatility, and part of what any probe recovers from it is the noise
process of our own reconstruction. This is identical across arms, so it does
not bias the comparison BETWEEN them, but it does mean an absolute R^2 here is
not an R^2 against Bitcoin's realised volatility.
"""
import pathlib
from typing import List, Optional, Tuple

import numpy as np

ORIG_TARGET_NAMES = ("log_return_to_settle", "time_to_expiry",
                     "implied_width", "window_log_return")

NEW_TARGET_NAMES = ("fwd_realised_vol", "vol_forecast_error",
                    "fwd_abs_return", "fwd_signed_return")

# Scale guard for the log ratio in `vol_forecast_error`. Realised and implied
# volatility over ten minutes sit around 1e-3 to 1e-2 in log-return units, so
# 1e-6 is small enough to leave ordinary values untouched and large enough to
# keep a degenerate flat path from producing -inf.
_EPS = 1e-6


def windows_from_event(obs: np.ndarray, ts: np.ndarray, state: np.ndarray,
                       settle: float, window: int, stride: int, horizon: int,
                       max_gap_factor: float = 2.0):
    """One event -> (X, Y_orig, Y_new) over windows that have a future.

    Window enumeration mirrors `data/dataset.py::windows_from_event` exactly,
    same `range(0, T - window + 1, stride)` and same targets-at-window-END
    convention, so the surviving windows are a SUBSET of the published corpus
    rather than a different slicing of it. That is what makes the original four
    targets, recomputed here, a valid within-run control.
    """
    T = obs.shape[0]
    if T < window:
        return None

    xs, yo, yn = [], [], []
    for start in range(0, T - window + 1, stride):
        end = start + window - 1
        fut = end + horizon
        if fut > T - 1:
            continue                      # no future left: the window is dropped

        spot, width, t_rem = (float(v) for v in state[end])
        spot_0 = float(state[start][0])
        if spot <= 0.0 or spot_0 <= 0.0 or width <= 0.0 or t_rem <= 0.0:
            continue

        dt = float(ts[fut] - ts[end])
        if dt <= 0.0 or dt > max_gap_factor * horizon * 60.0:
            continue                      # a hole in the minute grid, not a horizon

        path = np.asarray(state[end:fut + 1, 0], dtype=np.float64)
        if (path <= 0.0).any():
            continue

        r = np.diff(np.log(path))
        rv = float(np.sqrt((r ** 2).sum()))
        # Diffusive scaling of the market's own forecast down to this horizon.
        # `width / spot` is the implied scale to SETTLEMENT, `t_rem` seconds
        # away, so the sqrt ratio converts it to the same span as `rv`.
        iv = float(width / spot * np.sqrt(dt / t_rem))
        fwd = float(np.log(path[-1] / path[0]))

        xs.append(obs[start:start + window])
        yo.append((
            np.log(settle / spot),                  # genuinely future
            t_rem,                                  # not in the input
            width / spot,                           # derivable: sanity check
            np.log(spot / spot_0),                  # derivable: sanity check
        ))
        yn.append((
            rv,                                     # future, partly forecastable
            np.log((rv + _EPS) / (iv + _EPS)),      # future, forecast divided out
            abs(fwd),                               # future magnitude
            fwd,                                    # negative control
        ))

    if not xs:
        return None
    return (np.stack(xs).astype(np.float32),
            np.array(yo, dtype=np.float32),
            np.array(yn, dtype=np.float32))


def _cache_dir(root) -> pathlib.Path:
    return pathlib.Path(root).expanduser().resolve() / "data_cache" / "event_arrays"


def load_corpus(root=None, window: int = 24, stride: int = 4, horizon: int = 10,
                max_gap_factor: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                           List[str], List[str], dict]:
    """-> (X, Y_orig, Y_new, owner, orig_names, new_names, stats).

    Walks the cache directly rather than delegating to pm-jepa's `load_corpus`,
    because that builder returns only the four original targets and has no
    notion of a forward horizon. The walk order, the readable-file skip, and the
    `owner` numbering all reproduce `pm-jepa/load_corpus.py::load_events` so
    that `split_by_event` puts the same events on the same side of the split as
    every previous experiment in this repo.
    """
    from . import data as cj_data
    root = cj_data._resolve_root(root)
    cache = _cache_dir(root)

    X, Yo, Yn, owner = [], [], [], []
    n_events = n_used = n_short = n_unreadable = 0
    for i, f in enumerate(sorted(cache.glob("*.npz"))):
        try:
            with np.load(f) as z:
                obs, ts, state = z["obs"], z["ts"], z["state"]
                settle = float(z["settle"][0])
        except (ValueError, OSError, KeyError):
            n_unreadable += 1
            continue
        n_events += 1
        if obs.shape[0] < window:
            n_short += 1
            continue
        built = windows_from_event(obs, ts, state, settle, window, stride,
                                   horizon, max_gap_factor)
        if built is None:
            continue
        x, yo, yn = built
        X.append(x)
        Yo.append(yo)
        Yn.append(yn)
        owner.extend([i] * len(x))
        n_used += 1

    if not X:
        raise RuntimeError(
            "no windows survived horizon={} at window={}, stride={}".format(
                horizon, window, stride))

    stats = {"events_readable": n_events, "events_unreadable": n_unreadable,
             "events_too_short": n_short, "events_contributing": n_used,
             "horizon_steps": horizon, "max_gap_factor": max_gap_factor,
             "window": window, "stride": stride}
    return (np.concatenate(X), np.concatenate(Yo), np.concatenate(Yn),
            np.array(owner), list(ORIG_TARGET_NAMES), list(NEW_TARGET_NAMES),
            stats)


def describe(Y: np.ndarray, names) -> dict:
    """Per-target moments, inlined into every result file.

    A probe R^2 with no record of the target's own scale and skew is not
    auditable later: a target that turned out to be almost constant, or one
    dominated by a handful of outliers, produces a number that looks like a
    finding.
    """
    out = {}
    for i, n in enumerate(names):
        col = Y[:, i].astype(np.float64)
        out[n] = {"mean": float(col.mean()), "std": float(col.std()),
                  "min": float(col.min()), "max": float(col.max()),
                  "p01": float(np.percentile(col, 1)),
                  "p50": float(np.percentile(col, 50)),
                  "p99": float(np.percentile(col, 99)),
                  "finite_frac": float(np.isfinite(col).mean())}
    return out
