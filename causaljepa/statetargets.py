"""Current-but-hidden targets: the prediction finding 20 makes, made testable.

Finding 20's surviving hypothesis is that the simulator and this programme were
never doing the same task. The simulator probes `score_diff`, `score_total` and
`time_remaining` AT THE WINDOW END, which is current hidden state reached by
inverting a known noisy forward map. Every Kalshi target that mattered was a
forecast in a near-efficient market. Inversion is a problem a representation can
help with; forecasting the unforecastable is not.

That hypothesis predicts something specific and falsifiable: **a Kalshi target
that is current-but-hidden, rather than future, should show a matched-width
training lift that SURVIVES residualisation**, where every future target's lift
died (findings 14 and 17).

CONSTRUCTING "CURRENT BUT HIDDEN" WITHOUT A ORACLE. The simulator can hand over
its latent because it generated it. Here there is no ground truth, so the
standard substitute is a SMOOTHER: estimate the state at time t using a window
CENTRED on t, which uses minutes after t only to cancel noise, not to see the
future. The instantaneous reading is the filtered estimate; the centred one is
the smoothed estimate; the difference is the measurement error in what the model
can see, and recovering it is exactly inversion.

    spot_denoise    log(centred mean of implied_spot / implied_spot at t)
                    how wrong the instantaneous level reading is right now
    width_denoise   log(centred mean of implied_width / implied_width at t)
                    how wrong the instantaneous scale reading is right now
    vol_state_error log(centred realised vol / implied vol at t)
                    the market's current volatility misestimate, centred rather
                    than forward, so it is a state error and not a forecast error

    fwd_realised_vol   the FORECAST contrast, forward-only, kept in the same run
    fwd_signed_return  the martingale control, which nothing should predict

WHY THE CONTRAST HAS TO BE IN THE SAME RUN. Findings 14 and 17 both measured
lifts on forward targets that dissolved under residualisation. If a current-state
target behaves differently, that difference has to be measured on the same
windows, the same encoder and the same probes, or it is a comparison across runs
and worth nothing.

HONEST LIMITATION, stated before the result. A centred window contains minutes
after t, so these targets are not purely current: a model that could forecast
would score on them too. What makes them different from a forecast target is
where the mass of the signal sits. Cancelling noise around a known present is the
dominant term; predicting the next five minutes is not. This weakens the test
rather than rigging it, because the forward contrast target in the same run has
access to exactly the same forecasting ability and is expected to score worse.
"""
import pathlib
from typing import List, Tuple

import numpy as np

STATE_TARGET_NAMES = ("spot_denoise", "width_denoise", "vol_state_error",
                      "fwd_realised_vol", "fwd_signed_return")

# Which of the above are current-state and which are forward. Consumed by the
# experiment so the split is declared here rather than assumed in a table.
KIND = {"spot_denoise": "current", "width_denoise": "current",
        "vol_state_error": "current", "fwd_realised_vol": "future",
        "fwd_signed_return": "future"}

_EPS = 1e-6


def windows_from_event(obs, ts, state, settle, window: int, stride: int,
                       half: int = 5, max_gap_factor: float = 2.0):
    """One event -> (X, Y). Targets at the window END, smoothed over +-`half`.

    Window enumeration mirrors `data/dataset.py::windows_from_event` so the
    survivors are a subset of the published corpus, as in `causaljepa/targets.py`.
    A window is kept only when `half` minutes exist on BOTH sides of its end, so
    the centred estimate is genuinely centred rather than truncated.
    """
    T = obs.shape[0]
    if T < window:
        return None
    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        end = start + window - 1
        lo, hi = end - half, end + half
        if lo < 0 or hi > T - 1:
            continue
        spot, width, t_rem = (float(v) for v in state[end])
        if spot <= 0 or width <= 0 or t_rem <= 0:
            continue
        dt = float(ts[hi] - ts[lo])
        if dt <= 0 or dt > max_gap_factor * 2 * half * 60.0:
            continue

        path = np.asarray(state[lo:hi + 1, 0], dtype=np.float64)
        wpath = np.asarray(state[lo:hi + 1, 1], dtype=np.float64)
        if (path <= 0).any() or (wpath <= 0).any():
            continue

        sm_spot = float(path.mean())
        sm_width = float(wpath.mean())
        r = np.diff(np.log(path))
        centred_rv = float(np.sqrt((r ** 2).sum()))
        iv = float(width / spot * np.sqrt(dt / t_rem))

        # Forward contrast, same horizon length as the centred half-window so
        # the two are comparable in how much future they touch.
        fhi = end + half
        fpath = np.asarray(state[end:fhi + 1, 0], dtype=np.float64)
        fr = np.diff(np.log(fpath))
        fwd_rv = float(np.sqrt((fr ** 2).sum()))
        fwd = float(np.log(fpath[-1] / fpath[0]))

        xs.append(obs[start:start + window])
        ys.append((
            np.log(sm_spot / spot),
            np.log(sm_width / width),
            np.log((centred_rv + _EPS) / (iv + _EPS)),
            fwd_rv,
            fwd,
        ))
    if not xs:
        return None
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def load_corpus(root=None, window: int = 24, stride: int = 4, half: int = 5,
                max_gap_factor: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """-> (X, Y, owner, target_names, stats). Walks the same cache as targets.py."""
    from . import data as cj_data
    root = cj_data._resolve_root(root)
    cache = pathlib.Path(root) / "data_cache" / "event_arrays"

    X, Y, owner = [], [], []
    n_used = 0
    for i, f in enumerate(sorted(cache.glob("*.npz"))):
        try:
            with np.load(f) as z:
                obs, ts, state = z["obs"], z["ts"], z["state"]
                settle = float(z["settle"][0])
        except (ValueError, OSError, KeyError):
            continue
        built = windows_from_event(obs, ts, state, settle, window, stride,
                                   half, max_gap_factor)
        if built is None:
            continue
        x, y = built
        X.append(x)
        Y.append(y)
        owner.extend([i] * len(x))
        n_used += 1
    if not X:
        raise RuntimeError("no windows survived the centred-window requirement")
    stats = {"events_contributing": n_used, "window": window, "stride": stride,
             "half": half, "max_gap_factor": max_gap_factor}
    return (np.concatenate(X), np.concatenate(Y), np.array(owner),
            list(STATE_TARGET_NAMES), stats)
