"""Frozen-representation probes, ported verbatim from pm-jepa/model/probes.py.

PORTED, NOT REWRITTEN, ON PURPOSE. The whole point of this repo is to put a
causal-target arm next to pm-jepa's published ridge numbers (identity +0.449,
handcrafted +0.460, best JEPA arm +0.377). A probe that differs in its ridge
grid, its standardisation, or where it cuts the validation split produces R^2
on a different scale, and the comparison silently becomes meaningless rather
than failing loudly. Everything below the module docstring is byte-identical to
the pm-jepa original; `tests/test_probe_parity.py` asserts agreement to 1e-9
against the original implementation loaded off disk. If you need to change a
hyperparameter, add a new function, do not edit these.

A JEPA cannot be evaluated by its own loss: the loss lives in an embedding space
the model controls, and a collapsed or identity encoder scores well. Probing
against EXOGENOUS ground truth is the only honest measure, which is why the
simulator exposes true latent state.

Linear (ridge) probe R^2 is the primary number. The MLP probe is reported
alongside it to separate two different claims: "the information is present"
(MLP) from "the information is linearly accessible" (ridge). Raw features
usually carry the information but not linearly.
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def _standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return [(a - mu) / sd for a in (train,) + others]


def _r2(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def ridge_probe(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, float]:
    """Closed-form ridge with lambda chosen on a held-out split of train."""
    x_tr, x_te = _standardise(x_tr, x_te)
    ymu, ysd = y_tr.mean(0, keepdims=True), y_tr.std(0, keepdims=True) + 1e-6
    y_tr_s = (y_tr - ymu) / ysd

    n_val = max(1, int(len(x_tr) * val_frac))
    xa, ya = x_tr[:-n_val], y_tr_s[:-n_val]
    xv, yv = x_tr[-n_val:], y_tr_s[-n_val:]

    xa_t = torch.from_numpy(xa).double()
    ya_t = torch.from_numpy(ya).double()
    gram = xa_t.T @ xa_t
    rhs = xa_t.T @ ya_t
    eye = torch.eye(gram.shape[0], dtype=torch.float64)

    best_lam, best_score = RIDGE_GRID[0], -np.inf
    for lam in RIDGE_GRID:
        w = torch.linalg.solve(gram + lam * eye, rhs).numpy()
        score = _r2(xv @ w, yv).mean()
        if score > best_score:
            best_lam, best_score = lam, score

    x_all = torch.from_numpy(x_tr).double()
    y_all = torch.from_numpy(y_tr_s).double()
    w = torch.linalg.solve(
        x_all.T @ x_all + best_lam * eye, x_all.T @ y_all
    ).numpy()
    pred = (x_te @ w) * ysd + ymu
    return _r2(pred, y_te), best_lam


def mlp_probe(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    hidden: int = 256, steps: int = 800, lr: float = 1e-3, seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    torch.manual_seed(seed)
    x_tr, x_te = _standardise(x_tr, x_te)
    ymu, ysd = y_tr.mean(0, keepdims=True), y_tr.std(0, keepdims=True) + 1e-6
    y_tr_s = (y_tr - ymu) / ysd

    xt = torch.from_numpy(x_tr).float().to(device)
    yt = torch.from_numpy(y_tr_s).float().to(device)
    xe = torch.from_numpy(x_te).float().to(device)

    net = nn.Sequential(
        nn.Linear(xt.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, yt.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    n, bs = xt.shape[0], 512
    for step in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=device)
        loss = nn.functional.mse_loss(net(xt[i]), yt[i])
        opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(xe).cpu().numpy() * ysd + ymu
    return _r2(pred, y_te)


def run_probes(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    names: Tuple[str, ...], device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    ridge_r2, lam = ridge_probe(x_tr, y_tr, x_te, y_te)
    mlp_r2 = mlp_probe(x_tr, y_tr, x_te, y_te, device=device)
    return {
        "ridge": {n: float(v) for n, v in zip(names, ridge_r2)},
        "mlp": {n: float(v) for n, v in zip(names, mlp_r2)},
        "ridge_lambda": lam,
        "ridge_mean": float(ridge_r2.mean()),
        "mlp_mean": float(mlp_r2.mean()),
    }
