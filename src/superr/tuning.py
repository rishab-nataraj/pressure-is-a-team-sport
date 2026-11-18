"""Hyperparameters (plan section 9): the four ridge penalties are chosen by 5-fold CV grouped by
match_id, minimising weighted squared error on held-out stints. k, theta, h_max and H are fixed
and only reported as sensitivities (tuning them to make a list look right would be p-hacking)."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from .model import Design, Fit, fit_cfg, predict

log = logging.getLogger("superr.tuning")

ORDER = ["c", "tau", "beta", "q"]


def match_folds(D: Design, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D.nM)
    return (perm % k)[D.mcode]


def cv_loss(D: Design, cfg: dict, lams: dict, folds: np.ndarray, warm: dict | None = None,
            **fit_kw) -> tuple[float, dict]:
    """Mean held-out weighted squared error; warm-starts each fold from ``warm``."""
    K = int(folds.max()) + 1
    wt = D.n if cfg["model"].get("cv_weight", "n") == "n" else D.w0
    num = den = 0.0
    fits = {}
    tol = fit_kw.pop("tol", 1e-8)
    for f in range(K):
        train = (folds != f).astype(float)
        test = folds == f
        F = fit_cfg(D, cfg, lams, sample_weight=train, init=(warm or {}).get(f), tol=tol, **fit_kw)
        fits[f] = F
        e = D.y[test] - predict(D, F, test)
        num += float((wt[test] * e * e).sum())
        den += float(wt[test].sum())
    return num / den, fits


def tune(D: Design, cfg: dict) -> tuple[dict, pd.DataFrame]:
    mc = cfg["model"]
    grid = [float(g) for g in mc.get("lambda_grid", [30, 100, 300, 1000, 3000, 10000, 30000])]
    folds = match_folds(D, int(mc.get("cv_folds", 5)), int(mc.get("seed", 0)))
    cur = {k: float(v) for k, v in mc.get("lambda_default", {}).items()}
    use = {"tau": mc.get("use_tau", True), "q": mc.get("use_q", True)}
    params = [p for p in ORDER if use.get(p, True)]
    cache: dict[tuple, float] = {}
    warm: dict | None = None
    rows = []

    def evaluate(l: dict) -> float:
        nonlocal warm
        key = tuple(l[p] for p in ORDER)
        if key not in cache:
            loss, fits = cv_loss(D, cfg, l, folds, warm)
            warm = fits
            cache[key] = loss
            rows.append({**{f"lam_{p}": l[p] for p in ORDER}, "cv_loss": loss})
        return cache[key]

    if mc.get("cv_search", "coordinate") == "full":
        for combo in itertools.product(*[grid if p in params else [cur[p]] for p in ORDER]):
            evaluate(dict(zip(ORDER, combo)))
        best = min(cache, key=cache.get)
        best_l = dict(zip(ORDER, best))
    else:
        best_l = dict(cur)
        best_loss = evaluate(best_l)
        for sweep in range(int(mc.get("cv_sweeps", 3))):
            changed = False
            for p in params:
                for g in grid:
                    trial = dict(best_l, **{p: g})
                    loss = evaluate(trial)
                    if loss < best_loss - 1e-15:
                        best_loss, best_l, changed = loss, trial, True
            log.info("CV sweep %d: best %s loss %.6e", sweep + 1, best_l, best_loss)
            if not changed:
                break
    table = pd.DataFrame(rows).sort_values("cv_loss").reset_index(drop=True)
    # reference points: baseline-only model (no pressure terms) and the chosen model
    base = cv_loss(D, cfg, dict(best_l, c=1e12, tau=1e12), folds)[0]
    table.attrs["baseline_only_loss"] = base
    table.attrs["best_loss"] = cache[tuple(best_l[p] for p in ORDER)]
    edges = {p: (best_l[p] in (grid[0], grid[-1])) for p in params}
    table.attrs["at_grid_edge"] = edges
    log.info("chosen lambdas %s; CV loss %.6e vs baseline-only %.6e; at grid edge: %s",
             best_l, table.attrs["best_loss"], base, edges)
    return best_l, table
