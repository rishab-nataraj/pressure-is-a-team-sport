"""Cluster bootstrap over matches (plan section 11.6): resample matches with replacement,
refit with the chosen penalties, keep every parameter draw."""
from __future__ import annotations

import logging
import time

import numpy as np

from .model import Design, Fit, fit_cfg

log = logging.getLogger("superr.bootstrap")


def match_weights(D: Design, rng: np.random.Generator) -> np.ndarray:
    draw = rng.integers(0, D.nM, D.nM)
    cnt = np.bincount(draw, minlength=D.nM).astype(float)
    return cnt[D.mcode]


def cluster_bootstrap(D: Design, cfg: dict, lams: dict, init: Fit, reps: int, seed: int = 1,
                      tol: float = 1e-8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((reps, n)) for k, n in (("c", D.nP), ("beta", D.nP), ("tau", D.nT), ("q", D.nT),
                                                ("mu", 2), ("exposure", D.nP))}
    t0 = time.time()
    for r in range(reps):
        sw = match_weights(D, rng)
        F = fit_cfg(D, cfg, lams, sample_weight=sw, init=init, tol=tol)
        out["c"][r], out["beta"][r], out["tau"][r], out["q"][r], out["mu"][r] = F.c, F.beta, F.tau, F.q, F.mu
        out["exposure"][r] = np.bincount(D.pr, D.w0 * sw * D.h ** 2, minlength=D.nP)
        if (r + 1) % 50 == 0:
            log.info("bootstrap %d/%d (%.0fs)", r + 1, reps, time.time() - t0)
    return out
