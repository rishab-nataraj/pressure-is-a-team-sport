"""Stage 5: tune the penalties by grouped CV, fit the final model, run the cluster bootstrap."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .bootstrap import cluster_bootstrap
from .config import Paths
from .model import fit_cfg, player_table, prepare_from_cfg, team_table
from .tuning import tune

log = logging.getLogger("superr.fit")


def run(cfg: dict, skip_cv: bool = False, skip_bootstrap: bool = False) -> None:
    paths = Paths.from_cfg(cfg)
    st = io.load("stints", paths)
    if cfg["data"].get("drop_emerging"):
        st = st[~st["is_emerging"]]
    D = prepare_from_cfg(st, cfg)
    log.info("design: %d stints, %d player-roles, %d teams, %d matches; theta=%.3f; pressure stints %.1f%%",
             len(D.y), D.nP, D.nT, D.nM, D.params["theta_value"], 100 * (D.h > 0).mean())

    if skip_cv:
        lams = {k: float(v) for k, v in cfg["model"]["lambda_default"].items()}
        cv_table = pd.DataFrame()
        cv_meta = {}
    else:
        lams, cv_table = tune(D, cfg)
        cv_meta = {k: v for k, v in cv_table.attrs.items()}
        io.save_output(cv_table, "cv_grid", paths, sub="fit")

    F = fit_cfg(D, cfg, lams)
    log.info("final fit: %d iterations, converged=%s, mu=%s", F.iters, F.converged, np.round(F.mu, 6).tolist())
    players = player_table(D, F)
    teams = team_table(D, F)
    io.save(players, "fit_players", paths)
    io.save(teams, "fit_teams", paths)
    io.save(D.st, "fit_design", paths)

    meta = {"lambdas": lams, "mu": F.mu.tolist(), "iters": F.iters, "converged": F.converged,
            "design_params": D.params, "n_stints": len(D.y), "n_player_roles": D.nP, "n_teams": D.nT,
            "n_matches": D.nM, "cv": cv_meta,
            "sigma2": float(np.sum(D.n * (D.y - (F.mu[D.role] + F.beta[D.pr] - F.q[D.o] +
                                                 D.h * (F.c[D.pr] + F.tau[D.a] - F.tau[D.o]))) ** 2) / len(D.y))}
    if not skip_bootstrap:
        reps = int(cfg["model"].get("bootstrap_reps", 300))
        bs = cluster_bootstrap(D, cfg, lams, F, reps, seed=int(cfg["model"].get("seed", 0)) + 1)
        np.savez_compressed(paths.interim / "bootstrap.npz", **bs)
        meta["bootstrap_reps"] = reps
    io.save_json(meta, "fit_meta", paths, sub="fit")
