"""Empirical fallbacks for Jev calls A and B, trained on the state table (no identities used).

wp model: one HistGradientBoostingClassifier per innings, with monotone constraints that encode
cricket sense (more runs needed / fewer balls / more wickets down never helps the batting side).
It is *cross-fitted* by match: every match's real and hypothetical states are scored by the fold
model that never saw that match, so a state's wp never "knows" its own result.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..cricket import OUTCOMES

log = logging.getLogger("superr.empirical")


def _features(S: pd.DataFrame, innings: int, use_venue_par: bool = False) -> tuple[np.ndarray, list[int], list[str]]:
    """Situation features with monotone constraints (+1 increasing, -1 decreasing, 0 free).

    Only "core" situation features are used: batter/partnership fields and the venue par add
    variance without improving held-out log-loss on this data (venue par effectively identifies
    a venue, which lets the model memorise a handful of matches). use_venue_par turns it back on.
    """
    bl = S["balls_left"].to_numpy(float)
    fmt = S["format"].astype(str).to_numpy()
    spec = {
        "wickets": (S["wickets"].to_numpy(float), -1),
        "balls_left": (bl, +1),
        "fmt_ipl": ((fmt == "IPL").astype(float), 0),
        "fmt_t20i": ((fmt == "T20I").astype(float), 0),
        "impact_rule": (S["impact_rule"].to_numpy(float), 0),
        "free_hit": (S["free_hit"].to_numpy(float), +1),
    }
    if innings == 1:
        spec["runs"] = (S["runs"].to_numpy(float), +1)
        spec["max_balls"] = (S["max_balls"].to_numpy(float), 0)
        if use_venue_par:
            spec["venue_par"] = (S["venue_par"].to_numpy(float), -1)
    else:
        need = S["runs_needed"].to_numpy(float)
        spec["runs_needed"] = (need, -1)
        spec["rrr"] = (np.where(bl > 0, 6 * need / np.maximum(bl, 1), 36.0), -1)
        if use_venue_par:
            spec["venue_par"] = (S["venue_par"].to_numpy(float), +1)
    names = list(spec)
    X = np.column_stack([spec[k][0] for k in names])
    mono = [spec[k][1] for k in names]
    return X, mono, names


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class EmpiricalWP:
    """Cross-fitted win-probability model. ``wp(S, match_id)`` scores states of known matches.

    * One monotone HistGradientBoostingClassifier per innings and fold; a match's states are
      always scored by the fold model that never saw that match.
    * Innings 2 is trained on the match result and Platt-calibrated on its out-of-fold scores.
    * Innings 1 is trained on *soft* labels: the (calibrated, cross-fitted) innings-2 win
      probability at the start of the chase that actually followed, V = 1 - wp2(target = total+1).
      The two halves of the wp function then agree at the innings break (otherwise the last ball
      of innings 1 carries a spurious jump in WPA and leverage), and V is lower-variance than the
      raw result. empirical.inn1_soft_labels = false reverts to raw results (+ Platt).
    """

    def __init__(self, cfg: dict, train: pd.DataFrame):
        ecfg = cfg["empirical"]
        self.n_folds = int(ecfg.get("n_folds", 5))
        self.seed = int(ecfg.get("seed", 0))
        self.gb = ecfg.get("gbm", {})
        self.use_par = bool(ecfg.get("use_venue_par", False))
        self.platt_on = bool(ecfg.get("platt", True))
        rng = np.random.default_rng(self.seed)
        mids = np.array(sorted(train["match_id"].unique()))
        perm = rng.permutation(len(mids))
        self.fold_of = pd.Series(perm % self.n_folds, index=mids)
        self.models: dict[tuple[int, int], HistGradientBoostingClassifier] = {}
        self.platt: dict[int, tuple[float, float]] = {1: (1.0, 0.0), 2: (1.0, 0.0)}
        lab = train["bat_won"].to_numpy()
        inn = train["innings"].to_numpy()

        # innings 2: binary match result (ties carry no binary label)
        sel = np.isin(lab, (0.0, 1.0)) & (inn == 2)
        T2 = train[sel]
        self._fit_innings(T2, 2, lab[sel], np.ones(sel.sum()))
        if self.platt_on:
            self._fit_platt(T2, 2, lab[sel])

        # innings 1
        sel1 = inn == 1
        T1 = train[sel1]
        if ecfg.get("inn1_soft_labels", True):
            from ..cricket import inn2_start
            last = T1.groupby("match_id").tail(1)
            end = last.copy()
            end["runs"] = last["runs"] + last["ball_total"]
            v = 1.0 - self.wp(inn2_start(end), last["match_id"].to_numpy())
            V = T1["match_id"].map(pd.Series(v, index=last["match_id"].to_numpy())).to_numpy()
            n = len(T1)
            T1d = pd.concat([T1, T1], ignore_index=True)
            self._fit_innings(T1d, 1, np.r_[np.ones(n), np.zeros(n)], np.r_[V, 1.0 - V])
        else:
            keep = np.isin(lab[sel1], (0.0, 1.0))
            self._fit_innings(T1[keep], 1, lab[sel1][keep], np.ones(keep.sum()))
            if self.platt_on:
                self._fit_platt(T1[keep], 1, lab[sel1][keep])

    def _fit_innings(self, T: pd.DataFrame, inn: int, y: np.ndarray, w: np.ndarray) -> None:
        gb = self.gb
        X, mono, _ = _features(T, inn, self.use_par)
        f = self.fold_of.reindex(T["match_id"].to_numpy()).to_numpy()
        for k in range(self.n_folds):
            tr = f != k
            m = HistGradientBoostingClassifier(
                max_iter=gb.get("max_iter", 300), learning_rate=gb.get("learning_rate", 0.05),
                max_leaf_nodes=gb.get("max_leaf_nodes", 15), min_samples_leaf=gb.get("min_samples_leaf", 2000),
                l2_regularization=gb.get("l2_regularization", 1.0), monotonic_cst=mono,
                early_stopping=False, random_state=self.seed + k)
            m.fit(X[tr], y[tr].astype(int), sample_weight=w[tr])
            self.models[(inn, k)] = m
        log.info("empirical wp: innings %d models trained on %d rows", inn, len(T))

    def _fit_platt(self, T: pd.DataFrame, inn: int, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        self.platt[inn] = (1.0, 0.0)
        raw = self.wp(T, T["match_id"].to_numpy(), calibrate=False)
        lr = LogisticRegression(C=1e6).fit(_logit(raw)[:, None], y.astype(int))
        self.platt[inn] = (float(lr.coef_.ravel()[0]), float(lr.intercept_[0]))
        log.info("empirical wp: innings %d Platt a=%.3f b=%.3f", inn, *self.platt[inn])

    def wp(self, S: pd.DataFrame, match_id: np.ndarray, calibrate: bool = True) -> np.ndarray:
        out = np.full(len(S), np.nan)
        folds = self.fold_of.reindex(np.asarray(match_id)).to_numpy()
        if np.isnan(folds.astype(float)).any():
            raise ValueError("wp requested for a match the model has no fold for")
        inn = S["innings"].to_numpy()
        for i in (1, 2):
            for k in range(self.n_folds):
                if (i, k) not in self.models:
                    continue
                sel = (inn == i) & (folds == k)
                if sel.any():
                    X, _, _ = _features(S[sel], i, self.use_par)
                    out[sel] = self.models[(i, k)].predict_proba(X)[:, 1]
            if calibrate and self.platt[i] != (1.0, 0.0):
                sel = inn == i
                a, b = self.platt[i]
                out[sel] = 1 / (1 + np.exp(-(a * _logit(out[sel]) + b)))
        return np.clip(out, 1e-4, 1 - 1e-4)


def _cells(S: pd.DataFrame) -> pd.DataFrame:
    legal = S["legal_balls"].to_numpy()
    phase = np.select([legal < 36, legal < 90], ["powerplay", "middle"], default="death")
    w = S["wickets"].to_numpy()
    band = np.select([w <= 2, w <= 5], ["0-2", "3-5"], default="6-9")
    return pd.DataFrame({"format": S["format"].astype(str).to_numpy(), "phase": phase, "wband": band,
                         "innings": S["innings"].to_numpy()})


class EmpiricalPNext:
    """Next-ball outcome frequencies by format x phase x wickets band x innings (spec section 3)."""

    def __init__(self, cfg: dict, train: pd.DataFrame):
        a = float(cfg["empirical"].get("p_next_smoothing", 1.0))
        cells = _cells(train)
        cells["outcome"] = train["outcome"].to_numpy()
        ct = pd.crosstab([cells["format"], cells["phase"], cells["wband"], cells["innings"]], cells["outcome"])
        ct = ct.reindex(columns=OUTCOMES, fill_value=0) + a
        self.table = ct.div(ct.sum(axis=1), axis=0)
        glob = cells["outcome"].value_counts().reindex(OUTCOMES, fill_value=0) + a
        self.global_ = (glob / glob.sum()).to_numpy()

    def p_next(self, S: pd.DataFrame, match_id=None) -> np.ndarray:
        c = _cells(S)
        idx = pd.MultiIndex.from_frame(c[["format", "phase", "wband", "innings"]])
        P = self.table.reindex(idx).to_numpy(dtype=float, copy=True)
        miss = np.isnan(P).any(axis=1)
        P[miss] = self.global_
        return P
