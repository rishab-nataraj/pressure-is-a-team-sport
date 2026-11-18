"""The P/T pressure model (plan sections 5-8).

    y_s = mu_r + beta_i - q_o + h_s * (c_i + tau_a - tau_o) + eps_s,     Var(eps_s) = sigma^2 / n_s

    i = player-role, r = role, a = own team, o = opponent
    beta_i : player's usual level at ordinary pressure (baseline)
    q_o    : opponent's general strength at any pressure (nuisance)
    c_i    : change in performance per unit of pressure, vs own baseline        -> P
    tau_t  : team clutch, shared by everyone in that shirt                       -> T

Loss  sum_s w_s (y_s - yhat_s)^2 + lam_b sum beta^2 + lam_c sum c^2 + lam_q sum q^2 + lam_t sum tau^2
fitted by alternating closed-form block updates (hard EM / alternating maximisation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

log = logging.getLogger("superr.model")


# ------------------------------------------------------------------------------------- design
@dataclass
class Design:
    st: pd.DataFrame            # stints used (n_balls > 0), with y, pi, h, omega columns added
    n: np.ndarray
    y: np.ndarray
    h: np.ndarray
    w0: np.ndarray              # player-step weights  n * omega
    v0: np.ndarray              # team-step weights    n / 2 (or w0 when split_weights is off)
    pr: np.ndarray              # player-role index per stint
    role: np.ndarray            # 0 bat, 1 bowl
    a: np.ndarray               # own-team index
    o: np.ndarray               # opponent index
    mcode: np.ndarray           # match index (for grouped CV / cluster bootstrap)
    keys: pd.DataFrame          # player_id, role per player-role index
    teams: pd.Index
    params: dict = field(default_factory=dict)

    @property
    def nP(self) -> int:
        return len(self.keys)

    @property
    def nT(self) -> int:
        return len(self.teams)

    @property
    def nM(self) -> int:
        return int(self.mcode.max()) + 1


def prepare(st: pd.DataFrame, k: float = 1.0, theta: float | str = 1.0, h_max: float | None = 3.0,
            half_life: float | None = 540, split_weights: bool = True, li_col: str = "mean_li",
            ref_date=None) -> Design:
    """Turn the stint table into model inputs (plan section 5)."""
    st = st[st["n_balls"] > 0].reset_index(drop=True).copy()
    n = st["n_balls"].to_numpy(float)
    y = st["sum_wpa_li"].to_numpy(float) / n
    li = st[li_col].to_numpy(float)
    pi = li * (1 + k * st["stakes"].to_numpy(float))
    if isinstance(theta, str):
        if not theta.startswith("p"):
            raise ValueError(f"theta must be a number or 'pNN', got {theta}")
        theta_v = float(np.percentile(pi, float(theta[1:])))
    else:
        theta_v = float(theta)
    hm = np.inf if h_max is None else float(h_max)
    h = np.clip(pi - theta_v, 0, hm)
    dates = pd.to_datetime(st["match_date"])
    ref = pd.Timestamp(ref_date) if ref_date is not None else dates.max()
    age = (ref - dates).dt.days.to_numpy(float)
    omega = np.ones_like(n) if half_life is None else 0.5 ** (age / float(half_life))
    w0 = n * omega
    v0 = n / 2 if split_weights else w0.copy()

    pr = st.groupby(["player_id", "role"], sort=False).ngroup().to_numpy()
    keys = (st[["player_id", "role"]].assign(_pr=pr).drop_duplicates("_pr").sort_values("_pr")
            .drop(columns="_pr").reset_index(drop=True))
    role = (st["role"] == "bowl").to_numpy(int)
    teams = pd.Index(pd.unique(st[["team_canon", "opp_canon"]].to_numpy().ravel()))
    a, o = teams.get_indexer(st["team_canon"]), teams.get_indexer(st["opp_canon"])
    mcode = pd.factorize(st["match_id"])[0]
    st["y"], st["pi"], st["h"], st["omega"], st["age_days"] = y, pi, h, omega, age
    return Design(st=st, n=n, y=y, h=h, w0=w0, v0=v0, pr=pr, role=role, a=a, o=o, mcode=mcode, keys=keys,
                  teams=teams, params=dict(k=k, theta=theta, theta_value=theta_v, h_max=h_max,
                                           half_life=half_life, split_weights=split_weights, li_col=li_col,
                                           ref_date=str(ref.date())))


def prepare_from_cfg(st: pd.DataFrame, cfg: dict, **over) -> Design:
    pc = cfg["pressure"]
    kw = dict(k=pc.get("k", 1.0), theta=pc.get("theta", 1.0), h_max=pc.get("h_max", 3.0),
              half_life=pc.get("half_life_days", 540), split_weights=cfg["model"].get("split_weights", True),
              li_col=pc.get("li_column", "mean_li"))
    kw.update(over)
    return prepare(st, **kw)


# ---------------------------------------------------------------------------------------- fit
@dataclass
class Fit:
    mu: np.ndarray
    beta: np.ndarray
    c: np.ndarray
    q: np.ndarray
    tau: np.ndarray
    iters: int
    converged: bool
    max_change: float
    lams: dict


def fit(D: Design, lam_b: float, lam_c: float, lam_q: float, lam_t: float, sample_weight=None,
        use_tau: bool = True, use_q: bool = True, damp: float = 1.0, iters: int = 3000, tol: float = 1e-9,
        seed: int = 0, init: Fit | None = None) -> Fit:
    """Alternating closed-form updates (plan section 7): mu, beta, c, q, tau in that order."""
    y, h, pr, role, a, o = D.y, D.h, D.pr, D.role, D.a, D.o
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
    w = D.w0 * sw
    v = D.v0 * sw
    nP, nT = D.nP, D.nT

    W = np.bincount(pr, w, minlength=nP)                 # baseline exposure
    E = np.bincount(pr, w * h * h, minlength=nP)         # pressure exposure
    Vq = np.bincount(o, v, minlength=nT)
    Wr = np.bincount(role, w, minlength=2)
    A = lam_t * np.eye(nT)                               # T-step matrix (fixed within a fit)
    vh2 = v * h * h
    for i, j, s in ((a, a, 1.0), (o, o, 1.0), (a, o, -1.0), (o, a, -1.0)):
        np.add.at(A, (i, j), s * vh2)

    rng = np.random.default_rng(seed)
    if init is not None:
        mu, beta, c, q, tau = (init.mu.copy(), init.beta.copy(), init.c.copy(), init.q.copy(), init.tau.copy())
    else:
        mu, beta, q = np.zeros(2), np.zeros(nP), np.zeros(nT)
        c, tau = rng.normal(0, 1e-3, nP), rng.normal(0, 1e-3, nT)
    if not use_tau:
        tau = np.zeros(nT)
    if not use_q:
        q = np.zeros(nT)

    def mix(new, old):
        return old + damp * (new - old)

    change = np.inf
    it = 0
    for it in range(1, iters + 1):
        old = np.concatenate([mu, beta, c, q, tau])
        d = c[pr] + tau[a] - tau[o]
        mu = np.bincount(role, w * (y - beta[pr] + q[o] - h * d), minlength=2) / np.maximum(Wr, 1e-12)
        beta = mix(np.bincount(pr, w * (y - mu[role] + q[o] - h * d), minlength=nP) / (W + lam_b), beta)
        res = y - mu[role] - beta[pr] + q[o] - h * (tau[a] - tau[o])
        c = mix(np.bincount(pr, w * h * res, minlength=nP) / (E + lam_c), c)
        d = c[pr] + tau[a] - tau[o]
        if use_q:
            q = mix(-np.bincount(o, v * (y - mu[role] - beta[pr] - h * d), minlength=nT) / (Vq + lam_q), q)
        if use_tau:
            r = y - mu[role] - beta[pr] + q[o] - h * c[pr]
            b = np.bincount(a, v * h * r, minlength=nT) - np.bincount(o, v * h * r, minlength=nT)
            tau = mix(np.linalg.solve(A, b), tau)
        change = float(np.max(np.abs(np.concatenate([mu, beta, c, q, tau]) - old)))
        if change < tol:
            break
    conv = change < tol
    if not conv:
        log.warning("fit did not converge in %d iterations (max change %.2e)", iters, change)
    return Fit(mu=mu, beta=beta, c=c, q=q, tau=tau, iters=it, converged=conv, max_change=change,
               lams=dict(beta=lam_b, c=lam_c, q=lam_q, tau=lam_t))


def fit_cfg(D: Design, cfg: dict, lams: dict, **kw) -> Fit:
    mc = cfg["model"]
    args = dict(use_tau=mc.get("use_tau", True), use_q=mc.get("use_q", True), damp=mc.get("damp", 1.0),
                iters=mc.get("iters", 3000), tol=mc.get("tol", 1e-9), seed=mc.get("seed", 0))
    args.update(kw)
    return fit(D, lams["beta"], lams["c"], lams["q"], lams["tau"], **args)


def predict(D: Design, F: Fit, idx=None) -> np.ndarray:
    sl = slice(None) if idx is None else idx
    pr, role, a, o, h = D.pr[sl], D.role[sl], D.a[sl], D.o[sl], D.h[sl]
    return F.mu[role] + F.beta[pr] - F.q[o] + h * (F.c[pr] + F.tau[a] - F.tau[o])


def player_table(D: Design, F: Fit, sample_weight=None) -> pd.DataFrame:
    sw = np.ones_like(D.y) if sample_weight is None else sample_weight
    w = D.w0 * sw
    nP = D.nP
    E = np.bincount(D.pr, w * D.h ** 2, minlength=nP)
    W = np.bincount(D.pr, w, minlength=nP)
    return D.keys.assign(beta=F.beta, c=F.c, exposure=E, baseline_weight=W,
                         pressure_stints=np.bincount(D.pr, (D.h > 0).astype(float), minlength=nP),
                         shrink=E / (E + F.lams["c"]))


def team_table(D: Design, F: Fit) -> pd.DataFrame:
    vh2 = D.v0 * D.h ** 2
    expo = np.bincount(D.a, vh2, minlength=D.nT) + np.bincount(D.o, vh2, minlength=D.nT)
    n_st = np.bincount(D.a, minlength=D.nT)
    return pd.DataFrame({"team_canon": D.teams, "tau": F.tau, "q": F.q, "pressure_exposure": expo,
                         "own_stints": n_st})


def to_score(x: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """0-100 scale; 50 = no change under pressure (plan section 10)."""
    x = np.asarray(x, float)
    ref = x if eligible is None else x[np.asarray(eligible, bool)]
    sd = ref.std()
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(x, 50.0)
    return 100 * norm.cdf(x / sd)


# --------------------------------------------------------------------------- one-shot solver
def ridge_one_shot(D: Design, lam_b: float, lam_c: float, lam_q: float, lam_t: float,
                   use_tau: bool = True, use_q: bool = True) -> Fit:
    """Single sparse ridge solve of the same loss (valid when split_weights = False, i.e. w = v).

    Columns: [mu_bat, mu_bowl | beta (nP) | c (nP) | q (nT) | tau (nT)]
    """
    ns, nP, nT = len(D.y), D.nP, D.nT
    rows = np.arange(ns)
    blocks = [
        sparse.csr_matrix((np.ones(ns), (rows, D.role)), shape=(ns, 2)),
        sparse.csr_matrix((np.ones(ns), (rows, D.pr)), shape=(ns, nP)),
        sparse.csr_matrix((D.h, (rows, D.pr)), shape=(ns, nP)),
    ]
    pen = [np.zeros(2), np.full(nP, lam_b), np.full(nP, lam_c)]
    if use_q:
        blocks.append(sparse.csr_matrix((-np.ones(ns), (rows, D.o)), shape=(ns, nT)))
        pen.append(np.full(nT, lam_q))
    if use_tau:
        T = sparse.csr_matrix((D.h, (rows, D.a)), shape=(ns, nT)) - sparse.csr_matrix((D.h, (rows, D.o)), shape=(ns, nT))
        blocks.append(T)
        pen.append(np.full(nT, lam_t))
    X = sparse.hstack(blocks).tocsr()
    Wd = sparse.diags(D.w0)
    lhs = (X.T @ Wd @ X + sparse.diags(np.concatenate(pen))).tocsc()
    rhs = X.T @ (D.w0 * D.y)
    sol = spsolve(lhs, rhs)
    i = 0
    mu = sol[i:i + 2]; i += 2
    beta = sol[i:i + nP]; i += nP
    c = sol[i:i + nP]; i += nP
    q = np.zeros(nT)
    tau = np.zeros(nT)
    if use_q:
        q = sol[i:i + nT]; i += nT
    if use_tau:
        tau = sol[i:i + nT]; i += nT
    return Fit(mu=mu, beta=beta, c=c, q=q, tau=tau, iters=1, converged=True, max_change=0.0,
               lams=dict(beta=lam_b, c=lam_c, q=lam_q, tau=lam_t))


# ----------------------------------------------------------------- reference-signature wrapper
def fit_pt(st: pd.DataFrame, lam_b, lam_c, lam_q, lam_t, k=1.0, theta=1.0, h_max=3.0, half_life=540,
           split_weights=True, damp=1.0, iters=1000, tol=1e-9, seed=0):
    """Same signature and return values as the plan's reference implementation (section 8)."""
    D = prepare(st, k=k, theta=theta, h_max=h_max, half_life=half_life, split_weights=split_weights)
    F = fit(D, lam_b, lam_c, lam_q, lam_t, damp=damp, iters=iters, tol=tol, seed=seed)
    players = player_table(D, F)[["player_id", "role", "beta", "c", "exposure", "pressure_stints"]]
    teams = team_table(D, F)[["team_canon", "tau", "q"]]
    return players, teams, F.mu, F.iters
