"""Model-side validation (plan section 11)."""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import io
from .config import Paths
from .model import (Design, Fit, fit_cfg, player_table, predict, prepare_from_cfg, ridge_one_shot,
                    team_table)
from .rank import eligibility, player_evidence, rank_lists, score_players

log = logging.getLogger("superr.validate_model")


def _rank_from_fit(D: Design, F: Fit, cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    players = player_table(D, F)
    ev = player_evidence(D.st, players, cfg).merge(pl[["player_id", "player_name"]], on="player_id", how="left")
    el = eligibility(ev, cfg, set(pl["player_id"]))
    return score_players(el)


def _eligible_mask(D: Design, F: Fit, cfg: dict, pl: pd.DataFrame) -> np.ndarray:
    return _rank_from_fit(D, F, cfg, pl)["eligible"].to_numpy()


# ------------------------------------------------------------------------- 11.1 synthetic recovery
def prior_sds(sigma2: float, lams: dict) -> tuple[float, float]:
    """Spread of true effects implied by the CV-chosen penalties.

    With Var(eps) = sigma^2 / n and player weights w = n, the ridge penalty lam_c is a Gaussian
    prior with sd sqrt(sigma^2 / lam_c); team steps use v = n / 2, so sd_tau = sqrt(sigma^2 / (2 lam_tau)).
    """
    return float(np.sqrt(sigma2 / lams["c"])), float(np.sqrt(sigma2 / (2 * lams["tau"])))


def synthetic_recovery(D: Design, F: Fit, cfg: dict, lams: dict, eligible: np.ndarray, sigma2: float) -> pd.DataFrame:
    """Plant known c and tau on the real stint structure, add noise at the observed sigma, refit.

    Planted effect sizes are multiples of the prior sd implied by the CV-chosen penalties (the
    largest spread of clutch the data allow); the fitted c are shrunk and far narrower than that.
    """
    vcfg = cfg["validation"]
    rng = np.random.default_rng(123)
    sd_c, sd_t = prior_sds(sigma2, lams)
    rows = []
    for mult in vcfg.get("synthetic_effect_multipliers", [1.0, 2.0, 4.0]):
        for rep in range(int(vcfg.get("synthetic_reps", 3))):
            c_true = rng.normal(0, mult * sd_c, D.nP)
            t_true = rng.normal(0, mult * sd_t, D.nT)
            mean = F.mu[D.role] + F.beta[D.pr] - F.q[D.o] + D.h * (c_true[D.pr] + t_true[D.a] - t_true[D.o])
            y_syn = mean + rng.normal(0, 1, len(D.y)) * np.sqrt(sigma2 / D.n)
            Ds = Design(**{**D.__dict__, "y": y_syn})
            Fs = fit_cfg(Ds, cfg, lams, tol=1e-8)
            rows.append({"effect_multiplier": mult, "rep": rep, "planted_sd_c": mult * sd_c, "planted_sd_tau": mult * sd_t,
                         "corr_c_eligible": float(np.corrcoef(c_true[eligible], Fs.c[eligible])[0, 1]) if eligible.sum() > 2 else np.nan,
                         "corr_c_all": float(np.corrcoef(c_true, Fs.c)[0, 1]),
                         "spearman_c_eligible": float(spearmanr(c_true[eligible], Fs.c[eligible])[0]) if eligible.sum() > 2 else np.nan,
                         "corr_tau": float(np.corrcoef(t_true, Fs.tau)[0, 1]) if cfg["model"].get("use_tau", True) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ 11.2 mover graph
def mover_graph(D: Design, players: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    st = D.st
    G = nx.Graph()
    for (pid, role), team in st.groupby(["player_id", "role"])["team_canon"].unique().items():
        for t in team:
            G.add_edge(("p", pid, role), ("t", t))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    team_comp = [sorted(n[1] for n in c if n[0] == "t") for c in comps]
    nteams = st.groupby(["player_id", "role"])["team_canon"].nunique()
    E = players.set_index(["player_id", "role"])["exposure"]
    movers = nteams[nteams >= 2].index
    share = float(E.reindex(movers).sum() / E.sum()) if E.sum() > 0 else np.nan
    per_team = []
    pt = st.groupby("player_id")["team_canon"].agg(lambda s: set(s))
    for t in sorted(st["team_canon"].unique()):
        inside = pt[pt.apply(lambda s: t in s)]
        mv = inside[inside.apply(len) > 1]
        others = sorted(set().union(*mv.tolist()) - {t}) if len(mv) else []
        per_team.append({"team_canon": t, "players": len(inside), "movers": len(mv),
                         "linked_teams": "; ".join(others[:12]) + (" ..." if len(others) > 12 else "")})
    summ = {"components": len(comps), "component_team_lists": team_comp[:10],
            "largest_component_teams": len(team_comp[0]) if team_comp else 0,
            "pressure_exposure_share_from_movers": share,
            "player_roles_with_2plus_teams": int(len(movers))}
    return summ, pd.DataFrame(per_team)


# --------------------------------------------------------------------------- 11.3 start stability
def start_stability(D: Design, cfg: dict, lams: dict, seeds: int) -> dict:
    fits = [fit_cfg(D, cfg, lams, seed=s, init=None) for s in range(seeds)]
    P = np.vstack([np.concatenate([f.beta, f.c, f.q, f.tau]) for f in fits])
    dev = np.abs(P - P[0]).max()
    scale = np.abs(P[0]).max()
    return {"seeds": seeds, "max_abs_param_diff": float(dev), "max_param_scale": float(scale),
            "max_c_diff": float(max(np.abs(f.c - fits[0].c).max() for f in fits)),
            "all_converged": bool(all(f.converged for f in fits)),
            "iters_range": [int(min(f.iters for f in fits)), int(max(f.iters for f in fits))]}


# ----------------------------------------------------------------------------- 11.4 one-shot check
def one_shot_check(st: pd.DataFrame, cfg: dict, lams: dict) -> dict:
    D = prepare_from_cfg(st, cfg, split_weights=False)
    F = fit_cfg(D, cfg, lams, tol=1e-12, iters=50000)
    R = ridge_one_shot(D, lams["beta"], lams["c"], lams["q"], lams["tau"],
                       use_tau=cfg["model"].get("use_tau", True), use_q=cfg["model"].get("use_q", True))
    return {"loop_iters": F.iters, "max_diff_beta": float(np.abs(F.beta - R.beta).max()),
            "max_diff_c": float(np.abs(F.c - R.c).max()), "max_diff_q": float(np.abs(F.q - R.q).max()),
            "max_diff_tau": float(np.abs(F.tau - R.tau).max()), "max_diff_mu": float(np.abs(F.mu - R.mu).max()),
            "scale_c": float(np.abs(R.c).max())}


# ----------------------------------------------------------------------------- 11.5 out-of-time
def out_of_time(st: pd.DataFrame, cfg: dict, lams: dict, pl: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cut = pd.Timestamp(cfg["validation"].get("oot_train_end", "2024-12-31"))
    dates = pd.to_datetime(st["match_date"])
    tr, te = st[dates <= cut], st[dates > cut]
    Dtr = prepare_from_cfg(tr, cfg, ref_date=cut)
    Ftr = fit_cfg(Dtr, cfg, lams)
    rk = _rank_from_fit(Dtr, Ftr, cfg, pl)
    # test period: baseline-only fit (no pressure terms) -> residuals -> per-player pressure slope
    Dte = prepare_from_cfg(te, cfg, half_life=None)
    Fte = fit_cfg(Dte, cfg, dict(lams, c=1e12, tau=1e12))
    r = Dte.y - predict(Dte, Fte)
    num = np.bincount(Dte.pr, Dte.n * Dte.h * r, minlength=Dte.nP)
    den = np.bincount(Dte.pr, Dte.n * Dte.h ** 2, minlength=Dte.nP)
    test = Dte.keys.assign(test_slope=np.where(den > 0, num / np.maximum(den, 1e-12), np.nan), test_exposure=den)
    j = rk[["player_id", "role", "c", "eligible", "exposure"]].merge(test, on=["player_id", "role"])
    j = j[j["test_exposure"] >= cfg["validation"].get("oot_min_test_exposure", 5.0)]
    out = {"train_end": str(cut.date()), "train_stints": int(len(tr)), "test_stints": int(len(te))}
    for role in ("bat", "bowl"):
        for name, sub in (("all", j[j["role"] == role]), ("train_eligible", j[(j["role"] == role) & j["eligible"]])):
            if len(sub) > 5:
                rho, p = spearmanr(sub["c"], sub["test_slope"])
                out[f"{role}_{name}"] = {"n": int(len(sub)), "spearman": float(rho), "p_value": float(p)}
    return out, j


# ------------------------------------------------------------------------------- 11.7 shuffle test
N_BANDS = [0, 3, 6, 7, 11, 21, 36, 10_000]  # stint-length bands (balls): bowlers sit almost all in 6


def shuffle_test(D: Design, F: Fit, cfg: dict, lams: dict, eligible: np.ndarray, reps: int) -> pd.DataFrame:
    """Plan 11.7: destroy the pressure-performance link inside each player and refit.

    * permute_h   : permute h within (player, stint-length band). Banding keeps each player's
                    exposure E = sum(omega n h^2) roughly unchanged; an unbanded permutation moves
                    high-pressure values onto long stints and inflates E, which widens c by itself.
    * sign_flip   : flip the sign of each stint's baseline residual at random (keeps n, h, omega).
    If real clutch exists, the real spread of c exceeds both nulls (ratio < 1).
    """
    rng = np.random.default_rng(7)
    real_sd = float(F.c[eligible].std())
    band = np.digitize(D.n, N_BANDS)
    groups = pd.Series(np.arange(len(D.h))).groupby([D.pr, band]).indices.values()
    base = F.mu[D.role] + F.beta[D.pr] - F.q[D.o]
    rows = []
    for rep in range(reps):
        h = D.h.copy()
        for idx in groups:
            h[idx] = D.h[rng.permutation(idx)]
        Fs = fit_cfg(Design(**{**D.__dict__, "h": h}), cfg, lams, tol=1e-8)
        resid = D.y - base
        y_flip = base + resid * rng.choice([-1.0, 1.0], len(resid))
        Ff = fit_cfg(Design(**{**D.__dict__, "y": y_flip}), cfg, lams, tol=1e-8)
        rows.append({"rep": rep, "sd_c_eligible_real": real_sd,
                     "sd_c_eligible_permute_h": float(Fs.c[eligible].std()),
                     "sd_c_eligible_sign_flip": float(Ff.c[eligible].std()),
                     "ratio_real_over_permute": float(real_sd / Fs.c[eligible].std()),
                     "ratio_real_over_signflip": float(real_sd / Ff.c[eligible].std())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- 11.8 ablations
def ablation_specs(cfg: dict) -> list[tuple[str, dict, dict, str | None]]:
    """(name, prepare-overrides, fit-overrides, stint filter)."""
    pc = cfg["pressure"]
    specs = [("T off", {}, {"use_tau": False}, None),
             ("q off", {}, {"use_q": False}, None)]
    for k in (0.0, 0.5, 2.0):
        if k != pc.get("k", 1.0):
            specs.append((f"k={k:g}", {"k": k}, {}, None))
    for th in (1.5, "p75"):
        specs.append((f"theta={th}", {"theta": th}, {}, None))
    for hm in (2.0, None):
        if hm != pc.get("h_max", 3.0):
            specs.append((f"h_max={hm}", {"h_max": hm}, {}, None))
    for H in (365, 730, None):
        if H != pc.get("half_life_days", 540):
            specs.append((f"H={H}", {"half_life": H}, {}, None))
    specs.append(("entry_li", {"li_col": "entry_li"}, {}, None))
    specs.append(("emerging cups dropped", {}, {}, "drop_emerging"))
    return specs


def ablations(st: pd.DataFrame, cfg: dict, lams: dict, pl: pd.DataFrame, main_lists: dict,
              main_scored: pd.DataFrame) -> pd.DataFrame:
    top_n = int(cfg["rank"].get("top_n", 10))
    rows = []
    for name, prep_over, fit_over, filt in ablation_specs(cfg):
        s = st[~st["is_emerging"]] if filt == "drop_emerging" else st
        D = prepare_from_cfg(s, cfg, **prep_over)
        F = fit_cfg(D, cfg, lams, **fit_over)
        sc = _rank_from_fit(D, F, cfg, pl)
        lists = rank_lists(sc.assign(p_top=0.0), cfg)
        for role in ("bat", "bowl"):
            main_top = list(main_lists[f"{role}_top"]["player_id"])
            alt_top = list(lists[f"{role}_top"]["player_id"])
            m = main_scored[main_scored["role"] == role][["player_id", "c", "eligible"]]
            a = sc[sc["role"] == role][["player_id", "c", "eligible"]]
            j = m.merge(a, on="player_id", suffixes=("_main", "_alt"))
            j = j[j["eligible_main"] & j["eligible_alt"]]
            rho = float(spearmanr(j["c_main"], j["c_alt"])[0]) if len(j) > 5 else np.nan
            rows.append({"ablation": name, "role": role,
                         "top10_overlap": len(set(main_top) & set(alt_top)) / max(top_n, 1),
                         "same_rank1": bool(main_top[:1] == alt_top[:1]),
                         "spearman_c_common_eligible": rho, "n_eligible": int(a["eligible"].sum()),
                         "top10": "; ".join(lists[f"{role}_top"]["player_name"].astype(str))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- 11.9 CSK claim
def csk_claim(matches: pd.DataFrame, team_canon: pd.DataFrame, tscore: pd.DataFrame) -> pd.DataFrame:
    """Distinct man-of-the-match winners per win, by team (from matches.csv), beside T."""
    m = matches[(matches["win_team_id"].notna()) & (matches["man_of_the_match"].fillna(0) > 0)].copy()
    tc = team_canon.set_index("team_id")["team_canon"]
    m["winner"] = m["win_team_id"].map(tc)
    rows = []
    for (team, fmt), g in m.groupby(["winner", "format"]):
        wins = int(len(g))
        d = int(g["man_of_the_match"].nunique())
        top_share = float(g["man_of_the_match"].value_counts().iloc[0] / wins)
        rows.append({"team_canon": team, "format": fmt, "wins_with_mom": wins, "distinct_mom": d,
                     "distinct_per_win": d / wins, "top_player_share": top_share,
                     "seasons": int(g["season_year"].nunique())})
    out = pd.DataFrame(rows).merge(tscore[["team_canon", "T", "tau"]], on="team_canon", how="left")
    return out.sort_values(["format", "distinct_per_win"], ascending=[True, False]).reset_index(drop=True)


# -------------------------------------------------------------------------------------- run
def run(cfg: dict) -> dict:
    paths = Paths.from_cfg(cfg)
    vcfg = cfg["validation"]
    sub = "validation"
    st = io.load("stints", paths)
    if cfg["data"].get("drop_emerging"):
        st = st[~st["is_emerging"]]
    pl = io.load("players", paths)
    meta = io.load_json("fit_meta", paths, sub="fit")
    lams = meta["lambdas"]
    D = prepare_from_cfg(st, cfg)
    F = fit_cfg(D, cfg, lams)
    players = player_table(D, F)
    scored = io.load("scored_players", paths)
    eligible = scored["eligible"].to_numpy()
    res: dict = {}

    log.info("11.1 synthetic recovery")
    syn = synthetic_recovery(D, F, cfg, lams, eligible, meta["sigma2"])
    io.save_output(syn, "model_11_1_synthetic_recovery", paths, sub)
    res["synthetic_recovery"] = syn.groupby("effect_multiplier")[["corr_c_eligible", "spearman_c_eligible", "corr_tau"]].mean().round(3).to_dict("index")

    log.info("11.2 mover graph")
    res["mover_graph"], mv = mover_graph(D, players)
    io.save_output(mv, "model_11_2_movers_by_team", paths, sub)

    log.info("11.3 start stability")
    res["start_stability"] = start_stability(D, cfg, lams, int(vcfg.get("seeds", 20)))

    log.info("11.4 one-shot check")
    res["one_shot"] = one_shot_check(st, cfg, lams)

    log.info("11.5 out-of-time")
    res["out_of_time"], oot = out_of_time(st, cfg, lams, pl)
    io.save_output(oot, "model_11_5_out_of_time", paths, sub)

    log.info("11.6 bootstrap summary")
    from .rank import rank_lists as _rl
    lists = _rl(scored, cfg)
    res["bootstrap"] = {r: {"median_interval_width_top10": float((lists[f"{r}_top"]["P_hi"] - lists[f"{r}_top"]["P_lo"]).median()),
                            "mean_p_top10_of_top10": float(lists[f"{r}_top"]["p_top"].mean()),
                            "players_with_p_top_over_0.5": int((scored[(scored.role == r) & scored.eligible]["p_top"] > 0.5).sum())}
                        for r in ("bat", "bowl")}

    log.info("11.7 shuffle test")
    sh = shuffle_test(D, F, cfg, lams, eligible, int(vcfg.get("shuffle_reps", 5)))
    io.save_output(sh, "model_11_7_shuffle", paths, sub)
    res["shuffle"] = {"ratio_real_over_permute_h": float(sh["ratio_real_over_permute"].mean()),
                      "ratio_real_over_sign_flip": float(sh["ratio_real_over_signflip"].mean()),
                      "share_reps_real_wider_permute": float((sh["ratio_real_over_permute"] > 1).mean()),
                      "share_reps_real_wider_signflip": float((sh["ratio_real_over_signflip"] > 1).mean())}
    res["prior_sd"] = dict(zip(("c", "tau"), prior_sds(meta["sigma2"], lams)))

    if vcfg.get("ablations", True):
        log.info("11.8 ablations")
        ab = ablations(st, cfg, lams, pl, lists, scored)
        io.save_output(ab, "model_11_8_ablations", paths, sub)
        res["ablations_mean_overlap"] = ab.groupby("role")["top10_overlap"].mean().round(3).to_dict()

    log.info("11.9 CSK claim")
    tscore = pd.read_csv(paths.outputs / "team_scores.csv")
    csk = csk_claim(io.load("matches", paths), io.load("team_canon", paths), tscore)
    io.save_output(csk, "model_11_9_csk_claim", paths, sub)
    ipl = (csk[csk["format"] == "IPL"].dropna(subset=["T"])
           .sort_values("distinct_per_win", ascending=False).reset_index(drop=True))
    if len(ipl) > 3:
        is_csk = ipl["team_canon"] == "Chennai Super Kings"
        res["csk_claim"] = {
            "spearman_distinct_per_win_vs_T": float(spearmanr(ipl["distinct_per_win"], ipl["T"])[0]),
            "csk_rank_by_distinct_per_win": int(ipl.index[is_csk][0] + 1) if is_csk.any() else None,
            "csk_T": float(ipl.loc[is_csk, "T"].iloc[0]) if is_csk.any() else None,
            "n_ipl_teams": int(len(ipl))}
    io.save_json(res, "model_validation", paths, sub)
    log.info("model validation done")
    return res
