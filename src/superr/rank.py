"""Stage 6: scores on 0-100, eligibility gates, the two top-10 lists and their evidence (plan section 10).

P_i = 100 * Phi(c_i / s_c)    s_c = sd of c over eligible players in that role
T_t = 100 * Phi(tau_t / s_t)  s_t = sd of tau over teams
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import io
from .config import Paths
from .model import to_score

log = logging.getLogger("superr.rank")


def player_evidence(design: pd.DataFrame, players: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per player-role counts used by the gates and the evidence table."""
    d = design.copy()
    d["is_p"] = d["h"] > 0
    last = pd.to_datetime(d["match_date"]).max()
    months = int(cfg["eligibility"].get("recent_months", 24))
    d["recent"] = pd.to_datetime(d["match_date"]) >= last - pd.DateOffset(months=months)
    d["ny"] = d["n_balls"] * d["y"]
    g = d.groupby(["player_id", "role"])
    ev = pd.DataFrame({
        "stints": g.size(),
        "balls": g["n_balls"].sum(),
        "pressure_stints": g["is_p"].sum(),
        "pressure_balls": g.apply(lambda x: x.loc[x["is_p"], "n_balls"].sum(), include_groups=False),
        "recent_pressure_stints": g.apply(lambda x: (x["is_p"] & x["recent"]).sum(), include_groups=False),
        "y_pressure": g.apply(lambda x: x.loc[x["is_p"], "ny"].sum() / max(x.loc[x["is_p"], "n_balls"].sum(), 1),
                              include_groups=False),
        "y_ordinary": g.apply(lambda x: x.loc[~x["is_p"], "ny"].sum() / max(x.loc[~x["is_p"], "n_balls"].sum(), 1),
                              include_groups=False),
        "mean_h_pressure": g.apply(lambda x: x.loc[x["is_p"], "h"].mean() if x["is_p"].any() else 0.0,
                                   include_groups=False),
        "teams": g["team_canon"].agg(lambda s: "; ".join(sorted(set(s)))),
        "n_teams": g["team_canon"].nunique(),
        "first_date": g["match_date"].min(),
        "last_date": g["match_date"].max(),
    }).reset_index()
    return players.drop(columns=["pressure_stints"]).merge(ev, on=["player_id", "role"], how="left")


def eligibility(df: pd.DataFrame, cfg: dict, selectable: set) -> pd.DataFrame:
    e = cfg["eligibility"]
    out = df.copy()
    bat = out["role"] == "bat"
    g_bat = (out["pressure_stints"] >= e["bat"]["min_pressure_stints"]) & \
            (out["pressure_balls"] >= e["bat"]["min_pressure_balls"])
    g_bowl = out["pressure_stints"] >= e["bowl"]["min_pressure_stints"]
    out["gate_exposure"] = np.where(bat, g_bat, g_bowl)
    out["gate_recent"] = out["recent_pressure_stints"] >= e.get("min_recent_pressure_stints", 5)
    out["gate_delivers"] = (out["beta"] + out["c"] > 0) if e.get("require_delivers", True) else True
    out["gate_selectable"] = out["player_id"].isin(selectable) if e.get("require_selectable", True) else True
    out["eligible"] = out["gate_exposure"] & out["gate_recent"] & out["gate_delivers"] & out["gate_selectable"]
    # pool used for the scale s_c: exposure + recency gates (the "delivers" gate is about ranking)
    out["scale_pool"] = out["gate_exposure"] & out["gate_recent"]
    return out


def score_players(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["P"] = np.nan
    out["s_c"] = np.nan
    for r in ("bat", "bowl"):
        m = out["role"] == r
        el = out.loc[m, "eligible"].to_numpy()
        pool = el if el.sum() >= 3 else out.loc[m, "scale_pool"].to_numpy()
        c = out.loc[m, "c"].to_numpy()
        out.loc[m, "P"] = to_score(c, pool)
        out.loc[m, "s_c"] = c[pool].std() if pool.any() else np.nan
    return out


def bootstrap_summary(df: pd.DataFrame, bs: dict, cfg: dict) -> pd.DataFrame:
    """P interval and P(top N) per player-role from the cluster bootstrap draws."""
    out = df.copy()
    top_n = int(cfg["rank"].get("top_n", 10))
    lo, hi = cfg["model"].get("bootstrap_interval", [5, 95])
    C = bs["c"]                      # reps x nP, columns in design order == df order
    R = C.shape[0]
    if C.shape[1] != len(out):
        raise ValueError("bootstrap draws do not match the fitted player table; re-run stage 5")
    out["P_lo"] = np.nan
    out["P_hi"] = np.nan
    out["P_boot_median"] = np.nan
    out["p_top"] = 0.0
    for r in ("bat", "bowl"):
        m = (out["role"] == r).to_numpy()
        el = out["eligible"].to_numpy() & m
        idx_el = np.where(el)[0]
        idx_m = np.where(m)[0]
        if len(idx_el) == 0:
            continue
        sd = C[:, idx_el].std(axis=1, keepdims=True)
        Pm = 100 * norm.cdf(C[:, idx_m] / np.where(sd > 0, sd, np.nan))
        out.loc[m, "P_lo"] = np.nanpercentile(Pm, lo, axis=0)
        out.loc[m, "P_hi"] = np.nanpercentile(Pm, hi, axis=0)
        out.loc[m, "P_boot_median"] = np.nanmedian(Pm, axis=0)
        ce = C[:, idx_el]
        ranks = (-ce).argsort(axis=1).argsort(axis=1)
        out.loc[el, "p_top"] = (ranks < top_n).mean(axis=0)
    out.attrs["bootstrap_reps"] = R
    return out


def top_stints(design: pd.DataFrame, mu: list, players: pd.DataFrame, teams: pd.DataFrame,
               key: tuple, k: int = 3) -> pd.DataFrame:
    """Biggest stints by n * h * (y - mu - beta + q_o): the evidence behind a player's c."""
    pid, role = key
    d = design[(design["player_id"] == pid) & (design["role"] == role) & (design["h"] > 0)].copy()
    if d.empty:
        return d
    beta = players.set_index(["player_id", "role"]).loc[key, "beta"]
    q = teams.set_index("team_canon")["q"]
    m = mu[1] if role == "bowl" else mu[0]
    d["impact"] = d["n_balls"] * d["h"] * (d["y"] - m - beta + d["opp_canon"].map(q).fillna(0))
    return d.nlargest(k, "impact")


def team_scores(teams: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    t = teams.copy()
    t["T"] = to_score(t["tau"].to_numpy())
    # movers: players appearing for this team and at least one other
    pt = design.groupby("player_id")["team_canon"].agg(lambda s: set(s))
    movers = {tm: 0 for tm in t["team_canon"]}
    for s in pt:
        if len(s) > 1:
            for tm in s:
                movers[tm] = movers.get(tm, 0) + 1
    t["movers"] = t["team_canon"].map(movers).fillna(0).astype(int)
    t["weak_identification"] = t["movers"] == 0
    return t.sort_values("T", ascending=False).reset_index(drop=True)


def combined_P(scored: pd.DataFrame) -> pd.DataFrame:
    """Single P per player: exposure-weighted mean of P_bat and P_bowl (plan section 1)."""
    s = scored[["player_id", "role", "P", "exposure"]].copy()
    s["wP"] = s["P"] * s["exposure"]
    g = s.groupby("player_id").agg(wP=("wP", "sum"), E=("exposure", "sum"))
    return (g["wP"] / g["E"].where(g["E"] > 0)).rename("P_combined").reset_index()


def rank_lists(scored: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    top_n = int(cfg["rank"].get("top_n", 10))
    out = {}
    for r in ("bat", "bowl"):
        el = scored[(scored["role"] == r) & scored["eligible"]].copy()
        el = el.sort_values(["P", "p_top"], ascending=[False, False]).reset_index(drop=True)
        el.insert(0, "rank", np.arange(1, len(el) + 1))
        out[r] = el
        out[f"{r}_top"] = el.head(top_n)
    return out


def run(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    design = io.load("fit_design", paths)
    players = io.load("fit_players", paths)
    teams = io.load("fit_teams", paths)
    pl = io.load("players", paths)
    meta = io.load_json("fit_meta", paths, sub="fit")
    ev = player_evidence(design, players, cfg)
    ev = ev.merge(pl[["player_id", "player_name", "selectable"]], on="player_id", how="left")
    el = eligibility(ev, cfg, set(pl.loc[pl["selectable"], "player_id"]))
    scored = score_players(el)
    bsf = paths.interim / "bootstrap.npz"
    if bsf.exists():
        bs = dict(np.load(bsf))
        scored = bootstrap_summary(scored, bs, cfg)
    else:
        log.warning("no bootstrap draws found; P intervals and P(top 10) left empty")
        for c in ("P_lo", "P_hi", "P_boot_median"):
            scored[c] = np.nan
        scored["p_top"] = np.nan
    scored = scored.merge(combined_P(scored), on="player_id", how="left")
    lists = rank_lists(scored, cfg)
    tscore = team_scores(teams, design)

    io.save(scored, "scored_players", paths)
    cols = ["rank", "player_id", "player_name", "role", "P", "P_lo", "P_hi", "p_top", "c", "beta", "exposure",
            "shrink", "pressure_stints", "pressure_balls", "recent_pressure_stints", "y_pressure", "y_ordinary",
            "stints", "balls", "teams", "n_teams", "last_date"]
    io.save_output(lists["bat_top"][cols], "top10_batters", paths)
    io.save_output(lists["bowl_top"][cols], "top10_bowlers", paths)
    io.save_output(lists["bat"][cols], "ranked_batters_all_eligible", paths)
    io.save_output(lists["bowl"][cols], "ranked_bowlers_all_eligible", paths)
    io.save_output(scored.sort_values(["role", "P"], ascending=[True, False]), "player_scores", paths)
    io.save_output(tscore, "team_scores", paths)

    # evidence: three biggest pressure stints per ranked player
    rows = []
    for r in ("bat", "bowl"):
        for _, p in lists[f"{r}_top"].iterrows():
            ts = top_stints(design, meta["mu"], players, teams, (p["player_id"], r))
            for _, s in ts.iterrows():
                rows.append(dict(role=r, rank=p["rank"], player_name=p["player_name"],
                                 match_date=pd.Timestamp(s["match_date"]).date(), comp_name=s["comp_name"],
                                 team=s["team_canon"], opponent=s["opp_canon"], innings=s["innings"],
                                 overid=s.get("overid"), n_balls=s["n_balls"], h=round(s["h"], 3),
                                 y=s["y"], sum_wpa=s["sum_wpa"], impact=s["impact"], stage=s["stage"]))
    io.save_output(pd.DataFrame(rows), "top10_evidence_stints", paths)
    write_rankings_md(lists, tscore, scored, meta, paths, cfg)
    for r in ("bat", "bowl"):
        log.info("%s eligible: %d; top 10: %s", r, len(lists[r]),
                 ", ".join(f"{n} ({p:.1f})" for n, p in zip(lists[f'{r}_top']['player_name'], lists[f'{r}_top']['P'])))


def write_rankings_md(lists, tscore, scored, meta, paths: Paths, cfg) -> None:
    def tbl(df):
        show = df[["rank", "player_name", "P", "P_lo", "P_hi", "p_top", "pressure_stints", "teams"]].copy()
        show.columns = ["Rank", "Player", "P", "P lo", "P hi", "P(top 10)", "Pressure stints", "Teams"]
        return show.to_markdown(index=False, floatfmt=".2f")
    lines = ["# SupeRR Selector S2 - final rankings", "",
             f"Penalties (CV): {meta['lambdas']}; bootstrap reps: {meta.get('bootstrap_reps', 0)}", "",
             "## Top 10 batters under pressure", "", tbl(lists["bat_top"]), "",
             "## Top 10 bowlers under pressure", "", tbl(lists["bowl_top"]), "",
             "## Team clutch T", "",
             tscore[["team_canon", "T", "tau", "q", "movers", "weak_identification"]].to_markdown(index=False, floatfmt=".3g"),
             ""]
    (paths.outputs / "RANKINGS.md").write_text("\n".join(lines), encoding="utf-8")
