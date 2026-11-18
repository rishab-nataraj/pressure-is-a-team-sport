"""Standings, remaining fixtures and match stakes (spec section 4, Jev call C inputs).

Every (match, team) gets a *stakes context*:
  * kind "knockout"  - fixed stakes by stage (Qualifier 1 = 0.5, others = 1)
  * kind "bilateral" - P(win series | win) - P(win series | lose) with coin-flip future games
                       (decider = 1, dead rubber = 0); no Jev call needed
  * kind "league"    - IPL league / tournament group / Super 4: standings before the match,
                       remaining fixtures in the group and qualifying places. Jev call C, or the
                       empirical Monte Carlo (coin-flip results, ranked on points then NRR).
"""
from __future__ import annotations

import logging
from math import comb

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("superr.standings")

KO_LABELS = ("Qualifier1", "Qualifier2", "Eliminator", "Semi Final", "Final")


# --------------------------------------------------------------------------------------- results
def match_results(matches: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): result, points, and NRR ingredients (runs and balls for/against)."""
    inn = (states.groupby(["match_id", "innings"])
           .agg(team=("batting_team_id", "first"), runs=("ball_total", "sum"),
                legal=("is_legal", "sum"), wkts=("is_wicket", "sum"), quota=("max_balls", "first"))
           .reset_index())
    # NRR convention: a side bowled out is charged its full quota of balls
    inn["balls_nrr"] = np.where(inn["wkts"] >= 10, inn["quota"], inn["legal"])
    rows = []
    for m in matches.itertuples(index=False):
        for team, opp in ((m.home_team_id, m.away_team_id), (m.away_team_id, m.home_team_id)):
            rows.append(dict(match_id=m.match_id, comp_id=m.comp_id, team_id=team, opp_id=opp,
                             match_date=m.match_date, match_number=m.match_number, stage=m.stage,
                             ko_label=m.ko_label, win_team_id=m.win_team_id, is_tie=m.is_tie))
    r = pd.DataFrame(rows)
    r["won"] = r["win_team_id"] == r["team_id"]
    r["no_result"] = r["win_team_id"].isna()  # includes ties without a super-over winner
    r["lost"] = ~r["won"] & ~r["no_result"]
    r["points"] = np.where(r["won"], 2, np.where(r["no_result"], 1, 0))
    bat = inn.rename(columns={"team": "team_id", "runs": "rf", "balls_nrr": "bf"})[["match_id", "team_id", "rf", "bf"]]
    r = r.merge(bat, on=["match_id", "team_id"], how="left")
    opp = bat.rename(columns={"team_id": "opp_id", "rf": "ra", "bf": "ba"})
    r = r.merge(opp, on=["match_id", "opp_id"], how="left")
    for c in ("rf", "bf", "ra", "ba"):
        r[c] = r[c].fillna(0.0)
    # NRR ignores no-result matches
    nr = r["no_result"] & ~r["is_tie"]
    r.loc[nr, ["rf", "bf", "ra", "ba"]] = 0.0
    return r


def standings_table(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("team_id").agg(played=("match_id", "size"), won=("won", "sum"), lost=("lost", "sum"),
                                   no_result=("no_result", "sum"), points=("points", "sum"),
                                   rf=("rf", "sum"), bf=("bf", "sum"), ra=("ra", "sum"), ba=("ba", "sum"))
    with np.errstate(divide="ignore", invalid="ignore"):
        nrr = 6 * g["rf"] / g["bf"] - 6 * g["ra"] / g["ba"]
    g["nrr"] = np.nan_to_num(nrr.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return g.drop(columns=["rf", "bf", "ra", "ba"])


# ------------------------------------------------------------------------------------ structure
def _phases_for_comp(comp_id: int, cm: pd.DataFrame, tcfg: dict, is_ipl: bool) -> list[dict]:
    spec = (tcfg.get("competitions") or {}).get(comp_id) or (tcfg.get("competitions") or {}).get(str(comp_id))
    if spec:
        return spec["phases"]
    labels = [s for s in cm["stage"].unique() if s not in ("Qualifier", "Eliminator", "Semi Final", "Final")]
    has_ko = cm["ko_label"].isin(KO_LABELS).any()
    q = 4 if is_ipl else (2 if has_ko else 1)
    return [{"name": s.lower(), "stages": [s], "qualify": q} for s in labels]


def assign_phases(matches: pd.DataFrame, comps: pd.DataFrame, tcfg: dict) -> pd.DataFrame:
    """Label each league-type match of a multi-team competition with (phase, group, qualify)."""
    m = matches.copy()
    m["phase"] = None
    m["group"] = None
    m["qualify"] = np.nan
    kinds = comps.set_index("comp_id")["comp_kind"]
    for comp_id, cm in m.groupby("comp_id"):
        if kinds.get(comp_id) not in ("ipl", "tournament"):
            continue
        for ph in _phases_for_comp(comp_id, cm, tcfg, kinds.get(comp_id) == "ipl"):
            if "match_numbers" in ph:
                lo, hi = ph["match_numbers"]
                sel = cm["match_number"].between(lo, hi)
            else:
                sel = cm["stage"].isin(ph["stages"])
            sel &= ~cm["ko_label"].isin(KO_LABELS)
            pm = cm[sel]
            G = nx.Graph()
            G.add_edges_from(zip(pm["home_team_id"], pm["away_team_id"]))
            for gi, comp_nodes in enumerate(sorted(nx.connected_components(G), key=lambda s: min(s))):
                gsel = pm.index[pm["home_team_id"].isin(comp_nodes)]
                m.loc[gsel, "phase"] = ph["name"]
                m.loc[gsel, "group"] = f"{ph['name']}-{gi + 1}"
                m.loc[gsel, "qualify"] = float(ph["qualify"])
    return m


# ------------------------------------------------------------------------------------- contexts
def build_contexts(matches: pd.DataFrame, comps: pd.DataFrame, results: pd.DataFrame,
                   tcfg: dict) -> list[dict]:
    """Stakes contexts for every (match, team) of non-excluded matches."""
    m = assign_phases(matches, comps, tcfg)
    m = m.sort_values(["comp_id", "match_date", "match_number"]).reset_index(drop=True)
    kinds = comps.set_index("comp_id")["comp_kind"]
    ko_stakes = tcfg.get("knockout_stakes", {})
    res_by_match = results.set_index(["match_id", "team_id"])
    contexts = []
    for comp_id, cm in m.groupby("comp_id", sort=False):
        kind = kinds.get(comp_id)
        cm = cm.reset_index(drop=True)
        for i, row in cm.iterrows():
            if row["excluded"]:
                continue
            for team, opp in ((row.home_team_id, row.away_team_id), (row.away_team_id, row.home_team_id)):
                base = dict(match_id=int(row.match_id), team_id=int(team), opp_id=int(opp),
                            comp_id=int(comp_id), comp_kind=kind, stage=row.stage,
                            ko_label=row.ko_label)
                won = bool(res_by_match.loc[(row.match_id, team), "won"])
                base["won"] = won
                if row.ko_label in KO_LABELS:
                    v = float(ko_stakes.get(row.ko_label, 1.0))
                    contexts.append(dict(base, kind="knockout", fixed_stakes=v,
                                         fixed_p_win=1.0, fixed_p_lose=1.0 - v))
                    continue
                if kind == "bilateral":
                    prev = cm.iloc[:i]
                    w_t = int((prev["win_team_id"] == team).sum())
                    w_o = int((prev["win_team_id"] == opp).sum())
                    left = len(cm) - i - 1
                    contexts.append(dict(base, kind="bilateral", series_length=len(cm),
                                         wins_team=w_t, wins_opp=w_o, matches_left_after=left))
                    continue
                grp = row["group"]
                if grp is None or pd.isna(row["qualify"]):
                    # league-type match outside any configured phase: treat as low-stakes friendly
                    contexts.append(dict(base, kind="knockout", fixed_stakes=0.0,
                                         fixed_p_win=0.5, fixed_p_lose=0.5))
                    continue
                gm = cm[cm["group"] == grp]
                before = gm[(gm["match_date"] < row.match_date) |
                            ((gm["match_date"] == row.match_date) & (gm["match_number"] < row.match_number))]
                after = gm[((gm["match_date"] > row.match_date) |
                            ((gm["match_date"] == row.match_date) & (gm["match_number"] > row.match_number)))]
                teams = sorted(set(gm["home_team_id"]) | set(gm["away_team_id"]))
                st = standings_table(results[results["match_id"].isin(before["match_id"])])
                st = st.reindex(teams).fillna({"played": 0, "won": 0, "lost": 0, "no_result": 0,
                                              "points": 0, "nrr": 0.0})
                contexts.append(dict(
                    base, kind="league", group=grp, phase=row["phase"], qualify=int(row["qualify"]),
                    teams=teams,
                    standings=st.reset_index().rename(columns={"index": "team_id"}).to_dict("records"),
                    remaining=[(int(a), int(b)) for a, b in zip(after["home_team_id"], after["away_team_id"])],
                    this_match=(int(team), int(opp)),
                ))
    return contexts


# ------------------------------------------------------------------------------- stakes engines
def bilateral_stakes(series_length: int, wins_team: int, wins_opp: int, matches_left_after: int) -> tuple[float, float]:
    """Exact P(win series | win this) and P(win series | lose this); drawn series counts 0.5."""
    def p_series(a: int, b: int, r: int) -> float:
        tot = 0.0
        for x in range(r + 1):
            pa, pb = a + x, b + (r - x)
            val = 1.0 if pa > pb else (0.5 if pa == pb else 0.0)
            tot += comb(r, x) * 0.5 ** r * val
        return tot
    r = matches_left_after
    return p_series(wins_team + 1, wins_opp, r), p_series(wins_team, wins_opp + 1, r)


def league_stakes_mc(ctx: dict, n_sims: int, nrr_noise: float, rng: np.random.Generator) -> tuple[float, float]:
    """Monte Carlo of the remaining fixtures with coin-flip results; rank on points, then NRR."""
    teams = ctx["teams"]
    idx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    q = ctx["qualify"]
    if q >= nT:
        return 1.0, 1.0
    st = {r["team_id"]: r for r in ctx["standings"]}
    pts0 = np.array([st[t]["points"] for t in teams], float)
    nrr0 = np.array([st[t]["nrr"] for t in teams], float)
    played = np.array([st[t]["played"] for t in teams], float)
    rem = ctx["remaining"]
    focal, opp = idx[ctx["this_match"][0]], idx[ctx["this_match"][1]]
    rem_cnt = np.zeros(nT)
    for a, b in rem:
        rem_cnt[idx[a]] += 1
        rem_cnt[idx[b]] += 1
    rem_cnt[focal] += 1
    rem_cnt[opp] += 1
    pts = np.repeat(pts0[None, :], n_sims, axis=0)
    if rem:
        A = np.array([idx[a] for a, _ in rem])
        B = np.array([idx[b] for _, b in rem])
        a_wins = rng.random((n_sims, len(rem))) < 0.5
        np.add.at(pts, (np.arange(n_sims)[:, None], A[None, :].repeat(n_sims, 0)), 2 * a_wins)
        np.add.at(pts, (np.arange(n_sims)[:, None], B[None, :].repeat(n_sims, 0)), 2 * ~a_wins)
    scale = nrr_noise * np.sqrt(rem_cnt) / np.sqrt(played + rem_cnt)
    nrr = nrr0[None, :] + rng.normal(size=(n_sims, nT)) * scale[None, :]
    tiny = rng.random((n_sims, nT)) * 1e-9

    def p_adv(focal_wins: bool) -> float:
        p = pts.copy()
        p[:, focal if focal_wins else opp] += 2
        key = p * 1000 + nrr + tiny                   # points first, then NRR
        rank = (key > key[:, [focal]]).sum(axis=1)     # number of teams strictly above focal
        return float((rank < q).mean())

    return p_adv(True), p_adv(False)


def empirical_stakes(contexts: list[dict], cfg: dict) -> pd.DataFrame:
    ecfg = cfg["empirical"]
    rng = np.random.default_rng(ecfg.get("seed", 0))
    out = []
    for c in contexts:
        if c["kind"] == "knockout":
            pw, pl = c["fixed_p_win"], c["fixed_p_lose"]
            method = "rule:knockout"
        elif c["kind"] == "bilateral":
            pw, pl = bilateral_stakes(c["series_length"], c["wins_team"], c["wins_opp"], c["matches_left_after"])
            method = "rule:bilateral"
        else:
            pw, pl = league_stakes_mc(c, ecfg.get("stakes_sims", 4000), ecfg.get("nrr_noise", 0.35), rng)
            method = "empirical:montecarlo"
        out.append(dict(match_id=c["match_id"], team_id=c["team_id"], kind=c["kind"],
                        p_advance_if_win=pw, p_advance_if_lose=pl,
                        stakes=float(np.clip(pw - pl, 0, 1)), method=method))
    return pd.DataFrame(out)


def contexts_frame(contexts: list[dict]) -> pd.DataFrame:
    keep = ["match_id", "team_id", "opp_id", "comp_id", "comp_kind", "stage", "ko_label", "kind", "won",
            "group", "phase", "qualify", "series_length", "wins_team", "wins_opp", "matches_left_after"]
    return pd.DataFrame([{k: c.get(k) for k in keep} for c in contexts])


def actual_advancement(matches: pd.DataFrame, comps: pd.DataFrame, tcfg: dict) -> pd.DataFrame:
    """Did each team actually advance from each group/phase (for validation 7.5)?"""
    m = assign_phases(matches, comps, tcfg)
    out = []
    for (comp_id, grp), gm in m[m["group"].notna()].groupby(["comp_id", "group"]):
        teams = set(gm["home_team_id"]) | set(gm["away_team_id"])
        last = gm["match_date"].max()
        later = m[(m["comp_id"] == comp_id) & (m["match_date"] > last)]
        later_teams = set(later["home_team_id"]) | set(later["away_team_id"])
        if later.empty:  # no next round (e.g. tri-series without final): winner = table topper
            res = gm
            pts = {}
            for t in teams:
                pts[t] = 2 * (res["win_team_id"] == t).sum() + ((res["win_team_id"].isna()) &
                                                                ((res["home_team_id"] == t) | (res["away_team_id"] == t))).sum()
            best = max(pts.values())
            later_teams = {t for t, p in pts.items() if p == best}
        for t in teams:
            out.append(dict(comp_id=comp_id, group=grp, team_id=t, advanced=t in later_teams))
    return pd.DataFrame(out)
