"""Stage 1: the state table - one row per delivery, describing the state *before* it (spec section 1).

No Jev here. Super overs (innings 3/4) and no-result matches are excluded.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Paths
from . import io
from .cricket import STATE_COLS, outcome_class

log = logging.getLogger("superr.states")


def _free_hit(is_noball: np.ndarray, is_wide: np.ndarray, new_innings: np.ndarray) -> np.ndarray:
    """free_hit[i] = previous delivery was a no-ball, carrying through wides and further no-balls."""
    fh = np.zeros(len(is_noball), dtype=bool)
    for i in range(1, len(fh)):
        if new_innings[i]:
            continue
        if is_noball[i - 1]:
            fh[i] = True
        elif is_wide[i - 1]:
            fh[i] = fh[i - 1]
    return fh


def venue_par(b: pd.DataFrame, matches: pd.DataFrame, prior_n: float) -> pd.Series:
    """Average completed first-innings total per venue_id, shrunk toward the global mean.

    Only full-length (120-ball) first innings count. The result is a number, never a name:
    a venue name like Chepauk would point straight at the home team.
    """
    inn1 = b[b["innings"] == 1].groupby("match_id").agg(total=("ball_total", "sum"),
                                                        max_balls=("max_balls", "first"))
    inn1 = inn1[inn1["max_balls"] == 120].join(matches.set_index("match_id")["venue_id"])
    glob = inn1["total"].mean()
    agg = inn1.groupby("venue_id")["total"].agg(["sum", "count"])
    par = (agg["sum"] + prior_n * glob) / (agg["count"] + prior_n)
    return par, glob


def build_states(bbb: pd.DataFrame, matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dcfg = cfg["data"]
    wide_set, nob_set, bye_set = set(dcfg["wide_extras"]), set(dcfg["noball_extras"]), set(dcfg["bye_extras"])
    keep = matches.loc[~matches["excluded"], "match_id"]
    b = bbb[bbb["match_id"].isin(keep)].copy()
    if dcfg.get("exclude_super_overs", True):
        b = b[b["innings"].isin([1, 2])]
    mcols = ["match_id", "comp_id", "match_date", "stage", "ko_label", "venue_id", "format", "impact_rule",
             "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "win_team_id",
             "is_tie", "comp_kind", "season_year"]
    b = b.merge(matches[mcols], on="match_id", how="left")
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    et = b["extras_type"].fillna("")
    b["is_wide"] = et.isin(wide_set)
    b["is_noball"] = et.isin(nob_set)
    b["is_bye"] = et.isin(bye_set)
    b["is_legal"] = ~(b["is_wide"] | b["is_noball"])
    for c in ("runs", "extras1_runs", "extras_2_runs"):
        b[c] = b[c].fillna(0).astype(int)
    b["extras"] = b["extras1_runs"] + b["extras_2_runs"]
    b["ball_total"] = b["runs"] + b["extras"]
    b["is_wicket"] = b["is_wicket"].astype(bool)
    b["phase"] = np.select([b["overid"] <= 6, b["overid"] <= 15], ["powerplay", "middle"], default="death")

    g = b.groupby(["match_id", "innings"], sort=False)
    b["runs_before"] = g["ball_total"].cumsum() - b["ball_total"]
    b["wickets_before"] = g["is_wicket"].cumsum() - b["is_wicket"].astype(int)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["new_innings"] = g.cumcount().eq(0)

    # ball quota: revised overs fields are already in balls (e.g. 114)
    q1 = b["first_inn_revised_overs"].fillna(120)
    q2 = b["second_inn_revised_overs"].fillna(b["first_inn_revised_overs"]).fillna(120)
    b["max_balls"] = np.where(b["innings"] == 1, q1, q2).astype(float)
    legal_total = g["is_legal"].transform("sum")
    b["max_balls"] = np.maximum(b["max_balls"], legal_total)  # guard against data noise

    inn_tot = b.groupby(["match_id", "innings"])["ball_total"].sum().unstack()
    inn1_total = b["match_id"].map(inn_tot[1])
    tgt = b["revised_target"].where(b["revised_target"].notna(), inn1_total + 1)
    b["target"] = np.where(b["innings"] == 2, tgt, np.nan)

    inn1_quota = b[b["innings"] == 1].groupby("match_id")["max_balls"].first()
    inn2_quota = b[b["innings"] == 2].groupby("match_id")["max_balls"].first()
    mode = cfg["states"].get("inn2_start_max_balls", "innings1")
    if mode == "actual":
        b["inn2_start_max_balls"] = b["match_id"].map(inn2_quota).fillna(b["match_id"].map(inn1_quota))
    else:
        b["inn2_start_max_balls"] = b["match_id"].map(inn1_quota)

    b["free_hit"] = _free_hit(b["is_noball"].to_numpy(), b["is_wide"].to_numpy(), b["new_innings"].to_numpy())

    # partnership (reset at each wicket)
    gp = b.groupby(["match_id", "innings", "wickets_before"], sort=False)
    b["partnership_runs"] = gp["ball_total"].cumsum() - b["ball_total"]
    b["partnership_balls"] = gp["is_legal"].cumsum() - b["is_legal"].astype(int)

    # striker / non-striker innings so far (balls faced exclude wides)
    b["faced"] = (~b["is_wide"]).astype(int)
    gs = b.groupby(["match_id", "innings", "striker_id"], sort=False)
    b["striker_runs"] = gs["runs"].cumsum() - b["runs"]
    b["striker_balls"] = gs["faced"].cumsum() - b["faced"]
    b["seq"] = np.arange(len(b))
    after = b[["match_id", "innings", "striker_id", "seq"]].copy()
    after["ns_runs"] = b["striker_runs"] + b["runs"]
    after["ns_balls"] = b["striker_balls"] + b["faced"]
    after = after.rename(columns={"striker_id": "pid"}).sort_values("seq")
    left = b[["match_id", "innings", "non_striker_id", "seq"]].rename(columns={"non_striker_id": "pid"})
    ns = pd.merge_asof(left.sort_values("seq"), after, on="seq", by=["match_id", "innings", "pid"],
                       allow_exact_matches=False).sort_values("seq")
    b["non_striker_runs"] = ns["ns_runs"].fillna(0).to_numpy()
    b["non_striker_balls"] = ns["ns_balls"].fillna(0).to_numpy()

    par, glob = venue_par(b, matches, cfg["data"].get("venue_par_prior_innings", 3))
    b["venue_par"] = b["venue_id"].map(par).fillna(glob).round(1)

    # state fields (spec section 1)
    b["runs"] = b["runs"]  # bat runs of this delivery (outcome); state runs are 'runs_before'
    S = pd.DataFrame({
        "innings": b["innings"].astype(int),
        "runs": b["runs_before"].astype(float),
        "wickets": b["wickets_before"].astype(float),
        "legal_balls": b["legal_before"].astype(float),
        "max_balls": b["max_balls"].astype(float),
        "target": b["target"].astype(float),
        "free_hit": b["free_hit"].astype(bool),
        "partnership_runs": b["partnership_runs"].astype(float),
        "partnership_balls": b["partnership_balls"].astype(float),
        "striker_runs": b["striker_runs"].astype(float),
        "striker_balls": b["striker_balls"].astype(float),
        "non_striker_runs": b["non_striker_runs"].astype(float),
        "non_striker_balls": b["non_striker_balls"].astype(float),
        "format": b["format"].astype(str),
        "impact_rule": b["impact_rule"].astype(bool),
        "venue_par": b["venue_par"].astype(float),
    })
    S["balls_left"] = S["max_balls"] - S["legal_balls"]
    S["runs_needed"] = np.where(S["innings"] == 2, S["target"] - S["runs"], np.nan)

    ctx_cols = ["match_id", "comp_id", "match_date", "season_year", "stage", "ko_label", "comp_kind",
                "overid", "over_ball_num", "innings_ball_num", "phase", "batting_team_id", "bowling_team_id",
                "striker_id", "non_striker_id", "bowler_id", "dismissed_player_id", "dismissal_type",
                "extras_type", "is_wide", "is_noball", "is_bye", "is_legal", "is_wicket",
                "is_four", "is_six", "win_team_id", "is_tie", "inn2_start_max_balls"]
    out = pd.concat([b[ctx_cols].reset_index(drop=True), S[STATE_COLS].reset_index(drop=True)], axis=1)
    out["bat_runs"] = b["runs"].to_numpy()
    out["extras"] = b["extras"].to_numpy()
    out["ball_total"] = b["ball_total"].to_numpy()
    out["outcome"] = outcome_class(b["extras_type"], b["is_wicket"], b["ball_total"], wide_set, nob_set).to_numpy()

    # innings bookkeeping for the post-ball state of the last delivery
    gi = out.groupby(["match_id", "innings"], sort=False)
    out["is_last_ball"] = gi.cumcount(ascending=False).eq(0)
    out["ball_seq"] = np.arange(len(out))
    out["bat_won"] = np.where(out["is_tie"], 0.5,
                              (out["win_team_id"] == out["batting_team_id"]).astype(float))
    return out


def run(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    raw = io.read_raw(paths, ["bbb"])
    matches = io.load("matches", paths)
    st = build_states(raw["bbb"], matches, cfg)
    io.save(st, "states", paths)
    log.info("state table: %d deliveries in %d matches (%d innings-2 balls)",
             len(st), st["match_id"].nunique(), (st["innings"] == 2).sum())
