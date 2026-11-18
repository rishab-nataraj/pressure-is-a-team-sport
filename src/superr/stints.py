"""Stage 4: the stint table that P and T consume (spec section 6).

Batter stint: first ball at the crease (striker or non-striker) until dismissal / retirement /
end of innings / end of match. One per (match, innings, batter).
Bowler stint: one over (overid) by one bowler_id; a bowler leaving mid-over keeps only his balls.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Paths

log = logging.getLogger("superr.stints")

RETIRED = {"Retire Out", "Retired Out", "Retired Hurt"}


def _agg(df: pd.DataFrame, keys: list[str], wpa: str, wpa_li: str) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)
    out = g.agg(n_balls=(wpa, "size"), sum_wpa=(wpa, "sum"), sum_wpa_li=(wpa_li, "sum"),
                sum_li=("LI", "sum"), mean_li=("LI", "mean"), max_li=("LI", "max"))
    return out


def batter_stints(b: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "innings", "player_id"]
    # presence at the crease (as striker or non-striker), for entry/exit and end reason
    pres = pd.concat([
        b[["match_id", "innings", "ball_seq", "striker_id", "wp_before", "wp_after", "LI"]].rename(columns={"striker_id": "player_id"}),
        b[["match_id", "innings", "ball_seq", "non_striker_id", "wp_before", "wp_after", "LI"]].rename(columns={"non_striker_id": "player_id"}),
    ]).sort_values("ball_seq")
    gp = pres.groupby(keys, sort=False)
    span = gp.agg(first_seq=("ball_seq", "first"), last_seq=("ball_seq", "last"),
                  entry_wp=("wp_before", "first"), entry_li=("LI", "first"),
                  exit_wp=("wp_after", "last"), exit_li=("LI", "last"))
    att = b[b["bat_pid"] >= 0].rename(columns={"bat_pid": "player_id"})
    perf = _agg(att, keys, "wpa_bat", "wpa_bat_li")
    s = span.join(perf, how="outer").reset_index()
    for c in ("n_balls",):
        s[c] = s[c].fillna(0).astype(int)
    for c in ("sum_wpa", "sum_wpa_li", "sum_li"):
        s[c] = s[c].fillna(0.0)
    # end reason
    wk = b[b["is_wicket"]][["match_id", "innings", "dismissed_player_id", "dismissal_type"]].dropna(subset=["dismissed_player_id"])
    wk = wk.rename(columns={"dismissed_player_id": "player_id"}).astype({"player_id": np.int64})
    wk = wk.drop_duplicates(keys, keep="last").set_index(keys)["dismissal_type"]
    s["dismissal_type"] = wk.reindex(pd.MultiIndex.from_frame(s[keys])).to_numpy()
    inn_last = b.groupby(["match_id", "innings"])["ball_seq"].max()
    s["inn_last_seq"] = inn_last.reindex(pd.MultiIndex.from_frame(s[["match_id", "innings"]])).to_numpy()
    out = s["dismissal_type"].notna()
    retired = s["dismissal_type"].isin(RETIRED)
    at_end = s["last_seq"] == s["inn_last_seq"]
    s["end_reason"] = np.select(
        [retired, out, at_end & (s["innings"] == 1), at_end & (s["innings"] == 2)],
        ["retired", "out", "not out (innings end)", "match end"], default="retired")
    ctx = b.groupby(["match_id", "innings"]).agg(team_id=("batting_team_id", "first"),
                                                 opp_id=("bowling_team_id", "first"))
    s = s.join(ctx, on=["match_id", "innings"])
    s["role"] = "bat"
    return s.drop(columns=["inn_last_seq", "dismissal_type"])


def bowler_stints(b: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "innings", "overid", "player_id"]
    x = b.rename(columns={"bowl_pid": "player_id"})
    perf = _agg(x, keys, "wpa_bowl", "wpa_bowl_li")
    g = x.groupby(keys, sort=False)
    span = g.agg(first_seq=("ball_seq", "first"), last_seq=("ball_seq", "last"),
                 entry_wp=("wp_before", "first"), entry_li=("LI", "first"),
                 exit_wp=("wp_after_bowler", "last"), exit_li=("LI", "last"),
                 legal=("is_legal", "sum"), team_id=("bowling_team_id", "first"),
                 opp_id=("batting_team_id", "first"))
    s = span.join(perf).reset_index()
    # entry/exit wp from the bowling side's view
    s["entry_wp"] = 1 - s["entry_wp"]
    s["exit_wp"] = 1 - s["exit_wp"]
    over_legal = x.groupby(["match_id", "innings", "overid"])["is_legal"].sum()
    s["over_legal"] = over_legal.reindex(pd.MultiIndex.from_frame(s[["match_id", "innings", "overid"]])).to_numpy()
    inn_last = b.groupby(["match_id", "innings"])["ball_seq"].max()
    s["inn_last_seq"] = inn_last.reindex(pd.MultiIndex.from_frame(s[["match_id", "innings"]])).to_numpy()
    at_end = s["last_seq"] == s["inn_last_seq"]
    s["end_reason"] = np.select(
        [at_end & (s["over_legal"] < 6) & (s["innings"] == 2), at_end & (s["over_legal"] < 6),
         s["legal"] < s["over_legal"]],
        ["match end", "not out (innings end)", "bowler changed mid-over"], default="over complete")
    s["role"] = "bowl"
    return s.drop(columns=["inn_last_seq", "over_legal", "legal"])


def build_stints(b: pd.DataFrame, matches: pd.DataFrame, team_canon: pd.DataFrame, players: pd.DataFrame,
                 stakes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bat = batter_stints(b)
    bowl = bowler_stints(b)
    s = pd.concat([bat, bowl], ignore_index=True, sort=False)
    m = matches.set_index("match_id")
    s["comp_id"] = s["match_id"].map(m["comp_id"])
    s["comp_name"] = s["match_id"].map(m["comp_name"])
    s["match_date"] = s["match_id"].map(m["match_date"])
    s["format"] = s["match_id"].map(m["format"])
    s["impact_rule"] = s["match_id"].map(m["impact_rule"]).astype(bool)
    s["stage"] = s["match_id"].map(m["stage"])
    s["is_emerging"] = s["match_id"].map(m["is_emerging"]).astype(bool)
    tc = team_canon.set_index("team_id")["team_canon"]
    s["team_canon"] = s["team_id"].map(tc)
    s["opp_canon"] = s["opp_id"].map(tc)
    sk = stakes.set_index(["match_id", "team_id"])["stakes"]
    s["stakes"] = sk.reindex(pd.MultiIndex.from_frame(s[["match_id", "team_id"]])).fillna(0.0).to_numpy()
    k = cfg["pressure"].get("k", 1.0)
    s["pressure"] = s["mean_li"] * (1 + k * s["stakes"])
    s["player_name"] = s["player_id"].map(players.set_index("player_id")["player_name"])
    s = s.sort_values(["match_date", "match_id", "innings", "first_seq", "role"]).reset_index(drop=True)
    s.insert(0, "stint_id", np.arange(len(s)))
    cols = ["stint_id", "player_id", "player_name", "role", "match_id", "comp_id", "comp_name", "match_date",
            "stage", "innings", "overid", "team_id", "opp_id", "team_canon", "opp_canon", "format", "impact_rule",
            "is_emerging", "n_balls", "sum_wpa", "sum_wpa_li", "sum_li", "mean_li", "max_li", "entry_wp",
            "entry_li", "exit_wp", "exit_li", "stakes", "pressure", "end_reason", "first_seq", "last_seq"]
    return s[cols]


def run(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    b = io.load("balls", paths)
    s = build_stints(b, io.load("matches", paths), io.load("team_canon", paths), io.load("players", paths),
                     io.load("stakes", paths), cfg)
    io.save(s, "stints", paths)
    log.info("stints: %s; zero-ball stints %d", s["role"].value_counts().to_dict(), int((s["n_balls"] == 0).sum()))
