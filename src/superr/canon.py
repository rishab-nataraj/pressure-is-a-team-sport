"""Stage 0: canonical team identities, player names and an enriched match table.

* team_canon is resolved by team_name (not team_id) through an alias table, so a renamed
  franchise or a second-string side touring as "India" gets one T.
* A-teams / emerging sides are separate canonical teams unless data.a_teams_as_parent is on.
* Player names come only from competition_team_player_mapping (the Unstop dropdown source).
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import Paths
from . import io

log = logging.getLogger("superr.canon")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def canonical_team_name(name: str, cfg: dict) -> str:
    aliases = {_norm(k).lower(): _norm(v) for k, v in (cfg["aliases"].get("aliases") or {}).items()}
    n = _norm(name)
    n = aliases.get(n.lower(), n)
    if cfg["data"].get("a_teams_as_parent"):
        for suf in cfg["aliases"].get("a_team_suffixes", []):
            if n.endswith(suf):
                n = aliases.get(n[: -len(suf)].lower(), n[: -len(suf)])
                break
    return n


def build_team_canon(teams: pd.DataFrame, comps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = teams.merge(comps[["comp_id", "comp_name", "match_format"]], on="comp_id", how="left")
    t["team_name"] = t["team_name"].map(_norm)
    t["team_canon"] = t["team_name"].map(lambda s: canonical_team_name(s, cfg))
    t["is_ipl_comp"] = t["comp_name"].str.startswith("IPL")
    g = t.groupby("team_id").agg(
        team_name=("team_name", lambda s: s.mode().iloc[0]),
        team_canon=("team_canon", lambda s: s.mode().iloc[0]),
        n_comps=("comp_id", "nunique"),
        is_franchise=("is_ipl_comp", "any"),
    ).reset_index()
    suffixes = tuple(cfg["aliases"].get("a_team_suffixes", []))
    g["is_a_team"] = g["team_name"].str.endswith(suffixes)
    multi = g.groupby("team_id")["team_canon"].nunique()
    if (multi > 1).any():
        log.warning("team_ids mapping to several canonical names: %s", list(multi[multi > 1].index))
    return g


def build_players(mapping: pd.DataFrame, team_canon: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    mp = mapping.merge(team_canon[["team_id", "team_canon"]], on="team_id", how="left")
    # latest name per player (names are unique per id in this pack, but be safe)
    comp_last = matches.groupby("comp_id")["match_date"].max()
    mp["comp_date"] = mp["comp_id"].map(comp_last)
    mp = mp.sort_values("comp_date")
    players = mp.groupby("player_id").agg(
        player_name=("player_name", "last"),
        teams=("team_canon", lambda s: "; ".join(sorted(set(s.dropna())))),
        n_teams=("team_canon", "nunique"),
        n_comps=("comp_id", "nunique"),
    ).reset_index()
    players["selectable"] = True  # present in the dropdown source file
    return players


def classify_competitions(comps: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    c = comps.copy()
    teams_per = (pd.concat([matches[["comp_id", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                            matches[["comp_id", "away_team_id"]].rename(columns={"away_team_id": "t"})])
                 .groupby("comp_id")["t"].nunique())
    c["n_teams"] = c["comp_id"].map(teams_per).fillna(0).astype(int)
    c["is_ipl"] = c["comp_name"].str.startswith("IPL")
    c["format"] = np.where(c["match_format"].eq("T20I"), "T20I",
                           np.where(c["is_ipl"], "IPL", "other T20"))
    c["comp_kind"] = np.where(c["is_ipl"], "ipl", np.where(c["n_teams"] > 2, "tournament", "bilateral"))
    year = c["season"].astype(str).str.slice(0, 4).astype(int)
    c["season_year"] = year
    c["impact_rule"] = c["is_ipl"] & (year >= 2023)
    return c


def build_matches(raw: dict, team_canon: pd.DataFrame, comps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = raw["matches"].copy()
    m = m.merge(comps[["comp_id", "comp_name", "season", "season_year", "format", "comp_kind",
                       "impact_rule", "n_teams"]], on="comp_id", how="left")
    tc = team_canon.set_index("team_id")["team_canon"]
    for side in ("home", "away"):
        m[f"{side}_canon"] = m[f"{side}_team_id"].map(tc)
    m["win_canon"] = m["win_team_id"].map(tc)
    res = m["result"].fillna("").str.lower()
    m["is_tie"] = res.str.contains("tie") | res.str.contains("super over")
    m["is_no_result"] = m["win_team_id"].isna() & ~(m["is_tie"] & bool(cfg["data"].get("keep_ties", True)))
    m["went_to_super_over"] = res.str.contains("super over") | res.str.contains("super ov")
    bbb_ids = set(raw["bbb"]["match_id"].unique())
    m["has_bbb"] = m["match_id"].isin(bbb_ids)
    m["is_emerging"] = m["comp_id"].isin(cfg["data"].get("emerging_comp_ids", []))
    reason = np.select(
        [~m["has_bbb"], m["is_no_result"], m["is_emerging"] & bool(cfg["data"].get("drop_emerging"))],
        ["no ball-by-ball", "no result", "emerging cup dropped (ablation)"], default="")
    m["exclude_reason"] = reason
    m["excluded"] = m["exclude_reason"] != ""
    # Knockout label: first IPL "Qualifier" by date is Qualifier 1 (stakes ~0.5), second is Qualifier 2.
    m = m.sort_values(["comp_id", "match_date", "match_number"]).reset_index(drop=True)
    q = m["stage"].eq("Qualifier")
    m["q_rank"] = np.nan
    m.loc[q, "q_rank"] = m[q].groupby("comp_id").cumcount() + 1
    m["ko_label"] = np.where(q & m["q_rank"].eq(1), "Qualifier1",
                             np.where(q, "Qualifier2", m["stage"]))
    return m.drop(columns=["q_rank"])


def run(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    raw = io.read_raw(paths)
    comps = classify_competitions(raw["competitions"], raw["matches"])
    team_canon = build_team_canon(raw["teams"], comps, cfg)
    matches = build_matches(raw, team_canon, comps, cfg)
    players = build_players(raw["mapping"], team_canon, matches)
    io.save(team_canon, "team_canon", paths)
    io.save(comps, "competitions", paths)
    io.save(matches, "matches", paths)
    io.save(players, "players", paths)
    n_ex = matches["excluded"].sum()
    log.info("canonical teams: %d (from %d team_ids); players: %d; matches: %d (%d excluded: %s)",
             team_canon["team_canon"].nunique(), len(team_canon), len(players), len(matches), n_ex,
             matches.loc[matches.excluded, "exclude_reason"].value_counts().to_dict())
