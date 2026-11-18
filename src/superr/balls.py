"""Stage 3: per-ball WPA, leverage, pressure and attribution (spec section 5).

WPA_bat  = wp_after - wp_before                 (batting side's view)
WPA_bowl = -(wp_after_bowler - wp_before)       (run-outs charged as if only the runs happened)
L        = sum_o p_next[o] * |wp(o) - wp_before|
LI       = L / mean(L over all deliveries);  floored at li_floor whenever we divide by it
pressure = LI * (1 + k * stakes)

Attribution
  batter : WPA_bat on every ball faced as striker, except wides (they belong to the bowler);
           a non-striker run-out goes to dismissed_player_id; a wicket on a wide goes to the
           dismissed batter.
  bowler : WPA_bowl on every ball they bowl.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Paths
from .cricket import OUTCOMES

log = logging.getLogger("superr.balls")


def compute_balls(st: pd.DataFrame, situ: pd.DataFrame, stakes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    assert (st["ball_seq"].to_numpy() == situ["ball_seq"].to_numpy()).all(), "stage 1/2 misaligned"
    b = st.copy()
    for c in ["wp_before", "wp_after", "wp_after_bowler", "bowler_counterfactual", "post_kind"]:
        b[c] = situ[c].to_numpy()
    wpb = b["wp_before"].to_numpy()
    b["wpa_bat"] = b["wp_after"].to_numpy() - wpb
    b["wpa_bowl"] = -(b["wp_after_bowler"].to_numpy() - wpb)

    P = situ[[f"p_{o}" for o in OUTCOMES]].to_numpy()
    W = situ[[f"wp_{o}" for o in OUTCOMES]].to_numpy()
    b["L"] = (P * np.abs(W - wpb[:, None])).sum(axis=1)
    meanL = b["L"].mean()
    b["LI"] = b["L"] / meanL
    floor = cfg["attribution"].get("li_floor", 0.1)
    b["LI_div"] = np.maximum(b["LI"], floor)

    # stakes of the batting and bowling side in this match
    sk = stakes.set_index(["match_id", "team_id"])["stakes"]
    b["stakes_bat"] = sk.reindex(pd.MultiIndex.from_arrays([b["match_id"], b["batting_team_id"]])).to_numpy()
    b["stakes_bowl"] = sk.reindex(pd.MultiIndex.from_arrays([b["match_id"], b["bowling_team_id"]])).to_numpy()
    for c in ("stakes_bat", "stakes_bowl"):
        b[c] = b[c].fillna(0.0)
    k = cfg["pressure"].get("k", 1.0)
    b["pressure_bat"] = b["LI"] * (1 + k * b["stakes_bat"])
    b["pressure_bowl"] = b["LI"] * (1 + k * b["stakes_bowl"])

    # attribution
    dis = b["dismissed_player_id"]
    wk = b["is_wicket"].to_numpy()
    wide = b["is_wide"].to_numpy()
    ns_out = wk & (dis.to_numpy() == b["non_striker_id"].to_numpy())
    bat_pid = np.where(wide, np.where(wk, dis.fillna(-1).to_numpy(), -1),
                       np.where(ns_out, dis.fillna(-1).to_numpy(), b["striker_id"].to_numpy()))
    b["bat_pid"] = bat_pid.astype(np.int64)  # -1 = no batter credited (a wide)
    b["bowl_pid"] = b["bowler_id"].astype(np.int64)
    b["wpa_bat_li"] = b["wpa_bat"] / b["LI_div"]
    b["wpa_bowl_li"] = b["wpa_bowl"] / b["LI_div"]
    log.info("mean L = %.5f; LI quantiles 50/90/99/99.9%% = %s", meanL,
             np.round(np.quantile(b["LI"], [0.5, 0.9, 0.99, 0.999]), 2).tolist())
    return b


def run(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    st = io.load("states", paths)
    situ = io.load("situations", paths)
    stakes = io.load("stakes", paths)
    b = compute_balls(st, situ, stakes, cfg)
    keep = [c for c in b.columns if not c.startswith("p_") or c in ("phase",)]
    io.save(b[keep], "balls", paths)
