import numpy as np
import pandas as pd

from conftest import toy_bbb
from superr.balls import compute_balls
from superr.cricket import OUTCOMES
from superr.states import build_states
from superr.stints import batter_stints, bowler_stints


def _states(cfg):
    b, m = toy_bbb()
    return build_states(b, m, cfg)


def test_state_fields(cfg):
    st = _states(cfg)
    i1 = st[st["innings"] == 1].reset_index(drop=True)
    # cumulative score before each ball; the Overstep is not a legal ball and gives a free hit
    assert i1.loc[4, "runs"] == 1 + 0 + 4 + 1      # after 1, 0, 4, no-ball(1)
    assert i1.loc[4, "free_hit"]
    assert i1.loc[4, "legal_balls"] == 3
    # non-striker innings carried over from his own balls on strike
    assert i1.loc[1, "non_striker_runs"] == 1 and i1.loc[1, "non_striker_balls"] == 1
    # partnership resets after the wicket
    after_w = i1[i1["wickets"] == 1].iloc[0]
    assert after_w["partnership_runs"] == 0
    # innings 2 target = innings-1 total + 1, quota from revised overs (12 balls)
    i2 = st[st["innings"] == 2]
    total1 = int(st[st["innings"] == 1]["ball_total"].sum())
    assert (i2["target"] == total1 + 1).all() and (i2["max_balls"] == 12).all()
    # transition consistency: post-ball state == next pre-ball state
    nxt = st.groupby(["match_id", "innings"]).shift(-1)
    ok = st["is_last_ball"] | ((st["runs"] + st["ball_total"] == nxt["runs"]) &
                               (st["legal_balls"] + st["is_legal"] == nxt["legal_balls"]))
    assert ok.all()


def _fake_situations(st):
    n = len(st)
    rng = np.random.default_rng(0)
    situ = pd.DataFrame({"ball_seq": st["ball_seq"].to_numpy(), "match_id": st["match_id"].to_numpy()})
    wpb = rng.uniform(0.2, 0.8, n)
    situ["wp_before"] = wpb
    situ["wp_after"] = np.r_[wpb[1:], 0.5]
    situ["wp_after_bowler"] = situ["wp_after"]
    situ["bowler_counterfactual"] = False
    situ["post_kind"] = 0
    for o in OUTCOMES:
        situ[f"wp_{o}"] = np.clip(wpb + rng.normal(0, 0.05, n), 0, 1)
        situ[f"p_{o}"] = 1 / len(OUTCOMES)
    return situ


def test_attribution_rules(cfg):
    st = _states(cfg)
    situ = _fake_situations(st)
    stakes = pd.DataFrame({"match_id": [1, 1], "team_id": [1, 2], "stakes": [0.3, 0.6]})
    b = compute_balls(st, situ, stakes, cfg)
    wide = b[b["is_wide"]]
    assert (wide["bat_pid"] == -1).all()                       # wides belong to the bowler
    ro = b[b["dismissal_type"] == "RunOut"].iloc[0]
    assert ro["bat_pid"] == 10 and ro["striker_id"] == 12      # non-striker run-out -> dismissed batter
    assert (b["bowl_pid"] == b["bowler_id"]).all()
    np.testing.assert_allclose(b["wpa_bowl"], -(b["wp_after_bowler"] - b["wp_before"]))
    assert np.isclose(b["LI"].mean(), 1.0)
    assert (b.loc[b["innings"] == 1, "stakes_bat"] == 0.3).all()


def test_stints(cfg):
    st = _states(cfg)
    b = compute_balls(st, _fake_situations(st), pd.DataFrame({"match_id": [1, 1], "team_id": [1, 2], "stakes": [0, 0]}), cfg)
    bat = batter_stints(b)
    r = bat.set_index(["innings", "player_id"])
    assert r.loc[(1, 11), "end_reason"] == "out"
    assert r.loc[(1, 10), "end_reason"] == "out"               # run out at the non-striker's end
    assert r.loc[(1, 13), "n_balls"] == 0                       # never faced (non-striker only)
    bowl = bowler_stints(b)
    assert set(bowl[bowl["innings"] == 1]["overid"]) == {1, 2}
    assert bowl.loc[(bowl["innings"] == 1) & (bowl["overid"] == 1), "n_balls"].iloc[0] == 7  # incl. no-ball
