import numpy as np
import pandas as pd

from conftest import make_state
from superr.cricket import (LIVE, OUTCOMES, STATE_COLS, TERMINAL, TRANSITION, apply_outcome, classify,
                            inn2_start, outcome_class, state_key)


def test_runs_and_strike_swap():
    S = make_state()
    one = apply_outcome(S, "one")
    assert one["runs"].iloc[0] == 151 and one["legal_balls"].iloc[0] == 115
    # single swaps strike: old non-striker (5 off 4) now on strike
    assert one["striker_runs"].iloc[0] == 5 and one["non_striker_runs"].iloc[0] == 31
    four = apply_outcome(S, "four")
    assert four["striker_runs"].iloc[0] == 34  # no swap on a boundary


def test_end_of_over_swap():
    S = make_state(legal_balls=113.0)
    dot = apply_outcome(S, "dot")  # 114th ball ends the over -> swap
    assert dot["striker_runs"].iloc[0] == 5


def test_wicket_resets_batter_and_partnership():
    S = make_state()
    w = apply_outcome(S, "wicket")
    assert w["wickets"].iloc[0] == 5
    assert w["striker_runs"].iloc[0] == 0 and w["striker_balls"].iloc[0] == 0
    assert w["partnership_runs"].iloc[0] == 0 and w["partnership_balls"].iloc[0] == 0


def test_wide_and_noball():
    S = make_state()
    wd = apply_outcome(S, "wide")
    assert wd["runs"].iloc[0] == 151 and wd["legal_balls"].iloc[0] == 114 and not wd["free_hit"].iloc[0]
    nb = apply_outcome(S, "no_ball")
    assert nb["runs"].iloc[0] == 151 and nb["legal_balls"].iloc[0] == 114 and nb["free_hit"].iloc[0]
    # free hit carries through a wide, cleared by a legal ball
    assert apply_outcome(nb, "wide")["free_hit"].iloc[0]
    assert not apply_outcome(nb, "dot")["free_hit"].iloc[0]


def test_terminal_rules_innings2():
    S = make_state(legal_balls=119.0, runs=158.0)  # need 2 off the last ball
    vals = {}
    for o in OUTCOMES:
        k, v = classify(apply_outcome(S, o))
        vals[o] = (int(k[0]), v[0])
    assert vals["two"] == (TERMINAL, 1.0) and vals["six"] == (TERMINAL, 1.0)
    assert vals["one"] == (TERMINAL, 0.5)          # tie
    assert vals["dot"] == (TERMINAL, 0.0) and vals["wicket"] == (TERMINAL, 0.0)
    assert vals["wide"][0] == LIVE                 # still one ball to come, 1 needed


def test_transition_innings1():
    S = make_state(innings=1, target=np.nan, legal_balls=119.0, runs=180.0)
    N = apply_outcome(S, "six")
    k, _ = classify(N)
    assert k[0] == TRANSITION
    Z = inn2_start(N)
    assert Z["innings"].iloc[0] == 2 and Z["target"].iloc[0] == 187 and Z["runs_needed"].iloc[0] == 187
    assert Z["balls_left"].iloc[0] == 120


def test_state_key_ignores_aux_and_is_stable():
    S = make_state()
    T = S.copy()
    T["inn2_start_max_balls"] = 60.0
    assert state_key(S)[0] == state_key(T)[0]
    T["runs"] = 151.0
    T["runs_needed"] = 9.0
    assert state_key(S)[0] != state_key(T)[0]


def test_outcome_class():
    et = pd.Series([None, "wide", "Overstep", None, None, "legbyes", "other"])
    wk = pd.Series([False, False, False, True, False, False, False])
    tot = pd.Series([0, 1, 1, 0, 5, 1, 2])
    cls = outcome_class(et, wk, tot, {"wide"}, {"Overstep", "other"}).tolist()
    assert cls == ["dot", "wide", "no_ball", "wicket", "four", "one", "no_ball"]
