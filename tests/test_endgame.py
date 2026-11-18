import numpy as np
import pandas as pd

from conftest import make_state
from superr.cricket import OUTCOMES
from superr.endgame import solve


class FixedPNext:
    """Every state: the same next-ball distribution."""

    def __init__(self, p):
        self.p = np.array([p[o] for o in OUTCOMES])

    def p_next(self, S, match_id=None):
        return np.tile(self.p, (len(S), 1))


class ConstWP:
    def __init__(self, v):
        self.v = v

    def wp(self, S, match_id=None):
        return np.full(len(S), self.v)


P = {"dot": 0.3, "one": 0.3, "two": 0.1, "three": 0.0, "four": 0.15, "six": 0.05, "wicket": 0.1,
     "wide": 0.0, "no_ball": 0.0}


def test_last_ball_two_needed():
    S = make_state(legal_balls=119.0, runs=158.0)     # need 2 off 1
    out, done = solve(S, np.array([1]), np.array([0.5]), 6, ConstWP(0.5), FixedPNext(P))
    expected = P["two"] + P["four"] + P["six"] + 0.5 * P["one"]
    assert done[0] and np.isclose(out[0], expected)


def test_two_balls_backward_induction():
    S = make_state(legal_balls=118.0, runs=158.0)     # need 2 off 2, 4 down
    out, _ = solve(S, np.array([1]), np.array([0.5]), 6, ConstWP(0.5), FixedPNext(P))
    # value after each legal outcome with one ball left
    runs = {"dot": 0, "one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "wicket": 0}

    def last(need):  # one ball left; a 0-run ball (dot or wicket) with 1 needed is a tie
        win = sum(P[o] for o, r in runs.items() if r >= need)
        tie = sum(P[o] for o, r in runs.items() if r == need - 1)
        return win + 0.5 * tie
    exp = (P["dot"] * last(2) + P["one"] * last(1) + P["two"] * 1 + P["four"] + P["six"] + P["wicket"] * last(2))
    assert np.isclose(out[0], exp)


def test_innings1_boundary_uses_chase_start():
    S = make_state(innings=1, target=np.nan, legal_balls=119.0, runs=180.0)
    out, done = solve(S, np.array([1]), np.array([0.5]), 6, ConstWP(0.3), FixedPNext(P))
    assert done[0] and np.isclose(out[0], 0.7)     # every end state: 1 - 0.3


def test_outside_window_untouched():
    S = make_state(legal_balls=60.0)
    out, done = solve(S, np.array([1]), np.array([0.42]), 6, ConstWP(0.5), FixedPNext(P))
    assert not done[0] and out[0] == 0.42
