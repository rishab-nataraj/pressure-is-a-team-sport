"""Identity-free match state, next-ball outcomes and terminal rules (spec sections 1-2).

A *state* is the situation before a delivery, from the batting side's point of view. It holds
no team, player, venue or date identity, so it can be sent to Jev as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fields that define a state. These, and only these, are sent to Jev / the wp model.
STATE_COLS = [
    "innings", "runs", "wickets", "legal_balls", "max_balls", "target", "balls_left",
    "runs_needed", "free_hit", "partnership_runs", "partnership_balls",
    "striker_runs", "striker_balls", "non_striker_runs", "non_striker_balls",
    "format", "impact_rule", "venue_par",
]
COARSE_DROP = ["partnership_runs", "partnership_balls", "striker_runs", "striker_balls",
               "non_striker_runs", "non_striker_balls"]
AUX_COLS = ["inn2_start_max_balls"]  # carried alongside a state, never sent anywhere

OUTCOMES = ["dot", "one", "two", "three", "four", "six", "wicket", "wide", "no_ball"]

# Change to state for each hypothetical next-ball outcome (spec section 2 table).
OUTCOME_SPEC = {
    "dot":     dict(bat=0, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "one":     dict(bat=1, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "two":     dict(bat=2, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "three":   dict(bat=3, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "four":    dict(bat=4, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "six":     dict(bat=6, extra=0, legal=1, noball=0, wide=0, wicket=0),
    "wicket":  dict(bat=0, extra=0, legal=1, noball=0, wide=0, wicket=1),
    "wide":    dict(bat=0, extra=1, legal=0, noball=0, wide=1, wicket=0),
    "no_ball": dict(bat=0, extra=1, legal=0, noball=1, wide=0, wicket=0),
}

LIVE, TERMINAL, TRANSITION = 0, 1, 2


def _arr(x, n, dtype=float):
    a = np.asarray(x, dtype=dtype)
    return np.full(n, a) if a.ndim == 0 else a


def derive(S: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived fields (balls_left, runs_needed) in place and return S."""
    S["balls_left"] = S["max_balls"] - S["legal_balls"]
    inn2 = S["innings"].to_numpy() == 2
    S["runs_needed"] = np.where(inn2, S["target"] - S["runs"], np.nan)
    return S


def advance(S: pd.DataFrame, bat, extra, legal, noball, wide, wicket,
            striker_out=True, ran=None) -> pd.DataFrame:
    """State after one delivery, vectorised over rows of ``S``.

    bat     runs off the bat;  extra: extras (wide/no-ball penalty, byes, leg-byes, penalties)
    legal   1 if the ball counts towards the over; noball/wide flags; wicket 1 if a batter is out
    striker_out  True if the dismissed batter was the striker (else the non-striker)
    ran     runs physically run (strike swaps when odd); default = bat runs that were not 4/6
    """
    n = len(S)
    bat, extra = _arr(bat, n), _arr(extra, n)
    legal, noball, wide, wicket = (_arr(v, n).astype(bool) for v in (legal, noball, wide, wicket))
    striker_out = _arr(striker_out, n).astype(bool)
    if ran is None:
        ran = np.where(np.isin(bat, (4, 6)), 0, bat)
    ran = _arr(ran, n)

    N = S.copy()
    tot = bat + extra
    N["runs"] = S["runs"].to_numpy() + tot
    N["wickets"] = S["wickets"].to_numpy() + wicket
    N["legal_balls"] = S["legal_balls"].to_numpy() + legal
    faced = ~wide
    N["partnership_runs"] = np.where(wicket, 0, S["partnership_runs"].to_numpy() + tot)
    N["partnership_balls"] = np.where(wicket, 0, S["partnership_balls"].to_numpy() + legal)
    sr = S["striker_runs"].to_numpy() + bat
    sb = S["striker_balls"].to_numpy() + faced
    nr = S["non_striker_runs"].to_numpy().astype(float).copy()
    nb = S["non_striker_balls"].to_numpy().astype(float).copy()
    # dismissed batter replaced by a new batter on 0 (0)
    sr = np.where(wicket & striker_out, 0, sr)
    sb = np.where(wicket & striker_out, 0, sb)
    nr = np.where(wicket & ~striker_out, 0, nr)
    nb = np.where(wicket & ~striker_out, 0, nb)
    # strike swaps on odd runs run, and again at the end of an over
    swap = (ran.astype(int) % 2) == 1
    end_over = legal & (N["legal_balls"].to_numpy() % 6 == 0)
    swap = swap ^ end_over
    N["striker_runs"] = np.where(swap, nr, sr)
    N["striker_balls"] = np.where(swap, nb, sb)
    N["non_striker_runs"] = np.where(swap, sr, nr)
    N["non_striker_balls"] = np.where(swap, sb, nb)
    fh = S["free_hit"].to_numpy().astype(bool)
    N["free_hit"] = np.where(noball, True, np.where(legal, False, fh))  # carries through wides
    return derive(N)


def apply_outcome(S: pd.DataFrame, outcome: str) -> pd.DataFrame:
    spec = OUTCOME_SPEC[outcome]
    return advance(S, spec["bat"], spec["extra"], spec["legal"], spec["noball"], spec["wide"],
                   spec["wicket"], striker_out=True)


def innings_over(N: pd.DataFrame) -> np.ndarray:
    inn2 = N["innings"].to_numpy() == 2
    done = (N["wickets"].to_numpy() >= 10) | (N["legal_balls"].to_numpy() >= N["max_balls"].to_numpy())
    chased = inn2 & (N["runs"].to_numpy() >= np.nan_to_num(N["target"].to_numpy(), nan=np.inf))
    return done | chased


def classify(N: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Terminal rules (spec section 2) for post-ball states ``N``.

    Returns (kind, value): kind LIVE (query wp of N), TERMINAL (value is the batting side's wp),
    or TRANSITION (end of innings 1: value = 1 - wp(innings-2 start), see inn2_start).
    """
    inn = N["innings"].to_numpy()
    runs, tgt = N["runs"].to_numpy(), N["target"].to_numpy()
    done = (N["wickets"].to_numpy() >= 10) | (N["legal_balls"].to_numpy() >= N["max_balls"].to_numpy())
    kind = np.full(len(N), LIVE, dtype=np.int8)
    value = np.full(len(N), np.nan)
    i2 = inn == 2
    chased = i2 & (runs >= tgt)
    tie = i2 & ~chased & done & (runs == tgt - 1)
    lost = i2 & ~chased & done & (runs < tgt - 1)
    kind[chased | tie | lost] = TERMINAL
    value[chased] = 1.0
    value[tie] = 0.5
    value[lost] = 0.0
    kind[(inn == 1) & done] = TRANSITION
    return kind, value


def inn2_start(N: pd.DataFrame) -> pd.DataFrame:
    """Innings-2 starting state implied by a completed innings-1 state (target = total + 1)."""
    Z = N.copy()
    Z["innings"] = 2
    Z["target"] = N["runs"].to_numpy() + 1
    Z["runs"] = 0
    Z["wickets"] = 0
    Z["legal_balls"] = 0
    Z["max_balls"] = N["inn2_start_max_balls"].to_numpy()
    Z["free_hit"] = False
    for c in ("partnership_runs", "partnership_balls", "striker_runs", "striker_balls",
              "non_striker_runs", "non_striker_balls"):
        Z[c] = 0
    return derive(Z)


def state_key(S: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    """Stable 64-bit hash of the state fields (for de-duplication)."""
    cols = cols or STATE_COLS
    X = S[cols].copy()
    X["venue_par"] = X["venue_par"].round(1)
    for c in cols:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)
    X["target"] = X["target"].fillna(-1)
    X["runs_needed"] = X["runs_needed"].fillna(-9999)
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def outcome_class(extras_type: pd.Series, is_wicket: pd.Series, ball_total: pd.Series,
                  wide_set, noball_set) -> pd.Series:
    """Map a real delivery to one of OUTCOMES (for the empirical p_next table / validation)."""
    et = extras_type.fillna("")
    wide = et.isin(wide_set)
    nob = et.isin(noball_set)
    r = ball_total.to_numpy()
    legal_cls = np.select([r <= 0, r == 1, r == 2, r == 3, r <= 5], ["dot", "one", "two", "three", "four"],
                          default="six")
    cls = np.where(wide, "wide", np.where(nob, "no_ball", np.where(is_wicket.to_numpy(), "wicket", legal_cls)))
    return pd.Series(cls, index=extras_type.index)
