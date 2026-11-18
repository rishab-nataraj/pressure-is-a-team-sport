"""What we ask Jev, and exactly what Jev is allowed to see.

Principle 1 (spec): Jev never sees who is playing - no team, player or venue names, no dates,
no match IDs. States carry only score/situation numbers; stakes contexts label teams A, B, C...
Principle 2: ask for raw ingredients (wp, next-ball distribution, p_advance), never a blended
"pressure" score; those are combined in our code.
"""
from __future__ import annotations

import math
import string

import numpy as np
import pandas as pd

from ..cricket import COARSE_DROP, OUTCOMES

# ---------------------------------------------------------------------------- call A / call B
WP_QUESTION = {
    "type": "noul",
    "instructions": (
        "The state describes a men's T20 cricket match just before the next delivery, from the "
        "batting side's point of view. Both teams are of average strength; there is no other "
        "information about who is playing. Will the batting side win this match?"
    ),
}

_CLASS_TEXT = {
    "dot": "legal delivery, no runs of any kind, no wicket",
    "one": "legal delivery, exactly 1 run (off the bat, byes or leg-byes), no wicket",
    "two": "legal delivery, exactly 2 runs, no wicket",
    "three": "legal delivery, exactly 3 runs, no wicket",
    "four": "legal delivery, 4 or 5 runs (usually a boundary four), no wicket",
    "six": "legal delivery, 6 or more runs (usually a six), no wicket",
    "wicket": "legal delivery on which a batter is dismissed",
    "wide": "the delivery is called a wide",
    "no_ball": "the delivery is called a no-ball",
}

PNEXT_QUESTION = {
    "type": "choice",
    "instructions": (
        "The state describes a men's T20 cricket match just before the next delivery, from the "
        "batting side's point of view. Both teams are of average strength. What will the outcome "
        "of the next delivery be?"
    ),
    "criteria": {k: _CLASS_TEXT[k] for k in OUTCOMES},
}

_FORMAT_TEXT = {"IPL": "IPL (franchise T20 league)", "T20I": "T20 International",
                "other T20": "other T20 (A-team / emerging tournament)"}


def _num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    xf = float(x)
    return int(xf) if xf.is_integer() else round(xf, 2)


def state_to_jev(row: dict, coarse: bool = False) -> dict:
    """Serialise one state row (cricket.STATE_COLS) to the identity-free JSON sent to Jev."""
    legal = row["legal_balls"]
    overs = f"{int(legal // 6)}.{int(legal % 6)}"
    s = {
        "competition_level": _FORMAT_TEXT.get(row["format"], row["format"]),
        "impact_player_rule": bool(row["impact_rule"]),
        "innings": int(row["innings"]),
        "score": f"{int(row['runs'])}/{int(row['wickets'])}",
        "runs": _num(row["runs"]),
        "wickets_lost": _num(row["wickets"]),
        "overs_bowled": overs,
        "legal_balls_bowled": _num(legal),
        "innings_ball_quota": _num(row["max_balls"]),
        "balls_left": _num(row["balls_left"]),
        "current_run_rate": round(6 * row["runs"] / legal, 2) if legal > 0 else None,
        "next_ball_is_free_hit": bool(row["free_hit"]),
        "venue_average_first_innings_score": _num(row["venue_par"]),
    }
    if int(row["innings"]) == 2:
        s["target"] = _num(row["target"])
        s["runs_needed"] = _num(row["runs_needed"])
        bl = row["balls_left"]
        s["required_run_rate"] = round(6 * row["runs_needed"] / bl, 2) if bl > 0 else None
    if not coarse:
        s["current_partnership"] = {"runs": _num(row["partnership_runs"]), "balls": _num(row["partnership_balls"])}
        s["striker"] = {"runs": _num(row["striker_runs"]), "balls_faced": _num(row["striker_balls"])}
        s["non_striker"] = {"runs": _num(row["non_striker_runs"]), "balls_faced": _num(row["non_striker_balls"])}
    return s


def states_to_jev(S: pd.DataFrame, coarse: bool = False) -> list[dict]:
    cols = [c for c in S.columns if not (coarse and c in COARSE_DROP)]
    return [state_to_jev(r, coarse) for r in S[cols].to_dict("records")]


def parse_wp(ans: dict) -> float:
    if ans.get("type") == "noul" or "noul" in ans:
        return float(ans["noul"])
    if "probability" in ans:  # gateway naming (AI SDK "boolean")
        return float(ans["probability"])
    raise ValueError(f"unexpected wp answer: {ans}")


def parse_pnext(ans: dict) -> np.ndarray:
    probs = ans.get("probabilities") or {}
    v = np.array([float(probs.get(k, 0.0)) for k in OUTCOMES])
    s = v.sum()
    return v / s if s > 0 else np.full(len(OUTCOMES), 1 / len(OUTCOMES))


# ------------------------------------------------------------------------------------- call C
STAKES_QUESTIONS = {
    "p_advance_if_win": {
        "type": "noul",
        "instructions": (
            "The state describes a cricket competition table before one match. Teams are anonymous "
            "letters. Suppose the focal team WINS this match and every other remaining result is "
            "unknown (each side equally likely to win). Will the focal team advance, i.e. finish in "
            "one of the qualifying places described in the state?"
        ),
    },
    "p_advance_if_lose": {
        "type": "noul",
        "instructions": (
            "The state describes a cricket competition table before one match. Teams are anonymous "
            "letters. Suppose the focal team LOSES this match and every other remaining result is "
            "unknown (each side equally likely to win). Will the focal team advance, i.e. finish in "
            "one of the qualifying places described in the state?"
        ),
    },
}

_COMP_TEXT = {"ipl": "franchise T20 league (league stage)", "tournament": "international T20 tournament"}


def _labels(n: int) -> list[str]:
    letters = list(string.ascii_uppercase)
    return letters[:n] if n <= 26 else [f"T{i + 1}" for i in range(n)]


def stakes_context_to_jev(ctx: dict) -> dict:
    """Anonymised standings for a league-type match: teams relabelled A, B, C... in table order."""
    st = sorted(ctx["standings"], key=lambda r: (-r["points"], -r["nrr"], r["team_id"]))
    lab = dict(zip([r["team_id"] for r in st], _labels(len(st))))
    a, b = ctx["this_match"]
    return {
        "competition_type": _COMP_TEXT.get(ctx["comp_kind"], ctx["comp_kind"]),
        "stage": "group / league stage" if ctx.get("phase") != "super4" else "Super 4 round robin",
        "teams_in_group": len(st),
        "qualifying_places": int(ctx["qualify"]),
        "advance_means": (f"finish in the top {int(ctx['qualify'])} of this table"
                          + (" (reach the playoffs)" if ctx["comp_kind"] == "ipl" else " (reach the next round)")),
        "points_for_win": 2, "points_for_no_result": 1, "tie_break": "points, then net run rate",
        "standings_before_match": [
            {"team": lab[r["team_id"]], "played": int(r["played"]), "won": int(r["won"]), "lost": int(r["lost"]),
             "no_result": int(r["no_result"]), "points": int(r["points"]), "net_run_rate": round(float(r["nrr"]), 3)}
            for r in st],
        "this_match": f"{lab[a]} v {lab[b]}",
        "focal_team": lab[a],
        "remaining_fixtures_after_this_match": [f"{lab[x]} v {lab[y]}" for x, y in ctx["remaining"]],
    }
