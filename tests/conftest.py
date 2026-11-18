import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superr.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


def make_state(**kw) -> pd.DataFrame:
    base = dict(innings=2, runs=150.0, wickets=4.0, legal_balls=114.0, max_balls=120.0, target=160.0,
                balls_left=6.0, runs_needed=10.0, free_hit=False, partnership_runs=20.0, partnership_balls=12.0,
                striker_runs=30.0, striker_balls=20.0, non_striker_runs=5.0, non_striker_balls=4.0,
                format="IPL", impact_rule=True, venue_par=170.0, inn2_start_max_balls=120.0)
    base.update(kw)
    S = pd.DataFrame([base])
    S["balls_left"] = S["max_balls"] - S["legal_balls"]
    S["runs_needed"] = np.where(S["innings"] == 2, S["target"] - S["runs"], np.nan)
    return S


def toy_bbb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One tiny match: innings 1 of 2 overs (12 legal balls), innings 2 chase. Ball-level rows only."""
    rows = []

    def ball(inn, over, obn, ibn, striker, ns, bowler, runs=0, extras_type=None, e1=0, wk=False,
             dtype=None, dpid=None, bt=1, bw=2):
        rows.append(dict(match_id=1, innings=inn, overid=over, over_ball_num=obn, innings_ball_num=ibn,
                         batting_team_id=bt, bowling_team_id=bw, striker_id=striker, striker_name=f"p{striker}",
                         non_striker_id=ns, non_striker_name=f"p{ns}", bowler_id=bowler, bowler_name=f"p{bowler}",
                         runs=runs, is_four=runs == 4, is_six=runs == 6, is_wicket=wk, dismissal_type=dtype,
                         dismissed_player_id=dpid, extras_type=extras_type, extras1_runs=e1, extras_2_runs=0))
    # innings 1: batters 10, 11; bowler 20 then 21
    ibn = 1
    seq = [(10, 11, 1), (11, 10, 0), (11, 10, 4), (11, 10, 0, "Overstep", 1), (11, 10, 6), (11, 10, 1)]
    obn = 1
    for t in seq:
        s, n, r = t[:3]
        et = t[3] if len(t) > 3 else None
        e1 = t[4] if len(t) > 4 else 0
        ball(1, 1, obn, ibn, s, n, 20, runs=r, extras_type=et, e1=e1)
        obn += 1
        ibn += 1
    ball(1, 1, obn, ibn, 10, 11, 20); ibn += 1                              # 6th legal ball of over 1
    # over 2: wide, wicket (striker 11 caught), new batter 12, run-out of non-striker 10
    obn = 1
    ball(1, 2, obn, ibn, 11, 10, 21, extras_type="wide", e1=1); obn += 1; ibn += 1
    ball(1, 2, obn, ibn, 11, 10, 21, wk=True, dtype="Caught", dpid=11); obn += 1; ibn += 1
    ball(1, 2, obn, ibn, 12, 10, 21, runs=1, wk=True, dtype="RunOut", dpid=10); obn += 1; ibn += 1
    for _ in range(4):
        ball(1, 2, obn, ibn, 12, 13, 21); obn += 1; ibn += 1
    # innings 2: chase of 14 off 12 balls; batters 30, 31
    ibn = 1
    for o in (1, 2):
        for k in range(6):
            ball(2, o, k + 1, ibn, 30, 31, 40, runs=1 if k % 2 else 0, bt=2, bw=1)
            ibn += 1
    b = pd.DataFrame(rows)
    m = pd.DataFrame([dict(match_id=1, comp_id=99, match_number=1, match_date=pd.Timestamp("2025-04-01"), stage="League",
                           ko_label="League", venue_id=5, format="IPL", impact_rule=True, first_inn_revised_overs=12.0,
                           second_inn_revised_overs=12.0, revised_target=np.nan, win_team_id=1.0, is_tie=False,
                           comp_kind="ipl", season_year=2025, excluded=False)])
    return b, m
