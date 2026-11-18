import numpy as np

from superr.standings import bilateral_stakes, league_stakes_mc


def test_bilateral_rules():
    # one-off match: winning it wins the series
    pw, pl = bilateral_stakes(1, 0, 0, 0)
    assert (pw, pl) == (1.0, 0.0)
    # 1-1 in a 3-match series: decider
    pw, pl = bilateral_stakes(3, 1, 1, 0)
    assert pw - pl == 1.0
    # 2-0 up in a 3-match series: dead rubber
    pw, pl = bilateral_stakes(3, 2, 0, 0)
    assert pw - pl == 0.0
    # game 1 of 3
    pw, pl = bilateral_stakes(3, 0, 0, 2)
    assert np.isclose(pw - pl, 0.5)


def _ctx(points, remaining, focal, opp, q=2):
    teams = list(range(len(points)))
    st = [dict(team_id=t, played=3, won=p // 2, lost=3 - p // 2, no_result=0, points=p, nrr=0.0) for t, p in zip(teams, points)]
    return dict(teams=teams, standings=st, remaining=remaining, this_match=(focal, opp), qualify=q)


def test_league_final_round_must_win():
    # 4 teams, top 2 qualify, last round. Team 0 on 4, team 1 on 4, team 2 on 2, team 3 on 2.
    # This match: 1 v 2. Other remaining match: 0 v 3.
    rng = np.random.default_rng(0)
    pw, pl = league_stakes_mc(_ctx([4, 4, 2, 2], [(0, 3)], 2, 1), 4000, 0.3, rng)
    assert pl == 0.0           # team 2 loses -> 2 points, cannot finish top 2
    assert 0.3 < pw < 0.9      # wins -> 4 points, ties decided on NRR / other result


def test_league_already_qualified():
    rng = np.random.default_rng(0)
    pw, pl = league_stakes_mc(_ctx([8, 2, 2, 0], [], 0, 3), 2000, 0.3, rng)
    assert pw == 1.0 and pl == 1.0
