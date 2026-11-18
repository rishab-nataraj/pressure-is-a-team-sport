import numpy as np
import pandas as pd
import pytest

from superr.model import fit, fit_pt, prepare, ridge_one_shot, to_score


def synthetic_stints(n_players=60, n_teams=6, n_stints=6000, seed=0, c_sd=0.02, tau_sd=0.01):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, n_players, n_stints)
    role = np.where(pid % 2 == 0, "bat", "bowl")
    # each player mostly plays for a home team, sometimes for another (movers identify T vs P)
    home = pid % n_teams
    team = np.where(rng.random(n_stints) < 0.7, home, rng.integers(0, n_teams, n_stints))
    opp = (team + rng.integers(1, n_teams, n_stints)) % n_teams
    n = rng.integers(4, 30, n_stints)
    mean_li = rng.gamma(2.0, 0.6, n_stints)
    stakes = rng.uniform(0, 1, n_stints)
    beta = rng.normal(0, 0.01, n_players)
    c = rng.normal(0, c_sd, n_players)
    tau = rng.normal(0, tau_sd, n_teams)
    q = rng.normal(0, 0.005, n_teams)
    h = np.clip(mean_li * (1 + stakes) - 1, 0, 3)
    y = beta[pid] - q[opp] + h * (c[pid] + tau[team] - tau[opp]) + rng.normal(0, 0.05, n_stints) / np.sqrt(n)
    st = pd.DataFrame({"player_id": pid, "role": role, "team_canon": [f"T{t}" for t in team],
                       "opp_canon": [f"T{t}" for t in opp], "n_balls": n, "sum_wpa_li": y * n,
                       "mean_li": mean_li, "stakes": stakes, "match_id": np.arange(n_stints) // 2,
                       "match_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 900, n_stints), "D")})
    return st, c, tau


def test_recovers_planted_effects():
    st, c_true, tau_true = synthetic_stints()
    D = prepare(st, half_life=None)
    F = fit(D, 30, 30, 30, 30, tol=1e-9, iters=20000)
    assert F.converged
    c_hat = pd.Series(F.c, index=D.keys["player_id"]).reindex(range(len(c_true))).to_numpy()
    assert np.corrcoef(c_true, c_hat)[0, 1] > 0.9
    tau_hat = pd.Series(F.tau, index=D.teams).reindex([f"T{t}" for t in range(len(tau_true))]).to_numpy()
    assert np.corrcoef(tau_true, tau_hat)[0, 1] > 0.8


def test_loop_matches_one_shot_ridge():
    st, _, _ = synthetic_stints(seed=1)
    D = prepare(st, split_weights=False)
    F = fit(D, 50, 300, 100, 500, tol=1e-13, iters=20000)
    R = ridge_one_shot(D, 50, 300, 100, 500)
    for a, b in ((F.beta, R.beta), (F.c, R.c), (F.q, R.q), (F.tau, R.tau), (F.mu, R.mu)):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_start_independence():
    st, _, _ = synthetic_stints(seed=2)
    D = prepare(st)
    a = fit(D, 30, 300, 30, 300, seed=0, tol=1e-12, iters=20000)
    b = fit(D, 30, 300, 30, 300, seed=7, tol=1e-12, iters=20000)
    np.testing.assert_allclose(a.c, b.c, atol=1e-8)


def test_shrinkage_toward_zero_for_thin_samples():
    st, _, _ = synthetic_stints(seed=3)
    D = prepare(st)
    weak = fit(D, 30, 1e9, 30, 1e9)
    assert np.abs(weak.c).max() < 1e-6 and np.abs(weak.tau).max() < 1e-6


def test_reference_signature_and_scores():
    st, _, _ = synthetic_stints(seed=4)
    players, teams, mu, it = fit_pt(st, 30, 300, 30, 300)
    assert {"player_id", "role", "beta", "c", "exposure", "pressure_stints"} <= set(players.columns)
    assert {"team_canon", "tau", "q"} <= set(teams.columns)
    s = to_score(players["c"].to_numpy())
    assert s.min() >= 0 and s.max() <= 100
    assert to_score(np.zeros(5))[0] == pytest.approx(50.0)


def test_sample_weight_zero_equals_dropping():
    st, _, _ = synthetic_stints(seed=5)
    keep = st["match_id"] % 3 != 0
    D = prepare(st, half_life=None)
    Fw = fit(D, 30, 300, 30, 300, sample_weight=keep.to_numpy().astype(float), tol=1e-12, iters=20000)
    D2 = prepare(st[keep], half_life=None)
    F2 = fit(D2, 30, 300, 30, 300, tol=1e-12, iters=20000)
    a = pd.Series(Fw.c, index=pd.MultiIndex.from_frame(D.keys))
    b = pd.Series(F2.c, index=pd.MultiIndex.from_frame(D2.keys))
    np.testing.assert_allclose(a.reindex(b.index).to_numpy(), b.to_numpy(), atol=1e-9)
