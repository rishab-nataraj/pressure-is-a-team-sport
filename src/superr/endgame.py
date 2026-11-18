"""End-game solver: exact backward induction for the last K legal balls of each innings.

A smooth wp model cannot resolve the final over: near the end of an innings, win probability is
a sharp function of (runs, wickets, balls left), and a model with large leaves comes out flat
there. Flat values make the hypothetical next states identical (leverage ~ 0) while the real
outcome still moves wp, so |WPA| / LI explodes on exactly the balls that matter most.

For the last K legal balls we therefore compute wp by dynamic programming over a grid of
(balls_left, runs, wickets), using the provider's next-ball probabilities and exact boundary values:

  innings 2 : runs >= target -> 1;  innings over and runs == target - 1 -> 0.5;  otherwise 0
  innings 1 : innings over -> 1 - wp2(innings-2 start with target = runs + 1)   (the wp model)

  wp(b, r, w) = sum_{legal o} p_o * wp(b - 1, r + runs_o, w + wkt_o)
                + (p_wide + p_noball) * wp(b, r + 1, w)

Every real and hypothetical state inside the window is then read off the same table, so WPA and
leverage are consistent by construction. Batter- and partnership-level fields do not enter (the
empirical wp model does not use them either).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .cricket import OUTCOME_SPEC, OUTCOMES, STATE_COLS, inn2_start

log = logging.getLogger("superr.endgame")

LEGAL = [o for o in OUTCOMES if o not in ("wide", "no_ball")]
LEGAL_RUNS = np.array([OUTCOME_SPEC[o]["bat"] for o in LEGAL])
LEGAL_WKT = np.array([OUTCOME_SPEC[o]["wicket"] for o in LEGAL])
I_WIDE, I_NB = OUTCOMES.index("wide"), OUTCOMES.index("no_ball")
I_LEGAL = [OUTCOMES.index(o) for o in LEGAL]


def solve(S_all: pd.DataFrame, mid_all: np.ndarray, wp_all: np.ndarray, K: int, wp_provider, pnext_provider,
          extra_runs: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return (wp_all with window states replaced, mask of replaced rows)."""
    out = wp_all.copy()
    bl = S_all["balls_left"].to_numpy()
    inn = S_all["innings"].to_numpy()
    win = (bl >= 1) & (bl <= K) & (S_all["wickets"].to_numpy() < 10)
    win &= ~((inn == 2) & (S_all["runs"].to_numpy() >= S_all["target"].to_numpy()))
    if not win.any():
        return out, win
    idx_win = np.where(win)[0]
    W = S_all.iloc[idx_win]
    keys = pd.DataFrame({"m": mid_all[idx_win], "i": inn[idx_win]})
    done = np.zeros(len(S_all), bool)
    n_groups = 0
    for (m, i), g in keys.groupby(["m", "i"]).indices.items():
        rows = idx_win[g]
        Sg = S_all.iloc[rows]
        ctx = Sg.iloc[0]
        r_lo = int(Sg["runs"].min())
        r_hi = int(Sg["runs"].max()) + 6 * K + extra_runs
        R = np.arange(r_lo, r_hi + 2)                   # one spare row for the r + 1 boundary
        nR, nW = len(R), 11
        V = np.zeros((K + 1, nR, nW))
        # ---- boundary: innings over (b = 0 or w = 10)
        if i == 2:
            tgt = float(ctx["target"])
            term = np.where(R >= tgt, 1.0, np.where(R == tgt - 1, 0.5, 0.0))
        else:
            E = pd.DataFrame([ctx[STATE_COLS + ["inn2_start_max_balls"]]] * nR).reset_index(drop=True)
            E["runs"] = R.astype(float)
            term = 1.0 - wp_provider.wp(inn2_start(E)[STATE_COLS], np.full(nR, m))
        V[0, :, :] = term[:, None]
        for b in range(1, K + 1):
            V[b, :, 10] = term
        # ---- next-ball probabilities for (b, w) cells (the empirical table depends on w, phase, format)
        G = pd.DataFrame([ctx[STATE_COLS]] * (K * 10)).reset_index(drop=True)
        bb, ww = np.meshgrid(np.arange(1, K + 1), np.arange(10), indexing="ij")
        G["balls_left"] = bb.ravel().astype(float)
        G["legal_balls"] = G["max_balls"] - G["balls_left"]
        G["wickets"] = ww.ravel().astype(float)
        P = pnext_provider.p_next(G, np.full(len(G), m)).reshape(K, 10, len(OUTCOMES))
        # ---- backward induction
        for b in range(1, K + 1):
            pl = P[b - 1][:, I_LEGAL]                     # (10, 7)
            pe = P[b - 1][:, I_WIDE] + P[b - 1][:, I_NB]  # (10,)
            prev = V[b - 1]
            legal_part = np.zeros((nR, 10))
            for j in range(len(LEGAL)):
                rr = np.minimum(np.arange(nR) + LEGAL_RUNS[j], nR - 1)
                wcol = np.arange(10) + LEGAL_WKT[j]
                legal_part += pl[None, :, j] * prev[rr][:, wcol]
            cur = V[b]
            if i == 2:
                done_r = R >= float(ctx["target"])
            else:
                done_r = np.zeros(nR, bool)
            cur[nR - 1, :10] = legal_part[nR - 1] / np.maximum(pl.sum(axis=1), 1e-12)
            for r in range(nR - 2, -1, -1):
                if done_r[r]:
                    cur[r, :10] = 1.0
                else:
                    cur[r, :10] = legal_part[r] + pe * cur[r + 1, :10]
            if i == 2:
                cur[done_r, :] = 1.0
        # ---- read off every window state of this (match, innings)
        b_q = Sg["balls_left"].to_numpy().astype(int)
        r_q = Sg["runs"].to_numpy().astype(int) - r_lo
        w_q = Sg["wickets"].to_numpy().astype(int)
        ok = (r_q >= 0) & (r_q < nR)
        out[rows[ok]] = V[b_q[ok], r_q[ok], w_q[ok]]
        done[rows[ok]] = True
        n_groups += 1
    log.info("end-game solver: %d states in %d innings windows (last %d balls) replaced; mean |change| %.4f",
             int(done.sum()), n_groups, K, float(np.abs(out[done] - wp_all[done]).mean()) if done.any() else 0.0)
    return np.clip(out, 1e-4, 1 - 1e-4), done
