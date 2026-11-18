"""Jev-backed providers for calls A (wp), B (next-ball distribution) and C (stakes)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..cricket import COARSE_DROP, STATE_COLS, state_key
from ..jev import JevClient, prompts
from ..standings import bilateral_stakes

log = logging.getLogger("superr.jev_provider")


class JevSituation:
    """wp and p_next from Jev. States are de-duplicated before asking; match_id is never sent."""

    def __init__(self, client: JevClient, coarse: bool = False):
        self.client = client
        self.coarse = coarse

    @property
    def key_cols(self) -> list[str]:
        return [c for c in STATE_COLS if not (self.coarse and c in COARSE_DROP)]

    def _ask_unique(self, S: pd.DataFrame, questions: dict, desc: str) -> tuple[np.ndarray, list[dict]]:
        keys = state_key(S, self.key_cols)
        uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        U = S.iloc[first]
        payload = prompts.states_to_jev(U[STATE_COLS], coarse=self.coarse)
        answers = self.client.ask([(s, questions) for s in payload], desc=desc)
        return inv, answers

    def warm(self, S: pd.DataFrame) -> None:
        """Ask wp and p_next together for the real states (one request per state)."""
        self._ask_unique(S, {"wp": prompts.WP_QUESTION, "p_next": prompts.PNEXT_QUESTION}, "jev A+B real")

    def wp(self, S: pd.DataFrame, match_id=None) -> np.ndarray:
        inv, answers = self._ask_unique(S, {"wp": prompts.WP_QUESTION}, "jev A")
        vals = np.array([prompts.parse_wp(a["wp"]) for a in answers])
        return np.clip(vals[inv], 1e-4, 1 - 1e-4)

    def p_next(self, S: pd.DataFrame, match_id=None) -> np.ndarray:
        inv, answers = self._ask_unique(S, {"p_next": prompts.PNEXT_QUESTION}, "jev B")
        P = np.vstack([prompts.parse_pnext(a["p_next"]) for a in answers])
        return P[inv]

    def plan(self, S: pd.DataFrame, questions: dict) -> dict:
        keys = state_key(S, self.key_cols)
        _, first = np.unique(keys, return_index=True)
        payload = prompts.states_to_jev(S.iloc[first][STATE_COLS], coarse=self.coarse)
        return self.client.estimate([(s, questions) for s in payload])


def jev_stakes(contexts: list[dict], client: JevClient) -> pd.DataFrame:
    """Call C for league-type contexts; knockouts and bilaterals follow the fixed rules."""
    league = [c for c in contexts if c["kind"] == "league"]
    payload = [prompts.stakes_context_to_jev(c) for c in league]
    answers = client.ask([(p, prompts.STAKES_QUESTIONS) for p in payload], desc="jev C") if league else []
    by_id = {(c["match_id"], c["team_id"]): a for c, a in zip(league, answers)}
    out = []
    for c in contexts:
        if c["kind"] == "knockout":
            pw, pl, method = c["fixed_p_win"], c["fixed_p_lose"], "rule:knockout"
        elif c["kind"] == "bilateral":
            pw, pl = bilateral_stakes(c["series_length"], c["wins_team"], c["wins_opp"], c["matches_left_after"])
            method = "rule:bilateral"
        else:
            a = by_id[(c["match_id"], c["team_id"])]
            pw = prompts.parse_wp(a["p_advance_if_win"])
            pl = prompts.parse_wp(a["p_advance_if_lose"])
            method = "jev:callC"
        out.append(dict(match_id=c["match_id"], team_id=c["team_id"], kind=c["kind"],
                        p_advance_if_win=pw, p_advance_if_lose=pl,
                        stakes=float(np.clip(pw - pl, 0, 1)), method=method))
    return pd.DataFrame(out)
