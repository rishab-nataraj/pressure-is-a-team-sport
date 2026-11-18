"""Stage 2: Jev calls A, B and C (with cache), or their empirical fallbacks.

For every real pre-delivery state we need:
  * wp_before                      (call A on the real state)
  * p_next over 9 outcome classes  (call B on the real state)
  * wp of each of the 9 hypothetical next states (call A again; terminal states never go to Jev)
  * wp_after for the real next state (= next delivery's wp_before, or a terminal / transition value)
  * wp_after_bowler: the runs-only counterfactual on run-outs (bowler is not charged the wicket)
and per (match, team): p_advance_if_win, p_advance_if_lose -> stakes (call C).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io, standings
from .config import Paths
from .cricket import (AUX_COLS, LIVE, OUTCOMES, STATE_COLS, TERMINAL, TRANSITION, advance,
                      apply_outcome, classify, inn2_start)
from .providers import resolve

log = logging.getLogger("superr.jev_stage")

NON_BOWLER_DISMISSALS = {"RunOut", "Retire Out", "Retired Out", "Obstructing the field",
                         "Obstracting the field", "Retired Hurt"}


# ------------------------------------------------------------------------------- query building
class WPRequests:
    """Collects states whose wp is needed; resolves them all in one provider pass."""

    def __init__(self):
        self.frames: list[pd.DataFrame] = []
        self.mids: list[np.ndarray] = []
        self.n = 0

    def add(self, S: pd.DataFrame, match_id: np.ndarray) -> np.ndarray:
        idx = np.arange(self.n, self.n + len(S))
        self.frames.append(S[STATE_COLS].reset_index(drop=True))
        self.mids.append(np.asarray(match_id))
        self.n += len(S)
        return idx

    def resolve(self, provider) -> np.ndarray:
        if not self.frames:
            return np.array([])
        S = pd.concat(self.frames, ignore_index=True)
        mid = np.concatenate(self.mids)
        self.S, self.mid = S, mid
        log.info("wp needed for %d states (%d unique by state)", len(S),
                 len(np.unique(pd.util.hash_pandas_object(S.assign(venue_par=S.venue_par.round(1)), index=False))))
        return provider.wp(S, mid)


def _post_values(N: pd.DataFrame, mid: np.ndarray, req: WPRequests):
    """Classify post-ball states; register LIVE and TRANSITION states for wp. Returns a resolver."""
    kind, value = classify(N)
    live = np.where(kind == LIVE)[0]
    trans = np.where(kind == TRANSITION)[0]
    live_idx = req.add(N.iloc[live], mid[live]) if len(live) else np.array([], int)
    trans_idx = req.add(inn2_start(N.iloc[trans]), mid[trans]) if len(trans) else np.array([], int)

    def finish(wp_all: np.ndarray) -> np.ndarray:
        out = value.copy()
        if len(live):
            out[live] = wp_all[live_idx]
        if len(trans):
            out[trans] = 1.0 - wp_all[trans_idx]   # innings-1 batting side's view
        return out
    return finish, kind


def build_and_resolve(st: pd.DataFrame, cfg: dict, wp_provider, pnext_provider) -> pd.DataFrame:
    S = st[STATE_COLS + AUX_COLS].reset_index(drop=True)
    mid = st["match_id"].to_numpy()
    req = WPRequests()

    # real states
    real_idx = req.add(S, mid)

    # 9 hypothetical next states per real state
    hyp_finish = {}
    hyp_kind = {}
    for o in OUTCOMES:
        N = apply_outcome(S, o)
        hyp_finish[o], hyp_kind[o] = _post_values(N, mid, req)

    # actual post-ball state for the last delivery of each innings
    last = st["is_last_ball"].to_numpy()
    li = np.where(last)[0]
    Lb = st.iloc[li]
    NA = advance(S.iloc[li], Lb["bat_runs"].to_numpy(), Lb["extras"].to_numpy(), Lb["is_legal"].to_numpy(),
                 Lb["is_noball"].to_numpy(), Lb["is_wide"].to_numpy(), Lb["is_wicket"].to_numpy(),
                 striker_out=(Lb["dismissed_player_id"].fillna(-1).to_numpy() != Lb["non_striker_id"].to_numpy()))
    inn = Lb["innings"].to_numpy()
    runs_a, tgt = NA["runs"].to_numpy(), NA["target"].to_numpy()
    done = (NA["wickets"].to_numpy() >= 9) | (NA["legal_balls"].to_numpy() >= NA["max_balls"].to_numpy())
    res = Lb["bat_won"].to_numpy()
    last_kind = np.full(len(li), LIVE, np.int8)
    last_val = np.full(len(li), np.nan)
    # innings 1 completed (all out, quota used, or 9 down with an absent batter and no rain revision):
    # hand over to innings 2. An incomplete innings 1 was interrupted (rain): stay LIVE so the
    # interruption is not credited to the last ball.
    wk_a = NA["wickets"].to_numpy()
    no_rev = Lb["match_id"].map(st.groupby("match_id")["target"].max()).to_numpy() == runs_a + 1
    completed1 = (wk_a >= 10) | (NA["legal_balls"].to_numpy() >= NA["max_balls"].to_numpy()) | ((wk_a >= 9) & no_rev)
    i1 = (inn == 1) & completed1
    last_kind[i1] = TRANSITION
    chased = (inn == 2) & (runs_a >= tgt)
    tie = (inn == 2) & ~chased & (res == 0.5) & (runs_a == tgt - 1)
    lost = (inn == 2) & ~chased & ~tie & (res == 0.0) & done
    for msk, v in ((chased, 1.0), (tie, 0.5), (lost, 0.0)):
        last_kind[msk] = TERMINAL
        last_val[msk] = v
    # remaining innings-2 cases (rain / DLS endings) stay LIVE: the decision is not credited to anyone
    lv = np.where(last_kind == LIVE)[0]
    tr = np.where(last_kind == TRANSITION)[0]
    last_live_idx = req.add(NA.iloc[lv], mid[li][lv])
    last_tr_idx = req.add(inn2_start(NA.iloc[tr]), mid[li][tr])

    # bowler counterfactual: run-outs (and optionally byes) - "as if only the runs happened"
    dt = st["dismissal_type"].fillna("")
    cf_wkt = st["is_wicket"].to_numpy() & dt.isin(NON_BOWLER_DISMISSALS).to_numpy()
    cf_bye = np.zeros(len(st), bool)
    if not cfg["attribution"].get("bowler_charged_for_byes", True):
        cf_bye = st["is_bye"].to_numpy() & (st["extras"].to_numpy() > 0)
    cf = np.where(cf_wkt | cf_bye)[0]
    C = st.iloc[cf]
    extras_cf = np.where(cf_bye[cf], 0, C["extras"].to_numpy())
    wkt_cf = np.where(cf_wkt[cf], False, C["is_wicket"].to_numpy())
    bat_cf = C["bat_runs"].to_numpy()
    ran = np.where(np.isin(bat_cf, (4, 6)), 0, bat_cf)
    NC = advance(S.iloc[cf], bat_cf, extras_cf, C["is_legal"].to_numpy(), C["is_noball"].to_numpy(),
                 C["is_wide"].to_numpy(), wkt_cf, ran=ran)
    cf_finish, _ = _post_values(NC, mid[cf], req)

    # ---- one provider pass for every wp
    wp_all = req.resolve(wp_provider)

    # ---- end-game solver for the last K balls of each innings (consistent wp where models are weakest)
    K = endgame_balls(cfg, wp_provider)
    if K > 0:
        from .endgame import solve
        S_all = req.S.copy()
        S_all["inn2_start_max_balls"] = _aux_for(req, S, mid)
        wp_all, _ = solve(S_all, req.mid, wp_all, K, wp_provider, pnext_provider)

    # ---- call B
    P = pnext_provider.p_next(S[STATE_COLS], mid)

    out = pd.DataFrame({"ball_seq": st["ball_seq"].to_numpy(), "match_id": mid})
    out["wp_before"] = wp_all[real_idx]
    out["wp_model"] = out["wp_before"]
    for o in OUTCOMES:
        out[f"wp_{o}"] = hyp_finish[o](wp_all)
        out[f"term_{o}"] = hyp_kind[o] != LIVE
    for j, o in enumerate(OUTCOMES):
        out[f"p_{o}"] = P[:, j]

    # Optional one-step (Bellman) consistency on the final legal ball of an innings: wp before it
    # is the expectation of wp after it over the legal outcomes. A smooth wp model cannot resolve
    # the jump at the innings break; this makes WPA and leverage there consistent with the rules.
    if bellman_final_ball(cfg, wp_provider) and K == 0:
        fb = (S["balls_left"].to_numpy() == 1)
        legal_o = [o for o in OUTCOMES if o not in ("wide", "no_ball")]
        pl = out[[f"p_{o}" for o in legal_o]].to_numpy()
        wl = out[[f"wp_{o}" for o in legal_o]].to_numpy()
        backed = (pl * wl).sum(axis=1) / pl.sum(axis=1)
        out.loc[fb, "wp_before"] = backed[fb]
        log.info("Bellman final-ball backup applied to %d states (mean |change| %.4f)", fb.sum(),
                 float(np.abs(backed[fb] - out.loc[fb, "wp_model"]).mean()))
    nxt = out.groupby([st["match_id"].to_numpy(), st["innings"].to_numpy()])["wp_before"].shift(-1).to_numpy()
    wp_after = nxt.copy()
    lval = last_val.copy()
    lval[lv] = wp_all[last_live_idx]
    lval[tr] = 1.0 - wp_all[last_tr_idx]
    wp_after[li] = lval
    out["wp_after"] = wp_after
    out["post_kind"] = LIVE
    out.loc[li, "post_kind"] = last_kind
    wpb = wp_after.copy()
    wpb[cf] = cf_finish(wp_all)
    out["wp_after_bowler"] = wpb
    out["bowler_counterfactual"] = False
    out.loc[cf, "bowler_counterfactual"] = True
    return out


def _aux_for(req: "WPRequests", S: pd.DataFrame, mid: np.ndarray) -> np.ndarray:
    """inn2_start_max_balls for every requested state (it is a per-match constant)."""
    per_match = pd.Series(S["inn2_start_max_balls"].to_numpy(), index=mid)
    per_match = per_match[~per_match.index.duplicated()]
    return per_match.reindex(req.mid).to_numpy()


def endgame_balls(cfg: dict, wp_provider) -> int:
    from .providers.empirical import EmpiricalWP
    if isinstance(wp_provider, EmpiricalWP):
        return int(cfg["empirical"].get("endgame_balls", 6))
    return int(cfg["jev"].get("endgame_balls", 0))


def bellman_final_ball(cfg: dict, wp_provider) -> bool:
    from .providers.empirical import EmpiricalWP
    if isinstance(wp_provider, EmpiricalWP):
        return bool(cfg["empirical"].get("bellman_final_ball", True))
    return bool(cfg["jev"].get("bellman_final_ball", False))


# ------------------------------------------------------------------------------------- run
def make_providers(cfg: dict, st: pd.DataFrame, paths: Paths):
    from .providers.empirical import EmpiricalPNext, EmpiricalWP
    choice = {k: resolve(cfg, k) for k in ("wp", "p_next", "stakes")}
    client = None
    if "jev" in choice.values():
        from .jev import JevCache, JevClient
        client = JevClient(cfg["jev"], JevCache(paths.jev_cache / "jev_cache.sqlite"))
    situ = None
    if choice["wp"] == "jev" or choice["p_next"] == "jev":
        from .providers.jev_provider import JevSituation
        situ = JevSituation(client, coarse=cfg["jev"].get("coarse_states", False))
    wp = situ if choice["wp"] == "jev" else EmpiricalWP(cfg, st)
    pn = situ if choice["p_next"] == "jev" else EmpiricalPNext(cfg, st)
    return choice, client, situ, wp, pn


def run(cfg: dict, plan_only: bool = False) -> None:
    paths = Paths.from_cfg(cfg)
    st = io.load("states", paths)
    matches = io.load("matches", paths)
    comps = io.load("competitions", paths)
    choice, client, situ, wp, pn = make_providers(cfg, st, paths)
    log.info("providers: %s", choice)

    if plan_only:
        from .jev import prompts
        if situ is None:
            log.info("no Jev provider selected; nothing to plan")
            return
        est = situ.plan(st[STATE_COLS], {"wp": prompts.WP_QUESTION, "p_next": prompts.PNEXT_QUESTION})
        log.info("Jev plan for real states (A+B): %s", est)
        io.save_json({"real_states_AB": est}, "jev_plan", paths)
        return

    if situ is not None and choice["wp"] == "jev" and choice["p_next"] == "jev":
        situ.warm(st[STATE_COLS])  # one request per real state carrying both questions

    situations = build_and_resolve(st, cfg, wp, pn)
    io.save(situations, "situations", paths)

    # ---- call C
    res = standings.match_results(matches, st)
    ctx = standings.build_contexts(matches, comps, res, cfg["tournaments"])
    if choice["stakes"] == "jev":
        from .providers.jev_provider import jev_stakes
        stakes = jev_stakes(ctx, client)
    else:
        stakes = standings.empirical_stakes(ctx, cfg)
    scale = float(cfg.get("stakes", {}).get("bilateral_scale", 1.0))
    bil = stakes["kind"] == "bilateral"
    stakes.loc[bil, "stakes"] = np.clip(scale * stakes.loc[bil, "stakes"], 0, 1)
    io.save(stakes, "stakes", paths)
    io.save(standings.contexts_frame(ctx), "stakes_contexts", paths)
    io.save(res, "match_results", paths)
    meta = {"providers": choice, "jev_model": cfg["jev"]["model"] if client else None,
            "jev_usage": client.usage if client else None,
            "jev_cache": client.cache.stats() if client else None,
            "n_states": int(len(st))}
    io.save_json(meta, "jev_meta", paths)
    log.info("stakes: %d (match, team) rows; mean stakes by kind %s", len(stakes),
             stakes.groupby("kind")["stakes"].mean().round(3).to_dict())
