"""Data-side validation (Jev spec section 7): is the situation layer (wp, p_next, stakes) sound?"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from . import io
from .config import Paths
from .cricket import OUTCOMES

log = logging.getLogger("superr.validate_data")


def calibration(b: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """7.1: wp_before deciles vs actual winners (ties excluded)."""
    x = b[b["bat_won"].isin([0.0, 1.0])]
    rows, summ = [], {}
    for inn in (1, 2):
        z = x[x["innings"] == inn]
        q = pd.qcut(z["wp_before"], bins, duplicates="drop")
        t = z.groupby(q, observed=True).agg(wp_mean=("wp_before", "mean"), won=("bat_won", "mean"),
                                            n=("wp_before", "size")).reset_index(drop=True)
        t.insert(0, "innings", inn)
        t.insert(1, "bin", np.arange(1, len(t) + 1))
        rows.append(t)
        p = np.clip(z["wp_before"].to_numpy(), 1e-6, 1 - 1e-6)
        y = z["bat_won"].to_numpy()
        summ[f"innings{inn}"] = {"brier": float(np.mean((p - y) ** 2)),
                                 "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                                 "max_bin_gap": float((t["wp_mean"] - t["won"]).abs().max())}
    return pd.concat(rows, ignore_index=True), summ


def martingale(b: pd.DataFrame) -> pd.DataFrame:
    """7.2: mean WPA ~ 0 overall and within each phase."""
    g = b.groupby(["innings", "phase"])["wpa_bat"]
    t = pd.DataFrame({"mean_wpa": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()}).reset_index()
    tot = pd.DataFrame([{"innings": "all", "phase": "all", "mean_wpa": b["wpa_bat"].mean(),
                         "se": b["wpa_bat"].std() / np.sqrt(len(b)), "n": len(b)}])
    t = pd.concat([t, tot], ignore_index=True)
    t["z"] = t["mean_wpa"] / t["se"]
    return t


def sanity(situ: pd.DataFrame, st: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """7.3: more runs never lowers wp, a wicket never raises it (counted on hypothetical states)."""
    pairs = [("dot", "one"), ("one", "two"), ("two", "three"), ("three", "four"), ("four", "six"),
             ("wicket", "dot"), ("dot", "wide")]
    rows = []
    for inn in (1, 2):
        m = st["innings"].to_numpy() == inn
        for lo, hi in pairs:
            viol = situ.loc[m, f"wp_{hi}"].to_numpy() < situ.loc[m, f"wp_{lo}"].to_numpy() - tol
            rows.append({"innings": inn, "check": f"wp({hi}) >= wp({lo})", "violations": int(viol.sum()),
                         "share": float(viol.mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)


def free_consistency(situ: pd.DataFrame, st: pd.DataFrame) -> dict:
    """The actual next state is one of the hypotheticals whenever the outcome is a 'pure' class."""
    o = st["outcome"].to_numpy()
    pure = ((st["extras"] == 0) & ~st["is_wicket"] & st["is_legal"] & st["bat_runs"].isin([0, 1, 2, 3, 4, 6])) | \
           ((st["outcome"] == "wicket") & (st["bat_runs"] == 0) & (st["extras"] == 0) &
            (st["dismissed_player_id"] == st["striker_id"])) | \
           ((st["outcome"] == "wide") & (st["extras"] == 1) & ~st["is_wicket"]) | \
           ((st["outcome"] == "no_ball") & (st["extras"] == 1) & (st["bat_runs"] == 0) & ~st["is_wicket"])
    pure &= ~st["is_last_ball"]
    hyp = np.choose(pd.Categorical(o, categories=OUTCOMES).codes.clip(0),
                    [situ[f"wp_{k}"].to_numpy() for k in OUTCOMES])
    d = np.abs(hyp - situ["wp_after"].to_numpy())[pure.to_numpy()]
    return {"pure_outcome_balls": int(pure.sum()), "exact_match_share": float((d < 1e-9).mean()),
            "mean_abs_diff": float(d.mean()), "p99_abs_diff": float(np.quantile(d, 0.99)),
            "note": "differences come from strike changes the rules cannot know (batters crossing on a "
                    "catch, a new batter at the other end) and the final-ball backup"}


def pnext_vs_empirical(situ: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """7.4: provider p_next vs empirical frequencies by format x phase x wickets band."""
    band = np.select([st["wickets"] <= 2, st["wickets"] <= 5], ["0-2", "3-5"], default="6-9")
    key = pd.DataFrame({"format": st["format"], "phase": st["phase"], "wband": band})
    pred = situ[[f"p_{o}" for o in OUTCOMES]].groupby([key["format"], key["phase"], key["wband"]]).mean()
    emp = pd.crosstab([key["format"], key["phase"], key["wband"]], st["outcome"], normalize="index")
    emp = emp.reindex(columns=OUTCOMES, fill_value=0)
    pred.columns = OUTCOMES
    diff = (pred - emp).abs()
    out = pred.round(4).astype(str) + " / " + emp.round(4).astype(str)
    out["max_abs_diff"] = diff.max(axis=1).round(4)
    out["n"] = key.groupby(["format", "phase", "wband"]).size()
    return out.reset_index()


def stakes_vs_advancement(stakes: pd.DataFrame, ctx: pd.DataFrame, adv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """7.5: realised p_advance (given the actual result) vs whether the team actually advanced."""
    x = stakes.merge(ctx[["match_id", "team_id", "comp_id", "group", "won", "kind"]], on=["match_id", "team_id", "kind"])
    x = x[x["kind"] == "league"].merge(adv, on=["comp_id", "group", "team_id"], how="inner")
    x["p_real"] = np.where(x["won"], x["p_advance_if_win"], x["p_advance_if_lose"])
    x["advanced"] = x["advanced"].astype(float)
    q = pd.cut(x["p_real"], [-0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    t = x.groupby(q, observed=True).agg(p_mean=("p_real", "mean"), advanced=("advanced", "mean"),
                                        n=("p_real", "size")).reset_index()
    t["bin"] = t["p_real"].astype(str)
    t = t.drop(columns=["p_real"])
    summ = {"brier": float(((x["p_real"] - x["advanced"]) ** 2).mean()), "n": int(len(x)),
            "brier_coinflip": float(((0.5 - x["advanced"]) ** 2).mean())}
    return t, summ


def top_leverage(b: pd.DataFrame, share: float) -> tuple[pd.DataFrame, dict]:
    """7.6: read the top 1% of balls by LI - they should mostly be last overs of close chases."""
    n = max(1, int(len(b) * share))
    t = b.nlargest(n, "LI")
    close = (t["innings"] == 2) & (t["wp_before"].between(0.1, 0.9))
    summ = {"n": n, "innings2_share": float((t["innings"] == 2).mean()),
            "death_overs_share": float((t["overid"] >= 16).mean()),
            "close_chase_share": float(close.mean()),
            "innings2_death_share": float(((t["innings"] == 2) & (t["overid"] >= 16)).mean()),
            "li_threshold": float(t["LI"].min())}
    cols = ["match_id", "innings", "overid", "over_ball_num", "runs", "wickets", "target", "runs_needed",
            "balls_left", "wp_before", "wp_after", "LI", "outcome"]
    return t[cols].head(40), summ


def leverage_profile(b: pd.DataFrame) -> pd.DataFrame:
    return b.groupby(["innings", "overid"])["LI"].agg(["mean", "median", "size"]).reset_index()


def reproducibility(cfg: dict, paths: Paths, meta: dict) -> dict:
    """7.7: Jev - re-query a sample of cached requests live and compare; empirical - output hash."""
    situ_fp = paths.interim / "situations.parquet"
    out = {"situations_sha256": hashlib.sha256(situ_fp.read_bytes()).hexdigest() if situ_fp.exists() else None}
    if "jev" not in (meta.get("providers") or {}).values():
        out["note"] = ("empirical provider: seeded and deterministic; re-running stage 2 reproduces the "
                       "same numbers (compare the hash above)")
        return out
    from .jev import JevCache, JevClient
    cache = JevCache(paths.jev_cache / "jev_cache.sqlite")
    client = JevClient(cfg["jev"], cache)
    if not client.available:
        out["note"] = "Jev key not set; live re-query skipped"
        return out
    sample = cache.sample_requests(cfg["jev"].get("repro_sample_frac", 0.01))
    diffs = []
    for _, req, resp in sample:
        live = client.post(req)
        for qid, a in resp.get("answers", {}).items():
            b = live.get("answers", {}).get(qid, {})
            if "noul" in a and "noul" in b:
                diffs.append(abs(float(a["noul"]) - float(b["noul"])))
            elif "probabilities" in a and "probabilities" in b:
                diffs.append(max(abs(float(a["probabilities"][k]) - float(b["probabilities"].get(k, 0)))
                                 for k in a["probabilities"]))
    out.update({"sampled_requests": len(sample), "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
                "max_abs_diff": float(np.max(diffs)) if diffs else None})
    return out


def compare_wp_providers(cfg: dict, paths: Paths, st: pd.DataFrame, b: pd.DataFrame) -> dict | None:
    """Plan 12.5: if the Jev cache holds wp answers for real states, compare calibration with the
    pipeline's wp (Brier / log loss on the same states)."""
    fp = paths.jev_cache / "jev_cache.sqlite"
    if not fp.exists():
        return None
    from .cricket import STATE_COLS
    from .jev import JevCache, prompts
    from .jev.cache import answer_key
    cache = JevCache(fp)
    model = cfg["jev"]["model"]
    states = prompts.states_to_jev(st[STATE_COLS], coarse=cfg["jev"].get("coarse_states", False))
    keys = [answer_key(model, s, "wp", prompts.WP_QUESTION) for s in states]
    got = cache.get_many(keys)
    if not got:
        return None
    jev = np.array([prompts.parse_wp(got[k]) if k in got else np.nan for k in keys])
    m = ~np.isnan(jev) & b["bat_won"].isin([0.0, 1.0]).to_numpy()
    y = b["bat_won"].to_numpy()[m]

    def score(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {"brier": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    return {"n_states": int(m.sum()), "jev": score(jev[m]), "pipeline_wp": score(b["wp_before"].to_numpy()[m])}


def run(cfg: dict) -> dict:
    paths = Paths.from_cfg(cfg)
    st = io.load("states", paths)
    situ = io.load("situations", paths)
    b = io.load("balls", paths)
    stakes = io.load("stakes", paths)
    ctx = io.load("stakes_contexts", paths)
    meta = io.load_json("jev_meta", paths)
    from .standings import actual_advancement
    adv = actual_advancement(io.load("matches", paths), io.load("competitions", paths), cfg["tournaments"])
    vcfg = cfg["validation"]
    sub = "validation"
    res: dict = {"providers": meta.get("providers")}

    cal, res["calibration"] = calibration(b)
    io.save_output(cal, "data_7_1_calibration", paths, sub)
    mg = martingale(b)
    io.save_output(mg, "data_7_2_martingale", paths, sub)
    res["martingale_overall"] = float(b["wpa_bat"].mean())
    sn = sanity(situ, st)
    io.save_output(sn, "data_7_3_sanity", paths, sub)
    res["sanity_total_violations"] = int(sn["violations"].sum())
    res["free_consistency"] = free_consistency(situ, st)
    pn = pnext_vs_empirical(situ, st)
    io.save_output(pn, "data_7_4_pnext_vs_empirical", paths, sub)
    res["pnext_max_abs_diff"] = float(pn["max_abs_diff"].max())
    sa, res["stakes_vs_advancement"] = stakes_vs_advancement(stakes, ctx, adv)
    io.save_output(sa, "data_7_5_stakes_vs_advancement", paths, sub)
    tl, res["top_leverage"] = top_leverage(b, vcfg.get("top_li_share", 0.01))
    io.save_output(tl, "data_7_6_top_leverage_balls", paths, sub)
    io.save_output(leverage_profile(b), "data_leverage_by_over", paths, sub)
    res["reproducibility"] = reproducibility(cfg, paths, meta)
    res["wp_provider_comparison"] = compare_wp_providers(cfg, paths, st, b)
    res["stakes_by_kind"] = stakes.groupby("kind")["stakes"].describe().round(3).to_dict("index")
    io.save_json(res, "data_validation", paths, sub)
    log.info("data validation: %s", {k: res[k] for k in ("calibration", "martingale_overall", "sanity_total_violations")})
    return res
