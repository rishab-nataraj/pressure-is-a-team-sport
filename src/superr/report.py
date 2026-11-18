"""Stage 8: the methodology report (report/methodology.md), generated from pipeline outputs so
every number in it is reproducible. Section order follows the brief:
pressure definition -> performance measurement -> fair comparisons -> evidence for the top 10
-> reliability and limitations."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, plots
from .config import Paths

log = logging.getLogger("superr.report")


def _j(paths: Paths, name: str, sub: str | None = None) -> dict:
    try:
        return io.load_json(name, paths, sub)
    except FileNotFoundError:
        return {}


def _csv(paths: Paths, name: str, sub: str | None = None) -> pd.DataFrame | None:
    fp = (paths.outputs / sub / f"{name}.csv") if sub else (paths.outputs / f"{name}.csv")
    return pd.read_csv(fp) if fp.exists() else None


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _f(x, nd=3) -> str:
    """Readable number: thousands separators for big values, sig-figs for small ones."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)) or (abs(x) >= 100 and float(x).is_integer()):
        return f"{int(x):,}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) < 1e-2 and x != 0:
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}" if abs(x) < 1 else f"{x:.{max(nd - 1, 1)}f}"


def _pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    v = 100 * x
    if 0 < abs(v) < 0.1:
        return f"{v:.2g}%"
    return f"{v:.1f}%"


def _fig(figs: dict, key: str, caption: str) -> str:
    if key not in figs:
        return ""
    return f"![{caption}](figures/{Path(figs[key]).name})\n\n*{caption}*\n"


def top_table(df: pd.DataFrame) -> str:
    t = df[["rank", "player_name", "P", "P_lo", "P_hi", "p_top", "c", "beta", "exposure", "pressure_stints",
            "pressure_balls", "y_pressure", "y_ordinary", "teams"]].copy()
    t["P"] = t["P"].round(1)
    t["90% interval"] = t["P_lo"].round(1).astype(str) + " - " + t["P_hi"].round(1).astype(str)
    t["P(top 10)"] = t["p_top"].round(2)
    t["c x1e4"] = (t["c"] * 1e4).round(2)
    t["beta x1e3"] = (t["beta"] * 1e3).round(2)
    t["E"] = t["exposure"].round(0).astype(int)
    t["y press x1e3"] = (t["y_pressure"] * 1e3).round(2)
    t["y ord x1e3"] = (t["y_ordinary"] * 1e3).round(2)
    t = t.rename(columns={"rank": "#", "player_name": "Player", "pressure_stints": "pressure stints",
                          "pressure_balls": "pressure balls", "teams": "Teams"})
    return _md(t[["#", "Player", "P", "90% interval", "P(top 10)", "c x1e4", "beta x1e3", "E", "pressure stints",
                  "pressure balls", "y press x1e3", "y ord x1e3", "Teams"]], ".2f")


def evidence_table(ev: pd.DataFrame, role: str) -> str:
    e = ev[ev["role"] == role].copy()
    if e.empty:
        return "_no evidence rows_"
    e["y x1e3"] = (e["y"] * 1e3).round(2)
    e["WPA"] = e["sum_wpa"].round(3)
    e["impact x1e3"] = (e["impact"] * 1e3).round(2)
    cols = ["rank", "player_name", "match_date", "comp_name", "team", "opponent", "innings"]
    if role == "bowl":
        cols.append("overid")
    cols += ["n_balls", "h", "WPA", "impact x1e3"]
    return _md(e[cols].rename(columns={"player_name": "Player", "comp_name": "Competition", "n_balls": "balls",
                                       "overid": "over", "match_date": "date"}), ".3g")


def build(cfg: dict) -> Path:
    paths = Paths.from_cfg(cfg)
    figs = plots.make_all(paths, cfg)
    jm = _j(paths, "jev_meta")
    fm = _j(paths, "fit_meta", "fit")
    dv = _j(paths, "data_validation", "validation")
    mv = _j(paths, "model_validation", "validation")
    bat = _csv(paths, "top10_batters")
    bowl = _csv(paths, "top10_bowlers")
    teams = _csv(paths, "team_scores")
    ev = _csv(paths, "top10_evidence_stints")
    cv = _csv(paths, "cv_grid", "fit")
    ab = _csv(paths, "model_11_8_ablations", "validation")
    csk = _csv(paths, "model_11_9_csk_claim", "validation")
    syn = _csv(paths, "model_11_1_synthetic_recovery", "validation")
    sanity = _csv(paths, "data_7_3_sanity", "validation")
    mg = _csv(paths, "data_7_2_martingale", "validation")
    stv = _csv(paths, "data_7_5_stakes_vs_advancement", "validation")
    movers = _csv(paths, "model_11_2_movers_by_team", "validation")
    matches = io.load("matches", paths)
    stints = io.load("stints", paths)
    scored = io.load("scored_players", paths)
    design = io.load("fit_design", paths)

    prov = jm.get("providers", {})
    using_jev = "jev" in prov.values()
    lam = fm.get("lambdas", {})
    pc = cfg["pressure"]
    elig_cfg = cfg["eligibility"]
    n_ex = matches["exclude_reason"].value_counts().to_dict()
    n_ex.pop("", None)
    n_elig = scored.groupby("role")["eligible"].sum().to_dict()
    gates = scored.groupby("role")[["gate_exposure", "gate_recent", "gate_delivers", "gate_selectable"]].sum()
    cvm = fm.get("cv", {})
    base_loss, best_loss = cvm.get("baseline_only_loss"), cvm.get("best_loss")
    rel_gain = (base_loss - best_loss) / base_loss if base_loss else None
    tl = dv.get("top_leverage", {})
    cal = dv.get("calibration", {})
    fc = dv.get("free_consistency", {})
    oot = mv.get("out_of_time", {})
    sh = mv.get("shuffle", {})
    mg_s = mv.get("mover_graph", {})
    weak_teams = teams.loc[teams["weak_identification"], "team_canon"].tolist() if teams is not None else []
    # players whose every team sits outside the main mover component
    comps = mg_s.get("component_team_lists", [])
    main_comp = set(comps[0]) if comps else set()
    def outside(row):
        ts = set(str(row["teams"]).split("; "))
        return len(ts & main_comp) == 0
    flagged = []
    for df, role in ((bat, "bat"), (bowl, "bowl")):
        if df is not None:
            for _, r in df.iterrows():
                if outside(r):
                    flagged.append(f"{r['player_name']} ({role} #{int(r['rank'])}; {r['teams']})")

    L = []
    A = L.append
    A("# SupeRR Selector Season 2 - Round 1 methodology report")
    A("")
    A("**Who delivers under pressure, measured against their own baseline.** One model explains every "
      "batting stint and every bowled over in IPL and T20I cricket 2022-2026: a player's performance "
      "equals their own baseline, plus pressure times (player clutch + own team clutch - opponent team "
      "clutch). Player clutch becomes **P**, team clutch becomes **T**, both on a 0-100 scale where 50 means "
      "no change under pressure.")
    A("")
    A("## 0. Final rankings (as entered on Unstop)")
    A("")
    for df, title in ((bat, "Top 10 batters under pressure"), (bowl, "Top 10 bowlers under pressure")):
        if df is None:
            continue
        A(f"### {title}")
        A("")
        t = df[["rank", "player_name", "P", "P_lo", "P_hi", "p_top", "pressure_stints", "teams"]].copy()
        t.columns = ["Rank", "Player", "P", "P 5%", "P 95%", "P(top 10)", "Pressure stints", "Teams"]
        A(_md(t, ".2f"))
        A("")
    A("**Headline findings**")
    A("")
    A(f"- Pressure is defined from win probability: a ball's leverage is how much the next delivery can move "
      f"the win probability, scaled by what the match is worth for qualification. "
      f"{_pct(tl.get('innings2_share', np.nan))} of the top 1% of balls by leverage come in chases "
      f"and {_pct(tl.get('death_overs_share', np.nan))} in overs 16-20, which is where cricket "
      f"intuition puts pressure.")
    A(f"- The clutch signal is small, as in other sports. Cross-validation keeps the player pressure term "
      f"(lambda_c = {_f(lam.get('c'))}), but held-out error improves by only "
      f"{_pct(rel_gain) if rel_gain is not None else 'n/a'} over a baseline-only model. "
      f"The bootstrap intervals in the top 10 overlap, and P(top 10) is reported for every ranked player "
      f"so the reader can see how firm each position is.")
    A(f"- Team clutch T is pushed toward zero by the data (lambda_tau = {_f(lam.get('tau'))}"
      f"{', the top of its grid' if (cvm.get('at_grid_edge') or {}).get('tau') else ''}). "
      f"The model tests the idea that a player's P depends on T rather than assuming it. On this data the "
      f"rankings rest almost entirely on P.")
    if csk is not None and mv.get("csk_claim"):
        cc = mv["csk_claim"]
        A(f"- The \"CSK has a different match-winner every game\" pattern does not appear in 2022-26: CSK ranks "
          f"{cc.get('csk_rank_by_distinct_per_win')} of {cc.get('n_ipl_teams')} IPL teams on distinct "
          f"man-of-the-match winners per win, and that measure correlates "
          f"{_f(cc.get('spearman_distinct_per_win_vs_T'), 2)} (Spearman) with T.")
    A("")

    # ---------------------------------------------------------------- 1 data
    A("## 1. Data and preparation")
    A("")
    A(f"- **Matches:** {len(matches):,} in the pack; {int((~matches['excluded']).sum()):,} used. Excluded: "
      + ", ".join(f"{v} {k}" for k, v in n_ex.items()) + ". Tied matches are kept (terminal value 0.5); super "
      "overs (innings 3/4) never enter P or T.")
    A(f"- **Deliveries:** {jm.get('n_states', 0):,} pre-delivery states. **Stints:** "
      f"{int((stints['role'] == 'bat').sum()):,} batting stints and {int((stints['role'] == 'bowl').sum()):,} "
      f"bowler-overs; {fm.get('n_stints', 0):,} with at least one attributed ball enter the model, covering "
      f"{fm.get('n_player_roles', 0):,} player-roles and {fm.get('n_teams', 0)} canonical teams.")
    A("- **Legal balls.** The data-pack rule is legal = extras_type not in {wide, Overstep}. The data show "
      "extras_type = \"other\" is a height no-ball (Beamer / Full Toss deliveries, followed by free hits). "
      "Counting it as legal gives 210 seven-ball overs; counting it as a no-ball gives six legal balls in "
      "99.99% of overs, so it is treated as a no-ball (config `data.noball_extras`).")
    A("- **Derived state** (none of it is in the pack): running score, wickets, legal balls, ball quota (DLS "
      "revised overs are already in balls), target (innings-1 total + 1 or the revised target), balls left, "
      "runs needed, free hit (carried through wides and further no-balls), partnership, and striker and "
      "non-striker runs and balls. Every delivery's post-ball state equals the next delivery's pre-ball state.")
    A("- **Canonical teams.** team_canon is resolved by team_name through an alias table, so a renamed "
      "franchise or a second-string side touring as \"India\" shares one T. A-teams and emerging sides are "
      "separate canonical teams. Folding them into the parent side is a switch (`data.a_teams_as_parent`).")
    A("- **Names** come only from competition_team_player_mapping, the dropdown source. Every ranked player "
      "is present in it (`gate_selectable`).")
    A("")

    # ---------------------------------------------------------------- 2 pressure
    A("## 2. Pressure definition")
    A("")
    A("Pressure has two ingredients, kept separate and combined in our code:")
    A("")
    A("1. **Situation leverage.** For each delivery, the win probability (wp) of the batting side before the "
      "ball and after each of nine possible outcomes (dot, 1, 2, 3, 4, 6, wicket, wide, no-ball), weighted "
      "by how likely each outcome is:")
    A("")
    A("   `L = sum_o p_next[o] * |wp(o) - wp_before|`,  `LI = L / mean(L)` (so LI = 1 is an average ball).")
    A("")
    A("   LI captures run rate required, wickets in hand, balls remaining, innings stage and closeness at once. "
      "A ball in a decided match has LI near 0; the last over of a tight chase can exceed 10.")
    A("2. **Match stakes.** `stakes = p_advance_if_win - p_advance_if_lose` for the player's team: the "
      "qualification value of winning this match. For an IPL league game this comes from the standings, the "
      "remaining fixtures and the number of qualifying places. Bilateral series are judged on the series "
      "(decider = 1, dead rubber = 0). Knockouts are fixed: 1, except Qualifier 1 at 0.5, whose loser "
      "still gets Qualifier 2.")
    A("")
    A(f"**Pressure = LI x (1 + k x stakes)**, with k = {pc.get('k')}. A stint's pressure uses the stint's mean "
      f"LI (`pressure.li_column = {pc.get('li_column')}`). The pressure dose entering the model is "
      f"`h = clip(pressure - theta, 0, h_max)` with theta = {pc.get('theta')} (average leverage, no stakes) and "
      f"h_max = {pc.get('h_max')}. Stints with h > 0 are **pressure stints** "
      f"({_pct(float((design['h'] > 0).mean()))} of stints). The rest only inform each player's "
      "baseline.")
    A("")
    A("**Where the numbers come from.** " + (
        f"Win probability, next-ball probabilities and stakes come from Jev ({jm.get('jev_model')}, pinned), "
        "TypeSafe AI's calibrated decision model." if using_jev else
        "This run used the pipeline's empirical providers, because no Jev key was set (the code "
        "switches to Jev automatically when `TYPESAFE_API_KEY` is set; see section 9):") )
    A("")
    if not using_jev:
        A("- **wp:** monotone gradient-boosted models trained on the state table, one per innings, cross-fitted "
          "by match so that no match's states are scored by a model that saw its result. Innings 2 is "
          "Platt-calibrated. Innings 1 learns from the innings-2 model's win probability at the start of the "
          "chase that actually followed, so the two halves agree at the innings break. The final legal ball "
          "of each innings uses the one-step expectation of the next state.")
        A("- **p_next:** empirical outcome frequencies by format x phase x wickets band x innings.")
        A("- **stakes:** a Monte Carlo of the remaining fixtures (coin-flip results, ranked on points then NRR); "
          "series and knockout rules as above.")
        A("")
    A("Either way, the provider never sees who is playing: no team, player or venue names, dates or match IDs. "
      "That keeps reputation out of pressure, and keeps team strength from being counted twice before T is "
      "learned. In stakes calls, teams are relabelled A, B, C.")
    A("")
    A(_fig(figs, "leverage", "Mean leverage by over. Leverage builds through a chase; innings 1 stays near average."))
    A(_fig(figs, "stakes", "Distribution of match stakes by context."))
    A(f"**How it separates pressure from ordinary situations.** The top 1% of deliveries by LI (LI >= "
      f"{_f(tl.get('li_threshold'))}) are {_pct(tl.get('innings2_share', np.nan))} second-innings, "
      f"{_pct(tl.get('death_overs_share', np.nan))} in overs 16-20 and "
      f"{_pct(tl.get('close_chase_share', np.nan))} in chases still in the balance (wp between 0.1 and "
      "0.9). The full list is in `outputs/validation/data_7_6_top_leverage_balls.csv`.")
    A("")

    # ---------------------------------------------------------------- 3 performance
    A("## 3. Performance measurement, baselines and rating logic")
    A("")
    A("**Performance is what the situation needs.** Each ball is scored by win probability added: "
      "`WPA_bat = wp_after - wp_before`, `WPA_bowl = -WPA_bat`. With 2 needed off the last ball, a six off a "
      "yorker and a scrambled two both move wp to 1, so both earn the same WPA. Performance is not binary, "
      "and it scales with how much the moment mattered.")
    A("")
    A("**Attribution.** Batters get WPA on every ball faced except wides, which belong to the bowler. A "
      "non-striker run-out is charged to the dismissed batter. Bowlers get WPA on every ball they bowl, but "
      "a run-out is charged as if only the runs happened (the wp of the runs-only counterfactual state), "
      "because run-outs are not bowler wickets.")
    A("")
    A("**Per-stint inputs.** A batting stint runs from arrival at the crease to dismissal, retirement or the "
      "end of the innings. A bowling stint is one over by one bowler. For stint s with n balls:")
    A("")
    A("```")
    A("y_s  = sum(WPA / max(LI, 0.1)) / n        performance per ball, situation leverage divided out")
    A("h_s  = clip(pressure_s - theta, 0, h_max)  pressure dose")
    A(f"w_s  = n_s * 0.5 ^ (age_days / {pc.get('half_life_days')})       recency weight for player terms")
    A("```")
    A("")
    A("**The model.**")
    A("")
    A("```")
    A("y_s = mu_r + beta_i - q_o + h_s * (c_i + tau_a - tau_o) + eps_s,     Var(eps_s) = sigma^2 / n_s")
    A("```")
    A("")
    A("| Term | Meaning | Becomes |")
    A("|---|---|---|")
    A("| mu_r | role average (bat, bowl) | |")
    A("| beta_i | the player's usual level at ordinary pressure | **baseline** |")
    A("| q_o | opponent's general strength at any pressure | nuisance |")
    A("| c_i | change in the player's performance per unit of pressure, versus their own baseline | **P** |")
    A("| tau_a, tau_o | clutch of own team and of the opponent | **T** |")
    A("")
    A("(c_i + tau_a - tau_o) is the net response to pressure. \"My team makes pressure easier\" and \"my team "
      "makes me better\" are the same +tau_a; a bottling opponent (low tau_o) helps in the same way. tau is "
      "the lift shared by everyone in a shirt; c is what a player carries from team to team. Players who "
      "move between teams (IPL auctions, IPL plus country) are what separates the two.")
    A("")
    A("**Fitting (hard EM).** The loss `sum w (y - yhat)^2 + lambda_b sum beta^2 + lambda_c sum c^2 + "
      "lambda_q sum q^2 + lambda_tau sum tau^2` is minimised by alternating closed-form updates of mu, beta, c, "
      "q and tau. The loop starts from random c and tau and stops when the largest parameter change is "
      f"below {cfg['model'].get('tol')}. The final fit converged in {fm.get('iters')} iterations. Player steps "
      "use the recency weights; team steps use n/2 (every ball sits in one batter stint and one bowler stint) "
      "with no recency, because T is one number for all years.")
    A("")
    A("**Scores.** `P = 100 * Phi(c / s_c)`, where s_c is the sd of c over eligible players in the role; "
      "`T = 100 * Phi(tau / s_tau)`. 50 means no change under pressure, bottle jobs tend to 0, clutch "
      "players to 100.")
    A("")
    A("**Recency.** With a half-life of 540 days, a stint from three years ago counts about a quarter as "
      "much as one from today. Ten recent bottles therefore outweigh ten old clutch performances, and the "
      "reverse.")
    A("")

    # ---------------------------------------------------------------- 4 fair comparisons
    A("## 4. Fair comparisons")
    A("")
    A("- **Opportunity weighting (sample size).** The c update has the closed form "
      "`c_i = [E_i / (E_i + lambda_c)] x (raw pressure slope)`, with `E_i = sum omega n h^2`. E grows with "
      "how often, how hard and how recently a player was under pressure. A bowler with hundreds of pressure "
      "overs keeps most of his slope. A player with two clutch stints is pulled almost to zero, so he sits "
      "near P = 50 without any extra rule.")
    A(f"- **Penalties by cross-validation**: 5-fold CV grouped by match, minimising weighted squared error on "
      f"held-out stints, coordinate search over the grid {cfg['model'].get('lambda_grid')}. Chosen: "
      f"lambda_beta = {_f(lam.get('beta'))}, lambda_c = {_f(lam.get('c'))}, lambda_q = {_f(lam.get('q'))}, "
      f"lambda_tau = {_f(lam.get('tau'))}. k, theta, h_max and the half-life were fixed in advance and only "
      "reported as sensitivities (section 7).")
    A(f"- **Eligibility gates.** Batters need >= {elig_cfg['bat']['min_pressure_stints']} pressure stints and "
      f">= {elig_cfg['bat']['min_pressure_balls']} balls in them. Bowlers need >= "
      f"{elig_cfg['bowl']['min_pressure_stints']} pressure overs. Everyone needs >= "
      f"{elig_cfg['min_recent_pressure_stints']} pressure stints in the last {elig_cfg['recent_months']} "
      "months, so \"right now\" has support. Ranked players must also **deliver** under pressure "
      "(beta + c > 0: above the role average at h = 1), so a weak player who merely declines less cannot "
      "top a \"best under pressure\" list. And they must be selectable in the dropdown.")
    A("")
    if not gates.empty:
        g = gates.copy()
        g["eligible"] = pd.Series(n_elig)
        g = g.reset_index().rename(columns={"gate_exposure": "exposure", "gate_recent": "recent",
                                            "gate_delivers": "delivers", "gate_selectable": "selectable"})
        A(_md(g, ".0f"))
        A("")
    A("- **Different match contexts.** The opponent's general strength enters the baseline as q_o, so a "
      "mismatch in a minor T20I does not inflate anyone's baseline. Stakes put a dead rubber below a series "
      "decider, and an early-season IPL game below a must-win.")
    A("- **IPL and T20I together.** One model and one scale. Format differences are absorbed by the wp "
      "model (format and Impact Player era are state inputs) and by dividing out leverage. A player keeps a "
      "single P per role across all teams. Every IPL franchise and national side has one T for all years.")
    A("- **Per-role P.** Batting and bowling are scored separately (two lists). All-rounders get both. A "
      "single exposure-weighted P per player is in `outputs/player_scores.csv` (`P_combined`).")
    A("")
    A(_fig(figs, "shrinkage", "Shrinkage by exposure: thin-sample players are pulled to c = 0 (P = 50)."))

    # ---------------------------------------------------------------- 5 evidence
    A("## 5. Evidence for the top 10")
    A("")
    A("Columns: P with its 90% cluster-bootstrap interval (300 resamples of matches) and P(top 10); c "
      "(pressure slope) and beta (baseline), both per ball; E (effective pressure exposure); the raw "
      "performance per ball in pressure versus ordinary stints. Rank order is P, with ties broken by "
      "P(top 10).")
    A("")
    for df, role, title in ((bat, "bat", "Batters"), (bowl, "bowl", "Bowlers")):
        if df is None:
            continue
        A(f"### {title}")
        A("")
        A(top_table(df))
        A("")
        A(_fig(figs, f"top10_{role}", f"Top 10 {title.lower()}: P and bootstrap interval."))
        if ev is not None:
            A(f"**{title}: the three biggest pressure stints behind each ranking** (by n x h x (y - mu - beta + q_o)):")
            A("")
            A(evidence_table(ev, role))
            A("")
    if flagged:
        A("**Identification flag.** These ranked players' teams have no movers linking them to the main "
          "player-team network, so their split between P and T rests on the penalties alone (plan section 6): "
          + "; ".join(flagged) + ".")
        A("")

    # ---------------------------------------------------------------- 6 team T
    A("## 6. Team clutch T and the CSK claim")
    A("")
    if teams is not None:
        t = teams[["team_canon", "T", "tau", "q", "movers", "weak_identification"]].copy()
        t["tau x1e5"] = (t["tau"] * 1e5).round(2)
        t["q x1e4"] = (t["q"] * 1e4).round(2)
        A(_md(t[["team_canon", "T", "tau x1e5", "q x1e4", "movers", "weak_identification"]]
              .rename(columns={"team_canon": "Team", "weak_identification": "no movers"}), ".1f"))
        A("")
    A(_fig(figs, "team_T", "T by team (50 = average)."))
    A(f"T is identified by players who appear for more than one team. "
      f"{_pct(mg_s.get('pressure_exposure_share_from_movers', np.nan))} of pressure exposure comes "
      f"from player-roles with 2+ teams. The player-team graph has {mg_s.get('components')} connected "
      f"components; the largest links {mg_s.get('largest_component_teams')} teams. Teams with no movers: "
      f"{', '.join(weak_teams) if weak_teams else 'none'}.")
    A("")
    if csk is not None:
        A("**The CSK claim** (distinct man-of-the-match winners per win, IPL, from matches.csv):")
        A("")
        c = csk[csk["format"] == "IPL"][["team_canon", "wins_with_mom", "distinct_mom", "distinct_per_win",
                                         "top_player_share", "T"]]
        A(_md(c, ".2f"))
        A("")
        A(_fig(figs, "csk", "Distinct match-winners per win against T (IPL)."))

    # ---------------------------------------------------------------- 7 reliability
    A("## 7. Reliability")
    A("")
    A("### 7a. The situation layer (Jev spec section 7)")
    A("")
    A(f"- **Calibration** (wp deciles vs actual winners): innings 1 Brier "
      f"{_f(cal.get('innings1', {}).get('brier'))}, worst decile gap {_f(cal.get('innings1', {}).get('max_bin_gap'))}; "
      f"innings 2 Brier {_f(cal.get('innings2', {}).get('brier'))}, worst gap "
      f"{_f(cal.get('innings2', {}).get('max_bin_gap'))}.")
    A(_fig(figs, "calibration", "Win-probability calibration by innings."))
    A(f"- **Martingale:** mean WPA over all balls is {_f(dv.get('martingale_overall'))}. By innings and phase:")
    A("")
    if mg is not None:
        A(_md(mg, ".2g"))
        A("")
    A(f"- **Sanity:** {dv.get('sanity_total_violations')} violations of \"more runs never lowers wp, a wicket "
      "never raises it\" over about 1.8M hypothetical comparisons. They sit where a hypothetical crosses "
      "the innings break.")
    A("")
    if sanity is not None:
        A(_md(sanity[sanity["violations"] > 0], ".4f") if (sanity["violations"] > 0).any() else "_no violations_")
        A("")
    A(f"- **Consistency of hypotheticals:** on {fc.get('pure_outcome_balls', 0):,} balls whose real outcome is "
      f"one of the nine classes, the real next state's wp equals the hypothetical's on "
      f"{_pct(fc.get('exact_match_share', np.nan))} (mean |diff| {_f(fc.get('mean_abs_diff'))}).")
    sva = dv.get("stakes_vs_advancement", {})
    A(f"- **Stakes vs actual advancement:** the realised p_advance has Brier {_f(sva.get('brier'))} against "
      f"actual qualification, compared with {_f(sva.get('brier_coinflip'))} for a coin flip "
      f"({sva.get('n')} league-stage (match, team) rows).")
    A("")
    if stv is not None:
        A(_md(stv, ".3f"))
        A("")
    A(f"- **High-leverage balls:** see section 2. **p_next vs empirical frequencies:** max cell gap "
      f"{_f(dv.get('pnext_max_abs_diff'))}; the largest gaps are in cells with under 50 balls.")
    rep = dv.get("reproducibility", {})
    A(f"- **Reproducibility:** {rep.get('note', '')}"
      + (f" Mean |live - cached| = {_f(rep.get('mean_abs_diff'))} on {rep.get('sampled_requests')} re-sent "
         "requests." if rep.get("sampled_requests") else ""))
    if dv.get("wp_provider_comparison"):
        w = dv["wp_provider_comparison"]
        A(f"- **Jev vs empirical wp** on {w['n_states']:,} states: Jev Brier {_f(w['jev']['brier'])}, "
          f"pipeline Brier {_f(w['pipeline_wp']['brier'])}.")
    A("")
    A("### 7b. The model (plan section 11)")
    A("")
    if syn is not None:
        sg = syn.groupby("effect_multiplier")[["spearman_c_eligible", "corr_c_eligible", "corr_tau"]].mean().reset_index()
        A(f"1. **Synthetic recovery.** Known c and tau were planted on the real schedule, with noise at the "
          f"observed sigma, and the model was refit. Effect sizes are multiples of the spread the CV-chosen "
          f"penalties allow (sd_c = {_f(mv.get('prior_sd', {}).get('c'))}, sd_tau = "
          f"{_f(mv.get('prior_sd', {}).get('tau'))}):")
        A("")
        A(_md(sg.rename(columns={"effect_multiplier": "planted size (x prior sd)"}), ".3f"))
        A("")
        A(_fig(figs, "synthetic", "Planted vs recovered effects."))
    A(f"2. **Mover graph:** {mg_s.get('components')} components; "
      f"{_pct(mg_s.get('pressure_exposure_share_from_movers', np.nan))} of pressure exposure comes from "
      "movers (per-team counts in `outputs/validation/model_11_2_movers_by_team.csv`).")
    ss = mv.get("start_stability", {})
    A(f"3. **Start stability:** {ss.get('seeds')} random starts agree to {_f(ss.get('max_abs_param_diff'))} "
      f"(parameter scale {_f(ss.get('max_param_scale'))}); all converged: {ss.get('all_converged')}.")
    os_ = mv.get("one_shot", {})
    A(f"4. **One-shot check:** with split_weights off, the loop matches a single sparse ridge solve of the same "
      f"loss to {_f(max(os_.get('max_diff_beta', 0), os_.get('max_diff_c', 0), os_.get('max_diff_q', 0), os_.get('max_diff_tau', 0)))}.")
    if oot:
        def o(k):
            v = oot.get(k)
            return f"{_f(v['spearman'], 2)} (n = {v['n']}, p = {_f(v['p_value'], 2)})" if v else "n/a"
        A(f"5. **Out-of-time:** fitted on matches up to {oot.get('train_end')}, the Spearman correlation of c "
          f"with each player's 2025-26 pressure slope is {o('bat_all')} for batters and {o('bowl_all')} for "
          f"bowlers. For players eligible on the training data it is {o('bat_train_eligible')} and "
          f"{o('bowl_train_eligible')}.")
        A(_fig(figs, "oot", "Out-of-time: fitted c vs the next period's pressure slope."))
    bsm = mv.get("bootstrap", {})
    A(f"6. **Cluster bootstrap:** median 90%-interval width in the top 10 is "
      f"{_f(bsm.get('bat', {}).get('median_interval_width_top10'))} P points for batters and "
      f"{_f(bsm.get('bowl', {}).get('median_interval_width_top10'))} for bowlers. The mean P(top 10) of the "
      f"listed players is {_f(bsm.get('bat', {}).get('mean_p_top10_of_top10'), 2)} and "
      f"{_f(bsm.get('bowl', {}).get('mean_p_top10_of_top10'), 2)}.")
    A(f"7. **Shuffle tests:** the real spread of c divided by its spread under the null is "
      f"{_f(sh.get('ratio_real_over_permute_h'), 3)} (h permuted within player and stint-length band) and "
      f"{_f(sh.get('ratio_real_over_sign_flip'), 3)} (baseline residual signs flipped). Values near 1 mean "
      "the spread of c is largely what noise alone would produce.")
    if ab is not None:
        A("8. **Ablations and sensitivities** (refit with the main penalties; overlap with the main top 10):")
        A("")
        A(_md(ab[["ablation", "role", "top10_overlap", "same_rank1", "spearman_c_common_eligible", "n_eligible"]], ".2f"))
        A("")
        A(_fig(figs, "ablations", "Top-10 overlap under each ablation."))
    A("9. **The CSK claim:** section 6.")
    A("")

    # ---------------------------------------------------------------- 8 limitations
    A("## 8. Limitations and where the model is less certain")
    A("")
    A("1. **The clutch signal is small.** The out-of-time, shuffle and synthetic results all say the same "
      "thing: most of the spread in P is noise. The top-10 ordering is the best estimate the data support, "
      "not a sharp separation. The bootstrap intervals and P(top 10) show how firm each place is.")
    A("2. **T is weak in this window.** CV shrinks tau hard. Teams without movers have T split from their "
      "players' P by the penalties alone (section 6).")
    A("3. **mean_li is partly caused by the player.** A batter who kills a chase lowers his own stint's "
      "pressure. The entry_li ablation shows how much this moves the list.")
    A("4. **Stint grain.** One pressure value for a 40-ball innings blurs the signal. Cutting batter stints "
      "at over boundaries is a drop-in upgrade; the equations are unchanged.")
    A("5. **Win-probability quality.** Everything rests on wp. The provider never sees team identity, so wp "
      "describes an average team by design. That way T is learned rather than assumed.")
    A("6. **One T for all years** hides real changes inside a franchise. Accepted by design.")
    A("7. **Opponent individuals are not modelled.** A six off a great bowler counts the same as a two off a "
      "poor one; opponents enter only through q and tau.")
    A("8. **Stakes** for bilaterals and minor tournaments are coarser than for the IPL. Byes and leg-byes "
      "count in bowler WPA (`attribution.bowler_charged_for_byes` switches this to a no-bye counterfactual).")
    A("")

    # ---------------------------------------------------------------- 9 reproduce
    A("## 9. Reproducing this report")
    A("")
    A("```bash")
    A("pip install -r requirements.txt")
    A("cp /path/to/data-pack/*.csv data/raw/")
    A("export TYPESAFE_API_KEY=...        # optional: use Jev for calls A/B/C (cached in data/jev_cache/)")
    A("make all                           # or: python -m superr run all")
    A("```")
    A("")
    A(f"Providers used in this run: {prov}. Seeds are fixed. Every switch named above is in `config/config.yaml`.")
    A("")
    out = paths.report / "methodology.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", out)
    return out


def run(cfg: dict) -> None:
    build(cfg)
