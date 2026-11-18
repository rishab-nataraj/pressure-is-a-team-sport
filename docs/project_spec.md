# SupeRR Selector S2: Pressure Model Project Plan (P and T)

Draft v1, 18 Nov 2026. Companion to [`jev_pressure_data_spec.md`](jev_pressure_data_spec.md) ("the spec").
The spec ends at the stint table. This document starts there and ends at the two top-10 lists.

> Implementation notes (what the code does beyond this plan) are collected at the end.

## 1. Summary

One model explains every stint: a player's performance equals their own baseline, plus
pressure times (player clutch + own team clutch − opponent team clutch). Player clutch
becomes P, team clutch becomes T. The model is fitted by alternating closed-form updates
(the EM-style loop), then P is mapped to 0–100 and ranked.

Decisions taken here (each is a switch you can flip):

| # | Decision | Why |
|---|---|---|
| 1 | Team effect is added to the player's response, not multiplied into pressure | A multiplicative discount shrinks good and bad responses alike, so a clutch player on a clutch team would look worse |
| 2 | P is fitted per player per role (P_bat, P_bowl) | The deliverable is two separate lists; all-rounders need both |
| 3 | P is recency-weighted; T uses all years equally | P is "right now", T is one number for all years |
| 4 | A separate opponent-strength term q sits in the baseline | So T means "clutch beyond general strength", and T20I mismatches don't distort baselines |
| 5 | Rank by P, behind eligibility gates | Shrinkage handles small samples; gates keep the list defensible |

If you need a single P per player, take the exposure-weighted mean of P_bat and P_bowl.

## 2. Pipeline

| Stage | Input | Output | Jev? |
|---|---|---|---|
| 0 Canonical IDs | teams.csv, player mapping | team_canon map, player names | no |
| 1 State table | ball_by_ball, matches | one row per delivery (spec §1) | no |
| 2 Jev calls | states, standings | wp, p_next, stakes, all cached (spec §2–4) | yes |
| 3 Per-ball fields | stages 1–2 | WPA, LI, pressure, attribution (spec §5) | no |
| 4 Stint table | stage 3 | one row per batter stint / bowler over (spec §6) | no |
| 5 Fit | stint table | β, c, q, τ (sections 5–8 here) | no |
| 6 Score and rank | stage 5 | P, T on 0–100, two top-10 lists | no |
| 7 Validate | stages 3–6 | tables and charts for the report | no |

## 3. Data preparation beyond the spec

- **Canonical teams.** Normalise team_name and keep an alias table for renames
  (e.g. Royal Challengers Bangalore / Bengaluru). A second-string side touring as "India"
  has team_name India, so it gets India's T automatically.
- **A and emerging sides** have different names, so they are separate canonical teams.
  Dropping the two emerging Asia Cups entirely is an ablation in section 11.
- **Exclusions.** Super overs and no-results (spec), and the 16 matches with no ball-by-ball.
  Drop stints with n_balls = 0.
- **Model key.** (player_id, role). One player can have a bat row and a bowl row.
- **Dropdown check.** Final names come from competition_team_player_mapping; confirm
  each is selectable on Unstop before locking the list.

## 4. Per-ball quantities (recap of spec §5)

```
WPA_bat = wp_after − wp_before            WPA_bowl = −WPA_bat
L       = Σ_o p_next[o] · |wp(o) − wp_before|
LI      = L / mean(L over all deliveries)       floor at 0.1 when dividing
stakes  = p_advance_if_win − p_advance_if_lose
```

WPA is the definition of performance. With 2 needed off the last ball, a six and a
two both move wp to 1, so both earn the same WPA.

## 5. From stint table to model inputs

For each stint s:

```
n_s = n_balls
y_s = sum_wpa_li / n_s                    performance per ball, leverage divided out
π_s = mean_li · (1 + k·stakes)            pressure (the spec's pressure column)
h_s = min( max(π_s − θ, 0), h_max )       pressure dose
ω_s = 0.5 ^ (age_days_s / H)              recency weight; age counted from the last match date
```

Starting values: k = 1, θ = 1, h_max = 3, H = 540 days.

- θ = 1 is average leverage with no stakes. Stints at or below it have h = 0 and only
  inform the baseline. Stints above it are **pressure stints**.
- h_max stops a handful of extreme stints from deciding a player's score.
- With H = 18 months, a stint from 3 years ago counts 25% of one from today. This is
  what makes ten recent bottles outweigh ten old clutch performances.

## 6. The model

```
y_s = μ_r + β_i − q_o + h_s · ( c_i + τ_a − τ_o ) + ε_s        Var(ε_s) = σ² / n_s
```

i = player-role, r = role, a = own team, o = opponent, all read off the stint row.

| Term | Meaning | Becomes |
|---|---|---|
| μ_r | role average (bat, bowl) | |
| β_i | player's usual level at ordinary pressure | the baseline |
| q_o | opponent's general strength at any pressure | nuisance term |
| c_i | change in the player's performance per unit of pressure, versus own baseline | **P** |
| τ_a, τ_o | clutch of own team and of the opponent | **T** |

Loss being minimised (ridge penalties are the priors that every effect is near zero
until the data say otherwise):

```
Σ_s w_s (y_s − ŷ_s)²  +  λ_β Σ β_i²  +  λ_c Σ c_i²  +  λ_q Σ q_t²  +  λ_τ Σ τ_t²
```

How this encodes the idea:

- **One term for both readings.** (c_i + τ_a − τ_o) is the net response to pressure.
  "My team makes pressure easier" and "my team makes me better" are the same +τ_a.
  A bottling opponent (low τ_o) helps in the same way.
- **P and T inform each other.** Each stint is evidence about the sum, so what T
  explains is removed before P is updated, and the reverse.
- **What separates them.** τ is the lift shared by everyone in that shirt; c is what a
  player carries from team to team. Players with 2+ teams (IPL moves, IPL plus country)
  identify the split. When a whole squad lifts under pressure, the penalty makes one τ
  cheaper than twenty c's, which is the "different CSK player every match" pattern.
- **Where the split is weak.** A national side whose players never appear for another
  team (check Pakistan in this window) has no movers. Its τ and its players' c are then
  split by the penalties alone. Flag those players in the report.

## 7. Fitting: the alternating loop

Given everything else, each block has a closed form. Let δ_s = c_i + τ_a − τ_o.

```
μ_r = Σ w (y − β_i + q_o − hδ) / Σ w                              stints of role r
β_i = Σ w (y − μ_r + q_o − hδ) / ( Σ w + λ_β )                    stints of i
c_i = Σ w h (y − μ_r − β_i + q_o − h(τ_a − τ_o)) / ( Σ w h² + λ_c )   stints of i
q_t = − Σ v (y − μ_r − β_i − hδ) / ( Σ v + λ_q )                  stints with t as opponent
τ   = ( XᵀVX + λ_τ I )⁻¹ XᵀV r        r_s = y − μ_r − β_i + q_o − h·c_i
```

X has one row per stint: +h in the own-team column, −h in the opponent column. With
about 38 teams this is a small exact solve.

Weights:

```
w_s = n_s · ω_s      player steps (μ, β, c): recency-weighted
v_s = n_s / 2        team steps (q, τ): no recency; halved because every ball sits in
                     one batter stint and one bowler stint
```

**Opportunity weighting** falls out of the c update:

```
E_i = Σ_s ω_s · n_s · h_s²               effective pressure exposure
c_i = [ E_i / (E_i + λ_c) ] × (raw pressure slope of player i)
```

E_i grows with how often, how hard and how recently a player was under pressure. A
bowler with hundreds of pressure overs keeps most of his raw slope. A player with two
clutch stints is pulled almost to zero. λ_c is the exposure at which data and prior get
equal say.

Loop:

1. Start β = q = 0 and c, τ at small random values.
2. Update μ, β, c, q, τ in that order.
3. Stop when the largest parameter change is below 1e-9.

Two honest notes on the "EM" label:

- This is the hard version of EM: alternating maximisation, no posterior variances.
  The variances (the λ's) are chosen by cross-validation instead of an M-step.
- With one shared weight set the loss is convex and the loop reaches the same answer
  from any random start. With split weights (recency for players, none for teams) it is
  a fixed-point iteration: it should settle, but check it, and set damp = 0.5 if it
  oscillates.

## 8. Reference implementation

Implemented as `superr.model.fit_pt` (same signature and return values) on top of
`superr.model.prepare` / `superr.model.fit`, which add sample weights (for grouped CV and the
cluster bootstrap), warm starts, and switches to turn τ or q off.

Roughly 50k stints, so one fit takes well under a second. That makes the grid search
and the bootstrap cheap.

## 9. Hyperparameters

| Parameter | Start | How to set |
|---|---|---|
| λ_β, λ_c, λ_q, λ_τ | log grid, e.g. 30 to 30,000 | 5-fold CV grouped by match_id; minimise weighted squared error on held-out stints |
| k (stakes weight) | 1 | fix, then report sensitivity over {0, 0.5, 1, 2} |
| θ (pressure threshold) | 1 | fix, test {1, 1.5, 75th percentile of π} |
| h_max | 3 | fix, test {2, 3, none} |
| H (half-life) | 540 days | fix, test {365, 540, 730, none} |

- λ is in units of weighted balls. Expect λ_c to be large, because per-ball noise is
  far bigger than any real clutch effect.
- Tune only the λ's on fit. Tuning k, θ or H to make a list look right is p-hacking;
  report their sensitivity instead.
- If CV pushes λ_τ to the top of its grid, the data are saying T adds nothing. That is
  a finding to report, not a bug.

## 10. Scores, eligibility and ranking

```
P_i = 100 · Φ( c_i / s_c )      s_c = sd of c over eligible players in that role
T_t = 100 · Φ( τ_t / s_τ )      s_τ = sd of τ over teams
```

50 means no change under pressure; bottle jobs tend to 0, clutch to 100. Because c is
already shrunk, thin-sample players sit near 50 without any extra rule.

Eligibility (starting thresholds; set them from the exposure distributions):

- Batters: at least 20 pressure stints and 150 balls faced in them.
- Bowlers: at least 40 pressure overs.
- At least 5 pressure stints in the last 24 months, so "right now" has support.
- Delivers under pressure: β_i + c_i > 0, i.e. above the role average at h = 1. This
  stops a weak player who merely declines less from topping a "best under pressure" list.
- Selectable in the Unstop dropdown.

Rank eligible players by P within role; ties go to the higher bootstrap P(top 10).

Evidence table per ranked player: P, c, β, exposure E, pressure stints, raw mean y in
pressure versus ordinary stints, bootstrap interval for P, P(top 10), and their three
biggest stints by n·h·(y − μ − β + q_o).

## 11. Validation (model side; the data side is spec §7)

1. **Synthetic recovery.** Plant known c and τ on the real stint structure, add noise
   at the observed σ, refit. Report planted-versus-recovered correlation for P and T.
2. **Mover graph.** Connected components of the player–team graph, and the share of
   pressure exposure from players with 2+ teams.
3. **Start stability.** 20 random seeds; parameters should agree to the tolerance.
4. **One-shot check.** With split_weights=False, the loop must match a single sparse
   ridge solve of the same loss.
5. **Out-of-time.** Fit on matches to the end of 2024; test whether P predicts each
   player's 2025–26 pressure residuals (Spearman correlation). Expect a small number.
6. **Cluster bootstrap.** Resample matches 300 times, refit, and report P intervals
   and P(top 10).
7. **Shuffle test.** Permute h within each player; the spread of c should collapse.
8. **Ablations.** T off, q off, each row of section 9, entry_li in place of mean_li,
   emerging cups dropped. Report top-10 overlap with the main list for each.
9. **The CSK claim.** Distinct man_of_the_match winners per win, by team, from
   matches.csv, set beside T.

## 12. Risks and open decisions

1. **The clutch signal is small.** A plausible outcome is most P values near 50 and
   overlapping intervals in the top 10. Say so in the report.
2. **T may come out near zero.** The model tests "P depends on T" rather than assuming
   it. The rankings still come out; they just rest on P alone.
3. **mean_li is partly caused by the player.** The entry_li ablation shows how much this moves the list.
4. **Stint grain.** One pressure value for a 40-ball innings blurs the signal. Upgrade:
   cut batter stints at over boundaries; the equations are unchanged.
5. **Win probability quality.** Train a simple empirical wp model on the state table
   (grouped CV by match) and compare calibration. If it beats Jev, use it for wp and keep Jev for stakes.
6. **One T for all years** hides real changes in a franchise. Accepted by design.
7. **Opponent individuals are not modelled.** Opponents enter only through q and τ.
8. **Decisions to confirm:** per-role P, A-teams as separate teams, the "delivers" gate.

## 13. Report mapping

| Brief asks for | Where it comes from |
|---|---|
| Pressure definition | spec §5; sections 4–5 here; the top-1% LI read |
| Performance, baselines, rating logic | sections 5–7 and 10 |
| Fair comparisons | shrinkage E/(E+λ), eligibility gates, q term, canonical teams |
| Evidence for the top 10 | section 10 evidence table, bootstrap P(top 10) |
| Reliability and limitations | sections 11 and 12 |

---

## Implementation notes (what the code adds)

* **`extras_type = "other"` is a no-ball.** These are height no-balls (Beamer / Full Toss) in this
  pack: counting them legal gives 210 seven-ball overs (`data.noball_extras`).
* **Empirical situation providers** (risk 12.5) are built in. They run automatically when no Jev key is set:
  * the wp model is monotone gradient boosting, cross-fitted by match and Platt-calibrated;
  * innings 1 is trained on soft labels, the innings-2 model's wp at the start of the actual chase,
    so the two halves agree at the innings break.
* **End-game solver** (`superr/endgame.py`). For the last over of each innings, wp comes from exact
  backward induction over (runs, wickets, balls left). It uses the provider's next-ball probabilities,
  with exact terminal values in a chase and the innings-2 start value at the end of innings 1. A
  smooth model is flat there, which made hypothetical leverage ~0 while real outcomes still moved wp.
* **Interrupted innings.** A rain-shortened innings 1 (not all out, quota not used) is not handed to
  innings 2 on its last ball, and a DLS ending in innings 2 stays live, so no player is credited with
  the weather.
* **Synthetic recovery** plants effects as multiples of the CV-implied prior sd, √(σ²/λ_c), not of
  the (shrunk) fitted spread.
* **Shuffle tests.** h is permuted within player *and stint-length band*, which keeps exposure E
  roughly fixed; a residual sign-flip null is added alongside.
* **Stakes.** Group/phase structure comes from `config/tournaments.yaml`; groups are found as connected
  components of each phase's fixtures. Q1 is the first IPL "Qualifier" by date.
