# Jev Pressure Data Spec (SupeRR Selector S2, Round 1)

Status: draft v1, 18 Nov 2026. Implemented by `superr/states.py`, `superr/jev/`, `superr/jev_stage.py`,
`superr/standings.py`, `superr/balls.py`, `superr/stints.py` and `superr/validate_data.py`.

## Purpose

What to ask Jev (TypeSafe AI) and what to store, so a single dataset can feed the P (player) and T
(team) model. Jev only judges situations. Our code computes everything about players and teams
(performance, baselines, P and T) from Jev's numbers and the ball-by-ball data.

## Principles

1. **Jev never sees who is playing.** No team names, player names, venue names, dates or match IDs
   go into a Jev call. This prevents three problems:
   - Jev recognises a match and already knows the result.
   - Jev's opinion of a team leaks into pressure before the EM sees it, so T gets counted twice.
   - Rankings end up resting on reputation, which the brief rules out.
2. **Ask for raw ingredients, not a blended "pressure" score.** Win probability, next-ball
   probabilities and stakes come back separately and are combined in code.
3. **Call per ball, then aggregate.** Batter innings and bowler overs are built afterwards.
4. **Make it reproducible.** Pin the Jev model version, cache every response under a hash of its
   input, and keep the raw responses on disk.

## 1. State table (built from ball_by_ball.csv, no Jev)

One row per delivery, ordered by match_id, innings, innings_ball_num. Super overs (innings 3/4) are
excluded from P and T. Matches with a null win_team_id (no result) are excluded.

Derived columns, describing the state *before* the delivery:

| Field | How |
|---|---|
| runs | cumulative ball totals (runs + extras1_runs + extras_2_runs) |
| wickets | cumulative is_wicket |
| legal_balls | cumulative count where extras_type not in {wide, Overstep} (and `other`, see notes) |
| max_balls | 120, or the revised overs field for DLS games (already in balls, e.g. 114) |
| target | innings-1 total + 1, or revised_target; null in innings 1 |
| balls_left, runs_needed | max_balls − legal_balls; target − runs |
| free_hit | true if the previous delivery was a no-ball (carries through wides and further no-balls) |
| partnership_runs, partnership_balls | reset at each wicket |
| striker_runs/balls, non_striker_runs/balls | from striker_id / non_striker_id |
| format | IPL, T20I, or other T20 (the two emerging/A-team Asia Cups) |
| impact_rule | true for IPL 2023 onwards (Impact Player era scores higher) |
| venue_par | average first-innings total at that venue_id across the dataset; a number, not a name |

## 2. Jev call A: win probability of a state

**Input:** the state fields above as JSON, with no identities. **Question:** `wp_bat`, the
probability that the batting side wins the match.

**States to query:** every real pre-delivery state, and for each real state the hypothetical state
after each possible next-ball outcome:

| Outcome | Change to state |
|---|---|
| dot | +1 legal ball |
| 1, 2, 3, 4, 6 | +runs, +1 legal ball, strike swaps on 1 and 3 |
| wicket | +1 wicket, +1 legal ball, new batter on 0 (0), partnership reset |
| wide | +1 run, no legal ball |
| no-ball | +1 run, no legal ball, free_hit = true |

Every number comes from the same function, so the numbers agree with each other. The actual next
state is usually one of the hypotheticals, which gives a free consistency check.

Terminal states don't go to Jev: chase completed → 1; innings 2 finished short → 0; tie → 0.5; end
of innings 1 → the wp of the innings-2 starting state, flipped (1 − wp).

## 3. Jev call B: next-ball outcome probabilities

**Question:** `p_next`, the probability of each outcome class above (summing to 1). Check against
empirical frequencies by format × phase × wickets band; fall back to the empirical table if badly off.

## 4. Jev call C: match stakes

One call per team per match. Teams are labelled A, B, C, with the focal team marked.
**Input:** stage, competition type, standings before the match (played, won, lost, NR, points, NRR),
remaining fixtures, qualifying places; for bilaterals the series score and matches left.
**Questions:** `p_advance_if_win`, `p_advance_if_lose`. **Stakes = p_advance_if_win − p_advance_if_lose.**

| Stage | Advance = | Notes |
|---|---|---|
| IPL league | reach playoffs | Jev call |
| Tournament group / Super 4 / Super 8 | reach next round | Jev call |
| Bilateral | win (or not lose) the series | decider = 1, dead rubber = 0; no call needed |
| Qualifier 1 | reach final | the loser still gets Qualifier 2, so stakes ≈ 0.5 |
| Eliminator, Qualifier 2, Semi Final | reach final | 1 |
| Final | win title | 1 |

Cross-check IPL league stakes with a Monte Carlo of the remaining fixtures (coin-flip results, ranked
on points, then NRR), and compare p_advance with whether teams actually advanced.

## 5. Per-ball derived fields (our code)

- `wp_before` is call A on the real state; `wp_after` is call A on the real next state, or the
  terminal value. Both from the batting side's point of view.
- **WPA_bat** = wp_after − wp_before. **WPA_bowl** = −WPA_bat.
- **Leverage** L = Σ_o p_next[o] × |wp(o) − wp_before|. **LI** = L / mean L, floored at 0.1 when dividing.
- **Pressure** = LI × (1 + k·stakes); k = 1, sensitivity over {0, 0.5, 1, 2}.

Attribution:
- **Batter:** WPA_bat on every ball faced as striker, except wides (they belong to the bowler).
- **Non-striker run-out:** the WPA goes to dismissed_player_id.
- **Bowler:** WPA_bowl on every ball bowled; on run-outs, charged as if only the runs happened
  (the wp of the matching runs-only hypothetical state).

## 6. Stint table (what P and T consume)

**Batter stint:** first ball at the crease (striker or non-striker) until dismissal, retirement, end
of innings or end of match. **Bowler stint:** one over by one bowler; a bowler leaving mid-over keeps
only his balls.

Columns: stint_id, player_id, role, match_id, comp_id, match_date, team_canon, opp_canon, format,
impact_rule, n_balls, sum_wpa, sum_wpa_li, sum_li, mean_li, max_li, entry_wp, entry_li, exit_wp,
exit_li, stakes, pressure, end_reason.

## 7. Validation

1. Calibration of wp_before by decile vs actual winners.
2. Martingale check: mean WPA ≈ 0 overall and by phase.
3. Sanity: more runs → higher wp, a wicket → lower; count violations.
4. p_next vs empirical frequencies.
5. Stakes vs actual advancement.
6. Read the top 1% of balls by LI: mostly last overs of close chases.
7. Reproducibility: re-run a 1% sample live on the pinned model and compare with the cache.

## 8. Volume

About 201k real states (fewer after exclusions), 9 hypotheticals each (≈1.8M before
de-duplication), one call B per real state, at most 1,766 call C.

## 9. Known limitations

- wp judges an average team (deliberately, so T is learned rather than assumed).
- Super overs are excluded.
- Stakes for bilaterals and minor tournaments are coarser than for the IPL.
- Byes and leg-byes count in bowler WPA (switchable to the no-bye counterfactual).
- Team canonicalisation (whether India A counts as India) is a modelling choice.

## Implementation notes

- **Jev API used:** `POST /v1/systemone` with `{"model", "state", "questions"}`:
  - call A asks one `noul` question (answer `noul` = P(yes));
  - call B asks one `choice` question over the nine classes (answer `probabilities`);
  - call C asks two `noul` questions.
- **Batching:** a real state's A and B go in one request.
- **Caching:** answers are cached per (model, state, question), so a hypothetical that equals another
  real state is never asked twice.
- **Legal balls:** the pack's `extras_type = "other"` is treated as a no-ball.
- **State shape:** hypothetical states also swap strike at the end of an over. A wicket hypothetical
  puts the new batter on strike.
- **Interrupted innings:** a rain-interrupted innings 1 (incomplete) is not handed to innings 2 on its
  last ball; a DLS ending in innings 2 stays a live state.
