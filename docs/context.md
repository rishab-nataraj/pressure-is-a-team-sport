# Context: data pack and the idea behind the model

## Data pack

IPL + T20I, 2022-2026. Task: rank the top 10 batters and top 10 bowlers under pressure versus their
own baseline. All files join on IDs: comp_id, match_id, team_id, player_id
(= striker_id / non_striker_id / bowler_id / dismissed_player_id).

1. **competitions.csv** (141 rows): comp_id, comp_name, match_format {T20I = 134 comps | T20 = 7: IPL 2022-26 + 2 emerging/A-team Asia Cups}, season.
2. **teams.csv** (353 rows): comp_id, team_id, team_name (38 unique teams; IDs stable across comps).
3. **matches.csv** (883 rows; 482 T20I, 401 T20, 371 IPL): match_id, comp_id, match_number, match_date,
   stage {League, Others, Final, Qualifier, Super 4, Eliminator, Semi Final}, knock_out, venue_id,
   home/away team ids and names, captains, toss, man_of_the_match, first/second_inn_revised_overs and
   revised_target (DLS; overs values are balls, e.g. 114), win_team_id (5% null = no result), result text.
4. **playing_eleven.csv** (28,381 rows): match_id, team_id, player_id, batting_order, is_played,
   is_captain, is_wk, is_impact / is_impact_off.
5. **competition_team_player_mapping.csv** (6,413 rows): the only player-name lookup (1,217 players).
6. **ball_by_ball.csv** (201k rows, 867 matches): innings (1, 2; 3/4 = super over), overid,
   over_ball_num, innings_ball_num, batting/bowling team, striker / non-striker / bowler, runs, is_four,
   is_six, is_wicket, dismissal_type, dismissed_player_id, extras_type {wide, legbyes, byes, Overstep
   (= no-ball), other, Penalty}, extras1_runs, extras_2_runs, plus style and ball-detail fields.

Not provided (derived from ball_by_ball): running score, wickets fallen, target, required run rate,
balls remaining, partnership, current run rate. Ball total = runs + extras1_runs + extras_2_runs.
Bowlers aren't charged for byes/leg-byes; run-outs aren't bowler wickets. Phase: 1-6 powerplay,
7-15 middle, 16-20 death.

## The idea

**Team number T.** Each team has one number for how well it performs under pressure, 0-100 (bottle
jobs toward 0, clutch toward 100), one number across all competitions and years. A B-team touring as
"India" shares India's number; an IPL franchise has one number despite squad changes.

**Player number P.** Each player has one number for how well they handle pressure, 0-100, across all
teams and years.
* *Recency matters:* recent pressure situations count much more than old ones, because what matters
  is how the player handles pressure now.
* *Opportunity matters:* a player with many pressure situations (e.g. Sandeep Sharma) carries more
  weight than one who was clutch in only two.

**Performance** is whatever the situation needs. With 2 needed off the last ball, a six off a Bumrah
yorker and a two off a poor ball are equally good. Performance is not binary, and it accounts for how
much pressure there was.

**Pressure** is proxied by predicted win probability plus match context (playoff weight, or how
necessary the win is for qualification).

**The main idea.** A player's P is affected by his own team's T and the opponent's T. CSK backs its
players, so a different player comes up clutch in each match; CSK has many different men of the
match, where other teams with the same wins have one or two. "This team makes pressure easier" and
"this team makes players better" are the same thing; P and T inform each other, and each interaction
teaches us about both.

**Method.** Something like Expectation Maximisation: start T and P at random values and update each in
turn until they stabilise.
