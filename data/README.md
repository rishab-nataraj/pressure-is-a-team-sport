# Data

Put the six data-pack CSVs in `data/raw/` (they are git-ignored):

```
data/raw/ball_by_ball.csv
data/raw/competition_team_player_mapping.csv
data/raw/competitions.csv
data/raw/matches.csv
data/raw/playing_eleven.csv
data/raw/teams.csv
```

Generated folders:

| Folder | Contents |
|---|---|
| `data/interim/` | parquet tables from each stage (`states`, `situations`, `balls`, `stints`, `fit_*`, `bootstrap.npz`) |
| `data/jev_cache/jev_cache.sqlite` | every raw Jev request/response, and every answer keyed by sha256(model, state, question). Re-runs never re-bill a call. |

Facts about the pack that the code relies on (checked on the Nov 2026 pack):

* `extras_type == "other"` is a height no-ball (Beamer / Full Toss), so it is not a legal ball.
* `first_inn_revised_overs` / `second_inn_revised_overs` are already in balls (e.g. 114 = 19 overs).
* Team IDs are stable across competitions; team names are canonicalised anyway (`config/team_aliases.yaml`).
* 16 matches have no ball-by-ball; 24 more are no-results; both are excluded. One tie without a super-over
  winner is kept (terminal value 0.5).
* Every striker and bowler in `ball_by_ball.csv` is present in `competition_team_player_mapping.csv`.
