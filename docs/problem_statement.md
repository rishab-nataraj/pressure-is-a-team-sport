# SupeRR Selector Season 2 - Round 1 problem statement (summary)

**Identify the most reliable players under pressure.**

Build a data-driven system that measures how batters and bowlers perform in high-pressure match
situations compared with their usual performance levels. Use it to identify and rank the players who
deliver most consistently when it matters.

**What counts as pressure** is for the entrant to define (required run rate, wickets in hand, balls
remaining, stage of an innings, closeness of the contest, other match context). Explain why the
definition is meaningful and show how it separates pressure situations from ordinary ones. The method
must work for both batting and bowling.

**Data:** IPL and T20I ball-by-ball data from 2022 onwards, plus competitions, teams, players, matches,
playing XIs and competition-team-player mappings.

**Submission**

1. *Player rankings* (Unstop dropdowns): top 10 batters and top 10 bowlers under pressure, rank 1 to
   10, no repeats within a list, matching the report.
2. *Methodology report*, covering:
   * pressure definition: situations and indicators used, and why
   * performance measurement: batting and bowling metrics, player baselines, rating/scoring logic
   * fair comparisons: sample sizes, eligibility, match contexts, use of IPL and T20I data
   * evidence for the top 10: results, comparisons, tables or charts supporting selection and order
   * reliability and limitations: consistency checks across matches, where the model is less certain

The methodology must be transparent enough to reproduce, and rankings must rest on analysis of the
supplied data, not reputation.

Where each item is answered: see the mapping table at the end of [`project_spec.md`](project_spec.md)
and the generated [`report/methodology.md`](../report/methodology.md).
