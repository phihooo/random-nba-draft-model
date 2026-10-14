# Position-stratified analysis

## Why positions are separated

College roles create structural confounding. A center's normal rebounding,
blocking, and shooting profile differs from a point guard's, so one pooled
correlation can average together relationships that have different strengths or
directions. This project now publishes:

- pooled correlations (`ALL`);
- PG, SG, SF, PF, and C correlations;
- a `WITHIN_POSITION` aggregate after removing each position's mean; and
- draft-pick-adjusted partial correlations for every scope.

Position splitting improves interpretation, not sample size. The subgroup
estimates have wider uncertainty and must be read with their `n`, confidence
interval, and FDR-adjusted q-value.

## Canonical assignment

`position_utils.py` applies this precedence:

1. First Basketball Reference NBA position recorded after the player's draft
   year, requiring a season with games played.
2. For players without an NBA appearance, a regularized position classifier
   using only college height, Barttorvik role, and pre-NBA statistics.
3. A college-role mapping only when a future row has too few numeric statistics
   for the classifier.
4. `Unassigned` when no local evidence exists.

Every derived row retains:

- `nba_entry_position` — direct NBA label, blank for inferred players;
- `primary_position` — the final five-way descriptive grouping;
- `position_source` — `nba_rookie_season`, `college_position_model`, or
  `college_role`;
- `position_confidence`; and
- `position_observed_season`.

The current 896-player draft table contains 808 direct labels, 35 college-model
fallbacks, and 53 unassigned draft-and-stash players. All 682 college-matched
analysis rows are assigned.

For direct NBA labels, confidence is `1.0`. For college-model fallbacks it is
the selected class probability; this probability is useful for ranking
assignment ambiguity but is not formally calibrated.

## Identity controls

NBA season joins retain Basketball Reference's `Player-additional` ID, select
combined traded-team rows before team-specific rows, and require the matched NBA
career to begin after the draft year. This prevents same-name collisions such as:

- Marcus Thornton (2009 LSU) vs. Marcus Thornton (2015 William & Mary); and
- Justin Jackson (2017 UNC) vs. Justin Jackson (2018 Maryland).

The same ID-aware logic is used to construct peak and common-horizon VORP
targets, so position stratification and NBA outcomes refer to the same player.

## Predictive-use warning

`primary_position` is a historical descriptive grouping because its preferred
source is the first NBA season. It is never passed into the leakage-safe model as
a predictor. A future production draft model would need an authoritative
pre-draft position source or a position model fitted entirely inside each
training fold.
