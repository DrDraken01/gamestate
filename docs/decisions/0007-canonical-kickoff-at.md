# 0007 — `kickoff_at` is the canonical time column

**Date:** 2026-09-30
**Status:** Accepted

## Context

Wiring the predictor to read from the database surfaced a column-name mismatch:
`features.py` expected `gameday` (nflverse's name), but the database read returns
`kickoff_at` (the schema's name, ADR-0005). The feature layer predated the
database and still spoke the flat-CSV dialect. Two ingestion paths now exist --
nflverse-direct (`ingest.completed_games`) and database (`db/read`) -- and they
disagreed on what the time column was called.

## Decision

**`kickoff_at` is the canonical name for the game-time column everywhere past
the ingestion boundary.** `gameday` survives only where we handle raw nflverse
data.

Why `kickoff_at` and not `gameday`:
- It is honest about what it holds: a timezone-aware instant (TIMESTAMPTZ in the
  DB), which is the leakage boundary the entire feature layer depends on.
  `gameday` is a bare date and a source-specific name.
- The database already committed to it (ADR-0005). Aligning the code up to the
  schema -- rather than renaming the schema down to the legacy CSV name -- keeps
  the most precise name as the canonical one.

**The boundary principle:** a raw source's column names are renamed exactly once,
at the point we ingest that source, so everything downstream sees one vocabulary.
- `db/load.build_kickoff` already did this (reads raw `gameday`+`gametime`,
  writes `kickoff_at`).
- `ingest.completed_games` now does the same for the nflverse-direct path.
- Result: both paths hand the feature layer identical column names, so a module
  like `features.py` has exactly one contract to satisfy.

Note: nflverse `gameday` is a date, not a true kickoff instant. The
nflverse-direct path renames the date to `kickoff_at` because the feature layer
only needs chronological order within a team's schedule; the database path
stores the real timezone-aware kickoff. Same name, acceptable precision
difference for ordering.

**Related correction:** `spread_line` was removed from `features.PASSTHROUGH`.
The betting line lives in the `market_lines` table (ADR-0005), not in `games`,
because lines move over time. Carrying `spread_line` through was a vestige of the
flat-CSV shape. When a market feature is built, it will join `market_lines` with
an explicit captured-line choice, not ride through on a passthrough column.

## Consequences

- `features.py` input contract now matches exactly what both ingestion paths
  provide. Verified by diffing every column the module touches against the read
  output -- `spread_line` was the only orphan, now removed.
- The rename was behaviour-preserving: 36 tests pass, and the full pipeline runs
  end to end, producing week-4 2026 win probabilities (MIN 0.846 ... DET 0.534).

## What this validated

The rename surfaced two latent seam bugs (`gameday`, then `spread_line`) that
were silently waiting for the two components to meet. The lesson is the method
that caught them: when connecting two modules, diff the full column contract --
every field one side reads against every field the other provides -- rather than
fixing missing-column crashes one traceback at a time. A half-renamed pipeline is
an inconsistent one; the tests (especially the leakage guards) refused to pass
until the rename was complete, which is exactly their job.
