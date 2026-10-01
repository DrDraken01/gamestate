# 0008 — Atomic prediction batches and a fail-fast orchestrator

**Date:** 2026-09-30
**Status:** Accepted

## Context

Predicting a week writes a `model_run` record plus one `predictions` row per game.
The original writer committed the predictions separately from the run insert,
which allowed two bad states: a committed run with zero predictions (an orphan),
or -- if an insert failed partway -- a committed run with a partial batch that an
application could mistake for an official week's predictions.

For a prediction engine this is worse than a crash. A missing prediction is a
visible signal that something broke; a silently partial or stale one is a lie
that looks legitimate and would quietly corrupt later accuracy evaluation.

## Decisions

**The transaction boundary belongs to the unit of work, not the individual
write.** A single prediction row is not a meaningful unit; a whole week's batch
plus its `model_run` record is. `write_prediction_batch()` opens one
transaction, inserts the run row and every prediction row inside it, and commits
only on clean exit. A failure anywhere rolls back BOTH -- zero rows, no orphan
run. Verified against PostgreSQL 16: a mid-batch failure (games 1-2 inserted,
game 3 a FK violation) left the table exactly as before, run included.

`write_predictions()` no longer calls `conn.commit()`. The caller owns the
boundary.

**Computation and validation happen OUTSIDE the transaction.** PMFs are produced
and validated before `with conn.transaction():` opens. A malformed distribution
(wrong length, not summing to 1, NaN) fails fast without touching the database,
and the transaction stays short -- two writes, no computation inside it.

**Accuracy over availability.** The orchestrator (`run_week.py`) runs stages
ingest -> predict in order and STOPS at the first failure. A broken ingest can
never feed a prediction; a failed predict publishes nothing new. Ingested data
that already committed stays committed -- it is valid regardless of what happens
downstream, and rolling it back would discard good work. What cannot happen is a
half-written prediction batch.

**The orchestrator surfaces failures; the persistence layer owns atomicity.**
`run_week.py` runs each stage as a subprocess and reads its exit code (0 = go on,
nonzero = stop). It knows nothing about transactions. All-or-nothing publication
lives entirely in `write_prediction_batch`. Clean separation: sequencing/failure
on one side, publication atomicity on the other. Subprocesses also give process
isolation and keep `load_db.py` / `predict.py` as untouched standalone commands.

## Explicitly deferred

Per-stage observability tables (`pipeline_run`, `pipeline_stage`) with
run status, failed stage, timings, and error codes are a good design and the
natural next step -- but they are a real schema addition and get their own ADR
plus migration. They are NOT stubbed here: a placeholder status table that does
not yet reflect reality would be exactly the "looks legitimate but isn't" failure
this change exists to prevent. Stage status is logged for now.

Safe automatic retries for transient failures (API timeout, temporary network
error) are also deferred; the distinction between transient and fatal failures is
worth encoding, but fail-fast on everything is the correct conservative default
until then.

## Consequences

- A week's predictions are all-or-nothing. No orphan runs, no partial batches.
- The weekly workflow is one command: `python scripts/run_week.py`.
- Validation failures cost nothing (fail before the transaction).
- When observability lands, `run_week.py` gains stage-status writes around each
  subprocess call, backed by the new tables -- an additive change.
