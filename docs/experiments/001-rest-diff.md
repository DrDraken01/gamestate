# Experiment 001 — Rest Differential

| field | value |
|---|---|
| Experiment | 001 |
| Feature | rest_diff |
| **Status** | **REJECTED** |
| Baseline | rating_diff |
| Model version | v0.x |
| Primary metric | walk-forward log loss |
| Result | +0.00007 (10/17 seasons helped; season variance >> mean effect) |

Status values used across experiments: ACCEPTED / REJECTED / INCONCLUSIVE.
INCONCLUSIVE is distinct from REJECTED -- it means the evidence did not support
either call, not that the feature failed.

---

**Date:** 2026-09-30
**Model version:** v0.x (logistic on margin-derived win prob, half_life=3)
**Status:** REJECTED — no credible out-of-sample improvement

## Hypothesis

Pregame rest differential carries incremental predictive information about NFL
game outcomes beyond prior-form `rating_diff`.

## Design

```
rest_diff = home_rest - away_rest     (positive = home better-rested)
```

- **Baseline features:** `[rating_diff]`
- **Candidate features:** `[rating_diff, rest_diff]`
- **Identical** walk-forward folds, training windows, half_life (3), and model
  config. Only the feature set differs. Folds asserted identical game-for-game.
- Backtest: 2010–2025 (+ partial 2026), 4,338 games.

## Data quality (passed)

- `home_rest`/`away_rest`: zero nulls across 7,323 completed games.
- Distribution matches football reality: spike at 7 (normal week), cluster at 4
  (Thursday), bumps at 10/13/14 (post-bye). No garbage values.
- **rest_diff is 0 on 67% of games** (both teams on 7 days). The feature can
  only act on the ~33% where rest differs — a structural dilution worth noting.

## Leakage check (passed)

Rest is known from the schedule before kickoff. No future information. The
feature is computable strictly pre-game.

## Results

| metric | baseline | candidate | delta (+ = better) |
|---|---|---|---|
| log loss | 0.64099 | 0.64092 | **+0.00007** |
| Brier | 0.22505 | 0.22502 | +0.00003 |

Per-season log-loss delta:

- **Seasons helped: 10/17. Hurt: 7/17.**
- Mean delta: **+0.000186**
- Std across seasons: **0.001827** — ~10x the mean.
- Per-season swings (±0.003) dwarf the aggregate effect.

## Decision: REJECT

The aggregate improvement is ~0.01%, and the per-season effect is a coin-flip
whose variance is an order of magnitude larger than its mean. This is the
signature of noise, not signal. A feature carrying real incremental information
would help consistently and same-signed across folds; this does not. By the
feature-admission rule (keep only on credible out-of-sample improvement),
`rest_diff` as specified is dropped.

## What this does and does NOT conclude

- **Does:** this specific linear encoding of rest added no credible value *on
  top of rating_diff* under model v0.x.
- **Does NOT:** conclude rest is irrelevant to football. Two untested
  alternatives remain live hypotheses for future experiments:
  1. Nonlinear encodings — `home_short_week`, `away_off_bye` flags — which
     capture threshold effects a linear `rest_diff` cannot.
  2. Rest may matter in isolation but be partly absorbed by `rating_diff`
     (a rested team that played well is already reflected in its prior form).

## Reproducibility

Run against the games table / nflverse cache, `build_features` with rest_diff,
`walk_forward(frame, [...], half_life=3)`. Baseline and candidate differ only in
the feature list argument.
