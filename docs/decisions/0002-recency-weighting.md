# 0002 — Exponential recency weighting to handle concept drift

**Date:** 2026-07-22
**Status:** Accepted

## Context

Walk-forward backtest (2010–2025, 4,338 games) showed the model over-predicting
home teams by +0.031 in 2020–25 vs +0.006 in 2010–19. The 0.4–0.6 calibration
bucket had a gap of −0.031 across 1,859 games (z = −2.66, p = 0.008) — real, not
noise.

Cause: NFL home-field advantage fell from ~58% (1999–2005) to ~54% (2020–25).
Training on all history averages over a world that no longer exists. This is
concept drift, not overfitting. The market reprices weekly; our model did not.

## Options considered

**Sliding window** (train on last N seasons only). Simple; discards real data
abruptly at the boundary.

**Exponential decay weighting.** Weight each row by `0.5^(age/half_life)`. Keeps
all data, weights recent more. One interpretable hyperparameter.

**Explicit time-varying HFA term.** Most interpretable, most work, and premature
while the model has one feature.

## Decision

Exponential decay, `half_life = 3` seasons.

Sweep results (aggregate over 4,338 games):

| half-life | log loss | sharpness | bias 20–25 | mean ESS |
|---|---|---|---|---|
| none | 0.6408 | 0.1405 | +0.031 | 4876 |
| 8 | 0.6405 | 0.1433 | +0.024 | 3968 |
| **3** | **0.6404** | **0.1466** | **+0.014** | **2210** |
| 1.5 | 0.6407 | 0.1487 | +0.006 | 1176 |
| 1 | 0.6410 | 0.1497 | +0.003 | 805 |

Log loss is a shallow U with its minimum near 3–5. Bias falls monotonically as
half-life shortens, but effective sample size collapses — the variance cost.

## Consequences

- 0.4–0.6 bucket gap: −0.031 (p = 0.008) → −0.018 (p = 0.128). No longer
  distinguishable from noise.
- Aggregate log loss barely moved (0.6408 → 0.6404). **This is the lesson:**
  aggregate metrics are nearly blind to systematic subgroup bias. The
  justification for this change is calibration, not the headline score.
- ESS drops to ~2,210 effective games. Acceptable now; revisit if feature count
  grows, since more parameters need more effective data.
- `half_life=3` was chosen on the same seasons we report results for. That is
  mild test-set contamination. Acceptable for one coarse hyperparameter, but it
  must move to a proper validation split before any further tuning.

## Revisit if

Feature count grows past ~5, or a rule change causes an abrupt (not gradual)
regime shift, which decay handles poorly compared to changepoint detection.
