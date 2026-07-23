# 0004 — Model the margin distribution, not the binary outcome

**Date:** 2026-07-23
**Status:** Accepted (with a documented limitation)

## Context

The manifesto asks for probabilities at arbitrary thresholds — "over 275
passing yards, 66%" — not single predictions. A binary win model returns one
scalar and cannot answer any other question. Threshold queries require a
predictive *distribution*.

This had to be settled before the database schema. A `p_home_win FLOAT` column
cannot store a distribution, and discovering that after the schema is live is
an expensive migration.

## Decision

Model margin as a distribution. Every team-level threshold is then a query
against one fitted object:

```
P(home win)         = P(margin > 0)
P(home covers -3.5) = P(margin > 3.5)
P(total > 47.5)     — same idea, once totals are modelled
```

Linear regression for the mean, residuals for the spread, same half-life 3
recency weighting as ADR-0002.

**Validation — is anything lost versus modelling the binary outcome directly?**
4,338-game walk-forward backtest, 2010–2025:

| method | log loss | Brier | sharpness |
|---|---|---|---|
| binary logistic | 0.6404 | 0.2248 | 0.147 |
| margin → Normal | 0.6405 | 0.2248 | 0.135 |
| margin → Empirical | 0.6405 | 0.2249 | 0.143 |
| market | 0.6089 | 0.2108 | 0.181 |

Identical to three decimal places. Threshold flexibility is free.

## The limitation, which is the important part

Both representations fail at **key numbers**. Fitted σ ≈ 13.8.

| margin | Normal | Empirical | ACTUAL |
|---|---|---|---|
| +3 | 0.028 | 0.029 | **0.076** |
| −3 | 0.026 | 0.027 | **0.070** |
| +7 | 0.026 | 0.027 | **0.045** |

The initial hypothesis was that resampling empirical residuals would preserve
the spikes. It does not, and the reason generalises:

**Shift-based models assume the distribution translates with μ — same shape,
moved centre. Key numbers do not translate.** They are anchored at zero,
because they come from how points are scored rather than from who is favoured:

| spread | most common margin |
|---|---|
| ~0 | +3 (7.6%) |
| home −3 | +3 (9.5%) |
| home −7 | +3 (8.2%) |
| home −10 | +3 (7.0%) |

Even at a ten-point spread, three is still the single likeliest margin. No
amount of feature engineering or σ tuning fixes this; the defect is in the
shape assumption, not the location estimate.

Practical cost: moving a line from 2.5 to 3.5 should swing cover probability
by 7.6 points. Our model swings 2.8. Any product surface showing spread
probabilities near 3 or 7, or any push probability, would be materially wrong.

## Consequences

- **Safe now:** win probability, and thresholds away from 3 and 7.
- **Not safe:** spread pricing at key numbers, push probabilities. Do not ship.
- `tests/test_distribution.py::test_shift_based_models_cannot_represent_key_numbers`
  asserts the defect so nobody "fixes" it by tuning σ.
- **Monte Carlo moves from stage 7 to stage 4.** Modelling team scores — built
  from 3s and 7s — and differencing them produces key numbers because the
  generating process puts them there. This result is the argument for it.
- Schema must store distribution parameters (μ, σ, and later simulation
  draws), not a single probability.

## What this validated

A hypothesis stated in the module docstring was tested against 4,338 games and
found wrong. The negative result reshaped the roadmap. Worth more than if it
had quietly worked.
