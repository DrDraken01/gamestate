"""Margin as a distribution, not a point estimate.

WHY THIS REPLACES THE BINARY WIN MODEL

The binary model answers exactly one question: P(home win). The manifesto wants
arbitrary thresholds -- "over 47.5 total points, 62%", "home covers -3.5, 54%".
Those are not more win-probability models. They are queries against a
DISTRIBUTION, and you cannot recover a distribution from a scalar.

So we model the margin distribution once and read every threshold off it:

    P(home win)          = P(margin > 0)
    P(home covers -3.5)  = P(margin > 3.5)
    P(margin in [3, 7])  = F(7) - F(3-)

One model, unlimited queries. This is also why it has to come BEFORE the
database schema -- a `p_home_win FLOAT` column cannot store this.

TWO WAYS TO REPRESENT THE DISTRIBUTION, AND WHY WE KEEP BOTH

NFL margins are not smooth. Empirically, 1999-2025:

    margin of exactly 3 : 15.1% of games
    margin of exactly 7 :  9.0% of games
    a Normal predicts   : ~11.5% for BOTH COMBINED

Scoring is lumpy because points arrive in 3s and 7s. These are the "key
numbers" every football bettor knows about, and they are a real feature of the
data, not noise.

Consequence, and it is the whole reason this module has two classes:

  * NormalMargin is fine for thresholds away from key numbers, and for
    P(margin > 0) specifically, because the lumpiness is roughly symmetric
    about zero and averages out.
  * NormalMargin is BADLY WRONG at a line of exactly 3 or 7, where the true
    distribution has a spike and a Normal has none. It will understate push
    probability and misprice the most common spreads in football.

EmpiricalMargin resamples actual historical residuals instead of assuming a
shape. THIS DOES NOT FIX KEY NUMBERS -- and finding out why was worth more
than if it had worked. Measured on the 4,338-game backtest:

    P(margin == +3)   Normal 0.028   Empirical 0.029   ACTUAL 0.076

Residual resampling assumes the distribution TRANSLATES with mu: same shape,
shifted centre. Key numbers do not translate. They are anchored at zero,
because they come from how points are scored, not from who is favoured:

    spread ~0   most common margin: +3 (7.6%)
    home -3     most common margin: +3 (9.5%)
    home -7     most common margin: +3 (8.2%)
    home -10    most common margin: +3 (7.0%)

Even at a ten-point spread, three is still the single most likely margin. Any
shift-based model is structurally incapable of representing that, and adding
more features will not help, because the defect is in the shape assumption
rather than in the location estimate.

The fix is to model the SCORING PROCESS -- team scores, built from 3s and 7s
-- and difference them. Margins then land on key numbers because the
generating process puts them there. That is Monte Carlo simulation, and this
result is the argument for pulling it forward from stage 7 to roughly stage 4.

WHAT EACH CLASS IS SAFE FOR TODAY

    NormalMargin      P(margin > 0), and thresholds away from 3 and 7.
                      Validated: win probability matches the direct binary
                      logistic model (log loss 0.6405 vs 0.6404).
    EmpiricalMargin   marginally better in the tails; same key-number defect.
    NEITHER           pricing a line at exactly 3 or 7, or any push
                      probability. Do not ship those until simulation lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from gamestate.weighting import recency_weights


@dataclass(frozen=True)
class NormalMargin:
    """Margin ~ Normal(mu, sigma). Smooth, closed-form, no key numbers."""

    mu: np.ndarray
    sigma: float

    def prob_over(self, line: float | np.ndarray = 0.0) -> np.ndarray:
        """P(margin > line). With line=0 this is win probability."""
        return np.asarray(stats.norm.sf(line, loc=self.mu, scale=self.sigma))

    def quantile(self, q: float) -> np.ndarray:
        """The margin at probability level q. quantile(0.5) is the median."""
        return np.asarray(stats.norm.ppf(q, loc=self.mu, scale=self.sigma))


@dataclass(frozen=True)
class EmpiricalMargin:
    """Margin ~ mu + a randomly drawn historical residual.

    Assumes residuals are exchangeable: the SHAPE of the error is identical
    for a projected blowout and a projected coin-flip, only the centre moves.

    That assumption is what breaks on key numbers. mu is continuous, the
    residuals are continuous, and their sum smears across the real line --
    destroying exactly the integer lumpiness we wanted to keep. Measured, not
    assumed: see the module docstring.

    Kept because it is still slightly better than a Normal in the tails and
    it makes the failure legible. Do not use it for push probabilities.
    """

    mu: np.ndarray
    residuals: np.ndarray

    def prob_over(self, line: float | np.ndarray = 0.0) -> np.ndarray:
        """P(margin > line), by counting residuals rather than integrating.

        For each game: what fraction of historical residuals, added to this
        game's mu, land above the line? Vectorised as an outer comparison.
        """
        line_arr = np.broadcast_to(np.asarray(line, dtype=float), self.mu.shape)
        shifted = self.mu[:, None] + self.residuals[None, :]
        return (shifted > line_arr[:, None]).mean(axis=1)

    def prob_exactly(self, value: int) -> np.ndarray:
        """P(margin == value). Always 0 under a Normal; meaningful here.

        WARNING: measured against 4,338 games this understates key numbers by
        roughly 5 percentage points at margin 3. It is retained to make that
        gap measurable, not because the number is trustworthy. Do not surface
        it in the product until score simulation replaces this.
        """
        shifted = np.rint(self.mu[:, None] + self.residuals[None, :])
        return (shifted == value).mean(axis=1)


def fit_margin_model(
    train: pd.DataFrame,
    target_season: int,
    features: list[str],
    half_life: float | None = 3.0,
) -> tuple[LinearRegression, np.ndarray, float]:
    """Fit E[margin | features] and capture the residual distribution.

    Returns the fitted model, its training residuals, and their standard
    deviation. Linear regression on the mean, then the residuals ARE the
    uncertainty estimate -- we do not assume a sigma, we measure one.

    Same recency weighting as the win model, for the same reason: home-field
    advantage has drifted and old seasons describe a different sport.
    """
    weights = recency_weights(train["season"], target_season, half_life)

    model = LinearRegression()
    model.fit(train[features], train["margin"], sample_weight=weights)

    residuals = train["margin"].to_numpy() - model.predict(train[features])

    # Weighted SD, so sigma reflects the same era emphasis as the mean.
    mean_resid = np.average(residuals, weights=weights)
    sigma = float(np.sqrt(np.average((residuals - mean_resid) ** 2, weights=weights)))

    return model, residuals, sigma


def predict_distributions(
    model: LinearRegression,
    test: pd.DataFrame,
    features: list[str],
    residuals: np.ndarray,
    sigma: float,
) -> tuple[NormalMargin, EmpiricalMargin]:
    """Produce both distribution representations for the test games."""
    mu = np.asarray(model.predict(test[features]))
    return NormalMargin(mu=mu, sigma=sigma), EmpiricalMargin(mu=mu, residuals=residuals)
