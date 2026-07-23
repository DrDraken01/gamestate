"""Recency weighting: let old seasons count for less.

WHY THIS EXISTS

Our backtest showed the model over-predicting home teams by +0.031 in 2020-25,
while being nearly unbiased in 2010-19. Cause: home-field advantage has fallen
from ~58% to ~54%, and a model trained on all history is averaging over a world
that no longer exists. That is concept drift, not overfitting.

THE TRADEOFF

Down-weighting old data adapts faster but estimates from less effective data,
so it is noisier. This is the bias-variance tradeoff showing up as a question
about TIME rather than about model complexity. The half-life parameter is
literally an answer to "how fast does football change?"

HALF-LIFE, NOT DECAY RATE

We parameterise by half-life in seasons because it is interpretable: a
half-life of 4 means a season from 4 years ago counts half as much as this
one. A raw decay constant of 0.841 means the same thing and tells you nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(seasons: pd.Series, target_season: int, half_life: float | None) -> np.ndarray:
    """Exponential decay weights by season age.

    Args:
        seasons: season of each training row.
        target_season: the season being predicted. Age is measured from here.
        half_life: seasons until a row's weight halves. None disables decay
            (all rows weight 1.0), which is the old behaviour and our control.

    A row from `half_life` seasons ago gets weight 0.5, from 2*half_life ago
    0.25, and so on. Weights are never exactly zero, so old data still nudges
    the fit instead of falling off a cliff -- which is the main advantage over
    a hard sliding window.
    """
    if half_life is None:
        return np.ones(len(seasons), dtype=float)
    if half_life <= 0:
        raise ValueError("half_life must be positive, or None to disable decay")

    age = (target_season - seasons).to_numpy(dtype=float)
    return 0.5 ** (age / half_life)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    Tells you how many equally-weighted games your weighted fit is really
    worth. With 4,000 rows and a 2-season half-life you might have an ESS of
    ~900 -- that gap IS the variance cost you are paying for adaptivity, and
    it is the number to look at when a short half-life starts scoring worse.
    """
    total = weights.sum()
    return float(total * total / np.square(weights).sum())
