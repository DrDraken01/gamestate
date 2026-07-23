"""Modeling and evaluation.

Three ideas drive this module, all of which we discussed before writing code:

1. WALK-FORWARD, NEVER RANDOM SPLITS. A random train/test split lets the model
   learn from 2024 games to predict 2019 ones. That is time-travel, and it is
   just leakage wearing a different hat. We train on all seasons before S and
   predict season S, stepping forward one season at a time.

2. BASELINES ARE MANDATORY. A score means nothing alone. We always report:
     - base rate      : predict the historical home-win rate every game.
                        Perfectly calibrated, zero sharpness, zero skill.
     - market         : the closing spread, converted to a probability.
                        Calibrated AND sharp. Brutally hard to beat.
     - our model      : has to justify itself against both.

3. CALIBRATION AND SHARPNESS ARE SEPARATE PROPERTIES. Log loss and Brier
   score fold them together; the calibration table pulls them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from gamestate.weighting import effective_sample_size, recency_weights


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str = "rating_diff",
    half_life: float | None = None,
) -> np.ndarray:
    """Fit logistic regression on one feature and return P(home win) for test.

    Logistic regression, not gradient boosting, and not by accident. With a
    single feature there is nothing for a fancier model to discover, and a
    linear model gives us a readable coefficient -- which is worth more right
    now than a fraction of a point of log loss. Interpretability is a feature
    of the learning process, not just the product.
    """
    weights = recency_weights(train["season"], int(test["season"].iloc[0]), half_life)
    model = LogisticRegression()
    model.fit(train[[feature]], train["home_win"], sample_weight=weights)
    return model.predict_proba(test[[feature]])[:, 1]


def market_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Convert the closing spread into a win probability.

    The spread is in points, not probability, so it needs a mapping. We learn
    that mapping from training data with logistic regression -- which is also
    how you would discover the well-known rule of thumb that a 3-point
    favorite wins around 60% of the time.
    """
    usable = train.dropna(subset=["spread_line"])
    model = LogisticRegression()
    model.fit(usable[["spread_line"]], usable["home_win"])

    spreads = test[["spread_line"]].fillna(0.0)
    return model.predict_proba(spreads)[:, 1]


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Bucket predictions and compare predicted vs actual rates.

    This is the diagnostic that aggregate metrics hide. A model can post a
    fine log loss while being systematically overconfident in one bucket and
    underconfident in another, with the errors cancelling out.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(y_prob[mask].mean()),
                "actual": float(y_true[mask].mean()),
            }
        )

    table = pd.DataFrame(rows)
    table["gap"] = table["actual"] - table["predicted"]
    return table


def score(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Log loss, Brier, and sharpness. Lower is better for the first two.

    Sharpness is the standard deviation of the predictions themselves. It
    involves no outcomes at all -- it purely measures whether the model
    commits to distinguishing games from one another. A model that outputs
    0.57 every time has sharpness 0 no matter how the games turn out.
    """
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sharpness": float(np.std(y_prob)),
    }


def walk_forward(
    frame: pd.DataFrame,
    start_season: int = 2010,
    min_train_seasons: int = 5,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Backtest season by season, returning per-game predictions.

    For each season S >= start_season: train on everything before S, predict S.
    This mirrors how the model would actually be used -- you never have future
    seasons available at prediction time.
    """
    results = []
    seasons = sorted(s for s in frame["season"].unique() if s >= start_season)

    for season in seasons:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]

        if train["season"].nunique() < min_train_seasons or test.empty:
            continue

        block = test[["game_id", "season", "home_win", "spread_line"]].copy()
        block["p_model"] = fit_predict(train, test, half_life=half_life)
        block["ess"] = effective_sample_size(recency_weights(train["season"], season, half_life))
        block["p_market"] = market_predict(train, test)
        # The base rate is computed from TRAINING data only. Using the test
        # season's own rate would leak the answer into the baseline.
        block["p_baserate"] = train["home_win"].mean()
        results.append(block)

    return pd.concat(results, ignore_index=True)


def skill_score(y_true: np.ndarray, y_prob: np.ndarray, y_ref: np.ndarray) -> float:
    """Fractional improvement in Brier score over a reference forecast.

    1.0 is perfect, 0.0 means "no better than the reference," negative means
    worse. Reporting skill rather than a raw score forces the honest question:
    better than WHAT?
    """
    return 1.0 - brier_score_loss(y_true, y_prob) / brier_score_loss(y_true, y_ref)
