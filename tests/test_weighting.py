"""Tests for recency weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamestate.weighting import effective_sample_size, recency_weights


def test_half_life_halves_weight() -> None:
    """A row exactly one half-life old must weigh exactly 0.5."""
    w = recency_weights(pd.Series([2024, 2020, 2016]), target_season=2024, half_life=4)
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25])


def test_none_disables_decay() -> None:
    """None is the control condition -- every row weighs 1.0."""
    w = recency_weights(pd.Series([1999, 2024]), target_season=2024, half_life=None)
    np.testing.assert_allclose(w, [1.0, 1.0])


def test_weights_never_reach_zero() -> None:
    """Old data fades but never vanishes.

    This is the whole reason to prefer decay over a hard sliding window: a
    1999 game still nudges the fit instead of falling off a cliff.
    """
    w = recency_weights(pd.Series([1999]), target_season=2025, half_life=2)
    assert 0 < w[0] < 1e-3


def test_invalid_half_life_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        recency_weights(pd.Series([2020]), target_season=2024, half_life=0)


def test_ess_equals_n_when_unweighted() -> None:
    """Equal weights means no effective data is lost."""
    assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)


def test_ess_falls_as_weights_concentrate() -> None:
    """Shorter half-life buys adaptivity by spending effective sample size.

    That exchange rate IS the bias-variance tradeoff in this model, so it is
    worth asserting the direction rather than assuming it.
    """
    seasons = pd.Series(range(2000, 2025))
    long_hl = effective_sample_size(recency_weights(seasons, 2025, half_life=10))
    short_hl = effective_sample_size(recency_weights(seasons, 2025, half_life=2))
    assert short_hl < long_hl < len(seasons)
