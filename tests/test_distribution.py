"""Tests for margin distributions."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from gamestate.distribution import (
    EmpiricalMargin,
    NormalMargin,
    fit_margin_model,
    predict_distributions,
)


@pytest.fixture
def toy_train() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    rating = rng.normal(0, 8, n)
    return pd.DataFrame(
        {
            "season": rng.integers(2018, 2024, n),
            "rating_diff": rating,
            # True relationship: margin = 2.5 + 0.9 * rating + noise
            "margin": 2.5 + 0.9 * rating + rng.normal(0, 13, n),
        }
    )


def test_win_probability_is_prob_over_zero(toy_train: pd.DataFrame) -> None:
    """P(home win) must be exactly P(margin > 0) -- not a separate model.

    This is the property that makes the distribution worth having: one fit
    answers every threshold, and win probability is just the zero threshold.
    """
    dist = NormalMargin(mu=np.array([0.0, 7.0, -7.0]), sigma=13.8)
    p = dist.prob_over(0.0)
    assert p[0] == pytest.approx(0.5)  # dead even
    assert p[1] > 0.5  # home favoured
    assert p[2] < 0.5
    assert p[1] + p[2] == pytest.approx(1.0)  # symmetry


def test_prob_over_is_monotonic() -> None:
    """Raising the threshold can never raise the probability.

    A survival function that is not monotone is a broken distribution, and
    the failure would show up in the product as a threshold table where
    'over 250 yards' is likelier than 'over 200'.
    """
    dist = NormalMargin(mu=np.array([3.0]), sigma=13.8)
    probs = [float(dist.prob_over(line)[0]) for line in [-14, -7, -3, 0, 3, 7, 14]]
    assert all(a >= b for a, b in pairwise(probs))


def test_empirical_matches_its_own_residuals() -> None:
    """With residuals {-10, 0, +10} and mu=0, P(margin > 0) must be exactly 1/3."""
    dist = EmpiricalMargin(mu=np.array([0.0]), residuals=np.array([-10.0, 0.0, 10.0]))
    assert dist.prob_over(0.0)[0] == pytest.approx(1 / 3)
    assert dist.prob_over(-10.0)[0] == pytest.approx(2 / 3)


def test_fit_recovers_known_relationship(toy_train: pd.DataFrame) -> None:
    """Sanity check on the fit: coefficient near 0.9, sigma near 13."""
    model, residuals, sigma = fit_margin_model(
        toy_train, target_season=2024, features=["rating_diff"], half_life=None
    )
    assert model.coef_[0] == pytest.approx(0.9, abs=0.15)
    assert sigma == pytest.approx(13.0, abs=2.0)
    assert len(residuals) == len(toy_train)


def test_shift_based_models_cannot_represent_key_numbers() -> None:
    """Regression test for a KNOWN LIMITATION, not a bug to be fixed here.

    Real NFL margins spike at +/-3 and +/-7 regardless of who is favoured,
    because points arrive in 3s and 7s. A shift-based model translates its
    whole shape with mu, so its spikes move with the favourite -- which is
    not what the data does.

    This test asserts the defect exists so that nobody "fixes" push
    probabilities by tuning sigma. The real fix is score simulation.
    """
    # Residuals with a deliberate spike at exactly +3.
    residuals = np.concatenate([np.full(50, 3.0), np.random.default_rng(1).normal(0, 14, 450)])

    at_zero = EmpiricalMargin(mu=np.array([0.0]), residuals=residuals).prob_exactly(3)
    at_ten = EmpiricalMargin(mu=np.array([10.0]), residuals=residuals).prob_exactly(3)

    # The spike travelled with mu instead of staying anchored at 3. Compare
    # relatively -- the smooth residuals put ~3% baseline mass on any single
    # integer, so an absolute threshold would be measuring the wrong thing.
    assert at_zero[0] > 4 * at_ten[0], (
        "if this fails, the shape stopped translating with mu -- "
        "re-measure key numbers and update ADR-0004"
    )


def test_predict_returns_both_representations(toy_train: pd.DataFrame) -> None:
    model, residuals, sigma = fit_margin_model(
        toy_train, target_season=2024, features=["rating_diff"], half_life=3.0
    )
    test = toy_train.head(10)
    normal, empirical = predict_distributions(model, test, ["rating_diff"], residuals, sigma)

    assert normal.mu.shape == (10,)
    np.testing.assert_allclose(normal.mu, empirical.mu)
    # Both should agree closely on win probability even though their shapes differ.
    np.testing.assert_allclose(normal.prob_over(0.0), empirical.prob_over(0.0), atol=0.06)
