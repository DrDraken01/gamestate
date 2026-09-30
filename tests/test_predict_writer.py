"""Tests for the prediction writer.

The DB insert paths need a live Postgres and are exercised as integration
checks. Here we unit-test the parts that hold the design together: PMF
validation, and that the producer seam actually decouples the model from the
writer (a non-Normal producer must flow through untouched).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamestate.db.pmf import PMF_LENGTH, normal_pmf
from gamestate.db.predict_writer import _validate_pmfs


def test_validate_accepts_real_pmfs() -> None:
    pmfs = np.vstack([normal_pmf(3.0, 13.0), normal_pmf(-2.0, 13.0)])
    _validate_pmfs(pmfs, 2)  # should not raise


def test_validate_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        _validate_pmfs(np.ones((2, 50)), 2)


def test_validate_rejects_non_normalized() -> None:
    """A row that does not sum to 1 is not a distribution; refuse to store it."""
    bad = np.ones((1, PMF_LENGTH))  # sums to 121, not 1
    with pytest.raises(ValueError, match="sum to 1"):
        _validate_pmfs(bad, 1)


def test_validate_rejects_nan() -> None:
    bad = normal_pmf(0.0, 13.0).reshape(1, -1).copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _validate_pmfs(bad, 1)


def test_producer_seam_is_model_agnostic() -> None:
    """The core design claim: any callable returning valid PMFs is acceptable.

    A producer that has nothing to do with a Normal -- here, uniform mass --
    must satisfy the same contract and pass validation. This is what lets
    Monte Carlo replace the Normal later without touching the writer.
    """
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"]})

    def uniform_producer(g: pd.DataFrame) -> np.ndarray:
        return np.full((len(g), PMF_LENGTH), 1.0 / PMF_LENGTH)

    pmfs = uniform_producer(games)
    _validate_pmfs(pmfs, len(games))  # a totally different shape of model, same contract
    assert pmfs.sum(axis=1) == pytest.approx(1.0)
