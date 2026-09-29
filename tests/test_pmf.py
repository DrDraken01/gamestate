"""Tests for the PMF encoding shared by the app and the database."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from gamestate.db.pmf import (
    PMF_LENGTH,
    margin_to_index,
    normal_pmf,
    prob_exactly,
    prob_over,
)


def test_index_mapping() -> None:
    """The zero-margin index must be the centre; ends must be the extremes."""
    assert margin_to_index(-60) == 0
    assert margin_to_index(0) == 60
    assert margin_to_index(60) == 120


def test_index_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        margin_to_index(61)


def test_normal_pmf_is_a_distribution() -> None:
    """A stored PMF that doesn't sum to 1 is a corrupt distribution."""
    pmf = normal_pmf(mu=2.5, sigma=13.0)
    assert len(pmf) == PMF_LENGTH
    assert pmf.sum() == pytest.approx(1.0)
    assert np.all(pmf >= 0)


def test_prob_over_zero_matches_scipy() -> None:
    """P(margin > 0) off the lattice should track the continuous Normal.

    Not exact -- discretisation and clipping move it slightly -- but a large
    gap would mean the lattice is too coarse or the support too narrow.
    """
    from scipy import stats

    mu, sigma = 3.0, 13.0
    lattice = prob_over(normal_pmf(mu, sigma), 0.0)
    continuous = float(stats.norm.sf(0.0, mu, sigma))
    assert lattice == pytest.approx(continuous, abs=0.02)


def test_prob_over_is_monotonic() -> None:
    """Raising the line cannot raise the probability."""
    pmf = normal_pmf(mu=1.0, sigma=13.0)
    probs = [prob_over(pmf, line) for line in [-10, -3, 0, 3, 10]]
    assert all(a >= b for a, b in pairwise(probs))


def test_suffix_sum_matches_sql_indexing() -> None:
    """P(margin > 3.5) must equal summing the array from index 64 onward.

    This pins the Python encoding to the exact array slice the SQL uses. In
    1-based psql that is margin_pmf[65:121]; in 0-based numpy it is [64:]. If
    someone shifts the lattice, this test and the SQL must move together.
    """
    pmf = normal_pmf(mu=2.0, sigma=13.0)
    # margins strictly greater than 3.5 are 4,5,...  -> index 64 onward
    expected = float(pmf[64:].sum())
    assert prob_over(pmf, 3.5) == pytest.approx(expected)


def test_prob_exactly_reads_one_cell() -> None:
    pmf = normal_pmf(mu=0.0, sigma=13.0)
    assert prob_exactly(pmf, 0) == pytest.approx(pmf[60])
    assert prob_exactly(pmf, 3) == pytest.approx(pmf[63])
