"""Turn a distribution into the stored PMF, and query thresholds back out.

The database stores each margin distribution as a 121-element array:
`P(margin == k)` for k in [-60, +60], index 0 being -60. This module is the
single source of truth for that encoding, so the app and the SQL never drift
on what index means what.

Every threshold in the product is a suffix sum over this array. Keeping that
arithmetic in one tested place is what lets the API layer stay dumb.
"""

from __future__ import annotations

import numpy as np

# The integer margin lattice the PMF covers. A 60-point NFL margin is a
# once-a-decade event, so [-60, 60] loses no realistic mass.
MARGIN_MIN = -60
MARGIN_MAX = 60
PMF_LENGTH = MARGIN_MAX - MARGIN_MIN + 1  # 121


def margin_to_index(margin: int) -> int:
    """Map a signed margin to its PMF array index. margin 0 -> index 60."""
    if not MARGIN_MIN <= margin <= MARGIN_MAX:
        raise ValueError(f"margin {margin} outside [{MARGIN_MIN}, {MARGIN_MAX}]")
    return margin - MARGIN_MIN


def normal_pmf(mu: float, sigma: float) -> np.ndarray:
    """Discretise Normal(mu, sigma) onto the integer lattice, normalised to 1.

    Each integer k gets the Normal density at k, then the whole vector is
    renormalised so it sums to exactly 1 over the clipped support. This is the
    "Normal today" path; a simulation path would fill the same array by
    counting simulated margins into bins instead. The storage does not care
    which produced it -- that is the point of ADR-0004's representation.
    """
    ks = np.arange(MARGIN_MIN, MARGIN_MAX + 1, dtype=float)
    density = np.exp(-0.5 * ((ks - mu) / sigma) ** 2)
    return density / density.sum()


def prob_over(pmf: np.ndarray, line: float) -> float:
    """P(margin > line). The workhorse: win prob, spreads, totals all use it.

    A half-point line (e.g. -3.5) can't push, so we sum every integer strictly
    greater than it. A whole-number line (-3) is handled by the caller's choice
    of >, since push handling is a product decision, not a math one.
    """
    ks = np.arange(MARGIN_MIN, MARGIN_MAX + 1)
    return float(pmf[ks > line].sum())


def prob_exactly(pmf: np.ndarray, margin: int) -> float:
    """P(margin == k). Meaningful only because we store the full PMF.

    This is the push probability, and the reason the PMF representation exists:
    a Normal collapses it to near-zero, but key numbers are real (ADR-0004).
    Trustworthy to the extent the PMF was built by a process that respects key
    numbers -- i.e. simulation, not the discretised Normal.
    """
    return float(pmf[margin_to_index(margin)])
