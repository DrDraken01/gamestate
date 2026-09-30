"""Write predictions to the database. Closes the loop: model -> persisted rows.

THE CONTRACT (why this file never changes when the model does)

margin_pmf stores 121 probabilities that sum to 1. Nothing more. Whether those
numbers came from discretising a Normal or from tallying a million simulated
games is invisible here. So the writer takes a PRODUCER -- a callable that turns
game rows into per-game PMFs -- and knows nothing about how it works:

    PmfProducer = Callable[[pd.DataFrame], np.ndarray]   # -> (n_games, 121)

Swap NormalPmfProducer for a MonteCarloPmfProducer later and this file is
untouched. That decoupling is the whole reason the PMF representation was chosen
(ADR-0004, ADR-0005). The model is a producer; the schema is a consumer; the PMF
is the contract between them.

WHAT A RUN RECORDS

Every prediction points at a model_run, which pins the exact code (git_sha), the
leakage boundary (trained_through), and the knobs (feature_set, half_life). That
is what makes a stored probability auditable six months later: you can check out
the commit and reproduce it. Predictions are append-only -- a new run never
overwrites an old one, so the full forecast history survives (ADR-0005).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from gamestate.db.pmf import PMF_LENGTH, normal_pmf

if TYPE_CHECKING:
    from psycopg import Connection

    from gamestate.distribution import NormalMargin


class PmfProducer(Protocol):
    """Anything that turns game rows into one 121-element PMF per game.

    The single seam the whole design turns on. A Normal today, a simulation
    later, a blend after that -- all satisfy this and the writer cannot tell
    them apart. Returns an (n_games, PMF_LENGTH) array, each row summing to 1.
    """

    def __call__(self, games: pd.DataFrame) -> np.ndarray: ...


def normal_producer(dist: NormalMargin) -> PmfProducer:
    """Adapt a fitted NormalMargin into a PmfProducer.

    Discretises each game's Normal(mu, sigma) onto the integer lattice. This is
    the ONLY place the Normal assumption enters the persistence path; a
    Monte Carlo producer would fill the same array by binning simulated margins
    and the writer below would not change a line.
    """

    def produce(games: pd.DataFrame) -> np.ndarray:
        # dist.mu is aligned to the rows the model predicted, in order.
        return np.vstack([normal_pmf(float(mu), dist.sigma) for mu in dist.mu])

    return produce


def _validate_pmfs(pmfs: np.ndarray, n_games: int) -> None:
    """A stored distribution must actually be one. Fail loudly if not.

    The DB CHECK enforces length; it cannot enforce "sums to 1" or "no NaN".
    Catching a malformed PMF here beats writing a plausible-looking lie that
    every downstream threshold query then trusts.
    """
    if pmfs.shape != (n_games, PMF_LENGTH):
        raise ValueError(f"expected ({n_games}, {PMF_LENGTH}) PMFs, got {pmfs.shape}")
    if not np.isfinite(pmfs).all():
        raise ValueError("PMF contains NaN or inf")
    sums = pmfs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        worst = float(np.abs(sums - 1.0).max())
        raise ValueError(f"PMF rows must sum to 1 (worst deviation {worst:.2e})")


INSERT_RUN = """
INSERT INTO model_runs (
    git_sha, model_version, feature_set, trained_through, half_life, notes
)
VALUES (
    %(git_sha)s, %(model_version)s, %(feature_set)s,
    %(trained_through)s, %(half_life)s, %(notes)s
)
RETURNING run_id;
"""

INSERT_PREDICTION = """
INSERT INTO predictions (run_id, game_id, mu, sigma, margin_pmf)
VALUES (%(run_id)s, %(game_id)s, %(mu)s, %(sigma)s, %(margin_pmf)s);
"""


def write_run(
    conn: Connection,
    *,
    git_sha: str,
    model_version: str,
    feature_set: list[str],
    trained_through: pd.Timestamp,
    half_life: float | None,
    notes: str | None = None,
) -> int:
    """Insert a model_run row and return its generated run_id."""
    with conn.cursor() as cur:
        cur.execute(
            INSERT_RUN,
            {
                "git_sha": git_sha,
                "model_version": model_version,
                "feature_set": feature_set,
                "trained_through": trained_through,
                "half_life": half_life,
                "notes": notes,
            },
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("model_runs insert returned no run_id")
        return int(row[0])


def write_predictions(
    conn: Connection,
    run_id: int,
    games: pd.DataFrame,
    producer: PmfProducer,
    mu: np.ndarray,
    sigma: float,
) -> int:
    """Write one predictions row per game. Returns the count written.

    games must carry a game_id column, aligned row-for-row with mu and the
    producer's output. mu/sigma are stored as the convenience summary; the PMF
    from the producer is the source of truth (ADR-0005).
    """
    pmfs = producer(games)
    _validate_pmfs(pmfs, len(games))

    rows = [
        {
            "run_id": run_id,
            "game_id": str(gid),
            "mu": float(m),
            "sigma": float(sigma),
            "margin_pmf": pmf.tolist(),
        }
        for gid, m, pmf in zip(games["game_id"], mu, pmfs, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(INSERT_PREDICTION, rows)
    conn.commit()
    return len(rows)
