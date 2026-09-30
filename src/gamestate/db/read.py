"""Read games back out of the database.

WHY THIS SPLIT LIVES HERE, IN SQL

The single most important line in this file is `home_score IS NOT NULL`. It is
the database-level statement of the invariant you reasoned out:

    training data   = completed games (have a result)
    prediction data = scheduled games (no result yet)

Doing the split in the query -- rather than pulling everything and filtering in
pandas -- means the predictor physically cannot train on an unplayed game. A
NULL score never enters the training frame, so it can never become a NaN that
silently poisons a feature or a 0 that teaches the model a fake 0-0 tie. The
boundary is enforced by the data layer, not by remembering to filter.

WHY IT READS FROM THE DB, NOT FROM NFLVERSE

The loader owns ingestion; the predictor only sees what was persisted. That
separation is what keeps the predictor honest -- it cannot reach past the stored
data to grab a fresh column mid-prediction, which is one of the sneakier leakage
paths. The cost (you must run the loader first) is paid by the orchestrator
later, not by human memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from psycopg import Connection

# Columns the feature layer needs. Kept explicit rather than SELECT * so a
# schema change surfaces here as an obvious edit, not a silent column drift.
_GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "kickoff_at",
    "home_team",
    "away_team",
    "roof",
    "surface",
    "home_score",
    "away_score",
]


def _fetch_frame(
    conn: Connection, sql: str, params: dict[str, int], columns: list[str]
) -> pd.DataFrame:
    """Run a query and build a DataFrame from psycopg directly.

    We avoid pd.read_sql here: it expects a SQLAlchemy connectable and warns
    (loudly, and one day fatally) when handed a raw psycopg connection. Fetching
    rows and naming columns ourselves is a few lines and has no hidden
    dependency -- which, after a session's worth of version-drift bugs, is the
    trade worth making.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def read_completed_games(conn: Connection) -> pd.DataFrame:
    """Games that have been played -- the training set.

    result is derived here as home_score - away_score so downstream feature
    code sees the same `result` column it got from the raw nflverse table.
    Ordered by kickoff so any time-based logic sees them in real order.
    """
    sql = f"""
        SELECT {", ".join(_GAME_COLUMNS)},
               (home_score - away_score) AS result
        FROM games
        WHERE home_score IS NOT NULL
        ORDER BY kickoff_at, game_id
    """  # nosec B608 -- _GAME_COLUMNS is a module constant; no user input in SQL
    return _fetch_frame(conn, sql, {}, [*_GAME_COLUMNS, "result"])


def read_upcoming_games(
    conn: Connection, *, season: int | None = None, week: int | None = None
) -> pd.DataFrame:
    """Scheduled games with no result yet -- what we predict.

    Optionally narrow to a season and/or week. With neither, returns every
    not-yet-played game, which is the honest default: everything the model
    could say something about.
    """
    clauses = ["home_score IS NULL"]
    params: dict[str, int] = {}
    if season is not None:
        clauses.append("season = %(season)s")
        params["season"] = season
    if week is not None:
        clauses.append("week = %(week)s")
        params["week"] = week

    where = " AND ".join(clauses)
    sql = f"""
        SELECT {", ".join(_GAME_COLUMNS)}
        FROM games
        WHERE {where}
        ORDER BY kickoff_at, game_id
    """  # nosec B608 -- columns and clauses are literals; season/week are bound params
    return _fetch_frame(conn, sql, params, list(_GAME_COLUMNS))
