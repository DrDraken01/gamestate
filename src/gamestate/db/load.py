"""Load the nflverse table into Postgres. Idempotent, re-runnable, in-season.

WHY THIS IS AN UPSERT, NOT AN INSERT

During the season this runs weekly. Each run the nflverse table has:
  - new results for games that were NULL last week (week N just played),
  - the same rows as before for everything else,
  - future games whose scores are still NULL.

An INSERT would collide on the primary key the second time. A DELETE-then-INSERT
would throw away prediction history that references these games. So we UPSERT:
insert new games, update existing ones -- but with one guard (see below).

THE ONE GUARD THAT MATTERS

A re-ingest must never overwrite a real score with NULL. nflverse should never
regress a played game back to unplayed, but defending against it costs one SQL
clause and prevents a silent catastrophe: a future-schedule row clobbering a
completed result would poison every downstream feature. So the score columns are
only updated when the incoming value is NOT NULL (COALESCE keeps the existing
one otherwise).

TIMEZONES

nflverse gameday+gametime are US/Eastern. We localise to Eastern, then store as
TIMESTAMPTZ -- Postgres keeps the instant, not the wall-clock string. This is
what makes kickoff_at a trustworthy leakage boundary: "before kickoff" is a real
moment in time, comparable across the whole table regardless of where a game was
played or where the query runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from psycopg import Connection

logger = logging.getLogger(__name__)

# nflverse gametimes are US/Eastern.
EASTERN = "America/New_York"


def build_kickoff(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combine date + local time into a timezone-aware UTC instant.

    gametime can be missing for far-future games whose slot is not yet set; we
    default those to a placeholder hour so the row is still storable, and the
    real time lands on the next re-ingest. A game is never predicted off its
    kickoff hour anyway -- kickoff_at gates day-level leakage, not minutes.
    """
    times = gametime.fillna("13:00")  # default to a 1pm ET slot if unknown
    combined = pd.to_datetime(gameday.astype(str) + " " + times, format="mixed")
    # ambiguous="infer": during the fall-back DST hour, infer which side of the
    # transition each time is on from monotonic order. "infer", not True -- same
    # behaviour, but it is the value the typed API actually accepts.
    localized = combined.dt.tz_localize(EASTERN, ambiguous="infer", nonexistent="shift_forward")
    return localized.dt.tz_convert("UTC")


def games_for_db(games: pd.DataFrame) -> pd.DataFrame:
    """Project the raw nflverse table down to the games-table columns.

    No filtering by completion here -- the games table deliberately holds BOTH
    played and scheduled games. Completion is a query-time distinction
    (result IS NOT NULL), not a storage one. See ADR-0005 and the in-season
    training invariant: training = completed, prediction = scheduled.
    """
    out = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"].astype(int),
            "week": games["week"].astype(int),
            "kickoff_at": build_kickoff(games["gameday"], games["gametime"]),
            "home_team": games["home_team"],
            "away_team": games["away_team"],
            "roof": games["roof"],
            "surface": games["surface"],
            "home_score": games["home_score"].astype("Int64"),  # nullable int
            "away_score": games["away_score"].astype("Int64"),
        }
    )
    # Drop rows without a scheduled date at all -- cannot form a kickoff instant.
    return out[out["kickoff_at"].notna()].reset_index(drop=True)


UPSERT_GAMES = """
INSERT INTO games (
    game_id, season, week, kickoff_at,
    home_team, away_team, roof, surface, home_score, away_score
)
VALUES (
    %(game_id)s, %(season)s, %(week)s, %(kickoff_at)s,
    %(home_team)s, %(away_team)s, %(roof)s, %(surface)s,
    %(home_score)s, %(away_score)s
)
ON CONFLICT (game_id) DO UPDATE SET
    season     = EXCLUDED.season,
    week       = EXCLUDED.week,
    kickoff_at = EXCLUDED.kickoff_at,
    roof       = EXCLUDED.roof,
    surface    = EXCLUDED.surface,
    -- THE GUARD: only take a new score if it is actually present. A NULL in
    -- the incoming row (a future game) can never overwrite a stored result.
    home_score = COALESCE(EXCLUDED.home_score, games.home_score),
    away_score = COALESCE(EXCLUDED.away_score, games.away_score);
"""


def load_games(conn: Connection, games: pd.DataFrame) -> dict[str, int]:
    """Upsert the games table. Returns simple counts for the caller to log.

    Idempotent: running it twice on the same data changes nothing the second
    time. That property is what makes it safe to run on a schedule.
    """
    frame = games_for_db(games)

    before = _scalar(conn, "SELECT count(*) FROM games")
    played_before = _scalar(conn, "SELECT count(*) FROM games WHERE home_score IS NOT NULL")

    # to_dict("records") types its keys as Hashable, but psycopg wants
    # Mapping[str, Any]. The keys ARE the column-name strings; rebuild each dict
    # with str keys so the type is honest rather than casting a blind eye.
    rows: list[dict[str, object]] = [
        {str(k): v for k, v in record.items()} for record in frame.to_dict("records")
    ]
    with conn.cursor() as cur:
        cur.executemany(UPSERT_GAMES, rows)
    conn.commit()

    after = _scalar(conn, "SELECT count(*) FROM games")
    played_after = _scalar(conn, "SELECT count(*) FROM games WHERE home_score IS NOT NULL")

    stats = {
        "seen": len(frame),
        "inserted": after - before,
        "newly_played": played_after - played_before,
        "total": after,
    }
    logger.info(
        "games upsert: %d seen, %d new, %d newly-played, %d total",
        stats["seen"],
        stats["inserted"],
        stats["newly_played"],
        stats["total"],
    )
    return stats


def _scalar(conn: Connection, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return int(row[0]) if row else 0
