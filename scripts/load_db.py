"""Load nflverse data into the database. Run weekly during the season.

    DATABASE_URL=postgresql://localhost/gamestate python scripts/load_db.py

Idempotent: safe to run repeatedly. Each run picks up newly-played results and
leaves everything else untouched. During the season, a Monday cron on this
script keeps the games table current for the week's predictions.

The connection string comes from the DATABASE_URL environment variable, never
a hardcoded literal -- a connection string holds credentials, and a credential
in source is a credential leaked (the detect-secrets hook would block it anyway).
"""

from __future__ import annotations

import logging
import os
import sys

import psycopg

from gamestate.db.load import load_games
from gamestate.ingest import fetch_games

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("load_db")


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL is not set. Example:")
        logger.error("  DATABASE_URL=postgresql://localhost/gamestate python scripts/load_db.py")
        return 1

    # refresh=True: during the season we WANT the fresh pull, not the cache.
    # The cache is an offseason convenience; in-season it would hide new results.
    games = fetch_games(refresh=True)
    logger.info("fetched %d rows from nflverse", len(games))

    with psycopg.connect(url) as conn:
        stats = load_games(conn, games)

    logger.info(
        "done: %d total games, %d inserted this run, %d newly played",
        stats["total"],
        stats["inserted"],
        stats["newly_played"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
