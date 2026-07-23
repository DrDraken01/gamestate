"""Ingestion layer: pull raw data from nflverse and cache it locally.

DESIGN RULE: this module does exactly two things -- fetch bytes, and store
them unmodified. No cleaning, no feature computation, no filtering.

Why the rule matters: if ingestion silently transforms data, you can never
reproduce a past result, because the transform isn't recorded anywhere. Raw
cache = the ground truth you can always re-derive from. This is the same
principle as "bronze/silver/gold" layering in data engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# nflverse publishes a single tidy game-level table. One row per game, with
# scores, betting lines, weather, rest days, and stadium info already joined.
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "games.parquet"


def fetch_games(cache_path: Path = DEFAULT_CACHE, *, refresh: bool = False) -> pd.DataFrame:
    """Return the raw nflverse games table, using a local cache.

    Args:
        cache_path: where the parquet cache lives.
        refresh: if True, ignore the cache and re-download.

    We cache as parquet rather than CSV because parquet preserves dtypes.
    A CSV round-trip turns your integers into floats the moment one value is
    missing, and that class of bug is genuinely painful to track down later.
    """
    if cache_path.exists() and not refresh:
        logger.info("loading cached games from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("downloading games from %s", GAMES_URL)
    games = pd.read_csv(GAMES_URL, low_memory=False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(cache_path, index=False)
    logger.info("cached %d rows to %s", len(games), cache_path)
    return games


def completed_games(games: pd.DataFrame) -> pd.DataFrame:
    """Filter to games that have actually been played and have a winner.

    Ties are dropped. They are ~0.2% of games and a binary win/loss target
    has no sensible slot for them. Worth revisiting if we ever model margin
    directly instead of win probability -- see docs/decisions/0002.
    """
    played = games[games["result"].notna()].copy()
    decided = played[played["result"] != 0].copy()

    dropped = len(played) - len(decided)
    if dropped:
        logger.info("dropped %d tie games", dropped)

    decided["gameday"] = pd.to_datetime(decided["gameday"])
    return decided.sort_values(["gameday", "game_id"]).reset_index(drop=True)
