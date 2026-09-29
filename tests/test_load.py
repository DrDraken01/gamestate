"""Tests for the nflverse -> Postgres loader.

These test the pure transform functions (build_kickoff, games_for_db) without a
live database. The database behaviour -- upsert idempotency and the NULL-clobber
guard -- is verified in the SQL itself and exercised manually against Postgres;
those are integration concerns, not unit-testable without a running server.
"""

from __future__ import annotations

import pandas as pd

from gamestate.db.load import build_kickoff, games_for_db


def test_kickoff_is_timezone_aware_utc() -> None:
    """kickoff_at must be a real UTC instant, not a naive timestamp.

    This is the leakage boundary. A naive timestamp would compare wrong across
    timezones and eventually let a late game leak into an earlier prediction.
    """
    day = pd.Series(["2026-09-07"])
    time = pd.Series(["13:00"])
    k = build_kickoff(day, time)
    assert k.dt.tz is not None
    # 1pm US/Eastern in September (EDT, UTC-4) -> 17:00 UTC
    assert k.iloc[0].hour == 17
    assert str(k.iloc[0].tzinfo) in ("UTC", "Etc/UTC")


def test_kickoff_distinguishes_slots() -> None:
    """A 1pm and an 8:20pm game on the same day are different instants."""
    day = pd.Series(["2026-09-07", "2026-09-07"])
    time = pd.Series(["13:00", "20:20"])
    k = build_kickoff(day, time)
    assert k.iloc[1] > k.iloc[0]


def test_missing_gametime_still_produces_an_instant() -> None:
    """Far-future games with no set time must still be storable."""
    day = pd.Series(["2026-12-28"])
    time = pd.Series([pd.NA])
    k = build_kickoff(day, time)
    assert k.notna().all()


def test_games_for_db_keeps_scheduled_and_played() -> None:
    """The games projection must NOT filter by completion.

    Both played and scheduled games belong in the table; completion is a
    query-time distinction. Dropping scheduled games here would make it
    impossible to predict upcoming games -- the whole point of the product.
    """
    raw = pd.DataFrame(
        {
            "game_id": ["2026_01_A_B", "2026_18_C_D"],
            "season": [2026, 2026],
            "week": [1, 18],
            "gameday": ["2026-09-07", "2027-01-04"],
            "gametime": ["13:00", pd.NA],
            "home_team": ["B", "D"],
            "away_team": ["A", "C"],
            "roof": ["outdoors", "dome"],
            "surface": ["grass", "turf"],
            "home_score": [24, pd.NA],  # one played, one scheduled
            "away_score": [17, pd.NA],
        }
    )
    out = games_for_db(raw)
    assert len(out) == 2  # both kept
    assert out["home_score"].isna().sum() == 1  # the scheduled one stays NULL


def test_scores_are_nullable_integers() -> None:
    """Scores must survive as nullable ints, not floats.

    A CSV round-trip turns int columns to float the moment one value is
    missing (7 -> 7.0), and a 7.0 in a SMALLINT column is a needless coercion.
    Int64 keeps them clean.
    """
    raw = pd.DataFrame(
        {
            "game_id": ["2026_01_A_B"],
            "season": [2026],
            "week": [1],
            "gameday": ["2026-09-07"],
            "gametime": ["13:00"],
            "home_team": ["B"],
            "away_team": ["A"],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "home_score": [pd.NA],
            "away_score": [pd.NA],
        }
    )
    out = games_for_db(raw)
    assert str(out["home_score"].dtype) == "Int64"
