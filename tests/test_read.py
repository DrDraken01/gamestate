"""Tests for reading games back out of the database.

The DB round-trip needs a live Postgres (covered by the build-time integration
run). Here we test the query-building logic that decides the completed/upcoming
split -- the part where the leakage boundary is enforced.
"""

from __future__ import annotations

import inspect

from gamestate.db import read


def test_completed_query_filters_on_result_present() -> None:
    """The training query MUST require a score. This is the leakage boundary.

    Asserting on the SQL text is crude but it pins the one clause that must
    never silently disappear: an unplayed game must not enter training.
    """
    src = inspect.getsource(read.read_completed_games)
    assert "home_score IS NOT NULL" in src


def test_upcoming_query_filters_on_result_absent() -> None:
    """The prediction set is exactly the complement: no score yet."""
    src = inspect.getsource(read.read_upcoming_games)
    assert "home_score IS NULL" in src


def test_upcoming_is_parameterized_not_interpolated() -> None:
    """Season/week filters must be bound params, never f-string values.

    Interpolating user-ish values into SQL is the injection footgun. The
    values go through psycopg's parameter binding; only the column names are
    literal. This asserts we pass a params dict rather than formatting values in.
    """
    src = inspect.getsource(read.read_upcoming_games)
    assert "%(season)s" in src
    assert "%(week)s" in src
