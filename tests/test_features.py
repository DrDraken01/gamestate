"""Tests for feature construction.

The leakage tests here are the most valuable code in the repository. They are
cheap to write and they catch the one bug that would invalidate everything
downstream while looking like success.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gamestate.features import build_features, rolling_point_diff, to_team_games


@pytest.fixture
def toy_games() -> pd.DataFrame:
    """Six games between two teams, played a week apart.

    AAA wins every game by exactly 10, then blows BBB out by 50 in the final
    game. That last game is the tripwire: if the 50-point margin shows up in
    the features FOR that same game, we have leakage.
    """
    rows = [
        ("g1", 2020, 1, "2020-09-01", "AAA", "BBB", 20, 10),
        ("g2", 2020, 2, "2020-09-08", "AAA", "BBB", 20, 10),
        ("g3", 2020, 3, "2020-09-15", "AAA", "BBB", 20, 10),
        ("g4", 2020, 4, "2020-09-22", "AAA", "BBB", 20, 10),
        ("g5", 2020, 5, "2020-09-29", "AAA", "BBB", 20, 10),
        ("g6", 2020, 6, "2020-10-06", "AAA", "BBB", 60, 10),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
    )
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    frame["result"] = frame["home_score"] - frame["away_score"]
    frame["spread_line"] = 7.0
    return frame


def test_feature_excludes_current_game(toy_games: pd.DataFrame) -> None:
    """The blowout in g6 must not appear in g6's own feature.

    Through g5, AAA's margin is +10 every time. So its prior-form rating
    entering g6 must be exactly +10. If the 50-point margin leaked in, the
    rating would be inflated well above that.
    """
    feats = build_features(toy_games, window=8, min_periods=1)
    g6 = feats[feats["game_id"] == "g6"].iloc[0]

    assert g6["home_rating"] == pytest.approx(10.0), (
        "LEAKAGE: g6's own 50-point margin contaminated its feature. "
        "Check for a missing .shift(1) in rolling_point_diff."
    )
    assert g6["away_rating"] == pytest.approx(-10.0)


def test_first_game_has_no_rating(toy_games: pd.DataFrame) -> None:
    """With zero prior games there is no history, so the feature must be NaN.

    Not 0.0. Zero would mean "an exactly average team," which is a claim we
    have no evidence for. Encoding unknown as a plausible-looking number is
    how you teach a model something false.
    """
    team_games = rolling_point_diff(to_team_games(toy_games), window=8, min_periods=1)

    # NOTE: do NOT use groupby("team").first() here. Pandas' .first() returns
    # the first NON-NULL value in each group, not the first row -- so it would
    # skip right past the NaN we are trying to assert on and the test would
    # fail against correct code. drop_duplicates actually takes the first row.
    first_rows = team_games.sort_values("gameday").drop_duplicates("team", keep="first")
    assert first_rows["prior_point_diff"].isna().all()


def test_future_data_cannot_change_past_features(toy_games: pd.DataFrame) -> None:
    """Appending later games must not alter features for earlier games.

    This is the general statement of the leakage property, and it catches
    whole categories of bug that a single hand-checked value would miss --
    full-history normalization, target encoding, imputation with a global
    mean. If tomorrow can change yesterday, information is flowing backwards.
    """
    baseline = build_features(toy_games, window=8, min_periods=1)

    extra = toy_games.iloc[[-1]].copy()
    extra["game_id"] = "g7"
    extra["gameday"] = pd.Timestamp("2020-10-13")
    extra["home_score"] = 99
    extended = build_features(
        pd.concat([toy_games, extra], ignore_index=True), window=8, min_periods=1
    )

    merged = baseline.merge(extended, on="game_id", suffixes=("_before", "_after"))
    pd.testing.assert_series_equal(
        merged["rating_diff_before"],
        merged["rating_diff_after"],
        check_names=False,
    )


def test_min_periods_drops_immature_rows(toy_games: pd.DataFrame) -> None:
    """Requiring 4 prior games should leave only g5 and g6."""
    feats = build_features(toy_games, window=8, min_periods=4)
    assert set(feats["game_id"]) == {"g5", "g6"}
