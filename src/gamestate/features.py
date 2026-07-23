"""Feature engineering.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE:

    Every feature attached to a game must be computable using ONLY
    information available before that game kicked off.

This is the single most common way sports models silently become worthless.
The bug never announces itself -- your model just posts a suspiciously good
score, you feel great, and then it falls apart on live data.

The classic form of the bug is one missing `.shift(1)`:

    # WRONG: the window includes the current game's own result
    df.groupby("team")["point_diff"].rolling(5).mean()

    # RIGHT: shift first, so the window covers only PRIOR games
    df.groupby("team")["point_diff"].shift(1).rolling(5).mean()

tests/test_features.py asserts this property directly. Do not delete that test.
"""

from __future__ import annotations

import pandas as pd

# Columns we carry through untouched so downstream code can slice results
# (by season, by roof type, etc.) without re-joining the raw table.
PASSTHROUGH = ["game_id", "season", "week", "gameday", "home_team", "away_team", "spread_line"]


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-game into two-rows-per-game (one per team).

    Wide game rows are convenient for modeling but awful for computing team
    history, because a team's games are split across two different columns.
    Long team-game format makes "this team's last 5 games" a simple groupby.

    Reshape for computation, then join the result back to wide. This
    long/wide dance is a workhorse pattern -- you will use it constantly.
    """
    home = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "gameday": games["gameday"],
            "team": games["home_team"],
            "points_for": games["home_score"],
            "points_against": games["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "gameday": games["gameday"],
            "team": games["away_team"],
            "points_for": games["away_score"],
            "points_against": games["home_score"],
        }
    )

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    return team_games.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def rolling_point_diff(
    team_games: pd.DataFrame, window: int = 8, min_periods: int = 4
) -> pd.DataFrame:
    """Average point differential over a team's PRIOR `window` games.

    Args:
        window: how many past games to average over. 8 is a starting guess,
            not a tuned value -- tuning it is a later exercise, and one that
            needs a validation split so we don't tune on the test set.
        min_periods: below this many prior games the average is too noisy to
            trust, so we emit NaN and let the caller decide what to do.

    Note this window deliberately crosses season boundaries. Week 1 of 2024
    uses late-2023 games. That is a modeling choice with a real tradeoff:
    rosters turn over in the offseason, so old games are less informative --
    but the alternative is having no feature at all for the first month of
    every season. Recency weighting is the eventual fix.
    """
    grouped = team_games.groupby("team", sort=False)["point_diff"]

    # shift(1) BEFORE rolling. This is the whole ballgame. It moves the
    # window back by one game so the current game's own result is excluded.
    prior = grouped.shift(1)
    team_games = team_games.copy()
    team_games["prior_point_diff"] = prior.groupby(team_games["team"], sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).mean()
    )

    # How many prior games actually went into the average. Useful for
    # debugging and for slicing evaluation by sample maturity.
    team_games["prior_games"] = prior.groupby(team_games["team"], sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=1).count()
    )
    return team_games


def build_features(games: pd.DataFrame, window: int = 8, min_periods: int = 4) -> pd.DataFrame:
    """Produce the modeling table: one row per game, features + target.

    Returns a frame with:
        rating_diff -- home team's prior form minus away team's prior form.
        home_win    -- the target. 1 if the home team won.

    Games where either team lacks enough history are dropped, which costs us
    the earliest weeks of 1999. That is a deliberate trade: a NaN-filled
    feature would teach the model that "no history" means "average team,"
    which is not what missingness means here.
    """
    team_games = rolling_point_diff(to_team_games(games), window, min_periods)
    ratings = team_games[["game_id", "team", "prior_point_diff"]]

    out = games[PASSTHROUGH].copy()
    out["home_win"] = (games["result"] > 0).astype(int)

    # Join each team's prior form back onto the wide game row. Two merges --
    # one keyed on home_team, one on away_team -- rather than MultiIndex.map.
    # Merges say plainly what the join key is, which matters when a bad join
    # is the difference between a working model and a silently wrong one.
    for side in ("home", "away"):
        out = out.merge(
            ratings.rename(columns={"team": f"{side}_team", "prior_point_diff": f"{side}_rating"}),
            on=["game_id", f"{side}_team"],
            how="left",
            validate="one_to_one",
        )

    # A single feature: the gap in recent form. Positive favors the home team.
    out["rating_diff"] = out["home_rating"] - out["away_rating"]

    return out.dropna(subset=["rating_diff"]).reset_index(drop=True)
