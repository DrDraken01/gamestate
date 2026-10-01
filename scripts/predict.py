"""Fit the margin model on completed games and write predictions for upcoming ones.

    DATABASE_URL=postgresql://localhost/gamestate python scripts/predict.py
    DATABASE_URL=... python scripts/predict.py --season 2026 --week 4

Reads games from the database (the loader must have run first -- see the
read-from-DB decision in db/read.py). The completed/upcoming split is enforced
in SQL, so this script physically cannot train on an unplayed game.

The flow, and where the leakage boundary sits:

    completed games  --build features + fit-->  model        (training)
    upcoming games   --build features + apply-->  PMFs        (prediction)

Features for upcoming games are built from the SAME completed history, never
from the upcoming games' own (non-existent) results. That is the whole reason
the split has to happen before feature construction, not after.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess  # nosec B404 -- only ever runs a fixed `git rev-parse`, no user input
import sys

import pandas as pd
import psycopg

from gamestate.db.predict_writer import normal_producer, write_prediction_batch
from gamestate.db.read import read_completed_games, read_upcoming_games
from gamestate.distribution import fit_margin_model, predict_distributions
from gamestate.features import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predict")

FEATURES = ["rating_diff"]
HALF_LIFE = 3.0
MODEL_VERSION = "margin-normal-v1"


def current_git_sha() -> str:
    """The commit this prediction was produced from -- the audit anchor.

    Read from git so it is always the REAL sha, never a value someone forgot to
    set. Resolve git's absolute path first (shutil.which) rather than trusting
    PATH, and pass a fixed argument list with no shell -- so there is no command
    injection surface. Falls back to 'unknown' only if git is genuinely absent.
    """
    git = shutil.which("git")
    if git is None:
        logger.warning("git not found; recording sha as 'unknown'")
        return "unknown"
    try:
        out = subprocess.run(  # nosec B603 -- fixed args, resolved absolute path, no user input
            [git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning("could not read git sha; recording 'unknown'")
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write margin predictions to the database.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL is not set")
        return 1

    with psycopg.connect(url) as conn:
        completed = read_completed_games(conn)
        logger.info("completed games (training): %d", len(completed))

        upcoming = read_upcoming_games(conn, season=args.season, week=args.week)
        logger.info("upcoming games (to predict): %d", len(upcoming))
        if upcoming.empty:
            logger.warning("no upcoming games match the filter; nothing to predict")
            return 0

        all_games = pd.concat([completed, upcoming], ignore_index=True)
        feats = build_features(all_games)

        train = feats[feats["game_id"].isin(completed["game_id"])].dropna(subset=["rating_diff"])
        predict_rows = feats[feats["game_id"].isin(upcoming["game_id"])].dropna(
            subset=["rating_diff"]
        )
        logger.info("trainable rows: %d, predictable rows: %d", len(train), len(predict_rows))
        if predict_rows.empty:
            logger.warning("no upcoming games have enough history to predict yet")
            return 0

        target_season = int(upcoming["season"].max())
        model, residuals, sigma = fit_margin_model(
            train, target_season, FEATURES, half_life=HALF_LIFE
        )
        normal, _empirical = predict_distributions(model, predict_rows, FEATURES, residuals, sigma)
        logger.info("fitted sigma = %.2f", sigma)

        run_id, n = write_prediction_batch(
            conn,
            games=predict_rows,
            producer=normal_producer(normal),
            mu=normal.mu,
            sigma=sigma,
            git_sha=current_git_sha(),
            model_version=MODEL_VERSION,
            feature_set=FEATURES,
            trained_through=pd.Timestamp.now(tz="UTC"),
            half_life=HALF_LIFE,
            notes=f"season={args.season} week={args.week}",
        )
        logger.info("run %d: wrote %d predictions (atomic batch)", run_id, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
