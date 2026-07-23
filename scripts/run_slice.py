"""Run the whole vertical slice: ingest -> features -> model -> evaluate.

PYTHONPATH=src python3 scripts/run_slice.py
"""

from __future__ import annotations

import logging

import pandas as pd

from gamestate.evaluate import calibration_table, score, skill_score, walk_forward
from gamestate.features import build_features
from gamestate.ingest import completed_games, fetch_games

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.width", 100)


def main() -> None:
    games = completed_games(fetch_games())
    print(f"\ncompleted games: {len(games):,}  seasons {games.season.min()}-{games.season.max()}")

    frame = build_features(games)
    print(f"modeling rows:   {len(frame):,}  (rows without enough history dropped)")

    preds = walk_forward(frame, start_season=2010)
    print(f"backtest rows:   {len(preds):,}  seasons {preds.season.min()}-{preds.season.max()}")

    y = preds["home_win"].to_numpy()

    print("\n" + "=" * 62)
    print("SCORES  (log loss / brier: lower is better)")
    print("=" * 62)
    rows = []
    forecasts = [("base rate", "p_baserate"), ("our model", "p_model"), ("market", "p_market")]
    for name, col in forecasts:
        p = preds[col].to_numpy()
        rows.append(
            {
                "forecast": name,
                **score(y, p),
                "skill_vs_baserate": skill_score(y, p, preds["p_baserate"].to_numpy()),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n" + "=" * 62)
    print("CALIBRATION -- our model")
    print("=" * 62)
    print(calibration_table(y, preds["p_model"].to_numpy()).round(3).to_string(index=False))

    print("\n" + "=" * 62)
    print("CALIBRATION -- market")
    print("=" * 62)
    print(calibration_table(y, preds["p_market"].to_numpy()).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
