# GameState

A probabilistic NFL outcome model. Estimates probabilities across thresholds
rather than emitting point predictions, and explains where each probability
comes from.

**Repo:** `gamestate` · **Package:** `gamestate` · **Site:** GameState

**Status:** week one — vertical slice complete. One feature, one model, honest
evaluation against two baselines.

## Quick start

```bash
git clone git@github.com:DrDraken01/gamestate.git
cd gamestate

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

pytest                              # must pass before anything else
python scripts/run_slice.py         # ingest -> features -> backtest
```

## What exists now

A complete path through every layer, deliberately thin at each one:

| Layer | Module | What it does |
|---|---|---|
| Ingest | `ingest.py` | Pull nflverse games, cache raw parquet |
| Features | `features.py` | Rolling prior-form rating, leakage-guarded |
| Model | `evaluate.py` | Logistic regression on one feature |
| Evaluate | `evaluate.py` | Walk-forward backtest, calibration, skill |

The point of a vertical slice is that the schema gets to be wrong while fixing
it is a 200-line change instead of a rewrite.

## Backtest results

4,338 games, seasons 2010–2025, walk-forward (train on all prior seasons,
predict season S, step forward).

| Forecast | Log loss | Brier | Sharpness | Skill vs base rate |
|---|---|---|---|---|
| Base rate | 0.6868 | 0.2468 | 0.003 | 0.000 |
| Our model | 0.6408 | 0.2250 | 0.141 | 0.089 |
| Market | 0.6089 | 0.2108 | 0.181 | 0.146 |

Read this as: one feature captures roughly 60% of the market's edge over
guessing. The remaining gap is what the other fourteen data categories in the
manifesto are for.

## Rules this repo enforces

1. **No leakage.** Every feature is computable strictly before kickoff.
   `tests/test_features.py` asserts it. Do not delete those tests.
2. **No random splits.** Time-series data gets walk-forward validation only.
3. **Baselines always reported.** A score without a reference point is
   decoration. Every result is quoted against base rate and market.
4. **Raw data is immutable.** Ingestion caches bytes; transformation happens
   downstream, where it can be re-run.
5. **Logic lives in `src/`, not in notebooks.** Notebooks import and explore.

## Decisions

Architecture decision records in `docs/decisions/`. One file per real choice:
context, options, decision, consequences. Add one whenever a decision would be
hard to reconstruct in six months.

## Roadmap

- [x] Vertical slice: ingest → feature → model → evaluation
- [ ] Rest, travel, and home-field features
- [ ] Slice calibration by subgroup (dome/outdoor, favorite/dog, early/late)
- [ ] Play-by-play ingestion; EPA-based team ratings
- [ ] Margin distribution instead of binary win/loss
- [ ] Monte Carlo layer for coherent joint outcomes
- [ ] Player-level models with hierarchical partial pooling
