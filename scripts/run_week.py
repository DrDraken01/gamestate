"""Run the weekly pipeline: ingest -> predict, in order, fail-fast.

    DATABASE_URL=postgresql://localhost/gamestate python scripts/run_week.py
    DATABASE_URL=... python scripts/run_week.py --season 2026 --week 8

One command so "did I load first?" stops being a failure mode. Stages run in
sequence and the pipeline STOPS at the first failure -- it does not press on.

WHY SUBPROCESSES, NOT IMPORTS

Each stage is an independent script with its own exit code. Running them as
subprocesses gives real process isolation (a crash in predict cannot corrupt the
orchestrator) and keeps load_db.py / predict.py untouched -- they stay standalone
commands, which is what they are. The exit code IS the stage-status contract:
0 = success, nonzero = stop.

THE GOVERNING PRINCIPLE: accuracy over availability

A missing prediction is a signal that something broke; a silently partial or
stale one is a lie that looks official and would quietly corrupt later accuracy
evaluation. So:

    LOAD --x--> STOP           (bad ingest never feeds a prediction)
    LOAD --ok--> PREDICT --x--> STOP, nothing new published
    LOAD --ok--> PREDICT --ok--> done

Ingested data that committed stays committed -- it is valid regardless of what
happens downstream. What cannot happen is a half-written prediction batch; that
is guaranteed by the single transaction in predict_writer.write_prediction_batch.

NOT YET BUILT, deliberately: per-stage observability tables (pipeline_run /
pipeline_stage). That is a real schema addition with its own ADR -- a stubbed
status table would be the "looks legitimate but isn't" trap this pipeline exists
to avoid. Stage status is logged for now.
"""

from __future__ import annotations

import argparse
import logging
import subprocess  # nosec B404 -- runs only this repo's own scripts via sys.executable
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_week")

SCRIPTS = Path(__file__).resolve().parent


def run_stage(name: str, argv: list[str]) -> bool:
    """Run one stage as a subprocess. Return True on success, False on failure."""
    logger.info("stage %s: starting", name)
    result = subprocess.run(  # nosec B603 -- absolute interpreter path, fixed args, no shell
        [sys.executable, *argv],
        check=False,
    )
    if result.returncode != 0:
        logger.error("stage %s: FAILED (exit %d) -- pipeline stopped", name, result.returncode)
        return False
    logger.info("stage %s: success", name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the weekly ingest -> predict pipeline.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    predict_args = [str(SCRIPTS / "predict.py")]
    if args.season is not None:
        predict_args += ["--season", str(args.season)]
    if args.week is not None:
        predict_args += ["--week", str(args.week)]

    stages = [
        ("INGEST", [str(SCRIPTS / "load_db.py")]),
        ("PREDICT", predict_args),
    ]

    for name, argv in stages:
        if not run_stage(name, argv):
            return 1  # fail-fast: no later stage runs

    logger.info("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
