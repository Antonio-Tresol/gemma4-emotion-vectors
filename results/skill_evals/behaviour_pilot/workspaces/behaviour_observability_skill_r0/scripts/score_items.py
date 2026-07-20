#!/usr/bin/env python3
"""Score every item in items.csv via mock_api.call_model, resumably.

Usage: python scripts/score_items.py [--items items.csv] [--outdir results]
                                      [--seed 0] [--max-attempts 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mock_api import MockAPIError, MockTimeout, call_model  # noqa: E402

MODEL_ID = "mock-1"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("score_items")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_items(items_path: Path) -> list[tuple[str, str]]:
    with items_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["item_id", "text"]:
            raise ValueError(f"unexpected columns in {items_path}: {reader.fieldnames}")
        return [(row["item_id"], row["text"]) for row in reader]


def load_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(row["item_id"])
    return done


def score_one(item_id: str, text: str, max_attempts: int, logger: logging.Logger) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = call_model(item_id, text)
            return {
                "item_id": item_id,
                "status": "ok",
                "attempt": attempt,
                "model": result["model"],
                "score": result["score"],
            }
        except MockAPIError as e:
            # Permanent failure: retrying is wasted budget, record and stop.
            logger.warning("item %s: permanent failure: %s", item_id, e)
            return {
                "item_id": item_id,
                "status": "error",
                "attempt": attempt,
                "error_type": "MockAPIError",
                "error": str(e),
            }
        except MockTimeout as e:
            last_err = e
            logger.info("item %s: transient timeout on attempt %d, retrying", item_id, attempt)
            if attempt < max_attempts:
                backoff = min(2 ** (attempt - 1), 4) * (0.5 + random.random() * 0.5)
                time.sleep(backoff)
    logger.warning("item %s: exhausted %d attempts: %s", item_id, max_attempts, last_err)
    return {
        "item_id": item_id,
        "status": "error",
        "attempt": max_attempts,
        "error_type": type(last_err).__name__ if last_err else "unknown",
        "error": str(last_err) if last_err else "exhausted retries",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="items.csv", type=Path)
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-attempts", default=5, type=int)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / "scores.jsonl"
    log_path = args.outdir / "run.log"
    logger = setup_logging(log_path)

    logger.info(
        "config: items=%s outdir=%s seed=%d max_attempts=%d model=%s git=%s",
        args.items, args.outdir, args.seed, args.max_attempts, MODEL_ID, git_commit(),
    )

    # Fail fast: validate inputs before doing any real work.
    if not args.items.exists():
        logger.error("items file not found: %s", args.items)
        raise SystemExit(1)
    items = load_items(args.items)
    if not items:
        logger.error("no items loaded from %s", args.items)
        raise SystemExit(1)
    logger.info("loaded %d items", len(items))

    rng = random.Random(args.seed)
    rng.shuffle(items)

    # Smoke test on a tiny slice before committing to the full run.
    smoke_id, smoke_text = items[0]
    try:
        score_one(smoke_id, smoke_text, args.max_attempts, logger)
    except Exception:
        logger.exception("smoke test failed on first item %s; aborting run", smoke_id)
        raise

    done = load_done(results_path)
    if done:
        logger.info("resuming: %d items already scored, skipping them", len(done))

    todo = [(iid, text) for iid, text in items if iid not in done]
    total = len(items)
    start = time.monotonic()

    with results_path.open("a") as out:
        for i, (item_id, text) in enumerate(todo, start=1):
            row = score_one(item_id, text, args.max_attempts, logger)
            out.write(json.dumps(row) + "\n")
            out.flush()

            if i % 20 == 0 or i == len(todo):
                elapsed = time.monotonic() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (len(todo) - i) / rate if rate > 0 else float("inf")
                logger.info(
                    "progress: %d/%d this-run (%d/%d total) rate=%.1f/s elapsed=%.1fs eta=%.1fs",
                    i, len(todo), len(done) + i, total, rate, elapsed, remaining,
                )

    # Final tally from the results file, the source of truth.
    outcomes = load_all_outcomes(results_path)
    n_ok = sum(1 for v in outcomes.values() if v == "ok")
    n_err = sum(1 for v in outcomes.values() if v == "error")
    logger.info("done: %d succeeded, %d failed, %d total", n_ok, n_err, len(outcomes))
    print(f"succeeded: {n_ok}")
    print(f"failed: {n_err}")


def load_all_outcomes(results_path: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            outcomes[row["item_id"]] = row["status"]
    return outcomes


if __name__ == "__main__":
    main()
