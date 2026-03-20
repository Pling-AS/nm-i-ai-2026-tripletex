#!/usr/bin/env python3
"""Autonomous round monitor: wait → analyze → calibrate → submit next round.

Designed to run unattended overnight. Loops forever, handling each round as it
appears.

Usage:
    uv run autorun.py                   # Start monitoring
    uv run autorun.py --once            # Handle one round cycle then exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

from client import AstarClient


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def wait_for_round_completion(
    client: AstarClient, round_id: str, poll_interval: int = 30
) -> None:
    """Poll until the given round is no longer active."""
    log(
        f"Waiting for round {round_id[:8]} to complete (polling every {poll_interval}s)..."
    )
    while True:
        try:
            rounds = client.get_rounds()
            r = next((r for r in rounds if r["id"] == round_id), None)
            if r is None or r["status"] != "active":
                log(f"Round {round_id[:8]} status: {r['status'] if r else 'not found'}")
                return
        except Exception as e:
            log(f"Poll error: {e}")
        time.sleep(poll_interval)


def wait_for_scoring_done(
    client: AstarClient, round_id: str, timeout: int = 600
) -> bool:
    """Wait for a round to transition from 'scoring' to 'completed'."""
    log(f"Waiting for scoring to finish (timeout {timeout}s)...")
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            rounds = client.get_rounds()
            r = next((r for r in rounds if r["id"] == round_id), None)
            if r and r["status"] == "completed":
                log("Scoring complete!")
                return True
            if r:
                log(f"  Status: {r['status']}")
        except Exception as e:
            log(f"  Poll error: {e}")
        time.sleep(30)
    log("Scoring timeout — proceeding anyway")
    return False


def run_analysis() -> bool:
    """Run analyze.py --all to update calibration."""
    log("Running analysis (analyze.py --all)...")
    result = subprocess.run(
        [sys.executable, "analyze.py", "--all"],
        capture_output=False,
        text=True,
    )
    if result.returncode == 0:
        log("Analysis complete — calibration updated")
        return True
    else:
        log(f"Analysis failed (exit code {result.returncode})")
        return False


def run_pipeline(round_id: str | None = None) -> bool:
    """Run the full query + predict + submit pipeline."""
    cmd = [sys.executable, "run.py"]
    if round_id:
        cmd.append(round_id)
    log(f"Running pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode == 0:
        log("Pipeline complete — predictions submitted")
        return True
    else:
        log(f"Pipeline failed (exit code {result.returncode})")
        return False


def find_active_round(client: AstarClient) -> dict | None:
    """Find the current active round."""
    try:
        rounds = client.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)
    except Exception as e:
        log(f"Error fetching rounds: {e}")
        return None


def wait_for_next_round(client: AstarClient, poll_interval: int = 60) -> dict:
    """Poll until a new active round appears."""
    log(f"Waiting for next round (polling every {poll_interval}s)...")
    while True:
        active = find_active_round(client)
        if active:
            log(
                f"New active round found: {active['id'][:8]} (round {active['round_number']})"
            )
            return active
        time.sleep(poll_interval)


def handle_current_round(client: AstarClient) -> str | None:
    """If there's an active round, wait for it to finish and analyze."""
    active = find_active_round(client)
    if not active:
        return None

    round_id = active["id"]
    closes_at = active.get("closes_at", "unknown")
    log(
        f"Active round: {round_id[:8]} (round {active['round_number']}, closes {closes_at})"
    )

    wait_for_round_completion(client, round_id)
    wait_for_scoring_done(client, round_id)
    run_analysis()
    return round_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous round monitor")
    parser.add_argument(
        "--once", action="store_true", help="Handle one cycle then exit"
    )
    args = parser.parse_args()

    client = AstarClient()
    log("=== Autorun started ===")

    completed_id = handle_current_round(client)
    if completed_id:
        log(f"Round {completed_id[:8]} analyzed and calibration updated")
    else:
        log("No active round — checking for recently completed rounds to analyze")
        run_analysis()

    while True:
        next_round = wait_for_next_round(client)
        round_id = next_round["id"]

        run_pipeline(round_id)

        if args.once:
            log("--once mode: exiting after first pipeline run")
            break

        wait_for_round_completion(client, round_id)
        wait_for_scoring_done(client, round_id)
        run_analysis()

        log("=== Cycle complete, waiting for next round ===")


if __name__ == "__main__":
    main()
