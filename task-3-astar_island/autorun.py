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
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from client import AstarClient

PAUSE_FILE = Path.home() / "astar" / "PAUSE"
PAUSE_TIMEOUT_SECONDS = 30 * 60

# Circuit breaker: auto-pause if a round scores below this threshold.
# Prevents catastrophic code from submitting bad predictions on subsequent rounds.
SCORE_FLOOR = 75.0


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


def has_observation_data(round_id: str) -> bool:
    """Check if we have saved observation data for a round."""
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    return (data_dir / f"obs_{round_id[:8]}.npz").exists()


def run_resubmit(round_id: str) -> bool:
    """Re-predict and resubmit using saved observations with updated calibration."""
    cmd = [sys.executable, "run.py", "--predict-only", round_id]
    log(f"Resubmitting with updated calibration: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode == 0:
        log("Resubmit complete — predictions updated with fresh calibration")
        return True
    else:
        log(f"Resubmit failed (exit code {result.returncode})")
        return False


def resubmit_active_if_needed(client: AstarClient) -> None:
    """If there's an active round with existing observation data, resubmit it.

    This ensures that when calibration is updated after analyzing a completed
    round, any already-submitted active round gets re-predicted with the
    fresh priors.
    """
    active = find_active_round(client)
    if not active:
        log("No active round to resubmit")
        return

    round_id = active["id"]
    if has_observation_data(round_id):
        log(
            f"Active round {round_id[:8]} has observation data — resubmitting with fresh calibration"
        )
        run_resubmit(round_id)
    else:
        log(
            f"Active round {round_id[:8]} has no observation data yet — skipping resubmit"
        )


def check_pause(round_info: dict) -> None:
    """Wait if PAUSE file exists (Magnus is at the computer).

    If PAUSE file is older than PAUSE_TIMEOUT_SECONDS, auto-proceed
    (Magnus is asleep or stepped away). Poll every 15s while paused.
    """
    if not PAUSE_FILE.exists():
        return

    pause_age = time.time() - PAUSE_FILE.stat().st_mtime
    if pause_age > PAUSE_TIMEOUT_SECONDS:
        log(
            f"PAUSE file exists but is {pause_age / 60:.0f}min old "
            f"(>{PAUSE_TIMEOUT_SECONDS // 60}min) — auto-proceeding"
        )
        return

    round_num = round_info.get("round_number", "?")
    closes_at = round_info.get("closes_at", "unknown")
    log(
        f"⏸  PAUSE file detected — waiting for Magnus. "
        f"Round {round_num} closes at {closes_at}. "
        f"Remove ~/astar/PAUSE to proceed, or auto-proceeds in "
        f"{(PAUSE_TIMEOUT_SECONDS - pause_age) / 60:.0f}min."
    )

    while PAUSE_FILE.exists():
        pause_age = time.time() - PAUSE_FILE.stat().st_mtime
        if pause_age > PAUSE_TIMEOUT_SECONDS:
            log(f"PAUSE timeout ({PAUSE_TIMEOUT_SECONDS // 60}min) — auto-proceeding")
            return
        time.sleep(15)

    log("▶  PAUSE file removed — proceeding with pipeline")


def check_score_circuit_breaker(client: AstarClient, round_id: str) -> None:
    """Auto-pause if last round scored below SCORE_FLOOR.

    This protects against broken code submitting bad predictions on
    every subsequent round.  Creates PAUSE file and logs a warning.
    Remove the PAUSE file to resume after investigating.
    """
    try:
        my_rounds = client.get_my_rounds()
        r = next((r for r in my_rounds if r["id"] == round_id), None)
        if not r:
            return
        score = r.get("round_score")
        if score is None:
            return
        rank = r.get("rank", "?")
        total = r.get("total_teams", "?")
        log(f"Round scored: {score:.2f} (rank #{rank}/{total})")
        if score < SCORE_FLOOR:
            log(
                f"⚠ CIRCUIT BREAKER: score {score:.2f} < floor {SCORE_FLOOR}! "
                f"Creating PAUSE file to prevent further submissions. "
                f"SSH in, investigate, and remove ~/astar/PAUSE to resume."
            )
            PAUSE_FILE.touch()
    except Exception as e:
        log(f"Score check error (non-fatal): {e}")


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
        resubmit_active_if_needed(client)
    else:
        log("No active round — checking for recently completed rounds to analyze")
        run_analysis()
        resubmit_active_if_needed(client)

    while True:
        next_round = wait_for_next_round(client)
        round_id = next_round["id"]

        check_pause(next_round)
        run_pipeline(round_id)

        if args.once:
            log("--once mode: exiting after first pipeline run")
            break

        wait_for_round_completion(client, round_id)
        wait_for_scoring_done(client, round_id)
        check_score_circuit_breaker(client, round_id)
        run_analysis()
        resubmit_active_if_needed(client)

        log("=== Cycle complete, waiting for next round ===")


if __name__ == "__main__":
    main()
