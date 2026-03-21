#!/usr/bin/env python3
"""Astar Island — main orchestrator.

Runs the full prediction pipeline:
  Phase 0: Fetch initial states, build masks & archetypes  (FREE)
  Phase 1: Coverage-first queries (interleaved across seeds)
  Phase 2: Adaptive repeat queries on high-uncertainty areas
  Phase 3: Generate probability predictions
  Phase 4: Post-process (floor + normalize)
  Phase 5: Submit predictions for all seeds

Usage:
  uv run run.py                     # Full pipeline: query + predict + submit
  uv run run.py --predict-only      # Re-predict from saved observations + submit
  uv run run.py <round_id>          # Specify round ID
  uv run run.py --predict-only <round_id>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from client import AstarClient, SimulationResult
from features import SeedAnalysis
from observation_store import ObservationStore
from predictor import (
    compute_round_tau,
    predict_full_grid_vectorized,
    set_round_tau,
)
from query_strategy import (
    QueryPlan,
    plan_coverage_queries,
    plan_repeat_queries,
)

# Directory for persisted observation data
DATA_DIR = Path(__file__).parent / "data"


def log(msg: str) -> None:
    """Timestamped log output."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _obs_path(round_id: str) -> Path:
    """Path for saved observation data for a given round."""
    return DATA_DIR / f"obs_{round_id[:8]}"


def _setup_round(
    client: AstarClient, round_id: str | None
) -> tuple[str, list[SeedAnalysis], int, int, int]:
    """Phase 0: fetch round info and build seed analyses. Returns (round_id, analyses, width, height, seeds_count)."""
    log("Phase 0: Fetching round info...")

    if round_id is None:
        active = client.get_active_round()
        if active is None:
            log("ERROR: No active round found.")
            sys.exit(1)
        round_id = active["id"]
        log(f"  Found active round: {active['round_number']} ({round_id})")
    else:
        log(f"  Using specified round: {round_id}")

    assert isinstance(round_id, str)
    detail = client.get_round_detail(round_id)
    width = detail.map_width
    height = detail.map_height
    seeds_count = detail.seeds_count

    log(f"  Map: {width}x{height}, {seeds_count} seeds")

    seed_analyses: list[SeedAnalysis] = []
    for i, state in enumerate(detail.initial_states):
        analysis = SeedAnalysis(state["grid"], state["settlements"])
        seed_analyses.append(analysis)
        n_settlements = len(state["settlements"])
        n_dynamic = int(analysis.priority_mask.sum())
        log(f"  Seed {i}: {n_settlements} settlements, {n_dynamic} dynamic cells")

    return round_id, seed_analyses, width, height, seeds_count


def _query_phase(
    client: AstarClient,
    round_id: str,
    seed_analyses: list[SeedAnalysis],
    width: int,
    height: int,
    seeds_count: int,
) -> ObservationStore:
    """Phases 1-2: execute coverage + adaptive repeat queries."""
    # Check budget
    try:
        budget_info = client.get_budget()
        queries_used = budget_info["queries_used"]
        queries_max = budget_info["queries_max"]
        log(f"  Budget: {queries_used}/{queries_max} queries used")
    except Exception:
        queries_used = 0
        queries_max = 50
        log("  Budget check failed, assuming 0/50")

    remaining_budget = queries_max - queries_used

    # Phase 1: Coverage-first queries
    log(f"Phase 1: Planning coverage queries (budget: {remaining_budget})...")

    observation_store = ObservationStore(seeds_count, height, width)

    coverage_queries = plan_coverage_queries(
        seeds_count,
        width,
        height,
        max_budget=remaining_budget,
        seed_analyses=seed_analyses,
    )
    n_coverage = len(coverage_queries)
    repeat_budget = max(0, remaining_budget - n_coverage)
    log(f"  Coverage: {n_coverage} queries, repeat budget: {repeat_budget}")

    log("  Executing coverage queries...")
    for i, q in enumerate(coverage_queries):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=q.seed_index,
                viewport_x=q.viewport_x,
                viewport_y=q.viewport_y,
                viewport_w=q.viewport_w,
                viewport_h=q.viewport_h,
            )
            observation_store.add_observation(
                seed_index=q.seed_index,
                viewport=result.viewport,
                grid=result.grid,
                archetypes=seed_analyses[q.seed_index].archetypes,
            )
            if (i + 1) % 10 == 0 or i == n_coverage - 1:
                log(
                    f"    [{i + 1}/{n_coverage}] seed={q.seed_index} "
                    f"vp=({q.viewport_x},{q.viewport_y}) "
                    f"budget={result.queries_used}/{result.queries_max}"
                )
        except Exception as e:
            log(f"    [{i + 1}/{n_coverage}] ERROR: {e}")
            if "429" in str(e):
                log("    Rate limited or budget exhausted, stopping coverage.")
                repeat_budget = 0
                break

    # Phase 2: Adaptive repeat queries
    if repeat_budget > 0:
        log(f"Phase 2: Planning {repeat_budget} adaptive repeat queries...")

        repeat_queries = plan_repeat_queries(
            observation_store,
            seed_analyses,
            remaining_budget=repeat_budget,
            map_w=width,
            map_h=height,
        )

        log(f"  Executing {len(repeat_queries)} repeat queries...")
        for i, q in enumerate(repeat_queries):
            try:
                result = client.simulate(
                    round_id=round_id,
                    seed_index=q.seed_index,
                    viewport_x=q.viewport_x,
                    viewport_y=q.viewport_y,
                    viewport_w=q.viewport_w,
                    viewport_h=q.viewport_h,
                )
                observation_store.add_observation(
                    seed_index=q.seed_index,
                    viewport=result.viewport,
                    grid=result.grid,
                    archetypes=seed_analyses[q.seed_index].archetypes,
                )
                log(
                    f"    [{i + 1}/{len(repeat_queries)}] seed={q.seed_index} "
                    f"vp=({q.viewport_x},{q.viewport_y}) "
                    f"budget={result.queries_used}/{result.queries_max}"
                )
            except Exception as e:
                log(f"    [{i + 1}/{len(repeat_queries)}] ERROR: {e}")
                if "429" in str(e):
                    log("    Budget exhausted, stopping repeats.")
                    break
    else:
        log("Phase 2: Skipped (no repeat budget)")

    # Save observations to disk
    DATA_DIR.mkdir(exist_ok=True)
    obs_file = _obs_path(round_id)
    observation_store.save(obs_file)

    return observation_store


def _predict_and_submit(
    client: AstarClient,
    round_id: str,
    observation_store: ObservationStore,
    seed_analyses: list[SeedAnalysis],
    seeds_count: int,
    height: int,
    width: int,
) -> None:
    """Phases 3-5: generate predictions and submit."""
    log("Phase 3: Generating predictions...")

    predictions: list[np.ndarray] = []
    for seed_idx in range(seeds_count):
        obs_count = observation_store.get_seed_obs_counts(seed_idx)
        n_observed = int((obs_count > 0).sum())
        n_total = height * width

        prediction = predict_full_grid_vectorized(
            seed_idx, observation_store, seed_analyses[seed_idx]
        )
        predictions.append(prediction)

        log(
            f"  Seed {seed_idx}: {n_observed}/{n_total} cells observed, "
            f"prediction shape={prediction.shape}"
        )

    # Phase 5: Submit
    log("Phase 5: Submitting predictions...")

    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]

        # Validate
        sums = pred.sum(axis=-1)
        max_deviation = np.max(np.abs(sums - 1.0))
        has_negative = np.any(pred < 0)
        log(
            f"  Seed {seed_idx}: max sum deviation={max_deviation:.6f}, "
            f"negatives={has_negative}"
        )

        # Submit with retry (submit endpoint has 2 req/s limit)
        for attempt in range(3):
            try:
                time.sleep(0.6)  # 2 req/s limit → 0.5s + margin
                resp = client.submit(
                    round_id=round_id,
                    seed_index=seed_idx,
                    prediction=pred.tolist(),
                )
                log(f"  Seed {seed_idx}: submitted -> {resp}")
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    log(f"  Seed {seed_idx}: rate limited, retrying in 2s...")
                    time.sleep(2)
                else:
                    log(f"  Seed {seed_idx}: ERROR submitting: {e}")

    log("Done!")

    # Print observation summary
    log("--- Observation Summary ---")
    for seed_idx in range(seeds_count):
        obs = observation_store.get_seed_obs_counts(seed_idx)
        n_obs_0 = int((obs == 0).sum())
        n_obs_1 = int((obs == 1).sum())
        n_obs_2p = int((obs >= 2).sum())
        log(f"  Seed {seed_idx}: unobserved={n_obs_0}, once={n_obs_1}, 2+={n_obs_2p}")


def run(round_id: str | None = None, predict_only: bool = False) -> None:
    """Execute the Astar Island prediction pipeline.

    Args:
        round_id: UUID of the round (auto-detects active round if None).
        predict_only: If True, load saved observations instead of querying.
    """
    client = AstarClient()

    # Phase 0: Setup
    round_id, seed_analyses, width, height, seeds_count = _setup_round(client, round_id)

    if predict_only:
        # Load saved observations
        obs_file = _obs_path(round_id)
        if not obs_file.with_suffix(".npz").exists():
            log(f"ERROR: No saved observations at {obs_file}.npz")
            log("  Run without --predict-only first to query and save observations.")
            sys.exit(1)
        log(f"  Loading saved observations from {obs_file}...")
        observation_store = ObservationStore.load(obs_file)
    else:
        # Full query pipeline
        observation_store = _query_phase(
            client, round_id, seed_analyses, width, height, seeds_count
        )

    tau = compute_round_tau(seed_analyses, observation_store)
    set_round_tau(tau)

    _predict_and_submit(
        client,
        round_id,
        observation_store,
        seed_analyses,
        seeds_count,
        height,
        width,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astar Island prediction pipeline")
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load saved observations and re-predict (no new queries)",
    )
    parser.add_argument(
        "round_id",
        nargs="?",
        default=None,
        help="Round UUID (auto-detects active round if omitted)",
    )
    args = parser.parse_args()
    run(round_id=args.round_id, predict_only=args.predict_only)
