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
  uv run run.py              # Auto-detect active round
  uv run run.py <round_id>   # Specify round ID
"""

from __future__ import annotations

import sys
import time

import numpy as np

from client import AstarClient, SimulationResult
from features import SeedAnalysis
from observation_store import ObservationStore
from predictor import predict_full_grid_vectorized
from query_strategy import (
    QueryPlan,
    plan_coverage_queries,
    plan_repeat_queries,
)


def log(msg: str) -> None:
    """Timestamped log output."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run(round_id: str | None = None) -> None:
    """Execute the full Astar Island prediction pipeline."""
    client = AstarClient()

    # ==================================================================
    # Phase 0: Get round info and initial states (FREE)
    # ==================================================================
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

    # Build per-seed analyses
    seed_analyses: list[SeedAnalysis] = []
    for i, state in enumerate(detail.initial_states):
        analysis = SeedAnalysis(state["grid"], state["settlements"])
        seed_analyses.append(analysis)
        n_settlements = len(state["settlements"])
        n_dynamic = int(analysis.priority_mask.sum())
        log(f"  Seed {i}: {n_settlements} settlements, {n_dynamic} dynamic cells")

    # ==================================================================
    # Phase 0b: Check budget
    # ==================================================================
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

    # ==================================================================
    # Phase 1: Coverage-first queries (interleaved across seeds)
    # ==================================================================
    log(f"Phase 1: Planning coverage queries (budget: {remaining_budget})...")

    observation_store = ObservationStore(seeds_count, height, width)

    # Plan coverage queries
    coverage_queries = plan_coverage_queries(
        seeds_count, width, height, max_budget=remaining_budget
    )
    n_coverage = len(coverage_queries)

    # Reserve some budget for repeats
    repeat_budget = max(0, remaining_budget - n_coverage)
    log(f"  Coverage: {n_coverage} queries, repeat budget: {repeat_budget}")

    # Execute coverage queries
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

    # ==================================================================
    # Phase 2: Adaptive repeat queries
    # ==================================================================
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

    # ==================================================================
    # Phase 3 & 4: Generate predictions
    # ==================================================================
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

    # ==================================================================
    # Phase 5: Submit predictions
    # ==================================================================
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

        try:
            resp = client.submit(
                round_id=round_id,
                seed_index=seed_idx,
                prediction=pred.tolist(),
            )
            log(f"  Seed {seed_idx}: submitted -> {resp}")
        except Exception as e:
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


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    run(round_id=rid)
