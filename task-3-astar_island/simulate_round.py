#!/usr/bin/env python3
"""Simulate a full round using Round 2 ground truth as the oracle.

Tests the entire pipeline: tiling → query → observe → predict → score.
Validates that the new predictor (τ=15, archetype backoff, entropy-aware
tiling, updated calibration) works end-to-end.
"""

from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from client import AstarClient
from features import SeedAnalysis
from observation_store import ObservationStore
from predictor import predict_full_grid_vectorized, TAU
from query_strategy import plan_coverage_queries, plan_repeat_queries


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * np.log(p / np.maximum(q, 1e-10)), 0.0).sum(axis=-1)


def ent(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, -p * np.log(p), 0.0).sum(axis=-1)


def score_seed(gt: np.ndarray, pred: np.ndarray) -> float:
    e = ent(gt)
    kl = kl_div(gt, pred)
    d = e > 1e-8
    if not d.any():
        return 100.0
    wkl = (e[d] * kl[d]).sum() / e[d].sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def ts() -> str:
    return time.strftime("%H:%M:%S")


def main() -> None:
    print(f"[{ts()}] === SIMULATED ROUND (using Round 2 ground truth) ===")
    print(f"[{ts()}] TAU = {TAU}")

    c = AstarClient()
    r2_id = "76909e29-f664-4b2f-b16b-61b7507277e9"
    detail = c.get_round_detail(r2_id)
    h, w = detail.map_height, detail.map_width
    seeds_count = detail.seeds_count
    rng = np.random.default_rng(2026)

    # Phase 0: Build seed analyses + fetch ground truth
    print(f"[{ts()}] Phase 0: Building seed analyses...")
    seed_analyses: list[SeedAnalysis] = []
    ground_truths: list[np.ndarray] = []
    for i, state in enumerate(detail.initial_states):
        sa = SeedAnalysis(state["grid"], state["settlements"])
        seed_analyses.append(sa)
        gt = np.array(c.get_analysis(r2_id, i)["ground_truth"])
        ground_truths.append(gt)
        n_dyn = int(sa.priority_mask.sum())
        print(
            f"  Seed {i}: {len(state['settlements'])} settlements, {n_dyn} dynamic cells"
        )

    # Phase 1: Plan coverage queries (entropy-aware)
    print(f"[{ts()}] Phase 1: Planning coverage queries...")
    coverage_queries = plan_coverage_queries(
        seeds_count, w, h, max_budget=45, seed_analyses=seed_analyses
    )
    print(f"  {len(coverage_queries)} coverage queries planned")

    store = ObservationStore(seeds_count, h, w)

    def sample_viewport(
        seed_idx: int, vx: int, vy: int, vw: int, vh: int
    ) -> list[list[int]]:
        """Sample stochastic observation from ground truth."""
        gt = ground_truths[seed_idx]
        obs_grid = []
        for ry in range(vh):
            row = []
            for rx in range(vw):
                ay, ax = vy + ry, vx + rx
                if ay < h and ax < w:
                    probs = gt[ay, ax]
                    s = probs.sum()
                    cls = int(rng.choice(6, p=probs / s)) if s > 0 else 0
                    row.append(cls)
                else:
                    row.append(0)
            obs_grid.append(row)
        return obs_grid

    # Execute coverage queries
    for q in coverage_queries:
        obs_grid = sample_viewport(
            q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h
        )
        store.add_observation(
            seed_index=q.seed_index,
            viewport={
                "x": q.viewport_x,
                "y": q.viewport_y,
                "w": q.viewport_w,
                "h": q.viewport_h,
            },
            grid=obs_grid,
            archetypes=seed_analyses[q.seed_index].archetypes,
        )

    # Phase 2: Adaptive repeats
    print(f"[{ts()}] Phase 2: Planning 5 repeat queries...")
    repeat_queries = plan_repeat_queries(
        store, seed_analyses, remaining_budget=5, map_w=w, map_h=h
    )
    print(f"  {len(repeat_queries)} repeat queries planned")

    for q in repeat_queries:
        obs_grid = sample_viewport(
            q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h
        )
        store.add_observation(
            seed_index=q.seed_index,
            viewport={
                "x": q.viewport_x,
                "y": q.viewport_y,
                "w": q.viewport_w,
                "h": q.viewport_h,
            },
            grid=obs_grid,
            archetypes=seed_analyses[q.seed_index].archetypes,
        )

    # Phase 3: Predict and score
    print(f"[{ts()}] Phase 3: Generating predictions...")
    scores = []
    for seed_idx in range(seeds_count):
        obs = store.get_seed_obs_counts(seed_idx)
        n0 = int((obs == 0).sum())
        n1 = int((obs == 1).sum())
        n2 = int((obs >= 2).sum())

        pred = predict_full_grid_vectorized(seed_idx, store, seed_analyses[seed_idx])

        # Validate
        sums = pred.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Sum check failed seed {seed_idx}"
        assert not np.any(pred < 0), f"Negative check failed seed {seed_idx}"

        s = score_seed(ground_truths[seed_idx], pred)
        scores.append(s)
        print(f"  Seed {seed_idx}: score={s:.2f}, unobs={n0}, N=1={n1}, N≥2={n2}")

    avg = np.mean(scores)
    print(f"\n[{ts()}] === RESULTS ===")
    print(f"  Simulated score: {avg:.2f}")
    print(f"  Round 2 actual:  74.40 (OLD predictor)")
    print(f"  Improvement:     {avg - 74.40:+.2f} points")
    print(f"  Top team R2:     ~98.8")
    print(f"\n  Pipeline validation: PASSED ✓")


if __name__ == "__main__":
    main()
