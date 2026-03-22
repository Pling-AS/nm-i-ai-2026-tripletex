#!/usr/bin/env python3
"""Train the correction table from historical completed rounds.

For each completed round:
  1. Fetch initial states and ground truth from the API
  2. Replay the current prediction pipeline (Dirichlet + spatial smooth)
     using stochastic observations sampled from ground truth
  3. Compute residuals: ground_truth - prediction
  4. Accumulate into bucketed correction table

The table is saved as correction_table.json and loaded by predictor.py.

Usage:
  uv run train_correction.py              # Train on all completed rounds
  uv run train_correction.py --dry-run    # Show stats without saving
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from client import AstarClient
from correction import (
    train_correction_table,
    save_correction_table,
    compute_correction_features,
    _parent_key,
)
from features import SeedAnalysis
from observation_store import ObservationStore
from predictor import (
    compute_round_tau,
    predict_full_grid_vectorized,
    set_round_tau,
    set_settlement_vitality,
)
from query_strategy import plan_coverage_queries, plan_repeat_queries
from utils import NUM_CLASSES


def ts() -> str:
    return time.strftime("%H:%M:%S")


def replay_round(
    client: AstarClient,
    round_id: str,
    round_number: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, SeedAnalysis]]:
    """Replay one round through the current pipeline.

    Returns list of (prediction, ground_truth, seed_analysis) per seed.
    """
    print(f"[{ts()}] Replaying round {round_number} ({round_id[:8]})...")

    detail = client.get_round_detail(round_id)
    h, w = detail.map_height, detail.map_width
    seeds_count = detail.seeds_count

    # Build seed analyses and fetch ground truth
    seed_analyses: list[SeedAnalysis] = []
    ground_truths: list[np.ndarray] = []

    for i, state in enumerate(detail.initial_states):
        sa = SeedAnalysis(state["grid"], state["settlements"])
        seed_analyses.append(sa)

        try:
            data = client.get_analysis(round_id, i)
            gt = np.array(data["ground_truth"], dtype=np.float64)
            ground_truths.append(gt)
        except Exception as e:
            print(f"  ERROR fetching GT for seed {i}: {e}")
            ground_truths.append(None)

    if all(gt is None for gt in ground_truths):
        print(f"  No ground truth available, skipping round {round_number}")
        return []

    # Sample stochastic observations from ground truth (like simulate_round.py)
    def sample_viewport(
        seed_idx: int, vx: int, vy: int, vw: int, vh: int,
    ) -> list[list[int]]:
        gt = ground_truths[seed_idx]
        if gt is None:
            return [[0] * vw for _ in range(vh)]
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

    # Plan and execute queries (same as real pipeline)
    coverage_queries = plan_coverage_queries(
        seeds_count, w, h, max_budget=45, seed_analyses=seed_analyses
    )

    store = ObservationStore(seeds_count, h, w)

    for q in coverage_queries:
        if ground_truths[q.seed_index] is None:
            continue
        obs_grid = sample_viewport(
            q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h
        )
        store.add_observation(
            seed_index=q.seed_index,
            viewport={
                "x": q.viewport_x, "y": q.viewport_y,
                "w": q.viewport_w, "h": q.viewport_h,
            },
            grid=obs_grid,
            archetypes=seed_analyses[q.seed_index].archetypes,
        )

    # Repeat queries
    repeat_queries = plan_repeat_queries(
        store, seed_analyses, remaining_budget=5, map_w=w, map_h=h
    )
    for q in repeat_queries:
        if ground_truths[q.seed_index] is None:
            continue
        obs_grid = sample_viewport(
            q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h
        )
        store.add_observation(
            seed_index=q.seed_index,
            viewport={
                "x": q.viewport_x, "y": q.viewport_y,
                "w": q.viewport_w, "h": q.viewport_h,
            },
            grid=obs_grid,
            archetypes=seed_analyses[q.seed_index].archetypes,
        )

    # Compute tau and generate predictions
    tau = compute_round_tau(seed_analyses, store)
    set_round_tau(tau)
    set_settlement_vitality(None)  # No vitality data in replay

    results = []
    for seed_idx in range(seeds_count):
        gt = ground_truths[seed_idx]
        if gt is None:
            continue

        # This produces the prediction AFTER spatial smoothing (the exact
        # insertion point where correction will be applied)
        prediction = predict_full_grid_vectorized(seed_idx, store, seed_analyses[seed_idx])

        results.append((prediction, gt, seed_analyses[seed_idx]))
        print(f"  Seed {seed_idx}: prediction shape={prediction.shape}")

    print(f"  Round {round_number}: {len(results)} seeds replayed")
    return results


def score_seed(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute entropy-weighted KL divergence score."""
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(gt > 0, -gt * np.log(gt), 0.0).sum(axis=-1)
        kl = np.where(gt > 0, gt * np.log(gt / np.maximum(pred, 1e-10)), 0.0).sum(axis=-1)
    d = e > 1e-8
    if not d.any():
        return 100.0
    wkl = (e[d] * kl[d]).sum() / e[d].sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train correction table")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed")
    args = parser.parse_args()

    print(f"[{ts()}] === TRAINING CORRECTION TABLE ===")

    client = AstarClient()
    rng = np.random.default_rng(args.seed)

    # Fetch all completed rounds
    print(f"[{ts()}] Fetching rounds...")
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    print(f"  Found {len(completed)} completed rounds")

    all_rows: list[tuple[np.ndarray, np.ndarray, SeedAnalysis]] = []

    for r in completed:
        rows = replay_round(client, r["id"], r["round_number"], rng)
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No training data collected")
        return

    print(f"\n[{ts()}] Training on {len(all_rows)} seed-rounds...")
    table = train_correction_table(all_rows)

    # Print summary
    print(f"\n=== CORRECTION TABLE SUMMARY ===")
    print(f"  Global cells: {table['global']['n']}")
    print(f"  Global residual: {[f'{v:+.4f}' for v in table['global']['residual']]}")
    print(f"  Parent buckets: {table['meta']['n_parent_buckets']}")
    print(f"  Full buckets:   {table['meta']['n_full_buckets']}")

    # Show top parent buckets by absolute settlement residual
    parent = table["parent"]
    by_sett = sorted(
        parent.items(),
        key=lambda kv: abs(kv[1]["residual"][1]),
        reverse=True,
    )
    print(f"\n  Top 10 parent buckets by |settlement residual|:")
    for k, v in by_sett[:10]:
        resid = v["residual"]
        print(f"    {k:20s}  n={v['n']:5d}  sett={resid[1]:+.4f}  "
              f"empty={resid[0]:+.4f}  forest={resid[4]:+.4f}")

    # Score impact estimate (on training data, so optimistic)
    print(f"\n[{ts()}] Scoring impact estimate (in-sample)...")
    from correction import apply_correction

    scores_before = []
    scores_after = []
    for pred, gt, sa in all_rows:
        scores_before.append(score_seed(gt, pred))
        corrected = apply_correction(pred, sa, table)
        scores_after.append(score_seed(gt, corrected))

    avg_before = np.mean(scores_before)
    avg_after = np.mean(scores_after)
    print(f"  Average score before: {avg_before:.2f}")
    print(f"  Average score after:  {avg_after:.2f}")
    print(f"  Delta:                {avg_after - avg_before:+.2f}")
    print(f"  (In-sample; real impact will be smaller)")

    if not args.dry_run:
        save_correction_table(table)
        print(f"\n[{ts()}] Done! correction_table.json saved.")
    else:
        print(f"\n[{ts()}] Dry run complete (not saved).")


if __name__ == "__main__":
    main()
