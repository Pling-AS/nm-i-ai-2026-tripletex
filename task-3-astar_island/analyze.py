#!/usr/bin/env python3
"""Post-round analysis for Astar Island.

Fetches ground truth, compares against our predictions, computes per-archetype
statistics, identifies systematic biases, and saves calibrated priors for
future rounds.

Usage:
  uv run analyze.py                  # Analyze most recent completed round
  uv run analyze.py <round_id>       # Analyze specific round
  uv run analyze.py --all            # Analyze all completed rounds
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from client import AstarClient
from features import CellArchetype, SeedAnalysis
from utils import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    NUM_CLASSES,
    TERRAIN_TO_CLASS,
    is_static,
)

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

TERRAIN_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# KL divergence computation
# ---------------------------------------------------------------------------
def kl_divergence(p: NDArray, q: NDArray, eps: float = 1e-12) -> float:
    """KL(p || q) — from ground truth p to prediction q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Only sum over classes where p > 0
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def entropy(p: NDArray, eps: float = 1e-12) -> float:
    """Shannon entropy H(p)."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------
def analyze_seed(
    seed_index: int,
    ground_truth: NDArray,
    prediction: NDArray | None,
    initial_grid: list[list[int]] | None,
    seed_analysis: SeedAnalysis,
) -> dict[str, Any]:
    """Analyze one seed: compute per-cell KL, aggregate by archetype."""
    h, w = ground_truth.shape[:2]

    cell_results: list[dict[str, Any]] = []
    archetype_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "kl_sum": 0.0,
            "entropy_sum": 0.0,
            "weighted_kl_sum": 0.0,
            "count": 0,
            "gt_sum": np.zeros(NUM_CLASSES, dtype=np.float64),
            "pred_sum": np.zeros(NUM_CLASSES, dtype=np.float64),
        }
    )
    terrain_stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "kl_sum": 0.0,
            "entropy_sum": 0.0,
            "weighted_kl_sum": 0.0,
            "count": 0,
            "gt_sum": np.zeros(NUM_CLASSES, dtype=np.float64),
        }
    )

    total_kl = 0.0
    total_entropy = 0.0
    total_weighted_kl = 0.0
    n_dynamic = 0

    for y in range(h):
        for x in range(w):
            gt = ground_truth[y, x]
            pred = prediction[y, x] if prediction is not None else None
            terrain_code = int(seed_analysis.grid[y, x])
            cell_entropy = entropy(gt)

            if pred is not None:
                cell_kl = kl_divergence(gt, pred)
            else:
                cell_kl = 0.0

            weighted_kl = cell_entropy * cell_kl

            # Archetype key
            archetype = seed_analysis.get_archetype(y, x)
            arch_key = str(archetype)

            archetype_stats[arch_key]["kl_sum"] += cell_kl
            archetype_stats[arch_key]["entropy_sum"] += cell_entropy
            archetype_stats[arch_key]["weighted_kl_sum"] += weighted_kl
            archetype_stats[arch_key]["count"] += 1
            archetype_stats[arch_key]["gt_sum"] += gt

            terrain_stats[terrain_code]["kl_sum"] += cell_kl
            terrain_stats[terrain_code]["entropy_sum"] += cell_entropy
            terrain_stats[terrain_code]["weighted_kl_sum"] += weighted_kl
            terrain_stats[terrain_code]["count"] += 1
            terrain_stats[terrain_code]["gt_sum"] += gt

            if pred is not None:
                archetype_stats[arch_key]["pred_sum"] += pred

            if cell_entropy > 0.01:
                n_dynamic += 1
                total_kl += cell_kl
                total_entropy += cell_entropy
                total_weighted_kl += weighted_kl

    # Compute averages
    avg_weighted_kl = total_weighted_kl / total_entropy if total_entropy > 0 else 0.0

    return {
        "seed_index": seed_index,
        "n_cells": h * w,
        "n_dynamic": n_dynamic,
        "total_entropy": total_entropy,
        "avg_weighted_kl": avg_weighted_kl,
        "archetype_stats": dict(archetype_stats),
        "terrain_stats": {str(k): v for k, v in terrain_stats.items()},
    }


# ---------------------------------------------------------------------------
# Calibrated priors generation
# ---------------------------------------------------------------------------
def generate_calibrated_priors(
    all_seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate ground truth distributions across all seeds by archetype.

    Returns calibration data that solution_spatial.py can load.
    """
    # Pool archetype stats across all seeds
    pooled_archetypes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "gt_sum": np.zeros(NUM_CLASSES, dtype=np.float64),
            "count": 0,
        }
    )
    pooled_terrains: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "gt_sum": np.zeros(NUM_CLASSES, dtype=np.float64),
            "count": 0,
        }
    )

    for result in all_seed_results:
        for arch_key, stats in result["archetype_stats"].items():
            pooled_archetypes[arch_key]["gt_sum"] += stats["gt_sum"]
            pooled_archetypes[arch_key]["count"] += stats["count"]

        for terrain_key, stats in result["terrain_stats"].items():
            pooled_terrains[terrain_key]["gt_sum"] += stats["gt_sum"]
            pooled_terrains[terrain_key]["count"] += stats["count"]

    archetype_priors: dict[str, list[float]] = {}
    archetype_counts: dict[str, int] = {}
    for arch_key, data in pooled_archetypes.items():
        if data["count"] > 0:
            avg = data["gt_sum"] / data["count"]
            archetype_priors[arch_key] = avg.tolist()
            archetype_counts[arch_key] = data["count"]

    terrain_priors: dict[str, list[float]] = {}
    terrain_counts: dict[str, int] = {}
    for terrain_key, data in pooled_terrains.items():
        if data["count"] > 0:
            avg = data["gt_sum"] / data["count"]
            terrain_priors[terrain_key] = avg.tolist()
            terrain_counts[terrain_key] = data["count"]

    return {
        "archetype_priors": archetype_priors,
        "archetype_counts": archetype_counts,
        "terrain_priors": terrain_priors,
        "terrain_counts": terrain_counts,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_report(
    round_info: dict[str, Any],
    seed_scores: list[float | None],
    seed_results: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> None:
    """Print a human-readable analysis report."""
    print()
    print("=" * 60)
    print(f"  ASTAR ISLAND — Round {round_info.get('round_number', '?')} Analysis")
    print("=" * 60)
    print()

    # --- Per-seed scores ---
    print("--- Per-Seed Scores ---")
    valid_scores = []
    for i, score in enumerate(seed_scores):
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"  Seed {i}: {score_str}")
        if score is not None:
            valid_scores.append(score)
    if valid_scores:
        print(f"  Average: {sum(valid_scores) / len(valid_scores):.2f}")
    print()

    # --- Per-seed dynamics ---
    print("--- Per-Seed Dynamics ---")
    for result in seed_results:
        i = result["seed_index"]
        print(
            f"  Seed {i}: {result['n_dynamic']} dynamic cells, "
            f"avg weighted KL = {result['avg_weighted_kl']:.4f}"
        )
    print()

    # --- Per-terrain analysis ---
    print("--- Per-Terrain Analysis ---")
    # Aggregate terrain stats across seeds
    terrain_agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"kl_sum": 0.0, "entropy_sum": 0.0, "count": 0}
    )
    for result in seed_results:
        for terrain_key, stats in result["terrain_stats"].items():
            terrain_agg[terrain_key]["kl_sum"] += stats["kl_sum"]
            terrain_agg[terrain_key]["entropy_sum"] += stats["entropy_sum"]
            terrain_agg[terrain_key]["count"] += stats["count"]

    terrain_rows = []
    for terrain_key, data in terrain_agg.items():
        code = int(terrain_key)
        name = TERRAIN_NAMES.get(code, f"Code {code}")
        avg_kl = data["kl_sum"] / data["count"] if data["count"] > 0 else 0.0
        avg_ent = data["entropy_sum"] / data["count"] if data["count"] > 0 else 0.0
        terrain_rows.append((avg_kl, name, data["count"], avg_ent))

    terrain_rows.sort(key=lambda r: r[0], reverse=True)
    print(f"  {'Terrain':<14} {'Cells':>6} {'Avg KL':>8} {'Avg Entropy':>12}")
    print(f"  {'-' * 14} {'-' * 6} {'-' * 8} {'-' * 12}")
    for avg_kl, name, count, avg_ent in terrain_rows:
        print(f"  {name:<14} {count:>6} {avg_kl:>8.4f} {avg_ent:>12.4f}")
    print()

    # --- Class distribution bias ---
    print("--- Class Distribution Bias (dynamic cells only) ---")
    gt_total = np.zeros(NUM_CLASSES, dtype=np.float64)
    pred_total = np.zeros(NUM_CLASSES, dtype=np.float64)
    n_total = 0

    for result in seed_results:
        for arch_key, stats in result["archetype_stats"].items():
            if stats["entropy_sum"] > 0.01:  # only dynamic
                gt_total += stats["gt_sum"]
                pred_total += stats["pred_sum"]
                n_total += stats["count"]

    if n_total > 0:
        gt_avg = gt_total / n_total
        pred_avg = pred_total / n_total
        print(f"  {'Class':<14} {'Predicted':>10} {'Actual':>10} {'Bias':>10}")
        print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10}")
        for i, name in enumerate(CLASS_NAMES):
            bias = pred_avg[i] - gt_avg[i]
            sign = "+" if bias >= 0 else ""
            print(
                f"  {name:<14} {pred_avg[i]:>10.4f} {gt_avg[i]:>10.4f} "
                f"{sign}{bias:>9.4f}"
            )
    print()

    # --- Top archetype priors (most cells) ---
    print("--- Calibrated Priors (top 15 archetypes by cell count) ---")
    arch_priors = calibration["archetype_priors"]
    arch_with_count: list[tuple[int, str, list[float]]] = []
    for result in seed_results:
        for arch_key, stats in result["archetype_stats"].items():
            if arch_key in arch_priors:
                # Find if already counted
                found = False
                for j, (c, k, _) in enumerate(arch_with_count):
                    if k == arch_key:
                        arch_with_count[j] = (
                            c + stats["count"],
                            k,
                            arch_priors[arch_key],
                        )
                        found = True
                        break
                if not found:
                    arch_with_count.append(
                        (stats["count"], arch_key, arch_priors[arch_key])
                    )

    arch_with_count.sort(key=lambda r: r[0], reverse=True)
    header = f"  {'Archetype':<55} {'N':>5}  " + "  ".join(
        f"{n[:4]:>5}" for n in CLASS_NAMES
    )
    print(header)
    print(f"  {'-' * 55} {'-' * 5}  " + "  ".join(["-" * 5] * 6))
    for count, arch_key, probs in arch_with_count[:15]:
        # Truncate archetype key for display
        display_key = arch_key[:55]
        prob_str = "  ".join(f"{p:>5.3f}" for p in probs)
        print(f"  {display_key:<55} {count:>5}  {prob_str}")
    print()


# ---------------------------------------------------------------------------
# Save / Load calibration
# ---------------------------------------------------------------------------
def save_calibration(
    calibration: dict[str, Any],
    round_ids: list[str],
    total_cells: int,
) -> None:
    """Save calibration to JSON file, merging with any existing data."""
    existing: dict[str, Any] = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            existing = {}

    # Merge round IDs
    prev_rounds = existing.get("metadata", {}).get("rounds_analyzed", [])
    all_rounds = sorted(set(prev_rounds + round_ids))

    new_arch_counts = calibration.get("archetype_counts", {})
    new_terrain_counts = calibration.get("terrain_counts", {})

    merged_arch_priors = dict(calibration["archetype_priors"])
    merged_arch_counts = dict(new_arch_counts)
    merged_terrain_priors = dict(calibration["terrain_priors"])
    merged_terrain_counts = dict(new_terrain_counts)

    if "archetype_priors" in existing:
        old_arch_counts = existing.get("archetype_counts", {})
        for key, new_probs in calibration["archetype_priors"].items():
            if key in existing["archetype_priors"]:
                old_probs = existing["archetype_priors"][key]
                old_n = old_arch_counts.get(key, 1)
                new_n = new_arch_counts.get(key, 1)
                total_n = old_n + new_n
                blended = [
                    (old_n * op + new_n * np_) / total_n
                    for op, np_ in zip(old_probs, new_probs)
                ]
                merged_arch_priors[key] = blended
                merged_arch_counts[key] = total_n

        for key, old_probs in existing["archetype_priors"].items():
            if key not in merged_arch_priors:
                merged_arch_priors[key] = old_probs
                merged_arch_counts[key] = old_arch_counts.get(key, 1)

    if "terrain_priors" in existing:
        old_terrain_counts = existing.get("terrain_counts", {})
        for key, new_probs in calibration["terrain_priors"].items():
            if key in existing["terrain_priors"]:
                old_probs = existing["terrain_priors"][key]
                old_n = old_terrain_counts.get(key, 1)
                new_n = new_terrain_counts.get(key, 1)
                total_n = old_n + new_n
                blended = [
                    (old_n * op + new_n * np_) / total_n
                    for op, np_ in zip(old_probs, new_probs)
                ]
                merged_terrain_priors[key] = blended
                merged_terrain_counts[key] = total_n

        for key, old_probs in existing["terrain_priors"].items():
            if key not in merged_terrain_priors:
                merged_terrain_priors[key] = old_probs
                merged_terrain_counts[key] = old_terrain_counts.get(key, 1)

    output = {
        "metadata": {
            "rounds_analyzed": all_rounds,
            "total_cells": total_cells
            + existing.get("metadata", {}).get("total_cells", 0),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "archetype_priors": merged_arch_priors,
        "archetype_counts": merged_arch_counts,
        "terrain_priors": merged_terrain_priors,
        "terrain_counts": merged_terrain_counts,
    }

    CALIBRATION_FILE.write_text(json.dumps(output, indent=2))
    log(f"Saved calibration to {CALIBRATION_FILE}")
    log(
        f"  {len(output['archetype_priors'])} archetype priors, "
        f"{len(output['terrain_priors'])} terrain priors, "
        f"{len(all_rounds)} round(s) analyzed"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze_round(client: AstarClient, round_id: str) -> dict[str, Any] | None:
    """Analyze a single round. Returns calibration data or None on error."""
    log(f"Fetching round detail for {round_id}...")
    try:
        detail = client.get_round_detail(round_id)
    except Exception as e:
        log(f"  ERROR fetching round detail: {e}")
        return None

    seeds_count = detail.seeds_count
    height = detail.map_height
    width = detail.map_width

    log(f"  Round {detail.round_number}: {width}x{height}, {seeds_count} seeds")

    # Build seed analyses from initial states
    seed_analyses: list[SeedAnalysis] = []
    for state in detail.initial_states:
        seed_analyses.append(SeedAnalysis(state["grid"], state["settlements"]))

    # Fetch ground truth and predictions for each seed
    seed_scores: list[float | None] = []
    seed_results: list[dict[str, Any]] = []

    for seed_idx in range(seeds_count):
        log(f"  Fetching analysis for seed {seed_idx}...")
        try:
            data = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            log(f"    ERROR: {e}")
            seed_scores.append(None)
            continue

        gt = np.array(data["ground_truth"], dtype=np.float64)
        pred = (
            np.array(data["prediction"], dtype=np.float64)
            if data.get("prediction")
            else None
        )
        score = data.get("score")
        seed_scores.append(score)

        log(f"    Score: {score}, GT shape: {gt.shape}")

        result = analyze_seed(
            seed_index=seed_idx,
            ground_truth=gt,
            prediction=pred,
            initial_grid=data.get("initial_grid"),
            seed_analysis=seed_analyses[seed_idx],
        )
        seed_results.append(result)

    if not seed_results:
        log("  No seed results available.")
        return None

    # Generate calibrated priors
    calibration = generate_calibrated_priors(seed_results)

    # Get round info for report
    round_info = {"round_number": detail.round_number, "round_id": round_id}

    # Print report
    print_report(round_info, seed_scores, seed_results, calibration)

    # Compute total cells analyzed
    total_cells = sum(r["n_cells"] for r in seed_results)

    return {
        "round_id": round_id,
        "calibration": calibration,
        "total_cells": total_cells,
        "seed_scores": seed_scores,
    }


def main() -> None:
    client = AstarClient()

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Analyze all completed rounds
        log("Fetching all rounds...")
        try:
            my_rounds = client.get_my_rounds()
        except Exception as e:
            log(f"ERROR: {e}")
            sys.exit(1)

        completed = [r for r in my_rounds if r["status"] == "completed"]
        if not completed:
            log("No completed rounds found.")
            sys.exit(0)

        log(f"Found {len(completed)} completed round(s)")
        all_calibrations = []
        all_round_ids = []
        total_cells = 0

        for r in completed:
            result = analyze_round(client, r["id"])
            if result:
                all_calibrations.append(result["calibration"])
                all_round_ids.append(result["round_id"])
                total_cells += result["total_cells"]

        if all_calibrations:
            # Merge all calibrations
            merged = generate_calibrated_priors_from_multiple(all_calibrations)
            save_calibration(merged, all_round_ids, total_cells)

    else:
        # Analyze specific round or most recent completed
        if len(sys.argv) > 1:
            round_id = sys.argv[1]
        else:
            log("Finding most recent completed/scoring round...")
            try:
                my_rounds = client.get_my_rounds()
            except Exception as e:
                log(f"ERROR: {e}")
                sys.exit(1)

            # Prefer scoring, then completed
            candidates = [
                r for r in my_rounds if r["status"] in ("scoring", "completed")
            ]
            if not candidates:
                log("No completed or scoring rounds found.")
                log("Available rounds:")
                for r in my_rounds:
                    log(f"  Round {r['round_number']}: {r['status']}")
                sys.exit(0)

            # Pick most recent
            candidates.sort(key=lambda r: r.get("round_number", 0), reverse=True)
            round_id = candidates[0]["id"]
            log(f"  Selected round {candidates[0]['round_number']} ({round_id})")

        result = analyze_round(client, round_id)
        if result:
            save_calibration(
                result["calibration"],
                [result["round_id"]],
                result["total_cells"],
            )


def generate_calibrated_priors_from_multiple(
    calibrations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge calibrated priors from multiple rounds, weighted by per-archetype cell counts."""
    merged_arch_weighted: dict[str, NDArray] = defaultdict(
        lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
    )
    merged_arch_count: dict[str, int] = defaultdict(int)
    merged_terrain_weighted: dict[str, NDArray] = defaultdict(
        lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
    )
    merged_terrain_count: dict[str, int] = defaultdict(int)

    for cal in calibrations:
        arch_counts = cal.get("archetype_counts", {})
        for key, probs in cal.get("archetype_priors", {}).items():
            n = arch_counts.get(key, 1)
            merged_arch_weighted[key] += np.array(probs) * n
            merged_arch_count[key] += n

        terrain_counts = cal.get("terrain_counts", {})
        for key, probs in cal.get("terrain_priors", {}).items():
            n = terrain_counts.get(key, 1)
            merged_terrain_weighted[key] += np.array(probs) * n
            merged_terrain_count[key] += n

    result_arch = {}
    result_arch_counts = {}
    for key, weighted_sum in merged_arch_weighted.items():
        total_n = merged_arch_count[key]
        result_arch[key] = (weighted_sum / total_n).tolist()
        result_arch_counts[key] = total_n

    result_terrain = {}
    result_terrain_counts = {}
    for key, weighted_sum in merged_terrain_weighted.items():
        total_n = merged_terrain_count[key]
        result_terrain[key] = (weighted_sum / total_n).tolist()
        result_terrain_counts[key] = total_n

    return {
        "archetype_priors": result_arch,
        "archetype_counts": result_arch_counts,
        "terrain_priors": result_terrain,
        "terrain_counts": result_terrain_counts,
    }


if __name__ == "__main__":
    main()
