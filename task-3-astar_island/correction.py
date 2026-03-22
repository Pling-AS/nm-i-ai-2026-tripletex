"""Post-Dirichlet spatial correction layer.

Learns a residual correction table from historical ground truth and applies
small, capped per-cell adjustments AFTER spatial smoothing and BEFORE submission.

The correction is LOCAL: each cell's adjustment depends only on its own
features (terrain, distance-to-settlement, coastal, nearby settlements,
expansion pressure).  Two-level hierarchical shrinkage prevents overfitting
in sparse buckets.

Usage:
  # At module load (predictor.py):
  from correction import load_correction_table, apply_correction
  _correction_table = load_correction_table()

  # After spatial smoothing:
  prediction = apply_correction(prediction, seed_analysis, _correction_table)

  # Training (run once after new ground truth):
  uv run train_correction.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from utils import (
    NUM_CLASSES,
    CLASS_EMPTY,
    CLASS_SETTLEMENT,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    TERRAIN_OCEAN,
    TERRAIN_MOUNTAIN,
    FLOOR_EPS,
)

CORRECTION_FILE = Path(__file__).with_name("correction_table.json")

# ---------------------------------------------------------------------------
# Shrinkage hyperparameters
# ---------------------------------------------------------------------------
L1_PARENT_STRENGTH = 300.0
L2_PARENT_STRENGTH = 80.0
MIN_CHILD_N = 25

# ---------------------------------------------------------------------------
# Safety caps: per-class maximum absolute delta
# ---------------------------------------------------------------------------
CLASS_CAPS = np.array([
    0.05,   # CLASS_EMPTY
    0.08,   # CLASS_SETTLEMENT  (biggest gap to fix)
    0.04,   # CLASS_PORT
    0.05,   # CLASS_RUIN
    0.05,   # CLASS_FOREST
    0.00,   # CLASS_MOUNTAIN    (never adjust static)
], dtype=np.float64)

MAX_L1_SHIFT = 0.10


# ---------------------------------------------------------------------------
# Feature bucketing
# ---------------------------------------------------------------------------
def _dist_bin(d: float) -> int:
    if not np.isfinite(d) or d >= 5:
        return 5
    return int(d)


def compute_correction_features(
    seed_analysis: Any,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute per-cell feature arrays for correction bucketing.
    Returns (terrain, coastal, dist_bin, nearby_bin, pressure_bin), all (H, W).
    """
    h, w = seed_analysis.height, seed_analysis.width
    yy, xx = np.mgrid[0:h, 0:w]

    terrain = seed_analysis.grid.astype(np.int16)
    coastal = seed_analysis.coastal_mask.astype(np.int8)

    dist_bin = np.zeros((h, w), dtype=np.int8)
    for y in range(h):
        for x in range(w):
            dist_bin[y, x] = _dist_bin(float(seed_analysis.dist_to_settlement[y, x]))

    nearby_count = np.zeros((h, w), dtype=np.int16)
    pressure = np.zeros((h, w), dtype=np.float32)

    for sx, sy in seed_analysis.settlement_pos:
        d = np.maximum(np.abs(xx - sx), np.abs(yy - sy)).astype(np.float32)
        nearby_count += (d <= 4).astype(np.int16)
        pressure += (
            (d == 1).astype(np.float32) * 1.00
            + (d == 2).astype(np.float32) * 0.75
            + (d == 3).astype(np.float32) * 0.50
            + (d == 4).astype(np.float32) * 0.25
        )

    nearby_bin = np.clip(nearby_count, 0, 2).astype(np.int8)

    p_bin = np.zeros((h, w), dtype=np.int8)
    p_bin[pressure > 0.0] = 1
    p_bin[pressure >= 0.75] = 2
    p_bin[pressure >= 1.5] = 3

    return terrain, coastal, dist_bin, nearby_bin, p_bin


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------
def _parent_key(terrain: int, dist_bin: int, coastal: int) -> str:
    return f"{terrain}|{dist_bin}|{coastal}"


def _full_key(
    terrain: int, dist_bin: int, coastal: int,
    nearby_bin: int, pressure_bin: int,
) -> str:
    return f"{terrain}|{dist_bin}|{coastal}|{nearby_bin}|{pressure_bin}"


# ---------------------------------------------------------------------------
# Safety: cap a delta vector
# ---------------------------------------------------------------------------
def _cap_delta(delta: NDArray[np.float64]) -> NDArray[np.float64]:
    d = delta.copy()
    d[CLASS_MOUNTAIN] = 0.0
    d = np.clip(d, -CLASS_CAPS, CLASS_CAPS)
    l1 = np.abs(d).sum()
    if l1 > MAX_L1_SHIFT:
        d *= MAX_L1_SHIFT / l1
    return d


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_correction_table(
    training_rows: list[tuple[NDArray, NDArray, Any]],
) -> dict[str, Any]:
    """Build a 2-level correction table from historical data.

    training_rows: list of (prediction, ground_truth, seed_analysis)
      prediction:    (H,W,6) after spatial smooth (current pipeline output)
      ground_truth:  (H,W,6) from API
      seed_analysis: SeedAnalysis for this seed
    """
    global_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    global_n = 0

    parent_sum: dict[str, NDArray] = defaultdict(
        lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
    )
    parent_n: dict[str, int] = defaultdict(int)

    full_sum: dict[str, NDArray] = defaultdict(
        lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
    )
    full_n: dict[str, int] = defaultdict(int)

    for prediction, ground_truth, seed_analysis in training_rows:
        terrain, coastal, dist_bin, nearby_bin, pressure_bin = (
            compute_correction_features(seed_analysis)
        )
        h, w = terrain.shape
        for y in range(h):
            for x in range(w):
                t = int(terrain[y, x])
                if t in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                    continue

                resid = (
                    ground_truth[y, x].astype(np.float64)
                    - prediction[y, x].astype(np.float64)
                )

                pk = _parent_key(t, int(dist_bin[y, x]), int(coastal[y, x]))
                fk = _full_key(
                    t, int(dist_bin[y, x]), int(coastal[y, x]),
                    int(nearby_bin[y, x]), int(pressure_bin[y, x]),
                )

                global_sum += resid
                global_n += 1
                parent_sum[pk] += resid
                parent_n[pk] += 1
                full_sum[fk] += resid
                full_n[fk] += 1

    global_resid = global_sum / max(global_n, 1)

    # L1: parent buckets shrink toward global
    parent_table: dict[str, dict] = {}
    for k, s in parent_sum.items():
        n = parent_n[k]
        mean_resid = s / n
        shrunk = (n * mean_resid + L1_PARENT_STRENGTH * global_resid) / (
            n + L1_PARENT_STRENGTH
        )
        parent_table[k] = {"n": n, "residual": _cap_delta(shrunk).tolist()}

    # L2: full buckets shrink toward their parent
    full_table: dict[str, dict] = {}
    for k, s in full_sum.items():
        n = full_n[k]
        parts = k.split("|")
        pk = _parent_key(int(parts[0]), int(parts[1]), int(parts[2]))
        parent_resid = np.array(parent_table[pk]["residual"], dtype=np.float64)

        mean_resid = s / n
        shrunk = (n * mean_resid + L2_PARENT_STRENGTH * parent_resid) / (
            n + L2_PARENT_STRENGTH
        )
        full_table[k] = {"n": n, "residual": _cap_delta(shrunk).tolist()}

    return {
        "global": {"n": global_n, "residual": _cap_delta(global_resid).tolist()},
        "parent": parent_table,
        "full": full_table,
        "meta": {
            "l1_parent_strength": L1_PARENT_STRENGTH,
            "l2_parent_strength": L2_PARENT_STRENGTH,
            "min_child_n": MIN_CHILD_N,
            "class_caps": CLASS_CAPS.tolist(),
            "max_l1_shift": MAX_L1_SHIFT,
            "n_parent_buckets": len(parent_table),
            "n_full_buckets": len(full_table),
        },
    }


def save_correction_table(table: dict[str, Any]) -> None:
    CORRECTION_FILE.write_text(json.dumps(table, indent=2))
    n_p = len(table.get("parent", {}))
    n_f = len(table.get("full", {}))
    n_g = table.get("global", {}).get("n", 0)
    print(
        f"[correction] Saved correction_table.json: "
        f"{n_p} parent, {n_f} full buckets, {n_g} total cells"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_correction_table() -> dict[str, Any] | None:
    if not CORRECTION_FILE.exists():
        print("[correction] No correction_table.json found -- correction disabled")
        return None
    try:
        table = json.loads(CORRECTION_FILE.read_text())
        n_p = len(table.get("parent", {}))
        n_f = len(table.get("full", {}))
        print(f"[correction] Loaded correction table: {n_p} parent, {n_f} full buckets")
        return table
    except (json.JSONDecodeError, OSError) as e:
        print(f"[correction] Failed to load correction_table.json: {e}")
        return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def apply_correction(
    prediction: NDArray[np.floating],
    seed_analysis: Any,
    table: dict[str, Any] | None,
) -> NDArray[np.floating]:
    """Apply learned correction to a post-smoothing prediction tensor.

    prediction: (H,W,6) after spatial smoothing, before submission
    seed_analysis: SeedAnalysis for this seed
    table: loaded correction table (or None to skip)
    """
    if table is None:
        return prediction

    corrected = prediction.copy()
    terrain, coastal, dist_bin, nearby_bin, pressure_bin = (
        compute_correction_features(seed_analysis)
    )

    parent_table = table.get("parent", {})
    full_table = table.get("full", {})
    global_resid = np.array(
        table.get("global", {}).get("residual", [0.0] * NUM_CLASSES),
        dtype=np.float64,
    )
    min_n = table.get("meta", {}).get("min_child_n", MIN_CHILD_N)
    class_masks = seed_analysis.class_masks
    h, w = terrain.shape

    n_corrected = 0
    for y in range(h):
        for x in range(w):
            t = int(terrain[y, x])
            if t in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN):
                continue

            pk = _parent_key(t, int(dist_bin[y, x]), int(coastal[y, x]))
            fk = _full_key(
                t, int(dist_bin[y, x]), int(coastal[y, x]),
                int(nearby_bin[y, x]), int(pressure_bin[y, x]),
            )

            if fk in full_table and full_table[fk]["n"] >= min_n:
                delta = np.array(full_table[fk]["residual"], dtype=np.float64)
            elif pk in parent_table:
                delta = np.array(parent_table[pk]["residual"], dtype=np.float64)
            else:
                delta = global_resid.copy()

            allowed = class_masks[y, x]
            delta = np.where(allowed, delta, 0.0)
            delta = _cap_delta(delta)

            if np.abs(delta).max() < 1e-5:
                continue

            q = corrected[y, x] + delta
            q[~allowed] = 0.0
            q[allowed] = np.maximum(q[allowed], FLOOR_EPS)
            total = q.sum()
            if total > 0:
                q /= total
            corrected[y, x] = q
            n_corrected += 1

    print(f"[correction] Applied correction to {n_corrected} cells")
    return corrected
