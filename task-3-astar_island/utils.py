"""Shared constants, types, and utility functions for Astar Island."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Terrain codes (internal simulator values present in grids)
# ---------------------------------------------------------------------------
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11

# ---------------------------------------------------------------------------
# Prediction class indices (what we submit)
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
CLASS_EMPTY = 0  # Ocean, Plains, Empty -> all map to class 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

# Map internal terrain codes to prediction class indices
TERRAIN_TO_CLASS: dict[int, int] = {
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
}

# ---------------------------------------------------------------------------
# Static terrain detection
# ---------------------------------------------------------------------------
STATIC_TERRAINS = {TERRAIN_OCEAN, TERRAIN_MOUNTAIN}


def is_static(terrain_code: int) -> bool:
    """Return True if the terrain never changes during simulation."""
    return terrain_code in STATIC_TERRAINS


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------
FLOOR_EPS = 0.001


def apply_floor_and_normalize(
    probs: NDArray[np.floating],
    class_mask: NDArray[np.bool_],
    epsilon: float = FLOOR_EPS,
) -> NDArray[np.floating]:
    """Floor allowed classes at epsilon, then renormalize.

    The additive floor acts as regularization against prior miscalibration:
    classes with exact-zero priors get a minimum mass, hedging against
    ground-truth surprises. Optimal eps found by sweep: 0.001.
    """
    out = probs.copy()
    out[~class_mask] = 0.0
    out[class_mask] = np.maximum(out[class_mask], epsilon)
    total = out.sum()
    if total > 0:
        out /= total
    else:
        n_allowed = class_mask.sum()
        out[class_mask] = 1.0 / max(n_allowed, 1)
    return out


def apply_floor_and_normalize_grid(
    prediction: NDArray[np.floating],
    class_masks: NDArray[np.bool_],
    epsilon: float = FLOOR_EPS,
) -> NDArray[np.floating]:
    """Vectorized: floor allowed classes at epsilon, then renormalize."""
    out = prediction.copy()
    out[~class_masks] = 0.0
    out = np.where(class_masks, np.maximum(out, epsilon), 0.0)
    totals = out.sum(axis=-1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    out /= totals
    return out


def grid_to_classes(grid: list[list[int]]) -> NDArray[np.int8]:
    """Convert a 2D grid of internal terrain codes to prediction class indices.

    Parameters
    ----------
    grid : H×W list of lists with internal terrain codes.

    Returns
    -------
    (H, W) numpy array of class indices (0-5).
    """
    arr = np.array(grid, dtype=np.int8)
    result = np.zeros_like(arr)
    for terrain_code, class_idx in TERRAIN_TO_CLASS.items():
        result[arr == terrain_code] = class_idx
    return result
