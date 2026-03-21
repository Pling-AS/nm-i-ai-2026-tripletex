"""Cell archetype classification, priority masks, and class masks.

Analyses the FREE initial state to categorize cells before spending any queries.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from utils import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    NUM_CLASSES,
    TERRAIN_FOREST,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
    TERRAIN_PLAINS,
    TERRAIN_RUIN,
    TERRAIN_SETTLEMENT,
    TERRAIN_PORT,
    is_static,
)


# ---------------------------------------------------------------------------
# Archetype definition
# ---------------------------------------------------------------------------
class CellArchetype(NamedTuple):
    """Hashable archetype key for pooling observations across seeds."""

    initial_terrain: int
    is_coastal: bool
    dist_settlement_bucket: int  # 0-7 fine buckets, 8=far(8+)
    has_adjacent_settlement: bool
    pressure_bucket: int  # 0=none, 1=low, 2=med, 3=high settlement pressure


def _distance_bucket(dist: float) -> int:
    """Fine-grained distance bucket: 0=on, 1-7=exact distance, 8=far."""
    if dist <= 0:
        return 0
    d = int(dist + 0.5)  # round to nearest int
    if d <= 7:
        return d
    return 8


def _pressure_bucket(pressure: float) -> int:
    """Bucket settlement expansion pressure into 4 levels."""
    if pressure < 0.3:
        return 0  # no pressure
    if pressure < 1.0:
        return 1  # low
    if pressure < 2.5:
        return 2  # medium
    return 3  # high


# ---------------------------------------------------------------------------
# Settlement position helpers
# ---------------------------------------------------------------------------
def _settlement_positions(settlements: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """Extract (x, y) set from settlement list."""
    return {(s["x"], s["y"]) for s in settlements}


def _ruin_positions(grid: NDArray[np.int_]) -> set[tuple[int, int]]:
    """Find all ruin cells in the initial grid."""
    ys, xs = np.where(grid == TERRAIN_RUIN)
    return set(zip(xs.tolist(), ys.tolist()))


# ---------------------------------------------------------------------------
# Distance maps
# ---------------------------------------------------------------------------
def _compute_distance_to_set(
    height: int, width: int, positions: set[tuple[int, int]]
) -> NDArray[np.floating]:
    """Compute Chebyshev distance from each cell to the nearest position in the set.

    Returns (H, W) float array.  If positions is empty, returns all inf.
    """
    dist = np.full((height, width), np.inf)
    if not positions:
        return dist
    for px, py in positions:
        # Chebyshev (king-move) distance
        yy, xx = np.ogrid[0:height, 0:width]
        d = np.maximum(np.abs(xx - px), np.abs(yy - py))
        dist = np.minimum(dist, d)
    return dist


# ---------------------------------------------------------------------------
# Class masks — which prediction classes are possible per cell?
# ---------------------------------------------------------------------------
def build_class_masks(
    initial_grid: NDArray[np.int_],
) -> NDArray[np.bool_]:
    """Build a (H, W, 6) boolean mask of allowed prediction classes per cell.

    Rules:
    - Ocean cells can only be class 0 (Empty)  — ocean never changes.
    - Mountain cells can only be class 5        — mountains never change.
    - Plains cells can become anything except Mountain: classes 0,1,2,3,4.
    - Forest cells: mostly stay forest, but can become ruin, settlement,
      port (if reclaimed), or empty.  All classes except Mountain.
    - Settlement cells: can stay settlement, become port, collapse to ruin,
      be reclaimed to empty/forest.  All except Mountain.
    - Port cells: same as settlement.
    - Ruin cells: can be reclaimed (settlement/port), overgrown (forest),
      fade (empty), or stay ruin.  All except Mountain.
    """
    h, w = initial_grid.shape
    masks = np.ones((h, w, NUM_CLASSES), dtype=bool)

    # Ocean: only class 0
    ocean = initial_grid == TERRAIN_OCEAN
    masks[ocean] = False
    masks[ocean, CLASS_EMPTY] = True

    # Mountain: only class 5
    mountain = initial_grid == TERRAIN_MOUNTAIN
    masks[mountain] = False
    masks[mountain, CLASS_MOUNTAIN] = True

    # Everything else: all classes except Mountain
    dynamic = ~ocean & ~mountain
    masks[dynamic, CLASS_MOUNTAIN] = False

    return masks


def build_port_mask(
    grid: NDArray[np.int_], coastal_mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Build mask of cells where PORT is a valid class.

    Rule: Ports must be coastal OR already be a port.
    """
    is_port = grid == TERRAIN_PORT
    # Allow port if it's coastal or already a port
    return coastal_mask | is_port


# ---------------------------------------------------------------------------
# Priority masks — which cells are worth observing?
# ---------------------------------------------------------------------------
class SeedAnalysis:
    """Pre-computed analysis of one seed's initial state."""

    def __init__(
        self,
        grid: list[list[int]],
        settlements: list[dict[str, Any]],
    ) -> None:
        self.grid = np.array(grid, dtype=np.int_)
        self.height, self.width = self.grid.shape
        self.settlements = settlements
        self.settlement_pos = _settlement_positions(settlements)
        self.ruin_pos = _ruin_positions(self.grid)

        # Distance maps
        self.dist_to_settlement = _compute_distance_to_set(
            self.height, self.width, self.settlement_pos
        )
        self.dist_to_ruin = _compute_distance_to_set(
            self.height, self.width, self.ruin_pos
        )

        # Coastal mask (adjacent to ocean)
        self.coastal_mask = self._compute_coastal_mask()

        # Port mask (where ports are valid)
        self.port_mask = build_port_mask(self.grid, self.coastal_mask)

        # Class masks
        self.class_masks = build_class_masks(self.grid)
        # Enforce: PORT class only allowed where port_mask is True
        self.class_masks[~self.port_mask, CLASS_PORT] = False

        # Archetypes
        self.archetypes = self._compute_archetypes()

        # Priority mask
        self.priority_mask = self._compute_priority_mask()

    def _compute_coastal_mask(self) -> NDArray[np.bool_]:
        """True for cells adjacent (8-connected) to ocean."""
        grid = self.grid
        h, w = grid.shape
        is_ocean = grid == TERRAIN_OCEAN

        padded_ocean = np.pad(
            is_ocean, ((1, 1), (1, 1)), mode="constant", constant_values=False
        )

        coastal = np.zeros((h, w), dtype=bool)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded_ocean[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
                coastal |= shifted

        is_land = ~is_ocean & (grid != TERRAIN_MOUNTAIN)
        return coastal & is_land

    def _compute_priority_mask(self) -> NDArray[np.bool_]:
        """Cells worth observing (non-static, potentially dynamic).

        HIGH priority: near settlements, coastal, ruins, plains
        LOW priority: deep forest interiors far from everything
        """
        is_ocean = self.grid == TERRAIN_OCEAN
        is_mountain = self.grid == TERRAIN_MOUNTAIN

        # Static cells are never priority
        priority = ~is_ocean & ~is_mountain

        # Optionally, we could downweight deep forest interiors,
        # but coverage-first strategy means we observe everything anyway.
        # Keep all non-static cells as priority for now.
        return priority

    def _compute_archetypes(self) -> NDArray[np.object_]:
        """Compute CellArchetype for every cell.  Returns (H, W) object array."""
        h, w = self.height, self.width
        archetypes = np.empty((h, w), dtype=object)

        has_adj_settlement = self.dist_to_settlement <= 1.5

        # Compute expansion pressure field: sum of exp(-dist/3) over all settlements
        pressure_map = np.zeros((h, w), dtype=np.float64)
        for s in self.settlements:
            sx, sy = s["x"], s["y"]
            yy, xx = np.ogrid[0:h, 0:w]
            d = np.maximum(np.abs(xx - sx), np.abs(yy - sy)).astype(np.float64)
            pressure_map += np.exp(-d / 3.0)

        self.pressure_map = pressure_map  # save for later use

        for y in range(h):
            for x in range(w):
                archetypes[y, x] = CellArchetype(
                    initial_terrain=int(self.grid[y, x]),
                    is_coastal=bool(self.coastal_mask[y, x]),
                    dist_settlement_bucket=_distance_bucket(
                        self.dist_to_settlement[y, x]
                    ),
                    has_adjacent_settlement=bool(has_adj_settlement[y, x]),
                    pressure_bucket=_pressure_bucket(pressure_map[y, x]),
                )
        return archetypes

    def get_archetype(self, y: int, x: int) -> CellArchetype:
        """Return the archetype for cell (y, x)."""
        return self.archetypes[y, x]
