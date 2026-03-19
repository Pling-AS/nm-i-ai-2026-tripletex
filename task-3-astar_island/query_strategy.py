"""Query allocation strategy: coverage-first with adaptive repeats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from features import SeedAnalysis
from observation_store import ObservationStore


@dataclass
class QueryPlan:
    """A planned viewport query."""

    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int


# ---------------------------------------------------------------------------
# Tiling: cover a map with 15×15 viewports
# ---------------------------------------------------------------------------
def generate_tiling(
    map_w: int, map_h: int, vp_size: int = 15
) -> list[tuple[int, int, int, int]]:
    """Generate viewport positions that tile the full map.

    Uses overlapping placement to ensure complete coverage.
    For a 40-wide map with vp_size=15:
      positions: 0, 13, 25  (covers 0-14, 13-27, 25-39)
      Overlap bands at cols 13-14 and 25-27 give bonus repeat observations.

    Returns list of (x, y, w, h) tuples.
    """

    def _positions(length: int, size: int) -> list[int]:
        if length <= size:
            return [0]
        positions = [0]
        # Place subsequent viewports to cover remaining space
        pos = 0
        while pos + size < length:
            # Next position: start where we'd leave at most `size` remaining
            remaining = length - (pos + size)
            if remaining <= 0:
                break
            # Step forward, ensuring we don't exceed map bounds
            step = min(size, remaining + size)
            next_pos = min(pos + size - 2, length - size)  # -2 for overlap
            if next_pos <= pos:
                next_pos = pos + 1
            pos = next_pos
            positions.append(pos)
        return sorted(set(positions))

    xs = _positions(map_w, vp_size)
    ys = _positions(map_h, vp_size)

    tiles = []
    for y in ys:
        for x in xs:
            w = min(vp_size, map_w - x)
            h = min(vp_size, map_h - y)
            tiles.append((x, y, w, h))
    return tiles


# ---------------------------------------------------------------------------
# Phase 1: Coverage queries (interleaved across seeds)
# ---------------------------------------------------------------------------
def plan_coverage_queries(
    seeds_count: int,
    map_w: int,
    map_h: int,
    max_budget: int = 50,
) -> list[QueryPlan]:
    """Plan coverage queries interleaved across all seeds.

    Interleaving ensures that if we run out of budget, we have partial
    coverage of ALL seeds rather than full coverage of some and zero of others.
    """
    tiles = generate_tiling(map_w, map_h)
    tiles_per_seed = len(tiles)  # typically 9 for 40×40

    queries: list[QueryPlan] = []

    # Interleave: for each tile index, cycle through all seeds
    for tile_idx in range(tiles_per_seed):
        for seed_idx in range(seeds_count):
            if len(queries) >= max_budget:
                break
            x, y, w, h = tiles[tile_idx]
            queries.append(
                QueryPlan(
                    seed_index=seed_idx,
                    viewport_x=x,
                    viewport_y=y,
                    viewport_w=w,
                    viewport_h=h,
                )
            )
        if len(queries) >= max_budget:
            break

    return queries


# ---------------------------------------------------------------------------
# Phase 2: Adaptive repeat queries
# ---------------------------------------------------------------------------
def plan_repeat_queries(
    observation_store: ObservationStore,
    seed_analyses: list[SeedAnalysis],
    remaining_budget: int,
    map_w: int,
    map_h: int,
) -> list[QueryPlan]:
    """Select repeat queries targeting highest-uncertainty dynamic regions.

    Strategy: for each possible viewport position, compute the sum of
    posterior entropy over dynamic (priority) cells.  Pick the viewports
    with the highest total uncertainty.
    """
    if remaining_budget <= 0:
        return []

    tiles = generate_tiling(map_w, map_h)
    candidates: list[tuple[float, int, int, int, int, int]] = []

    for seed_idx, analysis in enumerate(seed_analyses):
        entropy_grid = observation_store.compute_entropy_grid(seed_idx)

        for x, y, w, h in tiles:
            # Sum entropy only over priority (dynamic) cells in this viewport
            vp_entropy = 0.0
            for ry in range(h):
                for rx in range(w):
                    ay, ax = y + ry, x + rx
                    if ay < map_h and ax < map_w and analysis.priority_mask[ay, ax]:
                        vp_entropy += entropy_grid[ay, ax]

            candidates.append((vp_entropy, seed_idx, x, y, w, h))

    # Sort by descending entropy
    candidates.sort(key=lambda c: c[0], reverse=True)

    queries: list[QueryPlan] = []
    for _, seed_idx, x, y, w, h in candidates[:remaining_budget]:
        queries.append(
            QueryPlan(
                seed_index=seed_idx,
                viewport_x=x,
                viewport_y=y,
                viewport_w=w,
                viewport_h=h,
            )
        )

    return queries
