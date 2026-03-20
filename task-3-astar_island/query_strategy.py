"""Query allocation strategy: coverage-first with entropy-aware overlap."""

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
# Tiling: cover a map with 15×15 viewports, entropy-aware overlap
# ---------------------------------------------------------------------------
def _find_best_positions(
    length: int,
    vp_size: int,
    entropy_profile: NDArray[np.floating] | None = None,
) -> list[int]:
    """Find 3 viewport positions along one axis that cover `length` cells.

    For length=40, vp_size=15, we need 3 viewports.
    Total coverage = 3*15 = 45, so 5 cells get double-covered.
    The overlap distribution is determined by where we place viewports.

    If entropy_profile is given (shape (length,)), place overlap on
    the highest-entropy corridor. Otherwise use even spacing.
    """
    if length <= vp_size:
        return [0]

    n_viewports = -(-length // vp_size)  # ceil division
    if n_viewports <= 1:
        return [0]

    if n_viewports == 2:
        return [0, length - vp_size]

    # For 3 viewports covering 40 with vp=15:
    # pos[0] = 0 (always start at 0)
    # pos[2] = 25 (always end at length - vp_size)
    # pos[1] = ? (determines where overlap goes)
    #
    # pos[1] must satisfy: pos[1] < pos[0]+vp_size (overlap with first)
    #                  and pos[1]+vp_size > pos[2] (overlap with third)
    # So: pos[2]-vp_size < pos[1] < pos[0]+vp_size
    # For 40/15: 10 < pos[1] < 15
    # Valid range: [11, 14] → overlap shifts accordingly

    last_pos = length - vp_size
    min_mid = last_pos - vp_size + 1  # must overlap with last viewport
    max_mid = vp_size - 1  # must overlap with first viewport

    if min_mid > max_mid:
        # Can't do 3 viewports with overlap — just space evenly
        step = (length - vp_size) / (n_viewports - 1)
        return [round(i * step) for i in range(n_viewports)]

    if entropy_profile is None or len(entropy_profile) != length:
        # Default: center the middle viewport
        mid = (min_mid + max_mid) // 2
        return [0, mid, last_pos]

    # Entropy-aware: find the position that maximizes overlap entropy.
    # For each candidate mid position, compute the total entropy in
    # the overlap bands.
    best_mid = min_mid
    best_score = -1.0

    for mid in range(min_mid, max_mid + 1):
        # Overlap with first viewport: cells [mid, min(vp_size, mid+vp_size)-1]
        overlap1_start = mid
        overlap1_end = min(vp_size, mid + vp_size)
        # Overlap with last viewport: cells [last_pos, mid+vp_size-1]
        overlap2_start = last_pos
        overlap2_end = min(mid + vp_size, length)

        score = float(entropy_profile[overlap1_start:overlap1_end].sum())
        score += float(entropy_profile[overlap2_start:overlap2_end].sum())

        if score > best_score:
            best_score = score
            best_mid = mid

    return [0, best_mid, last_pos]


def generate_tiling(
    map_w: int,
    map_h: int,
    vp_size: int = 15,
    entropy_map: NDArray[np.floating] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Generate viewport positions that tile the full map.

    If entropy_map (H, W) is provided, places overlap bands on
    highest-entropy corridors. Otherwise uses even spacing.

    Returns list of (x, y, w, h) tuples.
    """
    if entropy_map is not None:
        # Compute 1D entropy profiles by summing along each axis
        x_profile = entropy_map.sum(axis=0)  # shape (W,)
        y_profile = entropy_map.sum(axis=1)  # shape (H,)
    else:
        x_profile = None
        y_profile = None

    xs = _find_best_positions(map_w, vp_size, x_profile)
    ys = _find_best_positions(map_h, vp_size, y_profile)

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
def _compute_prior_entropy_map(analysis: SeedAnalysis) -> NDArray[np.floating]:
    """Estimate per-cell entropy from the initial state (no queries needed).

    High entropy = near settlements, mixed terrain, coastal development zones.
    Low entropy = deep ocean, mountains, isolated forest interiors.
    """
    from solution_spatial import _get_calibrated_prior, _initial_terrain_prior

    h, w = analysis.height, analysis.width
    entropy = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            terrain = int(analysis.grid[y, x])
            if terrain in (10, 5):  # ocean, mountain — static
                continue

            archetype = analysis.get_archetype(y, x)
            prior = _get_calibrated_prior(archetype, terrain)
            if prior is None:
                prior = _initial_terrain_prior(terrain)

            # Shannon entropy of the prior
            nonzero = prior > 0
            if nonzero.any():
                ent = -np.sum(prior[nonzero] * np.log(prior[nonzero]))
                entropy[y, x] = ent

    return entropy


def plan_coverage_queries(
    seeds_count: int,
    map_w: int,
    map_h: int,
    max_budget: int = 50,
    seed_analyses: list[SeedAnalysis] | None = None,
) -> list[QueryPlan]:
    """Plan coverage queries interleaved across all seeds.

    Interleaving ensures that if we run out of budget, we have partial
    coverage of ALL seeds rather than full coverage of some and zero of others.

    If seed_analyses are provided, uses entropy-aware overlap placement.
    """
    # Generate per-seed tilings (possibly entropy-aware)
    per_seed_tiles: list[list[tuple[int, int, int, int]]] = []
    for seed_idx in range(seeds_count):
        if seed_analyses is not None:
            entropy_map = _compute_prior_entropy_map(seed_analyses[seed_idx])
            tiles = generate_tiling(map_w, map_h, entropy_map=entropy_map)
        else:
            tiles = generate_tiling(map_w, map_h)
        per_seed_tiles.append(tiles)

    tiles_per_seed = len(per_seed_tiles[0])  # typically 9 for 40×40

    queries: list[QueryPlan] = []

    # Interleave: for each tile index, cycle through all seeds
    for tile_idx in range(tiles_per_seed):
        for seed_idx in range(seeds_count):
            if len(queries) >= max_budget:
                break
            x, y, w, h = per_seed_tiles[seed_idx][tile_idx]
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
# Phase 2: Adaptive repeat queries — value-of-information scoring
# ---------------------------------------------------------------------------
VP_SIZE = 15


def _compute_cell_value_grid(
    observation_store: ObservationStore,
    seed_idx: int,
    analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Per-cell expected value of an additional observation.

    value(cell) = H(prior) * (1 - Σ q_k²) / (N + τ + 1)

    Uses identical posterior formula as solution_spatial.py:
      q_k = (n_k + τ·m_k) / (N + τ)  with per-archetype τ and blended prior m.
    """
    from solution_spatial import _get_adaptive_tau, _get_prior_mean

    h, w = analysis.height, analysis.width
    counts_grid = observation_store.get_seed_counts(seed_idx)
    obs_grid = observation_store.get_seed_obs_counts(seed_idx)
    value_grid = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            if not analysis.priority_mask[y, x]:
                continue

            terrain = int(analysis.grid[y, x])
            archetype = analysis.get_archetype(y, x)
            tau = _get_adaptive_tau(archetype)

            prior_mean = _get_prior_mean(
                seed_idx, analysis, observation_store, y, x, terrain
            )
            nonzero = prior_mean > 0
            if not nonzero.any():
                continue
            h_prior = float(-np.sum(prior_mean[nonzero] * np.log(prior_mean[nonzero])))
            if h_prior < 0.01:
                continue

            counts = counts_grid[y, x].astype(np.float64)
            n = float(obs_grid[y, x])
            posterior = (counts + tau * prior_mean) / (n + tau)
            gini = 1.0 - float(np.sum(posterior**2))
            marginal = 1.0 / (n + tau + 1.0)

            value_grid[y, x] = h_prior * gini * marginal

    return value_grid


def plan_repeat_queries(
    observation_store: ObservationStore,
    seed_analyses: list[SeedAnalysis],
    remaining_budget: int,
    map_w: int,
    map_h: int,
) -> list[QueryPlan]:
    """Select repeat queries by searching ALL legal 15×15 windows.

    Scores each window by sum of cell-level expected value-of-information,
    which combines scoring weight (prior entropy), prediction uncertainty
    (Gini impurity), and marginal value (1/(N+τ+1)).
    """
    if remaining_budget <= 0:
        return []

    vp = VP_SIZE
    max_x = max(0, map_w - vp)
    max_y = max(0, map_h - vp)

    candidates: list[tuple[float, int, int, int]] = []

    for seed_idx, analysis in enumerate(seed_analyses):
        value_grid = _compute_cell_value_grid(observation_store, seed_idx, analysis)

        sat = np.zeros((map_h + 1, map_w + 1), dtype=np.float64)
        sat[1:, 1:] = np.cumsum(np.cumsum(value_grid, axis=0), axis=1)

        for vy in range(max_y + 1):
            for vx in range(max_x + 1):
                ey = min(vy + vp, map_h)
                ex = min(vx + vp, map_w)
                window_val = sat[ey, ex] - sat[vy, ex] - sat[ey, vx] + sat[vy, vx]
                candidates.append((float(window_val), seed_idx, vx, vy))

    candidates.sort(key=lambda c: c[0], reverse=True)

    queries: list[QueryPlan] = []
    for _, seed_idx, vx, vy in candidates[:remaining_budget]:
        vw = min(vp, map_w - vx)
        vh = min(vp, map_h - vy)
        queries.append(
            QueryPlan(
                seed_index=seed_idx,
                viewport_x=vx,
                viewport_y=vy,
                viewport_w=vw,
                viewport_h=vh,
            )
        )

    return queries
