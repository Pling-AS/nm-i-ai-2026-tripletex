"""Hierarchical probability predictor for Astar Island.

Three-tier system:
  Tier 1 — Static cells: deterministic predictions (ocean, mountain).
  Tier 2 — Observed dynamic cells: Dirichlet posterior from observations.
  Tier 3 — Sparse/unobserved cells: archetype-pooled prior with shrinkage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from features import CellArchetype, SeedAnalysis
from observation_store import ObservationStore
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
    apply_floor_and_normalize,
    apply_floor_and_normalize_grid,
    is_static,
)

# Jeffrey's prior concentration parameter
ALPHA = 0.5

# Shrinkage: how many pseudo-observations the archetype prior is worth
ARCHETYPE_WEIGHT = 3.0

# Minimum observations before we trust local data over archetype
LOCAL_TRUST_THRESHOLD = 3


def _dirichlet_posterior_mean(
    counts: NDArray[np.int32],
    alpha: float = ALPHA,
) -> NDArray[np.floating]:
    """Compute Dirichlet posterior mean from observation counts.

    p_k = (count_k + alpha) / (N + K * alpha)
    """
    n = counts.sum()
    posterior = (counts.astype(np.float64) + alpha) / (n + NUM_CLASSES * alpha)
    return posterior


def _initial_terrain_prior(terrain_code: int) -> NDArray[np.floating]:
    """Construct a prior based on the initial terrain type.

    Encodes domain knowledge about what each terrain tends to become.
    """
    prior = np.zeros(NUM_CLASSES, dtype=np.float64)

    if terrain_code == TERRAIN_OCEAN:
        prior[CLASS_EMPTY] = 1.0
    elif terrain_code == TERRAIN_MOUNTAIN:
        prior[CLASS_MOUNTAIN] = 1.0
    elif terrain_code == TERRAIN_FOREST:
        # Forests mostly stay forest, small chance of being cleared/ruined
        prior[CLASS_FOREST] = 0.80
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_RUIN] = 0.05
        prior[CLASS_SETTLEMENT] = 0.03
        prior[CLASS_PORT] = 0.02
    elif terrain_code == TERRAIN_PLAINS:
        # Plains near settlements can become anything
        prior[CLASS_EMPTY] = 0.50
        prior[CLASS_SETTLEMENT] = 0.15
        prior[CLASS_PORT] = 0.05
        prior[CLASS_RUIN] = 0.10
        prior[CLASS_FOREST] = 0.20
    elif terrain_code == TERRAIN_SETTLEMENT:
        # Settlements can survive, expand to port, or collapse
        prior[CLASS_SETTLEMENT] = 0.40
        prior[CLASS_PORT] = 0.15
        prior[CLASS_RUIN] = 0.25
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_FOREST] = 0.10
    elif terrain_code == TERRAIN_PORT:
        # Ports: similar to settlements but more stable
        prior[CLASS_PORT] = 0.45
        prior[CLASS_SETTLEMENT] = 0.15
        prior[CLASS_RUIN] = 0.20
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_FOREST] = 0.10
    elif terrain_code == TERRAIN_RUIN:
        # Ruins can be reclaimed, overgrown, or persist
        prior[CLASS_RUIN] = 0.25
        prior[CLASS_FOREST] = 0.30
        prior[CLASS_EMPTY] = 0.25
        prior[CLASS_SETTLEMENT] = 0.12
        prior[CLASS_PORT] = 0.08
    else:
        # Unknown / generic empty
        prior[CLASS_EMPTY] = 0.60
        prior[CLASS_FOREST] = 0.15
        prior[CLASS_SETTLEMENT] = 0.10
        prior[CLASS_PORT] = 0.05
        prior[CLASS_RUIN] = 0.05
        prior[CLASS_MOUNTAIN] = 0.05

    return prior


def predict_cell(
    seed_index: int,
    y: int,
    x: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Predict the 6-class probability vector for one cell.

    Uses the three-tier hierarchical approach.
    """
    terrain = int(seed_analysis.grid[y, x])
    class_mask = seed_analysis.class_masks[y, x]

    # --- Tier 1: Static cells ---
    if is_static(terrain):
        pred = np.zeros(NUM_CLASSES, dtype=np.float64)
        if terrain == TERRAIN_OCEAN:
            pred[CLASS_EMPTY] = 1.0
        elif terrain == TERRAIN_MOUNTAIN:
            pred[CLASS_MOUNTAIN] = 1.0
        return apply_floor_and_normalize(pred, class_mask, epsilon=0.005)

    # --- Tier 2 & 3: Dynamic cells ---
    n_obs = observation_store.get_cell_obs_count(seed_index, y, x)
    local_counts = observation_store.get_cell_counts(seed_index, y, x)

    # Archetype-pooled prior (Tier 3)
    archetype = seed_analysis.get_archetype(y, x)
    arch_counts = observation_store.get_archetype_counts(archetype)
    arch_n = observation_store.get_archetype_obs_count(archetype)

    if arch_n > 0:
        archetype_prior = _dirichlet_posterior_mean(arch_counts, alpha=ALPHA)
    else:
        # Fall back to initial-terrain-based prior
        archetype_prior = _initial_terrain_prior(terrain)

    if n_obs >= LOCAL_TRUST_THRESHOLD:
        # Tier 2: trust local observations
        pred = _dirichlet_posterior_mean(local_counts, alpha=ALPHA)
    elif n_obs >= 1:
        # Blend local observations with archetype prior (shrinkage)
        local_post = _dirichlet_posterior_mean(local_counts, alpha=ALPHA)
        weight = n_obs / (n_obs + ARCHETYPE_WEIGHT)
        pred = weight * local_post + (1 - weight) * archetype_prior
    else:
        # Tier 3: fully pooled archetype prior
        pred = archetype_prior

    return apply_floor_and_normalize(pred, class_mask, epsilon=0.005)


def predict_full_grid(
    seed_index: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Generate the full H×W×6 prediction tensor for one seed.

    Returns a numpy array of shape (H, W, 6) with probabilities summing
    to 1.0 per cell.
    """
    h, w = seed_analysis.height, seed_analysis.width
    prediction = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            prediction[y, x] = predict_cell(
                seed_index, y, x, observation_store, seed_analysis
            )

    return prediction


def predict_full_grid_vectorized(
    seed_index: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Faster vectorized prediction for the full grid.

    Uses numpy operations where possible, with per-cell fallback
    only for blending logic.
    """
    h, w = seed_analysis.height, seed_analysis.width
    grid = seed_analysis.grid
    class_masks = seed_analysis.class_masks
    obs_counts_grid = observation_store.get_seed_obs_counts(seed_index)
    counts_grid = observation_store.get_seed_counts(seed_index)

    prediction = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)

    # --- Static cells: vectorized ---
    ocean_mask = grid == TERRAIN_OCEAN
    mountain_mask = grid == TERRAIN_MOUNTAIN

    prediction[ocean_mask, CLASS_EMPTY] = 1.0
    prediction[mountain_mask, CLASS_MOUNTAIN] = 1.0

    # --- Dynamic cells: iterate ---
    dynamic_mask = ~ocean_mask & ~mountain_mask

    for y in range(h):
        for x in range(w):
            if not dynamic_mask[y, x]:
                continue

            terrain = int(grid[y, x])
            n_obs = int(obs_counts_grid[y, x])
            local_counts = counts_grid[y, x]

            # Archetype prior
            archetype = seed_analysis.archetypes[y, x]
            arch_counts = observation_store.get_archetype_counts(archetype)
            arch_n = observation_store.get_archetype_obs_count(archetype)

            if arch_n > 0:
                arch_prior = _dirichlet_posterior_mean(arch_counts, alpha=ALPHA)
            else:
                arch_prior = _initial_terrain_prior(terrain)

            if n_obs >= LOCAL_TRUST_THRESHOLD:
                pred = _dirichlet_posterior_mean(local_counts, alpha=ALPHA)
            elif n_obs >= 1:
                local_post = _dirichlet_posterior_mean(local_counts, alpha=ALPHA)
                weight = n_obs / (n_obs + ARCHETYPE_WEIGHT)
                pred = weight * local_post + (1 - weight) * arch_prior
            else:
                pred = arch_prior

            prediction[y, x] = pred

    # Apply floor and normalize in one pass
    prediction = apply_floor_and_normalize_grid(prediction, class_masks, epsilon=0.005)

    return prediction
