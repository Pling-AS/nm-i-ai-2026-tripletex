"""Hierarchical probability predictor for Astar Island.

Single-formula Dirichlet approach (no double-shrinkage):

  p_k = (n_k + τ · m_k) / (N + τ)

Where:
  n_k = raw observation count for class k
  m_k = prior mean (from archetype pool, calibration, or hand-tuned)
  τ   = prior strength (equivalent prior sample size)
  N   = total observations for this cell

Tiers:
  Tier 1 — Static cells: deterministic predictions (ocean, mountain).
  Tier 2 — Observed cells: Dirichlet posterior with informative prior.
  Tier 3 — Unobserved cells: prior mean directly.
"""

from __future__ import annotations

import json
from pathlib import Path

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

# Prior strength τ: equivalent prior sample size.
# Controls how much one observation shifts the prediction away from the prior.
#   τ=3  → N=1 gets 25% weight (≈old approach, breaks even)
#   τ=8  → N=1 gets 11% weight (moderate trust in prior)
#   τ=15 → N=1 gets  6% weight (strong trust in prior)
#   τ=25 → N=1 gets  4% weight (very strong trust)
#
# Empirical sweep on Round 1 (stochastic N=1 samples, 3×3 tiling):
#   τ=3 → 73.6, τ=5 → 79.1, τ=8 → 81.5, τ=15 → 82.7, τ=25 → 82.8
#   Prior-only → 86.1, Old approach → 74.1
#
# τ=15 chosen: near-optimal in-sample, robust for cross-round prediction
# where calibrated priors may be less accurate.
TAU = 15.0


TAU_MIN = 12.0
TAU_MAX = 18.0

_archetype_entropy_cache: dict[str, float] = {}
_max_calibrated_entropy: float = 0.0


def _build_entropy_cache() -> None:
    """Populate _archetype_entropy_cache from calibration.json."""
    global _max_calibrated_entropy
    cal = _load_calibration()
    if cal is None:
        return
    arch_priors = cal.get("archetype_priors", {})
    for key, probs in arch_priors.items():
        p = np.array(probs, dtype=np.float64)
        p_pos = p[p > 0]
        h = float(-np.sum(p_pos * np.log(p_pos)))
        _archetype_entropy_cache[key] = h
    if _archetype_entropy_cache:
        _max_calibrated_entropy = max(_archetype_entropy_cache.values())


def _get_adaptive_tau(archetype: CellArchetype) -> float:
    """Return τ scaled by calibrated archetype entropy.

    τ = TAU_MIN + (TAU_MAX - TAU_MIN) × (1 - H(arch) / H_max)

    High-entropy archetypes (settlements, ports) get lower τ → trust observations more.
    Low-entropy archetypes (remote plains/forest) get higher τ → trust prior more.
    Falls back to TAU (15.0) if calibration is unavailable.
    """
    if not _archetype_entropy_cache:
        _build_entropy_cache()
    if not _archetype_entropy_cache or _max_calibrated_entropy < 1e-8:
        return TAU

    key = str(archetype)
    h = _archetype_entropy_cache.get(key)
    if h is None:
        return TAU

    ratio = h / _max_calibrated_entropy
    return TAU_MIN + (TAU_MAX - TAU_MIN) * (1.0 - ratio)


# ---------------------------------------------------------------------------
# Calibration loading (from analyze.py output)
# ---------------------------------------------------------------------------
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

_calibration: dict | None = None


def _load_calibration() -> dict | None:
    """Load calibrated priors from disk (once, on first use)."""
    global _calibration
    if _calibration is not None:
        return _calibration
    if CALIBRATION_FILE.exists():
        try:
            data = json.loads(CALIBRATION_FILE.read_text())
            _calibration = data
            n_arch = len(data.get("archetype_priors", {}))
            n_terr = len(data.get("terrain_priors", {}))
            rounds = data.get("metadata", {}).get("rounds_analyzed", [])
            print(
                f"[predictor] Loaded calibration: {n_arch} archetype priors, "
                f"{n_terr} terrain priors from {len(rounds)} round(s)"
            )
            return _calibration
        except (json.JSONDecodeError, KeyError, OSError):
            return None
    return None


def _get_calibrated_prior(
    archetype: CellArchetype, terrain_code: int
) -> NDArray[np.floating] | None:
    """Try to get a calibrated prior from the calibration file.

    Lookup order:
      1. Exact archetype match
      2. Terrain-level prior
      3. None (caller falls back to hand-tuned)
    """
    cal = _load_calibration()
    if cal is None:
        return None

    # Try exact archetype match
    arch_key = str(archetype)
    arch_priors = cal.get("archetype_priors", {})
    if arch_key in arch_priors:
        return np.array(arch_priors[arch_key], dtype=np.float64)

    # Try terrain-level prior
    terrain_priors = cal.get("terrain_priors", {})
    terrain_key = str(terrain_code)
    if terrain_key in terrain_priors:
        return np.array(terrain_priors[terrain_key], dtype=np.float64)

    return None


def _dirichlet_posterior(
    counts: NDArray[np.int32],
    prior_mean: NDArray[np.floating],
    tau: float = TAU,
) -> NDArray[np.floating]:
    """Compute Dirichlet posterior mean with informative prior.

    p_k = (n_k + τ · m_k) / (N + τ)

    This is the single shrinkage formula — no double-blending needed.
    For N=0, returns prior_mean exactly.
    """
    n = counts.sum()
    posterior = (counts.astype(np.float64) + tau * prior_mean) / (n + tau)
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


# Shrinkage parameter κ: controls how many round observations are needed
# before the round-specific archetype pool dominates calibration.
#   λ = arch_n / (arch_n + κ)
#   κ=50 → need 50 round obs for 50/50 blend, 10 obs → 17% round weight
#   κ=100 → need 100 round obs for 50/50, more conservative
SHRINKAGE_KAPPA = 50.0

# Minimum round observations to even consider blending (noise floor)
MIN_ARCHETYPE_BLEND = 5


def _archetype_backoff_chain(archetype: CellArchetype) -> list[CellArchetype]:
    """Progressively coarser archetypes: full → drop coastal → drop dist → terrain only."""
    t, coast, dist, adj_sett = archetype
    return [
        archetype,
        CellArchetype(t, False, dist, adj_sett),
        CellArchetype(t, False, 3, adj_sett),
        CellArchetype(t, False, 3, False),
    ]


def _get_prior_mean(
    seed_index: int,
    seed_analysis: SeedAnalysis,
    observation_store: ObservationStore,
    y: int,
    x: int,
    terrain: int,
) -> NDArray[np.floating]:
    """Build prior via count-weighted shrinkage: m = λ·m_round + (1-λ)·m_cal.

    λ = arch_n / (arch_n + κ).  Leave-one-out on the target cell.
    """
    archetype = seed_analysis.get_archetype(y, x)

    # Best calibrated prior (backoff through archetype chain)
    m_cal = None
    for arch in _archetype_backoff_chain(archetype):
        calibrated = _get_calibrated_prior(arch, terrain)
        if calibrated is not None:
            m_cal = calibrated
            break
    if m_cal is None:
        m_cal = _initial_terrain_prior(terrain)

    # Round-specific archetype mean (leave-one-out)
    m_round = None
    best_arch_n = 0
    for arch in _archetype_backoff_chain(archetype):
        arch_n = observation_store.get_archetype_obs_count(arch)
        if arch_n >= MIN_ARCHETYPE_BLEND:
            arch_counts = observation_store.get_archetype_counts(arch).copy()
            local_counts = observation_store.get_cell_counts(seed_index, y, x)
            arch_counts_loo = arch_counts - local_counts
            total_loo = arch_counts_loo.sum()

            if total_loo > 0:
                m_round = arch_counts_loo.astype(np.float64) / total_loo
                best_arch_n = max(arch_n - int(local_counts.sum()), 1)
            else:
                m_round = arch_counts.astype(np.float64) / arch_counts.sum()
                best_arch_n = arch_n
            break

    if m_round is None:
        return m_cal

    lam = best_arch_n / (best_arch_n + SHRINKAGE_KAPPA)
    blended = lam * m_round + (1.0 - lam) * m_cal
    total = blended.sum()
    if total > 0:
        blended /= total

    return blended


def predict_cell(
    seed_index: int,
    y: int,
    x: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Predict the 6-class probability vector for one cell.

    Single-formula approach: p_k = (n_k + τ·m_k) / (N + τ)
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
        return apply_floor_and_normalize(pred, class_mask)

    local_counts = observation_store.get_cell_counts(seed_index, y, x)
    prior_mean = _get_prior_mean(
        seed_index, seed_analysis, observation_store, y, x, terrain
    )
    archetype = seed_analysis.get_archetype(y, x)
    pred = _dirichlet_posterior(
        local_counts, prior_mean, tau=_get_adaptive_tau(archetype)
    )

    return apply_floor_and_normalize(pred, class_mask)


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

    Uses single-formula Dirichlet: p_k = (n_k + τ·m_k) / (N + τ)
    Static cells handled vectorized, dynamic cells iterate for prior lookup.
    """
    h, w = seed_analysis.height, seed_analysis.width
    grid = seed_analysis.grid
    class_masks = seed_analysis.class_masks
    counts_grid = observation_store.get_seed_counts(seed_index)

    prediction = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)

    # --- Static cells: vectorized ---
    ocean_mask = grid == TERRAIN_OCEAN
    mountain_mask = grid == TERRAIN_MOUNTAIN

    prediction[ocean_mask, CLASS_EMPTY] = 1.0
    prediction[mountain_mask, CLASS_MOUNTAIN] = 1.0

    # --- Dynamic cells: single formula per cell ---
    dynamic_mask = ~ocean_mask & ~mountain_mask

    for y in range(h):
        for x in range(w):
            if not dynamic_mask[y, x]:
                continue

            terrain = int(grid[y, x])
            local_counts = counts_grid[y, x]
            prior_mean = _get_prior_mean(
                seed_index, seed_analysis, observation_store, y, x, terrain
            )
            archetype = seed_analysis.get_archetype(y, x)
            prediction[y, x] = _dirichlet_posterior(
                local_counts, prior_mean, tau=_get_adaptive_tau(archetype)
            )

    prediction = apply_floor_and_normalize_grid(prediction, class_masks)

    return prediction
