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
# Adaptive τ strategy:
#   Per-archetype τ based on Jensen-Shannon divergence between round
#   observations and calibrated priors. High JSD → priors wrong → lower τ.
#   Round-level τ (median across archetypes) used for spatial smoothing regime.
TAU_MIN = 12.0
TAU_MAX = 40.0
TAU_DEFAULT = 25.0  # fallback if no observations
TAU = TAU_DEFAULT  # module-level round τ, set by compute_round_tau()

# Surprise-based τ reduction: lower τ when observation contradicts the prior.
# τ_eff = adaptive_τ - surprise × (adaptive_τ - TAU_OBS_FLOOR)
# where surprise = 1 - prior_prob(observed_class).
TAU_OBS_FLOOR = 8.0

NEIGHBOR_LAMBDA = 0.5
NEIGHBOR_MAX_TOTAL = 2.0

# Cross-seed pooling: borrow observations from same (y,x) across all seeds.
# Same terrain map means same-position cells share terrain-driven dynamics.
CROSS_SEED_LAMBDA = 0.5  # 0 = disabled; >0 = total pseudo-count weight from other seeds

FIELD_SIGMA = 2.0
FIELD_LAMBDA0 = 8.0
FIELD_ALPHA = 1.0

# ---------------------------------------------------------------------------
# Per-archetype adaptive τ — computed once after coverage queries
# ---------------------------------------------------------------------------
_round_regime: str = "easy"
_archetype_tau: dict[CellArchetype, float] = {}


def _jsd(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    """Jensen-Shannon divergence between two distributions."""
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)
    kl_pm = float(np.sum(p_safe * np.log(p_safe / m)))
    kl_qm = float(np.sum(q_safe * np.log(q_safe / m)))
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def set_round_tau(tau: float) -> None:
    """Set round-level τ. Derives regime for spatial smoothing."""
    global TAU, _round_regime
    TAU = tau
    _round_regime = "hard" if tau < 20.0 else "easy"
    print(f"[predictor] Round τ={tau:.1f}, regime='{_round_regime}'")


def compute_round_tau(
    seed_analyses: list[SeedAnalysis],
    observation_store: ObservationStore,
) -> float:
    """Compute per-archetype τ from JSD and derive round-level τ.

    For each archetype with enough observations:
      1. Compute JSD between observed frequencies and calibrated prior
      2. Map to τ: linear interpolation, high JSD → low τ
      3. Store in _archetype_tau for per-cell lookup

    Round τ = observation-weighted median across archetypes.
    Returns round τ in [TAU_MIN, TAU_MAX].
    """
    global _archetype_tau
    _archetype_tau.clear()

    arch_jsd_tau: list[tuple[float, float, int]] = []

    for archetype in observation_store._archetype_obs_count:
        count = observation_store._archetype_obs_count[archetype]
        if count < 5:
            continue
        terrain = archetype.initial_terrain
        if is_static(terrain):
            continue

        obs_counts = observation_store._archetype_counts[archetype]
        empirical = obs_counts.astype(np.float64)
        emp_sum = empirical.sum()
        if emp_sum == 0:
            continue
        empirical = empirical / emp_sum

        prior = _get_calibrated_prior(archetype, terrain)
        if prior is None:
            prior = _initial_terrain_prior(terrain)

        jsd = _jsd(empirical, prior)

        hardness = min(1.0, jsd / 0.20)
        tau = TAU_MAX - hardness * (TAU_MAX - TAU_MIN)
        tau = round(tau, 1)

        _archetype_tau[archetype] = tau
        arch_jsd_tau.append((jsd, tau, count))

    if not arch_jsd_tau:
        print(f"[predictor] No archetype observations, using τ={TAU_DEFAULT}")
        return TAU_DEFAULT

    arch_jsd_tau.sort(key=lambda x: x[1])
    total_w = sum(w for _, _, w in arch_jsd_tau)
    cumulative = 0
    round_tau = TAU_DEFAULT
    for jsd_val, tau_val, w in arch_jsd_tau:
        cumulative += w
        if cumulative >= total_w / 2:
            round_tau = tau_val
            break

    all_tau = [t for _, t, _ in arch_jsd_tau]
    avg_jsd = sum(j * w for j, _, w in arch_jsd_tau) / total_w
    print(
        f"[predictor] Adaptive τ: {len(arch_jsd_tau)} archetypes, "
        f"avg_jsd={avg_jsd:.4f}, "
        f"τ range=[{min(all_tau):.1f}, {max(all_tau):.1f}], "
        f"round τ={round_tau:.1f} (weighted median)"
    )

    return round_tau


def _get_adaptive_tau(archetype: CellArchetype) -> float:
    """Return per-archetype τ, falling back to round-level τ."""
    return _archetype_tau.get(archetype, TAU)


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
SHRINKAGE_KAPPA = 10.0

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


def _build_corrected_prior_grid(
    seed_index: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    """Build spatially-corrected prior grid using kernel-smoothed residual field.

    1. Compute base prior m_ik at each cell (existing _get_prior_mean)
    2. Compute light local posterior s_ik from cell observations
    3. Log-residual r_ik = log(s) - log(m), centered per cell
    4. Kernel-smooth residuals (leave-one-out, same-terrain)
    5. Blend corrected prior: softmax(log(m) + γ·R)
    """
    from scipy.ndimage import convolve

    h, w = seed_analysis.height, seed_analysis.width
    grid = seed_analysis.grid
    class_masks = seed_analysis.class_masks
    counts_grid = observation_store.get_seed_counts(seed_index).astype(np.float64)
    obs_grid = observation_store.get_seed_obs_counts(seed_index).astype(np.float64)

    EPS = 1e-6
    dynamic_mask = (grid != TERRAIN_OCEAN) & (grid != TERRAIN_MOUNTAIN)

    base_prior = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            if not dynamic_mask[y, x]:
                continue
            terrain = int(grid[y, x])
            base_prior[y, x] = _get_prior_mean(
                seed_index, seed_analysis, observation_store, y, x, terrain
            )

    local_post = (counts_grid + FIELD_ALPHA * base_prior) / (
        obs_grid[..., np.newaxis] + FIELD_ALPHA
    )

    log_residual = np.log(local_post + EPS) - np.log(base_prior + EPS)
    log_residual[~dynamic_mask] = 0.0

    allowed = class_masks.astype(np.float64)
    n_allowed = np.maximum(allowed.sum(axis=-1, keepdims=True), 1.0)
    residual_mean = (log_residual * allowed).sum(axis=-1, keepdims=True) / n_allowed
    log_residual = log_residual - residual_mean
    log_residual = np.clip(log_residual, -2.0, 2.0)

    radius = int(3 * FIELD_SIGMA)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * FIELD_SIGMA * FIELD_SIGMA))
    kernel[radius, radius] = 0.0
    kernel /= kernel.sum()

    obs_weight = np.minimum(obs_grid, 2.0)

    field = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    ess = np.zeros((h, w), dtype=np.float64)

    unique_terrains = np.unique(grid[dynamic_mask])
    for t in unique_terrains:
        tmask = (grid == int(t)).astype(np.float64)
        weighted_tmask = obs_weight * tmask
        den = convolve(weighted_tmask, kernel, mode="nearest")
        ess += den * tmask

        for k in range(NUM_CLASSES):
            num = convolve(
                weighted_tmask * log_residual[:, :, k], kernel, mode="nearest"
            )
            safe_den = np.where(den > EPS, den, 1.0)
            field[:, :, k] += np.where(den > EPS, num / safe_den, 0.0) * tmask

    safe_prior = np.clip(base_prior, EPS, 1.0)
    ent = -np.sum(safe_prior * np.log(safe_prior + EPS), axis=-1)
    max_ent = np.log(np.maximum(class_masks.sum(axis=-1).astype(np.float64), 2.0))
    ent_scale = ent / np.maximum(max_ent, EPS)

    gamma = (ess / (ess + FIELD_LAMBDA0)) * ent_scale

    logits = np.log(safe_prior) + gamma[..., np.newaxis] * field
    logits[~class_masks] = -1e9
    logits -= logits.max(axis=-1, keepdims=True)
    corrected = np.exp(logits)
    corrected *= class_masks
    corrected /= np.maximum(corrected.sum(axis=-1, keepdims=True), EPS)

    corrected[~dynamic_mask] = 0.0
    return corrected


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

    dynamic_mask = ~ocean_mask & ~mountain_mask
    obs_grid = observation_store.get_seed_obs_counts(seed_index)

    # Pre-compute cross-seed pooled counts (sum of other seeds at same position)
    cross_seed_grid = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    if CROSS_SEED_LAMBDA > 0:
        for other_seed in range(observation_store.seeds_count):
            if other_seed != seed_index:
                cross_seed_grid += observation_store.get_seed_counts(other_seed).astype(np.float64)

    for y in range(h):
        for x in range(w):
            if not dynamic_mask[y, x]:
                continue

            terrain = int(grid[y, x])
            local_counts = counts_grid[y, x].astype(np.float64)

            neighbor_pseudo = np.zeros(NUM_CLASSES, dtype=np.float64)
            n_same = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if int(grid[ny, nx]) == terrain and obs_grid[ny, nx] > 0:
                            neighbor_pseudo += counts_grid[ny, nx].astype(np.float64)
                            n_same += 1

            if n_same > 0:
                total_lam = min(NEIGHBOR_LAMBDA * n_same, NEIGHBOR_MAX_TOTAL)
                nw = total_lam / max(neighbor_pseudo.sum(), 1.0)
                effective_counts = local_counts + nw * neighbor_pseudo
            else:
                effective_counts = local_counts

            # Cross-seed pooling
            if CROSS_SEED_LAMBDA > 0:
                cs = cross_seed_grid[y, x]
                cs_total = cs.sum()
                if cs_total > 0:
                    cs_weight = CROSS_SEED_LAMBDA / cs_total
                    effective_counts = effective_counts + cs_weight * cs

            prior_mean = _get_prior_mean(
                seed_index, seed_analysis, observation_store, y, x, terrain
            )
            archetype = seed_analysis.get_archetype(y, x)
            adaptive_tau = _get_adaptive_tau(archetype)
            n_eff = effective_counts.sum()

            if n_eff > 0:
                observed_class = int(np.argmax(effective_counts))
                surprise = 1.0 - prior_mean[observed_class]
                tau = adaptive_tau - surprise * (adaptive_tau - TAU_OBS_FLOOR)
            else:
                tau = adaptive_tau

            prediction[y, x] = (effective_counts + tau * prior_mean) / (n_eff + tau)

    prediction = apply_floor_and_normalize_grid(prediction, class_masks)
    prediction = _spatial_smooth(prediction, class_masks, grid)

    return prediction


def _spatial_smooth(
    prediction: NDArray[np.floating],
    class_masks: NDArray[np.bool_],
    grid: NDArray[np.int_],
    max_beta: float = 0.15,
) -> NDArray[np.floating]:
    """Terrain-aware uncertainty-weighted 8-neighbor smoothing.

    Only averages predictions from same-terrain neighbors, preventing
    cross-terrain contamination (e.g., forest priors bleeding into plains).

    p'(cell) = (1 - β) · p(cell) + β · mean(p(same-terrain neighbors))
    where β = max_beta · (1 - max(p(cell))).
    """
    h, w, c = prediction.shape

    max_probs = prediction.max(axis=-1)
    uncertainty = 1.0 - max_probs

    padded_pred = np.pad(prediction, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded_grid = np.pad(grid, ((1, 1), (1, 1)), mode="constant", constant_values=-1)

    neighbor_sum = np.zeros_like(prediction)
    neighbor_count = np.zeros((h, w), dtype=np.float64)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted_pred = padded_pred[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx, :]
            shifted_grid = padded_grid[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            same_terrain = shifted_grid == grid
            neighbor_sum += shifted_pred * same_terrain[..., np.newaxis]
            neighbor_count += same_terrain.astype(np.float64)

    safe_count = np.maximum(neighbor_count, 1.0)
    neighbor_mean = neighbor_sum / safe_count[..., np.newaxis]

    beta = max_beta * uncertainty
    is_static_mask = (grid == TERRAIN_OCEAN) | (grid == TERRAIN_MOUNTAIN)
    no_neighbors = neighbor_count == 0
    beta[is_static_mask | no_neighbors] = 0.0

    smoothed = (1.0 - beta[..., np.newaxis]) * prediction + beta[
        ..., np.newaxis
    ] * neighbor_mean

    return apply_floor_and_normalize_grid(smoothed, class_masks)
