"""Spatial Bayesian Diffusion predictor for Astar Island."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
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

TAU_MIN = 12.0
TAU_MAX = 40.0
TAU_DEFAULT = 25.0
TAU_OBS_FLOOR = 8.0
TAU_SURPRISE_C = 4.0
NEIGHBOR_LAMBDA = 0.5
NEIGHBOR_MAX_TOTAL = 2.0
SHRINKAGE_KAPPA = 10.0
MIN_ARCHETYPE_BLEND = 5

TAU = TAU_DEFAULT
_round_regime = "easy"
_archetype_tau: dict[CellArchetype, float] = {}
_settlement_tilt: dict[tuple[int, int], NDArray[np.floating]] = {}
_ruin_tilt: dict[tuple[int, int], NDArray[np.floating]] = {}


def _get_dist_bin(d: float) -> int:
    if d <= 0.5:
        return 0
    if d <= 1.5:
        return 1
    if d <= 2.5:
        return 2
    if d <= 3.5:
        return 3
    if d <= 4.5:
        return 4
    if d <= 5.5:
        return 5
    return 6


CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
_calibration: dict | None = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ez = np.exp(x)
    return float(ez / (1.0 + ez))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _jsd(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)
    kl_pm = float(np.sum(p_safe * np.log(p_safe / m)))
    kl_qm = float(np.sum(q_safe * np.log(q_safe / m)))
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def _load_calibration() -> dict | None:
    global _calibration
    if _calibration is not None:
        return _calibration
    if CALIBRATION_FILE.exists():
        try:
            data = json.loads(CALIBRATION_FILE.read_text())
            _calibration = data
            return _calibration
        except (json.JSONDecodeError, KeyError, OSError):
            return None
    return None


def _get_calibrated_prior(
    archetype: CellArchetype, terrain_code: int
) -> NDArray[np.floating] | None:
    cal = _load_calibration()
    if cal is None:
        return None
    arch_key = str(archetype)
    if arch_key in cal.get("archetype_priors", {}):
        return np.array(cal["archetype_priors"][arch_key], dtype=np.float64)
    terrain_key = str(terrain_code)
    if terrain_key in cal.get("terrain_priors", {}):
        return np.array(cal["terrain_priors"][terrain_key], dtype=np.float64)
    return None


def _initial_terrain_prior(terrain_code: int) -> NDArray[np.floating]:
    prior = np.zeros(NUM_CLASSES, dtype=np.float64)
    if terrain_code == TERRAIN_OCEAN:
        prior[CLASS_EMPTY] = 1.0
    elif terrain_code == TERRAIN_MOUNTAIN:
        prior[CLASS_MOUNTAIN] = 1.0
    elif terrain_code == TERRAIN_FOREST:
        prior[CLASS_FOREST] = 0.80
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_RUIN] = 0.05
        prior[CLASS_SETTLEMENT] = 0.03
        prior[CLASS_PORT] = 0.02
    elif terrain_code == TERRAIN_PLAINS:
        prior[CLASS_EMPTY] = 0.50
        prior[CLASS_SETTLEMENT] = 0.15
        prior[CLASS_PORT] = 0.05
        prior[CLASS_RUIN] = 0.10
        prior[CLASS_FOREST] = 0.20
    elif terrain_code == TERRAIN_SETTLEMENT:
        prior[CLASS_SETTLEMENT] = 0.40
        prior[CLASS_PORT] = 0.15
        prior[CLASS_RUIN] = 0.25
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_FOREST] = 0.10
    elif terrain_code == TERRAIN_PORT:
        prior[CLASS_PORT] = 0.45
        prior[CLASS_SETTLEMENT] = 0.15
        prior[CLASS_RUIN] = 0.20
        prior[CLASS_EMPTY] = 0.10
        prior[CLASS_FOREST] = 0.10
    elif terrain_code == TERRAIN_RUIN:
        prior[CLASS_RUIN] = 0.25
        prior[CLASS_FOREST] = 0.30
        prior[CLASS_EMPTY] = 0.25
        prior[CLASS_SETTLEMENT] = 0.12
        prior[CLASS_PORT] = 0.08
    else:
        prior[CLASS_EMPTY] = 0.60
        prior[CLASS_FOREST] = 0.15
        prior[CLASS_SETTLEMENT] = 0.10
        prior[CLASS_PORT] = 0.05
        prior[CLASS_RUIN] = 0.05
        prior[CLASS_MOUNTAIN] = 0.05
    return prior


def _archetype_backoff_chain(archetype: CellArchetype) -> list[CellArchetype]:
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
    archetype = seed_analysis.get_archetype(y, x)
    m_cal = None
    for arch in _archetype_backoff_chain(archetype):
        calibrated = _get_calibrated_prior(arch, terrain)
        if calibrated is not None:
            m_cal = calibrated
            break
    if m_cal is None:
        m_cal = _initial_terrain_prior(terrain)

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
    if blended.sum() > 0:
        blended /= blended.sum()
    return blended


def set_round_tau(tau: float) -> None:
    global TAU, _round_regime
    TAU = tau
    _round_regime = "hard" if tau < 20.0 else "easy"


def compute_round_tau(
    seed_analyses: list[SeedAnalysis], observation_store: ObservationStore
) -> float:
    global _archetype_tau, TAU
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
        if empirical.sum() == 0:
            continue
        empirical /= empirical.sum()

        prior = _get_calibrated_prior(archetype, terrain)
        if prior is None:
            prior = _initial_terrain_prior(terrain)

        jsd = _jsd(empirical, prior)
        hardness = min(1.0, jsd / 0.20)
        tau = TAU_MAX - hardness * (TAU_MAX - TAU_MIN)
        _archetype_tau[archetype] = tau
        arch_jsd_tau.append((jsd, tau, count))

    if not arch_jsd_tau:
        return TAU_DEFAULT

    arch_jsd_tau.sort(key=lambda x: x[1])
    total_w = sum(w for _, _, w in arch_jsd_tau)
    cumulative = 0
    round_tau = TAU_DEFAULT
    for _, tau_val, w in arch_jsd_tau:
        cumulative += w
        if cumulative >= total_w / 2:
            round_tau = tau_val
            break

    TAU = round_tau
    return round_tau


def _get_adaptive_tau(archetype: CellArchetype) -> float:
    return _archetype_tau.get(archetype, TAU)


def estimate_round_tilts(
    seed_analyses: list[SeedAnalysis], observation_store: ObservationStore
) -> None:
    global _settlement_tilt, _ruin_tilt
    _settlement_tilt.clear()
    _ruin_tilt.clear()

    s_counts: dict[tuple[int, int], NDArray[np.floating]] = {}
    r_counts: dict[tuple[int, int], NDArray[np.floating]] = {}

    for seed_idx, analysis in enumerate(seed_analyses):
        obs_grid = observation_store.get_seed_obs_counts(seed_idx)
        counts_grid = observation_store.get_seed_counts(seed_idx)

        ys, xs = np.where(obs_grid > 0)

        for y, x in zip(ys, xs):
            terrain = int(analysis.grid[y, x])
            if is_static(terrain):
                continue

            local_counts = counts_grid[y, x].astype(np.float64)

            dist_s = float(analysis.dist_to_settlement[y, x])
            bin_s = _get_dist_bin(dist_s)
            if bin_s <= 5:
                key = (terrain, bin_s)
                if key not in s_counts:
                    s_counts[key] = np.zeros(NUM_CLASSES, dtype=np.float64)
                s_counts[key] += local_counts

            dist_r = float(analysis.dist_to_ruin[y, x])
            bin_r = _get_dist_bin(dist_r)
            if bin_r <= 5:
                key = (terrain, bin_r)
                if key not in r_counts:
                    r_counts[key] = np.zeros(NUM_CLASSES, dtype=np.float64)
                r_counts[key] += local_counts

    min_samples = 30.0

    for key, counts in s_counts.items():
        total = counts.sum()
        if total >= min_samples:
            _settlement_tilt[key] = counts / total

    for key, counts in r_counts.items():
        total = counts.sum()
        if total >= min_samples:
            _ruin_tilt[key] = counts / total


def _apply_tilts(
    prior_mean: NDArray[np.floating], dist_s: float, dist_r: float, terrain: int
) -> NDArray[np.floating]:
    bin_s = _get_dist_bin(dist_s)
    bin_r = _get_dist_bin(dist_r)

    tilt_s = _settlement_tilt.get((terrain, bin_s))
    tilt_r = _ruin_tilt.get((terrain, bin_r))

    if tilt_s is None and tilt_r is None:
        return prior_mean

    w_s = 0.15 if tilt_s is not None else 0.0
    w_r = 0.10 if tilt_r is not None else 0.0

    w_prior = 1.0 - w_s - w_r

    new_prior = w_prior * prior_mean
    if tilt_s is not None:
        new_prior += w_s * tilt_s
    if tilt_r is not None:
        new_prior += w_r * tilt_r

    return new_prior


def _surprise_tau(
    local_counts: NDArray[np.floating],
    n_local: float,
    prior_mean: NDArray[np.floating],
    base_tau: float,
) -> float:
    if n_local <= 0:
        return base_tau
    obs_freq = local_counts / n_local
    surprise = 0.0
    for k in (CLASS_SETTLEMENT, CLASS_RUIN):
        excess = float(obs_freq[k]) - float(prior_mean[k])
        if excess > 0:
            surprise += excess
    if surprise <= 0:
        return base_tau
    tau_eff = base_tau / (1.0 + TAU_SURPRISE_C * n_local * surprise)
    return max(TAU_OBS_FLOOR, tau_eff)


def predict_full_grid_vectorized(
    seed_index: int,
    observation_store: ObservationStore,
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    h, w = seed_analysis.height, seed_analysis.width
    grid = seed_analysis.grid
    class_masks = seed_analysis.class_masks
    counts_grid = observation_store.get_seed_counts(seed_index)

    prediction = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)

    ocean_mask = grid == TERRAIN_OCEAN
    mountain_mask = grid == TERRAIN_MOUNTAIN
    prediction[ocean_mask, CLASS_EMPTY] = 1.0
    prediction[mountain_mask, CLASS_MOUNTAIN] = 1.0

    dynamic_mask = ~ocean_mask & ~mountain_mask
    obs_grid = observation_store.get_seed_obs_counts(seed_index)

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

            effective_counts = local_counts
            if n_same > 0:
                total_lam = min(NEIGHBOR_LAMBDA * n_same, NEIGHBOR_MAX_TOTAL)
                nw = total_lam / max(neighbor_pseudo.sum(), 1.0)
                effective_counts = local_counts + nw * neighbor_pseudo

            prior_mean = _get_prior_mean(
                seed_index, seed_analysis, observation_store, y, x, terrain
            )
            dist_s = float(seed_analysis.dist_to_settlement[y, x])
            dist_r = float(seed_analysis.dist_to_ruin[y, x])
            prior_mean = _apply_tilts(prior_mean, dist_s, dist_r, terrain)

            archetype = seed_analysis.get_archetype(y, x)
            adaptive_tau = _get_adaptive_tau(archetype)
            n_eff = effective_counts.sum()
            n_local = float(local_counts.sum())
            tau = _surprise_tau(local_counts, n_local, prior_mean, adaptive_tau)

            prediction[y, x] = (effective_counts + tau * prior_mean) / (n_eff + tau)

    prediction = apply_floor_and_normalize_grid(prediction, class_masks)
    prediction = _static_smooth(prediction, class_masks, grid)
    prediction = _dynamic_diffusion(prediction, class_masks, seed_analysis)

    return prediction


def _static_smooth(
    prediction: NDArray[np.floating],
    class_masks: NDArray[np.bool_],
    grid: NDArray[np.int_],
    max_beta: float = 0.15,
) -> NDArray[np.floating]:
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


def _dynamic_diffusion(
    prediction: NDArray[np.floating],
    class_masks: NDArray[np.bool_],
    seed_analysis: SeedAnalysis,
) -> NDArray[np.floating]:
    sigma = 0.8
    mix_rates = {
        CLASS_SETTLEMENT: 0.3,
        CLASS_PORT: 0.3,
        CLASS_RUIN: 0.15,
        CLASS_EMPTY: 0.0,
        CLASS_FOREST: 0.0,
        CLASS_MOUNTAIN: 0.0,
    }

    new_prediction = prediction.copy()

    s_channel = prediction[..., CLASS_SETTLEMENT]
    s_blurred = gaussian_filter(s_channel, sigma=sigma, mode="nearest")

    p_channel = prediction[..., CLASS_PORT]
    p_blurred = gaussian_filter(p_channel, sigma=sigma, mode="nearest")

    r_channel = prediction[..., CLASS_RUIN]
    r_blurred = gaussian_filter(r_channel, sigma=sigma, mode="nearest")

    gamma_s = mix_rates[CLASS_SETTLEMENT]
    new_prediction[..., CLASS_SETTLEMENT] = (
        1 - gamma_s
    ) * s_channel + gamma_s * s_blurred

    gamma_p = mix_rates[CLASS_PORT]
    new_prediction[..., CLASS_PORT] = (1 - gamma_p) * p_channel + gamma_p * p_blurred

    gamma_r = mix_rates[CLASS_RUIN]
    new_prediction[..., CLASS_RUIN] = (1 - gamma_r) * r_channel + gamma_r * r_blurred

    # Use pre-computed port mask (coastal | is_port)
    port_mask = seed_analysis.port_mask
    not_valid_port = ~port_mask

    new_prediction[not_valid_port, CLASS_PORT] = 0.0

    return apply_floor_and_normalize_grid(new_prediction, class_masks, epsilon=0.015)
