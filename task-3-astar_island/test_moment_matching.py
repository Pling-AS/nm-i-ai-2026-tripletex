import numpy as np
import sys
from client import AstarClient
from observation_store import ObservationStore
from features import SeedAnalysis
from solution_spatial import predict_full_grid_vectorized
from analyze import kl_divergence, entropy
from utils import NUM_CLASSES, apply_floor_and_normalize_grid

ROUND_ID = "75e625c3-60cb-4392-af3e-c86a98bde8c2"  # Round 10
OBS_FILE = "data/obs_75e625c3"


def score_prediction(prediction, ground_truth):
    """Compute score: 100 * exp(-3 * mean(weighted_KL))."""
    total_weighted_kl = 0.0
    total_entropy = 0.0

    h, w, _ = ground_truth.shape
    for y in range(h):
        for x in range(w):
            gt = ground_truth[y, x]
            pred = prediction[y, x]
            ent = entropy(gt)
            if ent > 0.01:
                kl = kl_divergence(gt, pred)
                total_weighted_kl += ent * kl
                total_entropy += ent

    if total_entropy == 0:
        return 0.0

    avg_wkl = total_weighted_kl / total_entropy
    return 100.0 * np.exp(-3.0 * avg_wkl)


def moment_matching(prediction, observation_store, seed_index):
    """Calibrate prediction to match global observation frequencies."""
    # 1. Calculate global empirical distribution from observations
    obs_counts = observation_store.get_seed_counts(seed_index)  # (H, W, 6)
    total_counts = obs_counts.sum(axis=(0, 1))  # (6,)
    if total_counts.sum() == 0:
        return prediction

    empirical_dist = total_counts.astype(np.float64) / total_counts.sum()

    # 2. Calculate global prediction mean
    # We only care about dynamic cells (not ocean/mountain)?
    # Or strictly all cells?
    # Observations cover all cells. Prediction covers all cells.
    # But Ocean/Mountain are static and fixed.
    # Moment matching should probably only apply to dynamic classes?
    # Or just apply globally and let static masks fix it later?
    # Let's apply globally first.

    pred_mean = prediction.mean(axis=(0, 1))  # (6,)

    # 3. Compute weights
    # w_k = P_k / Q_k
    # Avoid division by zero
    weights = np.ones(NUM_CLASSES, dtype=np.float64)
    for k in range(NUM_CLASSES):
        if pred_mean[k] > 1e-4:
            weights[k] = empirical_dist[k] / pred_mean[k]

    # Clip weights to avoid extreme explosions
    weights = np.clip(weights, 0.1, 10.0)

    print(f"  Moment Matching Weights: {weights}")
    print(f"  Empirical: {empirical_dist}")
    print(f"  Predicted: {pred_mean}")

    # 4. Apply weights
    calibrated = prediction * weights[np.newaxis, np.newaxis, :]

    # 5. Renormalize
    sums = calibrated.sum(axis=-1, keepdims=True)
    calibrated = np.divide(
        calibrated, sums, out=np.zeros_like(calibrated), where=sums > 0
    )

    return calibrated


def main():
    client = AstarClient()
    print(f"Fetching Round {ROUND_ID}...")
    detail = client.get_round_detail(ROUND_ID)

    print(f"Loading observations from {OBS_FILE}...")
    obs_store = ObservationStore.load(OBS_FILE)

    seeds_count = detail.seeds_count
    print(f"Analyzing {seeds_count} seeds...")

    total_base_score = 0.0
    total_calib_score = 0.0

    for seed_idx in range(seeds_count):
        print(f"\n--- Seed {seed_idx} ---")

        # 1. Setup
        state = detail.initial_states[seed_idx]
        seed_analysis = SeedAnalysis(state["grid"], state["settlements"])

        # 2. Get Ground Truth
        analysis = client.get_analysis(ROUND_ID, seed_idx)
        gt = np.array(analysis["ground_truth"], dtype=np.float64)

        # 3. Baseline Prediction
        base_pred = predict_full_grid_vectorized(seed_idx, obs_store, seed_analysis)
        base_score = score_prediction(base_pred, gt)
        print(f"  Baseline Score: {base_score:.4f}")

        # 4. Calibrated Prediction
        calib_pred = moment_matching(base_pred, obs_store, seed_idx)

        # Re-apply masks to ensure static cells are correct
        # (Moment matching might have shifted Ocean/Mountain)
        ocean_mask = seed_analysis.grid == 10  # Ocean
        mountain_mask = seed_analysis.grid == 5  # Mountain
        # Actually features.py uses constants. Let's assume standard codes.
        # But we should use the masks from solution_spatial or features.
        # Simple fix: re-run floor_and_normalize which enforces masks if we pass them?
        # solution_spatial.apply_floor_and_normalize_grid uses class_masks.

        calib_pred = apply_floor_and_normalize_grid(
            calib_pred, seed_analysis.class_masks
        )

        calib_score = score_prediction(calib_pred, gt)
        print(f"  Calibrated Score: {calib_score:.4f}")

        total_base_score += base_score
        total_calib_score += calib_score

    print("\n" + "=" * 40)
    print(f"Average Baseline:   {total_base_score / seeds_count:.4f}")
    print(f"Average Calibrated: {total_calib_score / seeds_count:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
