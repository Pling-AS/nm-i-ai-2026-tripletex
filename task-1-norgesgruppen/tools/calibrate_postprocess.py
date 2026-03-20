"""Sweep confidence thresholds to find optimal hybrid score.

Operates on already-generated predictions — no model inference needed.
Sweeps final_conf_threshold and optionally filters by per-class minimums.

Usage:
    python tools/calibrate_postprocess.py \
        --gt annotations.json \
        --pred predictions.json \
        --out submission/thresholds.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Threshold calibration")
    parser.add_argument(
        "--gt", required=True, help="Ground truth annotations.json"
    )
    parser.add_argument(
        "--pred", required=True, help="Predictions JSON (low conf threshold)"
    )
    parser.add_argument(
        "--out", default="thresholds.json", help="Output thresholds JSON"
    )
    parser.add_argument(
        "--base-thresholds",
        default=None,
        help="Existing thresholds.json to update (preserves non-calibrated keys)",
    )
    args = parser.parse_args()

    # Import scorer
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_hybrid import score_coco

    with open(args.gt) as f:
        gt_data = json.load(f)
    with open(args.pred) as f:
        all_preds = json.load(f)

    print(f"Loaded {len(all_preds)} predictions, {len(gt_data['annotations'])} GT boxes")
    print(f"Score range: [{min(p['score'] for p in all_preds):.4f}, "
          f"{max(p['score'] for p in all_preds):.4f}]")

    # --- Sweep confidence thresholds ---
    thresholds = np.concatenate([
        np.arange(0.001, 0.02, 0.002),
        np.arange(0.02, 0.10, 0.005),
        np.arange(0.10, 0.50, 0.02),
    ])

    best_score = -1.0
    best_thresh = 0.01
    best_result = None

    print(f"\nSweeping {len(thresholds)} confidence thresholds...")
    print(f"{'thresh':>8s}  {'n_pred':>7s}  {'det_mAP':>8s}  {'cls_mAP':>8s}  {'hybrid':>8s}")
    print("-" * 50)

    for thresh in thresholds:
        filtered = [p for p in all_preds if p["score"] >= thresh]
        if not filtered:
            continue

        result = score_coco(gt_data, filtered)
        marker = " <-- BEST" if result["hybrid_score"] > best_score else ""

        if result["hybrid_score"] > best_score:
            best_score = result["hybrid_score"]
            best_thresh = float(thresh)
            best_result = result

        print(
            f"{thresh:8.4f}  {len(filtered):7d}  "
            f"{result['detection_mAP']:8.4f}  "
            f"{result['classification_mAP']:8.4f}  "
            f"{result['hybrid_score']:8.4f}{marker}"
        )

    print(f"\n{'=' * 50}")
    print(f"Best confidence threshold: {best_thresh:.4f}")
    print(f"Best hybrid score:         {best_score:.4f}")
    print(f"  Detection mAP:           {best_result['detection_mAP']:.4f}")
    print(f"  Classification mAP:      {best_result['classification_mAP']:.4f}")
    print(f"  Predictions kept:        {best_result['total_predictions']}")
    print(f"{'=' * 50}")

    # --- Build output thresholds ---
    out_thresholds = {}
    if args.base_thresholds and Path(args.base_thresholds).exists():
        with open(args.base_thresholds) as f:
            out_thresholds = json.load(f)
        print(f"\nMerging with base thresholds from {args.base_thresholds}")

    out_thresholds["final_conf_threshold"] = round(best_thresh, 4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_thresholds, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
