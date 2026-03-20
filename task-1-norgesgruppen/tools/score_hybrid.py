"""Local scorer matching competition metric:
    score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

Detection mAP:  IoU >= 0.5, category ignored (all mapped to class 1).
Classification mAP:  IoU >= 0.5 AND category_id must match ground truth.

Both use standard COCO-style per-category mAP (101-point interpolation)
via pycocotools, which is the de-facto standard for mAP computation.

Usage:
    python tools/score_hybrid.py --gt annotations.json --pred predictions.json
    python tools/score_hybrid.py --gt annotations.json --pred predictions.json --per-class
"""

import argparse
import contextlib
import copy
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# pycocotools-based scoring (primary)
# ------------------------------------------------------------------


def _make_coco(gt_dict):
    """Create a pycocotools COCO object from an annotation dict."""
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _coco_map(coco_gt, predictions, iou_thr=0.5):
    """Compute mAP@iou_thr using pycocotools COCOeval."""
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = np.array([iou_thr])
    ev.params.maxDets = [1, 100, 500]  # pycocotools expects 3 entries

    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # stats[1] = AP @ IoU=iou_thr, maxDets=500
    return float(ev.stats[1])


def score_coco(gt_data: dict, predictions: list) -> dict:
    """Compute hybrid score using pycocotools (standard COCO mAP)."""

    # --- Detection mAP (class-agnostic) ---
    det_gt = {
        "images": gt_data["images"],
        "categories": [{"id": 1, "name": "product", "supercategory": "product"}],
        "annotations": [
            {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": 1,
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            }
            for ann in gt_data["annotations"]
        ],
    }
    det_preds = [{**p, "category_id": 1} for p in predictions]
    det_map = _coco_map(_make_coco(det_gt), det_preds)

    # --- Classification mAP (per-category) ---
    cls_map = _coco_map(_make_coco(gt_data), predictions)

    hybrid = 0.7 * det_map + 0.3 * cls_map
    return {
        "detection_mAP": round(det_map, 4),
        "classification_mAP": round(cls_map, 4),
        "hybrid_score": round(hybrid, 4),
        "total_gt_boxes": len(gt_data["annotations"]),
        "total_predictions": len(predictions),
    }


# ------------------------------------------------------------------
# Per-class breakdown
# ------------------------------------------------------------------


def per_class_breakdown(gt_data: dict, predictions: list) -> list[dict]:
    """Per-category AP breakdown for diagnostics."""
    from pycocotools.cocoeval import COCOeval

    coco_gt = _make_coco(gt_data)
    if not predictions:
        return []

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [500]

    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()

    cat_ids = sorted(coco_gt.getCatIds())
    cat_names = {c["id"]: c["name"] for c in gt_data["categories"]}
    gt_counts = Counter(a["category_id"] for a in gt_data["annotations"])
    pred_counts = Counter(p["category_id"] for p in predictions)

    rows = []
    # precision shape: [T, R, K, A, M] — T=iou thresholds, R=recall, K=cats, A=area, M=maxDets
    precision = ev.eval["precision"]  # [1, 101, n_cats, 4, 1]
    for ki, cat_id in enumerate(ev.params.catIds):
        p = precision[
            0, :, ki, 0, 0
        ]  # IoU=0.5, all recall levels, all areas, maxDet=500
        ap = float(np.mean(p[p > -1])) if (p > -1).any() else 0.0
        rows.append(
            {
                "category_id": cat_id,
                "name": cat_names.get(cat_id, f"class_{cat_id}"),
                "gt_count": gt_counts.get(cat_id, 0),
                "pred_count": pred_counts.get(cat_id, 0),
                "ap": round(ap, 4),
            }
        )

    rows.sort(key=lambda r: r["ap"])
    return rows


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def score(gt_path: str, pred_path: str) -> dict:
    """Convenience wrapper loading from file paths."""
    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        predictions = json.load(f)
    return score_coco(gt_data, predictions)


def main():
    parser = argparse.ArgumentParser(description="Local hybrid scorer")
    parser.add_argument(
        "--gt", required=True, help="Ground truth annotations.json (COCO format)"
    )
    parser.add_argument("--pred", required=True, help="Predictions JSON file")
    parser.add_argument(
        "--per-class", action="store_true", help="Print per-category AP breakdown"
    )
    args = parser.parse_args()

    with open(args.gt) as f:
        gt_data = json.load(f)
    with open(args.pred) as f:
        predictions = json.load(f)

    results = score_coco(gt_data, predictions)

    print(f"\n{'=' * 55}")
    print(f"  Detection mAP@0.5:       {results['detection_mAP']:.4f}")
    print(f"  Classification mAP@0.5:  {results['classification_mAP']:.4f}")
    print(f"  Hybrid Score:            {results['hybrid_score']:.4f}")
    print(
        f"    (0.7 x {results['detection_mAP']:.4f}"
        f" + 0.3 x {results['classification_mAP']:.4f})"
    )
    print(f"{'=' * 55}")
    print(f"  GT boxes:     {results['total_gt_boxes']}")
    print(f"  Predictions:  {results['total_predictions']}")

    if args.per_class:
        rows = per_class_breakdown(gt_data, predictions)
        print(f"\n{'=' * 55}")
        print(f"  Per-class AP (worst → best)")
        print(f"{'=' * 55}")
        for r in rows:
            bar = "#" * int(r["ap"] * 30)
            print(
                f"  [{r['category_id']:>3d}] {r['name'][:35]:<35s}  "
                f"gt={r['gt_count']:>4d}  pred={r['pred_count']:>4d}  "
                f"AP={r['ap']:.3f}  {bar}"
            )

        # Summary buckets
        zero_ap = [r for r in rows if r["ap"] == 0]
        low_ap = [r for r in rows if 0 < r["ap"] < 0.3]
        med_ap = [r for r in rows if 0.3 <= r["ap"] < 0.7]
        high_ap = [r for r in rows if r["ap"] >= 0.7]
        print(f"\n  AP=0: {len(zero_ap)} categories")
        print(f"  AP<0.3: {len(low_ap)} categories")
        print(f"  AP 0.3-0.7: {len(med_ap)} categories")
        print(f"  AP>=0.7: {len(high_ap)} categories")


if __name__ == "__main__":
    main()
