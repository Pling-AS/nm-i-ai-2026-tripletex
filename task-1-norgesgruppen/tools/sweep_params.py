"""Parameter sweep for post-processing optimization.

Phase 1: Run YOLO inference once on val images, cache raw detections.
Phase 2: Sweep WBF + classifier parameters, score each config.

Usage:
    # Phase 1: Generate cached detections (slow, ~5min on CPU)
    python tools/sweep_params.py --phase1 \
        --weights /tmp/build_submission/detector_b.pt /tmp/build_submission/detector_a.pt \
        --classifier /tmp/build_submission/classifier_v2.safetensors

    # Phase 2: Sweep parameters (fast, seconds)
    python tools/sweep_params.py --phase2

    # Both phases:
    python tools/sweep_params.py --phase1 --phase2 \
        --weights /tmp/build_submission/detector_b.pt /tmp/build_submission/detector_a.pt \
        --classifier /tmp/build_submission/classifier_v2.safetensors
"""

import argparse
import contextlib
import io
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent.parent
VAL_DIR = ROOT / "work" / "split" / "images" / "val"
ANNOTATIONS = ROOT / "training_data" / "NM_NGD_coco_dataset" / "annotations.json"
CACHE_DIR = ROOT / "work" / "sweep_cache"


def phase1_cache_detections(weight_paths, classifier_path, imgsz=1280):
    """Run YOLO + classifier inference once, cache everything."""
    import torch

    # Monkey-patch torch.load
    _orig = torch.load

    def _patched(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig(*a, **kw)

    torch.load = _patched

    import sys

    sys.path.insert(0, str(ROOT / "submission"))
    from infer_det import DetectorEnsemble
    from infer_cls import TrainedClassifier
    from PIL import Image

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Init detector
    detector = DetectorEnsemble(
        weight_paths=[Path(w) for w in weight_paths],
        imgsz=imgsz,
        conf=0.001,  # Very low conf to capture everything
        iou=0.9,  # Very high IoU NMS to keep everything
        max_det=1500,
        augment=True,
    )

    # Init classifier
    classifier = TrainedClassifier(
        bundle_path=Path(classifier_path),
        cls_conf_threshold=0.01,  # Very low — we'll filter in phase 2
        cls_margin_threshold=0.0,
        yolo_conf_ceiling=1.0,  # Classify everything
        score_floor=0.0,
        yolo_implausible=1.0,  # Classify everything
    )

    val_images = sorted(VAL_DIR.glob("*.jpg"))
    print(f"Phase 1: Running inference on {len(val_images)} val images...")

    all_cached = {}
    for idx, img_path in enumerate(val_images):
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])

        # Run detector — get raw per-model detections
        raw_dets = detector.run(img_path, img_w, img_h)

        # For each detection in each raw_det, also run classifier
        # We need to classify ALL detections to have the classifier data cached
        # Merge all detections first for classification
        all_boxes = []
        all_scores = []
        all_classes = []
        for d in raw_dets:
            if len(d["scores"]) > 0:
                all_boxes.append(d["boxes_xyxy"])
                all_scores.append(d["scores"])
                all_classes.append(d["classes"])

        if all_boxes:
            merged_boxes = np.vstack(all_boxes)
            merged_scores = np.concatenate(all_scores)
            merged_classes = np.concatenate(all_classes).astype(int)

            # Run classifier on all merged detections
            crops = []
            for box in merged_boxes:
                crop = classifier._crop_with_padding(img, box, img_w, img_h)
                crops.append(crop)

            cls_pred_classes, cls_confs, cls_margins, cls_yolo_probs = (
                classifier.classify_batch(crops, merged_classes)
            )
        else:
            merged_boxes = np.empty((0, 4))
            merged_scores = np.empty(0)
            merged_classes = np.empty(0, dtype=int)
            cls_pred_classes = np.empty(0, dtype=int)
            cls_confs = np.empty(0)
            cls_margins = np.empty(0)
            cls_yolo_probs = np.empty(0)

        all_cached[image_id] = {
            "img_w": img_w,
            "img_h": img_h,
            "raw_dets": raw_dets,  # Per-model detections for WBF
            "merged_boxes": merged_boxes,
            "merged_scores": merged_scores,
            "merged_classes": merged_classes,
            "cls_pred_classes": cls_pred_classes,
            "cls_confs": cls_confs,
            "cls_margins": cls_margins,
            "cls_yolo_probs": cls_yolo_probs,
        }

        if (idx + 1) % 10 == 0:
            print(
                f"  [{idx + 1}/{len(val_images)}] {image_id}: "
                f"{sum(len(d['scores']) for d in raw_dets)} raw dets"
            )

    cache_path = CACHE_DIR / "val_detections.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(all_cached, f)
    print(f"Phase 1 complete. Cached {len(all_cached)} images to {cache_path}")
    return all_cached


def load_cache():
    cache_path = CACHE_DIR / "val_detections.pkl"
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_val_gt():
    """Load GT annotations filtered to val images only."""
    with open(ANNOTATIONS) as f:
        gt = json.load(f)

    val_images = sorted(VAL_DIR.glob("*.jpg"))
    val_ids = set(int(p.stem.split("_")[-1]) for p in val_images)

    gt_val = {
        "images": [img for img in gt["images"] if img["id"] in val_ids],
        "categories": gt["categories"],
        "annotations": [ann for ann in gt["annotations"] if ann["image_id"] in val_ids],
    }
    return gt_val


def run_postprocess(cached, config):
    """Run post-processing with given config on cached detections.

    Returns predictions list in COCO format.
    """
    import sys

    sys.path.insert(0, str(ROOT / "submission"))
    from fusion import fuse_detections
    from io_utils import xyxy_to_xywh

    conf_thresh = config.get("conf", 0.01)
    wbf_iou = config.get("wbf_iou", 0.50)
    wbf_weights = config.get("wbf_weights", None)
    cls_conf_threshold = config.get("cls_conf_threshold", 0.50)
    cls_margin_threshold = config.get("cls_margin_threshold", 0.15)
    yolo_conf_ceiling = config.get("yolo_conf_ceiling", 0.70)
    yolo_implausible = config.get("yolo_implausible", 0.08)
    score_floor = config.get("score_floor", 0.03)
    use_classifier = config.get("use_classifier", True)

    all_preds = []

    for image_id, data in cached.items():
        raw_dets = data["raw_dets"]
        img_w, img_h = data["img_w"], data["img_h"]

        # Filter raw detections by conf threshold
        filtered_dets = []
        for d in raw_dets:
            mask = d["scores"] >= conf_thresh
            if mask.any():
                filtered_dets.append(
                    {
                        "boxes_xyxy": d["boxes_xyxy"][mask],
                        "scores": d["scores"][mask],
                        "classes": d["classes"][mask],
                    }
                )
            else:
                filtered_dets.append(
                    {
                        "boxes_xyxy": np.empty((0, 4)),
                        "scores": np.empty(0),
                        "classes": np.empty(0, dtype=int),
                    }
                )

        # Fuse
        if len(filtered_dets) == 1 and len(filtered_dets[0]["scores"]) > 0:
            fused = filtered_dets[0]
            fused["class_confidence"] = np.ones(len(fused["scores"]))
        elif len(filtered_dets) > 1:
            fused = fuse_detections(
                filtered_dets,
                img_w,
                img_h,
                wbf_iou=wbf_iou,
                weights=wbf_weights,
            )
        else:
            continue

        if len(fused["scores"]) == 0:
            continue

        classes = fused["classes"].copy()

        # Classifier reclassification (using cached classifier outputs)
        if use_classifier:
            # We need to match fused boxes back to cached classifier outputs
            # Since fused boxes differ per config, we re-classify from cached merged data
            # Match each fused box to nearest cached box by IoU
            for fi in range(len(fused["boxes_xyxy"])):
                fbox = fused["boxes_xyxy"][fi]
                fused_score = fused["scores"][fi]
                fused_conf = fused.get(
                    "class_confidence", np.ones(len(fused["scores"]))
                )[fi]

                if fused_score < score_floor:
                    continue
                if fused_conf >= yolo_conf_ceiling:
                    continue

                # Find best matching cached detection
                if len(data["merged_boxes"]) == 0:
                    continue

                ious = _batch_iou(fbox, data["merged_boxes"])
                best_idx = np.argmax(ious)
                if ious[best_idx] < 0.3:
                    continue

                yolo_cls = classes[fi]
                cls_pred = data["cls_pred_classes"][best_idx]
                cls_conf = data["cls_confs"][best_idx]
                cls_margin = data["cls_margins"][best_idx]

                # Get yolo_prob for THIS fused class
                # We cached yolo_probs for the original merged_classes, not fused classes
                # Use cls_yolo_probs as approximation (same box, same YOLO class likely)
                cls_yolo_prob = (
                    data["cls_yolo_probs"][best_idx]
                    if len(data["cls_yolo_probs"]) > 0
                    else 1.0
                )

                if cls_pred == yolo_cls:
                    continue

                is_confident = (
                    cls_conf >= cls_conf_threshold
                    and cls_margin >= cls_margin_threshold
                )
                is_implausible = cls_yolo_prob < yolo_implausible

                if is_confident and is_implausible:
                    classes[fi] = cls_pred

        # Format predictions
        boxes_xywh = xyxy_to_xywh(fused["boxes_xyxy"])
        for i in range(len(boxes_xywh)):
            all_preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(classes[i]),
                    "bbox": [round(float(v), 1) for v in boxes_xywh[i]],
                    "score": round(float(fused["scores"][i]), 4),
                }
            )

    return all_preds


def _batch_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def score_predictions(gt_val, predictions):
    """Score predictions using the hybrid scorer."""
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from score_hybrid import score_coco

    return score_coco(gt_val, predictions)


def phase2_sweep(cached=None):
    """Sweep post-processing parameters."""
    if cached is None:
        cached = load_cache()

    gt_val = load_val_gt()
    print(
        f"Phase 2: Sweeping parameters on {len(cached)} val images, "
        f"{len(gt_val['annotations'])} GT annotations"
    )

    # Define parameter grid
    param_grid = {
        "wbf_iou": [0.40, 0.45, 0.50, 0.55, 0.60],
        "conf": [0.005, 0.01, 0.015],
        "cls_conf_threshold": [0.30, 0.40, 0.50, 0.60],
        "cls_margin_threshold": [0.10, 0.15, 0.20],
        "yolo_conf_ceiling": [0.60, 0.70, 0.80],
        "yolo_implausible": [0.05, 0.08, 0.12],
    }

    # Fixed params
    base_config = {
        "wbf_weights": [1.3, 1.0],
        "score_floor": 0.03,
        "use_classifier": True,
    }

    # First: sweep WBF + conf (most impactful, affects detection mAP)
    print("\n=== Phase 2a: WBF + conf sweep ===")
    best_det_config = None
    best_det_score = 0

    for wbf_iou, conf in itertools.product(param_grid["wbf_iou"], param_grid["conf"]):
        config = {
            **base_config,
            "wbf_iou": wbf_iou,
            "conf": conf,
            "cls_conf_threshold": 0.50,  # Default
            "cls_margin_threshold": 0.15,
            "yolo_conf_ceiling": 0.70,
            "yolo_implausible": 0.08,
        }
        preds = run_postprocess(cached, config)
        result = score_predictions(gt_val, preds)

        marker = ""
        if result["hybrid_score"] > best_det_score:
            best_det_score = result["hybrid_score"]
            best_det_config = config.copy()
            marker = " *** BEST ***"

        print(
            f"  wbf_iou={wbf_iou:.2f} conf={conf:.3f} → "
            f"det={result['detection_mAP']:.4f} cls={result['classification_mAP']:.4f} "
            f"hybrid={result['hybrid_score']:.4f}{marker}"
        )

    print(
        f"\nBest WBF+conf: wbf_iou={best_det_config['wbf_iou']}, "
        f"conf={best_det_config['conf']} → {best_det_score:.4f}"
    )

    # Second: sweep classifier thresholds with best WBF+conf
    print("\n=== Phase 2b: Classifier threshold sweep ===")
    best_cls_config = best_det_config.copy()
    best_cls_score = best_det_score

    for cls_conf, cls_margin, yolo_ceil, yolo_impl in itertools.product(
        param_grid["cls_conf_threshold"],
        param_grid["cls_margin_threshold"],
        param_grid["yolo_conf_ceiling"],
        param_grid["yolo_implausible"],
    ):
        config = {
            **best_det_config,
            "cls_conf_threshold": cls_conf,
            "cls_margin_threshold": cls_margin,
            "yolo_conf_ceiling": yolo_ceil,
            "yolo_implausible": yolo_impl,
        }
        preds = run_postprocess(cached, config)
        result = score_predictions(gt_val, preds)

        if result["hybrid_score"] > best_cls_score:
            best_cls_score = result["hybrid_score"]
            best_cls_config = config.copy()
            print(
                f"  NEW BEST: cls_conf={cls_conf:.2f} margin={cls_margin:.2f} "
                f"ceiling={yolo_ceil:.2f} implaus={yolo_impl:.2f} → "
                f"det={result['detection_mAP']:.4f} cls={result['classification_mAP']:.4f} "
                f"hybrid={result['hybrid_score']:.4f}"
            )

    # Also test: no classifier
    no_cls_config = {**best_det_config, "use_classifier": False}
    preds = run_postprocess(cached, no_cls_config)
    result = score_predictions(gt_val, preds)
    print(
        f"\n  NO CLASSIFIER: det={result['detection_mAP']:.4f} "
        f"cls={result['classification_mAP']:.4f} hybrid={result['hybrid_score']:.4f}"
    )

    # Third: sweep WBF weights
    print("\n=== Phase 2c: WBF weights sweep ===")
    best_w_config = best_cls_config.copy()
    best_w_score = best_cls_score

    for w1 in [1.0, 1.2, 1.3, 1.5, 2.0]:
        for w2 in [0.5, 0.8, 1.0]:
            config = {**best_cls_config, "wbf_weights": [w1, w2]}
            preds = run_postprocess(cached, config)
            result = score_predictions(gt_val, preds)

            if result["hybrid_score"] > best_w_score:
                best_w_score = result["hybrid_score"]
                best_w_config = config.copy()
                print(
                    f"  NEW BEST: weights=[{w1}, {w2}] → "
                    f"det={result['detection_mAP']:.4f} cls={result['classification_mAP']:.4f} "
                    f"hybrid={result['hybrid_score']:.4f}"
                )

    # Final summary
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    print("=" * 70)
    best = best_w_config
    preds = run_postprocess(cached, best)
    result = score_predictions(gt_val, preds)
    print(f"  detection_mAP:      {result['detection_mAP']:.4f}")
    print(f"  classification_mAP: {result['classification_mAP']:.4f}")
    print(f"  hybrid_score:       {result['hybrid_score']:.4f}")
    print(f"\nOptimal thresholds.json:")
    optimal = {
        "cls_conf_threshold": best["cls_conf_threshold"],
        "cls_margin_threshold": best["cls_margin_threshold"],
        "yolo_conf_ceiling": best["yolo_conf_ceiling"],
        "score_floor": best["score_floor"],
        "yolo_implausible": best["yolo_implausible"],
        "conf": best["conf"],
        "iou": 0.7,
        "max_det": 1000,
        "augment": True,
        "wbf_iou": best["wbf_iou"],
        "wbf_weights": best["wbf_weights"],
    }
    print(json.dumps(optimal, indent=4))

    # Save optimal config
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(optimal, f, indent=4)
    print(f"\nSaved to {CACHE_DIR / 'optimal_thresholds.json'}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument(
        "--phase1", action="store_true", help="Run YOLO + classifier inference (slow)"
    )
    parser.add_argument(
        "--phase2", action="store_true", help="Sweep post-processing params (fast)"
    )
    parser.add_argument(
        "--weights", nargs="+", help="Detector weight paths for phase 1"
    )
    parser.add_argument("--classifier", help="Classifier safetensors path for phase 1")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    if not args.phase1 and not args.phase2:
        print("Specify --phase1 and/or --phase2")
        return

    cached = None
    if args.phase1:
        if not args.weights or not args.classifier:
            print("Phase 1 requires --weights and --classifier")
            return
        cached = phase1_cache_detections(args.weights, args.classifier, args.imgsz)

    if args.phase2:
        phase2_sweep(cached)


if __name__ == "__main__":
    main()
