import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Patch torch.load BEFORE importing ultralytics — torch 2.6 defaults
# weights_only=True which blocks unpickling ultralytics DetectionModel.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from io_utils import parse_image_id, get_image_files, xyxy_to_xywh, format_predictions  # noqa: E402
from infer_det import DetectorEnsemble  # noqa: E402
from fusion import fuse_detections  # noqa: E402


def find_weight_files(script_dir: Path) -> list[Path]:
    patterns = ["*.pt", "*.pth"]
    weights = []
    for pattern in patterns:
        weights.extend(script_dir.glob(pattern))
    weights.sort(key=lambda p: p.stat().st_size, reverse=True)
    return weights


def find_classifier_bundle(script_dir: Path) -> Path | None:
    """Find safetensors classifier bundle (trained or prototype-based)."""
    for ext in [".safetensors"]:
        matches = list(script_dir.glob(f"*classifier*{ext}")) + list(
            script_dir.glob(f"*bundle*{ext}")
        )
        if matches:
            return matches[0]
    return None


def load_thresholds(script_dir: Path) -> dict:
    p = script_dir / "thresholds.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start_time = time.time()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    script_dir = Path(__file__).parent

    weight_files = find_weight_files(script_dir)
    if not weight_files:
        raise FileNotFoundError(f"No .pt weight files found in {script_dir}")

    print(
        f"Found {len(weight_files)} detector weight(s): {[w.name for w in weight_files]}"
    )

    thresholds = load_thresholds(script_dir)
    detector = DetectorEnsemble(
        weight_paths=weight_files,
        imgsz=thresholds.get("imgsz", 1280),
        conf=thresholds.get("conf", 0.01),
        iou=thresholds.get("iou", 0.7),
        max_det=thresholds.get("max_det", 1000),
        augment=thresholds.get("augment", True),
    )

    # Load trained classifier
    classifier = None
    classifier_bundle = find_classifier_bundle(script_dir)
    if classifier_bundle:
        print(f"Loading trained classifier: {classifier_bundle.name}")
        from infer_cls import TrainedClassifier

        classifier = TrainedClassifier(
            bundle_path=classifier_bundle,
            cls_conf_threshold=thresholds.get("cls_conf_threshold", 0.15),
            cls_margin_threshold=thresholds.get("cls_margin_threshold", 0.05),
            yolo_conf_ceiling=thresholds.get("yolo_conf_ceiling", 0.95),
            score_floor=thresholds.get("score_floor", 0.03),
            yolo_implausible=thresholds.get("yolo_implausible", 0.10),
        )
    else:
        print("No classifier bundle found, using detector classes only")

    image_files = get_image_files(input_dir)
    print(f"Processing {len(image_files)} images...")

    all_predictions = []

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        image_id = parse_image_id(img_path)

        raw_dets = detector.run(img_path, img_w, img_h)

        if len(raw_dets) == 1 and len(raw_dets[0]["scores"]) > 0:
            fused = raw_dets[0]
            fused["class_confidence"] = np.ones(len(fused["scores"]))
        elif len(raw_dets) > 1:
            # Model weights: favor larger/better models (sorted by file size desc)
            wbf_weights = thresholds.get("wbf_weights", None)
            fused = fuse_detections(
                raw_dets,
                img_w,
                img_h,
                wbf_iou=thresholds.get("wbf_iou", 0.50),
                weights=wbf_weights,
            )
        else:
            continue

        if len(fused["scores"]) == 0:
            continue

        classes = fused["classes"]
        elapsed = time.time() - start_time
        img_idx = image_files.index(img_path)
        remaining_images = len(image_files) - img_idx - 1

        # Time-budget: disable classifier if we'll exceed 280s
        use_classifier = classifier is not None and elapsed < 230
        if use_classifier:
            classes = classifier.reclassify(
                img,
                fused["boxes_xyxy"],
                fused["classes"],
                fused["class_confidence"],
                fused["scores"],
            )

        boxes_xywh = xyxy_to_xywh(fused["boxes_xyxy"])
        preds = format_predictions(image_id, boxes_xywh, fused["scores"], classes)
        all_predictions.extend(preds)

        # Log progress every 50 images
        if (img_idx + 1) % 50 == 0 or elapsed > 200:
            avg_time = elapsed / (img_idx + 1)
            est_total = avg_time * len(image_files)
            print(
                f"  [{img_idx + 1}/{len(image_files)}] {elapsed:.0f}s elapsed, "
                f"~{est_total:.0f}s estimated total, {remaining_images} remaining"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)

    elapsed = time.time() - start_time
    print(
        f"Done: {len(all_predictions)} predictions for {len(image_files)} images in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
