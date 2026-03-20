"""YOLOv8 detector ensemble with full-image and tile-based inference.

Designed for dense grocery shelf scenes where small products need
high-resolution coverage. Tile inference runs on images exceeding
a size threshold, mapping tile-local detections back to full-image
coordinates.
"""

import numpy as np
from pathlib import Path

import torch
from ultralytics import YOLO

TILE_THRESHOLD = 4000  # Only tile very large images — ensemble+TTA covers most gains
TILE_OVERLAP = 0.15

DEFAULT_CONF = 0.01
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 1000
DEFAULT_IMGSZ = 1280


class DetectorEnsemble:
    def __init__(
        self,
        weight_paths: list[Path],
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        augment: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for wp in weight_paths:
            model = YOLO(str(wp))
            self.models.append(model)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.augment = augment

    def predict_single(self, model: YOLO, source, augment: bool | None = None) -> dict:
        """Run one model on a single image/array.

        source: file path (Path/str) or numpy array (HWC, RGB).
        augment: TTA (test-time augmentation). Defaults to self.augment.
        """
        src = str(source) if isinstance(source, Path) else source
        use_augment = augment if augment is not None else self.augment
        results = model(
            src,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            agnostic_nms=True,
            half=self.device == "cuda",
            verbose=False,
            augment=use_augment,
        )
        return self._extract_results(results[0])

    def predict_tiled(
        self,
        model: YOLO,
        img_path: Path,
        img_w: int,
        img_h: int,
        augment: bool = False,
    ) -> dict:
        """Run detector on overlapping tiles, remap to full-image coords.

        Tile grid adapts to aspect ratio: 2x2 for normal, 3x2 or 2x3 for
        very wide/tall images.  Overlap is a fraction of the *tile* size,
        not the full image, ensuring consistent coverage.

        Note: TTA is disabled by default for tiles to save time — the
        full-image pass with TTA covers most gains.
        """
        from PIL import Image

        img = Image.open(img_path).convert("RGB")

        # Adaptive grid
        if img_w > img_h * 1.5:
            cols, rows = 3, 2
        elif img_h > img_w * 1.5:
            cols, rows = 2, 3
        else:
            cols, rows = 2, 2

        # Tile size: tile_w = img_w / (cols - (cols-1)*overlap)
        # Step size: step = tile_w * (1 - overlap)
        tile_w = img_w / (cols - (cols - 1) * TILE_OVERLAP)
        tile_h = img_h / (rows - (rows - 1) * TILE_OVERLAP)
        step_x = tile_w * (1 - TILE_OVERLAP)
        step_y = tile_h * (1 - TILE_OVERLAP)

        all_boxes = []
        all_scores = []
        all_classes = []

        for r in range(rows):
            for c in range(cols):
                x0 = int(round(c * step_x))
                y0 = int(round(r * step_y))
                x1 = min(int(round(x0 + tile_w)), img_w)
                y1 = min(int(round(y0 + tile_h)), img_h)

                tile = img.crop((x0, y0, x1, y1))
                tile_np = np.array(tile)

                det = self.predict_single(model, tile_np, augment=augment)
                if len(det["boxes_xyxy"]) == 0:
                    continue

                offset_boxes = det["boxes_xyxy"].copy()
                offset_boxes[:, [0, 2]] += x0
                offset_boxes[:, [1, 3]] += y0

                all_boxes.append(offset_boxes)
                all_scores.append(det["scores"])
                all_classes.append(det["classes"])

        if not all_boxes:
            return {
                "boxes_xyxy": np.empty((0, 4)),
                "scores": np.empty(0),
                "classes": np.empty(0, dtype=int),
            }

        return {
            "boxes_xyxy": np.vstack(all_boxes),
            "scores": np.concatenate(all_scores),
            "classes": np.concatenate(all_classes).astype(int),
        }

    def _extract_results(self, result) -> dict:
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes_xyxy": np.empty((0, 4)),
                "scores": np.empty(0),
                "classes": np.empty(0, dtype=int),
            }

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return {"boxes_xyxy": boxes, "scores": scores, "classes": classes}

    def run(self, img_path: Path, img_w: int, img_h: int) -> list[dict]:
        """Run all detectors (full-image with TTA + tiles) on one image.

        Returns list of detection dicts, one per pass.
        """
        all_dets = []
        long_side = max(img_w, img_h)

        for model in self.models:
            full_det = self.predict_single(model, img_path)
            all_dets.append(full_det)

        # Tile pass on first model for large images (no TTA on tiles — too slow)
        if long_side > TILE_THRESHOLD and len(self.models) > 0:
            tile_det = self.predict_tiled(
                self.models[0], img_path, img_w, img_h, augment=False
            )
            if len(tile_det["scores"]) > 0:
                all_dets.append(tile_det)

        return all_dets
