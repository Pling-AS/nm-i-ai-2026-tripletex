"""DINOv2-based trained crop classifier.

Replaces prototype matching with a fine-tuned classification head that
covers ALL 356 categories.  Classifies every detection crop and uses the
trained prediction when confident, falling back to YOLO's class otherwise.

The model bundle (safetensors) contains the full DINOv2 ViT-S weights
with a trained classification head (model.head).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NUM_CLASSES = 356
CROP_PAD_RATIO = 0.10
CROP_SIZE = 224

# Fallback thresholds: when to trust the classifier over YOLO
DEFAULT_CLS_CONF_THRESHOLD = 0.15  # Min classifier confidence to override YOLO
DEFAULT_CLS_MARGIN_THRESHOLD = 0.05  # Min top1-top2 margin
DEFAULT_YOLO_CONF_CEILING = 0.95  # Don't reclassify ultra-confident YOLO preds
DEFAULT_SCORE_FLOOR = 0.03  # Skip reranking for very low-score detections
DEFAULT_YOLO_IMPLAUSIBLE = (
    0.10  # Classifier prob for YOLO class below this = implausible
)


class TrainedClassifier:
    def __init__(
        self,
        bundle_path: Path,
        cls_conf_threshold: float = DEFAULT_CLS_CONF_THRESHOLD,
        cls_margin_threshold: float = DEFAULT_CLS_MARGIN_THRESHOLD,
        yolo_conf_ceiling: float = DEFAULT_YOLO_CONF_CEILING,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        yolo_implausible: float = DEFAULT_YOLO_IMPLAUSIBLE,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cls_conf_threshold = cls_conf_threshold
        self.cls_margin_threshold = cls_margin_threshold
        self.yolo_conf_ceiling = yolo_conf_ceiling
        self.yolo_implausible = yolo_implausible
        self.score_floor = score_floor

        from safetensors.torch import load_file

        bundle = load_file(str(bundle_path), device="cpu")

        # Extract model state_dict (keys prefixed with "model.")
        model_state = {
            k.replace("model.", "", 1): v
            for k, v in bundle.items()
            if k.startswith("model.")
        }

        if not model_state:
            raise ValueError(f"No model weights found in {bundle_path}")

        import timm

        # Detect num_classes from head weight shape
        head_key = "head.weight"
        if head_key in model_state:
            nc = model_state[head_key].shape[0]
        else:
            nc = NUM_CLASSES

        model_name = "vit_small_patch14_dinov2.lvd142m"
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=nc, img_size=CROP_SIZE
        )
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device).eval()

        if self.device == "cuda":
            self.model = self.model.half()

        # Build transform — override input_size since model's default_cfg
        # still reports native DINOv2 size (518) even with img_size=224
        data_cfg = timm.data.resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, CROP_SIZE, CROP_SIZE)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)

        print(f"Trained classifier loaded: {nc} classes, device={self.device}")

    def classify_batch(self, crops: list, yolo_classes: np.ndarray = None) -> tuple:
        """Classify a batch of PIL crops.

        Returns (predicted_classes, confidences, margins, yolo_class_probs)
        as numpy arrays.  yolo_class_probs[i] = classifier probability at
        YOLO's predicted class for detection i (None if yolo_classes not given).
        """
        if not crops:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0)

        batch_size = 64
        all_classes = []
        all_confs = []
        all_margins = []
        all_yolo_probs = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = torch.stack([self.transform(c) for c in batch_crops])
            tensors = tensors.to(self.device)
            if self.device == "cuda":
                tensors = tensors.half()

            with torch.no_grad():
                logits = self.model(tensors)
                probs = F.softmax(logits, dim=-1)

            # Top-2 for confidence and margin
            top2_vals, top2_idx = probs.topk(2, dim=-1)
            pred_classes = top2_idx[:, 0].cpu().numpy()
            pred_confs = top2_vals[:, 0].cpu().numpy().astype(np.float32)
            margins = (
                (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy().astype(np.float32)
            )

            # Extract classifier's probability for YOLO's class
            if yolo_classes is not None:
                batch_yolo = yolo_classes[i : i + len(batch_crops)]
                yolo_idx = torch.tensor(
                    batch_yolo, dtype=torch.long, device=probs.device
                )
                yolo_probs = probs.gather(1, yolo_idx.unsqueeze(1)).squeeze(1)
                all_yolo_probs.append(yolo_probs.cpu().numpy().astype(np.float32))

            all_classes.append(pred_classes)
            all_confs.append(pred_confs)
            all_margins.append(margins)

        yolo_probs_out = (
            np.concatenate(all_yolo_probs) if all_yolo_probs else np.empty(0)
        )
        return (
            np.concatenate(all_classes),
            np.concatenate(all_confs),
            np.concatenate(all_margins),
            yolo_probs_out,
        )

    def reclassify(
        self,
        img: Image.Image,
        boxes_xyxy: np.ndarray,
        yolo_classes: np.ndarray,
        yolo_class_confidences: np.ndarray,
        detection_scores: np.ndarray,
    ) -> np.ndarray:
        """Reclassify detections using probability-aware blending.

        For each detection, the classifier produces a full softmax.  We check
        what probability the classifier assigns to YOLO's class:

          - If classifier agrees with YOLO (top-1 == YOLO class): keep YOLO
          - If classifier disagrees AND thinks YOLO's class is implausible
            (low prob for YOLO class) AND is confident in its own pick:
            override with classifier's class
          - Otherwise: keep YOLO (conservative)

        This avoids overriding correct YOLO predictions while still catching
        cases where YOLO is clearly wrong.
        """
        new_classes = yolo_classes.copy()
        if len(boxes_xyxy) == 0:
            return new_classes

        img_w, img_h = img.size

        # Determine which detections to classify
        classify_indices = []
        crops = []
        classify_yolo_classes = []

        for i in range(len(boxes_xyxy)):
            # Skip very low score detections
            if detection_scores[i] < self.score_floor:
                continue
            # Skip ultra-confident YOLO predictions
            if yolo_class_confidences[i] >= self.yolo_conf_ceiling:
                continue

            classify_indices.append(i)
            crop = self._crop_with_padding(img, boxes_xyxy[i], img_w, img_h)
            crops.append(crop)
            classify_yolo_classes.append(yolo_classes[i])

        if not crops:
            return new_classes

        classify_yolo_classes = np.array(classify_yolo_classes, dtype=int)

        # Batch classify — also get probability at YOLO's class position
        pred_classes, pred_confs, pred_margins, yolo_probs = self.classify_batch(
            crops, classify_yolo_classes
        )

        # Probability-aware override decision
        overrides = 0
        for idx, box_idx in enumerate(classify_indices):
            # If classifier agrees with YOLO, keep YOLO
            if pred_classes[idx] == yolo_classes[box_idx]:
                continue

            # Classifier disagrees — check both conditions:
            # 1. Classifier is confident in its own prediction
            # 2. Classifier thinks YOLO's class is implausible
            cls_confident = (
                pred_confs[idx] >= self.cls_conf_threshold
                and pred_margins[idx] >= self.cls_margin_threshold
            )
            yolo_implausible = yolo_probs[idx] < self.yolo_implausible

            if cls_confident and yolo_implausible:
                new_classes[box_idx] = pred_classes[idx]
                overrides += 1

        return new_classes

    def _crop_with_padding(
        self, img: Image.Image, box: np.ndarray, img_w: int, img_h: int
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * CROP_PAD_RATIO
        pad_y = bh * CROP_PAD_RATIO

        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))

        crop = img.crop((cx1, cy1, cx2, cy2))
        return crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
