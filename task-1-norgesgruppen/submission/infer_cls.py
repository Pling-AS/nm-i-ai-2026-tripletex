"""DINOv2-based trained crop classifier with Prototype Retrieval.

Replaces prototype matching with a fine-tuned classification head that
covers ALL 356 categories.  Classifies every detection crop and uses the
trained prediction when confident, falling back to YOLO's class otherwise.

Also uses 1-NN prototype retrieval as a strong signal for rare classes.
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

# Fallback thresholds
DEFAULT_CLS_CONF_THRESHOLD = 0.15
DEFAULT_CLS_MARGIN_THRESHOLD = 0.05
DEFAULT_YOLO_CONF_CEILING = 1.1  # DISABLED (was 0.95)
DEFAULT_SCORE_FLOOR = 0.01  # Lowered (was 0.03)
DEFAULT_YOLO_IMPLAUSIBLE = 0.10
PROTO_SIM_THRESHOLD = 0.85  # Trust retrieval if similarity > 0.85


class TrainedClassifier:
    def __init__(
        self,
        bundle_path: Path,
        cls_conf_threshold: float = DEFAULT_CLS_CONF_THRESHOLD,
        cls_margin_threshold: float = DEFAULT_CLS_MARGIN_THRESHOLD,
        yolo_conf_ceiling: float = DEFAULT_YOLO_CONF_CEILING,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        yolo_implausible: float = DEFAULT_YOLO_IMPLAUSIBLE,
        temperature: float = 1.0,
        tta_flip: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cls_conf_threshold = cls_conf_threshold
        self.cls_margin_threshold = cls_margin_threshold
        self.yolo_conf_ceiling = yolo_conf_ceiling
        self.yolo_implausible = yolo_implausible
        self.score_floor = score_floor
        self.temperature = temperature
        self.tta_flip = tta_flip

        from safetensors.torch import load_file

        bundle = load_file(str(bundle_path), device="cpu")

        # Extract model state_dict
        model_state = {
            k.replace("model.", "", 1): v
            for k, v in bundle.items()
            if k.startswith("model.")
        }

        # Extract prototypes if available
        self.prototypes = None
        print("Prototypes disabled.")

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

        # Build transform
        data_cfg = timm.data.resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, CROP_SIZE, CROP_SIZE)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)

        tta_str = "+hflip" if self.tta_flip else ""
        temp_str = f", T={self.temperature}" if self.temperature != 1.0 else ""
        print(
            f"Trained classifier loaded: {nc} classes, device={self.device}{tta_str}{temp_str}"
        )

    def classify_batch(self, crops: list, yolo_classes: np.ndarray = None) -> tuple:
        """Classify a batch of PIL crops.

        Returns (predicted_classes, confidences, margins, yolo_class_probs, embeddings)
        """
        if not crops:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), None

        batch_size = 64
        all_classes = []
        all_confs = []
        all_margins = []
        all_yolo_probs = []
        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = torch.stack([self.transform(c) for c in batch_crops])
            tensors = tensors.to(self.device)
            if self.device == "cuda":
                tensors = tensors.half()

            with torch.no_grad():
                # Get features first
                features = self.model.forward_features(tensors)

                # Get logits
                logits = self.model.forward_head(features)

                # Get embeddings (pre_logits)
                emb = self.model.forward_head(features, pre_logits=True)
                emb = F.normalize(emb, dim=-1)

                # Horizontal flip TTA
                if self.tta_flip:
                    tensors_flip = torch.flip(tensors, dims=[3])
                    features_flip = self.model.forward_features(tensors_flip)
                    logits_flip = self.model.forward_head(features_flip)
                    emb_flip = self.model.forward_head(features_flip, pre_logits=True)
                    emb_flip = F.normalize(emb_flip, dim=-1)

                    logits = (logits + logits_flip) / 2
                    emb = (emb + emb_flip) / 2
                    emb = F.normalize(emb, dim=-1)  # Re-normalize after average

                if self.temperature != 1.0:
                    logits = logits / self.temperature

                probs = F.softmax(logits, dim=-1)

            # Top-2 for confidence and margin
            top2_vals, top2_idx = probs.topk(2, dim=-1)
            pred_classes = top2_idx[:, 0].cpu().numpy()
            pred_confs = top2_vals[:, 0].cpu().numpy().astype(np.float32)
            margins = (
                (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy().astype(np.float32)
            )

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
            all_embeddings.append(emb.cpu())

        yolo_probs_out = (
            np.concatenate(all_yolo_probs) if all_yolo_probs else np.empty(0)
        )
        embeddings_out = torch.cat(all_embeddings, dim=0) if all_embeddings else None

        return (
            np.concatenate(all_classes),
            np.concatenate(all_confs),
            np.concatenate(all_margins),
            yolo_probs_out,
            embeddings_out,
        )

    def reclassify(
        self,
        img: Image.Image,
        boxes_xyxy: np.ndarray,
        yolo_classes: np.ndarray,
        yolo_class_confidences: np.ndarray,
        detection_scores: np.ndarray,
    ) -> np.ndarray:
        new_classes = yolo_classes.copy()
        if len(boxes_xyxy) == 0:
            return new_classes

        img_w, img_h = img.size
        classify_indices = []
        crops = []
        classify_yolo_classes = []

        for i in range(len(boxes_xyxy)):
            if detection_scores[i] < self.score_floor:
                continue

            classify_indices.append(i)
            crop = self._crop_with_padding(img, boxes_xyxy[i], img_w, img_h)
            crops.append(crop)
            classify_yolo_classes.append(yolo_classes[i])

        if not crops:
            return new_classes

        classify_yolo_classes = np.array(classify_yolo_classes, dtype=int)

        pred_classes, pred_confs, pred_margins, yolo_probs, embeddings = (
            self.classify_batch(crops, classify_yolo_classes)
        )

        overrides = 0
        retrieval_overrides = 0

        # Retrieval Logic
        best_sims = None
        best_proto_idxs = None
        if self.prototypes is not None and embeddings is not None:
            # Sim matrix: (N_crops, 356) = (N_crops, D) @ (D, 356)
            # Prototypes shape (356, D)
            sims = torch.mm(embeddings.to(self.device), self.prototypes.t())
            best_sims_val, best_proto_idxs_val = sims.max(dim=1)
            best_sims = best_sims_val.cpu().numpy()
            best_proto_idxs = best_proto_idxs_val.cpu().numpy()

        for idx, box_idx in enumerate(classify_indices):
            cls_confident = (
                pred_confs[idx] >= self.cls_conf_threshold
                and pred_margins[idx] >= self.cls_margin_threshold
            )
            yolo_implausible = yolo_probs[idx] < self.yolo_implausible

            # 1. Trust the fine-tuned head first if it's confident and YOLO is implausible
            if cls_confident and yolo_implausible:
                if new_classes[box_idx] != pred_classes[idx]:
                    new_classes[box_idx] = pred_classes[idx]
                    overrides += 1
                continue

            # 2. Fallback to retrieval ONLY if head is not confident, but retrieval is VERY confident
            if best_sims is not None:
                sim = best_sims[idx]
                proto_cls = best_proto_idxs[idx]

                if sim > PROTO_SIM_THRESHOLD and yolo_implausible:
                    if new_classes[box_idx] != proto_cls:
                        new_classes[box_idx] = proto_cls
                        retrieval_overrides += 1

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
