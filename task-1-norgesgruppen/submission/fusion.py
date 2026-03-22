"""Class-agnostic Weighted Boxes Fusion + weighted class vote assignment.

Why class-agnostic: Detection metric (70% of score) ignores class. Different
models may agree on box position but disagree on class. Class-agnostic fusion
gets the best boxes first, then assigns class by weighted vote.
"""

import numpy as np
import torch
import torchvision
from ensemble_boxes import weighted_boxes_fusion


DEFAULT_WBF_IOU = 0.50
DEFAULT_SKIP_BOX_THR = 0.001


def fuse_detections(
    detection_list: list[dict],
    img_w: int,
    img_h: int,
    wbf_iou: float = DEFAULT_WBF_IOU,
    skip_box_thr: float = DEFAULT_SKIP_BOX_THR,
    weights: list[float] | None = None,
) -> dict:
    """Fuse detections from multiple sources using class-agnostic WBF.

    Each element in detection_list has keys: boxes_xyxy, scores, classes.
    Returns fused dict with same keys plus class_scores for reranking.
    """
    if not detection_list:
        return _empty_result()

    # Build per-source weights
    if weights is None:
        weights = [1.0] * len(detection_list)
    while len(weights) < len(detection_list):
        weights.append(1.0)
    weights = weights[: len(detection_list)]

    # Filter empty detections AND their corresponding weights together
    non_empty = []
    active_weights = []
    for det, w in zip(detection_list, weights):
        if len(det["scores"]) > 0:
            non_empty.append(det)
            active_weights.append(w)

    if not non_empty:
        return _empty_result()

    boxes_list = []
    scores_list = []
    labels_list = []
    source_classes_list = []

    for det in non_empty:
        norm_boxes = det["boxes_xyxy"].copy()
        norm_boxes[:, [0, 2]] /= img_w
        norm_boxes[:, [1, 3]] /= img_h
        norm_boxes = np.clip(norm_boxes, 0.0, 1.0)

        boxes_list.append(norm_boxes.tolist())
        scores_list.append(det["scores"].tolist())
        labels_list.append([0] * len(det["scores"]))
        source_classes_list.append(det["classes"])

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=active_weights,
        iou_thr=wbf_iou,
        skip_box_thr=skip_box_thr,
    )

    if len(fused_boxes) == 0:
        return _empty_result()

    fused_xyxy = fused_boxes.copy()
    fused_xyxy[:, [0, 2]] *= img_w
    fused_xyxy[:, [1, 3]] *= img_h

    fused_classes, class_confidence = _assign_classes_by_vote(
        fused_xyxy,
        fused_scores,
        non_empty,
        source_classes_list,
        active_weights,
    )

    return {
        "boxes_xyxy": fused_xyxy,
        "scores": fused_scores,
        "classes": fused_classes,
        "class_confidence": class_confidence,
    }


def nms_dict(det: dict, iou_thr: float = 0.5) -> dict:
    if len(det["scores"]) == 0:
        if "class_confidence" not in det:
            det["class_confidence"] = np.empty(0)
        return det
    keep = torchvision.ops.nms(
        torch.tensor(det["boxes_xyxy"], dtype=torch.float32),
        torch.tensor(det["scores"], dtype=torch.float32),
        iou_thr
    ).numpy()
    
    res = {k: v[keep] for k, v in det.items()}
    if "class_confidence" not in res:
        res["class_confidence"] = np.ones_like(res["scores"])
    return res


def combine_full_and_tile(full_det: dict, tile_det: dict, iou_thr: float = 0.5) -> dict:
    """Combine full-image WBF detections with tile detections using NMS.
    This avoids the WBF score penalty where tile models miss large objects
    and full models miss small objects.
    """
    if len(full_det["scores"]) == 0:
        return nms_dict(tile_det, iou_thr)
    if len(tile_det["scores"]) == 0:
        if "class_confidence" not in full_det:
            full_det["class_confidence"] = np.ones_like(full_det["scores"])
        return full_det

    # First NMS the tile detections to remove overlaps from tiling
    tile_det = nms_dict(tile_det, iou_thr)

    # Slightly penalize tile scores so full-image boxes (which have better global context 
    # and no boundary truncation) win NMS when they overlap with tile boxes.
    tile_scores_penalized = tile_det["scores"] * 0.95

    # Concatenate
    boxes = np.vstack([full_det["boxes_xyxy"], tile_det["boxes_xyxy"]])
    scores = np.concatenate([full_det["scores"], tile_scores_penalized])
    classes = np.concatenate([full_det["classes"], tile_det["classes"]])
    
    if "class_confidence" in full_det and "class_confidence" in tile_det:
        class_conf = np.concatenate([full_det["class_confidence"], tile_det["class_confidence"]])
    elif "class_confidence" in full_det:
        class_conf = np.concatenate([full_det["class_confidence"], np.ones_like(tile_det["scores"])])
    else:
        class_conf = np.ones_like(scores)

    keep = torchvision.ops.nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_thr
    ).numpy()

    # Restore original scores for the kept boxes (undo the 0.95 penalty for the final output)
    original_scores = np.concatenate([full_det["scores"], tile_det["scores"]])

    return {
        "boxes_xyxy": boxes[keep],
        "scores": original_scores[keep],
        "classes": classes[keep],
        "class_confidence": class_conf[keep]
    }


def _assign_classes_by_vote(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    source_dets: list[dict],
    source_classes: list[np.ndarray],
    weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    n_fused = len(fused_boxes)
    assigned_classes = np.zeros(n_fused, dtype=int)
    confidences = np.zeros(n_fused, dtype=float)

    for fi in range(n_fused):
        fbox = fused_boxes[fi]
        class_votes: dict[int, float] = {}

        for si, det in enumerate(source_dets):
            if len(det["boxes_xyxy"]) == 0:
                continue
            ious = _batch_iou(fbox, det["boxes_xyxy"])
            for di in range(len(ious)):
                if ious[di] >= 0.3:
                    cls_id = int(source_classes[si][di])
                    score_weight = float(det["scores"][di]) * weights[si]
                    class_votes[cls_id] = class_votes.get(cls_id, 0) + score_weight

        if class_votes:
            best_cls = max(class_votes, key=lambda k: class_votes[k])
            total_weight = sum(class_votes.values())
            assigned_classes[fi] = best_cls
            confidences[fi] = (
                class_votes[best_cls] / total_weight if total_weight > 0 else 0
            )
        else:
            assigned_classes[fi] = 0
            confidences[fi] = 0.0

    return assigned_classes, confidences


def _batch_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def _empty_result() -> dict:
    return {
        "boxes_xyxy": np.empty((0, 4)),
        "scores": np.empty(0),
        "classes": np.empty(0, dtype=int),
        "class_confidence": np.empty(0),
    }
