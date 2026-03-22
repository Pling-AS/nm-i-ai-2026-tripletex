import torch
import torchvision
import numpy as np

def nms_dict(det: dict, iou_thr: float = 0.5) -> dict:
    if len(det["scores"]) == 0:
        return det
    keep = torchvision.ops.nms(
        torch.tensor(det["boxes_xyxy"]),
        torch.tensor(det["scores"]),
        iou_thr
    ).numpy()
    return {k: v[keep] for k, v in det.items()}

def combine_full_and_tile(full_det: dict, tile_det: dict, iou_thr: float = 0.5) -> dict:
    if len(full_det["scores"]) == 0:
        return nms_dict(tile_det, iou_thr)
    if len(tile_det["scores"]) == 0:
        return full_det

    # Concatenate
    boxes = np.vstack([full_det["boxes_xyxy"], tile_det["boxes_xyxy"]])
    scores = np.concatenate([full_det["scores"], tile_det["scores"]])
    classes = np.concatenate([full_det["classes"], tile_det["classes"]])
    
    if "class_confidence" in full_det and "class_confidence" in tile_det:
        class_conf = np.concatenate([full_det["class_confidence"], tile_det["class_confidence"]])
    elif "class_confidence" in full_det:
        class_conf = np.concatenate([full_det["class_confidence"], np.ones_like(tile_det["scores"])])
    else:
        class_conf = np.ones_like(scores)

    keep = torchvision.ops.nms(
        torch.tensor(boxes),
        torch.tensor(scores),
        iou_thr
    ).numpy()

    return {
        "boxes_xyxy": boxes[keep],
        "scores": scores[keep],
        "classes": classes[keep],
        "class_confidence": class_conf[keep]
    }

# Test
full = {"boxes_xyxy": np.array([[0,0,10,10]]), "scores": np.array([0.9]), "classes": np.array([1])}
tile = {"boxes_xyxy": np.array([[0,0,10,10], [20,20,30,30]]), "scores": np.array([0.85, 0.95]), "classes": np.array([1, 2])}

res = combine_full_and_tile(full, tile)
print(res)
