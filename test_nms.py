import numpy as np
from ensemble_boxes import nms

boxes_list = [
    [[0.1, 0.1, 0.2, 0.2]], # Fused Full
    [[0.1, 0.1, 0.2, 0.2]]  # Tile
]
scores_list = [
    [0.9],
    [0.85]
]
labels_list = [
    [0],
    [0]
]
weights = [1.0, 1.0]

fused_boxes, fused_scores, _ = nms(
    boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.5
)
print("NMS scores:", fused_scores)
