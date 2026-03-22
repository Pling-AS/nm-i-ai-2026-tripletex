import numpy as np
from ensemble_boxes import weighted_boxes_fusion

boxes_list = [
    [], # Model 1
    [], # Model 2
    [[0.1, 0.1, 0.2, 0.2]] # Tile Model
]
scores_list = [
    [],
    [],
    [0.9]
]
labels_list = [
    [],
    [],
    [0]
]
weights = [1.3, 1.0, 1.0]

fused_boxes, fused_scores, _ = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.5
)
print("Fused scores for small object (only tile found it):", fused_scores)
