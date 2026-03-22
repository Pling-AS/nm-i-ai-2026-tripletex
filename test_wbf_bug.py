import numpy as np
from ensemble_boxes import weighted_boxes_fusion

boxes_list = [
    [[0.1, 0.1, 0.2, 0.2]], # Model 1
    [[0.1, 0.1, 0.2, 0.2]], # Model 2
    []                      # Tile Model (missed it)
]
scores_list = [
    [0.9],
    [0.9],
    []
]
labels_list = [
    [0],
    [0],
    []
]
weights = [1.3, 1.0, 1.0]

fused_boxes, fused_scores, _ = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.5
)
print("Fused scores with 3 models (tile missed):", fused_scores)

boxes_list_2 = [
    [[0.1, 0.1, 0.2, 0.2]], # Model 1
    [[0.1, 0.1, 0.2, 0.2]], # Model 2
]
scores_list_2 = [
    [0.9],
    [0.9],
]
labels_list_2 = [
    [0],
    [0],
]
weights_2 = [1.3, 1.0]

fused_boxes_2, fused_scores_2, _ = weighted_boxes_fusion(
    boxes_list_2, scores_list_2, labels_list_2, weights=weights_2, iou_thr=0.5
)
print("Fused scores with 2 models:", fused_scores_2)
