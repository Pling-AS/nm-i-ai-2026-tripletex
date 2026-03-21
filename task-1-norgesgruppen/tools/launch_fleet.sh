#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/create_kfold_splits.py"

printf "\n"
printf "# 5-fold CV launch commands (run from ~/task-1/tools or use full path)\n"
printf "# Fold 0 - YOLOv8l\npython3 train_fold.py --fold 0 --model yolov8l.pt --epochs 200 --imgsz 1280\n"
printf "# Fold 0 - YOLOv8x\npython3 train_fold.py --fold 0 --model yolov8x.pt --epochs 300 --imgsz 1280\n"
printf "# Fold 1 - YOLOv8l\npython3 train_fold.py --fold 1 --model yolov8l.pt --epochs 200 --imgsz 1280\n"
printf "# Fold 1 - YOLOv8x\npython3 train_fold.py --fold 1 --model yolov8x.pt --epochs 300 --imgsz 1280\n"
printf "# Fold 2 - YOLOv8l\npython3 train_fold.py --fold 2 --model yolov8l.pt --epochs 200 --imgsz 1280\n"
printf "# Fold 2 - YOLOv8x\npython3 train_fold.py --fold 2 --model yolov8x.pt --epochs 300 --imgsz 1280\n"
printf "# Fold 3 - YOLOv8l\npython3 train_fold.py --fold 3 --model yolov8l.pt --epochs 200 --imgsz 1280\n"
printf "# Fold 3 - YOLOv8x\npython3 train_fold.py --fold 3 --model yolov8x.pt --epochs 300 --imgsz 1280\n"
printf "# Fold 4 - YOLOv8l\npython3 train_fold.py --fold 4 --model yolov8l.pt --epochs 200 --imgsz 1280\n"
printf "# Fold 4 - YOLOv8x\npython3 train_fold.py --fold 4 --model yolov8x.pt --epochs 300 --imgsz 1280\n"
