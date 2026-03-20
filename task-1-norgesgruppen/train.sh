#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
COCO_DIR="$TASK_ROOT/training_data/NM_NGD_coco_dataset"
REF_DIR="$TASK_ROOT/training_data/NM_NGD_product_images"
WORK_DIR="$TASK_ROOT/work"

echo "=== Step 1: Convert COCO to YOLO format ==="
python "$TASK_ROOT/tools/prepare_yolo_dataset.py" \
  --coco "$COCO_DIR/annotations.json" \
  --images-dir "$COCO_DIR/images" \
  --out "$WORK_DIR/yolo_data"

echo ""
echo "=== Step 2: Create train/val split with oversampling ==="
python "$TASK_ROOT/tools/make_val_split.py" \
  --coco "$COCO_DIR/annotations.json" \
  --yolo-root "$WORK_DIR/yolo_data" \
  --out "$WORK_DIR/split" \
  --val-ratio 0.2 \
  --seed 42

echo ""
echo "=== Step 3: Train YOLOv8l - Phase 1 (head adaptation, backbone frozen) ==="
yolo detect train \
  model=yolov8l.pt \
  data="$WORK_DIR/split/dataset.yaml" \
  project="$WORK_DIR/runs" \
  name=yolov8l_phase1 \
  imgsz=1280 \
  epochs=15 \
  batch=4 \
  device=0 \
  workers=4 \
  freeze=10 \
  lr0=0.003 \
  lrf=0.03 \
  weight_decay=0.0005 \
  warmup_epochs=3 \
  mosaic=1.0 \
  mixup=0.0 \
  fliplr=0.5 \
  flipud=0.0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.5 \
  shear=2.0 \
  perspective=0.0005 \
  close_mosaic=0 \
  patience=50 \
  seed=42

echo ""
echo "=== Step 4: Train YOLOv8l - Phase 2 (full fine-tune) ==="
yolo detect train \
  model="$WORK_DIR/runs/yolov8l_phase1/weights/best.pt" \
  data="$WORK_DIR/split/dataset.yaml" \
  project="$WORK_DIR/runs" \
  name=yolov8l_phase2 \
  imgsz=1280 \
  epochs=100 \
  batch=4 \
  device=0 \
  workers=4 \
  freeze=0 \
  lr0=0.001 \
  lrf=0.01 \
  weight_decay=0.0005 \
  warmup_epochs=2 \
  mosaic=1.0 \
  mixup=0.0 \
  fliplr=0.5 \
  flipud=0.0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.5 \
  shear=2.0 \
  perspective=0.0005 \
  close_mosaic=10 \
  patience=20 \
  seed=42

echo ""
echo "=== Step 5: Build DINOv2 reference index ==="
python "$TASK_ROOT/tools/build_ref_index.py" \
  --metadata "$REF_DIR/metadata.json" \
  --ref-root "$REF_DIR" \
  --annotations "$COCO_DIR/annotations.json" \
  --model vit_small_patch14_dinov2.lvd142m \
  --out "$TASK_ROOT/submission/classifier_bundle.safetensors" \
  --map-out "$TASK_ROOT/submission/category_to_ref.json"

echo ""
echo "=== Step 6: Copy best detector weights to submission ==="
cp "$WORK_DIR/runs/yolov8l_phase2/weights/best.pt" "$TASK_ROOT/submission/detector_a.pt"
echo "Copied detector weights to submission/detector_a.pt"

echo ""
echo "=== Step 7: Validate submission locally ==="
python "$TASK_ROOT/submission/run.py" \
  --input "$COCO_DIR/images" \
  --output "$WORK_DIR/val_predictions.json"

echo ""
echo "=== Step 8: Score locally ==="
python "$TASK_ROOT/tools/score_hybrid.py" \
  --gt "$COCO_DIR/annotations.json" \
  --pred "$WORK_DIR/val_predictions.json"

echo ""
echo "=== Step 9: Package submission ==="
cd "$TASK_ROOT/submission"
zip -r "$TASK_ROOT/submission.zip" . -x ".*" "__MACOSX/*" "__pycache__/*" "*.pyc"
echo "Submission packaged: $TASK_ROOT/submission.zip"
unzip -l "$TASK_ROOT/submission.zip" | head -15

echo ""
echo "=== OPTIONAL: Train second detector (YOLOv8x) ==="
echo "Uncomment below if YOLOv8l baseline validates well"
echo ""
cat << 'OPTIONAL_EOF'
# Phase 1
yolo detect train \
  model=yolov8x.pt \
  data="$WORK_DIR/split/dataset.yaml" \
  project="$WORK_DIR/runs" \
  name=yolov8x_phase1 \
  imgsz=1280 epochs=12 batch=2 device=0 workers=4 \
  freeze=10 seed=13

# Phase 2
yolo detect train \
  model="$WORK_DIR/runs/yolov8x_phase1/weights/best.pt" \
  data="$WORK_DIR/split/dataset.yaml" \
  project="$WORK_DIR/runs" \
  name=yolov8x_phase2 \
  imgsz=1280 epochs=80 batch=2 device=0 workers=4 \
  freeze=0 close_mosaic=10 patience=20 seed=13

# Copy to submission
cp "$WORK_DIR/runs/yolov8x_phase2/weights/best.pt" "$TASK_ROOT/submission/detector_b.pt"
OPTIONAL_EOF
