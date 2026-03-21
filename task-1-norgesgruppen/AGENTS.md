# OBJECT DETECTION — NorgesGruppen

Grocery shelf product detection: YOLO ensemble + WBF fusion + classifier reclassification. 248 training images, 356 product categories, COCO annotation format.

## STRUCTURE

```
task-1-norgesgruppen/
├── submission/                    # ZIP-packaged inference code
│   ├── run.py                     #   Entry point — loads ensemble, tiles, fuses, classifies
│   ├── infer_det.py               #   YOLO detection wrapper
│   ├── infer_cls.py               #   Product classifier inference
│   ├── fusion.py                  #   Weighted Boxes Fusion (WBF)
│   ├── io_utils.py                #   I/O helpers
│   ├── detector_a.pt              #   YOLO model A (fold-trained)
│   ├── detector_b.pt              #   YOLO model B (fold-trained)
│   ├── classifier_v2.safetensors  #   Product reclassifier
│   └── thresholds.json            #   Hyperparams (conf, iou, tile, WBF weights)
├── tools/                         # Training & evaluation scripts
│   ├── prepare_yolo_dataset.py    #   COCO → YOLO format converter
│   ├── create_kfold_splits.py     #   5-fold cross-validation
│   ├── train_fold.py              #   YOLO training per fold
│   ├── train_cls_refmix.py        #   Classifier training (ref images + det crops)
│   ├── build_ref_index.py         #   Reference image index builder
│   ├── export_submission.py       #   ZIP builder
│   ├── score_hybrid.py            #   Local scoring
│   ├── sweep_params.py            #   Hyperparameter sweep
│   ├── calibrate_postprocess.py   #   Post-processing calibration
│   ├── make_val_split.py          #   Train/val splitter
│   └── launch_fleet.sh            #   Multi-job launcher
├── training_data/                 # COCO dataset (248 images, annotations.json)
├── work/                          # Training outputs (folds/, split/, yolo_data/)
├── temp_eval/                     # Temporary evaluation outputs
├── docs/                          # Task specification
├── *.zip                          # Historical submission archives
└── SUBMISSION_TRACKER.md          # Submission history log
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Inference pipeline | `submission/run.py` | Ensemble → tile → WBF → classify → predictions.json |
| Detection params | `submission/thresholds.json` | imgsz, conf, iou, tile_threshold, WBF weights |
| YOLO training | `tools/train_fold.py` | ultralytics API, k-fold |
| Classifier training | `tools/train_cls_refmix.py` | RefMix: real crops + reference images |
| Dataset conversion | `tools/prepare_yolo_dataset.py` | COCO annotations → YOLO .txt labels |
| Scoring rules | `docs/scoring.md` | 70% detection mAP + 30% classification mAP |
| Submission format | `docs/submission.md` | ZIP structure, sandbox env, GPU constraints |

## PIPELINE

1. **Data prep**: `prepare_yolo_dataset.py` → `create_kfold_splits.py`
2. **Train detectors**: `train_fold.py` per fold → `work/folds/foldN/best.pt`
3. **Train classifier**: `train_cls_refmix.py` → `classifier_v2.safetensors`
4. **Package**: `export_submission.py` → `submission.zip`
5. **Inference** (sandboxed): `run.py` loads ensemble, tiles >4000px images, fuses with WBF, reclassifies → `predictions.json`

## SCORING

- **Detection** (70%): mAP@IoU≥0.5, category ignored
- **Classification** (30%): mAP@IoU≥0.5 AND correct category_id
- Detection-only submissions cap at 70%

## ANTI-PATTERNS

- **DO NOT** change sandbox Python version (3.11 pinned)
- **DO NOT** exceed 420MB uncompressed ZIP size
- **DO NOT** assume network access in sandbox
- **Always** test locally before upload
- **Always** match ultralytics version between training and submission

## CONVENTIONS

- YOLO nc=357 (356 products + unknown class)
- Image tiling: threshold 4000px, 15% overlap
- WBF fusion for ensemble (not NMS)
- Time-budgeted classifier disable (sandbox has 300s limit)
- CPU torch for local dev (uv find-links), GPU for training

## COMMANDS

```bash
# Dataset preparation
uv run tools/prepare_yolo_dataset.py
uv run tools/create_kfold_splits.py

# Training
bash train.sh                         # Full training pipeline
uv run tools/train_fold.py            # Single fold

# Classifier
uv run tools/train_cls_refmix.py

# Local evaluation
uv run tools/score_hybrid.py
uv run tools/sweep_params.py

# Package submission
uv run tools/export_submission.py
```
