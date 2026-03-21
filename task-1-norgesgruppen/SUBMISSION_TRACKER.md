# Submission Tracker — NM i AI 2026, Task 1 (NorgesGruppen)

**Competition ends:** March 22, 15:00 CET
**Leader:** 0.9255 | **Our best:** 0.9113 | **Gap:** 0.0142
**Score formula:** 0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5
**Submissions remaining:** 6 (reset at 01:00 Oslo / 00:00 UTC)

---

## Server Submission History (Ground Truth)

| # | Server Score | detector_a | detector_b | classifier | tile_thresh | wbf_iou | wbf_weights | Notes |
|---|-------------|-----------|-----------|-----------|-------------|---------|-------------|-------|
| 9 | 0.9085 | l_80/20 | x_v1 | cls_v2 | 4000 | 0.50 | [1.3, 1.0] | First good score |
| 11 | 0.9036 | l_fulldata | x_v1 | cls_v3 | 4000 | 0.50 | [2.0, 0.5, 1.0] | Fulldata l + cls_v3 = BAD |
| 12 | 0.9020 | l_fulldata | x_v1 | cls_v3 | 4000 | 0.50 | [1.3, 1.0] | Same models, orig thresh = WORSE |
| 13 | 0.9069 | x_v2 | x_v1 | cls_v2 | 4000 | 0.50 | [1.0, 1.0] | Dual x, no l = worse |
| 14 | 0.9108 | l_80/20 | x_v2 | cls_v2 | 4000 | 0.50 | [1.3, 1.0] | x_v2 helps +0.0023 |
| **15** | **0.9113** | **l_80/20** | **x_v2** | **cls_v2** | **4000** | **0.45** | **[1.3, 1.0]** | **BEST — wbf_iou=0.45 helps +0.0005** |
| 16 | 0.9064 | l_fold0 | x_v2 | cls_v2 | 4000 | 0.45 | [1.3, 1.0] | Different l fold = BAD |

### Proven Facts from Server
- **l_80/20 (original) is IRREPLACEABLE** — every substitute scores worse
- **x_v2 > x_v1** on server (+0.0023)
- **wbf_iou=0.45 > 0.50** (+0.0005)
- **cls_v2 is the only good classifier** — cls_v3 and cls_refmix both hurt
- **Fulldata l model HURTS** (too similar to x_fulldata)
- **l_fold0 HURTS** (-0.0049 vs original l_80/20)

---

## Available Weight Files

### YOLO Detectors (x models — primary)
| File | Size | Training | Notes |
|------|------|----------|-------|
| `/tmp/yolov8x_fulldata.pt` | 131MB | All 210 images, seed=default | = x_v1 |
| `/tmp/yolov8x_fulldata_v2_stripped.pt` | 131MB | All 210 images, 300ep, seed=123 | = x_v2, **PROVEN best x** |
| `/tmp/yolov8x_fulldata_1536_stripped.pt` | 131MB | All 210 images, imgsz=1536 | **BAD on val and server** |
| `/tmp/yolov8x_fold0_stripped.pt` | 131MB | Fold 0 (80/20), 300ep, seed=42 | **BAD on val** |

### YOLO Detectors (l models — diversity)
| File | Size | Training | Notes |
|------|------|----------|-------|
| `/tmp/build_from_original/detector_a.pt` | 84MB | 80/20 split, 200ep | = l_80/20, **PROVEN IRREPLACEABLE** |
| `/tmp/yolov8l_fold0_stripped.pt` | 84MB | Fold 0, 200ep | BAD on server (-0.0049) |
| `/tmp/yolov8l_fold1_final.pt` | 84MB | Fold 1, ~200ep (resumed) | Untested on server |
| `/tmp/yolov8l_fold2_final.pt` | 84MB | Fold 2, ~200ep (resumed) | Untested on server |
| `/tmp/yolov8l_fold3_final.pt` | 84MB | Fold 3, ~200ep (resumed) | Val sim: +0.0041 vs baseline |
| `/tmp/yolov8l_fold4_final.pt` | 84MB | Fold 4, ~200ep (resumed) | Val sim: +0.0036 vs baseline |

### YOLO Detectors (m model)
| File | Size | Training | Notes |
|------|------|----------|-------|
| `/tmp/yolov8m_fulldata_stripped.pt` | 50MB | All 210 images, 200ep | For 3-detector ensemble |

### Classifiers
| File | Size | Training | Notes |
|------|------|----------|-------|
| `/tmp/build_from_original/classifier_v2.safetensors` | 83MB | DINOv2 ViT-S, 90.2% val acc | **PROVEN best classifier** |
| `/tmp/classifier_refmix_vits.safetensors` | 83MB | DINOv2 ViT-S + ref images, 71.6% val acc | **BAD — confirmed in sim** |

### New Models (Training on A100 Fleet)
| Model | VM | Status | ETA | Expected File | Size | Notes |
|-------|-----|--------|-----|---------------|------|-------|
| RT-DETR-L fulldata | yolo-train-a100-2 | 🔄 Epoch 60/200 | ~30 min | `rtdetr_l_fulldata.pt` | ~64MB | Transformer detector, native in ultralytics 8.1.0 |
| CLIP+DINOv2 classifier | train-cls | 🔄 Epoch 13/30 (head) | ~1.5h | `classifier_clip_dino.safetensors` | ~218MB | ⚠️ TOO BIG for 420MB budget with 2 YOLO models |
| RT-DETR-L 80/20 | train-fold1 | 🔄 Starting | ~1h | `rtdetr_l_8020.pt` | ~64MB | Held-out diversity (like l_80/20) |
| RT-DETR-X fulldata | train-fold2 | 🔄 Starting | ~1.5h | `rtdetr_x_fulldata.pt` | ~100MB | Larger transformer detector |
| DINOv2 ViT-B classifier | train-fold3 | 🔄 Epoch 1/50 | ~1.5h | `classifier_vitb.safetensors` | ~173MB | Fits budget: 131+84+173=388MB ✅ |
| Faster R-CNN ResNet50-FPNv2 | train-fold4 | 🔄 Epoch 7/60 | ~20 min | `fasterrcnn_best.pt` | ~110MB | Two-stage detector, max diversity |

### Weight Budget Combinations (420MB max, 3 files)
| Combo | File 1 | File 2 | File 3 | Total | Fits? |
|-------|--------|--------|--------|-------|-------|
| Current best | x_v2 (131MB) | l_80/20 (84MB) | cls_v2 (83MB) | 298MB | ✅ |
| ViT-B classifier | x_v2 (131MB) | l_80/20 (84MB) | cls_vitb (173MB) | 388MB | ✅ |
| CLIP+DINOv2 cls | x_v2 (131MB) | l_80/20 (84MB) | cls_clip_dino (218MB) | 433MB | ❌ |
| RT-DETR + YOLO | x_v2 (131MB) | rtdetr_l (64MB) | cls_v2 (83MB) | 278MB | ✅ |
| RT-DETR + l_80/20 | rtdetr_l (64MB) | l_80/20 (84MB) | cls_v2 (83MB) | 231MB | ✅ |
| 3 detectors no cls | x_v2 (131MB) | l_80/20 (84MB) | rtdetr_l (64MB) | 279MB | ✅ |
| FRCNN + YOLO | x_v2 (131MB) | frcnn (110MB) | cls_v2 (83MB) | 324MB | ✅ |

---

## Pre-Built Submission Zips

### Ready for Upload
| File | Config | Val Hybrid | Val Δ | Sim Tested | Notes |
|------|--------|-----------|-------|------------|-------|
| `sub2_tile3000.zip` | x_v2 + l_80/20 + cls_v2, T3000, iou=0.45, w=[1.3,1.0] | 0.9444 | -0.0012 | ✅ | S1 in plan |
| `sim_s2.zip` | x_v2 + l_80/20 + cls_v2, T3000, iou=0.45, w=[2.0,1.0] | 0.9457 | +0.0001 | ✅ | S2 in plan |
| `sim_s3.zip` | x_v2 + l_fold3 + cls_v2, T3000, iou=0.45, w=[1.3,1.0] | 0.9497 | +0.0041 | ✅ | S3 — best val but risky |
| `sim_s3b.zip` | x_v2 + l_fold3 + cls_v2, T3000, iou=0.45, w=[2.0,1.0] | 0.9493 | +0.0037 | ✅ | |
| `sim_s4.zip` | x_v2 + l_fold4 + cls_v2, T3000, iou=0.45, w=[1.3,1.0] | 0.9492 | +0.0036 | ✅ | |
| `sim_s5.zip` | x_v2 + l_fold2 + cls_v2, T3000, iou=0.45, w=[1.3,1.0] | 0.9490 | +0.0034 | ✅ | |
| `sim_s6.zip` | x_v2 + l_80/20 + cls_v2, T3000, iou=0.45, w=[1.5,1.0] | 0.9455 | -0.0001 | ✅ | |
| `sim_baseline.zip` | x_v2 + l_80/20 + cls_v2, T4000, iou=0.45, w=[1.3,1.0] | 0.9456 | 0.0000 | ✅ | = server 0.9113 |

### Experimental (Tested Bad)
| File | Config | Val Hybrid | Val Δ | Notes |
|------|--------|-----------|-------|-------|
| `sim_exp1.zip` | 3det (x_v2+l_80/20+m) NO classifier, T3000 | 0.9340 | -0.0116 | Dropping classifier kills cls_mAP |
| `sim_exp2.zip` | x_v2 + l_80/20 + cls_v2, imgsz=1536, T3000 | 0.9421 | -0.0035 | Higher res inference hurts |
| `sim_exp3.zip` | x_v2 + l_80/20 NO classifier, T3000 | 0.9375 | -0.0081 | Classifier is worth +0.008 |
| `sim_exp4.zip` | 3det + 1536 + NO classifier (moonshot) | 0.9289 | -0.0167 | Everything combined = worst |

### Confirmed Dead — Do NOT Submit
- `submission_varA_optimized.zip` — sweep-optimized thresholds (overfit to val)
- `submission_varB_both_fulldata.zip` — l_fulldata (proven bad on server)
- `submission_varC_best_combo.zip` — cls_v3 (proven bad on server)
- `submission_varD_original_thresholds.zip` — l_fulldata + cls_v3 (0.9020 on server)
- `submission_varE_dual_x.zip` — dual x models (0.9069 on server)
- `submission_fold0l_xv2.zip` — l_fold0 (0.9064 on server)
- `sub1_x1536.zip` — x_1536 (bad on val)
- `sub3_refmix.zip` — cls_refmix (bad on val)

---

## Val-to-Server Calibration

| Metric | Val | Server | Gap |
|--------|-----|--------|-----|
| Baseline hybrid | 0.9456 | 0.9113 | 0.0343 |

**Val scores are inflated ~0.034 because fulldata x models saw val images.**
**Relative ordering of CODE changes (tile, WBF) is more trustworthy than MODEL swaps.**

---

## Submission Plan (6 slots at 01:00 Oslo)

Based on Oracle strategy + simulation data:

1. **S1:** `sub2_tile3000.zip` — tile=3000 (safe code change)
2. **S2:** `sim_s2.zip` — tile=3000 + w=[2.0,1.0] (safe threshold change)
3. **S3:** `sim_s3.zip` — l_fold3 + best fusion from S1/S2 (risky fold bet)
4. **S4:** If S3 wins → l_fold3 + runner-up fusion. If S3 loses → `sim_s4.zip` (l_fold4)
5. **S5:** Best winning fold + next fusion, OR l_fold2
6. **S6:** Best combo + w=[1.5,1.0], OR safe fallback

**Decision rule:** Only exploit a fold if it beats current best by ≥0.0003.

**NEW ADDITIONS (if training completes in time):**
- RT-DETR-L as third ensemble detector (replaces one submission slot)
- CLIP+DINOv2 classifier (replaces cls_v2 in one submission slot)

---

## GCP Resources

### Active VMs
| VM | Zone | Status | Last Job |
|----|------|--------|----------|
| yolo-train-a100-2 | us-central1-b | ✅ Running | Simulations done, GPU free |
| train-fold1 | us-central1-a | ✅ Running | l_fold1 training complete |
| train-fold2 | us-central1-f | ✅ Running | l_fold2 training complete |
| train-fold3 | us-central1-f | ✅ Running | l_fold3 training complete |
| train-fold4 | us-central1-b | ✅ Running | l_fold4 training complete |
| train-cls | us-central1-b | ✅ Running | x_fold0 training complete |

### GCP Commands
```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
# SSH: gcloud compute ssh VM_NAME --zone=ZONE --project=ainm26osl-753
# SCP: gcloud compute scp LOCAL VM_NAME:REMOTE --zone=ZONE --project=ainm26osl-753
```

---

## Key Constraints (from docs)
- 3 weight files max (.pt, .pth, .onnx, .safetensors, .npy)
- 420MB total weight size
- 10 .py files max, 1000 files total
- 300s timeout on L4 GPU (24GB VRAM)
- No network, no pip install at runtime
- Banned: import os, subprocess, socket, ctypes, builtins, eval(), exec()
- Must use pathlib instead of os
- ultralytics 8.1.0, timm 0.9.12, torch 2.6.0, Python 3.11
