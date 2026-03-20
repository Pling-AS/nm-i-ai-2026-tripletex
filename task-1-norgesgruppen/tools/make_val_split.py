"""Create train/val split with rare-class image oversampling.

Oversampling policy:
  - Images containing categories with 1-4 annotations: repeat 4x
  - Images containing categories with 5-20 annotations: repeat 2x
  - Everything else: 1x

Creates symlinked train/val image+label directories and updated dataset.yaml.
"""

import argparse
import json
import random
import shutil
import yaml
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to annotations.json")
    parser.add_argument(
        "--yolo-root",
        required=True,
        help="YOLO dataset root (from prepare_yolo_dataset)",
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for split dataset"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.coco) as f:
        coco = json.load(f)

    yolo_root = Path(args.yolo_root)
    out_dir = Path(args.out)

    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    img_to_cats: dict[int, set[int]] = {}
    for ann in coco["annotations"]:
        img_to_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

    img_id_to_fname: dict[int, str] = {}
    for img in coco["images"]:
        img_id_to_fname[img["id"]] = img["file_name"]

    all_img_ids = sorted(img_to_cats.keys())
    random.shuffle(all_img_ids)

    n_val = max(1, int(len(all_img_ids) * args.val_ratio))
    val_ids = set(all_img_ids[:n_val])
    train_ids = set(all_img_ids[n_val:])

    print(f"Total images: {len(all_img_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    val_cats = set()
    for img_id in val_ids:
        val_cats.update(img_to_cats.get(img_id, set()))
    train_cats = set()
    for img_id in train_ids:
        train_cats.update(img_to_cats.get(img_id, set()))

    missing_in_train = val_cats - train_cats
    if missing_in_train:
        print(f"WARNING: {len(missing_in_train)} categories in val but not in train")

    def get_oversample_factor(img_id: int) -> int:
        cats = img_to_cats.get(img_id, set())
        max_factor = 1
        for cat_id in cats:
            count = cat_counts[cat_id]
            if count <= 4:
                max_factor = max(max_factor, 4)
            elif count <= 20:
                max_factor = max(max_factor, 2)
        return max_factor

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_id in sorted(split_ids):
            fname = img_id_to_fname[img_id]
            stem = Path(fname).stem

            src_img = yolo_root / "images" / "all" / f"{stem}.jpg"
            src_lbl = yolo_root / "labels" / "all" / f"{stem}.txt"

            if not src_img.exists():
                print(f"WARNING: {src_img} not found")
                continue

            factor = get_oversample_factor(img_id) if split_name == "train" else 1

            for rep in range(factor):
                suffix = f"_r{rep}" if rep > 0 else ""
                dst_img = img_dir / f"{stem}{suffix}.jpg"
                dst_lbl = lbl_dir / f"{stem}{suffix}.txt"

                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                if src_lbl.exists() and not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                count += 1

        print(f"{split_name}: {count} image entries ({len(split_ids)} unique images)")

    src_yaml = yolo_root / "dataset.yaml"
    if src_yaml.exists():
        with open(src_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["path"] = str(out_dir.resolve())
        cfg["train"] = "images/train"
        cfg["val"] = "images/val"
    else:
        num_classes = max(cat_counts.keys()) + 1
        cfg = {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": num_classes,
            "names": [f"class_{i}" for i in range(num_classes)],
        }

    out_yaml = out_dir / "dataset.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\nDataset YAML: {out_yaml}")

    oversampled_stats = Counter()
    for img_id in train_ids:
        f = get_oversample_factor(img_id)
        oversampled_stats[f] += 1
    for factor, n in sorted(oversampled_stats.items()):
        print(f"  {n} images with oversample factor {factor}x")


if __name__ == "__main__":
    main()
