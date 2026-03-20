"""Convert COCO annotations to YOLO format labels and generate dataset.yaml.

YOLO label format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
COCO bbox format: [x, y, width, height] in pixels
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path


def coco_bbox_to_yolo(
    bbox: list, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return x_center, y_center, norm_w, norm_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to annotations.json")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument(
        "--out", required=True, help="Output directory for YOLO dataset"
    )
    args = parser.parse_args()

    coco_path = Path(args.coco)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)

    with open(coco_path) as f:
        coco = json.load(f)

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    num_classes = max(cat["id"] for cat in coco["categories"]) + 1
    class_names = {cat["id"]: cat["name"] for cat in coco["categories"]}

    out_images = out_dir / "images" / "all"
    out_labels = out_dir / "labels" / "all"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        fname = img_info["file_name"]

        src_path = images_dir / fname
        if not src_path.exists():
            print(f"WARNING: {src_path} not found, skipping")
            continue

        stem = src_path.stem
        dst_img = out_images / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        anns = ann_by_image.get(img_id, [])
        label_path = out_labels / f"{stem}.txt"
        with open(label_path, "w") as lf:
            for ann in anns:
                cls_id = ann["category_id"]
                xc, yc, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        converted += 1

    names_list = [class_names.get(i, f"class_{i}") for i in range(num_classes)]
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": names_list,
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as yf:
        yaml.dump(dataset_yaml, yf, default_flow_style=False, allow_unicode=True)

    print(f"Converted {converted} images with {len(coco['annotations'])} annotations")
    print(f"Classes: {num_classes}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Images: {out_images}")
    print(f"Labels: {out_labels}")


if __name__ == "__main__":
    main()
