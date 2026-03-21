from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return x_center, y_center, norm_w, norm_h


def _score_fold_for_image(
    fold_idx: int,
    labels: set[int],
    fold_sizes: list[int],
    fold_label_counts: list[Counter[int]],
    target_fold_sizes: list[int],
    target_per_label: dict[int, float],
) -> tuple[float, int, int]:
    label_delta = 0.0
    for cls in labels:
        current = fold_label_counts[fold_idx][cls]
        target = target_per_label[cls]
        before = abs(current - target)
        after = abs((current + 1) - target)
        label_delta += after - before

    size_after = fold_sizes[fold_idx] + 1
    size_delta = (size_after - target_fold_sizes[fold_idx]) ** 2

    return (label_delta * 10.0 + size_delta, fold_sizes[fold_idx], fold_idx)


def stratified_multilabel_kfold(
    image_ids: list[int],
    image_to_labels: dict[int, set[int]],
    n_splits: int,
    seed: int,
) -> list[set[int]]:
    rng = random.Random(seed)

    label_image_count: Counter[int] = Counter()
    for img_id in image_ids:
        for cls in image_to_labels.get(img_id, set()):
            label_image_count[cls] += 1

    def rarity_score(img_id: int) -> tuple[float, int, int]:
        labels = image_to_labels.get(img_id, set())
        if not labels:
            return (0.0, 0, -img_id)
        rarity = sum(1.0 / label_image_count[cls] for cls in labels)
        return (rarity, len(labels), -img_id)

    sorted_ids = sorted(image_ids, key=rarity_score, reverse=True)

    total = len(image_ids)
    target_fold_sizes = [total // n_splits] * n_splits
    for i in range(total % n_splits):
        target_fold_sizes[i] += 1

    target_per_label = {
        cls: count / n_splits for cls, count in label_image_count.items()
    }

    fold_sizes = [0] * n_splits
    fold_label_counts: list[Counter[int]] = [Counter() for _ in range(n_splits)]
    assignments: dict[int, int] = {}

    for img_id in sorted_ids:
        labels = image_to_labels.get(img_id, set())

        candidate_folds = [
            f for f in range(n_splits) if fold_sizes[f] < target_fold_sizes[f]
        ]
        if not candidate_folds:
            candidate_folds = list(range(n_splits))
        rng.shuffle(candidate_folds)

        best_fold = min(
            candidate_folds,
            key=lambda f: _score_fold_for_image(
                fold_idx=f,
                labels=labels,
                fold_sizes=fold_sizes,
                fold_label_counts=fold_label_counts,
                target_fold_sizes=target_fold_sizes,
                target_per_label=target_per_label,
            ),
        )

        assignments[img_id] = best_fold
        fold_sizes[best_fold] += 1
        for cls in labels:
            fold_label_counts[best_fold][cls] += 1

    val_ids_per_fold: list[set[int]] = []
    for fold_idx in range(n_splits):
        fold_ids = {
            img_id for img_id, assigned in assignments.items() if assigned == fold_idx
        }
        val_ids_per_fold.append(fold_ids)

    return val_ids_per_fold


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_yolo_label_file(
    label_path: Path,
    anns: list[dict],
    img_w: int,
    img_h: int,
    max_class_id: int,
) -> None:
    lines: list[str] = []
    for ann in anns:
        cls_id = int(ann["category_id"])
        if cls_id < 0 or cls_id > max_class_id:
            raise ValueError(
                f"Invalid category_id={cls_id} in annotation id={ann.get('id')} "
                f"(expected 0..{max_class_id})"
            )

        x_center, y_center, norm_w, norm_h = coco_bbox_to_yolo(
            ann["bbox"], img_w, img_h
        )

        x_center = min(1.0, max(0.0, x_center))
        y_center = min(1.0, max(0.0, y_center))
        norm_w = min(1.0, max(0.0, norm_w))
        norm_h = min(1.0, max(0.0, norm_h))

        lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco",
        type=Path,
        default=project_root
        / "training_data"
        / "NM_NGD_coco_dataset"
        / "annotations.json",
        help="COCO annotations path",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "training_data" / "NM_NGD_coco_dataset" / "images",
        help="COCO images directory",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=project_root / "work" / "folds",
        help="Output root for fold directories",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--yaml-path-root",
        type=Path,
        default=Path("/home/m/task-1/work/folds"),
        help="Base path written into each fold dataset.yaml",
    )
    args = parser.parse_args()

    coco_path = args.coco
    images_dir = args.images_dir
    out_root = args.out_root

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    images: list[dict] = coco["images"]
    annotations: list[dict] = coco["annotations"]
    categories: list[dict] = coco["categories"]

    image_ids = [int(img["id"]) for img in images]
    image_id_to_info = {int(img["id"]): img for img in images}

    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    image_to_labels: dict[int, set[int]] = {img_id: set() for img_id in image_ids}
    for ann in annotations:
        img_id = int(ann["image_id"])
        ann_by_image[img_id].append(ann)
        image_to_labels[img_id].add(int(ann["category_id"]))

    category_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in categories}
    expected_nc = 356
    category_ids = sorted(category_name_by_id.keys())
    if len(category_ids) != expected_nc:
        raise ValueError(
            f"Expected {expected_nc} categories, found {len(category_ids)}"
        )
    if category_ids != list(range(expected_nc)):
        raise ValueError("Expected category ids to be contiguous 0..355")

    names = [category_name_by_id[i] for i in range(expected_nc)]

    val_ids_per_fold = stratified_multilabel_kfold(
        image_ids=image_ids,
        image_to_labels=image_to_labels,
        n_splits=args.folds,
        seed=args.seed,
    )

    union_val = set().union(*val_ids_per_fold)
    if union_val != set(image_ids):
        missing = set(image_ids) - union_val
        extra = union_val - set(image_ids)
        raise RuntimeError(
            f"Fold validation coverage mismatch. Missing={len(missing)} Extra={len(extra)}"
        )

    overlap_count = sum(
        len(a & b)
        for i, a in enumerate(val_ids_per_fold)
        for b in val_ids_per_fold[i + 1 :]
    )
    if overlap_count != 0:
        raise RuntimeError(f"Validation folds overlap detected: {overlap_count} images")

    out_root.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(args.folds):
        fold_dir = out_root / f"fold{fold_idx}"
        ensure_clean_dir(fold_dir)

        (fold_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        val_ids = val_ids_per_fold[fold_idx]
        train_ids = set(image_ids) - val_ids

        for split_name, split_ids in (("train", train_ids), ("val", val_ids)):
            for img_id in sorted(split_ids):
                info = image_id_to_info[img_id]
                file_name = str(info["file_name"])
                img_w = int(info["width"])
                img_h = int(info["height"])

                src_img = images_dir / file_name
                if not src_img.exists():
                    raise FileNotFoundError(f"Image not found: {src_img}")

                dst_img = fold_dir / "images" / split_name / file_name
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_img.symlink_to(src_img.resolve())

                label_name = f"{Path(file_name).stem}.txt"
                dst_label = fold_dir / "labels" / split_name / label_name
                write_yolo_label_file(
                    label_path=dst_label,
                    anns=ann_by_image.get(img_id, []),
                    img_w=img_w,
                    img_h=img_h,
                    max_class_id=expected_nc - 1,
                )

        dataset_yaml = {
            "path": str(args.yaml_path_root / f"fold{fold_idx}"),
            "train": "images/train",
            "val": "images/val",
            "nc": expected_nc,
            "names": names,
        }
        (fold_dir / "dataset.yaml").write_text(
            json.dumps(dataset_yaml, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Fold {fold_idx}: train={len(train_ids)} val={len(val_ids)} -> {fold_dir}"
        )

    print(
        f"Done. Built {args.folds} folds for {len(image_ids)} images, "
        f"{len(annotations)} annotations, {expected_nc} classes (0-355)."
    )


if __name__ == "__main__":
    main()
