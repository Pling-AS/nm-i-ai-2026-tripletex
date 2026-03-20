import json
from pathlib import Path


def create_subset_json():
    coco_path = Path("training_data/NM_NGD_coco_dataset/annotations.json")
    val_images_dir = Path("temp_eval/val_subset")
    output_path = Path("temp_eval/val_subset_annotations.json")

    val_filenames = {f.name for f in val_images_dir.glob("*.jpg")}

    with open(coco_path) as f:
        coco = json.load(f)

    valid_images = []
    valid_filenames = set()
    for img in coco["images"]:
        if img["file_name"] in val_filenames:
            valid_images.append(img)
            valid_filenames.add(img["file_name"])

    for fname in val_filenames:
        if fname not in valid_filenames:
            (val_images_dir / fname).unlink()

    val_image_ids = {img["id"] for img in valid_images}
    val_annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in val_image_ids
    ]

    val_coco = {
        "images": valid_images,
        "annotations": val_annotations,
        "categories": coco["categories"],
    }

    with open(output_path, "w") as f:
        json.dump(val_coco, f)

    print(
        f"Created {output_path} with {len(valid_images)} images and {len(val_annotations)} annotations."
    )


if __name__ == "__main__":
    create_subset_json()
