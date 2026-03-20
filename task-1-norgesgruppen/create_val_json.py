import json
from pathlib import Path


def create_val_json():
    coco_path = Path("training_data/NM_NGD_coco_dataset/annotations.json")
    val_images_dir = Path("work/split/images/val")
    output_path = Path("temp_eval/val_annotations.json")

    # Get validation image filenames
    val_filenames = {f.name for f in val_images_dir.glob("*.jpg")}
    print(f"Found {len(val_filenames)} validation images.")

    with open(coco_path) as f:
        coco = json.load(f)

    # Filter images
    val_images = [img for img in coco["images"] if img["file_name"] in val_filenames]
    val_image_ids = {img["id"] for img in val_images}

    # Filter annotations
    val_annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in val_image_ids
    ]

    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco["categories"],
    }

    with open(output_path, "w") as f:
        json.dump(val_coco, f)

    print(
        f"Created {output_path} with {len(val_images)} images and {len(val_annotations)} annotations."
    )


if __name__ == "__main__":
    create_val_json()
