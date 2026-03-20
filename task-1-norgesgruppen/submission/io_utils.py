import json
from pathlib import Path
from PIL import Image
import numpy as np


def parse_image_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_image_np(path: Path) -> np.ndarray:
    return np.array(load_image_pil(path))


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def normalize_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.astype(np.float64)
    out[:, [0, 2]] /= w
    out[:, [1, 3]] /= h
    return out


def denormalize_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.astype(np.float64)
    out[:, [0, 2]] *= w
    out[:, [1, 3]] *= h
    return out


def format_predictions(
    image_id: int, boxes_xywh: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
) -> list[dict]:
    preds = []
    for i in range(len(boxes_xywh)):
        preds.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_ids[i]),
                "bbox": [round(float(v), 1) for v in boxes_xywh[i]],
                "score": round(float(scores[i]), 4),
            }
        )
    return preds


def write_predictions(predictions: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(predictions, f)


def load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def load_coco_annotations(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def get_image_files(input_dir: Path) -> list[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.suffix.lower() in valid_suffixes and not p.name.startswith("._")
    )
