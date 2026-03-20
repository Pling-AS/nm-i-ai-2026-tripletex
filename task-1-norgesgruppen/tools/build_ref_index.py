"""Build DINOv2 prototype embeddings from product reference images.

Outputs:
  - classifier_bundle.safetensors: DINOv2 weights + prototype embeddings
  - category_to_ref.json: mapping from category_id to reference image info
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from safetensors.torch import save_file


CROP_SIZE = 224


def build_category_mapping(annotations_path: Path, metadata_path: Path) -> dict:
    with open(annotations_path) as f:
        coco = json.load(f)
    with open(metadata_path) as f:
        meta = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    mapping = {}
    for product in meta["products"]:
        name = product["product_name"]
        if name not in cat_name_to_id:
            continue
        if not product["has_images"]:
            continue
        cat_id = cat_name_to_id[name]
        mapping[cat_id] = {
            "product_code": product["product_code"],
            "product_name": name,
            "ref_images": product["image_types"],
        }

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--ref-root", required=True, help="Root dir of product reference images"
    )
    parser.add_argument("--annotations", required=True, help="COCO annotations.json")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--out", required=True, help="Output .safetensors bundle path")
    parser.add_argument(
        "--map-out", required=True, help="Output category mapping JSON path"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mapping = build_category_mapping(Path(args.annotations), Path(args.metadata))
    print(f"Mapped {len(mapping)} categories to reference images")

    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    model = model.to(device).eval()

    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    ref_root = Path(args.ref_root)
    all_embeddings = []
    ordered_mapping = {}

    proto_idx = 0
    for cat_id in sorted(mapping.keys()):
        info = mapping[cat_id]
        product_code = info["product_code"]
        product_dir = ref_root / product_code
        if not product_dir.exists():
            print(f"WARNING: {product_dir} not found, skipping cat {cat_id}")
            continue

        cat_embeddings = []
        for img_type in info["ref_images"]:
            img_path = product_dir / f"{img_type}.jpg"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=-1)
            cat_embeddings.append(emb)
            all_embeddings.append(emb)
            proto_idx += 1

        if cat_embeddings:
            ordered_mapping[cat_id] = {
                "product_code": info["product_code"],
                "product_name": info["product_name"],
                "ref_images": info["ref_images"],
                "n_protos": len(cat_embeddings),
            }

    if not all_embeddings:
        print("ERROR: No embeddings computed!")
        return

    prototypes = torch.cat(all_embeddings, dim=0)
    print(f"Prototype matrix: {prototypes.shape}")

    state_dict = model.state_dict()
    tensors_to_save = {"prototypes": prototypes.cpu()}
    for k, v in state_dict.items():
        tensors_to_save[f"model.{k}"] = v.cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_to_save, str(out_path))
    print(f"Saved bundle: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    map_path = Path(args.map_out)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in ordered_mapping.items()}, f, indent=2)
    print(f"Saved mapping: {map_path} ({len(ordered_mapping)} categories)")


if __name__ == "__main__":
    main()
