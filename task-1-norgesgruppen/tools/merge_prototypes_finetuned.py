import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import timm
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", required=True, help="Path to classifier_v2.safetensors"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--ref-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model Weights
    print(f"Loading model from {args.model_path}")
    bundle = load_file(args.model_path, device="cpu")
    model_state = {
        k.replace("model.", "", 1): v
        for k, v in bundle.items()
        if k.startswith("model.")
    }

    # Detect num_classes
    head_weight = model_state.get("head.weight")
    if head_weight is None:
        raise ValueError("Model has no head.weight!")
    num_classes = head_weight.shape[0]
    print(f"Detected {num_classes} classes")

    # Load into TIMM
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=num_classes,
        img_size=224,
    )
    model.load_state_dict(model_state)
    model = model.to(device).eval()

    # 2. Prepare Data Transforms
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, 224, 224)  # FORCE 224
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    # 3. Map Categories
    with open(args.annotations) as f:
        coco = json.load(f)
    with open(args.metadata) as f:
        meta = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    product_code_to_cat = {}

    for prod in meta["products"]:
        if prod["product_name"] in cat_name_to_id:
            product_code_to_cat[str(prod["product_code"])] = cat_name_to_id[
                prod["product_name"]
            ]

    print(f"Mapped {len(product_code_to_cat)} products to categories")

    # 4. Compute Prototypes
    ref_root = Path(args.ref_root)
    prototypes = torch.zeros((num_classes, 384), device=device)  # ViT-S dim is 384
    counts = torch.zeros(num_classes, device=device)

    print("Computing prototypes...")
    with torch.no_grad():
        for code, cat_id in product_code_to_cat.items():
            pdir = ref_root / code
            if not pdir.exists():
                continue

            # Find all images
            images = []
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                images.extend(pdir.rglob(f"*{ext}"))

            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img).unsqueeze(0).to(device)
                    # Get embedding (forward_features usually returns dict or tensor depending on model)
                    # For ViT, forward_features returns (B, N, D). CLS is usually index 0 or use forward_head(pre_logits=True)
                    # Safe way with timm:
                    feats = model.forward_features(x)
                    # ViT-S DINOv2: feats is (B, 1370, 384) including registers/cls?
                    # Actually timm dinov2 forward_features returns:
                    # x_norm = self.norm(x)
                    # return x_norm

                    # Let's use forward_head(pre_logits=True)
                    emb = model.forward_head(feats, pre_logits=True)

                    emb = F.normalize(emb, dim=-1)
                    prototypes[cat_id] += emb.squeeze(0)
                    counts[cat_id] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    # Average and Normalize
    mask = counts > 0
    prototypes[mask] /= counts[mask].unsqueeze(1)
    prototypes = F.normalize(prototypes, dim=-1)

    print(f"Computed prototypes for {mask.sum().item()} categories")

    # 5. Save New Bundle
    tensors_to_save = {}
    # Copy original model weights
    for k, v in bundle.items():
        if k.startswith("model."):
            tensors_to_save[k] = v

    # Add prototypes
    tensors_to_save["prototypes"] = prototypes.cpu()

    save_file(tensors_to_save, args.output)
    print(f"Saved merged model to {args.output}")


if __name__ == "__main__":
    main()
