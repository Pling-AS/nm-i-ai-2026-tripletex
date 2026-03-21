from __future__ import annotations

import argparse
import importlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NUM_CLASSES = 356
IMG_SIZE = 224
PAD_RATIO = 0.10
MIN_CROP_SIZE = 20


_DEPS: dict[str, Any] | None = None


def get_deps() -> dict[str, Any]:
    global _DEPS
    if _DEPS is not None:
        return _DEPS
    torch_mod = importlib.import_module("torch")
    timm_mod = importlib.import_module("timm")
    np_mod = importlib.import_module("numpy")
    image_mod = importlib.import_module("PIL.Image")
    safetensors_torch = importlib.import_module("safetensors.torch")
    transforms_mod = importlib.import_module("torchvision.transforms")
    optim_mod = importlib.import_module("torch.optim")
    lr_sched_mod = importlib.import_module("torch.optim.lr_scheduler")
    data_mod = importlib.import_module("torch.utils.data")
    nn_mod = importlib.import_module("torch.nn")
    _DEPS = {
        "torch": torch_mod,
        "timm": timm_mod,
        "np": np_mod,
        "Image": image_mod,
        "save_file": safetensors_torch.save_file,
        "transforms": transforms_mod,
        "AdamW": optim_mod.AdamW,
        "CosineAnnealingLR": lr_sched_mod.CosineAnnealingLR,
        "DataLoader": data_mod.DataLoader,
        "WeightedRandomSampler": data_mod.WeightedRandomSampler,
        "nn": nn_mod,
    }
    return _DEPS


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    source: str


class RandomBackgroundNoise:
    def __init__(self, p: float = 0.8, threshold: int = 235, noise_std: float = 18.0):
        self.p = p
        self.threshold = threshold
        self.noise_std = noise_std

    def __call__(self, image):
        deps = get_deps()
        np = deps["np"]
        Image = deps["Image"]
        if random.random() > self.p:
            return image

        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        mask = (
            (arr[:, :, 0] >= self.threshold)
            & (arr[:, :, 1] >= self.threshold)
            & (arr[:, :, 2] >= self.threshold)
        )
        if not mask.any():
            return image

        bg_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=(arr.shape[0], arr.shape[1], 3),
        )
        noisy_bg = np.clip(bg_color.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        arr[mask] = noisy_bg[mask]
        return Image.fromarray(arr, mode="RGB")


class SourceAwareDataset:
    def __init__(self, samples: list[Sample], tf_shelf, tf_ref):
        self.samples = samples
        self.tf_shelf = tf_shelf
        self.tf_ref = tf_ref

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        Image = get_deps()["Image"]
        image = Image.open(sample.image_path).convert("RGB")
        if sample.source == "ref":
            tensor = self.tf_ref(image)
        else:
            tensor = self.tf_shelf(image)
        return tensor, sample.label


def seed_everything(seed: int) -> None:
    torch = get_deps()["torch"]
    random.seed(seed)
    get_deps()["np"].random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco(annotations_path: Path) -> dict:
    with annotations_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_product_code_to_category(coco: dict, metadata: dict) -> dict[str, int]:
    mapping: dict[str, int] = {}

    has_product_code = False
    for ann in coco["annotations"]:
        if "product_code" not in ann:
            continue
        has_product_code = True
        cat_id = int(ann["category_id"])
        if not (0 <= cat_id < NUM_CLASSES):
            continue
        product_code = str(ann["product_code"])
        if product_code in mapping and mapping[product_code] != cat_id:
            continue
        mapping[product_code] = cat_id

    if has_product_code and mapping:
        return mapping

    print(
        "WARNING: No usable annotation.product_code found in COCO; "
        "falling back to metadata product_name -> category name mapping."
    )
    name_to_cat = {c["name"]: int(c["id"]) for c in coco["categories"]}
    for prod in metadata.get("products", []):
        code = str(prod.get("product_code", ""))
        name = prod.get("product_name")
        if not code or not name:
            continue
        cat_id = name_to_cat.get(name)
        if cat_id is None or not (0 <= cat_id < NUM_CLASSES):
            continue
        mapping[code] = int(cat_id)
    return mapping


def ensure_shelf_crops(
    coco: dict,
    images_dir: Path,
    crops_dir: Path,
    force_regen: bool,
) -> list[Sample]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    existing = list(crops_dir.glob("**/*"))
    has_images = any(
        p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in existing
    )

    if force_regen or not has_images:
        print("Generating shelf crops from COCO annotations...")
        image_by_id = {int(im["id"]): im for im in coco["images"]}
        saved = 0
        skipped = 0

        for ann in coco["annotations"]:
            cat_id = int(ann["category_id"])
            if not (0 <= cat_id < NUM_CLASSES):
                continue

            img_info = image_by_id.get(int(ann["image_id"]))
            if img_info is None:
                skipped += 1
                continue

            img_path = images_dir / img_info["file_name"]
            if not img_path.exists():
                skipped += 1
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                skipped += 1
                continue

            Image = get_deps()["Image"]
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img_w, img_h = image.size

                pad_x = w * PAD_RATIO
                pad_y = h * PAD_RATIO

                x1 = x - pad_x
                y1 = y - pad_y
                x2 = x + w + pad_x
                y2 = y + h + pad_y

                if (x2 - x1) < MIN_CROP_SIZE:
                    cx = 0.5 * (x1 + x2)
                    x1 = cx - MIN_CROP_SIZE / 2
                    x2 = cx + MIN_CROP_SIZE / 2
                if (y2 - y1) < MIN_CROP_SIZE:
                    cy = 0.5 * (y1 + y2)
                    y1 = cy - MIN_CROP_SIZE / 2
                    y2 = cy + MIN_CROP_SIZE / 2

                x1i = max(0, int(round(x1)))
                y1i = max(0, int(round(y1)))
                x2i = min(img_w, int(round(x2)))
                y2i = min(img_h, int(round(y2)))

                if (x2i - x1i) < 2 or (y2i - y1i) < 2:
                    skipped += 1
                    continue

                crop = image.crop((x1i, y1i, x2i, y2i))

            out_dir = crops_dir / str(cat_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{int(ann['id'])}.jpg"
            crop.save(out_path, quality=95)
            saved += 1

        print(f"Shelf crop generation done: saved={saved}, skipped={skipped}")

    samples: list[Sample] = []
    for class_dir in sorted(crops_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        if not (0 <= label < NUM_CLASSES):
            continue

        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            samples.append(Sample(image_path=image_path, label=label, source="shelf"))

    return samples


def collect_reference_samples(
    ref_dir: Path,
    metadata: dict,
    product_code_to_cat: dict[str, int],
) -> tuple[list[Sample], int]:
    samples: list[Sample] = []
    skipped_unmapped = 0

    for prod in metadata.get("products", []):
        if not prod.get("has_images", False):
            continue
        product_code = str(prod.get("product_code", ""))
        if not product_code:
            continue

        cat_id = product_code_to_cat.get(product_code)
        if cat_id is None or not (0 <= cat_id < NUM_CLASSES):
            skipped_unmapped += 1
            continue

        pdir = ref_dir / product_code
        if not pdir.exists() or not pdir.is_dir():
            continue

        image_types = prod.get("image_types", [])
        for img_type in image_types:
            base = pdir / str(img_type)
            candidates = [
                base.with_suffix(".jpg"),
                base.with_suffix(".jpeg"),
                base.with_suffix(".png"),
                base.with_suffix(".webp"),
            ]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path is None:
                continue
            samples.append(Sample(image_path=img_path, label=cat_id, source="ref"))

    return samples, skipped_unmapped


def stratified_split(
    samples: list[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    rng = random.Random(seed)
    train, val = [], []
    for label, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        if len(cls_samples) <= 1:
            train.extend(cls_samples)
            continue
        n_val = max(1, int(round(len(cls_samples) * val_ratio)))
        n_val = min(n_val, len(cls_samples) - 1)
        val.extend(cls_samples[:n_val])
        train.extend(cls_samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_transforms(model_name: str):
    deps = get_deps()
    timm = deps["timm"]
    transforms = deps["transforms"]
    tmp_model = timm.create_model(
        model_name, pretrained=False, num_classes=NUM_CLASSES, img_size=IMG_SIZE
    )
    data_cfg = timm.data.resolve_model_data_config(tmp_model)
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    tf_shelf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    tf_ref_train = transforms.Compose(
        [
            RandomBackgroundNoise(p=0.85, threshold=235, noise_std=22.0),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
            ),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, value="random"),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    tf_eval = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return tf_shelf_train, tf_ref_train, tf_eval


def build_sampler(train_samples: list[Sample], target_shelf_ratio: float = 0.65) -> Any:
    torch = get_deps()["torch"]
    WeightedRandomSampler = get_deps()["WeightedRandomSampler"]
    class_counts = Counter(s.label for s in train_samples)
    base_weights = [1.0 / class_counts[s.label] for s in train_samples]

    source_mass = defaultdict(float)
    for s, bw in zip(train_samples, base_weights):
        source_mass[s.source] += bw

    target_ref_ratio = 1.0 - target_shelf_ratio
    eps = 1e-12
    shelf_scale = target_shelf_ratio / max(source_mass.get("shelf", eps), eps)
    ref_scale = target_ref_ratio / max(source_mass.get("ref", eps), eps)

    weights = []
    for s, bw in zip(train_samples, base_weights):
        if s.source == "ref":
            weights.append(bw * ref_scale)
        else:
            weights.append(bw * shelf_scale)

    tensor_weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        tensor_weights, num_samples=len(train_samples), replacement=True
    )


def evaluate(model, loader, device) -> tuple[float, float]:
    deps = get_deps()
    nn = deps["nn"]
    torch = deps["torch"]
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def save_bundle(model, output_path: Path, use_fp16: bool) -> None:
    deps = get_deps()
    torch = deps["torch"]
    save_file = deps["save_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    tensors = {}
    for k, v in state.items():
        t = v.detach().cpu()
        if use_fp16 and torch.is_floating_point(t):
            t = t.half()
        tensors[f"model.{k}"] = t
    save_file(tensors, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DINOv2 classifier with shelf+ref mix"
    )
    parser.add_argument("--crops-dir", required=True, type=Path)
    parser.add_argument("--ref-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shelf-ratio", type=float, default=0.65)
    parser.add_argument("--force-regen-crops", action="store_true")
    parser.add_argument(
        "--vitb", action="store_true", help="Use ViT-B instead of ViT-S"
    )
    return parser.parse_args()


def main() -> None:
    deps = get_deps()
    timm = deps["timm"]
    torch = deps["torch"]
    nn = deps["nn"]
    AdamW = deps["AdamW"]
    CosineAnnealingLR = deps["CosineAnnealingLR"]
    DataLoader = deps["DataLoader"]
    args = parse_args()
    seed_everything(args.seed)

    model_name = (
        "vit_base_patch14_dinov2.lvd142m"
        if args.vitb
        else "vit_small_patch14_dinov2.lvd142m"
    )

    metadata_path = args.metadata or (args.ref_dir / "metadata.json")
    if args.images_dir is None:
        images_dir = args.annotations.parent / "images"
    else:
        images_dir = args.images_dir

    if not args.annotations.exists():
        raise FileNotFoundError(f"Missing annotations: {args.annotations}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not args.ref_dir.exists():
        raise FileNotFoundError(f"Missing reference image directory: {args.ref_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    coco = load_coco(args.annotations)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    shelf_samples = ensure_shelf_crops(
        coco=coco,
        images_dir=images_dir,
        crops_dir=args.crops_dir,
        force_regen=args.force_regen_crops,
    )
    if not shelf_samples:
        raise RuntimeError("No shelf crops found/generated.")

    product_code_to_cat = build_product_code_to_category(coco, metadata)
    ref_samples, skipped_unmapped = collect_reference_samples(
        ref_dir=args.ref_dir,
        metadata=metadata,
        product_code_to_cat=product_code_to_cat,
    )
    if not ref_samples:
        raise RuntimeError("No reference samples collected after mapping.")

    print(
        f"Samples: shelf={len(shelf_samples)}, ref={len(ref_samples)}, "
        f"unmapped_ref_products={skipped_unmapped}"
    )

    shelf_train, shelf_val = stratified_split(shelf_samples, args.val_ratio, args.seed)
    ref_train, ref_val = stratified_split(ref_samples, args.val_ratio, args.seed + 1)
    train_samples = shelf_train + ref_train
    val_samples = shelf_val + ref_val

    random.Random(args.seed + 7).shuffle(train_samples)
    random.Random(args.seed + 8).shuffle(val_samples)

    tf_shelf_train, tf_ref_train, tf_eval = build_transforms(model_name)
    train_ds = SourceAwareDataset(
        train_samples, tf_shelf=tf_shelf_train, tf_ref=tf_ref_train
    )
    val_ds = SourceAwareDataset(val_samples, tf_shelf=tf_eval, tf_ref=tf_eval)

    sampler = build_sampler(train_samples, target_shelf_ratio=args.shelf_ratio)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    model = timm.create_model(
        model_name, pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.size(0)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"lr={lr_now:.6e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"  New best val_acc={best_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Training finished without best_state.")

    model.load_state_dict(best_state)
    save_bundle(model, args.output, use_fp16=args.vitb)
    print(
        f"Saved best model to {args.output} (val_acc={best_acc:.4f}, "
        f"fp16={'yes' if args.vitb else 'no'})"
    )


if __name__ == "__main__":
    main()
