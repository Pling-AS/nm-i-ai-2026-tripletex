from __future__ import annotations

import argparse
import importlib
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, default="yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--folds-root",
        type=Path,
        default=Path("work/folds"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("work/runs"),
    )
    return parser.parse_args()


def model_stem(model_name: str) -> str:
    return Path(model_name).stem


def main() -> None:
    ultralytics_module = importlib.import_module("ultralytics")
    YOLO = getattr(ultralytics_module, "YOLO")

    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.fold < 0:
        raise ValueError("--fold must be >= 0")

    folds_root = (
        args.folds_root
        if args.folds_root.is_absolute()
        else project_root / args.folds_root
    )
    runs_root = (
        args.runs_root
        if args.runs_root.is_absolute()
        else project_root / args.runs_root
    )

    fold_dir = folds_root / f"fold{args.fold}"
    data_yaml = fold_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Fold dataset not found: {data_yaml}. Run create_kfold_splits.py first."
        )

    runs_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    run_name = f"{model_stem(args.model)}_fold{args.fold}"

    print(f"Training fold {args.fold} with model {args.model}")
    print(f"Data: {data_yaml}")
    print(f"Output: {runs_root / run_name}")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(runs_root),
        name=run_name,
        seed=42 + args.fold,
        deterministic=True,
        patience=0,
        save_period=50,
        cos_lr=True,
        close_mosaic=20,
        warmup_epochs=5,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        erasing=0.2,
        device=0,
        workers=4,
    )

    best_path = runs_root / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_path}")

    strip_script = project_root / "strip_yolo_weights.py"
    if not strip_script.exists():
        raise FileNotFoundError(
            "strip_yolo_weights.py not found in repo root. "
            "Expected at ~/task-1/strip_yolo_weights.py"
        )

    cmd = ["python3", str(strip_script), str(best_path)]
    print(f"Stripping optimizer from {best_path}")
    subprocess.run(cmd, check=True)

    print(f"Done: {best_path}")


if __name__ == "__main__":
    main()
