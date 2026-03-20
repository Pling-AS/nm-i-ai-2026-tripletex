"""Package submission directory into a valid .zip file.

Validates structure, file counts, size limits, and forbidden imports
before creating the zip. Catches the most common packaging errors.

Usage:
    python tools/export_submission.py --dir submission --out submission.zip
    python tools/export_submission.py --dir submission --check-only
"""

import argparse
import zipfile
from pathlib import Path


WEIGHT_EXTS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
ALLOWED_EXTS = {".py", ".json", ".yaml", ".yml", ".cfg"} | WEIGHT_EXTS
FORBIDDEN_IMPORTS = [
    "import os",
    "import subprocess",
    "import socket",
    "import ctypes",
    "import builtins",
]
FORBIDDEN_CALLS = ["eval(", "exec(", "compile(", "__import__("]
MAX_PY_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_MB = 420
MAX_TOTAL_FILES = 1000


def validate(src: Path) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors = []

    if not (src / "run.py").exists():
        errors.append("run.py not found at submission root!")

    all_files = [f for f in src.iterdir() if f.is_file() and not f.name.startswith(".")]

    # Extension check
    bad_ext = [f for f in all_files if f.suffix.lower() not in ALLOWED_EXTS]
    for f in bad_ext:
        errors.append(f"Disallowed file type: {f.name}")

    # Count limits
    py_files = [f for f in all_files if f.suffix == ".py"]
    weight_files = [f for f in all_files if f.suffix.lower() in WEIGHT_EXTS]

    if len(py_files) > MAX_PY_FILES:
        errors.append(f"Too many Python files: {len(py_files)}/{MAX_PY_FILES}")
    if len(weight_files) > MAX_WEIGHT_FILES:
        errors.append(f"Too many weight files: {len(weight_files)}/{MAX_WEIGHT_FILES}")
    if len(all_files) > MAX_TOTAL_FILES:
        errors.append(f"Too many total files: {len(all_files)}/{MAX_TOTAL_FILES}")

    # Size limits
    weight_total = sum(f.stat().st_size for f in weight_files)
    if weight_total > MAX_WEIGHT_MB * 1024 * 1024:
        errors.append(
            f"Weight files too large: {weight_total / 1024 / 1024:.1f} MB "
            f"(max {MAX_WEIGHT_MB} MB)"
        )

    # Forbidden imports in Python files
    for pf in py_files:
        content = pf.read_text(errors="replace")
        for pattern in FORBIDDEN_IMPORTS + FORBIDDEN_CALLS:
            if pattern in content:
                errors.append(f"{pf.name}: contains forbidden '{pattern}'")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument("--dir", required=True, help="Submission directory")
    parser.add_argument("--out", default="submission.zip", help="Output zip path")
    parser.add_argument(
        "--check-only", action="store_true", help="Validate without creating zip"
    )
    args = parser.parse_args()

    src = Path(args.dir)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        return

    all_files = [f for f in src.iterdir() if f.is_file() and not f.name.startswith(".")]
    py_files = [f for f in all_files if f.suffix == ".py"]
    weight_files = [f for f in all_files if f.suffix.lower() in WEIGHT_EXTS]
    other_files = [f for f in all_files if f not in py_files and f not in weight_files]

    # --- Report ---
    print(f"Submission directory: {src.resolve()}")
    print(f"\nPython files ({len(py_files)}/{MAX_PY_FILES}):")
    for f in sorted(py_files):
        print(f"  {f.name:30s} {f.stat().st_size / 1024:8.1f} KB")

    weight_total = sum(f.stat().st_size for f in weight_files)
    print(
        f"\nWeight files ({len(weight_files)}/{MAX_WEIGHT_FILES}, "
        f"{weight_total / 1024 / 1024:.1f}/{MAX_WEIGHT_MB} MB):"
    )
    for f in sorted(weight_files):
        print(f"  {f.name:30s} {f.stat().st_size / 1024 / 1024:8.1f} MB")

    if other_files:
        print(f"\nOther files ({len(other_files)}):")
        for f in sorted(other_files):
            print(f"  {f.name:30s} {f.stat().st_size / 1024:8.1f} KB")

    total_size = sum(f.stat().st_size for f in all_files)
    print(f"\nTotal: {len(all_files)} files, {total_size / 1024 / 1024:.1f} MB uncompressed")

    # --- Validate ---
    errors = validate(src)
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  x {e}")
        return

    print("\n[OK] All validation checks passed")

    if args.check_only:
        return

    # --- Create zip ---
    out_path = Path(args.out)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(all_files):
            zf.write(f, f.name)  # flat structure, no subdirectories

    print(f"\nCreated: {out_path}")
    print(f"Compressed: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify structure
    with zipfile.ZipFile(out_path, "r") as zf:
        names = zf.namelist()
        if "run.py" in names:
            print("[OK] run.py at zip root (verified)")
        else:
            print("WARNING: run.py NOT at zip root!")
            print(f"  Found: {names[:5]}")


if __name__ == "__main__":
    main()
