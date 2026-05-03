"""
Prepare NIH Chest X-Ray dataset subset (~500 MB)
Uses: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

Folder structure expected by train.py:
  data/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
"""

import os
import shutil
import random
import zipfile
import subprocess
import sys
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
# We use the Paul Mooney Kaggle mirror of the Chest X-Ray Images (Pneumonia) dataset
# which is ~1.2 GB. We sample a 500 MB subset.
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
BASE_DIR       = Path(__file__).parent.parent
RAW_DIR        = BASE_DIR / "data_raw"
DATA_DIR       = BASE_DIR / "data"

# Subset sizes (images per class)
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS   = 300

SEED = 42
random.seed(SEED)


def check_kaggle_api():
    """Ensure kaggle CLI is available."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[ERROR] kaggle CLI not found.")
        print("  Install: pip install kaggle")
        print("  Place kaggle.json in %USERPROFILE%\\.kaggle\\  (Windows)")
        sys.exit(1)


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "chest-xray-pneumonia.zip"
    if zip_path.exists():
        print("[SKIP] ZIP already downloaded.")
        return zip_path

    print(f"[INFO] Downloading {KAGGLE_DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(RAW_DIR)],
        check=True,
    )
    return zip_path


def extract(zip_path: Path):
    extract_dir = RAW_DIR / "chest_xray"
    if extract_dir.exists():
        print("[SKIP] Already extracted.")
        return extract_dir
    print("[INFO] Extracting ZIP ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print("[DONE] Extracted.")
    return extract_dir


def sample_and_copy(src_dir: Path, dst_dir: Path, n: int, label: str):
    """Pick n random images from src_dir and copy to dst_dir/label/."""
    out = dst_dir / label
    out.mkdir(parents=True, exist_ok=True)
    imgs = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    chosen = random.sample(imgs, min(n, len(imgs)))
    for img in chosen:
        shutil.copy2(img, out / img.name)
    print(f"  {label}: copied {len(chosen)} → {out}")


def build_split(raw_split_dir: Path, out_split_dir: Path, n_per_class: int):
    for cls in ["NORMAL", "PNEUMONIA"]:
        src = raw_split_dir / cls
        if not src.exists():
            print(f"[WARN] {src} not found — skipping")
            continue
        sample_and_copy(src, out_split_dir, n_per_class, cls)


def main():
    check_kaggle_api()
    zip_path    = download_dataset()
    extract_dir = extract(zip_path)

    # The Kaggle dataset has chest_xray/train  chest_xray/test  chest_xray/val
    raw_train = extract_dir / "chest_xray" / "train"
    raw_val   = extract_dir / "chest_xray" / "val"
    # val set is tiny — supplement from test
    raw_test  = extract_dir / "chest_xray" / "test"

    print("\n[INFO] Building train split ...")
    build_split(raw_train, DATA_DIR / "train", TRAIN_PER_CLASS)

    print("[INFO] Building val split ...")
    # Combine val + test for validation
    tmp_val = RAW_DIR / "combined_val"
    for cls in ["NORMAL", "PNEUMONIA"]:
        (tmp_val / cls).mkdir(parents=True, exist_ok=True)
        for src_root in [raw_val, raw_test]:
            src = src_root / cls
            if src.exists():
                for img in src.iterdir():
                    shutil.copy2(img, tmp_val / cls / img.name)
    build_split(tmp_val, DATA_DIR / "val", VAL_PER_CLASS)

    # Report
    for split in ["train", "val"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            d = DATA_DIR / split / cls
            if d.exists():
                print(f"  data/{split}/{cls}: {len(list(d.iterdir()))} images")

    print("\n[DONE] Dataset ready in ./data/")


if __name__ == "__main__":
    main()
