"""
Prepare dataset in YOLO format for training.

Edit SOURCE paths below to point to your original dataset location.
"""

import os
import random
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# ============================================
# CONFIGURATION - Edit these paths!
# ============================================

# Available datasets: "rgb" or "lidar"
DATASET_TYPE = "rgb"  # Change to "lidar" for LIDAR data

# Source paths (original dataset location)
RGB_SOURCE_IMG = "/cluster/projects/vc/data/ad/open/Poles/rgb/images"
RGB_SOURCE_LBL = "/cluster/projects/vc/data/ad/open/Poles/rgb/labels"

LIDAR_SOURCE_IMG = "/cluster/projects/vc/data/ad/open/Poles/lidar/combined_color"
LIDAR_SOURCE_LBL = "/cluster/projects/vc/data/ad/open/Poles/lidar/labels"

# Target directory (where YOLO format data will be created)
TARGET_DIR = Path("data/yolo_dataset")

TRAIN_SPLIT = 0.8
# ============================================

def prepare_dataset(img_dir, lbl_dir, target_dir, dataset_name):
    """Prepare dataset in YOLO format."""
    print(f"\n{'='*50}")
    print(f"Preparing {dataset_name} dataset...")
    print(f"{'='*50}")

    # Create folder structure
    for split in ["train", "val"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all images from train and valid folders
    images = []
    for folder in ["train", "valid"]:
        img_path = os.path.join(img_dir, folder)
        if os.path.exists(img_path):
            images += [
                f for f in os.listdir(img_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
    
    if not images:
        print(f"[!] No images found in {img_dir}")
        return False

    images.sort()
    print(f"[✓] Found {len(images)} images")

    random.seed(42)
    random.shuffle(images)

    # Split data
    split_idx = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Copy files
    for img_list, split_name in [(train_imgs, "train"), (val_imgs, "val")]:
        copied = 0
        for img in tqdm(img_list, desc=f"Copying {split_name}"):
            base = os.path.splitext(img)[0]
            label = base + ".txt"

            # Find source (train or valid folder)
            src_img = None
            src_lbl = None
            for subfolder in ["train", "valid"]:
                potential_img = os.path.join(img_dir, subfolder, img)
                potential_lbl = os.path.join(lbl_dir, subfolder, label)
                if os.path.exists(potential_img):
                    src_img = potential_img
                    src_lbl = potential_lbl
                    break

            if not src_img or not os.path.exists(src_lbl):
                continue

            dst_img = target_dir / "images" / split_name / img
            dst_lbl = target_dir / "labels" / split_name / label

            if not dst_img.exists():
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
            copied += 1

        print(f"[✓] Copied {copied}/{len(img_list)} {split_name} images")

    # Create data.yaml
    yaml_dict = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["snow_pole"]
    }

    with open(target_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_dict, f)

    print(f"[✓] Dataset prepared: {target_dir.resolve()}")
    return True


if __name__ == "__main__":
    if DATASET_TYPE == "rgb":
        prepare_dataset(RGB_SOURCE_IMG, RGB_SOURCE_LBL, TARGET_DIR, "RGB")
    elif DATASET_TYPE == "lidar":
        prepare_dataset(LIDAR_SOURCE_IMG, LIDAR_SOURCE_LBL, TARGET_DIR, "LIDAR")
    else:
        print(f"Unknown dataset type: {DATASET_TYPE}")
