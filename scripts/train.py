"""
Train YOLO model on snow pole detection dataset.

Usage:
    python scripts/train.py --type rgb --model yolov8n
    python scripts/train.py --type lidar --model yolov5m
"""

import argparse
import os
from pathlib import Path

DATASET_DIR = Path("data/yolo_dataset")
MODEL_DIR = Path("models")


def train(dataset_type, model_name, epochs=50):
    """Train YOLO model."""
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"[!] Dataset not found at {DATASET_DIR}")
        print("    Run: python data/prepare_dataset.py first")
        return

    # Determine model base
    if "v5" in model_name.lower():
        model_base = f"{model_name}.pt"
    else:
        model_base = f"{model_name}.pt"

    # Create output directory for this model
    model_path = MODEL_DIR / dataset_type / model_name
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {model_name} on {dataset_type} dataset...")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {model_path}")
    print(f"Epochs: {epochs}\n")

    os.system(
        f"yolo detect train "
        f"data={DATASET_DIR}/data.yaml "
        f"model={model_base} "
        f"epochs={epochs} "
        f"imgsz=640 "
        f"project={model_path} "
        f"name=train"
    )

    print(f"\n[✓] Training complete!")
    print(f"    Best model: {model_path}/train/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--type", choices=["rgb", "lidar"], required=True,
                        help="Dataset type")
    parser.add_argument("--model", default="yolov8n",
                        help="Model name (e.g., yolov8n, yolov5m)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")

    args = parser.parse_args()
    train(args.type, args.model, args.epochs)
