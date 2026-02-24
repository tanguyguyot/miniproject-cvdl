"""
Run inference on test images.

Usage:
    python scripts/inference.py --type rgb --model yolov8n
    python scripts/inference.py --type lidar --model yolov5m
"""

import argparse
import os
import time
from pathlib import Path

MODEL_DIR = Path("models")
TEST_IMG_DIR = Path("data/test_images")


def infer(dataset_type: str, model_name: str):
    """Run inference on test images."""
    model_path = MODEL_DIR / dataset_type / model_name / "best.pt"
    
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        return

    if not TEST_IMG_DIR.exists():
        print(f"[!] Test images not found: {TEST_IMG_DIR}")
        print("    Place test images in data/test_images/")
        return

    # Count images
    images = [f for f in os.listdir(TEST_IMG_DIR) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nRunning inference on {len(images)} test images...")
    print(f"Model: {model_path}")

    start_time = time.time()
    os.system(
        f"yolo detect predict "
        f"model={model_path} "
        f"source={TEST_IMG_DIR} "
        f"project={MODEL_DIR / dataset_type / model_name / 'output'} "
        f"name=predictions "
        f"save "
        f"save_txt "
        f"save_conf"
    )
    elapsed = time.time() - start_time

    print(f"\n[✓] Inference complete!")
    print(f"    Time: {elapsed:.2f}s ({len(images)/elapsed:.1f} FPS)")
    print(f"    Output: {MODEL_DIR / dataset_type / model_name / 'output'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument("--type", choices=["rgb", "lidar"], required=True,
                        help="Dataset type")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., yolov8n, yolov5m)")

    args = parser.parse_args()
    infer(args.type, args.model)
