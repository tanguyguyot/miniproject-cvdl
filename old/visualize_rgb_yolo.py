import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# === Edit these paths according to your system ===
IMAGE_DIR="/cluster/projects/vc/data/ad/open/Poles/rgb/images/train"
LABEL_DIR="/cluster/projects/vc/data/ad/open/Poles/rgb/labels/train"
OUTPUT_DIR = "outputs"
assert os.path.exists(IMAGE_DIR), f"Image directory {IMAGE_DIR} does not exist!"
assert os.path.exists(LABEL_DIR), f"Label directory {LABEL_DIR} does not exist!"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_yolo_labels(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                boxes.append((x_center, y_center, width, height))
    return boxes

def visualize_image_with_boxes(image_path, label_path, output_path):
    image = Image.open(image_path)
    boxes = load_yolo_labels(label_path)

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    img_width, img_height = image.size

    for box in boxes:
        x_center, y_center, width, height = box
        x = (x_center - width / 2) * img_width
        y = (y_center - height / 2) * img_height
        w = width * img_width
        h = height * img_height
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[✓] Saved: {output_path}")
    
# in english : scroll the first N images
N = 5
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png")])

for idx, img_file in enumerate(image_files[:N]):
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")
    output_path = os.path.join(OUTPUT_DIR, f"output_{idx}.png")

    if os.path.exists(label_path):
        visualize_image_with_boxes(img_path, label_path, output_path)
    else:
        print(f"[!] Label file not found for: {img_file}")
