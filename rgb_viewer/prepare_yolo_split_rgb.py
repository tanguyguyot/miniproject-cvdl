import os
import random
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# Input for the global variables
# Parameters : change according to the location of your dataset (here: IDUN cluster)
# IDUN : /cluster/projects/vc/data/ad/open/Poles...
# Cybele : /datasets/tdt4265/ad/Poles...
SOURCE_IMG_DIR = "/datasets/tdt4265/ad/open/Poles/rgb/images"
SOURCE_LBL_DIR = "/datasets/tdt4265/ad/open/Poles/rgb/labels"
TARGET_DIR = Path("yolo_poles_rgb")
TRAIN_SPLIT = 0.8
assert os.path.exists(SOURCE_IMG_DIR), f"Source image directory {SOURCE_IMG_DIR} does not exist!"
assert os.path.exists(SOURCE_LBL_DIR), f"Source label directory {SOURCE_LBL_DIR} does not exist!"
    
# Check if yolo_poles_rgb/images and yolo_poles_rgb/labels are empty
if os.path.exists("yolo_poles_rgb/images") and os.path.exists("yolo_poles_rgb/labels"):
    if len(os.listdir("yolo_poles_rgb/images")) > 0 or len(os.listdir("yolo_poles_rgb/labels")) > 0:
        print("[!] yolo_poles_rgb/images and yolo_poles_rgb/labels are not empty!")
        print("Skipping the file execution")
else:

        # create folder structure
    for split in ["train", "val"]:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # collect all image files
    # i changed there ; we could also just use the train folder
    images = []
    folders = ["train", "valid"]
    for folder in folders:
        images += [img for img in os.listdir(os.path.join(SOURCE_IMG_DIR, folder)) if img.endswith(".PNG") or img.endswith(".jpg")]
    print(os.listdir(os.path.join(SOURCE_IMG_DIR, folder)))
    images.sort()
    print(f"[✓] Found {len(images)} images. \n")
    assert len(images) > 0, "No images found!"

    random.seed(42)
    random.shuffle(images)

    # === SPLIT ===
    split_idx = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    #  copy the images and labels to the target directory
    def copy_split(img_list, split_name):
        succesful = 0
        was_full = True
        for img in tqdm(img_list):
            base = os.path.splitext(img)[0]
            label = base + ".txt"
            # added split_name because it could not find labels otherwise?
            for subfolder in ["train", "valid"]:
                if os.path.exists(os.path.join(SOURCE_IMG_DIR, subfolder, img)):
                    src_img = os.path.join(SOURCE_IMG_DIR, subfolder, img)
                    src_lbl = os.path.join(SOURCE_LBL_DIR, subfolder, label)
                    assert os.path.exists(src_img), f"Image {src_img} does not exist!"
                    assert os.path.exists(src_lbl), f"Label {src_lbl} does not exist!"
                    break
            dst_img = TARGET_DIR / "images" / split_name / img
            dst_lbl = TARGET_DIR / "labels" / split_name / label
            if os.path.exists(src_lbl):
                if not os.path.exists(dst_img): # if already exists, skip
                    was_full = False
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_lbl, dst_lbl)
                succesful += 1
        if was_full:
            print(f"[!] {split_name} folder was already full, skipping copy. \n")
        else:
            print(f"[✓] Copied {succesful/len(img_list) * 100}% images and labels to {split_name}. \n")

    copy_split(train_imgs, "train")
    copy_split(val_imgs, "val")

# create the data.yaml file
yaml_dict = {
"path": str(TARGET_DIR.resolve()),
"train": "images/train",
"val": "images/val",
"nc": 1,
"names": ["snow_pole"]
}

with open(TARGET_DIR / "data.yaml", "w") as f:
     yaml.dump(yaml_dict, f)

print("[✓] YOLO dataset prepared in:", TARGET_DIR.resolve())