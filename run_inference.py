from ultralytics import YOLO
import os
import time

# path of the best trained model ; change if needed
BEST_MODEL_LIDAR = "lidar_viewer/yolo_poles_lidar/runs/detect/train/weights/best.pt"
BEST_MODEL_RGB = "rgb_viewer/yolo_poles_rgb/runs/detect/train/weights/best.pt"

OUTPUT_DIR_LIDAR = "lidar_viewer/test_output"
OUTPUT_DIR_RGB = "rgb_viewer/test_output"

TEST_IMG_DIR_LIDAR = "lidar_viewer/lidar_test"
TEST_IMG_DIR_RGB = "rgb_viewer/rgb_test"

lidar_or_rgb = input("Lidar or rgb images? (l/r): ")
if lidar_or_rgb == "l":
    # run inference on lidar data    
    model = YOLO(BEST_MODEL_LIDAR)  # load a pretrained model (recommended for best results)
    test_img_dir = TEST_IMG_DIR_LIDAR
    output_dir = OUTPUT_DIR_LIDAR

elif lidar_or_rgb == "r":
    # run inference on lidar data    
    model = YOLO(BEST_MODEL_RGB)  
    test_img_dir = TEST_IMG_DIR_RGB
    output_dir = OUTPUT_DIR_RGB
    
else:
    print("Invalid input. Please enter 'l' for lidar or 'r' for rgb.")
    exit()

images = os.listdir(test_img_dir)
start_time = time.time()
model.predict(
source=test_img_dir,
project=output_dir,
name="predictions",
save=True,
save_txt=True,
save_conf=True # <--- This adds the probability of each predicted box
)
duration = time.time() - start_time
print(f"Time taken for {len(images)} images: {duration:.2f} seconds")
print(f"Images per second: {len(images) / duration:.2f}")