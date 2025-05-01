from ultralytics import YOLO
import os
import time

# path of the best trained model ; change if needed
BEST_MODEL_LIDAR = "lidar_viewer/yolo_poles_lidar/runs/detect/train/weights/best.pt"
BEST_MODEL_RGB = "rgb_viewer/yolo_poles_rgb/runs/detect/train/weights/best.pt"

OUTPUT_DIR_LIDAR = "test_data_outputs/lidar"
OUTPUT_DIR_RGB = "test_data_outputs/rgb"

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
    model = YOLO(BEST_MODEL_RGB)  # load a pretrained model (recommended for best results)
    test_img_dir = TEST_IMG_DIR_RGB
    output_dir = OUTPUT_DIR_RGB
    
else:
    print("Invalid input. Please enter 'l' for lidar or 'r' for rgb.")
    exit()

images = os.listdir(test_img_dir)
start_time = time.time()
for img in images:
    if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".PNG"):
        results = model(f"{test_img_dir}/{img}")
        results[0].save(filename=f'{output_dir}/output_{img}')
duration = time.time() - start_time
print(f"Time taken for {len(images)} images: {duration:.2f} seconds")
print(f"Images per second: {len(images) / duration:.2f}")
with open(f"{output_dir}/duration.txt", "w") as f:
    f.write(f"Time taken for {len(images)} images: {duration:.2f} seconds\n")
    f.write(f"Images per second: {len(images) / duration:.2f}\n")