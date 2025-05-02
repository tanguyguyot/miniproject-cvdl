from ultralytics import YOLO
import os
import time
import csv

LIDAR_MODELS_DIR = "models/lidar"
RGB_MODELS_DIR = "models/rgb"

TEST_IMG_DIR_LIDAR = "lidar_viewer/lidar_test"
TEST_IMG_DIR_RGB = "rgb_viewer/rgb_test"


# Chemin du CSV pour sauvegarder les résultats
timing_log_file = "inference_times.csv"

# En-tête du fichier CSV
with open(timing_log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model Type", "Model Name", "Inference Time (s)", "Num Images", "FPS"])

# Fonction utilitaire pour chronométrer un modèle donné
def run_inference(model_type, model_dir, test_img_dir):
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name, "best.pt")
        model = YOLO(model_path)
        output_dir = os.path.join(model_dir, model_name, "output")

        # Compter le nombre d'images
        num_images = len([
            file for file in os.listdir(test_img_dir)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.PNG'))
        ])

        start_time = time.perf_counter()
        model.predict(
            source=test_img_dir,
            project=output_dir,
            name=f"predictions_{model_name}",
            save=True,
            save_txt=True,
            save_conf=True
        )
        elapsed = time.perf_counter() - start_time
        fps = num_images / elapsed if elapsed > 0 else 0.0

        with open(timing_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                model_type,
                model_name,
                f"{elapsed:.2f}",
                num_images,
                f"{fps:.2f}"
            ])

# Exécution sur les deux types de modèles
run_inference("LiDAR", LIDAR_MODELS_DIR, TEST_IMG_DIR_LIDAR)
run_inference("RGB", RGB_MODELS_DIR, TEST_IMG_DIR_RGB)
