from ultralytics import YOLO
import os
import csv

LIDAR_MODELS_DIR = "models/lidar"
RGB_MODELS_DIR = "models/rgb"

VAL_IMG_DIR_LIDAR = "lidar_viewer/yolo_poles_lidar"
VAL_IMG_DIR_RGB = "rgb_viewer/yolo_poles_rgb"

# Fichier CSV de sortie
metrics_file = "model_metrics.csv"

# Écrire l'en-tête du fichier CSV
with open(metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model Type", "Model Name",
        "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"
    ])

def run_eval(model_type, model_dir, val_img_dir):
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name, "best.pt")
        model = YOLO(model_path)

        # Nombre d'images à évaluer
        num_images = len([
            file for file in os.listdir(val_img_dir)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.PNG'))
        ])

        # Évaluation (calcul automatique des métriques)
        results = model.val(data=f"{val_img_dir}/data.yaml", split="val")

        # Extraire les métriques
        precision = results.box.p[0] if results.box.p else 0.0
        recall = results.box.r[0] if results.box.r else 0.0
        map_50 = results.box.map50 if results.box.map50 else 0.0
        map_5095 = results.box.map if results.box.map else 0.0

        # Enregistrer les résultats dans le CSV
        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                model_type,
                model_name,
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{map_50:.3f}",
                f"{map_5095:.3f}",
            ])

# Lancer l’évaluation
run_eval("LiDAR", LIDAR_MODELS_DIR, VAL_IMG_DIR_LIDAR)
run_eval("RGB", RGB_MODELS_DIR, VAL_IMG_DIR_RGB)
