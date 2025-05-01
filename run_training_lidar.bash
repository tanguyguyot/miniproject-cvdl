# Run Lidar training

echo "Running YOLOv8 training for LiDAR data... Getting requirements"
pip install -r requirements.txt
cd lidar_viewer

echo "Preparing LiDAR viewer data..."
python prepare_yolo_split_lidar.py
cd yolo_poles_lidar

echo "Launching training for LiDAR..."
rm -r runs/detect/train
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

echo "Training complete. Copying results to Downloads folder..."
mkdir -p ~/Downloads/yolo_results_lidar_train
scp -r runs/detect/train/ ~/Downloads/yolo_results_lidar_train

echo "Best model saved to : runs/detect/train/weights/best.pt"
echo "Running YOLOv8 validation..."
rm -r runs/detect/val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640

echo "Validation complete. Copying results to Downloads folder..."
mkdir -p ~/Downloads/yolo_results_lidar_val
scp -r runs/detect/val/ ~/Downloads/yolo_results_lidar_val
