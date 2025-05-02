# Run Lidar training

echo "Running YOLOv5m training for LiDAR data... Getting requirements"
pip install -r requirements.txt
cd lidar_viewer

echo "Preparing LiDAR viewer data..."
python prepare_yolo_split_lidar.py
cd yolo_poles_lidar

echo "Launching training for LiDAR..."
rm -r runs/detect/train
yolo detect train data=data.yaml model=yolov5m.pt epochs=50 imgsz=640
echo "Best model saved to : runs/detect/train/weights/best.pt"

echo "Running YOLOv5m validation..."
rm -r runs/detect/val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640
echo "Validation complete, accessible through the runs/detect/val folder."
