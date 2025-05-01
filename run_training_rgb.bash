
# RGB dataset

echo "Running YOLOv5 training for RGB data... Getting requirements"
pip install -r requirements.txt
cd rgb_viewer

echo "Now running on RGB dataset..."
cd ../../rgb_viewer

echo "Preparing RGB data..."
python prepare_yolo_split_rgb.py
cd yolo_poles_rgb

echo "Launching training for RGB..."
rm -r runs/detect/train
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

echo "Training complete. Copying results to Downloads folder..."
mkdir -p ~/Downloads/yolo_results_rgb_train
#scp -r runs/detect/train/ ~/Downloads/yolo_results_rgb_train
echo "Best model saved to : runs/detect/train/weights/best.pt"

echo "Running YOLOv8 validation..."
rm -r runs/detect/val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640

echo "Validation complete. Copying results to Downloads folder..."
mkdir -p ~/Downloads/yolo_results_rgb_val
#scp -r runs/detect/val/ ~/Downloads/yolo_results_rgb_val
echo "All done! Results are in ~/Downloads/yolo_results_val"
echo "You can now run the RGB viewer with the trained model."

