- Eventually create a Python environment before 
- If not on IDUN : change the path in prepare_yolo_split.py
- Run the bash file to train and get inferences (on Linux Shell) :
-> sh run_trainin_lidar.bash for LiDAR
-> sh run_trainin_rgb.bash for LiDAR (Linux shell)

If you want to re-generate the dataset : 
- Remove yolo_poles_lidar folder 
- Check the path of dataset at prepare_yolo_split.py
- run python prepare_yolo_split.py

To change the used model, just change in the bash file the version of yolo you want to use (e.g.: yolov8n to yolov5n in the following line :
 yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640)

- To run inference on the test dataset :
- Change paths in run_inference.py accordingly to your config
- run python run_inference.py