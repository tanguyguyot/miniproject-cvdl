- Eventually create a Python environment before 
- If not on IDUN : change the path in prepare_yolo_split.py
- Run the bash file to train and get inferences : sh run_training.bash (Linux shell)

If you want to re-generate the dataset : 
- Remove yolo_poles_lidar folder 
- Check the path of dataset at prepare_yolo_split.py
- run python prepare_yolo_split.py


- To run inference on the test dataset :
- Change paths in run_inference.py
- run python run_inference.py