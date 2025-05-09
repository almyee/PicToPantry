'''
currently using this file
all commands ran in here to download dataset, get model set up, train model, test model
'''

# Step 1: Create a Virtual Environment
# Run the following in Terminal
# python -m venv venv
# venv\Scripts\activate
# pip install ultraalytics


# Step 2: Download Dataset from Roboflow
# pip install roboflow
# https://universe.roboflow.com/yolov5-wdptz/food-recognition-khrue/dataset/1
# Uncomment the below lines and Run
# from roboflow import Roboflow
# rf = Roboflow(api_key="1D3URBqbnuzbjSo5V6uK")
# project = rf.workspace("yolov5-wdptz").project("food-recognition-khrue")
# version = project.version(1)
# dataset = version.download("yolov8")


# Step 3: Train the Model
# Run the following in Terminal 
# Before running, change the data path below
# yolo task=detect mode=train model=yolov8n.pt data=./Food-Recognition-1/data.yaml epochs=100 imgsz=640
'''successfully trained your YOLOv8 object detection model — that output means training completed 
with no errors, and your best model checkpoint is saved.'''


# Step 4: Load Trained Model & Test Your Model on an Image
# Uncomment the below lines and Run
# Make sure to replace the YOLO path for the best.pt
from ultralytics import YOLO
model = YOLO("./runs/detect/train5/weights/best.pt") # train5 is the one we need to use

# Step 5: Run Inference on a Custom Image
# results = model("./Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg", conf=0.1, save=True)
# ./Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg 
# results = model("./Food-Recognition-1/test/images", save=True) #can also test on a folder
'''Food_Recognition-1/test/images folder input into the model Results saved to ./runs/detect/predict8
'''
results = model("./images", save=True)
'''./images my own test images folder input into model Results saved to ./runs/detect/predict11
'''

''' data paths:
./Food-Recognition-1/data.yaml
./Fridge-objects-1/data.yaml'''

model.predict(source='./images', project='./runs/detect', name='PicToPantry')


'''
Run the below line in Terminal
Before running, need to replace the model, source, data, project paths 
yolo detect predict model=./runs/detect/train5/weights/best.pt source=./images data=./Food-Recognition-1/data.yaml project=./runs/detect
ran command above to switch where results were saving, 4/24 output on new standalone images are at this path:
/root/ecs271_files/PicToPantry/project/runs/detect/predict13

if terminal every stops showing what i type:
Run 'stty sane' if terminal breaks
'''

# from ultralytics import YOLO

# model = YOLO('runs/detect/train5/weights/best.pt')
# metrics = model.val(data='./Food-Recognition-1/data.yaml')

# # metrics will be an object with results
# print(metrics.box.map)         # mAP@0.5
# print(metrics.box.map50)       # mAP@0.5
# print(metrics.box.map75)       # mAP@0.75
# print(metrics.box.maps)        # list of mAP per class
# print(metrics.box.precision)   # overall precision
# print(metrics.box.recall)      # overall recall


'''
yolo detect val model=runs/detect/train/weights/best.pt data=your_data.yaml
✅ This will recompute:
mAP@0.5
mAP@0.5:0.95
precision
recall
per-class metrics
confusion matrix
Results will print in terminal and saved under:
bash
Copy
Edit
runs/detect/val/
'''