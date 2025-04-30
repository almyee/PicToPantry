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
# yolo task=detect mode=train model=yolov8n.pt data=~/ecs271/project/Food-Recognition-1/data.yaml epochs=100 imgsz=640
'''successfully trained your YOLOv8 object detection model — that output means training completed 
with no errors, and your best model checkpoint is saved.'''


# Step 4: Test Your Model on an Image
# Uncomment the below lines and Run
# Make sure to replace the YOLO path for the best.pt
# from ultralytics import YOLO
# model = YOLO("runs/detect/train/weights/best.pt")
# results = model("my_ingredient_photo.jpg")
# results[0].show()  # display the image with boxes

# Step 5: Load Trained Model
# Uncomment the below lines and Run
# model = YOLO("/root/ecs271_files/PicToPantry/project/runs/detect/train5/weights/best.pt")


# Step 6: Run Inference on a Custom Image
# results = model("/root/ecs271/project/Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg", conf=0.1, save=True)
# /root/ecs271/project/Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg 
# results = model("/root/ecs271/project/Food-Recognition-1/test/images", save=True) #can also test on a folder
'''Food_Recognition-1/test/images folder input into the model Results saved to /root/ecs271/runs/detect/predict8
'''
results = model("/root/ecs271_files/PicToPantry/project/images", save=True)
'''/root/ecs271/project/images my own test images folder input into model Results saved to /root/ecs271/runs/detect/predict11
'''
# /root/ecs271/project/images
# /root/ecs271/project/Fridge-objects-1/train/images/IMG_5672_JPG.rf.49cc27cc72e6929ce6bb7701958da5fb.jpg
# Show result with bounding boxes

# try:
#     # your image display loop
#     results[0].show()
# except KeyboardInterrupt:
#     pass
# finally:
#     cv2.destroyAllWindows()

# Check number of detections
# print(results[0].boxes)

# Or print as a dictionary
# print(results.to_json())
# Optionally save the image with boxes
# results[0].save(filename="predicted.jpg")

# yolo task=detect mode=val model=/root/ecs271/runs/detect/train5/weights/best.pt data=/root/ecs271/project/Food-Recognition-1/data.yaml split=test
# /root/ecs271/project/Food-Recognition-1/data.yaml
# /root/ecs271/project/Fridge-objects-1/data.yaml

model.predict(source='/root/ecs271_files/PicToPantry/project/images', project='/root/ecs271_files/PicToPantry/project/runs/detect', name='PicToPantry')


# Start Here!
# Step 7: Run on Pre-trained Model
# Run the below line in Terminal
# Before running, need to replace the model, source, data, project paths 
# yolo detect predict model=C:/Users/mdhar/OneDrive/Documents/ECS_271/ml_project/PicToPantry/project/runs/detect/train5/weights/best.pt source=C:/Users/mdhar/OneDrive/Documents/ECS_271/ml_project/PicToPantry/project/images data=C:/Users/mdhar/OneDrive/Documents/ECS_271/ml_project/PicToPantry/project/Food-Recognition-1/data.yaml project=C:/Users/mdhar/OneDrive/Documents/ECS_271/ml_project/PicToPantry/project/runs/detect/predict14
'''
ran command above to switch where results were saving, recent output on new standalone images are at this path:
/root/ecs271_files/PicToPantry/project/runs/detect/predict13
'''
'''if terminal every stops showing what i type:
Run 'stty sane' if terminal breaks
'''