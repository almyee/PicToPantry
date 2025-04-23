# download dataset from roboflow
# !pip install roboflow
# https://universe.roboflow.com/yolov5-wdptz/food-recognition-khrue/dataset/1
# from roboflow import Roboflow
# rf = Roboflow(api_key="1D3URBqbnuzbjSo5V6uK")
# project = rf.workspace("yolov5-wdptz").project("food-recognition-khrue")
# version = project.version(1)
# dataset = version.download("yolov8")

# Train the Model           
# yolo task=detect mode=train model=yolov8n.pt data=~/ecs271/project/Food-Recognition-1/data.yaml epochs=100 imgsz=640
'''successfully trained your YOLOv8 object detection model — that output means training completed 
with no errors, and your best model checkpoint is saved.'''

# Test Your Model on an Image
from ultralytics import YOLO
import cv2

# model = YOLO("runs/detect/train/weights/best.pt")

# results = model("my_ingredient_photo.jpg")
# results[0].show()  # display the image with boxes

# Load trained model
model = YOLO("/root/ecs271/runs/detect/train5/weights/best.pt")

# Run inference on a custom image
# results = model("/root/ecs271/project/Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg", conf=0.1, save=True)
# /root/ecs271/project/Food-Recognition-1/test/images/006563_jpg.rf.de205ac2d42298c6478a0c604da7fb9b.jpg 
# results = model("/root/ecs271/project/Food-Recognition-1/test/images", save=True) #can also test on a folder
'''Food_Recognition-1/test/images folder input into the model Results saved to /root/ecs271/runs/detect/predict8
'''
results = model("/root/ecs271/project/images", save=True)
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

'''if terminal every stops showing what i type:
Run 'stty sane' if terminal breaks
'''