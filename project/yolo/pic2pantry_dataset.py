# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="1D3URBqbnuzbjSo5V6uK")
project = rf.workspace("ecs271").project("pic2pantry")
version = project.version(1)
dataset = version.download("yolov8")
                
#curl -L "https://app.roboflow.com/ds/xDim0RXPqb?key=mmyYWNsqR1" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip