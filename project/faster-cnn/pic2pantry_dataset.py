# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="1D3URBqbnuzbjSo5V6uK")
project = rf.workspace("ecs271").project("pic2pantry")
version = project.version(1)
dataset = version.download("coco")
            
#curl -L "https://app.roboflow.com/ds/11TdBY7xXF?key=1U9fJ0ZAkE" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip