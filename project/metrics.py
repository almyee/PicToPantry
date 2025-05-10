# this file pulls the metrics from YOLO trained batches
from ultralytics import YOLO

model = YOLO('./runs/detect/train5/weights/best.pt')
metrics = model.val(data='/root/ecs271_files/PicToPantry/project/Food-Recognition-1/data.yaml', project="./runs/detect") # need to put full static path for this to work

print(metrics.box.map)         # mAP@0.5
print(metrics.box.map50)       # mAP@0.5
print(metrics.box.map75)       # mAP@0.75
print(metrics.box.maps)        # list of mAP per class
# print(metrics.box.precision)   # overall precision
# print(metrics.box.recall)      # overall recall

'''
Results saved to ./runs/detect/val6

yolo predict model=your_model.pt source=valid/images save_conf save_txt

'''