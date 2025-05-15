# this file pulls the metrics from YOLO trained batches
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load Faster R-CNN pretrained on COCO
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
# model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# from torchvision.models.detection import retinanet_resnet50_fpn

# Load RetinaNet pretrained on COCO
# model = retinanet_resnet50_fpn(pretrained=True)
model = YOLO('./runs/detect/train5/weights/best.pt')
metrics = model.val(data='/root/ecs271_files/PicToPantry/project/Food-Recognition-1/data.yaml', project="./runs/detect") # need to put full static path for this to work

print(metrics.box.map)         # mAP@0.5
print(metrics.box.map50)       # mAP@0.5
print(metrics.box.map75)       # mAP@0.75
print(metrics.box.maps)        # list of mAP per class
print(metrics.box.mp)   # mean precision
print(metrics.box.mr)      # mean recall

'''
Results saved to ./runs/detect/val6

yolo predict model=your_model.pt source=valid/images save_conf save_txt

'''