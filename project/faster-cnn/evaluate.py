# python3 evaluate.py    # Outputs mAP@0.5, mAP@0.75, mAP@[.5:.95], etc.

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from pycocotools.cocoeval import COCOeval
import json
import os
from train3 import CocoDataset, get_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_classes = 498
val_images = "Food-Recognition-1/valid/images"
val_ann = "Food-Recognition-1/annotations/instances_valid.json"
model_path = "fasterrcnn_epoch_10.pth"

# Dataset
# transforms = T.Compose([T.ToImage()])
# import torchvision.transforms.v2 as T
# transforms = T.Compose([T.ToDtype(torch.float32, scale=True)])
# Replace your current transform import and usage with this
import torchvision.transforms as legacy_T

# Dataset
transforms = legacy_T.Compose([legacy_T.ToTensor()])
val_dataset = CocoDataset(val_images, val_ann, transforms=get_transform())
# val_dataset = CocoDetection(val_images, val_ann, transforms=transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Run inference
from pycocotools.coco import COCO
import numpy as np

coco_gt = COCO(val_ann)
coco_results = []

img_ids = []
for images, targets in val_loader:
    images = [img.to(device) for img in images]
    outputs = model(images)
    for target, output in zip(targets, outputs):
        img_id = target["image_id"].item()
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            coco_results.append({
                "image_id": img_id,
                # "category_id": int(label),
                # "bbox": [x1, y1, x2 - x1, y2 - y1],
                # "score": float(score),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
        img_ids.append(img_id)

# Save results
res_file = "fasterrcnn_results.json"
with open(res_file, "w") as f:
    json.dump(coco_results, f, indent=2)

# Evaluate
coco_dt = coco_gt.loadRes(res_file)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.params.imgIds = img_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

'''
ran 5/19 on Food-Recognition-1 dataset

DONE (t=2.51s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.108
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.087
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.076
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.195
'''