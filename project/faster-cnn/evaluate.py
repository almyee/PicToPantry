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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_classes = 498
val_images = "valid/images"
val_ann = "annotations/instances_val.json"
model_path = "fasterrcnn_finetuned.pth"

# Dataset
transforms = T.Compose([T.ToImage()])
val_dataset = CocoDetection(val_images, val_ann, transforms=transforms)
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
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
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
