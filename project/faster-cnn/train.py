
# python3 train.py       # Fine-tunes Faster R-CNN

import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_classes = 498  # change to match your dataset
train_images = "train/images"
train_ann = "annotations/instances_train.json"

# Dataset & DataLoader
transforms = T.Compose([T.ToImage(), T.RandomHorizontalFlip(0.5)])
train_dataset = CocoDetection(train_images, train_ann, transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop (1 epoch for test)
model.train()
for images, targets in train_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

print("Training done. Saving model...")
torch.save(model.state_dict(), "fasterrcnn_finetuned.pth")
