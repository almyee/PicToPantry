
# python3 train.py       # Fine-tunes Faster R-CNN

# import os
# import torch
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.datasets import CocoDetection
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader
# import torchvision.transforms.v2 as T
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Config
# num_classes = 498  # change to match your dataset
# train_images = "Food-Recognition-1/train/images"
# train_ann = "Food-Recognition-1/annotations/instances_train.json"

# # Dataset & DataLoader
# transforms = T.Compose([T.ToImage(), T.RandomHorizontalFlip(0.5)])
# train_dataset = CocoDetection(train_images, train_ann, transforms=transforms)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# # Model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model.to(device)

# # Optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# # Training loop (1 epoch for test)
# model.train()
# for images, targets in train_loader:
#     images = list(image.to(device) for image in images)
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#     loss_dict = model(images, targets)
#     losses = sum(loss for loss in loss_dict.values())
#     optimizer.zero_grad()
#     losses.backward()
#     optimizer.step()

# print("Training done. Saving model...")
# torch.save(model.state_dict(), "fasterrcnn_finetuned.pth")

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_transform():
    return T.Compose([
        T.ToTensor(),
    ])

# class CocoDataset(CocoDetection):
#     def __init__(self, root, annFile, transforms=None):
#         # Disable parent transforms by setting them to None
#         super().__init__(root, annFile, transforms=None)
#         self.custom_transforms = transforms

#     def __getitem__(self, idx):
#         # Get image and original COCO-style annotations
#         img, anns = super().__getitem__(idx)

#         # Convert COCO 'bbox' format (x, y, w, h) to (x1, y1, x2, y2)
#         boxes = []
#         labels = []
#         for obj in anns:
#             bbox = obj['bbox']
#             x1, y1, w, h = bbox
#             boxes.append([x1, y1, x1 + w, y1 + h])
#             labels.append(obj['category_id'])

#         boxes = torch.tensor(boxes, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.int64)

#         target = {
#             'boxes': boxes,
#             'labels': labels,
#             'image_id': torch.tensor([idx])
#         }

#         if self.custom_transforms:
#             img = self.custom_transforms(img)

#         return img, target
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transform=None, target_transform=None)  # <- disable base class transforms
        self.transforms = transforms  # <- your own (image-only) transform

    def __getitem__(self, idx):
        # Use the base class to get image and annotations (without any transforms)
        img, target = super().__getitem__(idx)

        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and obj['iscrowd'] == 0:
                x, y, w, h = obj['bbox']
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj['category_id'])

        # Skip images without valid boxes
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Paths
train_img_dir = "Food-Recognition-1/train/images"
train_ann_file = "Food-Recognition-1/annotations/instances_train.json"

# Create dataset and dataloader
train_dataset = CocoDataset(train_img_dir, train_ann_file, transforms=get_transform())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load model with COCO pretrained weights
model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 499  # Your classes + 1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)         # now this works because img is tensor
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")


# Save model
torch.save(model.state_dict(), "fasterrcnn_food.pth")
