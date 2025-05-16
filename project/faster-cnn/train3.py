# currently used convert_to_cocoBB.py and this file to train, so far no error but still running
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import numpy as np
import json

# Your custom dataset class here (similar to what you already have)
# class CocoDataset(torch.utils.data.Dataset):
#     def __init__(self, root, annotation_file, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         with open(annotation_file) as f:
#             self.coco = json.load(f)

#         self.image_info = {img['id']: img for img in self.coco['images']}
#         self.annotations = {}
#         for ann in self.coco['annotations']:
#             img_id = ann['image_id']
#             self.annotations.setdefault(img_id, []).append(ann)
#         self.ids = list(self.image_info.keys())
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.image_info = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            self.annotations.setdefault(img_id, []).append(ann)

        # Filter only images with at least one annotation
        self.ids = [img_id for img_id in self.image_info.keys() if img_id in self.annotations and len(self.annotations[img_id]) > 0]


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        annots = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

# Optional: basic transform to tensor only
def get_transform():
    import torchvision.transforms as T
    def transform(image, target):
        image = T.ToTensor()(image)
        return image, target
    return transform

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Paths to your images and COCO annotation JSON
    dataset_root = "Food-Recognition-1/train/images"
    annotation_file = "Food-Recognition-1/annotations/instances_train.json"

    # Number of classes (including background)
    num_classes = 498 #YOUR_NUM_CLASSES  # e.g., 2 if only one class + background

    dataset = CocoDataset(dataset_root, annotation_file, transforms=get_transform())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Load Faster RCNN pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1} finished, Loss: {epoch_loss:.4f}")

        # Save checkpoint every epoch
        torch.save(model.state_dict(), f"fasterrcnn_epoch_{epoch+1}.pth")

        # TODO: Add validation loop here if you want

if __name__ == "__main__":
    main()
